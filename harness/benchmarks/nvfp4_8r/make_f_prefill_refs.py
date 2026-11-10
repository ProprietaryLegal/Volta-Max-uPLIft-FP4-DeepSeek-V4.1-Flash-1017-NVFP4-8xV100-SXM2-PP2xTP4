"""CPU only (the 8-card host): fp64 references for plan 04 part B (IQ prefill of the
WARM tier), one layer, M = 8,188 rows; rows [:m] give a smaller M.

Everything is synthetic except the weights:
  tier   int8 [384]: a fixed permutation (seed 404) of the expert ids, its
         first 190 HOT (0), next 101 WARM (1), last 93 HOST (2). 101 WARM
         experts are three full tiles of 32 and one of 5.
  ids    uniform top-6 without replacement per row (no tally: nothing here
         derives from private text), weights random and normalized,
         x ~ N(0, 1) fp16.
Per row, in fp64, summed over all four TP ranks (full K, all 5,120 rows):
  ref_native  the HOT and HOST routes from the raw NVFP4 checkpoint
              (accept_p1.Checkpoint: e2m1 x E4M3 x global scale),
  ref_iq      the WARM routes from gguf.quants.dequantize of the GGUF copy,
both by weight * W2 (silu(min(W1 x, 10)) * clamp(W3 x, +-10)), the formula
of make_p2_refs.contribution applied to all rows of an expert at once.
--iq-only skips ref_native (layers used for the IQ part alone).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gguf
import numpy as np

from .accept_p1 import Checkpoint
from .make_p2_refs import HIDDEN, LIMIT, TOP_K, silu

M = 8188
HOT, WARM, HOST = 0, 1, 2
COUNTS = (190, 101, 93)
GGUF_NAMES = {"w1": "gate", "w3": "up", "w2": "down"}


def synthetic_tier() -> np.ndarray:
    order = np.random.default_rng(404).permutation(384)
    tier = np.empty(384, np.int8)
    tier[order[: COUNTS[0]]] = HOT
    tier[order[COUNTS[0] : COUNTS[0] + COUNTS[1]]] = WARM
    tier[order[COUNTS[0] + COUNTS[1] :]] = HOST
    return tier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True, help="first shard")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--iq-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index

    started = time.monotonic()
    rng = np.random.default_rng(40400 + a.layer)
    ids = np.argsort(rng.random((M, 384)), axis=1)[:, :TOP_K].astype(np.int32)
    tw = rng.random((M, TOP_K)) + 0.1
    tw = (tw / tw.sum(1, keepdims=True)).astype(np.float32)
    x = rng.standard_normal((M, HIDDEN)).astype(np.float16)
    xf = x.astype(np.float64)
    tier = synthetic_tier()
    index = gguf_tensor_index(gguf_shard_paths(a.gguf))
    tensors = {role: index[f"blk.{a.layer}.ffn_{name}_exps.weight"]
               for role, name in GGUF_NAMES.items()}
    types = {role: int(t.tensor_type) for role, t in tensors.items()}
    ckpt = Checkpoint(a.weights, a.verification_log)
    refs = {"native": np.zeros((M, HIDDEN), np.float64), "iq": np.zeros((M, HIDDEN), np.float64)}
    routes = {"native": 0, "iq": 0}
    for e in range(384):
        part = "iq" if tier[e] == WARM else "native"
        if part == "native" and a.iq_only:
            continue
        rows, slots = np.nonzero(ids == e)  # a row routes to an expert at most once
        if rows.size == 0:
            continue
        if part == "iq":
            w1, w3, w2 = (
                gguf.quants.dequantize(
                    np.asarray(tensors[role].data[e]), gguf.GGMLQuantizationType(types[role])
                ).astype(np.float64)
                for role in ("w1", "w3", "w2")
            )
        else:
            w1, w3, w2 = (ckpt.projection(a.layer, e, p).astype(np.float64)
                          for p in ("w1", "w3", "w2"))
        if w1.shape != (2304, HIDDEN) or w3.shape != (2304, HIDDEN) or w2.shape != (HIDDEN, 2304):
            raise ValueError(f"L{a.layer} expert {e}: shapes {w1.shape} {w3.shape} {w2.shape}")
        xe = xf[rows]
        g = np.minimum(xe @ w1.T, LIMIT)
        u = np.clip(xe @ w3.T, -LIMIT, LIMIT)
        refs[part][rows] += ((silu(g) * u) @ w2.T) * tw[rows, slots].astype(np.float64)[:, None]
        routes[part] += int(rows.size)
    if not np.isfinite(refs["native"]).all() or not np.isfinite(refs["iq"]).all():
        raise ValueError("a reference is not finite")
    arrays = {"tier": tier, "ids": ids, "tw": tw, "x": x, "ref_iq": refs["iq"].astype(np.float32)}
    if not a.iq_only:
        arrays["ref_native"] = refs["native"].astype(np.float32)
    arrays["meta"] = np.frombuffer(json.dumps({
        "layer": a.layer, "m": M, "iq_only": a.iq_only,
        "type13": types["w1"], "type3": types["w3"], "type2": types["w2"],
    }).encode(), np.uint8)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **arrays)
    print(json.dumps({
        "layer": a.layer, "m": M, "types": types, "iq_only": a.iq_only, "routes": routes,
        "experts": {"hot": COUNTS[0], "warm": COUNTS[1], "host": COUNTS[2]},
        "rows_without_warm_route": int((tier[ids] != WARM).all(1).sum()),
        "norm_iq": float(np.linalg.norm(refs["iq"])),
        "norm_native": float(np.linalg.norm(refs["native"])),
        "seconds": round(time.monotonic() - started, 1), "output": str(a.output),
    }), flush=True)


if __name__ == "__main__":
    main()
