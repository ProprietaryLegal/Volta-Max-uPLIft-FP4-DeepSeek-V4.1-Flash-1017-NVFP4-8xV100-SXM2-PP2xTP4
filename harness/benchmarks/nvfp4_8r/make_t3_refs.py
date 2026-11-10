"""CPU only (the 8-card host): fp64 references for A3 and the falsifier-1 run (plan 01).

Per routed slot of every sample, the whole layer-output contribution
(all four TP ranks summed: full K, all 5,120 rows) in fp64, twice:
  nat  from the raw NVFP4 checkpoint (accept_p1.Checkpoint: e2m1 x E4M3 x
       global scale; no converter, no engine code),
  iq   from gguf.quants.dequantize of the GGUF copy's expert tensors,
both through make_p2_refs.contribution (limited SwiGLU 10, router weight).
A test adds, slot by slot, the reference of the tier that slot runs in.
Samples are make_p2_refs.routes at M = 6 (tally-weighted draw, random
normalized weights, x ~ N(0, 1) fp16); rows [:m] of a sample give M = m.
The output holds expert ids, never counts. Local use and tests on the test host only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gguf
import numpy as np

from .accept_p1 import Checkpoint
from .make_p2_refs import HIDDEN, TOP_K, contribution, routes

M = 6
GGUF_NAMES = {"w1": "gate", "w3": "up", "w2": "down"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    parser.add_argument("--tally", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True, help="first shard")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index

    started = time.monotonic()
    with np.load(a.tally, allow_pickle=False) as data:
        phases = [str(p) for p in data["phases"]]
        counts = data["counts"][phases.index("teacher"), a.layer]
    drawn = routes(counts, 1000 * a.layer + M, M, a.samples)
    del counts
    index = gguf_tensor_index(gguf_shard_paths(a.gguf))
    tensors = {role: index[f"blk.{a.layer}.ffn_{name}_exps.weight"]
               for role, name in GGUF_NAMES.items()}
    types = {role: int(t.tensor_type) for role, t in tensors.items()}
    ckpt = Checkpoint(a.weights, a.verification_log)
    nat = [np.zeros((M * TOP_K, HIDDEN), np.float64) for _ in drawn]
    iq = [np.zeros((M * TOP_K, HIDDEN), np.float64) for _ in drawn]
    needed = sorted({int(e) for ids, _, _ in drawn for e in ids.ravel()})
    for e in needed:
        native = [ckpt.projection(a.layer, e, p).astype(np.float64) for p in ("w1", "w3", "w2")]
        quant = [
            gguf.quants.dequantize(
                np.asarray(tensors[role].data[e]), gguf.GGMLQuantizationType(types[role])
            ).astype(np.float64)
            for role in ("w1", "w3", "w2")
        ]
        for w in (native, quant):
            if w[0].shape != (2304, HIDDEN) or w[1].shape != (2304, HIDDEN) or w[2].shape != (HIDDEN, 2304):
                raise ValueError(f"L{a.layer} expert {e}: projection shapes {[v.shape for v in w]}")
        for s, (ids, tw, x) in enumerate(drawn):
            for row, slot in zip(*np.nonzero(ids == e)):
                r = int(row) * TOP_K + int(slot)
                nat[s][r] = contribution(*native, x[row], float(tw[row, slot]))
                iq[s][r] = contribution(*quant, x[row], float(tw[row, slot]))
    arrays: dict[str, np.ndarray] = {}
    for s, (ids, tw, x) in enumerate(drawn):
        arrays[f"ids_s{s}"], arrays[f"tw_s{s}"], arrays[f"x_s{s}"] = ids, tw, x
        arrays[f"nat_s{s}"], arrays[f"iq_s{s}"] = nat[s], iq[s]
    arrays["meta"] = np.frombuffer(json.dumps({
        "layer": a.layer, "m": M, "samples": a.samples,
        "type13": types["w1"], "type3": types["w3"], "type2": types["w2"],
    }).encode(), np.uint8)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **arrays)
    # Aggregate only: how far the IQ record is from the native one on these inputs.
    num = sum(float(np.linalg.norm(i.reshape(M, TOP_K, HIDDEN).sum(1)
                                   - n.reshape(M, TOP_K, HIDDEN).sum(1)) ** 2)
              for i, n in zip(iq, nat))
    den = sum(float(np.linalg.norm(n.reshape(M, TOP_K, HIDDEN).sum(1)) ** 2) for n in nat)
    print(json.dumps({"layer": a.layer, "distinct_experts": len(needed), "types": types,
                      "iq_vs_native_rel_l2": (num / den) ** 0.5,
                      "seconds": round(time.monotonic() - started, 1),
                      "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
