"""P2 references on CPU: TP-rank partial MoE outputs straight from raw NVFP4.

Weights come from accept_p1.Checkpoint (e2m1 x E4M3 block scale x global
scale, no converter or engine code). Routes are drawn top-6 without
replacement weighted by the private P16 decode ("teacher") tally, as in
an earlier unpublished per-layer bench; router weights are random and normalized; x ~ N(0, 1) fp16.
The output files hold expert ids, never counts. Local use only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .accept_p1 import Checkpoint

HIDDEN = 5120
TP_INTER = 576
TOP_K = 6
LIMIT = 10.0
MS = (1, 6, 12)
F64 = npt.NDArray[np.float64]


def routes(
    counts: npt.NDArray[np.int64], seed: int, m: int, samples: int
) -> list[tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.float16]]]:
    rng = np.random.default_rng(seed)
    weight = np.maximum(counts.astype(np.float64), 1e-9)
    weight /= weight.sum()
    out = []
    for _ in range(samples):
        ids = np.stack(
            [rng.choice(384, TOP_K, replace=False, p=weight) for _ in range(m)]
        ).astype(np.int32)
        tw = rng.random((m, TOP_K)) + 0.1
        tw = (tw / tw.sum(1, keepdims=True)).astype(np.float32)
        x = rng.standard_normal((m, HIDDEN)).astype(np.float16)
        out.append((ids, tw, x))
    return out


def silu(v: F64) -> F64:
    return v / (1.0 + np.exp(-v))


def contribution(
    w1: F64, w3: F64, w2: F64, x: npt.NDArray[np.float16], weight: float
) -> F64:
    """One routed slot: weight * W2 (silu(min(W1 x, 10)) * clamp(W3 x, +-10))."""
    xf = x.astype(np.float64)
    g = np.minimum(w1 @ xf, LIMIT)
    u = np.clip(w3 @ xf, -LIMIT, LIMIT)
    return weight * (w2 @ (silu(g) * u))


def build(
    weights: Path, log: Path, tally: Path, layer: int, ranks: list[int],
    samples: int, output: Path, ms: tuple[int, ...] = MS,
) -> None:
    started = time.monotonic()
    with np.load(tally, allow_pickle=False) as data:
        phases = list(data["phases"])
        counts = data["counts"][phases.index("teacher"), layer]
    drawn = {m: routes(counts, 1000 * layer + m, m, samples) for m in ms}
    needed = sorted({int(e) for m in ms for ids, _, _ in drawn[m] for e in ids.ravel()})
    ckpt = Checkpoint(weights, log)
    acc = {
        (rank, m, s): np.zeros((m, HIDDEN), np.float64)
        for rank in ranks for m in ms for s in range(samples)
    }
    # One expert at a time: dequantize once, apply to every slot routed to it.
    for e in needed:
        w1, w3, w2 = (ckpt.projection(layer, e, p).astype(np.float64)
                      for p in ("w1", "w3", "w2"))
        for rank in ranks:
            lo, hi = rank * TP_INTER, (rank + 1) * TP_INTER
            for m in ms:
                for s, (ids, tw, x) in enumerate(drawn[m]):
                    for row, slot in zip(*np.nonzero(ids == e)):
                        acc[rank, m, s][row] += contribution(
                            w1[lo:hi], w3[lo:hi], w2[:, lo:hi], x[row], float(tw[row, slot])
                        )
    arrays: dict[str, np.ndarray] = {
        f"ref_r{rank}_m{m}_s{s}": v.astype(np.float32) for (rank, m, s), v in acc.items()
    }
    for m in ms:
        for s, (ids, tw, x) in enumerate(drawn[m]):
            arrays[f"ids_m{m}_s{s}"] = ids
            arrays[f"tw_m{m}_s{s}"] = tw
            arrays[f"x_m{m}_s{s}"] = x
    arrays["meta"] = np.frombuffer(
        json.dumps(
            {"layer": layer, "ranks": ranks, "ms": list(ms), "samples": samples}
        ).encode(),
        np.uint8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    print(json.dumps({"layer": layer, "ranks": ranks, "distinct_experts": len(needed),
                      "seconds": round(time.monotonic() - started, 1),
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    parser.add_argument("--tally", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ms", type=int, nargs="+", default=list(MS))
    a = parser.parse_args()
    build(a.weights, a.verification_log, a.tally, a.layer, a.ranks, a.samples, a.output,
          tuple(a.ms))
