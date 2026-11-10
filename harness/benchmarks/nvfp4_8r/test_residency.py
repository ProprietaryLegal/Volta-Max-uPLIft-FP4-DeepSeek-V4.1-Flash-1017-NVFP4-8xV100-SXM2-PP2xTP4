"""CPU test of choose_residency on the real private P16 decode tally.

Properties checked against the tally itself (outside the implementation):
partition per layer; hot experts fit the budget and no further hot expert
would; every cold count <= every hot count within the stage. Budgets are the
ESTIMATED per-card values of the calibration boot (the real loader passes free HBM
measured at boot). Prints aggregate shares only, never per-expert counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat

G = 2**30


def main() -> int:
    with np.load(os.environ["LANE"] + "/engine/private/p16-tally.npz") as d:
        phases = list(d["phases"])
        decode = d["counts"][phases.index("teacher")]
    for layers, budget_gib in ((range(0, 20), 20.37), (range(20, 40), 16.27)):
        budget = int(budget_gib * G)
        res = nat.choose_residency(decode, layers, budget)
        hot_n = sum(len(res[l][0]) for l in layers)
        for l in layers:
            hot, cold = res[l]
            assert sorted(hot + cold) == list(range(384)), f"L{l} not a partition"
        assert hot_n * nat.EXPERT_BYTES <= budget < (hot_n + 1) * nat.EXPERT_BYTES
        max_cold = max((int(decode[l, e]) for l in layers for e in res[l][1]), default=-1)
        min_hot = min(int(decode[l, e]) for l in layers for e in res[l][0])
        assert max_cold <= min_hot, "a cold expert is routed more than a hot one"
        cold_routes = sum(int(decode[l, e]) for l in layers for e in res[l][1])
        share = cold_routes / int(decode[layers.start : layers.stop].sum())
        print(f"layers {layers.start}-{layers.stop - 1}: budget {budget_gib} GiB, "
              f"hot {hot_n} cold {len(layers) * 384 - hot_n}, decode cold share {share:.4f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
