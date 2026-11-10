"""P3 speed bench, ms/round per miss arm (CPU; ESTIMATE from MEASURED parts).

Inputs: the test host bench receipt (bench_p3_miss_paths.py) and the private P16
decode tally. Per layer, a round is the M = 6 MTP verification batch: 6 tokens,
each routed to 6 distinct experts drawn in proportion to the decode tally
(Plackett-Luce; tokens independent: no temporal locality, so a cache of recent
cold experts would only do better). Residency per stage from choose_residency
at the stage budgets of test_residency (20.37 / 16.27 GiB per card). For each
layer the distribution of k = distinct cold experts per round is simulated and
the per-layer added cost f(k) of each arm, from the receipt, is averaged; stages
run one after the other for one request, so per-round cost is the sum over the
40 layers. Base: production 27.3 ms/round; all-native all-hot +8.8 ms (owner).
Prints only aggregates (no counts leave this script).
"""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import numpy as np

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native

TALLY = Path(os.environ["LANE"] + "/engine/private/p16-tally.npz")
STAGES = ((range(0, 20), 20.37), (range(20, 40), 16.27))
BASE_MS, NATIVE_MS = 27.3, 8.8


def k_distribution(p: np.ndarray, cold: np.ndarray, rounds: int, gen) -> np.ndarray:
    # Plackett-Luce top-6 without replacement via Gumbel top-k.
    logp = np.log(np.maximum(p, 1e-12))
    g = logp[None, None, :] - np.log(-np.log(gen.random((rounds, 6, p.size))))
    top = np.argpartition(-g, 6, axis=2)[:, :, :6].reshape(rounds, -1)
    hits = cold[top]
    ks = np.array([len(set(t[h].tolist())) for t, h in zip(top, hits)])
    return np.bincount(ks, minlength=37)[:37] / rounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4000)
    a = ap.parse_args()
    r = json.loads(a.receipt.read_text())
    g, c, cpu = r["graph_us"], r["copy_us"], r["cpu"]
    rt = r["roundtrip_us"]["d2h_sync_h2d_sync"]

    def lin(table, k):  # per-expert extrapolation past the measured k = 6
        return table(min(k, 6)) + max(0, k - 6) * (table(6) - table(5))

    def res(k):
        return lin(lambda j: g[f"M6_k{j}"]["res"], k)

    base0 = g["M6_k0"]["res"]
    arms = {
        # includes the split overhead itself (f(0) > 0)
        "zero_copy_now": lambda k: lin(lambda j: g[f"M6_k{j}"]["zc"], k) - base0,
        "a1_dma_upper": lambda k: 0.0 if k == 0 else c["id_sync"]
        + lin(lambda j: c[f"k{j}"]["dma"], k) + res(k) - base0,
        "a1_dma_lower": lambda k: 0.0 if k == 0 else max(
            0.0, c["id_sync"] + lin(lambda j: c[f"k{j}"]["dma"], k) - base0) + res(k) - base0,
        "a2_gather_upper": lambda k: 0.0 if k == 0 else lin(
            lambda j: c[f"k{j}"]["sm_gather"], k) + res(k) - base0,
        "a2_gather_lower": lambda k: 0.0 if k == 0 else max(
            0.0, lin(lambda j: c[f"k{j}"]["sm_gather"], k) - base0) + res(k) - base0,
        "b_host_upper": lambda k: 0.0 if k == 0 else rt + lin(
            lambda j: cpu[f"k{j}_x4slices_us"], k),
        "b_host_lower": lambda k: 0.0 if k == 0 else max(
            0.0, rt + lin(lambda j: cpu[f"k{j}_x4slices_us"], k) - base0),
    }
    split_overhead_us = g["M6_k0"]["zc"] - base0
    with np.load(TALLY, allow_pickle=False) as d:
        phases = [str(x) for x in d["phases"]]
        counts = np.asarray(d["counts"][phases.index(native.TALLY_PHASE)], np.float64)
    gen = np.random.default_rng(2026)
    added = {name: 0.0 for name in arms}
    mean_k = []
    for layers, gib in STAGES:
        plan = native.choose_residency(counts, layers, int(gib * 2**30))
        for layer in layers:
            cold = np.zeros(384, bool)
            cold[plan[layer][1]] = True
            p = counts[layer] / counts[layer].sum()
            dist = k_distribution(p, cold, a.rounds, gen)
            mean_k.append(float((dist * np.arange(37)).sum()))
            for name, f in arms.items():
                added[name] += sum(dist[k] * f(k) for k in range(37) if dist[k] > 0)
    all_hot = BASE_MS + NATIVE_MS
    # every arm other than zero_copy_now still splits hot/cold: add its k = 0 cost
    for name in added:
        if name != "zero_copy_now":
            added[name] += 40 * split_overhead_us
    out = {
        "mean_cold_experts_per_layer_round": round(float(np.mean(mean_k)), 3),
        "stage_mean_k": [round(float(np.mean(mean_k[i : i + 20])), 3) for i in (0, 20)],
        "split_overhead_ms_per_round_40_layers": round(40 * split_overhead_us / 1000, 2),
        "all_hot_ms_per_round": all_hot,
        "added_ms_per_round": {k: round(v / 1000, 2) for k, v in added.items()},
        "total_ms_per_round": {k: round(all_hot + v / 1000, 1) for k, v in added.items()},
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
