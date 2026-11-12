"""Rank 4 tally mix and route budget (campaign card 8.2), CPU test.

Expected values are arithmetic done here by hand, not the functions' output:

  m1  Two tallies, one layer. A: expert 0 has 3 routes, expert 1 has 1.
      B: expert 1 has 1, expert 2 has 1. Equal weights: per layer A gives
      0.75 / 0.25 / 0, B gives 0 / 0.5 / 0.5, so the mix is
      0.375 / 0.375 / 0.25 of 2**40. Weights 1 and 3: 0.1875 / 0.4375 / 0.375.
  m2  Only ratios matter: weights (2, 2) equal (1, 1); a tally multiplied by
      10 gives the same mix (a longer run does not outweigh a shorter one).
  m3  Refused: no weights, a missing weight, weight 0, a negative or
      non-finite weight, a float tally, a layer without routes, tallies of
      different layer counts.
  s1  Route budget to share. 0.5 routes per round in a stage of 20 layers is
      0.5 / 720 = 0.000694444..., so the share is 0.000694 (six places,
      rounded down); 0.000694 x 720 = 0.49968 <= 0.5 and 0.000695 x 720 =
      0.5004 > 0.5. A budget of 0 gives 0.0; 1.0 gives 0.001388; a budget
      above the engine's limit (0.05 x 720 = 36 routes) is refused.
  t1  Shares by tier. One stage of two layers; layer 0 routes 8 / 2 / 0 to a
      HOT / WARM / HOST expert, layer 1 routes 1 / 1 / 2: the layer shares are
      0.8 / 0.2 / 0 and 0.25 / 0.25 / 0.5, their mean 0.525 / 0.225 / 0.25;
      0.25 of 2 x 36 routes is 18 HOST routes per round.
  p1  The property the setting is chosen for: tiers chosen by the engine's
      choose_tiers from a mix of three seeded tallies with the share of s1
      send, by the mix, at most 0.5 routes of a 720-route round to HOST
      (counted here from the tier arrays), and at least one expert is HOST.

Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tally_mix as tm
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers


def tally(rows: list[dict[int, int]]) -> np.ndarray:
    out = np.zeros((len(rows), 384), np.int64)
    for layer, row in enumerate(rows):
        for e, c in row.items():
            out[layer, e] = c
    return out


def refused(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    fails: list[str] = []
    one = 1 << 40
    ta, tb = tally([{0: 3, 1: 1}]), tally([{1: 1, 2: 1}])

    def first3(mix: np.ndarray) -> list[int]:
        return mix[0, :3].tolist() + [int(mix[0, 3:].sum())]

    got = first3(tm.mix_tallies([ta, tb], [1, 1]))
    if got != [3 * one // 8, 3 * one // 8, one // 4, 0]:
        fails.append(f"m1 equal weights: {got}")
    got = first3(tm.mix_tallies([ta, tb], [1, 3]))
    if got != [3 * one // 16, 7 * one // 16, 3 * one // 8, 0]:
        fails.append(f"m1 weights 1 and 3: {got}")
    base = tm.mix_tallies([ta, tb], [1, 1])
    if not np.array_equal(base, tm.mix_tallies([ta, tb], [2.0, 2.0])) or \
            not np.array_equal(base, tm.mix_tallies([ta * 10, tb], [1, 1])):
        fails.append("m2: the mix depends on more than the weights' ratio and the layer shares")
    bad = [
        lambda: tm.mix_tallies([ta, tb], []),
        lambda: tm.mix_tallies([ta, tb], [1]),
        lambda: tm.mix_tallies([ta, tb], [1, 0]),
        lambda: tm.mix_tallies([ta, tb], [1, -1]),
        lambda: tm.mix_tallies([ta, tb], [1, float("nan")]),
        lambda: tm.mix_tallies([ta.astype(np.float64), tb], [1, 1]),
        lambda: tm.mix_tallies([tally([{0: 1}, {}]), tally([{0: 1}, {1: 1}])], [1, 1]),
        lambda: tm.mix_tallies([ta, tally([{0: 1}, {1: 1}])], [1, 1]),
        lambda: tm.mix_tallies([], []),
    ]
    not_refused = [i for i, fn in enumerate(bad) if not refused(fn)]
    if not_refused:
        fails.append(f"m3: cases {not_refused} were accepted")

    share = tm.host_route_share(0.5, 20)
    if share != 0.000694 or not (0.000694 * 720 <= 0.5 < 0.000695 * 720):
        fails.append(f"s1: 0.5 routes per round over 20 layers gave {share!r}")
    if tm.host_route_share(0, 20) != 0.0 or tm.host_route_share(1.0, 20) != 0.001388:
        fails.append("s1: budgets 0 and 1.0")
    if not refused(lambda: tm.host_route_share(36.5, 20)) or not refused(lambda: tm.host_route_share(0.5, 0)) \
            or not refused(lambda: tm.host_route_share(-0.1, 20)):
        fails.append("s1: a budget above the engine's limit, a stage without layers or a negative budget was accepted")

    plan = {layer: np.full(384, tiers.Tier.WARM, np.int8) for layer in (0, 1)}
    for layer in plan:
        plan[layer][0], plan[layer][2] = tiers.Tier.HOT, tiers.Tier.HOST
    t = tally([{0: 8, 1: 2}, {0: 1, 1: 1, 2: 2}])
    got_shares = tm.tier_route_shares(t, plan, range(0, 2))
    if [round(v, 12) for v in got_shares] != [0.525, 0.225, 0.25] or \
            round(tm.host_routes_per_round(got_shares[2], 2), 9) != 18.0:
        fails.append(f"t1: shares {got_shares}")

    rng = np.random.default_rng(8208)
    layers = range(0, 20)
    # Skewed counts (most experts rare, a few heavy), capped so a layer total stays far below 2**40.
    three = [(np.minimum(rng.zipf(1.3, (20, 384)), 100_000) - 1).astype(np.int64) * scale
             + (rng.random((20, 384)) < 0.9) for scale in (1, 50, 3)]
    mixed = tm.mix_tallies(three, [1, 1, 1])
    iq = {layer: 2_000_000 for layer in layers}
    chosen = tiers.choose_tiers(mixed, layers, iq, 15 * 2**30, tiers.HotIqHost(share))
    host_counts = sum(int(mixed[layer][chosen.tiers[layer] == tiers.Tier.HOST].sum()) for layer in layers)
    per_round = host_counts / int(mixed[:20].sum()) * 720
    n_host = chosen.count(tiers.Tier.HOST)
    if not 0 < per_round <= 0.5 or n_host == 0:
        fails.append(f"p1: {n_host} HOST experts take {per_round:.4f} routes of a 720-route round by the mix")
    res = {"check": "rank4-tally-mix-cpu", "share_for_half_a_route": share,
           "p1": {"host_experts": n_host, "host_routes_per_round_by_mix": round(per_round, 4)},
           "fails": fails, "verdict": "FAIL" if fails else "PASS"}
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
