"""A1 (plan 01, CPU): three-tier residency types and the tier choice.

Expected values are hand arithmetic written below, never taken from the code
under test (nvfp4_8r_tiers):

- iq_record_bytes = 11520 x (2 x bb13 + bb2): gate and up are 576 rows x 20
  blocks of 256 weights, down is 1280 rows x 9 blocks; bb = 56 B (IQ1_M,
  type 29), 66 B (IQ2_XXS, 16), 74 B (IQ2_XS, 17).
- a 2-layer toy stage whose HOST / WARM / HOT sets are worked out in the
  comments of ``toy_cases``;
- the real private tally: properties of the plan checked against the tally
  itself (aggregates printed only: counts of experts and shares of routes).

Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE (a real input is absent; it is named).
Receipt: --output JSON with status.
"""

from __future__ import annotations

import os
import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(os.environ["LANE"])
TALLY = ROOT / "engine/private/p16-tally.npz"
GGUF = ROOT / "models/gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"
E = 4_874_240  # one native record per rank (glossary of docs/plans/README.md)
BB = {29: 56, 16: 66, 17: 74}
HOT, WARM, HOST = 0, 1, 2

# MEASURED on the 8-card two-tier boot (plans/00 section 2): free HBM before
# experts per card, GiB; reserves are the serve profile's (patch P10).
FREE_GIB = {0: 29.30, 20: 28.86}
RESERVE = {0: 4_831_838_208, 20: 8_053_063_680}


class Inconclusive(RuntimeError):
    pass


def expect_raises(fails: list[str], what: str, fn) -> None:
    try:
        fn()
    except ValueError:
        return
    fails.append(f"{what}: no ValueError")


def record_bytes(fails: list[str], tiers) -> None:
    # 11520 x (2 x bb13 + bb2), by hand:
    #  (29,29) 11520 x 168 = 1,935,360   (16,16) 11520 x 198 = 2,280,960
    #  (17,17) 11520 x 222 = 2,557,440   (17,29) 11520 x 204 = 2,350,080
    #  (16,29) 11520 x 188 = 2,165,760   (29,16) 11520 x 178 = 2,050,560
    want = {(29, 29): 1_935_360, (16, 16): 2_280_960, (17, 17): 2_557_440,
            (17, 29): 2_350_080, (16, 29): 2_165_760, (29, 16): 2_050_560}
    for (t13, t2), value in want.items():
        got = tiers.iq_record_bytes(t13, t2)
        if got != value:
            fails.append(f"iq_record_bytes({t13},{t2}) = {got}, hand value {value}")
    for bad in ((15, 16), (16, 0), (18, 29)):
        expect_raises(fails, f"iq_record_bytes{bad}", lambda b=bad: tiers.iq_record_bytes(*b))


def residency_parsing(fails: list[str], tiers) -> None:
    key, share = "nvfp4_8r_residency", "nvfp4_8r_host_route_share"
    cases = [
        ({key: "two_tier"}, tiers.TwoTier()),
        ({key: "hot_iq"}, tiers.HotIq()),
        ({key: "hot_iq_host", share: 0.0}, tiers.HotIqHost(0.0)),
        ({key: "hot_iq_host", share: 0.05}, tiers.HotIqHost(0.05)),
    ]
    for extra, want in cases:
        got = tiers.parse_residency(extra)
        if got != want or type(got) is not type(want):
            fails.append(f"parse_residency({extra}) = {got!r}, want {want!r}")
    refused = [
        {}, {key: "hot_iq_host"}, {key: "two_tier", share: 0.0}, {key: "hot_iq", share: 0.01},
        {key: "hot_iq_host", share: 0.0500001}, {key: "hot_iq_host", share: -0.001},
        {key: "hot_iq_host", share: "0.0"}, {key: "hot_iq_host", share: True},
        {key: "hot_iq_host", share: float("nan")}, {key: "three_tier"}, {key: 1},
        {key: "hot_iq_host", share: None},
    ]
    for extra in refused:
        expect_raises(fails, f"parse_residency({extra})", lambda e=extra: tiers.parse_residency(e))
    expect_raises(fails, "HotIqHost(0.06)", lambda: tiers.HotIqHost(0.06))


def toy_counts() -> np.ndarray:
    """Two layers x 384 experts, six routed experts, 762 with count 0.

    Promotion cost (native record minus IQ record):
      layer 0 is (29,29): d0 = 4,874,240 - 1,935,360 = 2,938,880
      layer 1 is (17,17): d1 = 4,874,240 - 2,557,440 = 2,316,800
    Density = count / cost:
      L1 e7  500 / d1 = 2.158e-4     (first: denser than L0 e10 although fewer routes)
      L0 e10 600 / d0 = 2.042e-4
      L0 e20 300 / d0 = 1.021e-4
      L1 e8  100 / d1 = 4.316e-5
      L1 e9    4 / d1 = 1.7265e-6    (4 x d0 = 11,755,520 > 5 x d1 = 11,584,000)
      L0 e30   5 / d0 = 1.7013e-6
    then the 762 zero-count experts in (layer, expert) order.
    Stage total = 600 + 300 + 5 + 500 + 100 + 4 = 1,509 routes.
    """
    counts = np.zeros((2, 384), np.int64)
    counts[0, 10], counts[0, 20], counts[0, 30] = 600, 300, 5
    counts[1, 7], counts[1, 8], counts[1, 9] = 500, 100, 4
    return counts


IQ_BYTES = {0: 1_935_360, 1: 2_557_440}
D0, D1 = 2_938_880, 2_316_800
ALL_IQ = 384 * 1_935_360 + 384 * 2_557_440  # 384 x 4,492,800
assert ALL_IQ == 1_725_235_200


def want_arrays(default: int, sets: dict[tuple[int, int], int]) -> dict[int, np.ndarray]:
    out = {layer: np.full(384, default, np.int8) for layer in (0, 1)}
    for (layer, expert), tier in sets.items():
        out[layer][expert] = tier
    return out


def toy_cases(fails: list[str], tiers) -> None:
    counts, layers = toy_counts(), range(0, 2)

    def run(budget: int, residency):
        return tiers.choose_tiers(counts, layers, IQ_BYTES, budget, residency)

    def check(name: str, plan, want: dict[int, np.ndarray], hbm: int, budget: int) -> None:
        for layer in (0, 1):
            got = plan.tiers[layer]
            if got.dtype != np.int8 or got.shape != (384,) or not np.array_equal(got, want[layer]):
                bad = int((np.asarray(got) != want[layer]).sum()) if got.shape == (384,) else -1
                fails.append(f"{name}: layer {layer} tiers differ from the hand set in {bad} experts")
        if plan.hbm_bytes != hbm:
            fails.append(f"{name}: hbm_bytes {plan.hbm_bytes}, hand value {hbm}")
        if plan.budget_bytes != budget or plan.layers != layers:
            fails.append(f"{name}: budget/layers not carried")
        digest = hashlib.sha256(want[0].tobytes() + want[1].tobytes()).hexdigest()
        if plan.digest != digest:
            fails.append(f"{name}: digest is not sha256 of the tier arrays in layer order")

    # T1 HotIq, budget one byte short of the third promotion.
    #   need = ALL_IQ; + d1 (L1 e7) + d0 (L0 e10) = 1,730,490,880; the next
    #   candidate L0 e20 costs d0 = 2,938,880 and 2,938,879 are left: stop.
    #   L1 e8 (d1 = 2,316,800) would fit but comes after the stop: stays WARM.
    hbm = ALL_IQ + D1 + D0
    assert hbm == 1_730_490_880
    budget = hbm + D0 - 1
    check("T1 one byte short", run(budget, tiers.HotIq()),
          want_arrays(WARM, {(1, 7): HOT, (0, 10): HOT}), hbm, budget)
    # T2 the same plus one byte: L0 e20 is promoted; nothing is left for L1 e8.
    budget = hbm + D0
    check("T2 exact fit", run(budget, tiers.HotIq()),
          want_arrays(WARM, {(1, 7): HOT, (0, 10): HOT, (0, 20): HOT}), hbm + D0, budget)
    # T3 HotIqHost(0.0): HOST = exactly the 762 zero-count experts. The six
    #   others start WARM: need = 3 x 1,935,360 + 3 x 2,557,440 = 13,478,400.
    #   Budget = need + d1 + d0 = 18,734,080: L1 e7 and L0 e10 HOT, then stop.
    need = 3 * 1_935_360 + 3 * 2_557_440
    assert need == 13_478_400
    budget = need + D1 + D0
    assert budget == 18_734_080
    check("T3 share 0.0", run(budget, tiers.HotIqHost(0.0)),
          want_arrays(HOST, {(1, 7): HOT, (0, 10): HOT, (0, 20): WARM, (0, 30): WARM,
                             (1, 8): WARM, (1, 9): WARM}), budget, budget)
    # T4 HotIqHost(0.004): 0.004 x 1,509 = 6.036 routes. Ascending by (count,
    #   layer, expert): 762 zeros (sum 0), L1 e9 (sum 4 <= 6.036: HOST),
    #   L0 e30 (sum 9 > 6.036: stop). Five experts start WARM:
    #   need = 3 x 1,935,360 + 2 x 2,557,440 = 10,920,960; budget = need + d1 + d0.
    need = 3 * 1_935_360 + 2 * 2_557_440
    assert need == 10_920_960
    budget = need + D1 + D0
    check("T4 share admits one routed expert", run(budget, tiers.HotIqHost(0.004)),
          want_arrays(HOST, {(1, 7): HOT, (0, 10): HOT, (0, 20): WARM, (0, 30): WARM,
                             (1, 8): WARM}), budget, budget)
    # T5 a budget below the all-IQ tier: refused.
    expect_raises(fails, "T5 budget below all-IQ", lambda: run(ALL_IQ - 1, tiers.HotIq()))
    # T6 budget = ALL_IQ + d1: only L1 e7 is HOT, layer 0 has no HOT: refused.
    expect_raises(fails, "T6 layer without HOT", lambda: run(ALL_IQ + D1, tiers.HotIq()))
    # T7 HotIqHost(0.0) with room for all six routed experts
    #   (13,478,400 + 3 x d0 + 3 x d1 = 29,245,440): no WARM is left: refused.
    assert 13_478_400 + 3 * D0 + 3 * D1 == 29_245_440
    expect_raises(fails, "T7 layer without WARM", lambda: run(29_245_440, tiers.HotIqHost(0.0)))
    # The plan type refuses what the chooser must never build.
    good = want_arrays(WARM, {(1, 7): HOT, (0, 10): HOT})
    host = want_arrays(WARM, {(1, 7): HOT, (0, 10): HOT, (0, 0): HOST})

    def plan(arrays, residency, budget=ALL_IQ + D0 + D1):
        return tiers.TierPlan(layers, arrays, IQ_BYTES, budget, residency)

    if plan(good, tiers.HotIq()).hbm_bytes != ALL_IQ + D0 + D1:
        fails.append("TierPlan.hbm_bytes is not HOT x native + WARM x IQ bytes")
    expect_raises(fails, "TierPlan HotIq with a HOST expert", lambda: plan(host, tiers.HotIq()))
    expect_raises(fails, "TierPlan over budget", lambda: plan(good, tiers.HotIq(), ALL_IQ + D0 + D1 - 1))
    expect_raises(fails, "TierPlan value 3",
                  lambda: plan(want_arrays(WARM, {(1, 7): HOT, (0, 10): HOT, (0, 1): 3}), tiers.HotIq()))
    expect_raises(fails, "TierPlan int64 arrays",
                  lambda: plan({k: v.astype(np.int64) for k, v in good.items()}, tiers.HotIq()))
    expect_raises(fails, "TierPlan missing layer", lambda: plan({0: good[0]}, tiers.HotIq()))


def slot_maps(fails: list[str], tiers) -> None:
    # HOT {3, 10}, WARM {5, 383}, every other expert HOST. By hand:
    #  hot slots: 3 -> 0, 10 -> 1.  iq slots: 5 -> 0, 383 -> 1.
    #  cold slots count the non-HOT ids in ascending order: 0,1,2 -> 0,1,2;
    #  4 -> 3; 5 -> 4; 6..9 -> 5..8; 11 -> 9; ...; 383 -> 381 (382 non-HOT).
    #  host slot = cold slot for HOST experts, -1 for 3, 10 (HOT) and 5, 383 (WARM).
    tier = np.full(384, HOST, np.int8)
    tier[[3, 10]] = HOT
    tier[[5, 383]] = WARM
    maps = tiers.slot_maps(tier)
    want_hot = np.full(384, -1, np.int32)
    want_hot[3], want_hot[10] = 0, 1
    want_iq = np.full(384, -1, np.int32)
    want_iq[5], want_iq[383] = 0, 1
    want_cold = np.full(384, -1, np.int32)
    slot = 0
    for e in range(384):
        if e not in (3, 10):
            want_cold[e] = slot
            slot += 1
    want_host = want_cold.copy()
    want_host[[5, 383]] = -1
    for name, got, want in (("hot", maps.hot, want_hot), ("cold", maps.cold, want_cold),
                            ("host", maps.host, want_host), ("iq", maps.iq, want_iq)):
        if got.dtype != np.int32 or not np.array_equal(got, want):
            fails.append(f"slot_maps.{name} differs from the hand map")
    if want_cold[4] != 3 or want_cold[383] != 381:
        fails.append("hand cold map is itself wrong")


def gguf_types() -> dict[int, tuple[int, int]]:
    """(type13, type2) per layer from the real GGUF headers (CPU, no payload)."""
    if not GGUF.is_file():
        raise Inconclusive(f"GGUF absent: {GGUF}")
    from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index

    index = gguf_tensor_index(gguf_shard_paths(GGUF))
    out = {}
    for layer in range(40):
        t = {p: int(index[f"blk.{layer}.ffn_{p}_exps.weight"].tensor_type)
             for p in ("gate", "up", "down")}
        if t["gate"] != t["up"]:
            raise ValueError(f"layer {layer}: gate type {t['gate']} != up type {t['up']}")
        out[layer] = (t["gate"], t["down"])
    return out


def real_tally(fails: list[str], tiers) -> dict:
    if not TALLY.is_file():
        raise Inconclusive(f"tally absent: {TALLY}")
    with np.load(TALLY, allow_pickle=False) as data:
        counts = np.asarray(data["counts"][[str(p) for p in data["phases"]].index("teacher")])
    types = gguf_types()
    report = {}
    for layers in (range(0, 20), range(20, 40)):
        iq = {layer: 11520 * (2 * BB[types[layer][0]] + BB[types[layer][1]]) for layer in layers}
        budget = int(FREE_GIB[layers.start] * 2**30) - RESERVE[layers.start]
        total = int(counts[layers.start : layers.stop].sum())
        for label, residency in (("hot_iq_host_0.0", tiers.HotIqHost(0.0)), ("hot_iq", tiers.HotIq())):
            plan = tiers.choose_tiers(counts, layers, iq, budget, residency)
            tier = np.stack([plan.tiers[layer] for layer in layers])
            stage = counts[layers.start : layers.stop]
            n = {name: int((tier == value).sum()) for name, value in (("hot", HOT), ("warm", WARM), ("host", HOST))}
            routes = {name: int(stage[tier == value].sum()) for name, value in (("hot", HOT), ("warm", WARM), ("host", HOST))}
            hbm = sum(int((plan.tiers[layer] == HOT).sum()) * E + int((plan.tiers[layer] == WARM).sum()) * iq[layer]
                      for layer in layers)
            tag = f"layers {layers.start}-{layers.stop - 1} {label}"
            if sum(n.values()) != len(layers) * 384:
                fails.append(f"{tag}: tiers do not cover the stage")
            if hbm != plan.hbm_bytes or hbm > budget:
                fails.append(f"{tag}: hbm {plan.hbm_bytes} vs recomputed {hbm}, budget {budget}")
            zero = stage == 0
            if label == "hot_iq":
                if n["host"]:
                    fails.append(f"{tag}: HotIq plan has HOST experts")
            elif not np.array_equal(tier == HOST, zero):
                fails.append(f"{tag}: HOST set is not exactly the zero-route experts")
            # No HOT expert is less dense than a WARM one (count / promotion cost).
            cost = [E - iq[layer] for layer in layers]
            dens = [[Fraction(int(stage[i, e]), cost[i]) for e in range(384)] for i in range(len(layers))]
            hot_d = [dens[i][e] for i in range(len(layers)) for e in range(384) if tier[i, e] == HOT]
            warm_d = [dens[i][e] for i in range(len(layers)) for e in range(384) if tier[i, e] == WARM]
            if min(hot_d) < max(warm_d):
                fails.append(f"{tag}: a WARM expert is denser than a HOT one")
            # The walk stops at the first candidate that does not fit, so less
            # than one promotion of the costliest layer is left.
            if warm_d and budget - hbm >= E - min(iq.values()):
                fails.append(f"{tag}: {budget - hbm} bytes left, a promotion would fit")
            report[tag] = {
                "budget_bytes": budget, "hbm_bytes": hbm, "experts": n,
                "route_share": {k: round(v / total, 6) for k, v in routes.items()},
                "host_ram_native_bytes_4_ranks": 4 * (n["warm"] + n["host"]) * E,
            }
            print(tag, json.dumps(report[tag]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

    fails: list[str] = []
    record_bytes(fails, tiers)
    residency_parsing(fails, tiers)
    toy_cases(fails, tiers)
    slot_maps(fails, tiers)
    report: dict = {}
    status = None
    try:
        report = real_tally(fails, tiers)
    except Inconclusive as exc:
        status = "INCONCLUSIVE"
        report = {"missing": str(exc)}
    if fails:
        status = "FAIL"
    elif status is None:
        status = "PASS"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"check": "A1", "status": status, "failures": fails,
                                    "real_tally": report}, indent=1) + "\n")
    print(status, *fails, sep="\n")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[status]


if __name__ == "__main__":
    sys.exit(main())
