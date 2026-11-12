"""CPU test of iq_live_core. Every expected value is arithmetic done by hand
from DECISION-RULE.md, not a value recorded from the code.

    python test_iq_live_core.py        (numpy only; exit 0 PASS, 1 FAIL)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iq_live_core as core  # noqa: E402

FAILS: list[str] = []


def expect(name: str, got, want) -> None:
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")


def part(a: float, b: float, c: float, settled: bool = True) -> dict:
    s = lambda v, ok=True: {"median_us": v, "values_us": [v] * 5, "settled": ok}  # noqa: E731
    return {"a": s(a, settled), "b": s(b), "c": s(c)}


def layer(a_by_k: dict[int, float], extra: float = 0.0, base: float = 300.0) -> dict:
    """A layer whose forward costs base without the op and base + a + extra with it."""
    return {k: part(a, base + a + extra, base) for k, a in a_by_k.items()}


def main() -> int:
    tier = core.bench_tier()
    expect("tier WARM count", int((tier == core.WARM).sum()), 96)   # ids 2, 6, ..., 382
    expect("tier HOT count", int((tier == core.HOT).sum()), 288)
    expect("tier dtype", str(tier.dtype), "int8")
    expect("tier of id 2 / 3", (int(tier[2]), int(tier[3])), (core.WARM, core.HOT))

    for m in (1, 6):
        for k in core.K_VALUES:
            if k > 6 * m:
                continue
            for seed in range(20):
                ids = core.live_routes(k, m, tier, seed)
                tag = f"live_routes(k={k}, m={m}, seed={seed})"
                expect(f"{tag} shape", ids.shape, (m, 6))
                expect(f"{tag} dtype", str(ids.dtype), "int32")
                expect(f"{tag} distinct", len(set(ids.ravel().tolist())), 6 * m)
                expect(f"{tag} live", int((ids % 4 == 2).sum()), k)
                expect(f"{tag} range", bool(((ids >= 0) & (ids < 384)).all()), True)
    expect("live_routes differs by seed",
           bool((core.live_routes(6, 6, tier, 1) != core.live_routes(6, 6, tier, 2)).any()), True)
    expect("live_routes repeats per seed",
           bool((core.live_routes(6, 6, tier, 1) == core.live_routes(6, 6, tier, 1)).all()), True)
    for bad in ((37, 6), (-1, 6), (7, 1)):
        try:
            core.live_routes(bad[0], bad[1], tier, 0)
            FAILS.append(f"live_routes{bad} did not raise")
        except ValueError:
            pass

    for j in (0, 1, 2, 6):
        ids = core.shared_routes(j, 6, tier, 3)
        expect(f"shared_routes({j}) rows equal", bool((ids == ids[0]).all()), True)
        expect(f"shared_routes({j}) row distinct", len(set(ids[0].tolist())), 6)
        expect(f"shared_routes({j}) live routes", int((ids % 4 == 2).sum()), 6 * j)

    # settled: median 100, tolerance max(3, 1.03) = 3
    expect("settled inside", core.summarize([100, 101, 99, 103, 97])["settled"], True)
    expect("settled outside", core.summarize([100, 101, 99, 103.5, 97])["settled"], False)
    # median 5.12, tolerance max(0.15, 1.03) = 1.03: one timer step is settled, two are not
    expect("settled one step", core.summarize([5.12, 5.12, 5.12, 6.14, 5.12])["settled"], True)
    expect("settled two steps", core.summarize([5.12, 5.12, 5.12, 7.17, 5.12])["settled"], False)
    expect("median", core.summarize([3.0, 1.0, 2.0, 5.0, 4.0])["median_us"], 3.0)

    # C1: tolerance max(3 % of the earlier value, 2.05)
    expect("mono ok", core.check_monotone({0: 30.0, 1: 28.0, 3: 40.0}), [])        # falls 2.0 <= 2.05
    expect("mono bad", len(core.check_monotone({0: 30.0, 1: 27.9, 3: 40.0})), 1)   # falls 2.1 > 2.05
    expect("mono 3 %", core.check_monotone({16: 200.0, 36: 194.5}), [])            # 5.5 <= 6.0
    expect("mono 3 % bad", len(core.check_monotone({16: 200.0, 36: 193.9})), 1)    # 6.1 > 6.0
    # C2: tolerance max(10 % of a, 2.05)
    expect("add ok", core.check_additive({6: 70.0}, {6: 76.9}), [])                # 6.9 <= 7.0
    expect("add bad", len(core.check_additive({6: 70.0}, {6: 77.1})), 1)           # 7.1 > 7.0
    expect("add floor ok", core.check_additive({0: 10.0}, {0: 12.0}), [])          # 2.0 <= 2.05
    expect("add floor bad", len(core.check_additive({0: 10.0}, {0: 12.1})), 1)     # 2.1 > 2.05

    # layer_verdict arithmetic: a(0) = 30, a(36) = 210: slope 5.0; ideal(6) = 15.4 + 30 = 45.4
    a = {0: 30.0, 1: 40.0, 3: 50.0, 6: 80.0, 12: 110.0, 16: 130.0, 36: 210.0}
    v = core.layer_verdict(layer(a))
    expect("slope", v["slope_us_per_route"], 5.0)
    expect("ideal", v["ideal_k6_us"], 45.4)
    expect("removable", v["removable_k6_us"], 34.6)                                 # 80 - 45.4
    expect("passes", v["passes"], True)
    expect("b - c", v["b_minus_c_us"][6], 80.0)

    # decide: three such layers: A6 = 80 >= 60, R6 = 34.6 >= 30
    d = core.decide({"L05": v, "L12": v, "L27": v})
    expect("verdict build", (d["verdict"], d["A6_us"], d["R6_us"]), ("BUILD WARRANTED", 80.0, 34.6))
    # A6 under the falsifier: a(6) = 59.9
    lo = core.layer_verdict(layer({**a, 6: 59.9}))
    expect("verdict falsifier", core.decide({"L05": lo, "L12": lo, "L27": lo})["verdict"], "NO BUILD")
    # A6 = 60.0 exactly is not under the falsifier; removable = 60 - 45.4 = 14.6 < 30
    eq = core.layer_verdict(layer({**a, 6: 60.0}))
    de = core.decide({"L05": eq, "L12": eq, "L27": eq})
    expect("verdict follows routes", (de["verdict"], de["R6_us"]), ("NO BUILD", 14.6))
    expect("why names the routes", "already follows" in de["why"], True)
    # removable exactly 30.0 builds: a(6) = 75.4
    ex = core.layer_verdict(layer({**a, 6: 75.4}))
    expect("verdict at 30.0", core.decide({"L05": ex, "L12": ex, "L27": ex})["verdict"],
           "BUILD WARRANTED")
    # medians over layers: a(6) = 59, 61, 90 -> A6 = 61; removable 13.6, 15.6, 44.6 -> R6 = 15.6
    ls = {n: core.layer_verdict(layer({**a, 6: x})) for n, x in (("L05", 59.0), ("L12", 61.0),
                                                                  ("L27", 90.0))}
    dm = core.decide(ls)
    expect("median verdict", (dm["verdict"], dm["A6_us"], dm["R6_us"]), ("NO BUILD", 61.0, 15.6))
    # a layer failing C2 (the op costs 12 us more inside the forward: 15 % of 80) gives no input
    bad = core.layer_verdict(layer(a, extra=12.0))
    expect("C2 fail", bad["passes"], False)
    expect("two layers decide", core.decide({"L05": v, "L12": v, "L27": bad})["verdict"],
           "BUILD WARRANTED")
    expect("one layer cannot", core.decide({"L05": v, "L12": bad, "L27": bad})["verdict"],
           "INCONCLUSIVE")
    # an unsettled a(6) gives no input
    parts = layer(a)
    parts[6] = part(80.0, 380.0, 300.0, settled=False)
    expect("unsettled", core.layer_verdict(parts)["passes"], False)
    # a non-monotone layer gives no input: a(12) = 70 after a(6) = 80 (falls 10 > 2.4)
    expect("C1 fail", core.layer_verdict(layer({**a, 12: 70.0}))["passes"], False)

    if FAILS:
        print("FAIL")
        for f in FAILS:
            print(" ", f)
        return 1
    print("PASS test_iq_live_core")
    return 0


if __name__ == "__main__":
    sys.exit(main())
