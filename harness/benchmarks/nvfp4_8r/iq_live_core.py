"""Card 8.5 (rank 7): the pure part of bench_iq_live_routes: routings, the
bench's own checks and the decision rule. numpy and the standard library
only, so it runs on any CPU (test_iq_live_core.py).

The rule is the one in agents/iq-live/DECISION-RULE.md, written before any
bench data existed; the constants below are copied from it, not tuned.
"""

from __future__ import annotations

import statistics

import numpy as np

NUM_EXPERTS = 384
TOP_K = 6
HOT, WARM = 0, 1
K_VALUES = (0, 1, 3, 6, 12, 16, 36)
TIMER_STEP_US = 1.03          # one step of the event timer seen in the receipts (1.02 to 1.03 us)
SETTLED_TOL = 0.03
MONO_TOL = 0.03
MONO_FLOOR_US = 2.05
ADD_TOL = 0.10
ADD_FLOOR_US = 2.05
FALSIFIER_US = 60.0           # card 8.5: under this at 6 live routes, no build
LAUNCH_FLOOR_US = 15.4        # three launches at 5.12 us (parts-20261111T010903Z.json)
REMOVABLE_US = 30.0           # 1.2 ms per round over 40 layers
DECISION_K = 6


def bench_tier() -> np.ndarray:
    """int8 [384]: WARM where id % 4 == 2 (96 experts), HOT elsewhere (288);
    no HOST expert, the hot_iq shape."""
    ident = np.arange(NUM_EXPERTS)
    return np.where(ident % 4 == 2, WARM, HOT).astype(np.int8)


def live_routes(k: int, m: int, tier: np.ndarray, seed: int) -> np.ndarray:
    """int32 [m, 6]: 6m distinct experts, exactly k of the routes on WARM
    experts, the rest on HOT ones, at seeded random positions."""
    n = m * TOP_K
    if not 0 <= k <= n:
        raise ValueError(f"k = {k} live routes of {n}")
    warm = np.flatnonzero(tier == WARM)
    hot = np.flatnonzero(tier == HOT)
    if k > warm.size or n - k > hot.size:
        raise ValueError("the tier array cannot supply that many distinct experts")
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.choice(warm, k, replace=False),
                          rng.choice(hot, n - k, replace=False)])
    return ids[rng.permutation(n)].reshape(m, TOP_K).astype(np.int32)


def shared_routes(j: int, m: int, tier: np.ndarray, seed: int) -> np.ndarray:
    """int32 [m, 6]: every row routes to the same 6 experts, j of them WARM:
    6m routes, j * m live ones on j distinct experts (the most sharing)."""
    if not 0 <= j <= TOP_K:
        raise ValueError(f"j = {j} WARM experts per row")
    rng = np.random.default_rng(seed)
    row = np.concatenate([rng.choice(np.flatnonzero(tier == WARM), j, replace=False),
                          rng.choice(np.flatnonzero(tier == HOT), TOP_K - j, replace=False)])
    row = row[rng.permutation(TOP_K)]
    return np.tile(row, (m, 1)).astype(np.int32)


def summarize(values: list[float]) -> dict:
    med = statistics.median(values)
    tol = max(SETTLED_TOL * med, TIMER_STEP_US)
    return {"median_us": round(med, 2), "values_us": [round(v, 2) for v in values],
            "settled": all(abs(v - med) <= tol for v in values)}


def check_monotone(a: dict[int, float]) -> list[str]:
    """C1: time does not fall as k rises. Violations as text; empty = pass."""
    ks = sorted(a)
    bad = []
    for k1, k2 in zip(ks, ks[1:]):
        tol = max(MONO_TOL * a[k1], MONO_FLOOR_US)
        if a[k2] < a[k1] - tol:
            bad.append(f"a({k2}) = {a[k2]:.2f} < a({k1}) = {a[k1]:.2f} - {tol:.2f}")
    return bad


def check_additive(a: dict[int, float], d: dict[int, float]) -> list[str]:
    """C2: (b) - (c) equals (a) at the same k within 10 % (floor 2.05 us)."""
    bad = []
    for k in sorted(a):
        tol = max(ADD_TOL * a[k], ADD_FLOOR_US)
        if abs(d[k] - a[k]) > tol:
            bad.append(f"k = {k}: b - c = {d[k]:.2f} vs a = {a[k]:.2f} (tolerance {tol:.2f})")
    return bad


def layer_verdict(parts: dict[int, dict[str, dict]]) -> dict:
    """One layer's checks from {k: {"a"|"b"|"c": summarize(...)}} at M = 6."""
    a = {k: parts[k]["a"]["median_us"] for k in parts}
    d = {k: round(parts[k]["b"]["median_us"] - parts[k]["c"]["median_us"], 2) for k in parts}
    need = (0, DECISION_K, 36)
    missing = [k for k in need if k not in parts]
    if missing:
        raise ValueError(f"the layer has no part at k = {missing}")
    unsettled = [k for k in need if not parts[k]["a"]["settled"]]
    c1, c2 = check_monotone(a), check_additive(a, d)
    slope = (a[36] - a[0]) / 36.0
    ideal = LAUNCH_FLOOR_US + DECISION_K * slope
    return {
        "a_us": a, "b_minus_c_us": d,
        "c1_monotone": c1, "c2_additive": c2, "a_unsettled_at": unsettled,
        "passes": not c1 and not c2 and not unsettled,
        "slope_us_per_route": round(slope, 3),
        "ideal_k6_us": round(ideal, 2),
        "removable_k6_us": round(a[DECISION_K] - ideal, 2),
    }


def decide(layers: dict[str, dict]) -> dict:
    """The verdict from {layer name: layer_verdict(...)}."""
    ok = {name: v for name, v in layers.items() if v["passes"]}
    out: dict = {"layers_passing": sorted(ok), "rule": {
        "falsifier_us": FALSIFIER_US, "removable_us": REMOVABLE_US,
        "launch_floor_us": LAUNCH_FLOOR_US, "k": DECISION_K}}
    if len(ok) < 2:
        out["verdict"] = "INCONCLUSIVE"
        out["why"] = f"{len(ok)} of {len(layers)} layers pass the bench's own checks; two are needed"
        return out
    a6 = statistics.median(v["a_us"][DECISION_K] for v in ok.values())
    r6 = statistics.median(v["removable_k6_us"] for v in ok.values())
    out["A6_us"], out["R6_us"] = round(a6, 2), round(r6, 2)
    if a6 < FALSIFIER_US:
        out["verdict"] = "NO BUILD"
        out["why"] = f"A6 = {a6:.2f} us < {FALSIFIER_US} us (the card's falsifier)"
    elif r6 >= REMOVABLE_US:
        out["verdict"] = "BUILD WARRANTED"
        out["why"] = f"A6 = {a6:.2f} us >= {FALSIFIER_US} and R6 = {r6:.2f} us >= {REMOVABLE_US}"
    else:
        out["verdict"] = "NO BUILD"
        out["why"] = (f"A6 = {a6:.2f} us >= {FALSIFIER_US} but R6 = {r6:.2f} us < {REMOVABLE_US}: "
                      "the op already follows the live routes")
    return out
