"""Plan 06: the pure part of bench_iq_tail: the one-card speed rule. The
standard library only, so it runs on any CPU (test_iq_tail_core.py).

The rule is plan 06 section 7 ("Speed, one card"), written before any
bench_iq_tail data existed; the constants below are copied from it, not tuned.
Per layer and per (M, k) the bench times four captured graphs in the order
torch, fused, fused, torch (t1, f1, f2, t2; us per replay):

  S1  b_torch(6) - b_fused(6) >= 30.0 us at M = 6, on at least two layers
  S2  b_fused(k) - b_torch(k) <= 2.05 us at every k, both M
  S3  t1 and t2 agree within max(3 %, 2.05 us) at every k, and so do f1 and
      f2; a layer where they do not is unsettled and gives no input

with b_torch = mean(t1, t2) and b_fused = mean(f1, f2).
"""

from __future__ import annotations

SAVE_US = 30.0
SLOWER_US = 2.05
AGREE_TOL = 0.03
AGREE_FLOOR_US = 2.05
DECISION_M = "M6"
DECISION_K = 6
ARMS = ("t1", "f1", "f2", "t2")


def cell(times: dict[str, float]) -> dict:
    """One (M, k): {"t1", "f1", "f2", "t2"} medians -> means, saving, S3."""
    if sorted(times) != sorted(ARMS):
        raise ValueError(f"a cell needs {ARMS}, got {sorted(times)}")
    torch_us = (times["t1"] + times["t2"]) / 2.0
    fused_us = (times["f1"] + times["f2"]) / 2.0
    disagree = []
    for name, a, b, mean in (("torch", times["t1"], times["t2"], torch_us),
                             ("fused", times["f1"], times["f2"], fused_us)):
        tol = max(AGREE_TOL * mean, AGREE_FLOOR_US)
        if abs(a - b) > tol:
            disagree.append(f"{name}: {a:.2f} vs {b:.2f} (tolerance {tol:.2f})")
    return {"torch_us": round(torch_us, 2), "fused_us": round(fused_us, 2),
            "saved_us": round(torch_us - fused_us, 2), "runs_disagree": disagree}


def layer_tail_verdict(parts: dict[str, dict[int, dict[str, float]]]) -> dict:
    """One layer from {"M6": {k: {arm: us}}, "M1": {...}}."""
    if DECISION_M not in parts or DECISION_K not in parts[DECISION_M]:
        raise ValueError(f"the layer has no part at {DECISION_M} k = {DECISION_K}")
    cells = {m: {k: cell(v) for k, v in rows.items()} for m, rows in parts.items()}
    unsettled = [f"{m} k{k} {d}" for m, rows in cells.items() for k, c in rows.items()
                 for d in c["runs_disagree"]]
    slower = [f"{m} k{k}: fused {c['fused_us']:.2f} vs torch {c['torch_us']:.2f}"
              for m, rows in cells.items() for k, c in rows.items()
              if c["fused_us"] - c["torch_us"] > SLOWER_US]
    saved6 = cells[DECISION_M][DECISION_K]["saved_us"]
    return {"cells": cells, "s3_unsettled": unsettled, "settled": not unsettled,
            "s2_slower": slower, "saved_k6_us": saved6, "s1_holds": saved6 >= SAVE_US}


def decide_tail(layers: dict[str, dict]) -> dict:
    """The one-card verdict from {layer name: layer_tail_verdict(...)}."""
    settled = {n: v for n, v in layers.items() if v["settled"]}
    s2 = {n: v["s2_slower"] for n, v in settled.items() if v["s2_slower"]}
    s1_yes = sorted(n for n, v in settled.items() if v["s1_holds"])
    s1_no = sorted(n for n, v in settled.items() if not v["s1_holds"])
    out: dict = {
        "rule": {"save_us": SAVE_US, "slower_us": SLOWER_US, "agree_tol": AGREE_TOL,
                 "agree_floor_us": AGREE_FLOOR_US, "k": DECISION_K},
        "layers_settled": sorted(settled), "s1_holds_on": s1_yes, "s1_fails_on": s1_no,
        "s2_slower_on": s2,
        "saved_k6_us": {n: v["saved_k6_us"] for n, v in layers.items()},
    }
    if s2:
        out["verdict"] = "NOT ACCEPTED"
        out["why"] = f"S2: the fused forward is slower by more than {SLOWER_US} us on {sorted(s2)}"
    elif len(s1_no) >= 2:
        out["verdict"] = "NOT ACCEPTED"
        out["why"] = f"S1: under {SAVE_US} us saved at k = {DECISION_K} on {s1_no}"
    elif len(s1_yes) >= 2:
        out["verdict"] = "BUILD ACCEPTED ON ONE CARD"
        out["why"] = (f"S1 holds on {s1_yes}, S2 holds on every settled layer "
                      f"({len(settled)} of {len(layers)} settled)")
    else:
        out["verdict"] = "INCONCLUSIVE"
        out["why"] = (f"{len(settled)} of {len(layers)} layers settled, S1 holds on {s1_yes} and "
                      f"fails on {s1_no}: two settled layers on one side are needed")
    return out
