#!/usr/bin/env python3
"""Score the NVFP4 closeness test: hot_iq and Volta sPLIt against the all-NVFP4 reference.

  score.py --plan PLAN --key KEYFILE --ref PART... --hot PART... [--volta PART...] --out FILE
  score.py --plan PLAN --key-id HEX ...   the same on published receipts, the key itself deleted
  score.py --dry --plan PLAN --key KEYFILE --ref PART... --hot PART... [--volta PART...]   inputs and rules only
  score.py --self-test                                                                     synthetic receipts

The rules are the constants of closeness_lib.py and the functions below, fixed before any capture (DESIGN.md
section 4). The score file holds every figure, its bootstrap 95 % interval and every verdict. Exit 0 when scored
(whatever the verdicts), 3 when an input cannot be used (INCONCLUSIVE, nothing written).

Layers:
  floor        the reference's floor pass against its main pass on the same items: identical or not, and the same
               figures as the arms (the noise floor)
  contexts     per arm, decode positions 1 .. min(first divergence, 31) of every context: top-1 agreement, mean
               |logprob difference| of the reference's token, mean KL(reference || arm) over the tokens in both
               top-20 lists plus one bucket for the rest (a lower bound on the full KL); position 0 (prompt
               reading) separately; by text kind
  common       the positions where BOTH arms still match the reference (1 .. min of the two first divergences):
               the same figures for both arms on the same positions, the paired difference and the ratio
  divergence   first-divergence position per context: share still identical through 0, 1, 2, 4, 8, 16, 31, median
  mc           accuracy of each arm, paired difference from the reference, share of answers that differ from the
               reference's, and hot_iq against Volta sPLIt on that share
Intervals: percentile bootstrap, 2,000 resamples, seed 20261110, over whole contexts (within text kind) or MC items.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closeness_lib as cl  # noqa: E402

INCONCLUSIVE = "INCONCLUSIVE"


def ci_json(point, ci, undefined=0) -> dict:
    return {"value": point, "ci95": list(ci) if ci is not None else None, "undefined_resamples": undefined}


def verdict_diff(ci) -> str:
    """ci of (Volta sPLIt figure - hot_iq figure) for a distance: above 0 means hot_iq is closer."""
    if ci is None:
        return INCONCLUSIVE
    if ci[0] > 0:
        return "HOT_IQ CLOSER"
    if ci[1] < 0:
        return "HOT_IQ FARTHER"
    return INCONCLUSIVE


def verdict_vs_floor(arm_ci, floor_ci) -> str:
    if arm_ci is None or floor_ci is None:
        return INCONCLUSIVE
    overlap = arm_ci[0] <= floor_ci[1] and floor_ci[0] <= arm_ci[1]
    return "INDISTINGUISHABLE AT THIS SAMPLE SIZE" if overlap else "DIFFERS FROM THE REFERENCE"


# ------------------------------------------------------------------------------------------- context figures

def ctx_arrays(rows: list[cl.ContextResult]):
    import numpy as np

    f = lambda name: np.array([getattr(r, name) for r in rows], dtype=float)  # noqa: E731
    return {k: f(k) for k in ("n", "disagree", "near_tie", "sum_absdlp", "censored", "sum_kl", "sum_shared",
                              "first_div", "pos0_absdlp", "pos0_kl")} | \
        {"pos0_agree": np.array([1.0 if r.pos0_agree else 0.0 for r in rows])}


def ratio(num, den):
    def stat(idx):
        d = den[idx].sum()
        return None if d == 0 else float(num[idx].sum() / d)
    return stat


def mean_of(x):
    return lambda idx: float(x[idx].mean())


def context_figures(rows: list[cl.ContextResult], strata: list[str] | None) -> dict:
    """Pooled figures over decode positions, position 0, and first divergence, each with its interval."""
    import numpy as np

    if not rows:
        return {"contexts": 0}
    a = ctx_arrays(rows)
    n = len(rows)
    out = {"contexts": n, "positions": int(a["n"].sum())}
    for name, stat in (("disagreement_rate", ratio(a["disagree"], a["n"])),
                       ("mean_abs_dlogprob", ratio(a["sum_absdlp"], a["n"])),
                       ("mean_kl", ratio(a["sum_kl"], a["n"])),
                       ("pos0_agreement", mean_of(a["pos0_agree"])),
                       ("pos0_mean_abs_dlogprob", mean_of(a["pos0_absdlp"])),
                       ("pos0_mean_kl", mean_of(a["pos0_kl"]))):
        point, ci, und = cl.bootstrap(stat, n, strata)
        out[name] = ci_json(point, ci, und)
    dr = out["disagreement_rate"]
    out["top1_agreement"] = {"value": None if dr["value"] is None else 1 - dr["value"],
                             "ci95": None if dr["ci95"] is None else [1 - dr["ci95"][1], 1 - dr["ci95"][0]]}
    dis = a["disagree"].sum()
    out["near_tie_share_of_disagreements"] = None if dis == 0 else float(a["near_tie"].sum() / dis)
    out["censored_share"] = float(a["censored"].sum() / a["n"].sum()) if a["n"].sum() else None
    out["mean_shared_reference_mass"] = float(a["sum_shared"].sum() / a["n"].sum()) if a["n"].sum() else None
    d = a["first_div"]
    surv = {}
    for p in cl.SURVIVAL_AT:
        point, ci, und = cl.bootstrap(lambda idx, p=p: float((d[idx] > p).mean()), n, strata)
        surv[str(p)] = ci_json(point, ci, und)
    out["identical_through"] = surv
    point, ci, und = cl.bootstrap(lambda idx: float(np.median(d[idx])), n, strata)
    out["median_first_divergence"] = ci_json(point, ci, und)
    out["median_is_censored"] = bool(point >= cl.CONT_TOKENS)
    return out


def by_kind(rows: list[cl.ContextResult]) -> dict:
    return {k: context_figures([r for r in rows if r.kind == k], None) for k in cl.KINDS}


def paired(hot: list[cl.ContextResult], volta: list[cl.ContextResult], strata: list[str]) -> dict:
    """Same contexts, same positions: Volta sPLIt minus hot_iq for each distance, and hot_iq over Volta sPLIt."""
    h, v = ctx_arrays(hot), ctx_arrays(volta)
    out = {}
    for name, num in (("disagreement_rate", "disagree"), ("mean_abs_dlogprob", "sum_absdlp"), ("mean_kl", "sum_kl")):
        rh, rv = ratio(h[num], h["n"]), ratio(v[num], v["n"])

        def diff(idx, rh=rh, rv=rv):
            a, b = rv(idx), rh(idx)
            return None if a is None or b is None else a - b

        def rat(idx, rh=rh, rv=rv):
            a, b = rh(idx), rv(idx)
            return None if a is None or b is None or b == 0 else a / b

        pd, cd, ud = cl.bootstrap(diff, len(hot), strata)
        pr, cr, ur = cl.bootstrap(rat, len(hot), strata)
        out[name] = {"volta_minus_hot": ci_json(pd, cd, ud), "verdict": verdict_diff(cd),
                     "hot_over_volta": ci_json(pr, cr, ur)}
    longer = sum(1 for x, y in zip(hot, volta) if x.first_div > y.first_div)
    shorter = sum(1 for x, y in zip(hot, volta) if x.first_div < y.first_div)
    out["first_divergence_hot_vs_volta"] = {"hot_later": longer, "hot_earlier": shorter,
                                            "same": len(hot) - longer - shorter}
    return out


# ------------------------------------------------------------------------------------------- MC figures

def mc_figures(items: list[str], gold: dict, ref: dict, arm: dict | None, strata=None) -> dict:
    import numpy as np

    n = len(items)
    if n == 0:
        return {"items": 0}
    rc = np.array([1.0 if ref[i].answer == gold[i] else 0.0 for i in items])
    out = {"items": n}
    p, c, u = cl.bootstrap(mean_of(rc), n, strata)
    out["reference_accuracy"] = ci_json(p, c, u)
    out["reference_no_answer"] = sum(1 for i in items if ref[i].answer is None)
    out["reference_answer_at_position_0"] = sum(1 for i in items if ref[i].answer_pos == 0)
    if arm is None:
        return out
    ac = np.array([1.0 if arm[i].answer == gold[i] else 0.0 for i in items])
    dif = np.array([1.0 if arm[i].answer != ref[i].answer else 0.0 for i in items])
    p, c, u = cl.bootstrap(mean_of(ac), n, strata)
    out["accuracy"] = ci_json(p, c, u)
    p, c, u = cl.bootstrap(lambda idx: float(ac[idx].mean() - rc[idx].mean()), n, strata)
    out["accuracy_minus_reference"] = ci_json(p, c, u)
    out["accuracy_verdict"] = (INCONCLUSIVE if c is None else
                               "NOT DISTINGUISHABLE FROM THE REFERENCE AT THIS SAMPLE SIZE" if c[0] <= 0 <= c[1] else
                               "LOWER THAN THE REFERENCE" if c[1] < 0 else "HIGHER THAN THE REFERENCE")
    p, c, u = cl.bootstrap(mean_of(dif), n, strata)
    out["answer_differs_from_reference"] = ci_json(p, c, u)
    out["discordant"] = {"arm_right_reference_wrong": int(((ac == 1) & (rc == 0)).sum()),
                         "arm_wrong_reference_right": int(((ac == 0) & (rc == 1)).sum())}
    out["no_answer"] = sum(1 for i in items if arm[i].answer is None)
    out["answer_at_position_0"] = sum(1 for i in items if arm[i].answer_pos == 0)
    return out


# ------------------------------------------------------------------------------------------- score

class Unusable(Exception):
    pass


def end_status(path) -> str:
    """The status on a part's end line; a part cut off before its end line is reported as NO END LINE (its item
    lines were written one by one with fsync, so they still count)."""
    last = Path(path).read_text().splitlines()[-1]
    d = json.loads(last)
    return d["status"] if d.get("type") == "end" else "NO END LINE"


def score(plan: cl.Plan, kid: str, ref_parts, hot_parts, volta_parts, min_contexts=cl.MIN_CONTEXTS,
          min_mc=cl.MIN_MC) -> dict:
    try:
        ref, ref_heads = cl.read_receipts(ref_parts, plan, kid)
        hot, hot_heads = cl.read_receipts(hot_parts, plan, kid)
        volta, volta_heads = cl.read_receipts(volta_parts, plan, kid) if volta_parts else ({}, [])
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise Unusable(str(exc)) from exc
    kinds = {"reference": ref_heads, "hot_iq": hot_heads, "volta_split": volta_heads}
    for want, heads in kinds.items():
        if any(h["arm_kind"] != want for h in heads):
            raise Unusable(f"a receipt given as {want} was captured as another arm kind")
    reasons: list[str] = []
    ctx_meta = plan.contexts()
    gold = {m["id"]: m["gold"] for m in plan.doc["mc"]}
    order = plan.doc["order"]

    def main_of(caps, kind_ids):
        return {i: caps[("main", i)] for i in kind_ids if ("main", i) in caps}

    # bridges: a continuation part must reproduce the main capture of its first contexts exactly
    for name, caps in (("reference", ref), ("hot_iq", hot), ("volta_split", volta)):
        for i in order["bridge_contexts"]:
            if ("bridge", i) in caps and ("main", i) in caps and not cl.identical(caps[("bridge", i)], caps[("main", i)]):
                raise Unusable(f"{name}: bridge capture of {i} differs from its main capture (boots not equal)")
    ref_c, hot_c, vol_c = (main_of(x, order["main_contexts"]) for x in (ref, hot, volta))
    ref_m, hot_m, vol_m = (main_of(x, order["main_mc"]) for x in (ref, hot, volta))

    # floor
    fl_c = [i for i in order["floor_contexts"] if ("floor", i) in ref and i in ref_c]
    fl_m = [i for i in order["floor_mc"] if ("floor", i) in ref and i in ref_m]
    floor_rows = [cl.compare_context(i, ctx_meta[i]["kind"], ref_c[i], ref[("floor", i)]) for i in fl_c]
    identical_c = sum(1 for i in fl_c if cl.identical(ref_c[i], ref[("floor", i)]))
    identical_m = sum(1 for i in fl_m if cl.identical(ref_m[i], ref[("floor", i)]))
    complete = len(fl_c) == cl.FLOOR_CONTEXTS and len(fl_m) == cl.FLOOR_MC
    floor = {"contexts": len(fl_c), "mc": len(fl_m), "identical_contexts": identical_c, "identical_mc": identical_m,
             "status": (INCONCLUSIVE if not complete else
                        "ZERO" if identical_c == len(fl_c) and identical_m == len(fl_m) else "NONZERO"),
             "figures": context_figures(floor_rows, [r.kind for r in floor_rows]),
             "mc_answer_differs": (sum(1 for i in fl_m if ref_m[i].answer != ref[("floor", i)].answer) / len(fl_m))
             if fl_m else None}
    if not complete:
        reasons.append(f"floor incomplete: {len(fl_c)} of {cl.FLOOR_CONTEXTS} contexts, {len(fl_m)} of {cl.FLOOR_MC} MC")

    def arm_contexts(arm_c):
        ids = [i for i in order["main_contexts"] if i in ref_c and i in arm_c]
        rows = [cl.compare_context(i, ctx_meta[i]["kind"], ref_c[i], arm_c[i]) for i in ids]
        return ids, rows

    result = {"format": cl.SCORE_FORMAT, "plan_sha256": plan.sha256, "key_id": kid,
              "rules": {"top_k": cl.TOP_K, "continuation_tokens": cl.CONT_TOKENS, "mc_max_tokens": cl.MC_MAX_TOKENS,
                        "min_contexts": min_contexts, "min_mc": min_mc, "near_tie_nats": cl.NEAR_TIE_NATS,
                        "identical_logprob_tolerance": cl.IDENTICAL_LP_TOL, "bootstrap": cl.BOOT_B,
                        "bootstrap_seed": cl.BOOT_SEED, "survival_at": list(cl.SURVIVAL_AT)},
              "inputs": {k: [{"file": str(p), "sha256": cl.sha256_bytes(Path(p).read_bytes())} for p in parts]
                         for k, parts in (("reference", ref_parts), ("hot_iq", hot_parts),
                                          ("volta_split", volta_parts or []))},
              "part_end_status": {k: [end_status(p) for p in parts] for k, parts in
                                  (("reference", ref_parts), ("hot_iq", hot_parts), ("volta_split", volta_parts or []))},
              "floor": floor, "arms": {}}
    metrics = ("disagreement_rate", "mean_abs_dlogprob", "mean_kl")
    if floor["status"] == "ZERO":
        floor_cis = {m: [0.0, 0.0] for m in metrics}
    elif floor["status"] == "NONZERO":
        floor_cis = {m: floor["figures"][m]["ci95"] for m in metrics}
    else:
        floor_cis = {m: None for m in metrics}
    rows_by_arm = {}
    for name, arm_c, arm_m in (("hot_iq", hot_c, hot_m), ("volta_split", vol_c, vol_m)):
        if not arm_c and not arm_m:
            result["arms"][name] = {"status": "NOT CAPTURED"}
            continue
        ids, rows = arm_contexts(arm_c)
        rows_by_arm[name] = dict(zip(ids, rows))
        fig = context_figures(rows, [r.kind for r in rows])
        enough = len(rows) >= min_contexts
        vs = {m: (verdict_vs_floor(fig[m]["ci95"], floor_cis.get(m)) if enough else INCONCLUSIVE)
              for m in ("disagreement_rate", "mean_abs_dlogprob", "mean_kl")} if rows else {}
        if not enough:
            reasons.append(f"{name}: {len(rows)} contexts compared, fewer than {min_contexts}")
        mc_ids = [i for i in order["main_mc"] if i in ref_m and i in arm_m]
        mcf = mc_figures(mc_ids, gold, ref_m, arm_m)
        if len(mc_ids) < min_mc:
            reasons.append(f"{name}: {len(mc_ids)} MC items compared, fewer than {min_mc}")
            mcf["accuracy_verdict"] = INCONCLUSIVE
        result["arms"][name] = {"contexts": fig, "by_kind": by_kind(rows), "vs_reference": vs,
                                "contexts_enough": enough, "mc": mcf, "mc_enough": len(mc_ids) >= min_mc,
                                "items_captured": {"contexts": len(arm_c), "mc": len(arm_m)}}
    # hot_iq against Volta sPLIt on the positions both still share with the reference
    if "hot_iq" in rows_by_arm and "volta_split" in rows_by_arm:
        ids = [i for i in order["main_contexts"] if i in rows_by_arm["hot_iq"] and i in rows_by_arm["volta_split"]]
        h_rows, v_rows = [], []
        for i in ids:
            upto = min(rows_by_arm["hot_iq"][i].first_div, rows_by_arm["volta_split"][i].first_div)
            k = ctx_meta[i]["kind"]
            h_rows.append(cl.compare_context(i, k, ref_c[i], hot_c[i], upto=upto))
            v_rows.append(cl.compare_context(i, k, ref_c[i], vol_c[i], upto=upto))
            h_rows[-1].first_div = rows_by_arm["hot_iq"][i].first_div
            v_rows[-1].first_div = rows_by_arm["volta_split"][i].first_div
        strata = [r.kind for r in h_rows]
        common = {"contexts": len(ids), "hot_iq": context_figures(h_rows, strata),
                  "volta_split": context_figures(v_rows, strata), "paired": paired(h_rows, v_rows, strata),
                  "by_kind": {"hot_iq": by_kind(h_rows), "volta_split": by_kind(v_rows)}}
        if len(ids) < min_contexts:
            for m in common["paired"].values():
                if isinstance(m, dict) and "verdict" in m:
                    m["verdict"] = INCONCLUSIVE
        mc_ids = [i for i in order["main_mc"] if i in ref_m and i in hot_m and i in vol_m]
        import numpy as np

        dh = np.array([1.0 if hot_m[i].answer != ref_m[i].answer else 0.0 for i in mc_ids])
        dv = np.array([1.0 if vol_m[i].answer != ref_m[i].answer else 0.0 for i in mc_ids])
        if mc_ids:
            p, c, u = cl.bootstrap(lambda idx: float(dv[idx].mean() - dh[idx].mean()), len(mc_ids), None)
            common["mc"] = {"items": len(mc_ids), "differs_volta_minus_hot": ci_json(p, c, u),
                            "verdict": verdict_diff(c) if len(mc_ids) >= min_mc else INCONCLUSIVE,
                            "three_arm_table": {"reference": mc_figures(mc_ids, gold, ref_m, None),
                                                "hot_iq": mc_figures(mc_ids, gold, ref_m, hot_m),
                                                "volta_split": mc_figures(mc_ids, gold, ref_m, vol_m)}}
        result["common"] = common
    else:
        reasons.append("Volta sPLIt not captured: no hot_iq against Volta sPLIt comparison")
    result["reasons"] = reasons
    result["status"] = "SCORED" if not reasons else "SCORED WITH INCONCLUSIVE PARTS"
    return result


# ------------------------------------------------------------------------------------------- self-test

def _pos(tops: list[tuple[str, float]]) -> dict:
    return {"tok": tops[0][0], "lp": math.log(tops[0][1]), "top": [t for t, _ in tops],
            "toplp": [math.log(p) for _, p in tops]}


def _tail(prefix: str) -> list[tuple[str, float]]:
    return [(f"{prefix}{j}", 0.001) for j in range(cl.TOP_K - 3)]


REF_TOP = [("a", 0.5), ("b", 0.3), ("c", 0.1)]
HOT_NEAR = [("a", 0.45), ("b", 0.35), ("c", 0.1)]
VOL_NEAR = [("a", 0.4), ("b", 0.35), ("c", 0.15)]
SWAP = [("b", 0.5), ("a", 0.3), ("c", 0.1)]


def _capture(item: str, pass_: str, kind: str, sha: str, layout: list[list[tuple[str, float]]],
             answer=None) -> dict:
    ps = [_pos(t + _tail("z")) for t in layout]
    return {"type": "item", "item": item, "kind": kind, "pass": pass_, "prompt_sha256": sha, "prompt_tokens": 100,
            "tok": [p["tok"] for p in ps], "lp": [p["lp"] for p in ps], "top": [p["top"] for p in ps],
            "toplp": [p["toplp"] for p in ps], "seq_sha": "x", "answer": answer,
            "answer_pos": None if answer is None else 2}


def synthetic_inputs(tmp: Path):
    """The hand-checked synthetic fixture: 60 contexts (12 per kind), 60 MC items, three arms whose figures are
    worked out by hand in self_test. Also used by report.py's self-test. Returns (plan, key id, files)."""
    g = cl.CONT_TOKENS
    ctx = [{"id": f"{k}-{j:02d}", "kind": k, "text": f"t{k}{j}", "text_sha256": cl.sha256_text(f"t{k}{j}"),
            "user_tokens": 100} for k in cl.KINDS for j in range(12)]
    mc = [{"id": f"mc-{j:02d}", "subject": "s", "gold": 0, "prompt": f"p{j}", "prompt_sha256": cl.sha256_text(f"p{j}"),
           "user_tokens": 100} for j in range(60)]
    main_c = [c["id"] for c in ctx]
    main_m = [m["id"] for m in mc]
    plan_doc = {"format": cl.PLAN_FORMAT, "contexts": ctx, "mc": mc, "tokenizer_sha256": "x", "request": {},
                "order": {"main_contexts": main_c, "floor_contexts": main_c[:cl.FLOOR_CONTEXTS],
                          "bridge_contexts": main_c[:cl.BRIDGE_CONTEXTS], "main_mc": main_m,
                          "floor_mc": main_m[:cl.FLOOR_MC]}}
    (tmp / "plan.json").write_text(json.dumps(plan_doc))
    plan = cl.Plan.load(tmp / "plan.json")
    kid = "k" * 16

    def head(arm, kind):
        return {"type": "header", "format": cl.RECEIPT_FORMAT, "arm": arm, "arm_kind": kind, "part": 1,
                "plan_sha256": plan.sha256, "key_id": kid}

    ref_layout = [REF_TOP] * g
    hot_layout = [REF_TOP] + [HOT_NEAR] * 7 + [SWAP] + [REF_TOP] * (g - 9)  # diverges at position 8
    vol_layout = [REF_TOP] + [VOL_NEAR] * 3 + [SWAP] + [REF_TOP] * (g - 5)  # diverges at position 4
    mc_ref = {m["id"]: (0 if j < 45 else 1) for j, m in enumerate(mc)}  # 45 of 60 right
    mc_hot = dict(mc_ref) | {"mc-00": 2}  # one answer changed (right -> wrong)
    mc_vol = dict(mc_ref) | {f"mc-{j:02d}": 3 for j in range(6)}  # six changed
    files = {}
    for arm, kind, layout, ans, passes in (("R", "reference", ref_layout, mc_ref, ("floor", "main")),
                                           ("H", "hot_iq", hot_layout, mc_hot, ("main",)),
                                           ("V", "volta_split", vol_layout, mc_vol, ("main",))):
        lines = [head(arm, kind)]
        for pss in passes:
            cids = plan_doc["order"]["floor_contexts"] if pss == "floor" else main_c
            mids = plan_doc["order"]["floor_mc"] if pss == "floor" else main_m
            lines += [_capture(c, pss, "context", cl.sha256_text(plan.contexts()[c]["text"]), layout) for c in cids]
            lines += [_capture(m, pss, "mc", cl.sha256_text(plan.mc()[m]["prompt"]), [REF_TOP] * 3, ans[m])
                      for m in mids]
        lines.append({"type": "end", "status": "COMPLETE"})
        files[arm] = tmp / f"{arm}.part1.jsonl"
        files[arm].write_text("".join(json.dumps(x) + "\n" for x in lines))
    return plan, kid, files


def self_test() -> int:
    fails: list[str] = []

    def close(name, got, want, tol=1e-6):
        if got is None or abs(got - want) > tol:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    # kl_both_top on a hand case: reference top a .5, b .3, c .1 (rest .1); arm top a .4, c .3, e .2.
    # shared {a, c}: P = (.5, .1, rest .4), Q = (.4, .3, rest .3);
    # KL = .5 ln(.5/.4) + .1 ln(.1/.3) + .4 ln(.4/.3) = .111572 - .109861 + .115073 = .116783
    r = cl.Pos("a", math.log(.5), ("a", "b", "c"), (math.log(.5), math.log(.3), math.log(.1)))
    x = cl.Pos("a", math.log(.4), ("a", "c", "e"), (math.log(.4), math.log(.3), math.log(.2)))
    kl, shared = cl.kl_both_top(r, x)
    close("kl hand case", kl, 0.116783, 1e-5)
    close("kl shared mass", shared, 0.6)
    d, cens = cl.abs_dlp(r, x)
    close("abs dlp hand case", d, math.log(.5 / .4))
    check("abs dlp not censored", cens, False)
    r2 = cl.Pos("b", math.log(.3), r.top, r.toplp)
    d, cens = cl.abs_dlp(r2, x)  # b is not in the arm's top: bound ln .2, |ln .2 - ln .3| = .405465
    close("abs dlp censored", d, 0.405465, 1e-5)
    check("abs dlp censored flag", cens, True)

    # synthetic plan: 60 contexts (12 per kind), 60 MC items
    g = cl.CONT_TOKENS
    with tempfile.TemporaryDirectory() as tmp:
        plan, kid, files = synthetic_inputs(Path(tmp))
        res = score(plan, kid, [files["R"]], [files["H"]], [files["V"]], min_contexts=50, min_mc=50)
        # hand arithmetic, per context (all 60 alike):
        # hot: positions 1..8; 7 x [abs .105361, KL .5 ln(.5/.45) + .3 ln(.3/.35) = .006435] + 1 x [abs .510826,
        #      KL .5 ln(.5/.3) + .3 ln(.3/.5) = .2 x .510826 = .102165]; n 8, disagree 1
        #      agreement 7/8 = .875, mean abs (7 x .105361 + .510826) / 8 = .156044, mean KL (7 x .006435 + .102165) / 8 = .018401
        # volta: positions 1..4; 3 x [abs .223144, KL .5 ln(.5/.4) + .3 ln(.3/.35) + .1 ln(.1/.15) = .024780] + 1 x divergence
        #      agreement 3/4 = .75, mean abs (3 x .223144 + .510826) / 4 = .295064, mean KL (3 x .024780 + .102165) / 4 = .044126
        # common positions 1..4: hot 4 x [.105361, .006435], no disagreement; KL ratio .006435 / .044126 = .145832
        h = res["arms"]["hot_iq"]["contexts"]
        v = res["arms"]["volta_split"]["contexts"]
        close("hot agreement", h["top1_agreement"]["value"], 0.875)
        close("hot mean abs", h["mean_abs_dlogprob"]["value"], 0.156044, 1e-5)
        close("hot mean kl", h["mean_kl"]["value"], 0.018401, 1e-5)
        close("volta agreement", v["top1_agreement"]["value"], 0.75)
        close("volta mean abs", v["mean_abs_dlogprob"]["value"], 0.295064, 1e-5)
        close("volta mean kl", v["mean_kl"]["value"], 0.044126, 1e-5)
        close("hot pos0 kl", h["pos0_mean_kl"]["value"], 0.0)
        check("hot positions", h["positions"], 60 * 8)
        check("hot survival 4 / 8", (h["identical_through"]["4"]["value"], h["identical_through"]["8"]["value"]), (1.0, 0.0))
        check("volta survival 2 / 4", (v["identical_through"]["2"]["value"], v["identical_through"]["4"]["value"]), (1.0, 0.0))
        check("hot median", h["median_first_divergence"]["value"], 8.0)
        com = res["common"]
        close("common hot kl", com["hot_iq"]["mean_kl"]["value"], 0.006435, 1e-5)
        close("common volta kl", com["volta_split"]["mean_kl"]["value"], 0.044126, 1e-5)
        close("common kl ratio", com["paired"]["mean_kl"]["hot_over_volta"]["value"], 0.145832, 1e-4)
        check("common verdicts", [com["paired"][m]["verdict"] for m in ("disagreement_rate", "mean_abs_dlogprob", "mean_kl")],
              ["HOT_IQ CLOSER"] * 3)
        check("first divergence later", com["paired"]["first_divergence_hot_vs_volta"], {"hot_later": 60, "hot_earlier": 0, "same": 0})
        check("floor", (res["floor"]["status"], res["floor"]["identical_contexts"], res["floor"]["identical_mc"]), ("ZERO", 50, 50))
        check("hot vs floor", res["arms"]["hot_iq"]["vs_reference"]["mean_kl"], "DIFFERS FROM THE REFERENCE")
        # MC: reference 45 / 60 = .75; hot 44 / 60 = .733333, differs 1 / 60; volta 39 / 60 = .65, differs 6 / 60
        mh, mv = res["arms"]["hot_iq"]["mc"], res["arms"]["volta_split"]["mc"]
        close("mc reference", mh["reference_accuracy"]["value"], 0.75)
        close("mc hot", mh["accuracy"]["value"], 44 / 60)
        close("mc volta", mv["accuracy"]["value"], 39 / 60)
        close("mc hot differs", mh["answer_differs_from_reference"]["value"], 1 / 60)
        close("mc volta differs", mv["answer_differs_from_reference"]["value"], 0.1)
        close("mc paired", com["mc"]["differs_volta_minus_hot"]["value"], 5 / 60)
        check("mc discordant hot", mh["discordant"], {"arm_right_reference_wrong": 0, "arm_wrong_reference_right": 1})
        ci = mv["accuracy"]["ci95"]
        check("mc interval holds its point", ci is not None and ci[0] <= 39 / 60 <= ci[1], True)
        check("status", res["status"], "SCORED")
        res2 = score(plan, kid, [files["R"]], [files["H"]], [files["V"]])  # the real minimums: 300 and 500
        check("minimums give INCONCLUSIVE", (res2["common"]["paired"]["mean_kl"]["verdict"],
                                             res2["arms"]["hot_iq"]["mc"]["accuracy_verdict"],
                                             res2["status"]), (INCONCLUSIVE, INCONCLUSIVE, "SCORED WITH INCONCLUSIVE PARTS"))
        try:
            score(plan, "q" * 16, [files["R"]], [files["H"]], [files["V"]])
            fails.append("a receipt made with another key was accepted")
        except Unusable:
            pass
        try:
            score(plan, kid, [files["H"]], [files["H"]], [files["V"]])
            fails.append("an arm given as the reference was accepted")
        except Unusable:
            pass
    # bootstrap: two items 0 and 1, the mean's resamples are 0, .5 and 1 with chances 1/4, 1/2, 1/4
    import numpy as np

    vals = np.array([0.0, 1.0])
    p, ci, _ = cl.bootstrap(lambda idx: float(vals[idx].mean()), 2, None)
    check("bootstrap point and range", (p, ci), (0.5, (0.0, 1.0)))
    for f in fails:
        print("FAIL", f)
    print("SELF-TEST PASS" if not fails else f"SELF-TEST FAIL ({len(fails)})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--key-id", help="the key's id (16 hex) in place of --key, for scoring published receipts after "
                                     "the key was deleted")
    ap.add_argument("--ref", type=Path, nargs="+")
    ap.add_argument("--hot", type=Path, nargs="+")
    ap.add_argument("--volta", type=Path, nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.plan or bool(a.key) == bool(a.key_id):
        ap.error("--plan and exactly one of --key and --key-id are required")
    plan = cl.Plan.load(a.plan)
    if a.key_id and not re.fullmatch(r"[0-9a-f]{16}", a.key_id):
        ap.error("--key-id must be 16 lower-case hex digits")
    kid = a.key_id or cl.key_id(cl.read_key(a.key))
    if a.dry and not a.ref:
        o = plan.doc["order"]
        print(f"plan {plan.sha256[:16]}: {len(o['main_contexts'])} contexts, {len(o['main_mc'])} MC items; floor "
              f"{len(o['floor_contexts'])} + {len(o['floor_mc'])}; bridge {len(o['bridge_contexts'])}; key {kid}")
        print("no receipts given: the score needs --ref (floor and main passes) and --hot, and --volta for the "
              "hot_iq against Volta sPLIt comparison")
        print(f"rules: minimum {cl.MIN_CONTEXTS} contexts and {cl.MIN_MC} MC items per comparison; top {cl.TOP_K}; "
              f"{cl.CONT_TOKENS} tokens; bootstrap {cl.BOOT_B} seed {cl.BOOT_SEED}; CPU only, no requests")
        return 0
    if not (a.ref and a.hot):
        ap.error("--ref and --hot are required")
    if a.dry:
        for name, parts in (("reference", a.ref), ("hot_iq", a.hot), ("volta_split", a.volta or [])):
            if not parts:
                print(f"{name}: no receipt (its comparisons will be absent)")
                continue
            caps, heads = cl.read_receipts(parts, plan, kid)
            by = {}
            for (p, _), c in caps.items():
                by[p] = by.get(p, 0) + 1
            print(f"{name}: {len(parts)} part(s), arm {heads[0]['arm']}, kind {heads[0]['arm_kind']}, items per pass {by}")
        print(f"rules: minimum {cl.MIN_CONTEXTS} contexts and {cl.MIN_MC} MC items per comparison; floor "
              f"{cl.FLOOR_CONTEXTS} + {cl.FLOOR_MC}; top {cl.TOP_K}; {cl.CONT_TOKENS} tokens; bootstrap {cl.BOOT_B} "
              f"seed {cl.BOOT_SEED}")
        return 0
    if not a.out:
        ap.error("--out is required")
    try:
        res = score(plan, kid, a.ref, a.hot, a.volta or [])
    except Unusable as exc:
        print(json.dumps({"status": INCONCLUSIVE, "reason": str(exc)}))
        return 3
    with a.out.open("x") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": res["status"], "reasons": res["reasons"], "out": str(a.out),
                      "bytes": a.out.stat().st_size}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
