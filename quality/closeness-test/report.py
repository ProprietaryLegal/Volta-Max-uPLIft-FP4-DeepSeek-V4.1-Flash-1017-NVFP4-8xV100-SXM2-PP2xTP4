#!/usr/bin/env python3
"""Tables, charts and filled prose from a closeness score file (score.py's output).

  report.py tables --score SCORE                         Markdown tables on standard output
  report.py charts --score SCORE --out DIR               survival.svg, kinds.svg, mc.svg (new files only)
  report.py fill   --score SCORE --in TEMPLATE --out FILE
                                                         replace every "CLOSENESS PENDING (slot)" marker
  report.py --dry [--score SCORE]                        the slots, the files each command writes, nothing written
  report.py --self-test                                  on score.py's hand-checked synthetic fixture

fill refuses (exit 3, nothing written) when the template names a slot this script does not know, and when a filled
value would break the public style (an em dash or a semicolon). Every slot is listed by --dry. A figure that the
score file does not hold (an arm not captured) fills as "not measured". Charts are static SVG (the README hosts
cannot run scripts): three fixed colours checked for colour-blind separation, a distinct marker shape per build,
direct labels, and the same numbers in the tables beside them.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closeness_lib as cl  # noqa: E402

MARKER = re.compile(r"CLOSENESS PENDING \(([a-z0-9_]+)\)")
NOT_MEASURED = "not measured"
LABEL = {"reference": "NVFP4 reference", "hot_iq": "hot_iq", "volta_split": "Volta sPLIt"}
# validated with the dataviz palette script (light surface, all pairs): CVD dE 9.2 worst, normal-vision dE 24.0;
# the aqua is below 3:1 on the surface, so every mark is also direct-labelled and every figure is in a table
COLOR = {"reference": "#2a78d6", "hot_iq": "#eb6834", "volta_split": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
FONT = "font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"


# ------------------------------------------------------------------------------------------- formatting

def pct(x, digits=1) -> str:
    return NOT_MEASURED if x is None else f"{100 * x:.{digits}f}%"


def nats(x) -> str:
    return NOT_MEASURED if x is None else f"{x:.4f}"


def num(x, digits=1) -> str:
    return NOT_MEASURED if x is None else f"{x:.{digits}f}"


def with_ci(fig: dict | None, fmt=pct) -> str:
    """'87.5% (95% interval 85.0% to 90.1%)'; an interval the bootstrap could not give is said plainly."""
    if not fig or fig.get("value") is None:
        return NOT_MEASURED
    ci = fig.get("ci95")
    if ci is None:
        return f"{fmt(fig['value'])} (no 95% interval: too many resamples undefined)"
    return f"{fmt(fig['value'])} (95% interval {fmt(ci[0])} to {fmt(ci[1])})"


def short_ci(fig: dict | None, fmt=pct) -> str:
    if not fig or fig.get("value") is None:
        return NOT_MEASURED
    ci = fig.get("ci95")
    return fmt(fig["value"]) + ("" if ci is None else f" ({fmt(ci[0])} to {fmt(ci[1])})")


def removed(fig: dict | None) -> dict | None:
    """hot_iq over Volta sPLIt (a ratio of distances) turned into the share of Volta sPLIt's distance removed."""
    if not fig or fig.get("value") is None:
        return None
    ci = fig.get("ci95")
    return {"value": 1 - fig["value"], "ci95": None if ci is None else [1 - ci[1], 1 - ci[0]]}


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ------------------------------------------------------------------------------------------- slots

def arm(s, a, *path):
    return get(s, "arms", a, *path)


def slots(s: dict) -> dict[str, str]:
    """Every value the publish section may name, as finished public text."""
    out: dict[str, str] = {}
    out["contexts_compared"] = str(get(s, "common", "contexts") or arm(s, "hot_iq", "contexts", "contexts")
                                   or NOT_MEASURED)
    out["mc_items"] = str(arm(s, "hot_iq", "mc", "items") or NOT_MEASURED)
    fl = s.get("floor", {})
    out["floor_result"] = {
        "ZERO": (f"identical on all {fl.get('contexts')} repeated contexts and all {fl.get('mc')} repeated questions, "
                 "so the noise floor is zero"),
        "NONZERO": (f"identical on {fl.get('identical_contexts')} of {fl.get('contexts')} repeated contexts and "
                    f"{fl.get('identical_mc')} of {fl.get('mc')} repeated questions, so the reference is not fully "
                    "repeatable and the arms are judged against its own repeat"),
    }.get(fl.get("status"), "INCONCLUSIVE (the repeat was not completed)")
    for a, tag in (("hot_iq", "hot"), ("volta_split", "volta")):
        c = arm(s, a, "contexts") or {}
        out[f"{tag}_top1"] = with_ci(c.get("top1_agreement"))
        out[f"{tag}_absdlp"] = with_ci(c.get("mean_abs_dlogprob"), nats)
        out[f"{tag}_kl"] = with_ci(c.get("mean_kl"), nats)
        out[f"{tag}_identical_all"] = with_ci(get(c, "identical_through", str(cl.CONT_TOKENS - 1)))
        med = c.get("median_first_divergence")
        out[f"{tag}_median_divergence"] = (NOT_MEASURED if not med or med.get("value") is None else
                                           f"beyond token {cl.CONT_TOKENS}" if c.get("median_is_censored") else
                                           with_ci({"value": med["value"] + 1, "ci95": None if med["ci95"] is None else
                                                    [med["ci95"][0] + 1, med["ci95"][1] + 1]}, num))
        out[f"{tag}_pos0"] = with_ci(c.get("pos0_agreement"))
        out[f"{tag}_vs_reference"] = arm(s, a, "vs_reference", "mean_kl") or NOT_MEASURED
        m = arm(s, a, "mc") or {}
        out[f"{tag}_mc_accuracy"] = with_ci(m.get("accuracy"))
        out[f"{tag}_mc_minus_reference"] = with_ci(m.get("accuracy_minus_reference"))
        out[f"{tag}_mc_verdict"] = m.get("accuracy_verdict") or NOT_MEASURED
        out[f"{tag}_mc_differs"] = with_ci(m.get("answer_differs_from_reference"))
    out["reference_mc_accuracy"] = with_ci(arm(s, "hot_iq", "mc", "reference_accuracy"))
    com = s.get("common") or {}
    for m in ("disagreement_rate", "mean_abs_dlogprob", "mean_kl"):
        out[f"common_verdict_{m}"] = get(com, "paired", m, "verdict") or NOT_MEASURED
    out["common_hot_kl"] = with_ci(get(com, "hot_iq", "mean_kl"), nats)
    out["common_volta_kl"] = with_ci(get(com, "volta_split", "mean_kl"), nats)
    out["common_hot_top1"] = with_ci(get(com, "hot_iq", "top1_agreement"))
    out["common_volta_top1"] = with_ci(get(com, "volta_split", "top1_agreement"))
    out["share_removed_kl"] = with_ci(removed(get(com, "paired", "mean_kl", "hot_over_volta")))
    out["share_removed_absdlp"] = with_ci(removed(get(com, "paired", "mean_abs_dlogprob", "hot_over_volta")))
    out["common_mc_verdict"] = get(com, "mc", "verdict") or NOT_MEASURED
    fd = get(com, "paired", "first_divergence_hot_vs_volta")
    out["hot_later_divergence"] = (NOT_MEASURED if not fd else
                                   f"hot_iq stayed identical longer on {fd['hot_later']} contexts, Volta sPLIt on "
                                   f"{fd['hot_earlier']}, and the two diverged at the same token on {fd['same']}")
    out["status"] = s.get("status", NOT_MEASURED)
    out["table_contexts"] = table_contexts(s)
    out["table_common"] = table_common(s)
    out["table_kinds"] = table_kinds(s)
    out["table_mc"] = table_mc(s)
    return out



# ------------------------------------------------------------------------------------------- tables

def md_table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def table_contexts(s: dict) -> str:
    h, v = arm(s, "hot_iq", "contexts") or {}, arm(s, "volta_split", "contexts") or {}
    f = get(s, "floor", "figures") or {}
    last = str(cl.CONT_TOKENS - 1)
    rows = [
        ["Contexts", str(h.get("contexts", NOT_MEASURED)), str(v.get("contexts", NOT_MEASURED)),
         str(f.get("contexts", NOT_MEASURED))],
        ["Tokens compared (after the first, up to the first difference)", str(h.get("positions", NOT_MEASURED)),
         str(v.get("positions", NOT_MEASURED)), str(f.get("positions", NOT_MEASURED))],
        ["Same next token as the reference"] + [short_ci(x.get("top1_agreement")) for x in (h, v, f)],
        ["Mean log-probability gap on the reference's token (nats)"]
        + [short_ci(x.get("mean_abs_dlogprob"), nats) for x in (h, v, f)],
        ["Mean KL divergence from the reference, top 20 (nats, lower bound)"]
        + [short_ci(x.get("mean_kl"), nats) for x in (h, v, f)],
        [f"All {cl.CONT_TOKENS} tokens identical"] + [short_ci(get(x, "identical_through", last)) for x in (h, v, f)],
        ["First token identical (prompt reading, control)"] + [short_ci(x.get("pos0_agreement")) for x in (h, v, f)],
    ]
    return md_table(["Measure", LABEL["hot_iq"], LABEL["volta_split"], "Reference repeat (noise floor)"], rows)


def table_common(s: dict) -> str:
    com = s.get("common") or {}
    rows = []
    for m, name, fmt in (("disagreement_rate", "Next token differs from the reference", pct),
                         ("mean_abs_dlogprob", "Mean log-probability gap (nats)", nats),
                         ("mean_kl", "Mean KL divergence, top 20 (nats, lower bound)", nats)):
        rows.append([name, short_ci(get(com, "hot_iq", m), fmt), short_ci(get(com, "volta_split", m), fmt),
                     short_ci(get(com, "paired", m, "volta_minus_hot"), fmt),
                     get(com, "paired", m, "verdict") or NOT_MEASURED])
    return md_table(["Measure (same tokens for both builds)", LABEL["hot_iq"], LABEL["volta_split"],
                     "Volta sPLIt minus hot_iq", "Verdict"], rows)


def table_kinds(s: dict) -> str:
    bk = get(s, "common", "by_kind") or {}
    rows = []
    for k in cl.KINDS:
        h, v = get(bk, "hot_iq", k) or {}, get(bk, "volta_split", k) or {}
        rows.append([k, str(h.get("contexts", NOT_MEASURED)), str(h.get("positions", NOT_MEASURED)),
                     short_ci(h.get("mean_kl"), nats), short_ci(v.get("mean_kl"), nats),
                     short_ci(h.get("top1_agreement")), short_ci(v.get("top1_agreement"))])
    return md_table(["Text kind", "Contexts", "Tokens compared", "hot_iq KL", "Volta sPLIt KL", "hot_iq same token",
                     "Volta sPLIt same token"], rows)


def table_mc(s: dict) -> str:
    three = get(s, "common", "mc", "three_arm_table")
    rows = []
    ref = (three or {}).get("reference") or arm(s, "hot_iq", "mc") or {}
    rows.append([LABEL["reference"], short_ci(ref.get("reference_accuracy")), "", "",
                 str(ref.get("reference_no_answer", NOT_MEASURED))])
    for a in ("hot_iq", "volta_split"):
        m = (three or {}).get(a) or arm(s, a, "mc") or {}
        rows.append([LABEL[a], short_ci(m.get("accuracy")), short_ci(m.get("accuracy_minus_reference")),
                     short_ci(m.get("answer_differs_from_reference")), str(m.get("no_answer", NOT_MEASURED))])
    return md_table(["Build", "Accuracy", "Minus the reference", "Answer differs from the reference",
                     "No answer"], rows)


SLOT_NAMES = sorted(slots({}).keys())


# ------------------------------------------------------------------------------------------- charts (static SVG)

def nice_max(v: float) -> float:
    """The smallest of 1, 2, 2.5, 5 times a power of ten that is at least v."""
    if v <= 0:
        return 1.0
    e = math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if m * 10 ** e >= v - 1e-12:
            return m * 10 ** e
    raise AssertionError("unreachable")


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_doc(w: int, h: int, title: str, desc: str, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-labelledby="t d"><title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>' + "".join(body) + "</svg>\n")


def text(x, y, t, size=12, color=INK2, anchor="start", weight="normal") -> str:
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}" style="{FONT}">{esc(t)}</text>')


def marker(shape: str, x: float, y: float, color: str, tip: str) -> str:
    ring = f'stroke="{SURFACE}" stroke-width="2"'
    if shape == "circle":
        m = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" {ring}>'
    elif shape == "square":
        m = f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" rx="1" fill="{color}" {ring}>'
    else:
        m = (f'<path d="M{x:.2f} {y - 6:.2f} L{x + 6:.2f} {y:.2f} L{x:.2f} {y + 6:.2f} L{x - 6:.2f} {y:.2f} Z" '
             f'fill="{color}" {ring}>')
    return m + f"<title>{esc(tip)}</title></{m[1:m.index(' ')]}>"


SHAPE = {"reference": "circle", "hot_iq": "square", "volta_split": "diamond"}


def chart_survival(s: dict) -> str:
    """Share of contexts whose first N greedy tokens all equal the reference's, N = 1, 2, 3, 5, 9, 17, 32."""
    w, h, left, top, pw, ph = 640, 360, 64, 56, 520, 220
    pts = list(cl.SURVIVAL_AT)
    xs = {p: left + i * pw / (len(pts) - 1) for i, p in enumerate(pts)}
    y = lambda v: top + (1 - v) * ph  # noqa: E731
    body = [text(left, 24, "Greedy continuations that stay identical to the NVFP4 reference", 15, INK, weight="600"),
            text(left, 42, "Share of contexts whose first N tokens all match, with 95% intervals", 12)]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="{GRID}" '
                    'stroke-width="1"/>')
        body.append(text(left - 8, y(v) + 4, pct(v, 0), 11, anchor="end"))
    for p in pts:
        body.append(text(xs[p], top + ph + 18, str(p + 1), 11, anchor="middle"))
    body.append(text(left + pw / 2, top + ph + 36, "First N tokens identical (points measured at N = 1, 2, 3, 5, 9, 17, 32)", 12, anchor="middle"))
    floor = get(s, "floor", "figures", "identical_through")
    series = [("reference", floor, "Reference repeat")] if floor else []
    series += [(a, arm(s, a, "contexts", "identical_through"), LABEL[a]) for a in ("hot_iq", "volta_split")]
    ends = []
    for a, surv, name in series:
        if not surv:
            continue
        col = COLOR[a]
        line = [(xs[p], y(surv[str(p)]["value"])) for p in pts if surv.get(str(p), {}).get("value") is not None]
        body.append(f'<polyline points="{" ".join(f"{px:.2f},{py:.2f}" for px, py in line)}" fill="none" '
                    f'stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for p in pts:
            f = surv.get(str(p)) or {}
            if f.get("value") is None:
                continue
            if f.get("ci95"):
                body.append(f'<line x1="{xs[p]:.2f}" x2="{xs[p]:.2f}" y1="{y(f["ci95"][0]):.2f}" '
                            f'y2="{y(f["ci95"][1]):.2f}" stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
            body.append(marker(SHAPE[a], xs[p], y(f["value"]), col,
                               f"{name}: first {p + 1} tokens identical on {pct(f['value'])} of contexts"))
        ends.append([line[-1][1] + 4, line[-1][0] + 10, name])
    # end labels: keep at least 15 px apart so overlapping lines still get readable names
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
    body += [text(lx, ly, name, 12, INK) for ly, lx, name in ends]
    lx = left
    for a, surv, name in series:
        if surv:
            body.append(marker(SHAPE[a], lx + 6, h - 20, COLOR[a], name))
            body.append(text(lx + 18, h - 16, name, 12, INK))
            lx += 160
    return svg_doc(w + 90, h, "Greedy continuations identical to the NVFP4 reference",
                   "Share of contexts whose first N greedy tokens equal the reference's, per build, with bootstrap "
                   "95% intervals. The same figures are in the table beside this chart.", body)


def chart_kinds(s: dict) -> str:
    """Mean KL divergence from the reference by text kind, on the tokens both builds share with the reference."""
    bk = get(s, "common", "by_kind") or {}
    w, h, left, top, pw, ph = 640, 340, 72, 56, 540, 220
    arms = ("hot_iq", "volta_split")
    vals = [x for a in arms for k in cl.KINDS for x in
            ([get(bk, a, k, "mean_kl", "value")] + list(get(bk, a, k, "mean_kl", "ci95") or [])) if x is not None]
    top_v = nice_max(max(vals) if vals else 1.0)
    y = lambda v: top + ph - v / top_v * ph  # noqa: E731
    body = [text(left, 24, "Distance from the NVFP4 reference by text kind (lower is closer)", 15, INK, weight="600"),
            text(left, 42, "Mean KL divergence in nats over the top 20 tokens, same tokens for both builds, "
                           "95% intervals", 12)]
    for i in range(5):
        v = top_v * i / 4
        body.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="{GRID}" '
                    'stroke-width="1"/>')
        body.append(text(left - 8, y(v) + 4, f"{v:.3g}", 11, anchor="end"))
    gw = pw / len(cl.KINDS)
    bw = 22
    for gi, k in enumerate(cl.KINDS):
        cx = left + gw * (gi + 0.5)
        body.append(text(cx, top + ph + 18, k, 12, anchor="middle"))
        for ai, a in enumerate(arms):
            f = get(bk, a, k, "mean_kl") or {}
            x0 = cx - bw - 1 + ai * (bw + 2)  # a 2px surface gap between the pair
            if f.get("value") is None:
                body.append(text(x0 + bw / 2, top + ph - 6, "none", 10, anchor="middle"))
                continue
            hgt = f["value"] / top_v * ph
            r = min(4.0, bw / 2, hgt / 2)
            yb, yt = top + ph, top + ph - hgt
            # rounded data end at the top, square at the baseline
            body.append(f'<path d="M{x0:.2f} {yb:.2f} L{x0:.2f} {yt + r:.2f} Q{x0:.2f} {yt:.2f} {x0 + r:.2f} '
                        f'{yt:.2f} L{x0 + bw - r:.2f} {yt:.2f} Q{x0 + bw:.2f} {yt:.2f} {x0 + bw:.2f} {yt + r:.2f} '
                        f'L{x0 + bw:.2f} {yb:.2f} Z" fill="{COLOR[a]}" data-height="{hgt:.4f}">'
                        f"<title>{esc(LABEL[a])}, {k}: {nats(f['value'])} nats</title></path>")
            if f.get("ci95"):
                xm = x0 + bw / 2
                body.append(f'<line x1="{xm:.2f}" x2="{xm:.2f}" y1="{y(f["ci95"][0]):.2f}" y2="{y(f["ci95"][1]):.2f}" '
                            f'stroke="{INK}" stroke-width="1.5" stroke-linecap="round"/>')
    lx = left
    for a in arms:
        body.append(f'<rect x="{lx}" y="{h - 26}" width="12" height="12" rx="2" fill="{COLOR[a]}"/>')
        body.append(text(lx + 18, h - 16, LABEL[a], 12, INK))
        lx += 140
    return svg_doc(w + 20, h, "Distance from the NVFP4 reference by text kind",
                   "Mean KL divergence from the reference per text kind for hot_iq and Volta sPLIt, on the tokens "
                   "where both still match the reference, with bootstrap 95% intervals. The same figures are in the "
                   "table beside this chart.", body)


def chart_mc(s: dict) -> str:
    """Multiple-choice accuracy of the three builds, dots with 95% intervals on a shared axis."""
    three = get(s, "common", "mc", "three_arm_table") or {}
    rows = [("reference", (three.get("reference") or arm(s, "hot_iq", "mc") or {}).get("reference_accuracy"))]
    rows += [(a, (three.get(a) or arm(s, a, "mc") or {}).get("accuracy")) for a in ("hot_iq", "volta_split")]
    rows = [(a, f) for a, f in rows if f and f.get("value") is not None]
    w, left, top, pw, rh = 640, 150, 56, 420, 40
    h = top + rh * max(len(rows), 1) + 50
    lo_v = min([f["ci95"][0] if f.get("ci95") else f["value"] for _, f in rows] or [0.0])
    hi_v = max([f["ci95"][1] if f.get("ci95") else f["value"] for _, f in rows] or [1.0])
    lo = max(0.0, (math.floor(round(lo_v * 20, 9)) - 1) / 20)
    hi = min(1.0, (math.ceil(round(hi_v * 20, 9)) + 1) / 20)
    x = lambda v: left + (v - lo) / (hi - lo) * pw  # noqa: E731
    body = [text(16, 24, "Multiple-choice accuracy, thinking off, temperature 0", 15, INK, weight="600"),
            text(16, 42, "Dots are accuracy, bars are 95% intervals. The axis does not start at zero.", 12)]
    steps = round((hi - lo) / 0.05)
    for i in range(steps + 1):
        v = lo + 0.05 * i
        body.append(f'<line x1="{x(v):.2f}" x2="{x(v):.2f}" y1="{top - 6}" y2="{top + rh * len(rows)}" '
                    f'stroke="{GRID}" stroke-width="1"/>')
        body.append(text(x(v), top + rh * len(rows) + 18, pct(v, 0), 11, anchor="middle"))
    for i, (a, f) in enumerate(rows):
        cy = top + rh * i + rh / 2
        body.append(text(left - 12, cy + 4, LABEL[a], 12, INK, anchor="end"))
        if f.get("ci95"):
            body.append(f'<line x1="{x(f["ci95"][0]):.2f}" x2="{x(f["ci95"][1]):.2f}" y1="{cy:.2f}" y2="{cy:.2f}" '
                        f'stroke="{COLOR[a]}" stroke-width="2" stroke-linecap="round"/>')
        body.append(marker(SHAPE[a], x(f["value"]), cy, COLOR[a], f"{LABEL[a]}: {pct(f['value'])}"))
        body.append(text(x(f["ci95"][1] if f.get("ci95") else f["value"]) + 10, cy + 4, pct(f["value"]), 12, INK))
    return svg_doc(w, h, "Multiple-choice accuracy by build",
                   "Accuracy of the NVFP4 reference, hot_iq and Volta sPLIt on the same multiple-choice items with "
                   "bootstrap 95% intervals. The same figures are in the table beside this chart.", body)


CHARTS = {"survival.svg": chart_survival, "kinds.svg": chart_kinds, "mc.svg": chart_mc}


# ------------------------------------------------------------------------------------------- fill

class Refused(Exception):
    pass


def fill(template: str, s: dict) -> str:
    names = MARKER.findall(template)
    unknown = sorted(set(names) - set(SLOT_NAMES))
    if unknown:
        raise Refused(f"the template names slots this script does not know: {', '.join(unknown)}")
    vals = slots(s)
    for n in set(names):
        if "—" in vals[n] or ";" in vals[n]:
            raise Refused(f"slot {n} would put an em dash or a semicolon into public prose")
    out = MARKER.sub(lambda m: vals[m.group(1)], template)
    if "CLOSENESS PENDING" in out:
        raise Refused("a CLOSENESS PENDING marker is left that is not in the form CLOSENESS PENDING (slot)")
    return out


# ------------------------------------------------------------------------------------------- self-test

def self_test() -> int:
    import score as sc

    fails: list[str] = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    check("pct", pct(0.875), "87.5%")
    check("nats", nats(0.0064351), "0.0064")
    check("with_ci", with_ci({"value": 0.5, "ci95": [0.25, 0.75]}), "50.0% (95% interval 25.0% to 75.0%)")
    check("with_ci none", with_ci({"value": 0.5, "ci95": None}),
          "50.0% (no 95% interval: too many resamples undefined)")
    check("removed", removed({"value": 0.2, "ci95": [0.1, 0.4]}), {"value": 0.8, "ci95": [0.6, 0.9]})
    check("nice_max", [nice_max(v) for v in (0.044126, 0.3, 1.0, 7.1)], [0.05, 0.5, 1.0, 10.0])
    check("unknown slot gives not measured", slots({})["hot_kl"], NOT_MEASURED)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plan, kid, files = sc.synthetic_inputs(tmp)
        s = sc.score(plan, kid, [files["R"]], [files["H"]], [files["V"]], min_contexts=50, min_mc=50)
        v = slots(s)
        # the score self-test's hand arithmetic: hot agreement 7/8 on every context, Volta 3/4; on the common
        # positions hot's KL is .006435 and Volta's .044126, so 1 - .006435/.044126 = 85.4% of the distance removed
        check("hot top1", v["hot_top1"], "87.5% (95% interval 87.5% to 87.5%)")
        check("volta top1", v["volta_top1"], "75.0% (95% interval 75.0% to 75.0%)")
        check("removed kl", v["share_removed_kl"], "85.4% (95% interval 85.4% to 85.4%)")
        check("floor", v["floor_result"], "identical on all 50 repeated contexts and all 50 repeated questions, "
                                          "so the noise floor is zero")
        # divergence at 0-based position 8 is the 9th token
        check("hot median", v["hot_median_divergence"], "9.0 (95% interval 9.0 to 9.0)")
        check("mc", (v["reference_mc_accuracy"].split(" ")[0], v["hot_mc_accuracy"].split(" ")[0],
                     v["volta_mc_accuracy"].split(" ")[0], v["volta_mc_differs"].split(" ")[0]),
              ("75.0%", "73.3%", "65.0%", "10.0%"))
        check("later", v["hot_later_divergence"], "hot_iq stayed identical longer on 60 contexts, Volta sPLIt on 0, "
                                                  "and the two diverged at the same token on 0")
        check("table rows", table_kinds(s).count("\n"), 1 + len(cl.KINDS))
        check("all slots known", sorted(v), SLOT_NAMES)
        for n, val in v.items():
            if "—" in val or ";" in val:
                fails.append(f"slot {n} holds an em dash or a semicolon")
        good = "Agreement was CLOSENESS PENDING (hot_top1) on CLOSENESS PENDING (contexts_compared) contexts."
        check("fill", fill(good, s), "Agreement was 87.5% (95% interval 87.5% to 87.5%) on 60 contexts.")
        for bad, why in (("CLOSENESS PENDING (no_such_slot)", "unknown slot"),
                         ("CLOSENESS PENDING (Hot)", "malformed marker")):
            try:
                fill(bad, s)
                fails.append(f"fill accepted a {why}")
            except Refused:
                pass
        # charts: well-formed XML and marks where the hand arithmetic puts them
        for name, fn in CHARTS.items():
            try:
                ET.fromstring(fn(s))
            except ET.ParseError as exc:
                fails.append(f"{name} is not well-formed: {exc}")
        surv = chart_survival(s)
        # hot_iq is identical through 0-based position 4 (top of the plot, y = 56) and not through 8 (y = 276)
        check("survival top", 'y="51.00" width="10"' in surv, True)
        check("survival bottom", 'y="271.00" width="10"' in surv, True)
        kinds = chart_kinds(s)
        # axis top = nice_max(.0441263) = .05 over 220 px: hot bar .00643505 / .05 x 220 = 28.3142, Volta .04412633 x 4400 = 194.1558
        check("kinds bar height", 'data-height="28.3142"' in kinds, True)
        check("kinds volta bar height", 'data-height="194.1558"' in kinds, True)
    # MC chart on hand-set intervals: lowest end 60%, highest 80%, so the axis runs 55% .. 85% (one 5-point step of
    # room each side) over 420 px from x = 150; Volta sPLIt's 65% sits at 150 + (.10 / .30) x 420 = 290
    hand = {"arms": {"hot_iq": {"mc": {"reference_accuracy": {"value": 0.75, "ci95": [0.70, 0.80]},
                                       "accuracy": {"value": 0.7333, "ci95": [0.68, 0.78]}}},
                     "volta_split": {"mc": {"accuracy": {"value": 0.65, "ci95": [0.60, 0.70]}}}}}
    mc = chart_mc(hand)
    check("mc diamond at 65%", 'd="M290.00 ' in mc, True)
    check("mc axis ends", ('>55%<' in mc, '>85%<' in mc, '>50%<' in mc, '>90%<' in mc), (True, True, False, False))
    for f in fails:
        print("FAIL", f)
    print("SELF-TEST PASS" if not fails else f"SELF-TEST FAIL ({len(fails)})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", nargs="?", choices=("tables", "charts", "fill"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--score", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--in", dest="inp", type=Path)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.dry:
        print("slots (each fills CLOSENESS PENDING (name)):")
        for n in SLOT_NAMES:
            print("  " + n)
        print("tables: prints four Markdown tables (contexts, common positions, text kinds, multiple choice)")
        print(f"charts: writes {', '.join(CHARTS)} into --out (refuses an existing file)")
        print("fill: writes --out from --in (refuses an existing file, an unknown slot, or an em dash or semicolon)")
        if a.score:
            s = json.loads(a.score.read_text())
            print(f"score {a.score}: status {s.get('status')}, reasons {s.get('reasons')}")
        print("CPU only, no requests")
        return 0
    if not a.command or not a.score:
        ap.error("a command and --score are required")
    s = json.loads(a.score.read_text())
    if s.get("format") != cl.SCORE_FORMAT:
        print(f"INCONCLUSIVE: {a.score} is not a {cl.SCORE_FORMAT} file")
        return 3
    if a.command == "tables":
        for name in ("table_contexts", "table_common", "table_kinds", "table_mc"):
            print(slots(s)[name] + "\n")
        return 0
    if not a.out:
        ap.error("--out is required")
    if a.command == "charts":
        a.out.mkdir(exist_ok=True)
        for name, fn in CHARTS.items():
            with (a.out / name).open("x") as f:
                f.write(fn(s))
            print(f"{a.out / name} {(a.out / name).stat().st_size} bytes")
        return 0
    if not a.inp:
        ap.error("fill needs --in")
    try:
        filled = fill(a.inp.read_text(), s)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 3
    with a.out.open("x") as f:
        f.write(filled)
    print(f"{a.out} {a.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
