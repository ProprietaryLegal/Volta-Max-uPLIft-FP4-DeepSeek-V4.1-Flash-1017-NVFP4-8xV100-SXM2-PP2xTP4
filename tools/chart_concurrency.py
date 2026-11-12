#!/usr/bin/env python3
"""Concurrency chart (aggregate and per-request decode tok/s by prompt slots and requests running together) as two SVG
files, light and dark. No dependencies.

  python3 chart_concurrency.py OUTDIR [DATA.json]

Without DATA.json it draws window 15 (2026-11-12, run 2, the declared reading, docs/measurements.md). DATA.json gives
another window in the same shape: {"window": "18", "date": "2026-11-12", "default_slots": 2,
"table": {"1": {"1": [aggregate, per_request]}, "2": {...}, ...}}, slots and requests running together as keys.
Aggregate tok/s = all tokens over the time from the first start to the last end. Per-request = mean decode tok/s of
the requests running together. Colors: an ordinal one-hue ramp (slot count is ordered), checked with the dataviz
validator in both modes (--ordinal, ALL CHECKS PASS).
"""
import json
import math
import sys
from pathlib import Path

W15 = {"window": "15", "date": "2026-11-12", "default_slots": 2, "table": {
    1: {1: (118.7, 137.2)},
    2: {1: (124.4, 143.5), 2: (160.1, 109.0)},
    3: {1: (123.7, 138.6), 2: (165.2, 108.4), 3: (194.0, 88.2)},
    4: {1: (123.7, 137.9), 2: (166.9, 110.8), 3: (193.3, 86.9), 4: (198.3, 61.7)},
}}
MODES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "axis": "#b9b8b2",
              "ramp": {1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 4: "#0d366b"}},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "grid": "#33332f", "axis": "#5c5b55",
             "ramp": {1: "#1c5cab", 2: "#3987e5", 3: "#86b6ef", 4: "#cde2fb"}},
}
WORDS = {2: "two", 3: "three", 4: "four"}
W, H = 960, 448
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
BAR, GAP = 22, 2          # thin bars, 2 px surface gap between neighbours
TOP, BOTTOM = 112, 360     # plot area, y in px


def load(path):
    """Parse DATA.json once: slots 1..n present in order, each with levels 1..slots, two positive numbers per cell."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    t = {int(s): {int(k): tuple(float(x) for x in v) for k, v in lv.items()} for s, lv in d["table"].items()}
    if sorted(t) != list(range(1, len(t) + 1)) or len(t) > 4:
        raise SystemExit(f"{path}: slots must be 1 to n (n at most 4), got {sorted(t)}")
    for s, lv in t.items():
        if sorted(lv) != list(range(1, s + 1)) or any(len(v) != 2 or min(v) <= 0 for v in lv.values()):
            raise SystemExit(f"{path}: slot {s} needs levels 1 to {s}, each [aggregate, per_request] > 0")
    if int(d["default_slots"]) not in t:
        raise SystemExit(f"{path}: default_slots {d['default_slots']} is not in the table")
    return {"window": str(d["window"]), "date": str(d["date"]), "default_slots": int(d["default_slots"]), "table": t}


def names(data):
    return {s: f"{s} slot" + ("s" if s > 1 else "") + (" (published default)" if s == data["default_slots"] else "")
            for s in data["table"]}


def bar_path(x, y0, y1, w, r=4):
    """Bar from baseline y0 up to y1 with 4 px rounded data end, square at the baseline."""
    r = min(r, (y0 - y1) / 2, w / 2)
    return (f"M{x:.1f},{y0:.1f} L{x:.1f},{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} "
            f"L{x + w - r:.1f},{y1:.1f} Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} L{x + w:.1f},{y0:.1f} Z")


def panel(c, data, x0, x1, title, idx, step):
    t, nm = data["table"], names(data)
    n = len(t)
    vmax = step * math.ceil(max(v[idx] for lv in t.values() for v in lv.values()) * 1.04 / step)
    out = []
    scale = (BOTTOM - TOP) / vmax
    out.append(f'<text x="{x0}" y="{TOP - 26}" fill="{c["text"]}" font-size="15" font-weight="600">{title}</text>')
    for v in range(0, vmax + 1, step):
        y = BOTTOM - v * scale
        out.append(f'<line x1="{x0 + 34}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c["grid"] if v else c["axis"]}" '
                   f'stroke-width="1"/>')
        out.append(f'<text x="{x0 + 28}" y="{y + 4:.1f}" fill="{c["text2"]}" font-size="12" text-anchor="end">{v}</text>')
    group_w = (x1 - x0 - 40) / n
    for k in range(1, n + 1):
        gx = x0 + 40 + (k - 1) * group_w
        present = [s for s in range(1, n + 1) if k in t[s]]
        width = len(present) * BAR + (len(present) - 1) * GAP
        bx = gx + (group_w - width) / 2
        for s in present:
            v = t[s][k][idx]
            y1 = BOTTOM - v * scale
            out.append(f'<path d="{bar_path(bx, BOTTOM, y1, BAR)}" fill="{c["ramp"][s]}">'
                       f'<title>{nm[s]}, {k} together: {v:.1f} tok/s</title></path>')
            if s == max(present, key=lambda q: t[q][k][idx]):  # label the tallest bar of each group only
                out.append(f'<text x="{bx + BAR / 2:.1f}" y="{y1 - 6:.1f}" fill="{c["text"]}" font-size="12" '
                           f'text-anchor="middle">{v:.1f}</text>')
            bx += BAR + GAP
        out.append(f'<text x="{gx + group_w / 2:.1f}" y="{BOTTOM + 20}" fill="{c["text2"]}" font-size="12" '
                   f'text-anchor="middle">{k}</text>')
    out.append(f'<text x="{x0 + 40 + n / 2 * group_w:.1f}" y="{BOTTOM + 40}" fill="{c["text2"]}" font-size="12" '
               f'text-anchor="middle">requests running together</text>')
    return out


def desc(data):
    t = data["table"]
    agg = [f"1 slot {t[1][1][0]:.1f}"] + [f"{s} slots {t[s][s][0]:.1f} with {WORDS[s]}" + (" requests" if s == 2 else "")
                                          for s in sorted(t) if s > 1]
    per = [f"{t[s][s][1]:.1f}" for s in sorted(t) if s > 1]
    per_txt = per[0] if len(per) == 1 else ", ".join(per[:-1]) + " and " + per[-1]
    return (f"Window {data['window']}, run 2. Aggregate tokens per second: " + ", ".join(agg)
            + (f". Per-request decode falls from {t[1][1][1]:.1f} to {per_txt} tok/s." if per else "."))


def svg(mode, data):
    c, nm = MODES[mode], names(data)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'font-family="{FONT}" role="img" aria-labelledby="t d">',
             '<title id="t">Volta Max uPLIft: throughput by prompt slots and requests running together</title>',
             f'<desc id="d">{desc(data)}</desc>',
             f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>']
    lx = 24
    for s in sorted(data["table"]):
        parts.append(f'<rect x="{lx}" y="24" width="12" height="12" rx="2" fill="{c["ramp"][s]}"/>')
        parts.append(f'<text x="{lx + 18}" y="35" fill="{c["text"]}" font-size="13">{nm[s]}</text>')
        lx += 18 + int(6.7 * len(nm[s])) + 26
    parts += panel(c, data, 24, 470, "Aggregate tok/s, all requests together", 0, 25)
    parts += panel(c, data, 500, 940, "Decode tok/s of each request", 1, 25)
    for i, line in enumerate((f"Measured {data['date']} on 8x V100-SXM2-32GB (window {data['window']}, run 2): three public "
                              "prompts, 1,024 tokens each, thinking off, temperature 0. Each request's output token-exact",
                              "against the same prompt run alone. Labels: the highest bar of each group. "
                              "Slot profiles: quality not established.")):
        parts.append(f'<text x="24" y="{H - 30 + 16 * i}" fill="{c["text2"]}" font-size="11">{line}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    data = load(sys.argv[2]) if len(sys.argv) == 3 else W15
    out.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        p = out / f"concurrency-{mode}.svg"
        p.write_text(svg(mode, data))
        print(f"WROTE {p} {p.stat().st_size:,} B")


if __name__ == "__main__":
    main()
