#!/usr/bin/env python3
"""Decode chart (one-stream decode tok/s by prompt length, 1K to 256K, close-out bench, code, legal and prose) as two
SVG files, light and dark. No dependencies, in the style of chart_first_token.py.

  python3 chart_decode_by_length.py OUTDIR

Writes decode-by-length-closeout-light.svg and decode-by-length-closeout-dark.svg.
"""
import sys
from pathlib import Path

# Copied from docs/measurements.md, section "Speed bench 2026-11-13 (close-out)" > "Long prompts: 1K to 256K, thinking
# off", the second figure of each cell (close-out build, MEASURED 2026-11-13, one request at a time on the two-slot
# default, thinking off, mean of 2 runs).
LENGTHS = ["1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K"]
SERIES = [
    ("code", [172.0, 168.5, 174.6, 169.5, 159.5, 162.6, 164.3, 147.9, 142.7]),
    ("legal", [98.8, 95.1, 90.9, 89.4, 84.8, 89.0, 90.0, 79.1, 83.5]),
    ("prose", [83.2, 81.9, 80.8, 81.6, 80.6, 82.3, 79.4, 76.9, 74.4]),
]
MODES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "axis": "#b9b8b2",
              "lines": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "grid": "#33332f", "axis": "#5c5b55",
             "lines": ["#3987e5", "#d95926", "#199e70"]},
}
W, H = 960, 440
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
LEFT, RIGHT, TOP, BOTTOM = 84, 880, 92, 332   # plot area in px
YMAX, STEP = 200, 50                          # tok/s


def svg(mode):
    c = MODES[mode]
    n = len(LENGTHS)
    xs = [LEFT + 20 + i * (RIGHT - LEFT - 40) / (n - 1) for i in range(n)]
    scale = (BOTTOM - TOP) / YMAX
    desc = ("Close-out build, one request at a time on the two-slot default, thinking off, decode tok/s by prompt "
            "length. " + " ".join(f"{name}: " + ", ".join(f"{k} {v}" for k, v in zip(LENGTHS, vals)) + "."
                                  for name, vals in SERIES))
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="{FONT}" role="img" aria-labelledby="t d">',
         '<title id="t">Volta Max uPLIft: decode speed by prompt length</title>',
         f'<desc id="d">{desc}</desc>',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>',
         f'<text x="24" y="36" fill="{c["text"]}" font-size="15" font-weight="600">'
         'Decode speed by prompt length, one request, tok/s</text>']
    lx = 24
    for (name, _), col in zip(SERIES, c["lines"]):
        p.append(f'<line x1="{lx}" x2="{lx + 22}" y1="60" y2="60" stroke="{col}" stroke-width="2.5"/>')
        p.append(f'<text x="{lx + 28}" y="64" fill="{c["text2"]}" font-size="12">{name}</text>')
        lx += 90
    for v in range(0, YMAX + 1, STEP):
        y = BOTTOM - v * scale
        p.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c["grid"] if v else c["axis"]}" '
                 'stroke-width="1"/>')
        p.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" fill="{c["text2"]}" font-size="12" text-anchor="end">{v}</text>')
    # value labels above the line for code and legal, below it for prose; end names nudged apart (legal and prose
    # end 9 tok/s apart)
    place = {"code": (-10, 4), "legal": (-10, -2), "prose": (20, 12)}
    for (name, vals), col in zip(SERIES, c["lines"]):
        ys = [BOTTOM - v * scale for v in vals]
        dv, dn = place[name]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for k, v, x, y in zip(LENGTHS, vals, xs, ys):
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}" stroke="{c["surface"]}" stroke-width="2">'
                     f'<title>{name}, {k} prompt: {v} tok/s</title></circle>')
        p.append(f'<text x="{xs[0]:.1f}" y="{ys[0] + dv:.1f}" fill="{c["text"]}" font-size="12" '
                 f'text-anchor="middle">{vals[0]}</text>')
        p.append(f'<text x="{xs[-1]:.1f}" y="{ys[-1] + dv:.1f}" fill="{c["text"]}" font-size="12" '
                 f'text-anchor="middle">{vals[-1]}</text>')
        p.append(f'<text x="{xs[-1] + 14:.1f}" y="{ys[-1] + dn:.1f}" fill="{c["text2"]}" font-size="12">{name}</text>')
    for k, x in zip(LENGTHS, xs):
        p.append(f'<text x="{x:.1f}" y="{BOTTOM + 20}" fill="{c["text2"]}" font-size="12" text-anchor="middle">{k}</text>')
    p.append(f'<text x="{(LEFT + RIGHT) / 2:.1f}" y="{BOTTOM + 40}" fill="{c["text2"]}" font-size="12" '
             'text-anchor="middle">prompt length, tokens (each step doubles)</text>')
    for i, line in enumerate(("Measured 2026-11-13 on 8x V100-SXM2-32GB, close-out build, one request at a time on the "
                              "two-slot default, thinking off, mean of 2 runs.",
                              "Bench conditions as in the README's Results. Fast prompt reading: quality not "
                              "established.")):
        p.append(f'<text x="24" y="{H - 30 + 16 * i}" fill="{c["text2"]}" font-size="11">{line}</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        f = out / f"decode-by-length-closeout-{mode}.svg"
        f.write_text(svg(mode))
        print(f"WROTE {f} {f.stat().st_size:,} B")


if __name__ == "__main__":
    main()
