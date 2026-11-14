#!/usr/bin/env python3
"""First-token chart (time to first token by prompt length, 1K to 256K, close-out bench) as two SVG files, light and
dark. No dependencies, in the style of chart_concurrency.py.

  python3 chart_first_token.py OUTDIR

Writes first-token-by-length-closeout-light.svg and first-token-by-length-closeout-dark.svg.
"""
import sys
from pathlib import Path

# Copied from README.md, section "Speed: tokens per second" > "Long prompts: 1K to 256K", column "first token"
# (close-out build, MEASURED 2026-11-13, one request at a time on the two-slot default, thinking off).
FIRST_TOKEN_S = [
    ("1K", 0.85),
    ("2K", 1.41),
    ("4K", 2.64),
    ("8K", 5.96),
    ("16K", 8.72),
    ("32K", 13.7),
    ("64K", 23.2),
    ("128K", 43.8),
    ("256K", 86.7),
]
MODES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "axis": "#b9b8b2",
              "line": "#3987e5"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "grid": "#33332f", "axis": "#5c5b55",
             "line": "#86b6ef"},
}
W, H = 960, 420
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
LEFT, RIGHT, TOP, BOTTOM = 84, 930, 80, 312   # plot area in px
YMAX, STEP = 100, 20                           # seconds


def svg(mode):
    c = MODES[mode]
    n = len(FIRST_TOKEN_S)
    xs = [LEFT + 20 + i * (RIGHT - LEFT - 40) / (n - 1) for i in range(n)]
    scale = (BOTTOM - TOP) / YMAX
    ys = [BOTTOM - v * scale for _, v in FIRST_TOKEN_S]
    desc = ("Close-out build, one request at a time on the two-slot default, thinking off. Seconds to the first token: "
            + ", ".join(f"{k} {v} s" for k, v in FIRST_TOKEN_S) + ".")
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="{FONT}" role="img" aria-labelledby="t d">',
         '<title id="t">Volta Max uPLIft: time to first token by prompt length</title>',
         f'<desc id="d">{desc}</desc>',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>',
         f'<text x="24" y="36" fill="{c["text"]}" font-size="15" font-weight="600">'
         'Time to first token by prompt length, seconds</text>']
    for v in range(0, YMAX + 1, STEP):
        y = BOTTOM - v * scale
        p.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c["grid"] if v else c["axis"]}" '
                 'stroke-width="1"/>')
        p.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" fill="{c["text2"]}" font-size="12" text-anchor="end">{v} s</text>')
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    p.append(f'<polyline points="{pts}" fill="none" stroke="{c["line"]}" stroke-width="2.5" stroke-linejoin="round"/>')
    for (k, v), x, y in zip(FIRST_TOKEN_S, xs, ys):
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c["line"]}" stroke="{c["surface"]}" stroke-width="2">'
                 f'<title>{k} prompt: first token {v} s</title></circle>')
        p.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" fill="{c["text"]}" font-size="12" text-anchor="middle">{v} s</text>')
        p.append(f'<text x="{x:.1f}" y="{BOTTOM + 20}" fill="{c["text2"]}" font-size="12" text-anchor="middle">{k}</text>')
    p.append(f'<text x="{(LEFT + RIGHT) / 2:.1f}" y="{BOTTOM + 40}" fill="{c["text2"]}" font-size="12" '
             'text-anchor="middle">prompt length, tokens (each step doubles)</text>')
    for i, line in enumerate(("Measured 2026-11-13 on 8x V100-SXM2-32GB, close-out build, one request at a time on the "
                              "two-slot default, thinking off,",
                              "at the default 300 W power limit and 1,530 MHz maximum clock. "
                              "Fast prompt reading: quality not established.")):
        p.append(f'<text x="24" y="{H - 30 + 16 * i}" fill="{c["text2"]}" font-size="11">{line}</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        f = out / f"first-token-by-length-closeout-{mode}.svg"
        f.write_text(svg(mode))
        print(f"WROTE {f} {f.stat().st_size:,} B")


if __name__ == "__main__":
    main()
