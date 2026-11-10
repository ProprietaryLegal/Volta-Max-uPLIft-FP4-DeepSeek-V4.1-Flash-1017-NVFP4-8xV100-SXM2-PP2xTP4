#!/usr/bin/python3 -I
"""bench_volta_summary.py RECEIPT_DIR: print Volta sPLIt's README speed tables from benchvolta receipts
(run.json + cells.jsonl), in the README's layout and with its aggregation:
  Long prompts / Thinking on: each cell = mean of its runs; first token = median of the kinds; prompt reading = range
  over the kinds; decode per kind. Prompt reading by text type: mean of runs. Short prompts: median of reps (range),
  tokens per round at T=0 = median; thinking-on row = token-weighted sum(tokens - 1) / sum(decode time) over the 4
  prompts, with the per-prompt range. A cell the run did not reach prints "-". Reads numbers only.
"""
import json, statistics, sys
from pathlib import Path

KINDS = ("code", "structured", "legal", "prose", "chat")
LABEL = {"code": "code", "structured": "structured (JSON)", "legal": "legal", "prose": "prose", "chat": "chat"}


def ft(s):
    return f"{s:.2f} s" if s < 10 else f"{s:.1f} s"


def n0(x):
    return f"{x:,.0f}"


def mean(xs):
    return sum(xs) / len(xs)


def main():
    d = Path(sys.argv[1])
    run = json.loads((d / "run.json").read_text())
    cells = [json.loads(x) for x in (d / "cells.jsonl").read_text().splitlines()] if (d / "cells.jsonl").exists() else []
    kinds = [k for k in KINDS if k in run["kinds"]]
    srv = run.get("server") or {}
    print(f"# benchvolta {d.name}: server model {srv.get('model')}, status {run['status']}, "
          f"{run.get('cells_done', len(cells))}/{run['cells_planned']} cells")
    if run.get("stop"):
        print(f"STOP: {json.dumps(run['stop'])}")
    pr = run.get("prompts") or {}
    if pr.get("drift"):
        print("PROMPT DRIFT from Volta's sweep (prompts are not Volta's): " + "; ".join(pr["drift"]))
    print()

    def long_part(part):
        by = {}
        for c in cells:
            if c["part"] == part:
                by.setdefault((c["length"], c["kind"]), []).append(c)
        return by, sorted({L for L, _ in by})

    def kL(L):
        return f"{L // 1024}K"

    # ---------------------------------------------------------------- table 1
    by, lengths = long_part("long-off")
    print("### Long prompts: 1K to 256K\n")
    print("| prompt | first token | prompt reading | " + " | ".join(kinds) + " |")
    print("|---|---:|---:|" + "---:|" * len(kinds))
    maxdiff, same, pairs, multi_first = 0.0, 0, 0, 0
    for L in lengths:
        tt, rd, dec = [], [], []
        for k in kinds:
            rs = by.get((L, k), [])
            if rs:
                tt.append(mean([r["ttft_s"] for r in rs]))
                rd.append(mean([r["prompt_read_tok_s"] for r in rs]))
                dec.append(f"{mean([r['decode_tok_s'] for r in rs]):.1f}")
                multi_first += sum(r["first_event_tokens"] != 1 for r in rs)
                if len(rs) >= 2:
                    pairs += 1
                    same += len({r["ids_sha"] for r in rs}) == 1
                    ds = [r["decode_tok_s"] for r in rs]
                    maxdiff = max(maxdiff, max(ds) - min(ds))
            else:
                dec.append("-")
        rng = f"{n0(min(rd))}-{n0(max(rd))} tok/s" if rd else "-"
        print(f"| {kL(L)} | {ft(statistics.median(tt)) if tt else '-'} | {rng} | " + " | ".join(dec) + " |")
    n_off = sum(c["part"] == "long-off" for c in cells)
    print(f"\n{n_off} requests. Largest difference between the two runs of a cell {maxdiff:.1f} tok/s; greedy output "
          f"byte-identical in {same} of {pairs} cells. Cells whose first event carried more than one token: "
          f"{multi_first} (decode = (tokens - 1) / span holds when 0).")
    acc = {k: [c["tok_per_round"] - 1 for c in cells if c["part"] == "long-off" and c["kind"] == k] for k in kinds}
    if any(acc.values()):
        print("Accepted drafts per round (thinking off, mean over lengths): "
              + ", ".join(f"{k} {mean(v):.2f}" for k, v in acc.items() if v) + ".")
    print()

    # ---------------------------------------------------------------- table 2
    by_on, lengths_on = long_part("long-on")
    print("### Thinking on\n")
    print("| prompt | first token | " + " | ".join(kinds) + " |")
    print("|---|---:|" + "---:|" * len(kinds))
    for L in lengths_on:
        tt, dec = [], []
        for k in kinds:
            rs = by_on.get((L, k), [])
            if rs:
                tt.append(mean([r["ttft_s"] for r in rs]))
                dec.append(f"{mean([r['decode_tok_s'] for r in rs]):.1f}")
            else:
                dec.append("-")
        print(f"| {kL(L)} | {ft(statistics.median(tt)) if tt else '-'} | " + " | ".join(dec) + " |")
    print()

    # ---------------------------------------------------------------- table 3
    print("### Prompt reading by text type\n")
    print("| prompt | " + " | ".join(kinds) + " |")
    print("|---|" + "---:|" * len(kinds))
    for L in lengths:
        row = []
        for k in kinds:
            rs = by.get((L, k), [])
            row.append(n0(mean([r["prompt_read_tok_s"] for r in rs])) if rs else "-")
        print(f"| {kL(L)} | " + " | ".join(row) + " |")
    print()

    # ---------------------------------------------------------------- table 4
    print("### Short prompts, by text type\n")
    print("| kind of output | greedy (T=0) | sampled (T=1) | tokens per round, T=0 |")
    print("|---|---:|---:|---:|")
    ident, toks = [], []
    for k in kinds:
        g = [c for c in cells if c["part"] == "short" and c["kind"] == k and c["temperature"] == 0]
        s = [c for c in cells if c["part"] == "short" and c["kind"] == k and c["temperature"] == 1]
        if not g and not s:
            continue

        def cell(rs, bold):
            if not rs:
                return "-"
            v = [r["decode_tok_s"] for r in rs]
            m = f"{statistics.median(v):.1f}"
            return (f"**{m}**" if bold else m) + f" ({min(v):.1f}-{max(v):.1f})"

        tpr = f"{statistics.median([r['tok_per_round'] for r in g]):.2f}" if g else "-"
        print(f"| {LABEL[k]} | {cell(g, True)} | {cell(s, False)} | {tpr} |")
        if g:
            ident.append(f"{k} {'yes' if len({r['text_sha'] for r in g}) == 1 else 'NO'} ({len(g)} reps)")
            toks.append(f"{k} {statistics.median([r['completion_tokens'] for r in g + s]):.0f}")
    th = [c for c in cells if c["part"] == "think-short"]
    if th:
        tw = sum(c["completion_tokens"] - 1 for c in th) / sum(c["decode_s"] for c in th)
        v = [c["decode_tok_s"] for c in th]
        print(f"| thinking on, {len(th)} prompts, token-weighted | - | **{tw:.1f}** ({min(v):.1f}-{max(v):.1f} per "
              f"prompt) | - |")
    print()
    if ident:
        print("Greedy repeats byte-identical: " + ", ".join(ident) + ".")
        print("Median completion tokens (natural stop, max 1,024): " + ", ".join(toks) + ".")
    if th:
        print("Thinking-on prompts: " + ", ".join(f"{c['pid']} {c['decode_tok_s']:.1f} tok/s "
                                                  f"({c['completion_tokens']} tokens)" for c in th) + ".")


if __name__ == "__main__":
    main()
