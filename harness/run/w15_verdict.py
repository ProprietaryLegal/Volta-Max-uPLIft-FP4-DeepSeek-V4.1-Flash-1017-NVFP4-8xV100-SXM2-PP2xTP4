#!/usr/bin/python3 -I
"""w15_verdict.py: the readings and gates of window 15 (three chat slots). Rule: docs/acceptance/WINDOW15-accept-rule.md. Standard library.

  trace LOG                       FREE 0=BYTES 20=BYTES: the least `card_free` at `at=enter` over each stage's
                                  four ranks (VLLM_NVFP4_8R_LOAD_TRACE=1 boots). Stage by the log prefix Worker_PP0 /
                                  Worker_PP1. Needs exactly one enter line per rank, eight in all.
  tiers LOG                       one TIERS line per stage and a DIGESTS line, from the loader lines; the four ranks
                                  of a stage must log the same figures.
  creserve PROFILE_B A0 A20 B0 B20
                                  RESERVE_C R0 R20 = B's reserve + free_A - free_B per stage (bytes): a one-slot
                                  boot with these reserves plans B's record budget.
  mem FILE --oom N [--limit MIB] [--card MIB] [--need-mark LABEL]...
                                  MEMORY PASS | FAIL | INCONCLUSIVE on the sampler file (V100s numbered 0..7 as in
                                  run/w9_common.sh cards_mib: stage 0 = 0 1 4 5, stage 1 = 2 3 6 7).
  rotation JSONL --slots N        ROTATION PASS | FAIL | INCONCLUSIVE on rotation_order.py's receipt.
  rotation JSONL --strict CHATS   the same checks on a strict A B C rotation of run/harness_loop.py (older receipts).
  speed MAIN NOISE A1 B1 A2 B2    SPEED PASS | FAIL | INCONCLUSIVE from pairs_arms_v2.py's outputs (window 12's rule).
  decide a=S b=S c=S d=S e=S t=S  the DECISION W15 line. S = PASS | FAIL | INCONCLUSIVE | NOT-RUN.
  self-test [--root DIR]          real lane files under DIR (default the lane root).
Exit: 0 PASS (or a reading printed), 1 FAIL, 3 INCONCLUSIVE, 2 usage or an unreadable input.
Prints counts, bytes, times and digest prefixes only.
"""
from __future__ import annotations

import os
import json
import re
import sys
from pathlib import Path

ROOT = Path(os.environ["LANE"])
LIMIT_MIB, CARD_MIB = 31998, 32510       # the build's memory line: card size less 512 MiB
STAGE_OF_CARD = {0: 0, 1: 0, 4: 0, 5: 0, 2: 1, 3: 1, 6: 1, 7: 1}   # run/w9_common.sh cards_mib numbering
HIT_TTFT_FRACTION = 0.5                   # a kept follow-up's first token within half the fastest full read
PASS, FAIL, INC = "PASS", "FAIL", "INCONCLUSIVE"
CODE = {PASS: 0, FAIL: 1, INC: 3}

TRACE = re.compile(r"\(Worker_PP([01])_TP([0-3]) pid=\d+\).*nvfp4_8r load trace: rank=\d+ at=enter .*?card_free=(\d+) ")
LOADER = re.compile(
    r"\(Worker_PP([01])_TP([0-3]) pid=\d+\).*nvfp4_8r: TP rank (\d) layers (\d+)-(\d+) residency \S+ (?:prefill_warm \S+ )?"
    r"free ([\d.]+) GiB \(stage least ([\d.]+)\) reserve ([\d.]+) GiB (?:prefill workspace [\d.]+ GiB )?budget ([\d.]+) GiB "
    r"records [\d.]+ GiB, (\d+) hot / (\d+) warm / (\d+) host experts, decode route share ([\d.]+) / [\d.]+ / [\d.]+, "
    r"tier digest ([0-9a-f]{12}), free after ([\d.]+) GiB")


class Undecided(Exception):
    """An input that cannot give the reading: the caller says INCONCLUSIVE."""


# --- readings -----------------------------------------------------------------------------------------------------
def trace_free(text: str) -> dict[int, int]:
    seen: dict[tuple[int, int], int] = {}
    for m in TRACE.finditer(text):
        key = (int(m[1]), int(m[2]))
        if key in seen:
            raise Undecided(f"two at=enter trace lines for PP{key[0]} TP{key[1]} (one boot per log expected)")
        seen[key] = int(m[3])
    if sorted(seen) != [(p, t) for p in (0, 1) for t in range(4)]:
        raise Undecided(f"{len(seen)} of 8 at=enter trace lines (was the boot made with VLLM_NVFP4_8R_LOAD_TRACE=1?)")
    return {0: min(v for (p, _), v in seen.items() if p == 0), 20: min(v for (p, _), v in seen.items() if p == 1)}


def tiers(text: str) -> dict[int, dict]:
    per: dict[int, list[dict]] = {0: [], 20: []}
    for m in LOADER.finditer(text):
        first = int(m[4])
        if first not in per or int(m[1]) != (0 if first == 0 else 1):
            raise Undecided(f"a loader line for layers {m[4]}-{m[5]} from PP{m[1]}")
        per[first].append({"layers": f"{m[4]}-{m[5]}", "least_gib": m[7], "reserve_gib": m[8], "budget_gib": m[9],
                           "counts": (int(m[10]), int(m[11]), int(m[12])), "hot_share": m[13], "digest12": m[14],
                           "free_after_gib": m[15]})
    out = {}
    for first, rows in per.items():
        if len(rows) != 4:
            raise Undecided(f"layers starting {first}: {len(rows)} of 4 loader lines")
        if any({k: v for k, v in r.items() if k != "free_after_gib"} != {k: v for k, v in rows[0].items() if k != "free_after_gib"}
               for r in rows):
            raise Undecided(f"layers starting {first}: the four ranks logged different tiers or budgets")
        out[first] = rows[0]
    return out


def profile_reserves(text: str) -> dict[int, int]:
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if l == "  nvfp4_8r_reserve_bytes:"]
    if len(idx) != 1:
        raise Undecided(f"{len(idx)} lines '  nvfp4_8r_reserve_bytes:' in the profile")
    got: dict[int, int] = {}
    for l in lines[idx[0] + 1:]:
        if l.strip().startswith("#"):
            continue
        m = re.fullmatch(r"    (\d+): (\d+)", l)
        if not m:
            break
        got[int(m[1])] = int(m[2])
    if sorted(got) != [0, 20]:
        raise Undecided(f"the profile's reserves name stages {sorted(got)}, expected [0, 20]")
    return got


def creserve(prof_b: str, fa: dict[int, int], fb: dict[int, int]) -> dict[int, int]:
    rb = profile_reserves(prof_b)
    rc = {k: rb[k] + fa[k] - fb[k] for k in (0, 20)}
    if any(not 0 < v < 32 * 2**30 for v in rc.values()):
        raise Undecided(f"reserve_C {rc} is not a positive byte count below 32 GiB")
    return rc


def mem(text: str, oom: int, limit: int, card: int, need_marks: list[str]) -> tuple[str, str]:
    high = {0: 0, 1: 0}; n = 0; ends: list[str] = []; marks: dict[str, int] = {}; per_mark: dict[str, int] = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 3 and f[1] == "MARK":
            marks[" ".join(f[2:])] = n
            continue
        if len(f) >= 2 and f[1] in ("STOPPED", "NO-ANSWER"):
            ends.append(" ".join(f[1:])[:120])
            continue
        if len(f) != 9:
            raise Undecided(f"a sampler line that is not eight cards: {line[:80]!r}")
        cards = dict(tuple(int(x) for x in c.split(",")) for c in f[1:])
        if sorted(cards) != list(range(8)):
            raise Undecided(f"a sampler line without cards 0..7: {line[:80]!r}")
        for c, v in cards.items():
            high[STAGE_OF_CARD[c]] = max(high[STAGE_OF_CARD[c]], v)
        n += 1
    for lab in need_marks:
        s, e = marks.get(f"{lab} start"), marks.get(f"{lab} end")
        per_mark[lab] = -1 if s is None or e is None else e - s
    fig = (f"stage 0 high mark {high[0]} MiB (to the card {card - high[0]}, to the limit {limit - high[0]}), "
           f"stage 1 high mark {high[1]} MiB (to the card {card - high[1]}, to the limit {limit - high[1]}); "
           .replace("; ", ", ") + f"samples {n}, out-of-memory lines {oom}"
           + "".join(f", samples in {k} {v if v >= 0 else 'no marks'}" for k, v in per_mark.items()))
    if oom > 0 or max(high.values()) > limit:
        return FAIL, fig
    if n == 0 or ends or any(v < 2 for v in per_mark.values()):
        return INC, fig + (f", sampler ended early: {ends[0]}" if ends else "")
    return PASS, fig


def rotation_order(slots: int) -> tuple[list[int], list]:
    """Visits and the expected outcome of each (None a first read, True kept, False re-read): chats 0..n-1, a round of
    follow-ups, chat n (evicts chat 0: the free slot freed longest ago, v41_multi_seq.py), chats 1..n-1, chat 0."""
    n = slots
    order = list(range(n)) + list(range(n)) + [n] + list(range(1, n)) + [0]
    expect = [None] * n + [True] * n + [None] + [True] * (n - 1) + [False]
    return order, expect


def rotation(recs: list[dict], order: list[int], expect: list, labels: bool) -> tuple[str, str]:
    rows = []
    if len(recs) < len(order):
        raise Undecided(f"{len(recs)} of {len(order)} visits in the receipt (a chat ended its tool loop early, or the run stopped)")
    if len(recs) > len(order):
        raise Undecided(f"{len(recs)} records for {len(order)} visits")
    firsts = []
    for k, (r, c, e) in enumerate(zip(recs, order, expect)):
        if r.get("chat") != c or (labels and r.get("req") != f"v{k}"):
            raise Undecided(f"visit {k}: record chat {r.get('chat')} req {r.get('req')}, expected chat {c} req v{k}")
        if not isinstance(r.get("ttft_s"), (int, float)) or not isinstance(r.get("kept_tokens"), int):
            raise Undecided(f"visit {k}: no ttft_s / kept_tokens")
        if (r.get("hit") is None) != (e is None):
            raise Undecided(f"visit {k}: first-read flag {r.get('hit')!r}, expected {'a first read' if e is None else 'a follow-up'}")
        if e is None:
            firsts.append(r["ttft_s"])
    fast = min(firsts)
    bad = []
    for k, (r, c, e) in enumerate(zip(recs, order, expect)):
        word = "first" if e is None else ("HIT" if r["hit"] else "MISS")
        rows.append(f"v{k} chat {c} {word} kept {r['kept_tokens']} of {r['ctx_tokens']} ttft {r['ttft_s']} s")
        if e is True and not r["hit"]:
            bad.append(f"v{k} chat {c} lost its prompt (expected kept)")
        if e is True and r["hit"] and not r["ttft_s"] < HIT_TTFT_FRACTION * fast:
            bad.append(f"v{k} chat {c} kept but first token {r['ttft_s']} s is not under {HIT_TTFT_FRACTION} x {fast} s")
        if e is False and r["hit"]:
            bad.append(f"v{k} chat {c} kept its prompt where the extra chat should have evicted it")
    hits = [r["ttft_s"] for r, e in zip(recs, expect) if e is True and r["hit"]]
    fig = (f"{sum(1 for e in expect if e is True)} kept follow-ups expected, {len(hits)} kept, first token "
           f"{min(hits) if hits else '-'} to {max(hits) if hits else '-'} s against full reads {min(firsts)} to {max(firsts)} s; "
           .replace("; ", ", ") + ("; ".join(bad) if bad else "every visit as expected").replace(";", ","))
    return (FAIL if bad else PASS), fig + "\n" + "\n".join(rows)


T0 = re.compile(r"^T0 (\S+) - (\S+): mean ([-+][0-9.]+) ms/round-dec")


def speed(main: str, noise: str, a1: str, b1: str, a2: str, b2: str) -> tuple[str, str]:
    def means(text: str) -> dict[tuple[str, str], float]:
        return {(m[1], m[2]): float(m[3]) for m in (T0.match(l) for l in text.splitlines()) if m}
    mm, nn = means(main), means(noise)
    try:
        p1, p2 = mm[(b1, a1)], mm[(b2, a2)]
        ns = [nn[(f"{a1}r", a1)], nn[(f"{b1}r", b1)], nn[(f"{b2}r", b2)], nn[(f"{a2}r", a2)], nn[(a2, a1)]]
    except KeyError as k:
        raise Undecided(f"no T0 line for the pair {k}")
    floor = max(abs(x) for x in ns); d = (p1 + p2) / 2
    fig = (f"p1 (B1-A1) {p1:+.3f}, p2 (B2-A2) {p2:+.3f}, mean {d:+.3f} ms/round-dec, noise floor {floor:.3f} "
           f"(largest of |{' '.join(f'{x:+.3f}' for x in ns)}|), 2x {2 * floor:.3f}")
    if p1 > 0 and p2 > 0 and d > 2 * floor:
        return FAIL, "slots arm slower beyond twice the noise floor: " + fig
    return PASS, "slots arm not slower beyond twice the noise floor: " + fig


GATES = ("a", "b", "c", "d", "e", "t")
NAMES = {"a": "memory", "b": "token-exact", "c": "speed, one chat", "d": "R10 rotation and 500K read",
         "e": "Xid count", "t": "tier reproducibility"}


def decide(st: dict[str, str]) -> tuple[str, str]:
    lines = [f"GATE ({g}) {NAMES[g]}: {st[g]}" for g in GATES]
    if any(st[g] == FAIL for g in GATES):
        return FAIL, "\n".join(lines) + "\nDECISION W15: NOT INSTALLABLE (a gate failed)"
    if all(st[g] == PASS for g in GATES):
        return PASS, "\n".join(lines) + "\nDECISION W15: INSTALLABLE by the rule (every gate PASS). Installing is the maintainer's call."
    return INC, "\n".join(lines) + "\nDECISION W15: INCONCLUSIVE (no gate failed, not every gate passed)"


# --- self-test on real lane files ---------------------------------------------------------------------------------
def self_test(root: Path) -> int:
    ok = True

    def check(name: str, cond: bool, got) -> None:
        nonlocal ok
        print(f"{'ok  ' if cond else 'FAIL'} {name}: {got}")
        ok = ok and cond

    rd = lambda p: (root / p).read_text(errors="replace")
    # 1 trace: a one-slot boot's load trace against the planner receipt that recorded the same boot's free bytes
    want = {int(k): v for k, v in json.loads(rd("stage/piq-load/exact-tier-profiles.json"))["free_bytes_scope_off"].items()}
    got = trace_free(rd("run/state/W4-piq-so.log"))
    check("trace W4-piq-so = exact-tier-profiles.json free_bytes_scope_off", got == want, got)
    try:
        trace_free(rd("run/state/W12-B1.log")); check("trace refuses a boot without the trace", False, "accepted")
    except Undecided as e:
        check("trace refuses a boot without the trace", True, e)
    # 2 tiers: the installed boot (the accepted tiers of 2026-11-11, which window 12 expected too)
    t = tiers(rd("run/state/W12-B1.log"))
    check("tiers W12-B1 = R1 (3833/3847/0 8e7829184090, 2459/5221/0 9473486077c6, reserves 4.50/7.00)",
          (t[0]["counts"], t[0]["digest12"], t[0]["reserve_gib"], t[20]["counts"], t[20]["digest12"], t[20]["reserve_gib"])
          == ((3833, 3847, 0), "8e7829184090", "4.50", (2459, 5221, 0), "9473486077c6", "7.00"), t)
    # the three-slot boot read by eye from its loader lines: 3346 / 1886 HOT, least 28.76 / 28.73
    t = tiers(rd("run/state/W9-hiq3b.log"))
    check("tiers W9-hiq3b = 3346/4334/0 least 28.76 and 1886/5794/0 least 28.73",
          (t[0]["counts"], t[0]["least_gib"], t[20]["counts"], t[20]["least_gib"])
          == ((3346, 4334, 0), "28.76", (1886, 5794, 0), "28.73"), {k: (v["counts"], v["least_gib"]) for k, v in t.items()})
    # 3 creserve: arithmetic. One-slot free (the boot of check 1) as A, the same minus the engine's three-slot state as B.
    prof = "  nvfp4_8r_reserve_bytes:\n    # c\n    0: 5167382528\n    20: 7851737088\n  nvfp4_8r_residency: hot_iq\n"
    fb = {0: want[0] - 572_833_382, 20: want[20] - 386_190_540}
    rc = creserve(prof, want, fb)
    check("creserve = reserve_B + free_A - free_B", rc == {0: 5167382528 + 572_833_382, 20: 7851737088 + 386_190_540}, rc)
    # 4 mem: two memory receipts of 2026-11-11, maxima as recorded then (31,694 / 30,994 and 31,456 on stage 1)
    st, fig = mem(rd("logs/baseline/W11-gpkmem-mem.txt"), 0, LIMIT_MIB, CARD_MIB, [])
    check("mem W11-gpkmem PASS with 31694 / 30994", st == PASS and "stage 0 high mark 31694 " in fig and "stage 1 high mark 30994 " in fig, (st, fig))
    st, fig = mem(rd("logs/baseline/W10a-mem-mem.txt"), 0, LIMIT_MIB, CARD_MIB, [])
    check("mem W10a-mem PASS with stage 1 31456", st == PASS and "stage 1 high mark 31456 " in fig, (st, fig))
    st, _ = mem(rd("logs/baseline/W10a-mem-mem.txt"), 1, LIMIT_MIB, CARD_MIB, [])
    check("mem FAIL on an out-of-memory line", st == FAIL, st)
    st, _ = mem(rd("logs/baseline/W10a-mem-mem.txt"), 0, 31700, CARD_MIB, [])
    check("mem FAIL above a limit of 31700 (W10a stage 0 31732)", st == FAIL, st)
    st, _ = mem(rd("logs/baseline/W10a-mem-mem.txt"), 0, LIMIT_MIB, CARD_MIB, ["long500k"])
    check("mem INCONCLUSIVE without the marks it needs", st == INC, st)
    st, _ = mem(rd("logs/baseline/W10a-mem-mem.txt") + "04:39:20 STOPPED Xid lines 3 -> 4; nvidia-smi not called\n", 0, LIMIT_MIB, CARD_MIB, [])
    check("mem INCONCLUSIVE when the sampler stopped early", st == INC, st)
    # 5 rotation: a three-slot boot's strict rotation (window 9: 6 of 6 kept)
    recs = [json.loads(l) for l in rd("logs/baseline/W9-hiq3b-rotate3.jsonl").splitlines() if l.strip()]
    order = [0, 1, 2] * 3; expect = [None] * 3 + [True] * 6
    st, fig = rotation(recs, order, expect, labels=False)
    check("rotation W9-hiq3b strict: PASS, 6 of 6 kept, 7.56 to 8.38 s", st == PASS and "6 kept, first token 7.56 to 8.38 s" in fig, fig.splitlines()[0])
    lost = [dict(r) for r in recs]; lost[5]["hit"] = False
    check("rotation FAIL when a kept follow-up is lost", rotation(lost, order, expect, labels=False)[0] == FAIL, "-")
    try:
        rotation(recs[:7], order, expect, labels=False); check("rotation refuses a short receipt", False, "accepted")
    except Undecided as e:
        check("rotation refuses a short receipt", True, e)
    o3, e3 = rotation_order(3); o2, e2 = rotation_order(2)
    check("order for 3 slots", (o3, e3) == ([0, 1, 2, 0, 1, 2, 3, 1, 2, 0], [None] * 3 + [True] * 3 + [None, True, True, False]), o3)
    check("order for 2 slots", (o2, e2) == ([0, 1, 0, 1, 2, 1, 0], [None, None, True, True, None, True, False]), o2)
    # 6 speed: window 12's receipts; its own VERDICT line (logs/dispatch/w12-verdict.out) says floor 2.605, p1 -2.634
    st, fig = speed(rd("logs/dispatch/w12-pairs-abba.out"), rd("logs/dispatch/w12-pairs-noise.out"), "W12-A1", "W12-B1", "W12-A2", "W12-B2")
    v = rd("logs/dispatch/w12-verdict.out")
    check("speed on window 12: PASS, numbers as its VERDICT line", st == PASS and "noise floor 2.605" in fig and "noise floor 2.605" in v
          and "p1 (B1-A1) -2.634" in fig and "p1 (B1-A1) -2.634" in v, fig)
    try:
        speed(rd("logs/dispatch/w12-pairs-abba.out"), rd("logs/dispatch/w12-pairs-noise.out"), "W12-B1", "W12-A1", "W12-B2", "W12-A2")
        check("speed refuses pairs the receipts lack (arms swapped)", False, "accepted")
    except Undecided as e:
        check("speed refuses pairs the receipts lack (arms swapped)", True, e)
    flip = rd("logs/dispatch/w12-pairs-abba.out").replace("mean -2.634", "mean +5.300").replace("mean -3.847", "mean +5.400")
    check("speed FAIL when both pairs are slower beyond 2x the floor (+5.30, +5.40 > 5.210)",
          speed(flip, rd("logs/dispatch/w12-pairs-noise.out"), "W12-A1", "W12-B1", "W12-A2", "W12-B2")[0] == FAIL, "-")
    # 7 decide
    allp = dict.fromkeys(GATES, PASS)
    check("decide all PASS", decide(allp)[0] == PASS, "-")
    check("decide one FAIL", decide({**allp, "c": FAIL, "b": INC})[0] == FAIL, "-")
    check("decide one INCONCLUSIVE", decide({**allp, "b": INC})[0] == INC, "-")
    check("decide one NOT-RUN", decide({**allp, "d": "NOT-RUN"})[0] == INC, "-")
    print("SELF-TEST PASS" if ok else "SELF-TEST FAIL")
    return 0 if ok else 1


def main(a: list[str]) -> int:
    try:
        if a[:1] == ["self-test"]:
            root = Path(a[2]) if len(a) == 3 and a[1] == "--root" else ROOT
            return self_test(root)
        if len(a) == 2 and a[0] == "trace":
            f = trace_free(Path(a[1]).read_text(errors="replace"))
            print(f"FREE 0={f[0]} 20={f[20]}")
            return 0
        if len(a) == 2 and a[0] == "tiers":
            t = tiers(Path(a[1]).read_text(errors="replace"))
            for k, r in t.items():
                print(f"TIERS layers {r['layers']} reserve {r['reserve_gib']} GiB least {r['least_gib']} GiB budget "
                      f"{r['budget_gib']} GiB {r['counts'][0]} hot / {r['counts'][1]} warm / {r['counts'][2]} host "
                      f"hot share {r['hot_share']} digest {r['digest12']} free after {r['free_after_gib']} GiB (4 ranks)")
            print(f"DIGESTS 0={t[0]['digest12']} 20={t[20]['digest12']}")
            print(f"COUNTS 0={'/'.join(map(str, t[0]['counts']))} 20={'/'.join(map(str, t[20]['counts']))}")
            return 0
        if len(a) == 6 and a[0] == "creserve" and all(x.isdecimal() for x in a[2:]):
            rc = creserve(Path(a[1]).read_text(), {0: int(a[2]), 20: int(a[3])}, {0: int(a[4]), 20: int(a[5])})
            print(f"RESERVE_C {rc[0]} {rc[20]}")
            return 0
        if len(a) >= 4 and a[0] == "mem" and a[2] == "--oom" and a[3].isdecimal():
            limit, card, marks, rest = LIMIT_MIB, CARD_MIB, [], a[4:]
            while rest:
                if rest[0] == "--limit" and len(rest) > 1: limit = int(rest[1])
                elif rest[0] == "--card" and len(rest) > 1: card = int(rest[1])
                elif rest[0] == "--need-mark" and len(rest) > 1: marks.append(rest[1])
                else: raise SystemExit(2)
                rest = rest[2:]
            st, fig = mem(Path(a[1]).read_text(), int(a[3]), limit, card, marks)
            print(f"MEMORY {st}: {fig}")
            return CODE[st]
        if len(a) == 4 and a[0] == "rotation" and a[2] in ("--slots", "--strict") and a[3].isdecimal():
            recs = [json.loads(l) for l in Path(a[1]).read_text().splitlines() if l.strip()]
            n = int(a[3])
            if a[2] == "--slots":
                if n not in (2, 3):
                    raise SystemExit(2)
                order, expect = rotation_order(n)
                st, fig = rotation(recs, order, expect, labels=True)
            else:
                k = len(recs) // n
                st, fig = rotation(recs, list(range(n)) * k, [None] * n + [True] * (n * (k - 1)), labels=False)
            print(f"ROTATION {st}: {fig}")
            return CODE[st]
        if len(a) == 7 and a[0] == "speed":
            st, fig = speed(Path(a[1]).read_text(), Path(a[2]).read_text(), *a[3:7])
            print(f"SPEED {st}: {fig}")
            return CODE[st]
        if a[:1] == ["decide"] and len(a) == 7:
            st = dict(x.split("=", 1) for x in a[1:])
            if sorted(st) != sorted(GATES) or any(v not in (PASS, FAIL, INC, "NOT-RUN") for v in st.values()):
                raise SystemExit(2)
            s, text = decide(st)
            print(text)
            return CODE[s]
    except Undecided as e:
        print(f"{INC}: {e}")
        return 3
    except (OSError, ValueError) as e:
        print(f"w15_verdict.py: cannot read the input: {e}", file=sys.stderr)
        return 2
    print(__doc__.split("\n\n")[0], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
