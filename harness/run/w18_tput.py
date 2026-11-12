#!/usr/bin/python3 -I
"""w18_tput.py: the throughput client and readings of window 18 (one to four chat slots). Rule: docs/acceptance/WINDOW18-accept-rule.md.
Standard library only. Prompts: the three public short prompts (TPUT_PROMPTS, as in harness/run/tput_prompts.py, from --ab)
and Python standard-library source for the long prompt. Receipts hold token ids, hashes, counts and times, no text.

  seq URL ARM RUN OUT.jsonl --ab AB.py
        level 1: p0, p1, p2 one after another (thinking off, temperature 0, ignore_eos, 1,024 tokens out).
  level URL K ARM RUN OUT.jsonl --ab AB.py
        level K (2..4): K requests started together behind a thread barrier, request i on p(i mod 3).
  mixed URL CHARS DELAY ARM OUT.jsonl --ab AB.py [--src-dir DIR]
        a long prompt of CHARS characters of DIR/*.py source (longprompt.py's text, 48 tokens out), and DELAY s later
        the short request p0 (1,024 tokens).
  compare ARM DIR PREFIX
        TOKENS MATCH | DIFFER: every request of DIR/PREFIX-ARM-L<k>-r<n>.jsonl (k >= 2) and of the mixed receipt
        against the same prompt in L1 run 2, and L1 run 1 against L1 run 2.
  table DIR PREFIX
        the TABLE W18 lines (run 2, the declared reading) and the run-1 table, from every receipt of PREFIX in DIR.
  self-test
        arithmetic checks of the readings on records built here.
Exit: 0 done (MATCH for compare), 1 a request incomplete or DIFFER, 2 usage or an unreadable input, 3 INCONCLUSIVE, 4 a client error (HTTP, stream or connection).
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

MAX_TOKENS = 1024
LONG_TOKENS = 48
ROUNDS_TOL = 0.02                       # docs/acceptance/WINDOW18-accept-rule.md (r)
ARMS = ("S1", "S2", "S3", "S4")
SLOTS_OF = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
NAME = re.compile(r"^(?P<arm>S[1-4])-L(?P<k>[1-4])-r(?P<run>[12])\.jsonl$")


def die(msg: str, code: int = 2) -> None:
    print(f"w15_tput: {msg}", file=sys.stderr)
    sys.exit(code)


def load_ab(path: str):
    spec = importlib.util.spec_from_file_location("ab_t1_v2_readonly", path)
    if spec is None or spec.loader is None:
        die(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompts = list(mod.TPUT_PROMPTS)
    if len(prompts) != 3:
        die(f"{path}: TPUT_PROMPTS has {len(prompts)} prompts, expected 3")
    return mod.MODEL, prompts


# --- the client ---------------------------------------------------------------------------------------------------
def drafts(url: str) -> float:
    text = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    total, seen = 0.0, False
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total{") or line.startswith("vllm:spec_decode_num_drafts_total "):
            total += float(line.split()[-1]); seen = True
    if not seen:
        die(f"{url}/metrics has no vllm:spec_decode_num_drafts_total")
    return total


def stream(url: str, body: dict) -> dict:
    """POST one streamed chat completion; times and token ids per chunk. Raises on any HTTP or stream error."""
    body = dict(body, stream=True, stream_options={"include_usage": True}, return_token_ids=True)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ids: list[int] = []
    marks: list[tuple[float, int]] = []          # (time, ids in the chunk) for every chunk that carried ids
    usage, finish = {}, None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("error"):
                raise RuntimeError(f"stream error: {str(obj['error'])[:200]}")
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                tid = ch.get("token_ids") or []
                if tid:
                    ids.extend(int(x) for x in tid)
                    marks.append((time.perf_counter(), len(tid)))
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    end = time.perf_counter()
    return {"t0": t0, "end": end, "ids": ids, "marks": marks, "usage": usage, "finish": finish}


def record(s: dict, want: int) -> dict:
    """The per-request reading of docs/acceptance/WINDOW18-accept-rule.md section 4 from one stream()."""
    n = int(s["usage"].get("completion_tokens") or 0)
    marks = s["marks"]
    first = marks[0][0] if marks else None
    last = marks[-1][0] if marks else None
    dec_tokens = sum(k for _, k in marks[1:])
    dec_s = (last - first) if marks else None
    chunks = len(marks)
    return {
        "tokens": n, "want": want, "ids_complete": len(s["ids"]) == n == want,
        "prompt_tokens": int(s["usage"].get("prompt_tokens") or 0),
        "t0": s["t0"], "first": first, "last": last, "end": s["end"],
        "wall_s": s["end"] - s["t0"], "ttft_s": (first - s["t0"]) if first is not None else None,
        "dec_s": dec_s, "dec_tokens": dec_tokens, "chunks": chunks,
        "dec_tok_s": (dec_tokens / dec_s) if dec_s else None,
        "ms_round": (dec_s * 1e3 / (chunks - 1)) if dec_s and chunks > 1 else None,
        "finish": s["finish"], "toksha": hashlib.sha256(json.dumps(s["ids"]).encode()).hexdigest()[:16],
        "ids": s["ids"],
    }


def short_body(model: str, prompt: str) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": MAX_TOKENS, "ignore_eos": True, "chat_template_kwargs": {"thinking": False}}


def long_body(model: str, chars: int, src_dir: str) -> dict:
    txt = ""
    for f in sorted(glob.glob(f"{src_dir}/*.py")) + sorted(glob.glob(f"{src_dir}/*/*.py")):
        txt += open(f, errors="replace").read()
        if len(txt) >= chars:
            break
    if len(txt) < chars:
        die(f"only {len(txt)} characters of source under {src_dir}, asked {chars}")
    return {"model": model, "temperature": 0, "max_tokens": LONG_TOKENS, "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": "Here is Python standard-library source code.\n\n" + txt[:chars]
                          + "\n\nIn one sentence, say what kind of code this is."}]}


class Out:
    """A receipt opened exclusively: a repeated command refuses instead of overwriting."""

    def __init__(self, path: str):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            die(f"{path} exists: not overwritten")
        self.f = os.fdopen(fd, "w")

    def put(self, obj: dict) -> None:
        self.f.write(json.dumps(obj) + "\n"); self.f.flush()

    def close(self) -> None:
        self.f.close()


def fmt(x, spec: str) -> str:
    return "nan" if x is None else format(x, spec)


def req_line(arm: str, level: int, run: int, i: int, pid: str, r: dict) -> str:
    return (f"REQ {arm} L{level} r{run} q{i} {pid} tokens {r['tokens']} ttft {fmt(r['ttft_s'], '.3f')} s "
            f"dec {fmt(r['dec_tok_s'], '.1f')} tok/s {fmt(r['ms_round'], '.2f')} ms/round chunks {r['chunks']} "
            f"toksha {r['toksha']} ids {'ok' if r['ids_complete'] else 'INCOMPLETE'}")


def summarize(recs: list[dict], level: int, server_drafts: float | None) -> dict:
    """The per-level reading of docs/acceptance/WINDOW18-accept-rule.md section 4 and checks (c) and (r)."""
    tokens = sum(r["tokens"] for r in recs)
    if level == 1:
        wall = sum(r["wall_s"] for r in recs)                       # one stream: three requests in a row
    else:
        wall = max(r["end"] for r in recs) - min(r["t0"] for r in recs)
    dts = [r["dec_tok_s"] for r in recs if r["dec_tok_s"] is not None]
    mss = [r["ms_round"] for r in recs if r["ms_round"] is not None]
    chunks = sum(r["chunks"] for r in recs)
    if server_drafts is None or server_drafts <= 0:
        rounds = "INCONCLUSIVE"
    else:
        rounds = "OK" if abs(chunks - server_drafts) <= ROUNDS_TOL * server_drafts else "INCONCLUSIVE"
    complete = sum(r["ids_complete"] for r in recs)
    return {
        "kind": "level", "level": level, "requests": len(recs), "complete": complete, "tokens": tokens,
        "wall_s": wall, "agg_tok_s": tokens / wall if wall > 0 else None,
        "dec_mean": sum(dts) / len(dts) if dts else None, "dec_min": min(dts) if dts else None,
        "dec_max": max(dts) if dts else None,
        "ms_mean": sum(mss) / len(mss) if mss and rounds == "OK" else None,
        "ms_min": min(mss) if mss and rounds == "OK" else None, "ms_max": max(mss) if mss and rounds == "OK" else None,
        "chunks": chunks, "server_drafts": server_drafts, "rounds": rounds,
        "start_skew_ms": (max(r["t0"] for r in recs) - min(r["t0"] for r in recs)) * 1e3 if level > 1 else 0.0,
        "status": "OK" if complete == len(recs) else "INCONCLUSIVE",
    }


def level_line(arm: str, run: int, s: dict) -> str:
    return (f"LEVEL {arm} L{s['level']} r{run} requests {s['requests']} tokens {s['tokens']} wall {s['wall_s']:.2f} s "
            f"aggregate {fmt(s['agg_tok_s'], '.1f')} tok/s per-request dec {fmt(s['dec_mean'], '.1f')} "
            f"({fmt(s['dec_min'], '.1f')}-{fmt(s['dec_max'], '.1f')}) tok/s ms/round {fmt(s['ms_mean'], '.2f')} "
            f"skew {s['start_skew_ms']:.0f} ms chunks {s['chunks']} server-drafts {fmt(s['server_drafts'], '.0f')} "
            f"rounds {s['rounds']} complete {s['complete']}/{s['requests']} {s['status']}")


def cmd_seq(url: str, arm: str, run: int, out: str, ab: str) -> int:
    model, prompts = load_ab(ab)
    o = Out(out); recs = []; d_all = 0.0
    for i, p in enumerate(prompts):
        a = drafts(url)
        r = record(stream(url, short_body(model, p)), MAX_TOKENS)
        b = drafts(url); d_all += b - a
        r.update(kind="req", arm=arm, level=1, run=run, req=i, prompt=f"p{i}", server_drafts=b - a)
        o.put(r); recs.append(r)
        print(req_line(arm, 1, run, i, f"p{i}", r) + f" server-drafts {b - a:.0f}", flush=True)
    s = summarize(recs, 1, d_all); s.update(arm=arm, run=run)
    o.put(s); o.close()
    print(level_line(arm, run, s), flush=True)
    return 0 if s["status"] == "OK" else 1


def cmd_level(url: str, k: int, arm: str, run: int, out: str, ab: str) -> int:
    model, prompts = load_ab(ab)
    o = Out(out)
    bar = threading.Barrier(k)
    res: list[dict | None] = [None] * k
    err: list[str] = []

    def one(i: int) -> None:
        try:
            bar.wait(timeout=60)
            res[i] = record(stream(url, short_body(model, prompts[i % 3])), MAX_TOKENS)
        except Exception as e:  # the error is reported and the level fails loudly below
            err.append(f"request {i}: {type(e).__name__}: {str(e)[:200]}")

    a = drafts(url)
    ts = [threading.Thread(target=one, args=(i,)) for i in range(k)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err:
        die("; ".join(err), 4)
    b = drafts(url)
    recs = []
    for i, r in enumerate(res):
        assert r is not None
        r.update(kind="req", arm=arm, level=k, run=run, req=i, prompt=f"p{i % 3}")
        o.put(r); recs.append(r)
        print(req_line(arm, k, run, i, f"p{i % 3}", r), flush=True)
    s = summarize(recs, k, b - a); s.update(arm=arm, run=run)
    o.put(s); o.close()
    print(level_line(arm, run, s), flush=True)
    return 0 if s["status"] == "OK" else 1


def cmd_mixed(url: str, chars: int, delay: float, arm: str, out: str, ab: str, src_dir: str) -> int:
    model, prompts = load_ab(ab)
    lb = long_body(model, chars, src_dir)
    o = Out(out)
    res: dict[str, dict] = {}
    err: list[str] = []

    def run(name: str, body: dict, want: int) -> None:
        try:
            res[name] = record(stream(url, body), want)
        except Exception as e:
            err.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")

    tl = threading.Thread(target=run, args=("long", lb, LONG_TOKENS))
    ts = threading.Thread(target=run, args=("short", short_body(model, prompts[0]), MAX_TOKENS))
    tl.start(); time.sleep(delay); ts.start()
    ts.join(); tl.join()
    if err:
        die("; ".join(err), 4)
    lr, sr = res["long"], res["short"]
    lr.update(kind="mixed-long", arm=arm, prompt="long", ids_complete=len(lr["ids"]) == lr["tokens"] > 0)
    sr.update(kind="mixed-short", arm=arm, level=0, run=0, req=0, prompt="p0")
    o.put(lr); o.put(sr)
    before = sr["end"] < lr["first"] if lr["first"] is not None else False
    line = (f"MIXED {arm} long prompt_tokens {lr['prompt_tokens']} first-token {fmt(lr['ttft_s'], '.1f')} s | short sent "
            f"+{delay:.0f} s: first-token {fmt(sr['ttft_s'], '.2f')} s dec {fmt(sr['dec_tok_s'], '.1f')} tok/s "
            f"{fmt(sr['ms_round'], '.2f')} ms/round tokens {sr['tokens']} finished-before-long-first-token "
            f"{'yes' if before else 'no'} ids {'ok' if sr['ids_complete'] else 'INCOMPLETE'}")
    o.put({"kind": "mixed", "arm": arm, "line": line, "short_before_long_first": before}); o.close()
    print(line, flush=True)
    return 0 if sr["ids_complete"] and lr["ids_complete"] else 1


# --- readings over receipts ---------------------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError) as e:
        die(f"cannot read {path}: {e}")
    return []


def receipts(d: Path, prefix: str) -> dict[tuple[str, int, int], list[dict]]:
    got = {}
    for p in sorted(d.glob(f"{prefix}-S*-L*-r*.jsonl")):
        m = NAME.match(p.name[len(prefix) + 1:])
        if m:
            got[(m["arm"], int(m["k"]), int(m["run"]))] = read_jsonl(p)
    return got


def first_diff(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def compare(arm: str, d: Path, prefix: str) -> tuple[str, str]:
    got = receipts(d, prefix)
    ref = got.get((arm, 1, 2))
    if not ref:
        return "INCONCLUSIVE", f"no L1 run 2 receipt for {arm}"
    by_prompt = {r["prompt"]: r["ids"] for r in ref if r.get("kind") == "req"}
    checks: list[tuple[str, list[int], list[int]]] = []
    for (a, k, run), recs in sorted(got.items()):
        if a != arm or (k, run) == (1, 2):
            continue
        for r in recs:
            if r.get("kind") == "req":
                checks.append((f"L{k} r{run} q{r['req']} {r['prompt']}", by_prompt.get(r["prompt"], []), r["ids"]))
    mixed = d / f"{prefix}-{arm}-mixed.jsonl"
    if mixed.exists():
        for r in read_jsonl(mixed):
            if r.get("kind") == "mixed-short":
                checks.append(("mixed short p0", by_prompt.get("p0", []), r["ids"]))
    if not checks:
        return "INCONCLUSIVE", f"nothing to compare on {arm}"
    bad = [(name, first_diff(x, y)) for name, x, y in checks if x != y or not x]
    if bad:
        return "DIFFER", f"{len(bad)} of {len(checks)} differ from L1 run 2: " + ", ".join(f"{n} at {i}" for n, i in bad[:6])
    return "MATCH", f"{len(checks)} of {len(checks)} equal to L1 run 2 of the same prompt"


def level_of(recs: list[dict]) -> dict | None:
    for r in recs:
        if r.get("kind") == "level":
            return r
    return None


def table(d: Path, prefix: str) -> list[str]:
    got = receipts(d, prefix)
    base = level_of(got.get(("S1", 1, 2), []))
    b = base["agg_tok_s"] if base and base["status"] == "OK" else None
    out = []
    present = [arm for arm in ARMS if any(key[0] == arm for key in got)]
    top = max((SLOTS_OF[arm] for arm in present), default=1)       # columns up to the largest arm measured
    cols = tuple(range(1, top + 1))
    for run, title in ((2, "TABLE W18 (run 2, the declared reading)"), (1, "TABLE W18 run 1 (noise view, not the reading)")):
        out.append(f"{title}: aggregate tok/s (ratio to S1 one stream run 2 = {fmt(b, '.1f')}), per-request decode tok/s, ms/round")
        out.append("| slots | " + " | ".join("1 request" if k == 1 else f"{k} together" for k in cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for arm in ARMS:
            if not any(key[0] == arm for key in got):
                continue
            cells = []
            for k in cols:
                if k > SLOTS_OF[arm]:
                    cells.append(""); continue
                s = level_of(got.get((arm, k, run), []))
                if s is None:
                    cells.append("not run"); continue
                ratio = s["agg_tok_s"] / b if b and s["agg_tok_s"] else None
                cell = (f"{fmt(s['agg_tok_s'], '.1f')} ({fmt(ratio, '.2f')}x), {fmt(s['dec_mean'], '.1f')}, "
                        f"{fmt(s['ms_mean'], '.2f') if s['rounds'] == 'OK' else 'ms INCONCLUSIVE'}")
                if s["status"] != "OK":
                    cell += " INCONCLUSIVE"
                cells.append(cell)
            out.append(f"| {SLOTS_OF[arm]} ({arm}) | " + " | ".join(cells) + " |")
    # the key row: the largest arm's levels, each step against the one before (the code predicts the gain at 2)
    karm = f"S{top}"
    row = [level_of(got.get((karm, k, 2), [])) for k in cols]
    if any(row):
        parts, prev = [], None
        for k, s in zip(cols, row):
            v = s["agg_tok_s"] if s and s["status"] == "OK" else None
            step = f" ({(v / prev - 1) * 100:+.1f} % on {k - 1})" if v and prev else ""
            parts.append(f"{k} together {fmt(v, '.1f')}{step}")
            prev = v
        out.append(f"KEY ROW W18 {karm} ({top} slot(s), run 2, aggregate tok/s): " + ", ".join(parts))
    out.append("NOISE W18 (aggregate tok/s run 1 -> run 2):")
    for arm in ARMS:
        for k in (1, 2, 3, 4):
            s1, s2 = level_of(got.get((arm, k, 1), [])), level_of(got.get((arm, k, 2), []))
            if s1 and s2 and s1["agg_tok_s"] and s2["agg_tok_s"]:
                out.append(f"  {arm} L{k}: {s1['agg_tok_s']:.1f} -> {s2['agg_tok_s']:.1f} "
                           f"({(s2['agg_tok_s'] / s1['agg_tok_s'] - 1) * 100:+.1f} %)")
    return out


# --- self-test ----------------------------------------------------------------------------------------------------
def fake(t0: float, step: float, n_chunks: int, per: int, first_at: float) -> dict:
    """A stream of n_chunks chunks of `per` ids, the first at t0 + first_at, then one every `step` s."""
    marks = [(t0 + first_at + i * step, per) for i in range(n_chunks)]
    ids = list(range(n_chunks * per))
    return {"t0": t0, "end": marks[-1][0] + 0.001, "ids": ids, "marks": marks,
            "usage": {"completion_tokens": len(ids), "prompt_tokens": 20}, "finish": "length"}


def self_test() -> int:
    fails = []

    def check(name: str, got, want, tol: float = 1e-9) -> None:
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        if not ok:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    # one request: 256 chunks of 4 ids = 1,024 tokens, first chunk 0.5 s after the start, then one every 0.02 s
    r = record(fake(100.0, 0.02, 256, 4, 0.5), 1024)
    check("tokens", r["tokens"], 1024)
    check("complete", r["ids_complete"], True)
    check("ttft", r["ttft_s"], 0.5, 1e-9)
    check("dec_s", r["dec_s"], 255 * 0.02, 1e-9)                   # 255 gaps
    check("dec_tok_s", r["dec_tok_s"], 1020 / (255 * 0.02), 1e-6)  # tokens after the first chunk
    check("ms_round", r["ms_round"], 20.0, 1e-6)
    # level 1: three of those in a row, wall 0.5 + 5.1 + 0.001 each
    recs = [record(fake(100.0 + 10 * i, 0.02, 256, 4, 0.5), 1024) for i in range(3)]
    s = summarize(recs, 1, 768.0)
    check("L1 aggregate", s["agg_tok_s"], 3072 / (3 * (0.5 + 5.1 + 0.001)), 1e-6)
    check("L1 rounds OK", s["rounds"], "OK")
    # level 2 together: two streams starting 0.01 s apart, each at 0.04 s per chunk; aggregate over the joint span
    recs = [record(fake(200.0, 0.04, 256, 4, 0.5), 1024), record(fake(200.01, 0.04, 256, 4, 0.5), 1024)]
    s = summarize(recs, 2, 512.0)
    span = (200.01 + 0.5 + 255 * 0.04 + 0.001) - 200.0
    check("L2 aggregate", s["agg_tok_s"], 2048 / span, 1e-6)
    check("L2 per-request", s["dec_mean"], 1020 / (255 * 0.04), 1e-6)
    check("L2 skew", round(s["start_skew_ms"], 6), 10.0, 1e-6)
    check("L2 ms", s["ms_mean"], 40.0, 1e-6)
    # rounds check on 512 chunks: 522 server drafts is within 2 % (10 <= 10.44), 501 is not (11 > 10.02)
    check("rounds within", summarize(recs, 2, 522.0)["rounds"], "OK")
    check("rounds outside", summarize(recs, 2, 501.0)["rounds"], "INCONCLUSIVE")
    check("rounds outside ms", summarize(recs, 2, 501.0)["ms_mean"], None)
    # an incomplete request makes the level INCONCLUSIVE
    short = record(fake(300.0, 0.02, 200, 4, 0.5), 1024)
    check("incomplete", short["ids_complete"], False)
    check("level incomplete", summarize([short, recs[0]], 2, 456.0)["status"], "INCONCLUSIVE")
    check("first_diff equal", first_diff([1, 2, 3], [1, 2, 3]), -1)
    check("first_diff at", first_diff([1, 2, 3], [1, 9, 3]), 1)
    check("first_diff length", first_diff([1, 2], [1, 2, 3]), 2)
    check("name", bool(NAME.match("S4-L3-r2.jsonl")), True)
    check("name 5", bool(NAME.match("S5-L1-r1.jsonl")), False)
    for f in fails:
        print("FAIL", f)
    print("SELF-TEST PASS" if not fails else f"SELF-TEST FAIL ({len(fails)})")
    return 0 if not fails else 1


def opt(a: list[str], name: str, default: str | None = None) -> tuple[list[str], str | None]:
    if name in a:
        i = a.index(name)
        if i + 1 >= len(a):
            die(f"{name} needs a value")
        return a[:i] + a[i + 2:], a[i + 1]
    return a, default


def main(a: list[str]) -> int:
    a, ab = opt(a, "--ab")
    a, src = opt(a, "--src-dir", "/usr/lib/python3.12")
    if a[:1] == ["seq"] and len(a) == 5 and ab and a[2] in ARMS and a[3] in ("1", "2"):
        return cmd_seq(a[1], a[2], int(a[3]), a[4], ab)
    if a[:1] == ["level"] and len(a) == 6 and ab and a[2] in ("2", "3", "4") and a[3] in ARMS and a[4] in ("1", "2"):
        if int(a[2]) > SLOTS_OF[a[3]]:
            die(f"level {a[2]} on {a[3]} has more requests than slots")
        return cmd_level(a[1], int(a[2]), a[3], int(a[4]), a[5], ab)
    if a[:1] == ["mixed"] and len(a) == 6 and ab and a[2].isdecimal() and a[4] in ARMS:
        return cmd_mixed(a[1], int(a[2]), float(a[3]), a[4], a[5], ab, src)
    if a[:1] == ["compare"] and len(a) == 4 and a[1] in ARMS:
        st, why = compare(a[1], Path(a[2]), a[3])
        print(f"TOKENS {a[1]} {st}: {why}")
        return {"MATCH": 0, "DIFFER": 1}.get(st, 3)
    if a[:1] == ["table"] and len(a) == 3:
        print("\n".join(table(Path(a[1]), a[2])))
        return 0
    if a == ["self-test"]:
        return self_test()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:   # a client error or any other failure: loud, its own exit code (1 means incomplete only)
        print(f"w15_tput: CLIENT ERROR {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
        sys.exit(4)
