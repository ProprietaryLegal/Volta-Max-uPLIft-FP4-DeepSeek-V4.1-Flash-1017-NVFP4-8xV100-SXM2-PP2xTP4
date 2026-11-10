#!/usr/bin/python3 -I
"""bench_volta_run.py URL [--max-prompt 256K] [--kinds code,structured,legal,prose,chat]
                          [--parts long-off,short,think-short,long-on] [--out-dir DIR] [--allow-prompt-drift]
                          [--kitdata DIR] [--plan [--prefill-rates 1300,2000] [--decode-rates 50,65]]

Runs Volta sPLIt's README speed tables against one OpenAI-compatible vLLM server, one request at a time, with the
method of the kit that produced them (Volta sPLIt's own bench scripts, not published):
  long-off, long-on  its context sweep (ctxsweep5) with its measuring module (measure.py)
                     (tables "Long prompts: 1K to 256K", "Thinking on", "Prompt reading by text type")
  short, think-short its head-to-head runner (run_h2h), suites speed and think
                     (table "Short prompts, by text type" and its "thinking on, 4 prompts, token-weighted" row)

Order (cut from the end): long-off by rising length (per length: rep 0 of every kind, then rep 1), then short (per kind
greedy x3 then T=1 x3), then think-short (4 prompts), then long-on by rising length.

Long cell (ctxsweep5): one user turn rendered by the server's own chat template (POST /tokenize with messages and
chat_template_kwargs, split at a mark via /detokenize), content = a real document of the kind cut to an exact token
count + "\\n\\n---\\n" + the kit's instruction for the kind, so the prompt is exactly N tokens; sent as token IDs to
/v1/completions: greedy, seed 42, ignore_eos, 1,024 tokens, return_token_ids, a random cache_salt. Thinking off =
{"thinking": false}; thinking on = {"thinking": true, "reasoning_effort": 50}.
Short cell (run_h2h speed): the kit's class instruction as one chat message, thinking off, max_tokens 1,024, natural
stop, T=0, or T=1 with top_p 1 and seed 1000 + rep; 3 reps. think-short (run_h2h think): the kit's 4 prompts, thinking
on at effort 50, T=1, top_p 1, the prompt's seed, max_tokens 2,048.

Measures, timed by this program: first token = first streamed token - request start; prompt reading = prompt tokens / first
token; decode = (tokens - tokens in the first event) / (last token - first token) for long cells (measure.py; equals
(tokens - 1) / span when the first event carries one token, which every receipt records), (tokens - 1) / (last text
chunk - first text chunk) for chat cells (run_h2h). Tokens per round = 1 + accepted / drafts from /metrics deltas,
reconciled with the usage count (INCONCLUSIVE on any contradiction). Greedy identity: sha256 of the token-id list
(long) or of reasoning + NUL + content (chat), computed in memory.

Receipts (DIR, default logs/benchvolta/UTC-TIME under the LANE directory): run.json (settings, server model, kit data
sha256, document and template token counts, status) and cells.jsonl (one numeric record per cell, fsync'd). No
generated text is ever written; it exists only in memory long enough to be hashed.

Fail fast, no retries. Exit 0 done; 3 server error (HTTP status, refused connection, timeout, stream error) - the run
stops at once and run.json says why; 4 INCONCLUSIVE (protocol or count contradiction, server not idle, effort
preflight); 5 prompt drift (document or template token counts differ from Volta's 2026-11-05 sweep) unless
--allow-prompt-drift; 1 usage or kit data error.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

KITDATA = Path(os.environ["LANE"] + "/stage/bench/kitdata")
LOGS = Path(os.environ["LANE"] + "/logs/benchvolta")
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
KINDS = ("code", "structured", "legal", "prose", "chat")
PARTS = ("long-off", "short", "think-short", "long-on")
CHARS = 1_600_000
MARK = "@@CTXSWEEP-CONTENT@@"
THINK_OFF = {"thinking": False}
THINK_ON = {"thinking": True, "reasoning_effort": 50}
EFFORT_MARK = "Reasoning Effort: 50 (range 1-100"
LONG_OUT, SHORT_OUT, THINK_OUT = 1024, 1024, 2048
LONG_REPS = {"long-off": 2, "long-on": 1}
SHORT_REPS = 3
SPEC_K = 5
TIMEOUT_S = 1800
OVERHEAD_S = 0.5          # planning only: per-request program and scheduling overhead (ESTIMATED)


# ------------------------------------------------------------------------------------------------ stop reasons
@dataclass(frozen=True)
class HttpStatus:
    status: int
    path: str


@dataclass(frozen=True)
class Unreachable:
    path: str
    error: str


@dataclass(frozen=True)
class TimedOut:
    path: str
    after_s: float


@dataclass(frozen=True)
class StreamError:
    path: str


@dataclass(frozen=True)
class Inconclusive:
    what: str


@dataclass(frozen=True)
class PromptDrift:
    what: str


EXIT = {HttpStatus: 3, Unreachable: 3, TimedOut: 3, StreamError: 3, Inconclusive: 4, PromptDrift: 5}


class Stop(Exception):
    def __init__(self, reason):
        super().__init__(repr(reason))
        self.reason = reason

    def record(self) -> dict:
        r = self.reason
        out = {"kind": type(r).__name__}
        if isinstance(r, HttpStatus):
            out.update(http_status=r.status, path=r.path)
        elif isinstance(r, Unreachable):
            out.update(path=r.path, error=r.error)
        elif isinstance(r, TimedOut):
            out.update(path=r.path, after_s=round(r.after_s, 1))
        elif isinstance(r, StreamError):
            out.update(path=r.path)
        else:
            out.update(what=r.what)
        return out


# ------------------------------------------------------------------------------------------------ cells
@dataclass(frozen=True)
class LongCell:
    part: str          # long-off | long-on
    kind: str
    length: int
    rep: int


@dataclass(frozen=True)
class ShortCell:
    kind: str
    temperature: int   # 0 | 1
    rep: int


@dataclass(frozen=True)
class ThinkCell:
    index: int
    pid: str
    seed: int


def parse_len(s: str) -> int:
    s = s.strip().upper()
    v = int(s[:-1]) * 1024 if s.endswith("K") else int(s)
    if v not in LENGTHS:
        raise SystemExit(f"--max-prompt {s}: must be one of {', '.join(f'{x // 1024}K' for x in LENGTHS)}")
    return v


def plan_cells(max_prompt: int, kinds: tuple[str, ...], parts: tuple[str, ...], speed: dict) -> list:
    lengths = [n for n in LENGTHS if n <= max_prompt]
    cells: list = []
    if "long-off" in parts:
        cells += [LongCell("long-off", k, n, r) for n in lengths for r in range(LONG_REPS["long-off"]) for k in kinds]
    if "short" in parts:
        order = [k for k in speed["classes"] if k in kinds]          # the kit's class order
        cells += [ShortCell(k, t, r) for k in order for t in (0, 1) for r in range(SHORT_REPS)]
    if "think-short" in parts:
        cells += [ThinkCell(i, p["id"], int(p["seed"])) for i, p in enumerate(speed["think"])]
    if "long-on" in parts:
        cells += [LongCell("long-on", k, n, r) for n in lengths for r in range(LONG_REPS["long-on"]) for k in kinds]
    return cells


# ------------------------------------------------------------------------------------------------ server
class Server:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]

    def open(self, path: str, body: dict | None, timeout: float):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        t0 = time.perf_counter()
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise Stop(HttpStatus(e.code, path)) from None
        except (TimeoutError, socket.timeout):
            raise Stop(TimedOut(path, time.perf_counter() - t0)) from None
        except urllib.error.URLError as e:
            r = getattr(e, "reason", e)
            if isinstance(r, (TimeoutError, socket.timeout)):
                raise Stop(TimedOut(path, time.perf_counter() - t0)) from None
            raise Stop(Unreachable(path, type(r).__name__)) from None
        except (ConnectionError, http.client.HTTPException) as e:
            raise Stop(Unreachable(path, type(e).__name__)) from None

    def get_json(self, path: str):
        with self.open(path, None, 60) as r:
            return json.load(r)

    def post_json(self, path: str, body: dict, timeout: float = 600):
        with self.open(path, body, timeout) as r:
            return json.load(r)

    def metrics(self) -> dict[str, float]:
        with self.open("/metrics", None, 30) as r:
            lines = r.read().decode("utf-8", "replace").splitlines()
        out = {}
        for name in ("spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total",
                     "num_requests_running", "num_requests_waiting"):
            vals = [float(x.split()[-1]) for x in lines if x.startswith(("vllm:" + name + "{", "vllm:" + name + " "))]
            if not vals or any(not math.isfinite(v) for v in vals):
                raise Stop(Inconclusive("missing or nonfinite /metrics series vllm:" + name))
            out[name] = sum(vals)
        return out

    def idle(self) -> dict[str, float]:
        m = self.metrics()
        if m["num_requests_running"] or m["num_requests_waiting"]:
            raise Stop(Inconclusive("server has other requests running or waiting"))
        return m

    def tokenize_text(self, model: str, text: str) -> list[int]:
        ids = self.post_json("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}).get("tokens")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise Stop(Inconclusive("/tokenize returned no valid tokens"))
        return ids

    def render(self, model: str, content: str, kwargs: dict) -> tuple[list[int], str]:
        out = self.post_json("/tokenize", {"model": model, "messages": [{"role": "user", "content": content}],
                                           "add_generation_prompt": True, "chat_template_kwargs": kwargs})
        ids = out.get("tokens")
        if not isinstance(ids, list) or not ids:
            raise Stop(Inconclusive("chat /tokenize returned no tokens"))
        text = self.post_json("/detokenize", {"model": model, "tokens": ids}).get("prompt")
        if not isinstance(text, str):
            raise Stop(Inconclusive("/detokenize returned no prompt text"))
        return ids, text


@dataclass(frozen=True)
class Counters:
    rounds: int
    accepted: int


def reconcile(before: dict, after: dict, tokens: int) -> Counters:
    d = []
    for name in ("spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total"):
        v = after[name] - before[name]
        if not math.isfinite(v) or v < 0 or not float(v).is_integer():
            raise Stop(Inconclusive("invalid server counter delta " + name))
        d.append(int(v))
    c = Counters(*d)
    if c.rounds <= 0 or c.accepted > SPEC_K * c.rounds:
        raise Stop(Inconclusive(f"contradictory spec-decode deltas: rounds {c.rounds} accepted {c.accepted}"))
    gen = 1 + c.rounds + c.accepted
    if not gen - SPEC_K <= tokens <= gen:
        raise Stop(Inconclusive(f"usage {tokens} tokens contradicts counters (1 + {c.rounds} + {c.accepted})"))
    return c


# ------------------------------------------------------------------------------------------------ streams
def sse(resp, path: str):
    """Parsed SSE objects; returns normally only after [DONE]."""
    for raw in resp:
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if data == b"[DONE]":
            return
        obj = json.loads(data)
        if obj.get("error"):
            raise Stop(StreamError(path))
        yield obj
    raise Stop(Inconclusive(f"{path}: stream ended without [DONE]"))


def stream_long(srv: Server, model: str, prompt: tuple[int, ...], out_tokens: int) -> dict:
    """measure.stream: token-ID prompt to /v1/completions, greedy, ignore_eos; timing per event with token IDs."""
    body = {"model": model, "temperature": 0, "seed": 42, "max_tokens": out_tokens, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True,
            "cache_salt": uuid.uuid4().hex[:12], "prompt": list(prompt)}
    t0 = time.perf_counter()
    events: list[tuple[float, int]] = []
    ids: list[int] = []
    usage, finish = None, None
    with srv.open("/v1/completions", body, TIMEOUT_S) as resp:
        try:
            for obj in sse(resp, "/v1/completions"):
                now = time.perf_counter()
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    t = ch.get("token_ids") or []
                    if t:
                        if any(type(i) is not int or i < 0 for i in t):
                            raise Stop(Inconclusive("malformed token id in stream"))
                        events.append((now, len(t)))
                        ids.extend(t)
        except (TimeoutError, socket.timeout):
            raise Stop(TimedOut("/v1/completions", time.perf_counter() - t0)) from None
        except (ConnectionError, http.client.HTTPException):
            raise Stop(StreamError("/v1/completions")) from None
        except json.JSONDecodeError:
            raise Stop(Inconclusive("/v1/completions: malformed SSE JSON")) from None
    t_end = time.perf_counter()
    n = (usage or {}).get("completion_tokens")
    if (not usage or finish is None or n != out_tokens or len(ids) != out_tokens
            or usage.get("prompt_tokens") != len(prompt)):
        raise Stop(Inconclusive(f"incomplete stream: finish {finish} usage {usage} ids {len(ids)} "
                                f"want {out_tokens} out, {len(prompt)} prompt"))
    span = events[-1][0] - events[0][0]
    if span <= 0:
        raise Stop(Inconclusive("no measured decode span"))
    ttft = events[0][0] - t0
    return {"prompt_tokens": len(prompt), "completion_tokens": n, "first_event_tokens": events[0][1],
            "events": len(events), "finish": finish, "ttft_s": ttft, "decode_s": span, "wall_s": t_end - t0,
            "decode_tok_s": (n - events[0][1]) / span, "prompt_read_tok_s": len(prompt) / ttft,
            "ids_sha": hashlib.sha256(str(ids).encode()).hexdigest()[:16]}


def stream_chat(srv: Server, body: dict) -> dict:
    """run_h2h Server.stream_chat + timing(): timing on chunks that carry content or reasoning text."""
    path = "/v1/chat/completions"
    t0 = time.perf_counter()
    first = last = None
    content: list[str] = []
    reasoning: list[str] = []
    usage, finish, chunks = None, None, 0
    with srv.open(path, body, TIMEOUT_S) as resp:
        try:
            for obj in sse(resp, path):
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    now = time.perf_counter()
                    d = ch.get("delta") or {}
                    rs = d.get("reasoning_content") or d.get("reasoning") or ""
                    ct = d.get("content") or ""
                    if rs or ct:
                        first = first or now
                        last = now
                        chunks += 1
                        reasoning.append(rs)
                        content.append(ct)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        except (TimeoutError, socket.timeout):
            raise Stop(TimedOut(path, time.perf_counter() - t0)) from None
        except (ConnectionError, http.client.HTTPException):
            raise Stop(StreamError(path)) from None
        except json.JSONDecodeError:
            raise Stop(Inconclusive(path + ": malformed SSE JSON")) from None
    t_end = time.perf_counter()
    if not usage or "completion_tokens" not in usage or "prompt_tokens" not in usage:
        raise Stop(Inconclusive("no usage chunk"))
    n = int(usage["completion_tokens"])
    if n < 2 or first is None or last is None or last <= first:
        raise Stop(Inconclusive(f"{n} completion tokens and no measurable decode window"))
    if n > body["max_tokens"] or finish not in ("stop", "length"):
        raise Stop(Inconclusive(f"finish {finish!r} with {n} tokens (max {body['max_tokens']})"))
    text_sha = hashlib.sha256(("".join(reasoning) + "\x00" + "".join(content)).encode()).hexdigest()[:16]
    rchars, cchars = sum(map(len, reasoning)), sum(map(len, content))
    del content, reasoning
    det = usage.get("completion_tokens_details") or {}
    pt = int(usage["prompt_tokens"])
    return {"prompt_tokens": pt, "completion_tokens": n, "reasoning_tokens": det.get("reasoning_tokens"),
            "chunks": chunks, "finish": finish, "ttft_s": first - t0, "decode_s": last - first, "wall_s": t_end - t0,
            "decode_tok_s": (n - 1) / (last - first), "prompt_read_tok_s": pt / (first - t0),
            "reasoning_chars": rchars, "content_chars": cchars, "text_sha": text_sha}


# ------------------------------------------------------------------------------------------------ documents
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_kitdata(kd: Path) -> dict[str, str]:
    sums = {}
    for line in (kd / "SHA256SUMS").read_text().splitlines():
        h, name = line.split(None, 1)
        sums[name.strip()] = h
    for name, h in sums.items():
        if sha256_file(kd / name) != h:
            raise SystemExit(f"kit data {name}: sha256 differs from SHA256SUMS")
    return sums


def gutenberg(p: Path) -> str:
    raw = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    s, e = raw.find("*** START OF"), raw.find("*** END OF")
    if s < 0 or e < 0:
        raise SystemExit(f"{p.name}: Gutenberg markers missing")
    return raw[raw.index("\n", s) + 1: e][:CHARS]


def struct_text(jsonl: Path) -> str:
    out, n = [], 0
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            json.loads(line)
            out.append(line)
            n += len(line)
            if n >= CHARS:
                return "".join(out)
    raise SystemExit("structured source too short")


def legal_text(corpus: Path, jsonl: Path) -> str:
    parts = [corpus.read_text(encoding="utf-8")]
    n = len(parts[0])
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)["prompt"]
            parts.append("\n\n" + p)
            n += len(p) + 2
            if n >= CHARS:
                return "".join(parts)
    raise SystemExit("legal source too short")


def document(kd: Path, kind: str) -> str:
    jsonl = kd / "legalbench_items.jsonl"
    if kind == "code":
        return (kd / "code_vllm_doc.txt").read_text(encoding="utf-8")
    if kind == "structured":
        return struct_text(jsonl)
    if kind == "legal":
        return legal_text(kd / "longctx_corpus.txt", jsonl)
    if kind == "prose":
        return gutenberg(kd / "pg2600.txt")
    if kind == "chat":
        return gutenberg(kd / "pg135.txt")
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ planning
def estimate(cells: list, prefill: float, decode: float) -> dict[str, float]:
    """Minutes per part from assumed rates (ESTIMATED, not measured)."""
    m: dict[str, float] = {}
    for c in cells:
        if isinstance(c, LongCell):
            s, part = c.length / prefill + (LONG_OUT - 1) / decode + OVERHEAD_S, c.part
        elif isinstance(c, ShortCell):
            s, part = SHORT_OUT / decode + OVERHEAD_S, "short"
        else:
            s, part = THINK_OUT / decode + OVERHEAD_S, "think-short"
        m[part] = m.get(part, 0.0) + s / 60
    return m


def label(c) -> str:
    if isinstance(c, LongCell):
        return f"{c.part:<11s} {c.length // 1024:>4d}K {c.kind:<10s} r{c.rep}"
    if isinstance(c, ShortCell):
        return f"{'short':<11s}  T={c.temperature} {c.kind:<10s} r{c.rep}"
    return f"{'think-short':<11s}       {c.pid:<10s}"


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("url")
    ap.add_argument("--max-prompt", default="256K")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--kitdata", type=Path, default=KITDATA)
    ap.add_argument("--allow-prompt-drift", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the cells and a time estimate; send nothing")
    ap.add_argument("--prefill-rates", default="1300,2000")
    ap.add_argument("--decode-rates", default="50,65")
    a = ap.parse_args()
    kinds = tuple(k.strip() for k in a.kinds.split(",") if k.strip())
    parts = tuple(p.strip() for p in a.parts.split(",") if p.strip())
    if not kinds or any(k not in KINDS for k in kinds) or len(set(kinds)) != len(kinds):
        raise SystemExit(f"--kinds: comma list from {KINDS}")
    if not parts or any(p not in PARTS for p in parts):
        raise SystemExit(f"--parts: comma list from {PARTS}")
    kinds = tuple(k for k in KINDS if k in kinds)
    max_prompt = parse_len(a.max_prompt)
    kd = a.kitdata
    sums = verify_kitdata(kd)
    speed = json.loads((kd / "speed_prompts.json").read_text())
    volta = json.loads((kd / "volta_ctxsweep5_sources.json").read_text())
    cells = plan_cells(max_prompt, kinds, parts, speed)

    if a.plan:
        for c in cells:
            print(label(c))
        print(f"{len(cells)} requests (+1 warm-up)")
        for pr in (float(x) for x in a.prefill_rates.split(",")):
            for dr in (float(x) for x in a.decode_rates.split(",")):
                e = estimate(cells, pr, dr)
                print(f"ESTIMATE prefill {pr:.0f} tok/s decode {dr:.0f} tok/s: "
                      + ", ".join(f"{p} {m:.1f} min" for p, m in e.items()) + f"; total {sum(e.values()):.1f} min")
        return 0

    out = a.out_dir or LOGS / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir(parents=True, exist_ok=False)
    run_path, cells_path = out / "run.json", out / "cells.jsonl"
    run: dict = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "url": a.url,
                 "max_prompt": max_prompt, "kinds": list(kinds), "parts": list(parts), "cells_planned": len(cells),
                 "kitdata": str(kd), "kitdata_sha256": sums, "status": "RUNNING",
                 "method": {"long_out": LONG_OUT, "short_out": SHORT_OUT, "think_out": THINK_OUT,
                            "long_reps": LONG_REPS, "short_reps": SHORT_REPS, "think_off": THINK_OFF,
                            "think_on": THINK_ON, "spec_k": SPEC_K}}

    def save_run():
        tmp = run_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(run, indent=2) + "\n")
        tmp.replace(run_path)

    save_run()
    srv = Server(a.url)
    done = 0
    try:
        models = srv.get_json("/v1/models").get("data") or []
        if len(models) != 1:
            raise Stop(Inconclusive(f"/v1/models lists {len(models)} models, need exactly one"))
        model = models[0]["id"]
        run["server"] = {"model": model, "max_model_len": models[0].get("max_model_len")}
        srv.idle()
        if {"think-short", "long-on"} & set(parts):        # chatkw.preflight
            _, on = srv.render(model, "preflight", THINK_ON)
            _, off = srv.render(model, "preflight", THINK_OFF)
            if EFFORT_MARK not in on or EFFORT_MARK in off or on == off:
                raise Stop(Inconclusive("effort preflight: thinking-on render lacks 'Reasoning Effort: 50' "
                                        "or thinking-off render carries it"))
            run["effort_preflight"] = "PASS"

        prompts: dict[LongCell, tuple[int, ...]] = {}
        long_parts = [p for p in ("long-off", "long-on") if p in parts]
        if long_parts:
            modes = {"long-off": THINK_OFF, "long-on": THINK_ON}
            volta_mode = {"long-off": "off", "long-on": "reasoning"}
            halves = {}
            for p in long_parts:
                _, text = srv.render(model, MARK, modes[p])
                if text.count(MARK) != 1:
                    raise Stop(Inconclusive("rendered template does not contain the content mark exactly once"))
                pre, post = text.split(MARK)
                halves[p] = (srv.tokenize_text(model, pre) if pre else [], srv.tokenize_text(model, post) if post else [])
            body = {}
            docs_sha = {}
            for k in kinds:
                doc = document(kd, k)
                docs_sha[k] = hashlib.sha256(doc.encode()).hexdigest()
                body[k] = srv.tokenize_text(model, doc)
            tail = {k: srv.tokenize_text(model, "\n\n---\n" + speed["classes"][k]) for k in kinds}
            drift = []
            for k in kinds:
                if len(body[k]) != volta["document_tokens"][k]:
                    drift.append(f"{k} document {len(body[k])} tokens, Volta {volta['document_tokens'][k]}")
            for p in long_parts:
                got = [len(halves[p][0]), len(halves[p][1])]
                if got != volta["template_tokens"][volta_mode[p]]:
                    drift.append(f"{p} template {got}, Volta {volta['template_tokens'][volta_mode[p]]}")
            run["prompts"] = {"document_tokens": {k: len(v) for k, v in body.items()},
                              "document_sha256": docs_sha,
                              "template_tokens": {p: [len(x) for x in halves[p]] for p in long_parts},
                              "tail_tokens": {k: len(v) for k, v in tail.items()},
                              "volta_document_tokens": volta["document_tokens"],
                              "volta_template_tokens": volta["template_tokens"], "drift": drift}
            if drift:
                print("PROMPT DRIFT from Volta sPLIt's 2026-11-05 sweep: " + "; ".join(drift), flush=True)
                if not a.allow_prompt_drift:
                    raise Stop(PromptDrift("; ".join(drift)))
                run["prompts"]["drift_allowed"] = True
            for c in cells:
                if isinstance(c, LongCell):
                    pre, post = halves[c.part]
                    k = c.length - (len(pre) + len(tail[c.kind]) + len(post))
                    if k < 1 or k > len(body[c.kind]):
                        raise SystemExit(f"{c.kind}: need {k} document tokens, have {len(body[c.kind])}")
                    prompts[c] = tuple(pre + body[c.kind][:k] + tail[c.kind] + post)
            del body
        save_run()

        # warm-up (ctxsweep5: one 256-token synthetic prompt, 32 tokens; not recorded)
        nonce = uuid.uuid4().hex[:12]
        warm = srv.tokenize_text(model, "\n".join(f"{nonce} public observation {i}: blue river stone tree wind."
                                                  for i in range(200)))
        if len(warm) < 256:
            raise Stop(Inconclusive("warm-up text shorter than 256 tokens"))
        before = srv.idle()
        w = stream_long(srv, model, tuple(warm[:256]), 32)
        reconcile(before, srv.metrics(), w["completion_tokens"])
        run["warm"] = "done"
        save_run()
        print(f"START {len(cells)} cells -> {out}", flush=True)

        for c in cells:
            before = srv.idle()
            if isinstance(c, LongCell):
                m = stream_long(srv, model, prompts[c], LONG_OUT)
                rec = {"part": c.part, "kind": c.kind, "length": c.length, "rep": c.rep, "temperature": 0}
            elif isinstance(c, ShortCell):
                samp = {"temperature": 0.0} if c.temperature == 0 else {"temperature": 1.0, "top_p": 1.0,
                                                                          "seed": 1000 + c.rep}
                m = stream_chat(srv, {"model": model, "messages": [{"role": "user",
                                                                     "content": speed["classes"][c.kind]}],
                                      "max_tokens": SHORT_OUT, **samp, "chat_template_kwargs": dict(THINK_OFF),
                                      "stream": True, "stream_options": {"include_usage": True}})
                rec = {"part": "short", "kind": c.kind, "length": None, "rep": c.rep, "temperature": c.temperature}
            else:
                p = speed["think"][c.index]
                m = stream_chat(srv, {"model": model, "messages": [{"role": "user", "content": p["prompt"]}],
                                      "max_tokens": THINK_OUT, "temperature": 1.0, "top_p": 1.0, "seed": c.seed,
                                      "chat_template_kwargs": dict(THINK_ON), "stream": True,
                                      "stream_options": {"include_usage": True}})
                rec = {"part": "think-short", "kind": None, "length": None, "rep": c.index, "temperature": 1,
                       "pid": c.pid}
            cnt = reconcile(before, srv.metrics(), m["completion_tokens"])
            rec.update(m, rounds=cnt.rounds, accepted=cnt.accepted, tok_per_round=1 + cnt.accepted / cnt.rounds,
                       utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            if isinstance(c, LongCell):
                rec["prompt_sha"] = hashlib.sha256(str(prompts[c]).encode()).hexdigest()[:16]
            with cells_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            done += 1
            h = rec.get("ids_sha") or rec.get("text_sha")
            print(f"{done:>3d}/{len(cells)} {label(c)} ttft {m['ttft_s']:7.2f} s  read {m['prompt_read_tok_s']:7.0f} "
                  f"tok/s  decode {m['decode_tok_s']:6.1f} tok/s  tok/round {rec['tok_per_round']:.2f}  "
                  f"n {m['completion_tokens']}  {h}", flush=True)
    except Stop as e:
        run.update(status="STOPPED", stop=e.record(), cells_done=done,
                   ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save_run()
        print(f"STOPPED after {done} cells: {json.dumps(e.record())}", flush=True)
        return EXIT[type(e.reason)]
    run.update(status="DONE", cells_done=done, ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save_run()
    print(f"DONE {done} cells -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
