#!/usr/bin/python3 -I
"""termination_test.py: does the tool loop's final long-thinking request end, or run to the token limit?

  termination_test.py run URL OUT.jsonl [--chats N] [--start I] [--followups K] [--chars C] [--tail T]
                                        [--label TEXT] [--model NAME]
  termination_test.py summary OUT.jsonl    the summary line of a receipt (several runs of one arm, appended)
  termination_test.py plan [--chats N] [--start I] [--followups K] [--chars C] [--tail T] [--label TEXT]
                                           the chats that would be sent: sizes and sha256 only; no request
  termination_test.py self-test            CPU checks of the measures and the interval (hand-computed values)

Campaign card 8.0. One chat is the chat of run/harness_loop.py: Anthropic Messages API format (POST /v1/messages, streamed),
the same system prompt, tool and task text, one long first user message of public text (Python standard-library
source), a tool loop in which every assistant message is sent back with its blocks as produced, then the final
request, in which the model writes its answer with thinking on. Fixed: max_tokens 32000, temperature 0, thinking
on. One request at a time; before every request the server must be idle.

A chat sends K + 2 requests: the first read (request 0), K tool-call follow-ups, the final request. K = 6 with the
default sizes is harness_loop.py's "830000 8 ... 1 12000" (the 216K-token loop); "--chars 120000 --tail 8000
--followups 2" is its 31K-token chat. Chat I takes its text from offset I * (C + T * (K + 2)) of the source, so a
chat index names the same chat on every server, and chat 0 without --label is byte for byte the chat
harness_loop.py sends. --label TEXT puts the line "Run TEXT chat I." first in every chat: use it when the server
may already have seen these chats since it booted (a server that keeps earlier prompts can match an older copy of
an identical chat), and then use the same label on every arm that is to be compared.

Copied from run/harness_loop.py, not imported (that file runs its loop when imported): the source-text files and
their order, SYSTEM, TOOLS, the task sentence, the first-message header, the tool-result rule, the stream reader,
the counters and the kept-read rule (kept tokens >= previous prompt length - 124). self-test compares the copies
with that file's source text.

Per chat one JSON line in OUT (numbers only; no text is written or printed anywhere): the final request's context
tokens, kept tokens, first-token time, output tokens, stop reason, decode tok/s, dec-ms/round and tokens per round
(null when the server has no draft counter), whether thinking closed, the measures below, and the same numbers for
the earlier requests under "steps".

Measures of the final answer (thinking and text blocks in order), to tell a repetition loop from long reasoning.
The /v1/messages stream carries no token ids, so the unit is a text unit held in memory only: a run of word
characters or one punctuation character ("units": "text").
  distinct20                share of distinct 20-unit windows over the whole answer (1.0: no window recurs)
  distinct20_last_quarter   the same over the last quarter of the units (the last len // 4)
  longest_repeat            length in units of the longest span that occurs at least twice (the occurrences may
                            overlap; a pure loop of period p over n units gives n - p)
  thinking_closed           true when a text or tool_use block started after the last thinking block; null when
                            there is no thinking block
A share is null when the answer (or its last quarter) is shorter than 20 units.

Outcome of a chat: ENDED (final request stopped end_turn), LIMIT (stopped max_tokens), INCONCLUSIVE (an earlier
request did not call the tool, or the final request stopped for another reason; the reason is recorded).
Last line: SUMMARY chats N: ended normally, ran to the limit, undecided, and the Wilson 95 % interval of the limit
share over the decided chats. Exit 0 when every chat decided, 3 when one did not or the server was busy
(INCONCLUSIVE). Any failed request stops the run with its error; nothing is retried.
"""
from __future__ import annotations

import os
import argparse
import ast
import datetime
import glob
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HARNESS = Path(os.environ["LANE"] + "/run/harness_loop.py")  # read as text by self-test only
SOURCE_GLOBS = ("/usr/lib/python3.12/*.py", "/usr/lib/python3.12/*/*.py")
HEADER = "Here is Python standard-library source code.\n\n"
SYSTEM = ("You are a careful code reviewer working through a large body of Python standard-library source. "
          "You work in steps and use the tool you are given to fetch each further section. Never fetch two "
          "sections in one step.")
TOOLS = [{"name": "read_section", "description": "Return the text of one further section of the source by number.",
          "input_schema": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}]
EXTRA_USE = "One section per step."
REACH = 124  # the engine's reach (VLLM_V41_RING_REACH), as harness_loop.py
MAX_TOKENS = 32000
WINDOW = 20
Z95 = 1.959963984540054  # standard normal quantile at 0.975
RUNNING, WAITING = "vllm:num_requests_running", "vllm:num_requests_waiting"
KEPT, DRAFTS, SUCCESS = "vllm:prefix_cache_hits", "vllm:spec_decode_num_drafts", "vllm:request_success"
UNIT = re.compile(r"\w+|[^\w\s]")


class Inconclusive(Exception):
    """The run cannot go on to a result; the message names why."""


def task(n_req: int) -> str:
    return (f"Above is the first part of the source. There are {n_req - 1} further sections, numbered 1 to {n_req - 1}. "
            "Fetch them one at a time with read_section, in order. After each one, note in two sentences what that "
            "section does, then fetch the next. After the last section, write a short memorandum (about 300 words) "
            "that summarizes the sections and names any function that looks error-prone.")


# ------------------------------------------------------------------ chats (pure)

@dataclass(frozen=True)
class Shape:
    chars: int      # characters of source in the first message
    tail: int       # characters of source in each tool result
    followups: int  # tool-call requests between the first read and the final request
    label: str      # "" = no label line

    def __post_init__(self) -> None:
        if self.chars < 1 or self.tail < 1 or self.followups < 0:
            raise ValueError("chars and tail must be at least 1, followups at least 0")
        if "\n" in self.label:
            raise ValueError("the label is one line")

    @property
    def n_req(self) -> int:
        return self.followups + 2

    @property
    def span(self) -> int:
        return self.chars + self.tail * self.n_req

    def to_json(self) -> dict:
        return {"chars": self.chars, "tail": self.tail, "followups": self.followups, "label": self.label}


@dataclass(frozen=True)
class Chat:
    index: int
    first_text: str
    task: str
    sections: tuple[str, ...]  # sections[i] answers the tool call of request i

    def sha(self) -> str:
        body = json.dumps([self.first_text, self.task, list(self.sections)]).encode()
        return hashlib.sha256(body).hexdigest()[:16]


def build_chat(txt: str, shape: Shape, index: int) -> Chat:
    lo = index * shape.span
    if len(txt) < lo + shape.span:
        raise ValueError(f"chat {index} needs {lo + shape.span} characters of source, {len(txt)} given")
    tag = f"Run {shape.label} chat {index}.\n\n" if shape.label else ""
    at = lo + shape.chars
    sections = tuple(txt[at + i * shape.tail: at + (i + 1) * shape.tail] for i in range(shape.followups + 1))
    return Chat(index, tag + HEADER + txt[lo: at], task(shape.n_req), sections)


def source_text(need: int) -> str:
    txt = ""
    for pattern in SOURCE_GLOBS:
        for name in sorted(glob.glob(pattern)):
            if len(txt) >= need:
                return txt
            txt += open(name, errors="replace").read()
    if len(txt) < need:
        raise SystemExit(f"only {len(txt)} characters of source available, asked {need}")
    return txt


# --------------------------------------------------------------- measures (pure)

def distinct_share(seq: list, window: int = WINDOW) -> float | None:
    """Distinct windows / windows; None when seq is shorter than one window."""
    count = len(seq) - window + 1
    if count < 1:
        return None
    return len({tuple(seq[i: i + window]) for i in range(count)}) / count


def longest_repeat(seq: list) -> int:
    """Length of the longest span that occurs at least twice in seq (occurrences may overlap).

    Suffix automaton: every state stands for spans that share their end positions; a state with two end
    positions or more is a repeated span, and its longest span has the state's length.
    """
    nxt: list[dict] = [{}]
    link, length, ends = [-1], [0], [0]
    last = 0
    for x in seq:
        cur = len(nxt)
        nxt.append({})
        link.append(0)
        length.append(length[last] + 1)
        ends.append(1)
        p = last
        while p != -1 and x not in nxt[p]:
            nxt[p][x] = cur
            p = link[p]
        if p != -1:
            q = nxt[p][x]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(nxt)
                nxt.append(dict(nxt[q]))
                link.append(link[q])
                length.append(length[p] + 1)
                ends.append(0)
                while p != -1 and nxt[p].get(x) == q:
                    nxt[p][x] = clone
                    p = link[p]
                link[q] = link[cur] = clone
        last = cur
    for v in sorted(range(1, len(nxt)), key=length.__getitem__, reverse=True):
        ends[link[v]] += ends[v]
    return max((length[v] for v in range(1, len(nxt)) if ends[v] >= 2), default=0)


def wilson95(k: int, n: int) -> tuple[float, float] | None:
    """Wilson score interval, 95 %, of k successes in n trials; None when n is 0."""
    if n == 0:
        return None
    if not 0 <= k <= n:
        raise ValueError(f"{k} of {n}")
    p, zz = k / n, Z95 * Z95
    centre = (p + zz / (2 * n)) / (1 + zz / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / (1 + zz / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Measures:
    units: int
    distinct_all: float | None
    distinct_last_quarter: float | None
    longest_repeat: int

    @classmethod
    def of(cls, seq: list) -> Measures:
        quarter = seq[len(seq) - len(seq) // 4:]
        return cls(len(seq), distinct_share(seq), distinct_share(quarter), longest_repeat(seq))

    def to_json(self) -> dict:
        def r(v: float | None) -> float | None:
            return None if v is None else round(v, 4)
        return {"units": "text", "unit_count": self.units, "distinct20": r(self.distinct_all),
                "distinct20_last_quarter": r(self.distinct_last_quarter), "longest_repeat": self.longest_repeat}


def text_units(blocks: list[dict]) -> list[str]:
    parts = [b.get("thinking", "") if b["type"] == "thinking" else b.get("text", "")
             for b in blocks if b["type"] in ("thinking", "text")]
    return UNIT.findall("\n".join(parts))


def thinking_closed(kinds: list[str]) -> bool | None:
    if "thinking" not in kinds:
        return None
    last = len(kinds) - 1 - kinds[::-1].index("thinking")
    return last < len(kinds) - 1


# --------------------------------------------------------------- outcomes (pure)

@dataclass(frozen=True)
class Ended:
    name = "ENDED"
    reason = None


@dataclass(frozen=True)
class Limit:
    name = "LIMIT"
    reason = None


@dataclass(frozen=True)
class Undecided:
    reason: str
    name = "INCONCLUSIVE"


Outcome = Ended | Limit | Undecided


def final_outcome(stop: str | None) -> Outcome:
    if stop == "end_turn":
        return Ended()
    if stop == "max_tokens":
        return Limit()
    return Undecided(f"the final request stopped {stop!r}")


def summary_line(names: list[str], kept_finals: int) -> str:
    """names: the outcome name of every chat. The limit share is taken over the decided chats."""
    ended, limit = names.count("ENDED"), names.count("LIMIT")
    undecided = len(names) - ended - limit
    head = (f"SUMMARY chats {len(names)}: ended normally {ended}, ran to the limit {limit}, undecided {undecided}; "
            f"final request kept its prompt in {kept_finals} of {len(names)}; ")
    interval = wilson95(limit, ended + limit)
    if interval is None:
        return head + "limit share INCONCLUSIVE (no chat reached a decided final request)"
    line = head + (f"limit share {limit} of {ended + limit} = {limit / (ended + limit):.3f}, Wilson 95 % interval "
                   f"{interval[0]:.3f} to {interval[1]:.3f}")
    if undecided:
        line += f"; INCONCLUSIVE for {undecided} chats (not in the share)"
    return line


# ------------------------------------------------------------------ requests (I/O)

@dataclass(frozen=True)
class Reply:
    blocks: list[dict]
    input_tokens: int
    output_tokens: int
    stop: str | None
    ttft_s: float
    dec_s: float
    wall_s: float


def metrics(url: str) -> dict[str, float | None]:
    """Sum of every series of each counter; None for a counter the server does not print."""
    got: dict[str, float | None] = dict.fromkeys((RUNNING, WAITING, KEPT, DRAFTS, SUCCESS))
    for line in urllib.request.urlopen(url + "/metrics", timeout=10).read().decode().splitlines():
        for name in got:
            if line.startswith((name + "_total{", name + "{", name + "_total ", name + " ")):
                got[name] = (got[name] or 0.0) + float(line.rsplit(" ", 1)[1])
    for name in (RUNNING, WAITING, KEPT):
        if got[name] is None:
            raise Inconclusive(f"{name} is not in {url}/metrics")
    return got


def call(url: str, model: str, messages: list[dict]) -> Reply:
    body = {"model": model, "max_tokens": MAX_TOKENS, "stream": True, "temperature": 0, "system": SYSTEM,
            "tools": TOOLS, "messages": messages, "chat_template_kwargs": {"thinking": True}}
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
    t0 = time.monotonic()
    first = last = None
    blocks: list[dict] = []
    usage: dict = {}
    stop = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            t = ev.get("type")
            if t == "error":
                raise RuntimeError(f"server error event: {str(ev)[:300]}")
            if t == "message_start":
                usage.update(ev["message"].get("usage") or {})
            elif t == "content_block_start":
                b = dict(ev["content_block"])
                if b["type"] == "tool_use":
                    b["_json"] = ""
                blocks.append(b)
            elif t == "content_block_delta":
                now = time.monotonic()
                first = first or now
                last = now
                d = ev["delta"]
                b = blocks[ev["index"]]
                if d["type"] == "thinking_delta":
                    b["thinking"] = b.get("thinking", "") + d["thinking"]
                elif d["type"] == "text_delta":
                    b["text"] = b.get("text", "") + d["text"]
                elif d["type"] == "input_json_delta":
                    b["_json"] += d["partial_json"]
                elif d["type"] == "signature_delta":
                    b["signature"] = b.get("signature", "") + d["signature"]
                else:
                    raise RuntimeError(f"unknown delta type {d['type']}")
            elif t == "message_delta":
                usage.update(ev.get("usage") or {})
                stop = ev["delta"].get("stop_reason")
    end = time.monotonic()
    if first is None:
        raise RuntimeError(f"nothing streamed (stop {stop})")
    for b in blocks:
        if b["type"] == "tool_use":
            b["input"] = json.loads(b.pop("_json") or "{}")
    return Reply(blocks, int(usage["input_tokens"]), int(usage["output_tokens"]), stop,
                 first - t0, last - first, end - t0)


def step(url: str, model: str, req: int | str, chat: int, messages: list[dict],
         prev_ctx: int | None) -> tuple[dict, Reply]:
    """One request. Returns its numbers (no text) and the reply."""
    a = metrics(url)
    if a[RUNNING] != 0 or a[WAITING] != 0:
        raise Inconclusive(f"server busy before request {req} of chat {chat}: {a[RUNNING]:g} running, "
                           f"{a[WAITING]:g} waiting")
    reply = call(url, model, messages)
    b = metrics(url)
    n = reply.output_tokens
    kept = int(b[KEPT] - a[KEPT])
    rounds = None if a[DRAFTS] is None or b[DRAFTS] is None else b[DRAFTS] - a[DRAFTS]
    rec = {"req": req, "ctx_tokens": reply.input_tokens, "kept_tokens": kept,
           "hit": None if prev_ctx is None else kept >= prev_ctx - REACH,
           "out_tokens": n, "stop": reply.stop, "kinds": [x["type"] for x in reply.blocks],
           "ttft_s": round(reply.ttft_s, 2), "dec_s": round(reply.dec_s, 2),
           "dec_tok_s": round((n - 1) / reply.dec_s, 2) if reply.dec_s > 0 and n > 1 else None,
           "dec_ms_round": round(reply.dec_s * 1e3 / rounds, 2) if rounds else None,
           "tok_per_round": round(n / rounds, 2) if rounds else None,
           "wall_s": round(reply.wall_s, 2)}
    print(f"REQ chat {chat} req {req} ctx {rec['ctx_tokens']} kept {kept} hit {rec['hit']} ttft {rec['ttft_s']} s "
          f"out {n} decode {rec['dec_tok_s']} tok/s {rec['tok_per_round']} tok/round wall {rec['wall_s']} s "
          f"stop {reply.stop} blocks {'+'.join(rec['kinds'])}", flush=True)
    return rec, reply


def run_chat(url: str, model: str, chat: Chat, shape: Shape) -> dict:
    """Send one chat; return its record (numbers only)."""
    before = metrics(url)
    messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": chat.first_text},
                                                         {"type": "text", "text": chat.task}]}]
    steps: list[dict] = []
    prev_ctx = None
    outcome: Outcome | None = None
    for i in range(shape.followups + 1):  # the first read and the follow-ups: each must call the tool
        rec, reply = step(url, model, i, chat.index, messages, prev_ctx)
        prev_ctx = rec["ctx_tokens"]
        messages = messages + [{"role": "assistant", "content": reply.blocks}]
        uses = [x for x in reply.blocks if x["type"] == "tool_use"]
        if reply.stop != "tool_use" or not uses:
            outcome = Undecided(f"request {i} did not call the tool (stop {reply.stop!r})")
            break
        steps.append(rec)
        messages = messages + [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u["id"],
             "content": chat.sections[i] if k == 0 else EXTRA_USE} for k, u in enumerate(uses)]}]
    else:
        rec, reply = step(url, model, "final", chat.index, messages, prev_ctx)
        outcome = final_outcome(reply.stop)
    after = metrics(url)
    sent = len(steps) + 1
    foreign = None if before[SUCCESS] is None or after[SUCCESS] is None else int(after[SUCCESS] - before[SUCCESS]) - sent
    if foreign:
        print(f"NOTE chat {chat.index}: the server finished {foreign + sent} requests while this chat sent {sent}",
              flush=True)
    measures = Measures.of(text_units(reply.blocks))
    record = {"test": "termination", "v": 1, "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
              "url": url, "model": model, **shape.to_json(), "chat": chat.index, "chat_sha": chat.sha(),
              "outcome": outcome.name, "reason": outcome.reason,
              **{k: v for k, v in rec.items() if k != "req"}, "last_req": rec["req"],
              "thinking_closed": thinking_closed(rec["kinds"]), **measures.to_json(),
              "foreign_requests": foreign, "steps": steps}
    print(f"CHAT {chat.index} {outcome.name} out {rec['out_tokens']} stop {rec['stop']} units {measures.units} "
          f"distinct20 {record['distinct20']} last-quarter {record['distinct20_last_quarter']} "
          f"longest-repeat {measures.longest_repeat} thinking-closed {record['thinking_closed']}"
          + (f" reason: {outcome.reason}" if outcome.reason else ""), flush=True)
    return record


# ---------------------------------------------------------------------- modes

ARM_KEYS = ("url", "model", "chars", "tail", "followups", "label")


def read_receipt(path: Path) -> list[dict]:
    """The records of one arm; ValueError when the file mixes arms or holds a chat twice."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in records:
        if r.get("test") != "termination" or r.get("outcome") not in ("ENDED", "LIMIT", "INCONCLUSIVE"):
            raise ValueError(f"{path}: a line is not a termination-test record")
        if [r[k] for k in ARM_KEYS] != [records[0][k] for k in ARM_KEYS]:
            raise ValueError(f"{path} mixes runs that differ in one of {ARM_KEYS}")
    chats = [r["chat"] for r in records]
    if len(set(chats)) != len(chats):
        raise ValueError(f"{path} holds a chat twice")
    return records


def summarize(records: list[dict]) -> tuple[str, int]:
    names = [r["outcome"] for r in records]
    line = summary_line(names, sum(1 for r in records if r["hit"]))
    return line, 0 if names and all(n != "INCONCLUSIVE" for n in names) else 3


def shape_of(a: argparse.Namespace) -> Shape:
    return Shape(a.chars, a.tail, a.followups, a.label)


def chats_of(a: argparse.Namespace, shape: Shape) -> list[Chat]:
    if a.chats < 1 or a.start < 0:
        raise SystemExit("--chats must be at least 1 and --start at least 0")
    txt = source_text(shape.span * (a.start + a.chats))
    return [build_chat(txt, shape, c) for c in range(a.start, a.start + a.chats)]


def mode_run(a: argparse.Namespace) -> int:
    shape, url, out = shape_of(a), a.url.rstrip("/"), Path(a.out)
    chats = chats_of(a, shape)
    old = read_receipt(out) if out.exists() else []
    if old:
        mine = [url, a.model, *(shape.to_json()[k] for k in ARM_KEYS[2:])]
        if [old[0][k] for k in ARM_KEYS] != mine:
            raise SystemExit(f"{out} holds another arm or shape; give a new file")
        seen = {r["chat"] for r in old} & {c.index for c in chats}
        if seen:
            raise SystemExit(f"{out} already holds chats {sorted(seen)}")
    records = []
    stopped = None
    try:
        for chat in chats:
            record = run_chat(url, a.model, chat, shape)
            with out.open("a") as f:
                f.write(json.dumps(record) + "\n")
            records.append(record)
    except Inconclusive as exc:
        stopped = str(exc)
    line, code = summarize(old + records) if old + records else ("SUMMARY chats 0: no chat finished", 3)
    print(line)
    if stopped is not None:
        print(f"INCONCLUSIVE: run stopped: {stopped}")
        return 3
    return code


def mode_summary(a: argparse.Namespace) -> int:
    line, code = summarize(read_receipt(Path(a.out)))
    print(line)
    return code


def mode_plan(a: argparse.Namespace) -> int:
    shape = shape_of(a)
    print(f"PLAN shape {shape.to_json()} requests per chat {shape.n_req} span {shape.span} characters")
    for chat in chats_of(a, shape):
        first = hashlib.sha256(chat.first_text.encode()).hexdigest()[:16]
        print(f"PLAN chat {chat.index} first message {len(chat.first_text)} characters sha {first} "
              f"sections {len(chat.sections)} x {len(chat.sections[0])} chat_sha {chat.sha()}")
    return 0


def harness_copies() -> dict:
    """SYSTEM, TOOLS and the task text for 8 requests, evaluated from harness_loop.py's source."""
    source = HARNESS.read_text()
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("SYSTEM", "TOOLS"):
                found[name] = ast.literal_eval(node.value)
            elif name == "TASK":
                found[name] = eval(compile(ast.Expression(node.value), str(HARNESS), "eval"), {"n_req": 8})
    found["source"] = source
    return found


def mode_self_test(_: argparse.Namespace) -> int:
    def check(name: str, got: object, want: object, tol: float = 0.0) -> None:
        if isinstance(want, float) and got is not None:
            ok = abs(got - want) <= tol
        else:
            ok = got == want
        print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r} want {want!r}")
        if not ok:
            raise SystemExit(f"self-test failed at: {name}")

    # A. pure loop, period 7, 700 units: 681 windows, 7 distinct; last quarter 175 units, 156 windows, 7 distinct;
    #    longest repeat 700 - 7 = 693.
    m = Measures.of([i % 7 for i in range(700)])
    check("loop units", m.units, 700)
    check("loop distinct20 = 7/681", m.distinct_all, 0.010279, 5e-7)
    check("loop last quarter = 7/156", m.distinct_last_quarter, 0.044872, 5e-7)
    check("loop longest repeat = 700-7", m.longest_repeat, 693)
    # B. 400 different units: 381 of 381 windows distinct; last quarter 81 of 81; nothing repeats.
    m = Measures.of(list(range(400)))
    check("distinct distinct20", m.distinct_all, 1.0)
    check("distinct last quarter", m.distinct_last_quarter, 1.0)
    check("distinct longest repeat", m.longest_repeat, 0)
    # C. 600 different units, then 200 units of a period-5 loop of five new units: 781 windows; the 600 that start
    #    in the first part are distinct, the 181 inside the loop give 5: 605/781. Last quarter = the loop: 181
    #    windows, 5 distinct. Longest repeat 200 - 5 = 195.
    m = Measures.of(list(range(600)) + [1000 + i % 5 for i in range(200)])
    check("late loop distinct20 = 605/781", m.distinct_all, 0.774648, 5e-7)
    check("late loop last quarter = 5/181", m.distinct_last_quarter, 0.027624, 5e-7)
    check("late loop longest repeat = 200-5", m.longest_repeat, 195)
    # D. 19 units "abababab...": no 20-unit window; longest repeat 19 - 2 = 17.
    m = Measures.of(["a", "b"] * 9 + ["a"])
    check("short distinct20", m.distinct_all, None)
    check("short last quarter", m.distinct_last_quarter, None)
    check("short longest repeat = 19-2", m.longest_repeat, 17)
    # E. "abcabd": "ab" occurs twice, no 3-unit span does.
    check("abcabd longest repeat", longest_repeat(list("abcabd")), 2)
    check("empty longest repeat", longest_repeat([]), 0)
    # Text units: six units in "The cat, the cat."; thinking then text are joined in order.
    check("units", text_units([{"type": "text", "text": "The cat, the cat."}]), ["The", "cat", ",", "the", "cat", "."])
    check("units order", text_units([{"type": "thinking", "thinking": "a b"}, {"type": "tool_use"},
                                     {"type": "text", "text": "c"}]), ["a", "b", "c"])
    check("thinking open", thinking_closed(["thinking"]), False)
    check("thinking closed", thinking_closed(["thinking", "text"]), True)
    check("thinking none", thinking_closed(["text", "tool_use"]), None)
    check("thinking reopened", thinking_closed(["thinking", "text", "thinking"]), False)
    # Wilson 95 %, z = 1.959964. 2 of 5: centre (0.4 + 0.384146) / 1.768292 = 0.443448; half-width
    # 1.959964 * sqrt(0.048 + 0.0384146) / 1.768292 = 0.325827; so 0.117621 to 0.769276.
    # 0 of 5: 0 to z^2 / (5 + z^2) = 3.841459 / 8.841459 = 0.434483. 5 of 5: 5 / 8.841459 = 0.565517 to 1.
    lo, hi = wilson95(2, 5)
    check("wilson 2 of 5 low", lo, 0.117621, 2e-6)
    check("wilson 2 of 5 high", hi, 0.769276, 2e-6)
    lo, hi = wilson95(0, 5)
    check("wilson 0 of 5 low", lo, 0.0, 1e-12)
    check("wilson 0 of 5 high", hi, 0.434483, 2e-6)
    lo, hi = wilson95(5, 5)
    check("wilson 5 of 5 low", lo, 0.565517, 2e-6)
    check("wilson 5 of 5 high", hi, 1.0, 1e-12)
    check("wilson 0 of 0", wilson95(0, 0), None)
    check("outcome end_turn", final_outcome("end_turn").name, "ENDED")
    check("outcome max_tokens", final_outcome("max_tokens").name, "LIMIT")
    check("outcome tool_use", final_outcome("tool_use").name, "INCONCLUSIVE")
    check("summary 3 ended 2 limit", summary_line(["ENDED", "LIMIT", "ENDED", "LIMIT", "ENDED"], 5),
          "SUMMARY chats 5: ended normally 3, ran to the limit 2, undecided 0; final request kept its prompt in 5 "
          "of 5; limit share 2 of 5 = 0.400, Wilson 95 % interval 0.118 to 0.769")
    check("summary undecided", summary_line(["INCONCLUSIVE"], 0),
          "SUMMARY chats 1: ended normally 0, ran to the limit 0, undecided 1; final request kept its prompt in 0 "
          "of 1; limit share INCONCLUSIVE (no chat reached a decided final request)")
    # Chat layout on a 60-character text, chars 10, tail 3, one follow-up: 3 requests, span 10 + 3 * 3 = 19.
    # Chat 1 starts at 19: first text = header + txt[19:29], sections txt[29:32] and txt[32:35].
    txt = "".join(chr(65 + i % 26) for i in range(60))
    c = build_chat(txt, Shape(10, 3, 1, ""), 1)
    check("chat 1 first text", c.first_text, HEADER + "TUVWXYZABC")
    check("chat 1 sections", c.sections, ("DEF", "GHI"))
    check("chat 1 task", c.task.startswith("Above is the first part of the source. There are 2 further sections, "
                                           "numbered 1 to 2. "), True)
    check("label line", build_chat(txt, Shape(10, 3, 1, "w4"), 1).first_text, "Run w4 chat 1.\n\n" + HEADER + "TUVWXYZABC")
    # The copies against harness_loop.py's own source text.
    h = harness_copies()
    check("SYSTEM = harness_loop.py's", SYSTEM, h["SYSTEM"])
    check("TOOLS = harness_loop.py's", TOOLS, h["TOOLS"])
    check("task(8) = harness_loop.py's TASK at 8 requests", task(8), h["TASK"])
    for name, piece in (("header", '"Here is Python standard-library source code.\\n\\n" + txt[c * span: c * span + chars]'),
                        ("extra tool use", '"One section per step."'), ("reach", "prev_ctx - 124"),
                        ("globs", 'sorted(glob.glob("/usr/lib/python3.12/*.py")) + sorted(glob.glob("/usr/lib/python3.12/*/*.py"))'),
                        ("span", "span = chars + tail * n_req"), ("section offset", "lo = c * span + chars + i * tail"),
                        ("request body", '"stream": True, "temperature": 0, "system": SYSTEM')):
        check(f"harness_loop.py still has its {name}", piece in h["source"], True)
    print("SELF-TEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run", help="send the chats to URL, append one line per chat to OUT.jsonl")
    run.add_argument("url")
    run.add_argument("out")
    run.add_argument("--model", default="deepseek-v4.1-flash", help="model name to send (default %(default)s)")
    plan = sub.add_parser("plan", help="print the sizes and sha256 of the chats; no request")
    for p in (run, plan):
        p.add_argument("--chats", type=int, default=5, help="number of chats (default %(default)s)")
        p.add_argument("--start", type=int, default=0, help="index of the first chat (default %(default)s)")
        p.add_argument("--followups", type=int, default=2,
                       help="tool-call follow-ups before the final request (default %(default)s; 6 = the 216K loop)")
        p.add_argument("--chars", type=int, default=830000,
                       help="characters of source in the first message (default %(default)s, about 216K tokens; "
                            "120000 is about 31K)")
        p.add_argument("--tail", type=int, default=12000,
                       help="characters of source per tool result (default %(default)s; the 31K chat used 8000)")
        p.add_argument("--label", default="", help="first line 'Run LABEL chat I.' in every chat (default: none)")
    summary = sub.add_parser("summary", help="print the summary line of OUT.jsonl")
    summary.add_argument("out")
    sub.add_parser("self-test", help="CPU checks with hand-computed values; no request")
    a = parser.parse_args()
    return {"run": mode_run, "summary": mode_summary, "plan": mode_plan, "self-test": mode_self_test}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
