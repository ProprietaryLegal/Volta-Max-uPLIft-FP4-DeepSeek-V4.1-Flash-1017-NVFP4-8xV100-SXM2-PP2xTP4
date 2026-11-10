#!/usr/bin/env python3
"""Capture one arm of the NVFP4 closeness test: one request at a time, keyed token hashes and logprobs only.

  capture.py --plan PLAN --url URL --arm LABEL --arm-kind KIND --passes P[,P] --budget-min M \\
             --key KEYFILE --tokenizer TOKENIZER_JSON --out DIR [--part N] [--resume PART ...] [--wait-idle]
  capture.py --dry --plan PLAN --arm-kind KIND --passes P[,P] --budget-min M     request plan and time estimate
  capture.py --self-test                                                          CPU checks; no request

KIND is reference, hot_iq or volta_split (it picks the time model of --dry and is recorded). Passes, in the order
given: floor (the first 50 contexts and first 50 MC items, captured a second time on the reference), bridge (the
first 10 contexts, for a continuation part), main (every context, then every MC item). --resume names earlier part
files of the same arm: their main items are not sent again.

Every request: POST /v1/chat/completions, temperature 0, thinking off, top_logprobs 20, tokens named by id, token
ids returned, streamed, and a fresh cache_salt so that no request reuses another's cached prompt (every prompt is
read in full, by the same path, in every arm). Contexts: 32 tokens with ignore_eos. MC: at most 8 tokens.

What is written (DIR/LABEL.partN.jsonl, opened exclusively, one line per item, flushed): for every generated
position the keyed hash of the chosen token and of each of the top 20, and their logprobs; the item's prompt sha256,
prompt token count, timing, and for MC the index of the parsed answer letter and the position of its token. The
token ids and any text exist only in memory. Without the key file the hashes cannot be turned back into tokens.

Before each request the server must be idle (requests running + waiting = 0). On a lane server anything else
stops the arm (exit 3). With --wait-idle (production) the client waits, polling every 5 s, until two polls in a row
show it idle, and records the wait. After each request the server's finished-request counter must have moved by
exactly one on a lane server. Any HTTP error, malformed stream or count mismatch ends the arm with a traceback:
no retries. --budget-min: no new item starts after that many minutes; the arm then ends with BUDGET_STOP.

Exit 0 COMPLETE, 4 BUDGET_STOP (receipt valid, some items not captured), 3 INCONCLUSIVE (server not idle or not
healthy at the start), 1 error.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closeness_lib as cl  # noqa: E402

NO_LOGPROB = -9999.0
PROGRESS_EVERY = 20
IDLE_POLL_S = 5.0
REQUEST_TIMEOUT_S = 300


class NotIdle(Exception):
    pass


# ------------------------------------------------------------------------------------------- server I/O

def http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {"running": None, "waiting": None, "success": None}
    names = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
             "vllm:request_success_total": "success"}
    for line in text.splitlines():
        for name, key in names.items():
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[key] = (out[key] or 0.0) + float(line.split()[-1])
    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise RuntimeError(f"{url}/metrics lacks {missing}")
    return out


def wait_idle(url: str, wait: bool) -> float:
    t0 = time.monotonic()
    quiet = 0
    while True:
        m = metrics(url)
        busy = m["running"] + m["waiting"]
        if busy == 0:
            quiet += 1
            if not wait or quiet == 2:
                return time.monotonic() - t0
        elif not wait:
            raise NotIdle(f"{url}: {busy:g} requests running or waiting")
        else:
            quiet = 0
        time.sleep(IDLE_POLL_S)


def token_id(token: object) -> int:
    if not (isinstance(token, str) and token.startswith("token_id:") and token[9:].isdigit()):
        raise RuntimeError("the server did not name a logprob token by id")
    return int(token[9:])


def parse_stream(lines, key: bytes) -> dict:
    """Parse an SSE stream of /v1/chat/completions. Text deltas are never read. Returns ids (in memory only),
    the per-position hashes and logprobs, usage, finish reason and the time of the first token."""
    ids: list[int] = []
    chosen: list[tuple[int, float]] = []
    tops: list[list[tuple[int, float]]] = []
    usage, finish, t_first = None, None, None
    for raw in lines:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
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
        for choice in obj.get("choices") or []:
            got = [int(t) for t in choice.get("token_ids") or []]
            if got and t_first is None:
                t_first = time.monotonic()
            ids.extend(got)
            for entry in (choice.get("logprobs") or {}).get("content") or []:
                chosen.append((token_id(entry["token"]), float(entry["logprob"])))
                tops.append([(token_id(t["token"]), float(t["logprob"])) for t in entry["top_logprobs"]])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if usage is None:
        raise RuntimeError("the stream carried no usage")
    if [t for t, _ in chosen] != ids or len(ids) != int(usage["completion_tokens"]):
        raise RuntimeError(f"token ids {len(ids)}, logprob entries {len(chosen)}, completion_tokens "
                           f"{usage['completion_tokens']}: they must agree")
    for i, (t, lp) in enumerate(chosen):
        if lp <= NO_LOGPROB:
            raise RuntimeError(f"position {i}: the server gave no logprob for the chosen token")
        if len(tops[i]) != cl.TOP_K:
            raise RuntimeError(f"position {i}: {len(tops[i])} top logprobs, want {cl.TOP_K}")
        if t not in [x for x, _ in tops[i]]:
            raise RuntimeError(f"position {i}: the greedy token is not in its own top {cl.TOP_K}")
    rec = {"tok": [cl.tok_hash(key, t) for t, _ in chosen], "lp": [round(lp, 7) for _, lp in chosen],
           "top": [[cl.tok_hash(key, x) for x, _ in tp] for tp in tops],
           "toplp": [[round(v, 7) for _, v in tp] for tp in tops],
           "chosen_not_top1": sum(1 for (t, _), tp in zip(chosen, tops) if t != tp[0][0]),
           "seq_sha": cl.sha256_bytes(b"".join(bytes.fromhex(cl.tok_hash(key, t)) for t in ids))}
    return {"ids": ids, "rec": rec, "usage": usage, "finish": finish, "t_first": t_first}


def request_body(model: str, message: str, kind: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": message}], "temperature": 0,
            "chat_template_kwargs": {"thinking": False}, "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "logprobs": True, "top_logprobs": cl.TOP_K,
            "return_tokens_as_token_ids": True, "cache_salt": secrets.token_hex(16)}
    if kind == "context":
        body.update(max_tokens=cl.CONT_TOKENS, ignore_eos=True)
    else:
        body.update(max_tokens=cl.MC_MAX_TOKENS, ignore_eos=False)
    return body


def mc_answer(ids: list[int], tokenizer) -> tuple[int | None, int | None]:
    """Parse the answer letter from the generated tokens (decoded in memory only) and find the position of the
    token that carries it."""
    text = tokenizer.decode(ids, skip_special_tokens=True)
    m = cl.MC_ANSWER.search(text)
    if m is None:
        return None, None
    at = m.start(1)
    for k in range(1, len(ids) + 1):
        if len(tokenizer.decode(ids[:k], skip_special_tokens=True)) > at:
            return cl.LETTERS.index(m.group(1)), k - 1
    raise RuntimeError("the answer letter is in the text but in no token prefix")


# ------------------------------------------------------------------------------------------- the arm

def item_list(plan: cl.Plan, passes: list[str], skip_main: set[str]) -> list[tuple[str, str, str]]:
    """(pass, item id, 'context' | 'mc') in capture order."""
    o = plan.doc["order"]
    out: list[tuple[str, str, str]] = []
    for p in passes:
        if p == "floor":
            out += [(p, i, "context") for i in o["floor_contexts"]] + [(p, i, "mc") for i in o["floor_mc"]]
        elif p == "bridge":
            out += [(p, i, "context") for i in o["bridge_contexts"]]
        elif p == "main":
            out += [(p, i, "context") for i in o["main_contexts"] if i not in skip_main]
            out += [(p, i, "mc") for i in o["main_mc"] if i not in skip_main]
        else:
            raise ValueError(f"unknown pass {p!r}")
    return out


def client_sha() -> str:
    here = Path(__file__).resolve().parent
    return cl.sha256_bytes((here / "capture.py").read_bytes() + (here / "closeness_lib.py").read_bytes())


def write_line(f, obj: dict) -> None:
    f.write(json.dumps(obj, separators=(",", ":")) + "\n")
    f.flush()
    os.fsync(f.fileno())


def run_arm(a) -> int:
    from tokenizers import Tokenizer

    plan = cl.Plan.load(a.plan)
    key = cl.read_key(a.key)
    kid = cl.key_id(key)
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    if cl.sha256_bytes(a.tokenizer.read_bytes()) != plan.doc["tokenizer_sha256"]:
        raise ValueError(f"{a.tokenizer} is not the tokenizer the plan was built with")
    skip: set[str] = set()
    if a.resume:
        done, heads = cl.read_receipts(a.resume, plan, kid)
        if any(h["arm"] != a.arm for h in heads):
            raise ValueError("--resume parts belong to another arm")
        skip = {i for (p, i) in done if p == "main"}
    passes = a.passes.split(",")
    if a.resume and passes[0] != "bridge":
        raise ValueError("a continuation part must start with the bridge pass")
    todo = item_list(plan, passes, skip)
    url = a.url.rstrip("/")
    out = a.out / f"{a.arm}.part{a.part}.jsonl"
    try:
        with urllib.request.urlopen(url + "/health", timeout=30) as r:
            if r.status != 200:
                raise NotIdle(f"{url}/health answered {r.status}")
        models = http_json(url + "/v1/models")["data"]
        if len(models) != 1:
            raise RuntimeError(f"{url} serves {len(models)} models, want one")
        model = models[0]["id"]
        wait_idle(url, a.wait_idle)
    except NotIdle as exc:
        print(json.dumps({"status": "INCONCLUSIVE", "arm": a.arm, "reason": str(exc)}))
        return 3
    ctx, mcs = plan.contexts(), plan.mc()
    head = {"type": "header", "format": cl.RECEIPT_FORMAT, "arm": a.arm, "arm_kind": a.arm_kind, "part": a.part,
            "plan_sha256": plan.sha256, "key_id": kid, "client_sha256": client_sha(), "url": url, "model": model,
            "passes": passes, "resumed_from": [str(p) for p in (a.resume or [])], "items_planned": len(todo),
            "budget_min": a.budget_min, "wait_idle": a.wait_idle,
            "utc_start": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request": plan.doc["request"]}
    t0 = time.monotonic()
    counts: dict[str, int] = {}
    status = "COMPLETE"
    with out.open("x") as f:
        write_line(f, head)
        try:
            for n, (pss, iid, kind) in enumerate(todo):
                if time.monotonic() - t0 >= a.budget_min * 60:
                    status = "BUDGET_STOP"
                    break
                waited = wait_idle(url, a.wait_idle)
                before = metrics(url)["success"]
                message = cl.context_message(ctx[iid]) if kind == "context" else mcs[iid]["prompt"]
                body = request_body(model, message, kind)
                req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                t_send = time.monotonic()
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as reply:
                    got = parse_stream(reply, key)
                t_end = time.monotonic()
                del message, body
                n_tok = len(got["ids"])
                if kind == "context" and (n_tok != cl.CONT_TOKENS or got["finish"] != "length"):
                    raise RuntimeError(f"{iid}: {n_tok} tokens, finish {got['finish']!r}; want {cl.CONT_TOKENS}, 'length'")
                if kind == "mc" and not 1 <= n_tok <= cl.MC_MAX_TOKENS:
                    raise RuntimeError(f"{iid}: {n_tok} tokens")
                after = metrics(url)["success"]
                others = after - before - 1
                if others != 0 and not a.wait_idle:
                    raise RuntimeError(f"{iid}: the finished-request counter moved by {after - before:g}, want 1")
                line = {"type": "item", "item": iid, "kind": kind, "pass": pss,
                        "prompt_sha256": ctx[iid]["text_sha256"] if kind == "context" else mcs[iid]["prompt_sha256"],
                        "prompt_tokens": int(got["usage"]["prompt_tokens"]), "finish": got["finish"],
                        "ttft_s": round(got["t_first"] - t_send, 4), "total_s": round(t_end - t_send, 4),
                        "idle_wait_s": round(waited, 2), "others_finished": others, **got["rec"]}
                if kind == "mc":
                    line["answer"], line["answer_pos"] = mc_answer(got["ids"], tokenizer)
                del got
                write_line(f, line)
                counts[pss] = counts.get(pss, 0) + 1
                if (n + 1) % PROGRESS_EVERY == 0 or n + 1 == len(todo):
                    el = time.monotonic() - t0
                    print(f"PROGRESS {a.arm} {n + 1}/{len(todo)} items, elapsed {el:.0f} s, "
                          f"{el / (n + 1):.2f} s/item, projected {el / (n + 1) * len(todo):.0f} s "
                          f"(budget {a.budget_min * 60:.0f} s)", flush=True)
        except BaseException as exc:
            write_line(f, {"type": "end", "status": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                           "counts": counts, "elapsed_s": round(time.monotonic() - t0, 1)})
            raise
        write_line(f, {"type": "end", "status": status, "counts": counts, "planned": len(todo),
                       "elapsed_s": round(time.monotonic() - t0, 1)})
    print(json.dumps({"status": status, "arm": a.arm, "receipt": str(out), "bytes": out.stat().st_size,
                      "counts": counts, "planned": len(todo), "elapsed_s": round(time.monotonic() - t0, 1)}))
    return 0 if status == "COMPLETE" else 4


# ------------------------------------------------------------------------------------------- dry and self-test

def dry(a) -> int:
    plan = cl.Plan.load(a.plan)
    tm = cl.TIME_MODELS[a.arm_kind]
    ctx, mcs = plan.contexts(), plan.mc()
    passes = a.passes.split(",")
    todo = item_list(plan, passes, set())
    print(f"PLAN {a.plan} sha256 {plan.sha256[:16]}; arm kind {a.arm_kind}; passes {passes}; budget {a.budget_min} min")
    total = [0.0, 0.0]
    for p in passes:
        rows = [(i, k) for (pp, i, k) in todo if pp == p]
        for kind in ("context", "mc"):
            items = [i for i, k in rows if k == kind]
            if not items:
                continue
            toks = [(ctx[i] if kind == "context" else mcs[i])["user_tokens"] + cl.TEMPLATE_OVERHEAD for i in items]
            outn = (cl.CONT_TOKENS, cl.CONT_TOKENS) if kind == "context" else (4, cl.MC_MAX_TOKENS)
            c = sum(tm.request(t, outn[0], False) for t in toks)
            u = sum(tm.request(t, outn[1], True) for t in toks)
            total[0] += c
            total[1] += u
            print(f"  {p:6s} {kind:7s} {len(items):4d} requests, prompt tokens mean {sum(toks) / len(toks):6.1f} "
                  f"max {max(toks):4d}; {outn[0]} to {outn[1]} new tokens; estimate {c / 60:5.1f} min central, "
                  f"{u / 60:5.1f} min upper; cumulative {total[0] / 60:5.1f} / {total[1] / 60:5.1f} min")
    verdict = ("fits" if total[1] <= a.budget_min * 60 else
               "fits on the central estimate only" if total[0] <= a.budget_min * 60 else "DOES NOT FIT")
    print(f"TOTAL {len(todo)} requests: {total[0] / 60:.1f} min central, {total[1] / 60:.1f} min upper; "
          f"budget {a.budget_min} min: {verdict}")
    body = request_body("<served model>", "<item text>", "context")
    body["cache_salt"] = "<fresh 32 hex per request>"
    print("REQUEST (context) " + json.dumps(body))
    body = request_body("<served model>", "<item prompt>", "mc")
    body["cache_salt"] = "<fresh 32 hex per request>"
    print("REQUEST (mc)      " + json.dumps(body))
    print(f"WRITES {a.out}/<ARM>.part<N>.jsonl: header, one line per item, end line (status COMPLETE / BUDGET_STOP / ERROR)")
    return 0


class _FakeTok:
    """Maps token ids to fixed strings; stands in for the tokenizer in the self-test of mc_answer."""

    table = {1: "Answer", 2: ":", 3: " C", 4: "\n", 5: " (", 6: "B", 7: ")", 8: "Cats"}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.table[i] for i in ids)


def self_test() -> int:
    fails: list[str] = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    key = bytes(range(32))

    def entry(t, lp, top):
        return {"token": f"token_id:{t}", "logprob": lp,
                "top_logprobs": [{"token": f"token_id:{x}", "logprob": v} for x, v in top]}

    top_a = [(7, -0.1)] + [(100 + j, -5.0 - j) for j in range(cl.TOP_K - 1)]
    top_b = [(9, -0.2), (7, -1.9)] + [(200 + j, -6.0 - j) for j in range(cl.TOP_K - 2)]
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
              {"choices": [{"delta": {"content": "xx"}, "token_ids": [7, 9],
                            "logprobs": {"content": [entry(7, -0.1, top_a), entry(9, -0.2, top_b)]}}]},
              {"choices": [{"delta": {}, "finish_reason": "length"}]},
              {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 2}}]
    stream = [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"data: [DONE]\n"]
    got = parse_stream(stream, key)
    check("stream ids", got["ids"], [7, 9])
    check("stream finish", got["finish"], "length")
    check("stream hashes", got["rec"]["tok"], [cl.tok_hash(key, 7), cl.tok_hash(key, 9)])
    check("stream top length", [len(t) for t in got["rec"]["top"]], [cl.TOP_K, cl.TOP_K])
    check("stream logprobs", got["rec"]["lp"], [-0.1, -0.2])
    check("stream no text kept", "xx" in json.dumps(got["rec"]), False)
    check("stream chosen is top-1", got["rec"]["chosen_not_top1"], 0)
    bad = [f"data: {json.dumps(c)}\n".encode() for c in chunks[:2]] + \
          [f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 50, 'completion_tokens': 3}})}\n".encode()]
    try:
        parse_stream(bad, key)
        fails.append("a count mismatch was not refused")
    except RuntimeError:
        pass
    # keyed hash: HMAC-SHA256(key, id as 4 bytes big-endian), first 16 hex; checked against hashlib directly
    import hashlib
    import hmac as _h
    check("tok_hash", cl.tok_hash(key, 7), _h.new(key, b"\x00\x00\x00\x07", hashlib.sha256).hexdigest()[:16])
    tok = _FakeTok()
    check("mc Answer: C", mc_answer([1, 2, 3, 4], tok), (2, 2))
    check("mc Answer: (B)", mc_answer([1, 2, 5, 6, 7], tok), (1, 3))
    check("mc Answer: Cats", mc_answer([1, 2, 8], tok), (None, None))
    check("parse **Answer:** D", cl.parse_mc_answer("**Answer:** D"), 3)
    check("parse The answer is A", cl.parse_mc_answer("The answer is A"), None)
    # time model by hand: reference, 170 prompt tokens, 32 new tokens, central:
    # ttft = 0.70 + 4.30 * (100 / 930) = 1.16237; rounds = ceil(31 / 2.9) = 11; 11 * 0.140 = 1.54; + 0.08 = 2.78237
    check("time central", round(cl.TIME_MODELS["reference"].request(170, 32, False), 5), 2.78237)
    # upper: ttft = 0.70 + 4.30 * sqrt(100 / 930) = 0.70 + 4.30 * 0.327913 = 2.110027; rounds = ceil(31 / 2.4) = 13;
    # 13 * 0.230 = 2.99; + 0.08 = 5.180027
    check("time upper", round(cl.TIME_MODELS["reference"].request(170, 32, True), 4), 5.18)
    # request bodies
    b1, b2 = request_body("m", "x", "context"), request_body("m", "x", "context")
    check("fresh cache salt", b1["cache_salt"] != b2["cache_salt"] and len(b1["cache_salt"]) == 32, True)
    check("context body", (b1["max_tokens"], b1["ignore_eos"], b1["temperature"], b1["top_logprobs"],
                           b1["chat_template_kwargs"]), (32, True, 0, 20, {"thinking": False}))
    m = request_body("m", "x", "mc")
    check("mc body", (m["max_tokens"], m["ignore_eos"]), (8, False))
    for f in fails:
        print("FAIL", f)
    print("SELF-TEST PASS" if not fails else f"SELF-TEST FAIL ({len(fails)})")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--url")
    ap.add_argument("--arm")
    ap.add_argument("--arm-kind", choices=sorted(cl.TIME_MODELS))
    ap.add_argument("--passes", default="main")
    ap.add_argument("--budget-min", type=float)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--resume", type=Path, nargs="*")
    ap.add_argument("--wait-idle", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    need = ["plan", "arm_kind", "budget_min"] + ([] if a.dry else ["url", "arm", "key", "tokenizer"])
    missing = [n for n in need if getattr(a, n) is None]
    if missing:
        ap.error(f"missing {missing}")
    if a.dry:
        return dry(a)
    if not (a.arm.replace("-", "").replace("_", "").isalnum()):
        ap.error("--arm: letters, digits, - and _ only")
    return run_arm(a)


if __name__ == "__main__":
    sys.exit(main())
