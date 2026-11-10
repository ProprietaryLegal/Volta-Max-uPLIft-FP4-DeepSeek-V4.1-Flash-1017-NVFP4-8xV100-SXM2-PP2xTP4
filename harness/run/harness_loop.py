#!/usr/bin/python3 -I
"""harness_loop.py URL CHARS N OUT.jsonl [THINK=1] [TAIL_CHARS=12000] [CHATS=1] [BOUNDARY=0]

One turn shaped like an agent tool loop, in the Anthropic Messages API format
(POST /v1/messages, streamed, max_tokens 32000), a system prompt, tools, one long first user message, then a tool loop:
every assistant message is sent back with its blocks as produced (thinking, text, tool_use) and the next user message
carries the tool_result (TAIL_CHARS of new text). Public text only (Python standard-library source). N requests at
most; the loop also ends when the model stops calling the tool. Temperature 0 for repeatability. Records hold
numbers only, no text.

CHATS > 1: that many separate chats, each with its own text, sent strictly one request at a time in rotation
(A B C A B C ...): the test of whether a server keeps several chats' prompts at once. A follow-up counts as a HIT
when the kept tokens reach the chat's previous prompt length less 124 (the engine's reach, VLLM_V41_RING_REACH).

BOUNDARY=1 (one chat): after the tool loop ends, two more requests test the turn boundary. "turn2-keep": a new human
message with every earlier block sent back as produced. "turn3-strip": after that reply, another human message with
the thinking blocks of all earlier assistant messages removed (as a router does when it strips earlier thinking at a new human message).
The kept tokens of each say whether the server reused the prompt across a human message.

Through a router: HARNESS_LOOP_METRICS_URL names the model server whose counters to read when URL is a router in front
of it, HARNESS_LOOP_MODEL the model name to send, HARNESS_LOOP_HEADERS a JSON object of extra request headers.
HARNESS_LOOP_GAP_S: seconds to wait before every follow-up request (default 0), to tell a kept-prompt race from other
misses. HARNESS_LOOP_MAX_TOKENS: the max_tokens to send (default 32000).
HARNESS_LOOP_UNIQUE=1: start each chat's first message with a line naming this run, so a repeat of the script never sends
a chat the server has seen (a server that keeps earlier prompts can match an older copy of an identical chat).

Per request: prompt tokens, kept tokens (vllm:prefix_cache_hits delta), time to first streamed block, decode tok/s
(first to last streamed delta), dec-ms/round (drafts counter), wall tok/s.
"""
import glob, json, os, sys, time, urllib.request

url, chars, n_req, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
think = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
tail = int(sys.argv[6]) if len(sys.argv) > 6 else 12000
chats = int(sys.argv[7]) if len(sys.argv) > 7 else 1
boundary = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
if boundary and chats != 1:
    raise SystemExit("BOUNDARY=1 needs CHATS=1")
MODEL = os.environ.get("HARNESS_LOOP_MODEL", "deepseek-v4.1-flash")
METRICS_URL = os.environ.get("HARNESS_LOOP_METRICS_URL", url)
EXTRA_HEADERS = json.loads(os.environ.get("HARNESS_LOOP_HEADERS", "{}"))
GAP_S = float(os.environ.get("HARNESS_LOOP_GAP_S", "0"))
MAX_TOKENS = int(os.environ.get("HARNESS_LOOP_MAX_TOKENS", "32000"))
RUN_TAG = f"Run {time.time_ns()}.\n\n" if os.environ.get("HARNESS_LOOP_UNIQUE", "0") == "1" else ""

span = chars + tail * n_req
need = span * chats
txt = ""
for f in sorted(glob.glob("/usr/lib/python3.12/*.py")) + sorted(glob.glob("/usr/lib/python3.12/*/*.py")):
    txt += open(f, errors="replace").read()
    if len(txt) >= need:
        break
if len(txt) < need:
    raise SystemExit(f"only {len(txt)} characters of source available, asked {need}")

SYSTEM = ("You are a careful code reviewer working through a large body of Python standard-library source. "
          "You work in steps and use the tool you are given to fetch each further section. Never fetch two "
          "sections in one step.")
TOOLS = [{"name": "read_section", "description": "Return the text of one further section of the source by number.",
          "input_schema": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}]
TASK = (f"Above is the first part of the source. There are {n_req - 1} further sections, numbered 1 to {n_req - 1}. "
        "Fetch them one at a time with read_section, in order. After each one, note in two sentences what that "
        "section does, then fetch the next. After the last section, write a short memorandum (about 300 words) "
        "that summarizes the sections and names any function that looks error-prone.")


def metrics():
    want = ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:spec_decode_num_drafts",
            "vllm:num_requests_running")
    got = dict.fromkeys(want, 0.0)
    for line in urllib.request.urlopen(METRICS_URL + "/metrics", timeout=10).read().decode().splitlines():
        for w in want:
            if line.startswith(w + "_total{") or line.startswith(w + "{"):
                got[w] += float(line.rsplit(" ", 1)[1])
    return got


def call(messages):
    body = {"model": MODEL, "max_tokens": MAX_TOKENS, "stream": True, "temperature": 0, "system": SYSTEM,
            "tools": TOOLS, "messages": messages, "chat_template_kwargs": {"thinking": think}}
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "anthropic-version": "2023-06-01", **EXTRA_HEADERS})
    t0 = time.monotonic(); first = last = None; blocks = []; usage = {}; stop = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            t = ev.get("type")
            if t == "error":
                raise SystemExit(f"server error event: {ev}")
            if t == "message_start":
                usage.update(ev["message"].get("usage") or {})
            elif t == "content_block_start":
                b = dict(ev["content_block"])
                if b["type"] == "tool_use":
                    b["_json"] = ""
                blocks.append(b)
            elif t == "content_block_delta":
                now = time.monotonic(); first = first or now; last = now
                d = ev["delta"]; b = blocks[ev["index"]]
                if d["type"] == "thinking_delta":
                    b["thinking"] = b.get("thinking", "") + d["thinking"]
                elif d["type"] == "text_delta":
                    b["text"] = b.get("text", "") + d["text"]
                elif d["type"] == "input_json_delta":
                    b["_json"] += d["partial_json"]
                elif d["type"] == "signature_delta":
                    b["signature"] = b.get("signature", "") + d["signature"]
                else:
                    raise SystemExit(f"unknown delta type {d['type']}")
            elif t == "message_delta":
                usage.update(ev.get("usage") or {}); stop = ev["delta"].get("stop_reason")
    end = time.monotonic()
    for b in blocks:
        if b["type"] == "tool_use":
            b["input"] = json.loads(b.pop("_json") or "{}")
    return blocks, usage, stop, t0, first, last, end


def first_message(c):
    return [{"role": "user", "content": [
        {"type": "text", "text": RUN_TAG + "Here is Python standard-library source code.\n\n" + txt[c * span: c * span + chars]},
        {"type": "text", "text": TASK}]}]


recs = []


def step(label, c, messages, prev_ctx):
    if prev_ctx is not None and GAP_S > 0:
        time.sleep(GAP_S)
    a = metrics()
    if a["vllm:num_requests_running"] != 0:
        raise SystemExit(f"INCONCLUSIVE: server busy before request {label} of chat {c}")
    blocks, usage, stop, t0, first, last, end = call(messages)
    b = metrics()
    n = int(usage["output_tokens"]); ctx = int(usage["input_tokens"])
    kept = int(b["vllm:prefix_cache_hits"] - a["vllm:prefix_cache_hits"])
    rounds = b["vllm:spec_decode_num_drafts"] - a["vllm:spec_decode_num_drafts"]
    if first is None:
        raise SystemExit(f"request {label} chat {c}: nothing streamed (stop {stop})")
    dec = last - first
    hit = None if prev_ctx is None else kept >= prev_ctx - 124
    rec = {"req": label, "chat": c, "think": think, "ctx_tokens": ctx, "kept_tokens": kept, "hit": hit,
           "out_tokens": n, "stop": stop, "kinds": [x["type"] for x in blocks], "ttft_s": round(first - t0, 2),
           "dec_s": round(dec, 2), "dec_tok_s": round((n - 1) / dec, 2) if dec > 0 and n > 1 else None,
           "dec_ms_round": round(dec * 1e3 / rounds, 2) if rounds > 0 else None,
           "wall_s": round(end - t0, 2), "wall_tok_s": round(n / (end - t0), 2)}
    recs.append(rec)
    open(out, "a").write(json.dumps(rec) + "\n")
    print(f"REQ {label} chat {c} ctx {ctx} kept {kept} hit {hit} ttft {rec['ttft_s']} s out {n} decode "
          f"{rec['dec_tok_s']} tok/s {rec['dec_ms_round']} dec-ms/round wall {rec['wall_s']} s stop {stop} "
          f"blocks {'+'.join(rec['kinds'])}", flush=True)
    return rec, blocks, stop


state = [{"messages": first_message(c), "prev_ctx": None, "live": True} for c in range(chats)]
for i in range(n_req):
    for c, st in enumerate(state):
        if not st["live"]:
            continue
        rec, blocks, stop = step(i, c, st["messages"], st["prev_ctx"])
        st["prev_ctx"] = rec["ctx_tokens"]
        st["messages"] = st["messages"] + [{"role": "assistant", "content": blocks}]
        uses = [x for x in blocks if x["type"] == "tool_use"]
        if stop != "tool_use" or not uses:
            st["live"] = False
            continue
        lo = c * span + chars + i * tail
        st["messages"] = st["messages"] + [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u["id"],
             "content": txt[lo: lo + tail] if k == 0 else "One section per step."} for k, u in enumerate(uses)]}]
warm = [r for r in recs if r["hit"] is not None]
if warm:
    tt = sorted(r["ttft_s"] for r in warm); ds = [r["dec_tok_s"] for r in warm if r["dec_tok_s"]]
    firsts = [r for r in recs if r["hit"] is None]
    print(f"LOOP chats {chats}; first reads {[r['ttft_s'] for r in firsts]} s ({[r['ctx_tokens'] for r in firsts]} tokens); "
          f"follow-ups {len(warm)}: HITS {sum(1 for r in warm if r['hit'])} of {len(warm)}, ttft median "
          f"{tt[len(tt) // 2]} s, kept {sum(r['kept_tokens'] for r in warm) // len(warm)} tokens mean, "
          f"decode mean {sum(ds) / len(ds):.1f} tok/s")
if boundary:
    st = state[0]
    if st["live"]:
        raise SystemExit("INCONCLUSIVE: the tool loop did not end its turn; raise N so the model finishes")

    def human(text):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def stripped(messages):
        return [m if m["role"] != "assistant" else
                {"role": "assistant", "content": [x for x in m["content"] if x["type"] != "thinking"]} for m in messages]

    msgs = st["messages"] + [human("Thank you. In two sentences, which section was the longest?")]
    rec, blocks, stop = step("turn2-keep", 0, msgs, st["prev_ctx"])
    msgs = msgs + [{"role": "assistant", "content": blocks}]
    msgs3 = stripped(msgs) + [human("In two sentences, which section was the shortest?")]
    rec3, _, _ = step("turn3-strip", 0, msgs3, rec["ctx_tokens"])
    print(f"BOUNDARY keep: kept {rec['kept_tokens']} of {rec['ctx_tokens']} ttft {rec['ttft_s']} s hit {rec['hit']}; "
          f"strip: kept {rec3['kept_tokens']} of {rec3['ctx_tokens']} ttft {rec3['ttft_s']} s hit {rec3['hit']}")
