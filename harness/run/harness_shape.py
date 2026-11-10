#!/usr/bin/env python3
# Public copy: the prompt wording was generalized for publication, so token counts differ by a few tokens from the measured ones.
"""harness_shape.py URL CHARS N OUT.jsonl [THINK] [MAX_TOKENS]: a speed test shaped like an agent tool loop
(one long prompt, then many short tool-call requests). One chat: request 0 is a cold read of CHARS characters of public text (Python standard-library
source) plus a legal-style drafting task with invented facts; requests 1..N-1 append the previous answer and a
new user message (a follow-up instruction plus about 8,000 characters of new public text, standing for a tool
result). T=0, ignore_eos, MAX_TOKENS (default 300) streamed tokens each, THINK=1 turns thinking on (default 0).
Uses production's bench5 client read-only. Prints numbers only; OUT.jsonl gets one record per request, no text."""
import os
import glob, importlib.util, json, statistics, sys
spec = importlib.util.spec_from_file_location("ab_t1_v2", os.environ["PROD"] + "/1cat-1277/bench5/ab_t1_v2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
url, chars, n_req, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
think = len(sys.argv) > 5 and sys.argv[5] == "1"
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 300
TAIL = 8000
TASK = ("Set the code above aside. Draft a memorandum of law in support of a motion for summary judgment in a "
        "civil contract case, addressing the undisputed facts, the governing standard and the damages "
        "claimed. Use generic party names and invented facts.")
FOLLOW = ["Continue the memorandum with the argument on the undisputed facts.",
          "Now draft the section on the summary judgment standard and how it applies.",
          "Add the section on the damages claimed and the factors a court weighs.",
          "Draft a short proposed order granting the relief requested.",
          "Summarize the memorandum in five numbered points for the reader."]
need = chars + TAIL * n_req
txt = ""
for f in sorted(glob.glob("/usr/lib/python3.12/*.py")) + sorted(glob.glob("/usr/lib/python3.12/*/*.py")):
    txt += open(f, errors="replace").read()
    if len(txt) >= need:
        break
if len(txt) < need:
    raise SystemExit(f"only {len(txt)} characters of source available, asked {need}")
messages = [{"role": "user", "content": "Here is Python standard-library source code.\n\n" + txt[:chars] + "\n\n" + TASK}]
recs = []
for i in range(n_req):
    body = {"messages": messages, "temperature": 0, "max_tokens": max_tokens, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": think}}
    a = mod.metrics(url)
    s = mod.stream_chat(url, body, timeout=3600)
    b = mod.metrics(url)
    n = int(s["usage"]["completion_tokens"]); ctx = int(s["usage"]["prompt_tokens"])
    rounds = b["vllm:spec_decode_num_drafts_total"] - a["vllm:spec_decode_num_drafts_total"]
    if s["first"] is None or s["last"] is None or rounds <= 0 or n < 2:
        raise SystemExit(f"request {i}: no streamed token ids or no draft rounds: cannot time decode")
    dec = s["last"] - s["first"]; ttft = s["first"] - s["t0"]
    rec = {"req": i, "think": think, "ctx_tokens": ctx, "tokens": n, "rounds": rounds, "ttft_s": round(ttft, 3),
           "dec_s": round(dec, 3), "dec_ms_round": round(dec * 1e3 / rounds, 2),
           "dec_tok_s": round((n - 1) / dec, 2), "tok_round": round(n / rounds, 3),
           "wall_tok_s": round(n / (s["end"] - s["t0"]), 2)}
    recs.append(rec)
    open(out, "a").write(json.dumps(rec) + "\n")
    print(f"REQ {i} ctx {ctx} ttft {ttft:.1f} s decode {rec['dec_tok_s']:.1f} tok/s {rec['dec_ms_round']:.2f} dec-ms/round "
          f"{rec['tok_round']:.2f} tok/round wall {rec['wall_tok_s']:.1f} tok/s", flush=True)
    answer = "".join(s["content"]) if isinstance(s["content"], list) else s["content"]
    messages = messages + [{"role": "assistant", "content": answer or "(continued)"},
                           {"role": "user", "content": FOLLOW[i % len(FOLLOW)] + "\n\nNew material:\n\n"
                            + txt[chars + i * TAIL: chars + (i + 1) * TAIL]}]
warm = recs[1:]
if warm:
    print(f"HARNESS think={int(think)} ctx {recs[0]['ctx_tokens']}..{recs[-1]['ctx_tokens']} cold-read {recs[0]['ttft_s']:.1f} s; "
          f"warm requests {len(warm)}: ttft median {statistics.median(r['ttft_s'] for r in warm):.1f} s, "
          f"decode mean {statistics.mean(r['dec_tok_s'] for r in warm):.1f} tok/s, "
          f"{statistics.mean(r['dec_ms_round'] for r in warm):.2f} dec-ms/round, wall mean "
          f"{statistics.mean(r['wall_tok_s'] for r in warm):.1f} tok/s")
