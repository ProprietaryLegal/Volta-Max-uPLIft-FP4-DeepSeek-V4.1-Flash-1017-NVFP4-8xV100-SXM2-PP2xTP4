#!/usr/bin/env python3
# Public copy: the prompt wording was generalized for publication, so token counts differ by a few tokens from the measured ones.
"""ctx_tput.py URL CHARS OUT.jsonl [MAX_TOKENS]: decode speed at a given context length. One request: CHARS
characters of public text (Python standard-library source; 0 = no filler) followed by a legal-style drafting task
with invented facts; T=0, thinking off, ignore_eos, MAX_TOKENS (default 512) streamed tokens. Uses production's
bench5 client read-only (same timing as ab_t1_v2.py --tput: decode-only = first to last streamed token).
Prints one CTX line (numbers only) and appends the record (without text) to OUT.jsonl. Fails loudly."""
import os
import glob, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("ab_t1_v2", os.environ["PROD"] + "/1cat-1277/bench5/ab_t1_v2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
url, chars, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 512
TASK = ("Draft a memorandum of law in support of a motion for summary judgment in a civil contract case, "
        "addressing the undisputed facts, the governing standard and the damages claimed. Use generic party "
        "names and invented facts.")
txt = ""
if chars:
    for f in sorted(glob.glob("/usr/lib/python3.12/*.py")) + sorted(glob.glob("/usr/lib/python3.12/*/*.py")):
        txt += open(f, errors="replace").read()
        if len(txt) >= chars:
            break
    if len(txt) < chars:
        raise SystemExit(f"only {len(txt)} characters of source available, asked {chars}")
    content = ("Here is Python standard-library source code.\n\n" + txt[:chars]
               + "\n\nSet the code above aside. " + TASK)
else:
    content = TASK
body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": max_tokens,
        "ignore_eos": True, "chat_template_kwargs": {"thinking": False}}
a = mod.metrics(url)
s = mod.stream_chat(url, body)
b = mod.metrics(url)
n = int(s["usage"]["completion_tokens"]); ctx = int(s["usage"]["prompt_tokens"])
rounds = b["vllm:spec_decode_num_drafts_total"] - a["vllm:spec_decode_num_drafts_total"]
if s["first"] is None or s["last"] is None or rounds <= 0 or n < 2:
    raise SystemExit("no streamed token ids or no draft rounds: cannot time decode")
dec = s["last"] - s["first"]; ttft = s["first"] - s["t0"]
rec = {"ctx_tokens": ctx, "tokens": n, "rounds": rounds, "ttft_s": round(ttft, 3), "dec_s": round(dec, 3),
       "dec_ms_round": round(dec * 1e3 / rounds, 2), "dec_tok_s": round((n - 1) / dec, 2),
       "tok_round": round(n / rounds, 3)}
open(out, "a").write(json.dumps(rec) + "\n")
print(f"CTX {ctx} tokens: ttft {ttft:.1f} s, decode {rec['dec_tok_s']:.1f} tok/s, {rec['dec_ms_round']:.2f} dec-ms/round, "
      f"{rec['tok_round']:.2f} tok/round, {n} tokens out")
