#!/usr/bin/env python3
"""longprompt.py URL CHARS [MAX_TOKENS [SKIP]]: one long public prompt (Python standard-library source, CHARS characters
after skipping SKIP characters, so a second prompt shares no cached prefix with the first)
with a short answer, to exercise long-context prefill and card memory. Prints numbers only (prompt tokens, wall
seconds, prompt tokens per second including the short decode). Fails loudly on any HTTP error."""
import glob, json, sys, time, urllib.request
url, chars = sys.argv[1], int(sys.argv[2])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32
skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0
txt = ""
for f in sorted(glob.glob("/usr/lib/python3.12/*.py")) + sorted(glob.glob("/usr/lib/python3.12/*/*.py")):
    txt += open(f, errors="replace").read()
    if len(txt) >= skip + chars:
        break
if len(txt) < skip + chars:
    raise SystemExit(f"only {len(txt)} characters of source available, asked {skip + chars}")
body = {"model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
        "messages": [{"role": "user", "content": "Here is Python standard-library source code.\n\n" + txt[skip:skip + chars]
                      + "\n\nIn one sentence, say what kind of code this is."}]}
t0 = time.time()
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=7200))
dt = time.time() - t0
u = r["usage"]
print(f"LONGPROMPT chars {chars} prompt_tokens {u['prompt_tokens']} completion_tokens {u['completion_tokens']} "
      f"wall {dt:.1f} s prompt-tok/s {u['prompt_tokens'] / dt:.0f} answer_chars {len(r['choices'][0]['message']['content'] or '')}")
