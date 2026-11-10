#!/usr/bin/env python3
"""tput_prompts.py URL PROMPTS.txt OUT.jsonl: the bench5 ab_t1_v2.py --tput measurement (T=0, thinking off,
ignore_eos, 1024 tokens, streamed, decode-only ms/round) over the prompts in PROMPTS.txt (one per line) instead
of its three built-in prompts. Imports production's client read-only; prints numbers and shas only."""
import os
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ab_t1_v2", os.environ["PROD"] + "/1cat-1277/bench5/ab_t1_v2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
url, prompts_path, out = sys.argv[1:4]
prompts = [line.strip() for line in open(prompts_path) if line.strip()]
if not prompts:
    raise SystemExit("no prompts")
mod.TPUT_PROMPTS = prompts
mod.run_tput(url, len(prompts), out)
