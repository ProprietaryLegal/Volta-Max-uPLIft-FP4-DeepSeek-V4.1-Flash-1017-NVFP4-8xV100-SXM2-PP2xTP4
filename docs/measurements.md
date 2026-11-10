# Measurements

Every number here is MEASURED on 2026-11-09 unless its row or section gives another date or it is labelled
ESTIMATED (arithmetic shown) or UPSTREAM (the named source). The raw receipts are private build logs and are not
published; this page reduces them to tables and names the receipt each row comes from (paths under the build's
`logs/`).

## Results log

Newest first. A later arm is one more row at the top, with its details in a section below.

| Date | Build or profile | Test | Result | Receipt |
|---|---|---|---|---|
| 2026-11-10 | IQ records in prompt reading (plan 04 part B) at the accepted tiers: NOT accepted (quality not established) | first read of chat 0 (216,245 tokens) | booted with the accepted configuration's hot, warm and host experts, then ran out of card memory on stage 0 in the second chunk of the read (a request for 108 MiB with 64.75 MiB free). The server ended by itself. INFERENCE from the code, nothing run: both reading modes peak at the same point and the variant is short by the workspace it keeps from load. The fix is built and checked on one card of the test host (MEASURED 2026-11-10: results bit-equal to the previous file at four chunk sizes, peak of the reading call 1.30 GB to 0.67 GB at 8,188 rows). It has not yet run on eight cards and is not in this snapshot ([section](#iq-records-in-prompt-reading-plan-04-part-b-not-accepted-quality-not-established)) | private boot log, private one-card receipt |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 10 for, 2 against, and mean difference 5 for, 1 against, 6 undecided ([section](#quality-on-twelve-prompts-inconclusive)) | `t3w/quality-wide-score-N2w-PIQAw-Pw-N2wr.out` |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | termination test, chats 1 and 2 (198,307 and 204,871 tokens), second boot | both final answers ended normally (749 and 1,491 tokens). First reads 102.12 s and 103.0 s. 7 of 7 follow-ups kept the prompt in each chat, first token median 4.42 s and 4.26 s. With chat 0 the variant stands at 1 of 3 at the limit (95 % Wilson interval 0.061 to 0.792) ([section](#iq-records-in-prompt-reading-plan-04-part-b-not-accepted-quality-not-established)) | `baseline/W5-piqa-term-c12.out`, `baseline/W5-piqa-term.jsonl` |
| 2026-11-10 | accepted tiers under the load setting (a profile that pins them, not in this snapshot) | termination test, chat 0 (216,245 tokens) | the same hot, warm and host experts and tier digests as the accepted configuration. First read 167.64 s. The final answer ended normally with 6,417 tokens at 66.97 tok/s, and all eight requests gave the token counts of the two earlier runs ([section](#long-thinking-answers-the-termination-test)) | `baseline/W5-t3so-term-c0.out`, `baseline/W5-t3so-term.jsonl` |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | termination test, chat 0 (216,245 tokens) | first read 107.82 s. 7 of 7 follow-ups kept the prompt, first token 4.02 to 5.38 s (median 4.44 s). Tool-call requests 49.4 to 62.6 tok/s. The final answer ran to the 32,000-token limit in thinking, a repetition loop (5.33 tokens per round) ([section](#iq-records-in-prompt-reading-plan-04-part-b-not-accepted-quality-not-established)) | `baseline/W4-piq-so-term-c0.out`, `baseline/W4-piq-so-term.jsonl` |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B): NOT accepted (quality not established) | load failure, and an eight-card boot with the allocator setting off | explained and fixed: the allocator setting during weight loading fills the cards, the driver error is reproduced on one card, and with the setting off the variant was ready in 349 s with no error. That the first boot failed in exactly this way is an inference. Not in this snapshot | private one-card receipts and boot log |
| 2026-11-10 | three slots on the accepted tiers: NOT accepted (quality not established) | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 10 for, 2 against, and mean difference 7 for, 0 against, 5 undecided ([section](#quality-on-twelve-prompts-inconclusive)) | `t3w/quality-wide-score-N2w-S3hw-Pw-N2wr.out` |
| 2026-11-10 | accepted configuration | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 11 for, 1 against, and mean difference 7 for, 0 against, 5 undecided ([section](#quality-on-twelve-prompts-inconclusive)) | `t3w/quality-wide-score-N2w-T3w-Pw-N2wr.out` |
| 2026-11-10 | accepted configuration | termination test, three chats of 198,307 to 216,245 tokens | 3 of 3 final answers ended normally (6,417, 1,499 and 1,642 tokens). 0 of 3 at the limit, 95 % Wilson interval 0.000 to 0.561, so the rate is bounded only below 0.56 ([section](#long-thinking-answers-the-termination-test)) | `baseline/W4-t3a-term-c0.out`, `baseline/W4-t3a-term-c12.out`, `baseline/W4-t3a-term.jsonl` |
| 2026-11-10 | this build, six configurations | chat 0 of the termination test (the 216K tool loop) | ended normally in the accepted configuration (three runs). Ran to the 32,000-token limit on chat 0 in the five boots with a lower share of routes on official-precision experts, none accepted, quality not established. One of the five, the part B variant, ended chats 1 and 2 normally (rows above), so this holds for chat 0 only. Cause not established ([section](#long-thinking-answers-the-termination-test)) | the receipts named in that section |
| 2026-11-10 | Volta sPLIt | termination test, the same three chats | 3 of 3 final answers ended normally (10,013, 732 and 1,812 tokens). 0 of 3 at the limit, 95 % Wilson interval 0.000 to 0.561, so the rate is bounded only below 0.56 | `baseline/W4-prod-term.out`, `baseline/W4-prod-term.jsonl` |
| 2026-11-10 | Volta sPLIt | a separate 31,173-token chat, two tool-call requests, thinking on | the final answer ran to the 32,000-token limit in thinking, a repetition loop (longest repeated span 18,185 of 23,336 text units), at 210.24 tok/s, 4.9 tokens per round | `baseline/W4-smoke-prod.out`, `baseline/W4-smoke-prod.jsonl` |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B) | first eight-card boot | built and passed its one-card checks. The boot failed while loading (driver error "invalid argument" during the build of the IQ bank). Explained and fixed later the same day (rows above). Not in this snapshot | private boot log |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | closeness, three prompts | INCONCLUSIVE by the rule | `t3/quality-score-N2-W3s3h-P-N2r.out` |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | tool loop, 216,255-token prompt (unique chat), thinking on | first read 173.54 s; 7 of 7 follow-ups kept the whole prompt; follow-up first token median 9.62 s; decode mean 64.8 tok/s; the final request ran to the 32,000-token limit in thinking (5.16 tokens per round) | `baseline/W3-s3h-loop-think1.jsonl` |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | 257,741-token prompt; card memory | read 207.1 s, decode 60.8 tok/s; highest card memory 31,704 MiB | `baseline/W3-s3h-ctx.out`, `W3-s3h-mem.txt` |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | three chats taking turns | 6 of 6 follow-ups kept their own chat's prompt, first token median 7.65 s; final answers ended normally (981 to 1,888 tokens) | `baseline/W3-s3h-rotate3.jsonl` |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | short sets, two runs; loader | general 50.9 / 51.3 tok/s, legal-style 63.9 / 64.2 tok/s; warm 5.84 % / 12.54 %; ready in 374 s | `baseline/W3-s3h-g1-tput.out`, `-g2-`, `-l1-`, `-l2-`, `baseline/W3-s3h-arm.out` |
| 2026-11-10 | three-tier, one slot, reserves 5.3 / 8.1 GiB, traced | tool loop, 216,245-token prompt, thinking on; prefill trace | first read 171.92 s; 7 of 7 kept; follow-up first token median 9.34 s; decode mean 65.4 tok/s (traced: slowed); the final request ran to the 32,000-token limit in thinking (5.21 tokens per round); 162.5 experts staged per layer call, staging 0.1485 of 0.1731 s per call (86 %) | `baseline/W3-ns-tr-loop-think1.jsonl` |
| 2026-11-09 | Volta sPLIt | short tool loop (31,164-token chat, three steps), ten runs | seven runs of the identical chat: 3, 2, 2, 2, 1, 0 and 0 of 3 follow-ups kept their prompt (run 2 through our router; a 3-second pause in runs 5 and 7 changed nothing); three runs with a unique first line: 3 of 3 each. HYPOTHESIS: the engine matches an older copy of the same chat, so every repeat of a loop test must be a unique chat. Final request: the 32,000-token limit once (the router run, 5.6 tokens per round); a 3,000-token cap reached in 5 of 7 later runs (long thinking on this prompt is common at temperature 0; the cap does not show a runaway) | `baseline/prod-boundary-20261109.jsonl`, `prod-router-boundary-20261109.jsonl` (the run to the limit), `prod-boundary-r2-20261109.jsonl`, `prod-gap-{g0a,g3a,g0b,g3b,u1,u2,u3}-20261109.jsonl` |
| 2026-11-09 | Volta sPLIt | turn boundary, 31,164-token chat | kept: 38,800 of 38,845, first token 0.54 s; stripped: 0 of 38,193, 14.07 s; through our router after its change: 69,728 of 69,772, 1.45 s | `baseline/prod-boundary-20261109.jsonl`, `baseline/prod-router-boundary-20261109.jsonl` |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | tool loop repeat, 216,245-token prompt | six tool-call requests 12.6 to 13.5 s each, 92.2 to 99.6 tok/s; the final request again ran to the 32,000-token limit (5.87 tokens per round) | `baseline/t3-S3n-loop-think1.jsonl` |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | closeness, three prompts | INCONCLUSIVE by the rule | `t3/quality-score-N2-T3nS-P-N2r.out` |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | 257,706-token prompt | 214.4 s (1,202 tok/s); highest card memory 31,400 / 30,118 MiB | `baseline/t3-S3n-long250k.out` |
| 2026-11-09 | three prompt slots (no host tier) | three chats taking turns | 6 of 6 follow-ups kept their own chat's prompt; final answers ended normally | `baseline/t3-S3n-rotate3.jsonl` |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | short sets, two runs; loader | general 86.8 / 87.3 tok/s, legal-style 67.8 / 68.1 tok/s; warm 7.23 % / 18.02 %; ready in 334 s | `baseline/t3-S3n-g1-tput.out`, `-g2-`, `-l1-`, `-l2-`, `baseline/t3-S3n-arm.out` |
| 2026-11-09 | no host tier (`hot_iq`), quality NOT established | tool loop, 216,245-token prompt, thinking on | first read 169.7 s; six tool-call requests 12.3 to 13.5 s each, 58.8 to 105.0 tok/s; final request did not finish (32,000 tokens of thinking, token limit, 5.66 tokens per round); closeness check not run | `baseline/t3-T3n-loop-think1.jsonl` |
| 2026-11-09 | no host tier (`hot_iq`), quality NOT established | short sets, two runs; loader | general 86.3 / 86.2 tok/s (33.36 / 33.37 ms), legal-style 66.9 / 71.9 tok/s; warm tier 4.74 % / 13.53 % of routes; ready in 389 s | `baseline/t3-T3n-g1-tput.out`, `-g2-`, `-l1-`, `-l2-`, `baseline/t3-T3n-arm.out` |
| 2026-11-09 | two-tier on the current code | short set, general | 18.3 tok/s over the whole request, with 178.95 ms per round and 3.32 tokens per round in decode | `baseline/t3-N2a-g1-tput.out` |
| 2026-11-09 | three-tier and Volta sPLIt against two-tier | closeness, three prompts | PASS ([Quality](#quality-three-tier-against-two-tier)) | `t3/quality-score-N2-T3-P-N2r.out` |
| 2026-11-09 | three-tier | turn boundary, 31,164-token chat | earlier thinking kept: 38,432 of 38,469 kept, first token 1.11 s; stripped: 0 of 38,396, 33.08 s | `baseline/t3-T3a-boundary.jsonl` |
| 2026-11-09 | three-tier | tool loop, 216,245-token prompt, thinking on | first read 164.0 s; 7 of 7 follow-ups kept the whole prompt; 12.5 to 15.0 s per tool-call request | `baseline/t3-T3a-loop-think1.jsonl` |
| 2026-11-09 | three-tier | context rows, 46 to 257,741 tokens | 60.9 to 75.9 tok/s; read 28.5 s (30,795), 83.1 s (106,531), 194.2 s (257,741) | `baseline/t3-T3a-ctx.jsonl` |
| 2026-11-09 | three-tier | short sets, two runs | general 50.6 / 51.8 tok/s, legal-style 56.9 / 58.4 tok/s | `baseline/t3-T3a-g1-tput.out`, `-g2-`, `-l1-`, `-l2-` |
| 2026-11-09 | two-tier, Volta sPLIt | short sets, context rows, long prompts, harness-shaped test | the sections below | private boot receipts of the same day |

## How it was measured

- **Host and builds.** The 8-card host (8x V100-SXM2-32GB, two NVLink boards, PP2 x TP4). One build served at a time,
  and all three were measured on the same day with the same prompts.
- **One request at a time**, greedy (T=0), streamed. A round is one verify pass, counted from the server's draft
  counters. "ms per round" is always decode time (first to last streamed token) divided by rounds.
- **Two measures of tok/s.** On the short sets, tok/s is the output tokens divided by the whole request time, the
  wait for the first token included. That is what `ab_t1_v2.py --tput` prints as its total. Tokens per round there
  is output tokens divided by rounds. So a short-set tok/s is lower than tokens per round divided by ms per round.
  Example (three-tier, general set, run 1): 3,072 tokens in 60.73 s is 50.6 tok/s over the three requests.
  The decode figures, 3.15 tokens per round at 59.38 ms per round, give 53.0 tok/s (arithmetic).
  The difference is mostly the three requests' wait for the first token (0.84 to 1.03 s each). On the context
  rows, the harness-shaped tests and the tool loop, tok/s is decode only: output tokens after the first, divided
  by the time from the first to the last streamed token. There tok/s x ms per round gives the tokens per round.
- **Short sets** (`harness/run/measure_boot.sh`): the Volta sPLIt harness's `ab_t1_v2.py --tput` with its three
  general public prompts (an essay, a hash table, a Python module), and the same measurement over three generic
  legal-style public prompts (`harness/run/tput_prompts.py`; the prompt file is not published). Thinking off,
  `ignore_eos`, 1,024 tokens out. Run 1 follows a warm-up; run 2 repeats in the same boot without one.
- **Context rows** (`harness/run/ctx_tput.py`): a legal-style memorandum task after public filler text (Python
  standard-library source), 512 tokens out. "Read" is the time to the first token of a prompt nobody read before.
- **Long prompts** (`harness/run/longprompt.py`): public Python source, 48 tokens out, card memory sampled every 15 s
  (`harness/run/memwatch.sh`).
- **Harness-shaped tests**: `harness/run/harness_shape.py` (one chat, a 216K-token public prompt, thinking on, 300
  tokens, each follow-up adding the answer and 8,000 characters) and `harness/run/harness_loop.py` (a tool loop in
  the Anthropic Messages API format).
- **Termination test** (2026-11-10, program not in this snapshot): one chat is the chat of
  `harness/run/harness_loop.py`, the 216K tool loop: a first read, six tool-call requests, then the final request
  with thinking on, temperature 0 and a 32,000-token limit, one request at a time. Chat 0 is byte for byte that
  loop's chat. Chat 1 and chat 2 take later sections of the same public source text. The outcome of a chat is
  ENDED (the final request stopped normally) or LIMIT (it stopped at the token limit). The test keeps the answer in
  memory only and writes numbers: of the answer cut into text units (a word or one punctuation character), the
  share of distinct 20-unit windows and the length of the longest span that occurs at least twice. The summary
  gives the 95 % Wilson interval of the share of chats at the limit.
- **Closeness on twelve prompts** (2026-11-10, INCONCLUSIVE, program not in this snapshot): the request and
  the per-prompt figures are those of `quality_t3.py` on a wider set of prompts invented for the check (code 3, JSON
  2, general prose 3, legal-style prose 2, reasoning 2). The receipts hold token ids and log-probabilities only.
- **Single-card checks** ran on one card of a separate 4-card V100 test host at a 150 W cap, with no other compute
  process (each check refuses to run otherwise).

## Short sets

In this table tok/s is over the whole request, the first-token wait included. The ms per round is decode only, and
tokens per round is output tokens divided by rounds. So tok/s x ms per round gives less than the last column
([How it was measured](#how-it-was-measured)).

| Build | Set | Run 1: ms per round (decode), tok/s (whole request) | Run 2: the same | Tokens per round |
|---|---|---|---|---|
| Volta sPLIt | general | 21.48 ms, 136.9 tok/s | 21.57 ms, 137.9 tok/s | |
| Volta sPLIt | legal-style | 21.54 ms, 118.7 tok/s | 21.62 ms, 118.3 tok/s | |
| Two-tier | general | 174.64 ms, 18.8 tok/s | 163.37 ms, 20.1 tok/s | 3.32 |
| Two-tier | legal-style | 87.58 ms, 28.2 tok/s | 81.79 ms, 30.2 tok/s | 2.52 |
| Three-tier | general | 59.38 ms, 50.6 tok/s | 57.95 ms, 51.8 tok/s | 3.15 |
| Three-tier | legal-style | 42.16 ms, 56.9 tok/s | 41.02 ms, 58.4 tok/s | 2.55 |
| No host tier (`hot_iq`), quality not established | general | 33.36 ms, 86.3 tok/s | 33.37 ms, 86.2 tok/s | 3.15 |
| No host tier (`hot_iq`), quality not established | legal-style | 35.99 ms, 66.9 tok/s | 33.27 ms, 71.9 tok/s | 2.61 |
| No host tier, three slots, not accepted (quality not established) | general | 33.05 ms, 86.8 tok/s | 32.88 ms, 87.3 tok/s | 3.19 |
| No host tier, three slots, not accepted (quality not established) | legal-style | 33.14 ms, 67.8 tok/s | 32.96 ms, 68.1 tok/s | 2.46 |

Within-boot noise on two-tier was 6.5 %. A later two-tier boot on the current code gave 18.3 tok/s (178.95 ms per
round) on the general set (`baseline/t3-N2a-g1-tput.out`), 2.5 % per round from run 1 above.

Time to first token on the short sets: Volta sPLIt 0.37 to 0.45 s, two-tier 0.69 s, three-tier 0.84 to 1.07 s, no
host tier (not accepted: quality not established) 1.04 to 1.18 s.

## Context

| Context tokens | Volta sPLIt | Two-tier | Three-tier |
|---|---|---|---|
| 46 | 125.9 tok/s | 33.7 tok/s (79.07 ms) | 75.9 tok/s (37.81 ms) |
| 30,795 | 103.2 tok/s, read 11.8 s | 30.6 tok/s (82.54 ms), read 21.7 s | 68.1 tok/s (41.25 ms), read 28.5 s |
| 106,531 | 101.0 tok/s, read 38.2 s | 31.5 tok/s (84.16 ms), read 67.9 s | 60.9 tok/s (41.37 ms), read 83.1 s |
| 257,741 | 98.1 tok/s (26.85 ms), read 88.7 s | | 66.1 tok/s (43.9 ms), read 194.2 s |
| 303,004 | | 30.2 tok/s (87.19 ms), read 190.0 s | |

Prompt reading from these rows, in prompt tokens per second (arithmetic): Volta sPLIt 2,610 / 2,789 / 2,906;
two-tier 1,419 / 1,569 / 1,595 (303,004 tokens); three-tier 1,080 / 1,282 / 1,327. The tool loop's first read of
216,245 tokens ([Harness-shaped tests](#harness-shaped-tests)) ran at 2,946 tok/s on Volta sPLIt and about 1,319
on three-tier. The README's prompt-reading row covers these context rows and that first read: 2,610 to 2,946,
1,419 to 1,595 and 1,080 to 1,327 tok/s. Speed tests stop at about 250K tokens of context. The longest prompt
used since is 257,741 tokens. The two-tier row of 303,004 tokens was run before that limit.

## Long prompts and memory (two-tier)

| Boot | Card reserve (stage 0 / 1) | Prompt | Time | Notes |
|---|---|---|---|---|
| Calibration | 6 / 9 GiB | 106,496 tokens | 74.6 s (1,428 tok/s) | peak card memory 30,456 / 30,304 MiB |
| Calibration | 6 / 9 GiB | 498,099 tokens | 274.9 s | no out-of-memory error |
| Final | 4.5 / 7.5 GiB | 104,049 tokens | 72.2 s (1,441 tok/s) | |
| Final | 4.5 / 7.5 GiB | 498,099 tokens | 326.6 s (1,525 tok/s) | peak card memory 32,086 / 31,356 MiB, limit about 32,494 |

An earlier boot with one 6 GiB reserve for both stages ran out of memory on a 100K-token prompt on stage 1, which
holds the drafter and the head. KV pool on every boot: 2,076,870 tokens at max-model-len 524,288 (the engine reports
3.96x concurrency for 524,288-token requests).

## Residency at load

| Mode | Stage | Free per card after dense load | Expert budget | Hot | Warm | Host | Free after |
|---|---|---|---|---|---|---|---|
| Two-tier | 0 (layers 0-19) | 29.30 GiB | 24.80 GiB | 5,463 | | 2,217 | 3.81 GiB |
| Two-tier | 1 (layers 20-39) | 28.86 GiB | 21.36 GiB | 4,704 | | 2,976 | 6.89 GiB |
| Three-tier | 0 | 29.30 GiB | 24.80 GiB | 4,107 | 3,213 | 360 | 3.78 GiB |
| Three-tier | 1 | 28.86 GiB | 21.36 GiB | 2,695 | 4,379 | 606 | 7.15 GiB |
| No host tier (`hot_iq`) | 0 | | | 3,843 | 3,837 | 0 | |
| No host tier (`hot_iq`) | 1 | | | 2,184 | 5,496 | 0 | |
| No host tier, three slots | 0 | | | 3,334 | 4,346 | 0 | |
| No host tier, three slots | 1 | | | 1,779 | 5,901 | 0 | |

Counts are expert slices per rank set (each of the four TP ranks of a stage holds the same experts); 7,680 per stage.
Decode route shares by the private tally (aggregates only): two-tier host tier 0.841 % (stage 0) and 1.392 % (stage 1);
three-tier hot / warm / host 96.26 / 3.74 / 0.00 % (stage 0) and 90.71 / 9.29 / 0.00 % (stage 1); no host tier: warm
4.74 % (stage 0) and 13.53 % (stage 1); no host tier with three slots (reserves 5.3 / 8.1 GiB): warm 7.23 % and 18.02 %. Boot to ready: 288 s
(two-tier), 307 s (three-tier), 389 s (no host tier), 334 s (no host tier, three slots).

## Harness-shaped tests

Two-tier, one chat, 216K-token public prompt, thinking on, 300 tokens per request:

| Request | Context tokens | First token | Decode |
|---|---|---|---|
| 0 | 215,901 | 136.0 s | 38.6 tok/s |
| 1 | 217,985 | 134.3 s | 30.6 tok/s |
| 2 | 219,903 | 137.3 s | 31.4 tok/s |
| 3 | 221,827 | 140.0 s | 43.9 tok/s |

Every follow-up re-read the whole prompt. That is the test's own doing, not an engine defect: the engine keeps a
prompt only when the next request repeats the previous sequence, generated tokens included, to within 124 tokens of
its end, and this test drops the reasoning of earlier turns. Volta sPLIt on the same test also re-read (first token
78.0, 74.9 and 76.5 s).

Volta sPLIt, tool loop (`harness_loop.py`, Anthropic Messages API format, thinking on, 8 requests): first read 216,245 tokens in
73.4 s (2,946 tok/s); the 7 follow-ups at 219K to 238K tokens kept 216,368 to 235,264 tokens, first token 3.34 to
3.75 s, decode 112 to 138 tok/s (26.1 to 30.4 ms per round), about 4.6 s per tool-call request.

Three-tier, tool loop (same test: `harness_loop.py`, Anthropic Messages API format, thinking on, a 216,245-token first prompt of
public source text, then seven follow-ups: six tool-call requests and the final answer; receipt
`baseline/t3-T3a-loop-think1.jsonl`): first read 216,245 tokens in 163.97 s (1,319 tok/s, arithmetic); all seven
follow-ups kept the whole previous prompt (216,224 to 235,776 tokens kept). The six tool-call requests: first token
8.72 to 9.40 s, decode 47.0 to 64.9 tok/s (60.5 to 78.2 ms per round), 12.5 to 15.0 s wall each. The final answer:
6,417 tokens at 66.1 tok/s (62.8 ms per round), first token 8.74 s. Tokens per round across the seven follow-ups:
3.4 to 4.3 (arithmetic: tok/s x ms per round). Against Volta sPLIt on the same test: 2.7 to 3.3 times the wall time
per tool-call request (arithmetic: 12.5 / 4.6 and 15.0 / 4.6).

Three-tier, turn boundary (`harness_loop.py` `BOUNDARY=1`, a 31,164-token chat, then a new human message; receipt
`baseline/t3-T3a-boundary.jsonl`):

| New human message | Prompt tokens kept | First token |
|---|---|---|
| earlier thinking sent back unchanged | 38,432 of 38,469 | 1.11 s |
| earlier thinking stripped from the conversation | 0 of 38,396 | 33.08 s |

Volta sPLIt, turn boundary (the same test at temperature 0, thinking on, 2026-11-09 23:38 to 23:46Z; receipts
`baseline/prod-boundary-20261109.jsonl` and, through our router after its change, `baseline/prod-router-boundary-20261109.jsonl`):

| New human message | Prompt tokens kept | First token |
|---|---|---|
| earlier thinking sent back unchanged | 38,800 of 38,845 | 0.54 s |
| earlier thinking stripped from the conversation | 0 of 38,193 | 14.07 s |
| through our router after its change | 69,728 of 69,772 | 1.45 s |

The engine keeps a prompt only when the new prompt reproduces the previous sequence to within 124 tokens of its end,
so the program sending the requests must send earlier thinking back unchanged. Caution on these short loops: on Volta
sPLIt, re-sending the identical chat made follow-ups re-read their whole prompt (MEASURED 2026-11-09 23:38 to 23:54Z,
direct unless noted). In run order, 3, 2, 2, 2, 1, 0 and 0 of 3 follow-ups kept their prompt; run 2 went through our
router, and a 3-second pause before each follow-up (runs 5 and 7) changed nothing. With a unique first line per run
(`HARNESS_LOOP_UNIQUE=1`), 3 of 3 in each of three runs. HYPOTHESIS: the engine matches an older copy of the same
chat. Method rule: every repeat of a loop test must be a unique chat. The earlier builds' loops in this repository
each ran once per boot, so they are not affected. The final request ran to the 32,000-token limit in thinking once
(the router run, 5.6 tokens per round); the seven later runs had a 3,000-token cap, reached in 5 of 7. That supports
only this: long thinking on this prompt is common on Volta sPLIt at temperature 0. A 3,000-token cap does not show a
runaway.

### No host tier (not accepted: quality not established)

No host tier (`hot_iq`), tool loop (same test; receipt `baseline/t3-T3n-loop-think1.jsonl`): first read 216,245
tokens in 169.71 s (1,274 tok/s, arithmetic); all seven follow-ups kept the whole previous prompt (216,224 to 235,808
tokens). The six tool-call requests: first token 9.45 to 10.11 s, decode 58.8 to 105.0 tok/s (39.5 to 55.1 ms per
round), 12.3 to 13.5 s wall each. The final request did not finish: 32,000 tokens of thinking, stopped at the token
limit (226.4 s), at 147.4 tok/s and 38.42 ms per round, so 5.66 tokens per round (arithmetic). The same request
ended normally in the first run on the host-tier profile (6,417 tokens) and on Volta sPLIt (10,013 tokens). All
three ran at temperature 0. The test keeps no output text, and the cause is not established. The closeness check
was not run on this one-slot profile.

No host tier with three slots, tool loop repeated (receipt `baseline/t3-S3n-loop-think1.jsonl`): first read 216,245
tokens in 179.81 s (1,203 tok/s, arithmetic); all seven follow-ups kept the whole previous prompt (216,224 to 235,904
tokens). The six tool-call requests: first token 9.89 to 10.68 s, decode 92.2 to 99.6 tok/s (38.8 to 40.9 ms per
round), 12.6 to 13.5 s wall each. The final request AGAIN ran to the 32,000-token limit in thinking (218.0 s, 153.8
tok/s, 38.15 ms per round: 5.87 tokens per round, arithmetic). That is two of two on the no-host-tier profiles.
The verdict of these profiles stays: not accepted, quality not established (closeness INCONCLUSIVE).

Full count on the 216K tool loop, all at temperature 0. The final request ended normally on Volta sPLIt (10,013
tokens, twice) and in the accepted configuration (6,417 tokens, twice). It ran to the 32,000-token limit in
thinking in both no-host-tier runs (5.66 and 5.87 tokens per round, quality not established) and in three runs with
the host tier and reserves of 5.3 / 8.1 GiB (2026-11-10: 5.21, 5.16 and 5.33 tokens per round, none accepted,
quality not established). The cause is not established. The count by configuration, with the two other chats of the
termination test, is in [Long thinking answers](#long-thinking-answers-the-termination-test). On Volta sPLIt, the
final request of the shorter turn-boundary loop above also ran to that limit in one of three runs with the
32,000-token limit.

## Host tier, one slot, reserves 5.3 / 8.1 GiB (traced)

Profile `serve-nvfp4-pp2-t3-a1r.yaml`: the accepted profile's mode (host tier, one slot) with the reserves of the
three-slot profile, 5.3 / 8.1 GiB per card (the accepted profile has 4.5 / 7.5). It is the three-slot profile's
one-slot control, not the accepted configuration, and its quality is not established. Hot / warm / host experts:
stage 0 3,801 / 3,519 / 360; stage 1 2,450 / 4,624 / 606. Warm decode routes by the tally: 4.92 % (stage 0) and
11.16 % (stage 1). MEASURED 2026-11-10, public text, temperature 0, receipt `baseline/W3-ns-tr-loop-think1.jsonl`.
The prefill trace was on. It synchronizes the card at every mark, so every speed in this section is slowed by the
trace and does not compare with untraced rows.

216K tool loop (unique chat, thinking on): first read 216,245 tokens in 171.92 s; 7 of 7 follow-ups kept the whole
previous prompt. The six tool-call requests: first token 9.08 to 9.82 s, decode 59.8 to 62.8 tok/s, 13.2 to 14.6 s
wall each; over all seven follow-ups the first-token median is 9.34 s and the decode mean 65.4 tok/s. The final
request ran to the 32,000-token limit in thinking (362.0 s wall, 90.66 tok/s, 57.45 ms per round: 5.21 tokens per
round, arithmetic).

Prefill trace (3,600 expert-layer calls, 6,130 rows per call on average): 162.5 experts staged from host RAM per
layer call; staging took 0.1485 s of 0.1731 s per call, so host staging is 86 % of the expert time in prompt reading
(arithmetic).

## Three prompt slots on the accepted tiers (not accepted: quality not established)

Profile `serve-nvfp4-pp2-t3-a3.yaml`: the accepted three-tier mode with `max-num-seqs 3` and reserves 5.3 / 8.1
GiB, trace off. Ready in 374 s. Hot / warm / host experts: stage 0 3,597 / 3,723 / 360; stage 1 2,290 / 4,784 /
606. Warm decode routes by the tally: 5.84 % (stage 0) and 12.54 % (stage 1). MEASURED 2026-11-10, public text,
temperature 0; receipts `baseline/W3-s3h-*`.

| | Three slots on the accepted tiers | Accepted profile, one slot (2026-11-09) |
|---|---|---|
| Short set, general (two runs) | 50.9 / 51.3 tok/s (59.77 / 59.34 ms per round) | 50.6 / 51.8 tok/s (59.38 / 57.95 ms) |
| Short set, legal-style (two runs) | 63.9 / 64.2 tok/s (39.19 / 38.98 ms per round) | 56.9 / 58.4 tok/s (42.16 / 41.02 ms) |
| Three chats taking turns (30,439 to 31,174 tokens each, unique chats, thinking on) | first reads 29.72 to 31.17 s; 6 of 6 follow-ups kept their own chat's prompt; first token 7.11 to 7.85 s (median 7.65 s); all three final answers ended normally (981 to 1,888 tokens) | not run |
| 257,741-token prompt | read 207.1 s; decode 60.8 tok/s (46.96 ms per round) | 194.2 s; 66.1 tok/s |
| 216K tool loop (unique chat, 216,255 tokens) | first read 173.54 s; 7 of 7 kept; six tool-call requests: first token 9.32 to 10.1 s, 53.8 to 64.5 tok/s, 12.5 to 14.1 s each; final request ran to the 32,000-token limit in thinking (356.7 s, 92.13 tok/s, 55.98 ms per round: 5.16 tokens per round, arithmetic) | first read 164.0 s; final request ended normally (6,417 tokens) |
| Highest card memory | 31,704 MiB (receipt `baseline/W3-s3h-mem.txt`) | not compared |
| Closeness check against two-tier | INCONCLUSIVE by the rule | PASS |

Closeness (receipt `t3/quality-score-N2-W3s3h-P-N2r.out`): the divergence part holds on three of three prompts
(first different token 25 / 56 / 12 against Volta sPLIt's 0 / 10 / 5). The mean part has one prompt below Volta
sPLIt (prompt 3: 0.0392 against 0.1429), one above (prompt 2: 0.0565 against 0.0392) and one undecidable (prompt 1:
Volta sPLIt has no common prefix), so the rule cannot decide. The pattern is the same as on the no-host-tier
profile with three slots. Three prompts are too few to decide. The wider set of 2026-11-10 is INCONCLUSIVE too
([Quality on twelve prompts](#quality-on-twelve-prompts-inconclusive)). The accepted configuration stays the
one-slot three-tier profile.

## Long thinking answers: the termination test

MEASURED 2026-11-10 on the 8-card host, one request at a time, public text, thinking on, temperature 0, a
32,000-token limit on the final request ([How it was measured](#how-it-was-measured)). Each chat is a first read,
six tool-call requests and the final answer. Receipts: `baseline/W4-prod-term.out`, `baseline/W4-t3a-term-c0.out`,
`baseline/W4-t3a-term-c12.out` and their `.jsonl` files.

| Chat, first prompt | Build | First read | Follow-ups: first token | Final answer | Distinct 20-unit windows | Longest repeated span |
|---|---|---|---|---|---|---|
| 0, 216,245 tokens | Volta sPLIt | 73.61 s | 3.33 to 3.81 s | ended normally: 10,013 tokens at 137.72 tok/s | 0.963 | 143 of 9,386 units |
| 0, 216,245 tokens | accepted configuration | 164.9 s | 8.76 to 9.43 s | ended normally: 6,417 tokens at 67.45 tok/s | 0.9938 | 47 of 5,012 units |
| 1, 198,307 tokens | Volta sPLIt | 69.96 s | 3.34 to 3.65 s | ended normally: 732 tokens at 122.51 tok/s | 1.0 | 5 of 631 units |
| 1, 198,307 tokens | accepted configuration | 155.56 s | 8.9 to 9.51 s | ended normally: 1,499 tokens at 56.92 tok/s | 1.0 | 18 of 1,348 units |
| 2, 204,871 tokens | Volta sPLIt | 72.35 s | 3.29 to 3.63 s | ended normally: 1,812 tokens at 124.48 tok/s | 0.9818 | 31 of 1,609 units |
| 2, 204,871 tokens | accepted configuration | 161.39 s | 8.79 to 9.12 s | ended normally: 1,642 tokens at 58.96 tok/s | 1.0 | 14 of 1,438 units |

Every follow-up kept the whole previous prompt (7 of 7 in each chat, on both builds). In this termination test 3 of
3 chats ended normally in the accepted configuration and 3 of 3 on Volta sPLIt. With 0 of 3 at the limit, the 95 %
Wilson interval of the share of chats that run to the limit is 0.000 to 0.561 (arithmetic for 0 of 3 with z =
1.95996: z squared over 3 is 1.2805, and 1.2805 / 2.2805 is 0.5615), so three chats bound that rate only below 0.56.

Chat 0 repeats itself. In the accepted configuration its eight requests had the context, kept-prompt and output
token counts of the 2026-11-09 run (receipt `baseline/t3-T3a-loop-think1.jsonl`): outputs of 48, 273, 264, 199, 256,
196 and 218 tokens, then 6,417. Volta sPLIt's chat 0 had the counts of its 2026-11-09 run: 191, 111, 127, 116, 141,
127 and 138 tokens, then 10,013. The receipts hold counts, not text. The engine files differed between the two
runs of this build: those of 2026-11-10 add the part B code, which the accepted profile leaves switched off.

Chat 0 by configuration of this build, all at temperature 0:

| Configuration | Hot / warm / host experts, stage 0 and stage 1 | Decode routes on official-precision experts by the tally | Chat 0, final answer | Receipt |
|---|---|---|---|---|
| Accepted: host tier, one slot, reserves 4.5 / 7.5 GiB | 4,107 / 3,213 / 360 and 2,695 / 4,379 / 606 | 96.26 % / 90.71 % | ended normally in three runs (6,417 tokens each), the third under the load setting (below) | `baseline/t3-T3a-loop-think1.jsonl`, `baseline/W4-t3a-term-c0.out`, `baseline/W5-t3so-term-c0.out` |
| No host tier, one slot (not accepted, quality not established) | 3,843 / 3,837 / 0 and 2,184 / 5,496 / 0 | 95.26 % / 86.47 % | ran to the limit, 5.66 tokens per round | `baseline/t3-T3n-loop-think1.jsonl` |
| Host tier, one slot, reserves 5.3 / 8.1 GiB, traced (not accepted, quality not established) | 3,801 / 3,519 / 360 and 2,450 / 4,624 / 606 | 95.08 % / 88.84 % | ran to the limit, 5.21 tokens per round | `baseline/W3-ns-tr-loop-think1.jsonl` |
| Plan 04 part B variant, host tier, one slot, reserves 5.3 / 8.1 GiB (not accepted, quality not established) | 3,656 / 3,664 / 360 and 2,381 / 4,693 / 606 | 94.43 % / 88.27 % | ran to the limit, 5.33 tokens per round. Its chats 1 and 2 ended normally (below) | `baseline/W4-piq-so-term-c0.out` |
| Host tier, three slots, reserves 5.3 / 8.1 GiB (not accepted, quality not established) | 3,597 / 3,723 / 360 and 2,290 / 4,784 / 606 | 94.16 % / 87.46 % | ran to the limit, 5.16 tokens per round (chat with a label line, 216,255 tokens) | `baseline/W3-s3h-loop-think1.jsonl` |
| No host tier, three slots (not accepted, quality not established) | 3,334 / 4,346 / 0 and 1,779 / 5,901 / 0 | 92.77 % / 81.98 % | ran to the limit, 5.87 tokens per round | `baseline/t3-S3n-loop-think1.jsonl` |

The share on official-precision experts is 100 % less the warm share the loader prints (arithmetic). On chat 0,
every configuration with a lower share than the accepted one ran the final answer to the 32,000-token limit in
thinking.

What this does and does not show. The lower share and the limit go together on chat 0 only. Those are five boots.
Four of them ran chat 0 and no other chat. The fifth, the part B variant (not accepted, quality not established),
ran chats 1 and 2 on a second boot of the same profile, and both ended normally (749 and 1,491 tokens,
[section](#iq-records-in-prompt-reading-plan-04-part-b-not-accepted-quality-not-established)).
With chat 0 it stands at 1 of 3 at the limit: the 95 % Wilson interval of its share of chats at the limit is 0.061
to 0.792 (arithmetic for 1 of 3 with z = 1.95996: centre 0.4269, half-width 0.3654). So this lower-share
configuration does not run chats to the limit in general (chat 0 did, chats 1 and 2 did not), and the rate of the
other four on other chats is not measured. The cause is not established, and chance is not excluded. Of the five
runs to the limit, only the part B variant's was measured for repetition: its answer is a repetition loop (distinct
20-unit windows 0.0422, longest repeated span 25,652 of 26,822 units). The four earlier runs kept no such measure.

Volta sPLIt also runs to the limit. On a separate chat the same day (31,173 tokens at the first read, two tool-call
requests, a label line first, receipt `baseline/W4-smoke-prod.out`) its final answer ran to the 32,000-token limit
with thinking never closed: 154.02 s wall, 210.24 tok/s, distinct 20-unit windows 0.2011 overall and 0.0984 in the
last quarter, longest repeated span 18,185 of 23,336 units. That is a repetition loop. So a run to the limit is
not, by itself, a fault of this build.

The accepted tiers under the load setting. MEASURED 2026-11-10 (receipt `baseline/W5-t3so-term-c0.out`): a profile
that pins the accepted tiers with the load-time allocator setting off
([below](#iq-records-in-prompt-reading-plan-04-part-b-not-accepted-quality-not-established)) was ready in 334 s.
Under that setting its reserves read 4.45 / 7.67 GiB. It booted with the accepted configuration's hot / warm /
host experts, 4,107 / 3,213 / 360 and 2,695 / 4,379 / 606, and the same tier digests. Free card memory after load
was 3.93 GiB (stage 0) and 7.34 GiB (stage 1). Chat 0: first read 167.64 s, follow-ups' first token 8.74 to 9.49 s,
final answer ended normally with 6,417 tokens at 66.97 tok/s, distinct 20-unit windows 0.9938, longest repeated
span 47 of 5,012 units. All eight requests had the context, kept-prompt and output token counts of the two earlier
runs. So on this chat the setting did not change the accepted configuration's token counts. The receipts hold
counts, not text. The profile is not in this snapshot.

## IQ records in prompt reading (plan 04 part B, not accepted: quality not established)

The mode that reads warm experts from their IQ2-class records during prompt reading was built and passed its CPU
tests and its layer check on one card of the test host. It is not in this snapshot.

Load failure. Its first eight-card boot (2026-11-10) failed while loading: the driver returned "invalid argument"
during the build of the IQ bank. MEASURED on one card of the test host afterwards: the engine loads weights under
an allocator setting that caps block splitting at 20 MiB. Under it, building one stage-1 rank of the failed boot's
plan (2,310 hot / 4,764 warm / 606 host experts) left 0.038 GiB of the card free, against 7.696 GiB with the
setting off. The records of all 20 layers were equal in both arms. A probe of three small arms then gave the boot's
error. With the setting on, a cached piece split from a larger block and a request the card cannot meet return
"CUDA error: invalid argument". The same request without the setting, or with the setting and a whole cached block,
returns an ordinary out-of-memory error. Fix: load with the setting off (`VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0`).
MEASURED on eight cards: with it off the variant was ready in 349 s with no error. INFERENCE: that the first boot
failed in exactly this way. No boot was made with the setting on and the allocator traced.

The launcher in this snapshot does not set it. Every boot of this section after the failed first one, and the
pinned-tier boot of the accepted tiers ([Long thinking answers](#long-thinking-answers-the-termination-test)), ran
with the setting off. Every other measurement on this page was made with the engine's default. With the setting
off the planner measures other free memory, so the accepted profile would not choose its measured tiers (ESTIMATED
from the planner on CPU). A profile that pins those tiers under the setting was measured on 2026-11-10: chat 0 gave
the accepted configuration's token counts again. That profile is not in this snapshot.

The first boot that was measured (receipt `baseline/W4-piq-so-term-c0.out`): host tier, one slot, reserves
5.3 / 8.1 GiB, a prompt-reading workspace of 0.33 / 0.34 GiB per card. Hot / warm / host experts: stage 0
3,656 / 3,664 / 360, stage 1 2,381 / 4,693 / 606. Warm decode routes by the tally:
5.57 % (stage 0) and 11.73 % (stage 1). Chat 0 of the termination test, against the accepted configuration on the
same test the same day:

| | Part B variant (not accepted, quality not established) | Accepted configuration |
|---|---|---|
| First read, 216,245 tokens | 107.82 s (2,006 tok/s, arithmetic) | 164.9 s (1,311 tok/s, arithmetic) |
| Follow-ups that kept the whole previous prompt | 7 of 7 | 7 of 7 |
| Seven follow-ups: first token | 4.02 to 5.38 s, median 4.44 s | 8.76 to 9.43 s, median 8.99 s |
| Six tool-call requests: decode | 49.41 to 62.57 tok/s | 55.49 to 65.29 tok/s |
| Six tool-call requests: wall time each | 7.85 to 9.12 s | 12.59 to 14.05 s |
| Final answer | ran to the 32,000-token limit in thinking: 356.05 s wall, 90.95 tok/s, 58.56 ms per round, 5.33 tokens per round | ended normally: 6,417 tokens, 67.45 tok/s, 61.65 ms per round, 4.16 tokens per round |
| Repetition in the final answer | distinct 20-unit windows 0.0422, longest repeated span 25,652 of 26,822 units | 0.9938 and 47 of 5,012 units |

Reading the prompt from compressed records changes the output from the first request on: request 0 gave 93 output
tokens here against 48 in the accepted configuration.

Chats 1 and 2. MEASURED 2026-11-10 on a second boot of the same profile (ready in 343 s, the same hot / warm /
host experts and tier digests as the first, 4.95 GiB of card memory free after load on stage 0 and 7.88 GiB on
stage 1, receipts `baseline/W5-piqa-term-c12.out` and `baseline/W5-piqa-term.jsonl`), against the accepted
configuration on the same chats:

| | Part B variant (not accepted, quality not established) | Accepted configuration |
|---|---|---|
| Chat 1, first read, 198,307 tokens | 102.12 s (1,942 tok/s, arithmetic) | 155.56 s (1,275 tok/s, arithmetic) |
| Chat 1, seven follow-ups: first token | 4.3 to 5.58 s, median 4.42 s | 8.9 to 9.51 s, median 9.09 s |
| Chat 1, final answer | ended normally: 749 tokens at 57.56 tok/s, distinct 20-unit windows 1.0, longest repeated span 6 of 669 units | ended normally: 1,499 tokens at 56.92 tok/s |
| Chat 2, first read, 204,871 tokens | 103.0 s (1,989 tok/s, arithmetic) | 161.39 s (1,269 tok/s, arithmetic) |
| Chat 2, seven follow-ups: first token | 4.14 to 4.61 s, median 4.26 s | 8.79 to 9.12 s, median 8.96 s |
| Chat 2, final answer | ended normally: 1,491 tokens at 62.88 tok/s, distinct 20-unit windows 0.9992, longest repeated span 20 of 1,239 units | ended normally: 1,642 tokens at 58.96 tok/s |

Every follow-up kept the whole previous prompt (7 of 7 in each chat). Request 0 gave 93 output tokens in each chat,
against 48 and 89 in the accepted configuration. With chat 0 the variant at these reserves stands at 1 of 3 at the
limit (95 % Wilson interval 0.061 to 0.792, quality not established).

At the accepted tiers the variant does not read a long prompt. MEASURED 2026-11-10 (private boot log): a profile
that pins the accepted tiers for the variant was ready in 309 s with the accepted configuration's hot / warm / host
experts and tier digests. Its reserves read 4.12 / 7.33 GiB plus a prompt-reading workspace of 0.33 / 0.34 GiB per
card, which together are the 4.45 / 7.67 GiB of the pinned profile with the staged reading. Free card memory after
load was 3.61 GiB on stage 0 and 7.02 GiB on stage 1, against 3.93 and 7.34 GiB for the staged reading at the same
tiers. On its first request, the 216,245-token read of chat 0, all four stage-0 cards ran out of card memory in the
second chunk of the read, a few seconds in: a request for 108 MiB with 64.75 MiB free, 29.75 GiB allocated and
750 MiB cached in pieces. The server ended by itself. Quality not established, not accepted.

INFERENCE, from reading the code with nothing run: the failing point lies in the part of the reading step that both
modes share. Both hold the same tensors there (1.21 GiB for a chunk of 8,188 rows, arithmetic), and it is the peak
of the step in both. The variant is short by the workspace it keeps from load, 0.33 GiB. Its free memory after
load is lower by about that much (3.61 against 3.93 GiB, 0.32 GiB). Why the second chunk failed where the first
passed is not shown by the log.

The fix is built and checked on one card of the test host. MEASURED 2026-11-10 (private one-card receipt, one
process, one layer with all its experts hot). The reading step allocates and frees its tensors in another order.
At chunks of 7, 64, 4,096 and 8,188 rows the changed file gave results bit-equal to the previous file, and the
previous file repeated its own results bit-equal. The peak of allocated memory in the reading call at 8,188 rows
fell from 1,302,698,496 to 673,694,208 bytes (1.30 GB to 0.67 GB, or 1.21 to 0.63 GiB), a fall of 0.59 GiB.
The check did not run the variant's own records in the reading step, or a read with routes to warm or host
experts. The fix has not yet run on eight cards, so whether the variant then reads a long prompt at the accepted
tiers is not measured. The changed file is not in this snapshot. The engine file here is the previous file of
that check.

Not accepted. The quality of this variant is not established. Its closeness check on twelve prompts is INCONCLUSIVE
([section](#quality-on-twelve-prompts-inconclusive)): first different token 10 for, 2 against, and mean difference
5 for, 1 against, 6 undecided. Its final answer on chat 0 ran to the limit, and those on chats 1 and 2 ended
normally. The run on chat 0 does not separate the variant from its tiers, because the one-slot profile with the
same reserves and the staged reading also ran chat 0 to the limit (the traced run above). Next: the variant at the
accepted tiers on eight cards with the fix for its memory shortfall, the termination test and the closeness capture.

## Three prompt slots (no host tier: not accepted, quality not established)

Profile `serve-nvfp4-pp2-t3n-a3.yaml` (no host tier, `max-num-seqs 3`, reserves 5.3 / 8.1 GiB). One request at a time;
three chats of 30,333 to 31,321 tokens take turns (`harness_loop.py` with `CHATS=3`; receipt
`baseline/t3-S3n-rotate3.jsonl`):

| | Result |
|---|---|
| First reads, one per chat | 30.55 to 32.57 s |
| Follow-ups that kept their own chat's whole prompt | 6 of 6 (Volta sPLIt with one slot, same day: 0 of 6, each follow-up re-read in 12 to 14 s) |
| Follow-ups: first token | 8.70 to 9.27 s |
| Follow-ups: decode | 86.6 to 98.7 tok/s (35.1 to 39.5 ms per round) |
| Final answers | ended normally: 3,084 / 1,530 / 3,584 tokens |
| Long prompt after the chats | 257,706 tokens read in 214.4 s (1,202 tok/s), server healthy; highest card memory 31,400 MiB (stage 0) and 30,118 MiB (stage 1) (receipts `baseline/t3-S3n-long250k.out`, `-long250k-mem.txt`) |

Decode with three slots configured shows no slowdown against one slot (short sets above). The price is card memory:
fewer experts are hot, so the warm shares rise from 4.74 % / 13.53 % to 7.23 % / 18.02 %. The slot mechanism works;
this profile is not accepted because of the no-host-tier quality findings.

## Quality (three-tier against two-tier)

`harness/benchmarks/nvfp4_8r/quality_t3.py` (receipts `t3/quality-capture-{N2,N2r,T3,P}.out`, score
`t3/quality-score-N2-T3-P-N2r.out`): the three public prompts of the Volta sPLIt harness's general set,
1,024 tokens each, greedy, thinking off, log-probabilities of the chosen tokens. Reference: this build's two-tier
output (all routed experts at official precision), which repeated token for token on a second capture in the same
boot. "Mean difference" is the mean absolute difference of the chosen token's log-probability over the tokens before
the first difference.

| Arm against the reference | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---|---|---|
| Three-tier, first different token | 25 | 6 | 12 |
| Three-tier, mean difference | 0.0486 | 0.0026 | 0.0327 |
| Volta sPLIt, first different token | 0 | 10 | 5 |
| Volta sPLIt, mean difference | not defined (no common prefix) | 0.0392 | 0.1429 |
| No host tier, three slots, first different token | 25 | 45 | 5 |
| No host tier, three slots, mean difference | 0.0710 | 0.0722 | 0.0796 |

Rule written before the run (plan 01 section 8): PASS if three-tier diverges no earlier than Volta sPLIt on at least
two prompts and has the smaller mean difference on at least two. Three-tier diverges later on prompts 1 and 3 and
has the smaller mean difference on prompts 2 and 3: PASS. Limits: three prompts, greedy decoding, thinking off. It is
a closeness check against the all-native output, not a benchmark score.

No host tier with three slots (receipt `t3/quality-score-N2-T3nS-P-N2r.out`): INCONCLUSIVE by the rule. The divergence
part holds on three of three prompts. The mean part has one prompt below Volta sPLIt (prompt 3), one above (prompt 2)
and one undecidable (prompt 1: Volta sPLIt has no common prefix), so the rule cannot decide.

## Quality on twelve prompts: INCONCLUSIVE

MEASURED 2026-11-10 ([How it was measured](#how-it-was-measured)): 1,024 tokens each, greedy, thinking off,
log-probabilities of the chosen tokens, against this build's two-tier output, which repeated token for token on a
second capture in the same boot. Score receipts `t3w/quality-wide-score-N2w-T3w-Pw-N2wr.out` (accepted
configuration) and `t3w/quality-wide-score-N2w-S3hw-Pw-N2wr.out` (three slots on the accepted tiers: host tier,
reserves 5.3 / 8.1 GiB, not accepted, quality not established). The part B variant (reserves 5.3 / 8.1 GiB, not
accepted, quality not established) was captured the same day, score receipt
`t3w/quality-wide-score-N2w-PIQAw-Pw-N2wr.out`, and has its own table below.

| Prompt (kind) | First different token: accepted | three slots | Volta sPLIt | Mean difference: accepted | three slots | Volta sPLIt |
|---|---|---|---|---|---|---|
| 1 (code) | 19 | 19 | 13 | 0.0213 | 0.0324 | 0.0648 |
| 2 (JSON) | 26 | 52 | 34 | 0.0279 | 0.0239 | 0.0507 |
| 3 (general prose) | 16 | 16 | 1 | 0.0098 | 0.0145 | 0.5154 |
| 4 (legal-style prose) | 16 | 5 | 9 | 0.0146 | 0.0000 | 0.0334 |
| 5 (reasoning) | 40 | 23 | 0 | 0.0506 | 0.0351 | not defined |
| 6 (code) | 38 | 38 | 0 | 0.0503 | 0.0491 | not defined |
| 7 (general prose) | 2 | 12 | 2 | 0.0004 | 0.0640 | 0.2006 |
| 8 (legal-style prose) | 14 | 14 | 10 | 0.0029 | 0.0066 | 0.1104 |
| 9 (reasoning) | 18 | 18 | 0 | 0.0346 | 0.0539 | not defined |
| 10 (JSON) | 73 | 73 | 0 | 0.0076 | 0.0098 | not defined |
| 11 (code) | 15 | 1 | 4 | 0.0528 | 0.0000 | 0.0616 |
| 12 (general prose) | 17 | 17 | 0 | 0.0264 | 0.0187 | not defined |

The part B variant (not accepted, quality not established) on the same twelve prompts, INCONCLUSIVE:

| Prompt (kind) | First different token: part B variant | Volta sPLIt | Mean difference: part B variant | Volta sPLIt |
|---|---|---|---|---|
| 1 (code) | 19 | 13 | 0.0381 | 0.0648 |
| 2 (JSON) | 41 | 34 | 0.0332 | 0.0507 |
| 3 (general prose) | 16 | 1 | 0.0458 | 0.5154 |
| 4 (legal-style prose) | 5 | 9 | 0.0001 | 0.0334 |
| 5 (reasoning) | 40 | 0 | 0.0491 | not defined |
| 6 (code) | 0 | 0 | not defined | not defined |
| 7 (general prose) | 0 | 2 | not defined | 0.2006 |
| 8 (legal-style prose) | 37 | 10 | 0.0337 | 0.1104 |
| 9 (reasoning) | 36 | 0 | 0.0491 | not defined |
| 10 (JSON) | 73 | 0 | 0.0140 | not defined |
| 11 (code) | 15 | 4 | 0.1105 | 0.0616 |
| 12 (general prose) | 0 | 0 | not defined | not defined |

Rule fixed before the run: PASS if the arm diverges no earlier than Volta sPLIt on at least 8 prompts and has the
smaller mean difference on at least 8. FAIL if 5 prompts refuse either part. Otherwise INCONCLUSIVE. A prompt with
no common prefix decides nothing.

| Arm | First different token no earlier than Volta sPLIt | Smaller mean difference | Result |
|---|---|---|---|
| Accepted configuration | 11 for, 1 against (prompt 2) | 7 for, 0 against, 5 undecided | INCONCLUSIVE |
| Three slots on the accepted tiers (not accepted, quality not established) | 10 for, 2 against (prompts 4 and 11) | 7 for, 0 against, 5 undecided | INCONCLUSIVE |
| Part B variant, reserves 5.3 / 8.1 GiB (not accepted, quality not established) | 10 for, 2 against (prompts 4 and 7) | 5 for, 1 against (prompt 11), 6 undecided | INCONCLUSIVE |

The first part holds for all three arms. The second cannot decide: on prompts 5, 6, 9, 10 and 12 Volta sPLIt
differs from the reference at the first token, so it has no mean difference to compare with, and seven prompts
cannot reach eight. The part B variant itself differs from the reference at the first token on prompts 6, 7 and
12, so prompt 7 is undecided for it as well, and five prompts cannot reach eight. The rule was not changed after
the data. The accepted configuration stays accepted on the three-prompt PASS above. Three slots and the part B
variant stay not accepted. The missing input is a yardstick on those five prompts. Next: replace them
with prompts on which Volta sPLIt shares a prefix with the reference, chosen without looking at the arms, and score
again.

## Single-card checks (test host, one card, 150 W)

| Check | Result |
|---|---|
| Pack fold (CPU) | 44,696,862,720 values bit-equal to the NVFP4 formula in FP32, relative L2 0; 887.9 s |
| Scale rebase (CPU) | 61,440 expert records, 0 failures, widest exponent span 7 (W13) and 8 (W2), kernel limit 29 |
| Expert layer accuracy | worst relative L2 4.78e-4 against FP32 references from the raw NVFP4 weights (layers 5, 12, 27; 1, 6, 12 rows; ranks 0 and 3) |
| Expert layer time, CUDA graph | 1 row 130.05 to 131.07 us; 6 rows 279.55 to 348.16 us |
| Expert layer time, eager, 6 rows | 337.15 to 364.42 us |
| Graph replay | bit-equal to eager on two routes, 1 to 6 rows |
| Hot and host split | bit-equal to all experts on the card (natural and forced host routes, 1/6/12 rows, prompt chunks of 64 and 512 rows with up to 118 host experts staged) |
| Host expert, zero copy | one host expert per token at 1 row: 1,127.81 us against 152.45 us all on the card (eager); about 1.0 to 1.4 ms per host expert per layer at 6 rows |
| Split overhead with no host route (graph) | +63.5 us per layer at 1 row, +101.4 us at 6 rows |
| Miss path: copy-engine DMA | 743.7 us per expert (6.6 GB/s), linear to 6 experts; host-issued copies cannot be captured in a graph |
| Miss path: GPU gather | 1,014.8 us (1 expert) to 7,788 us (6) |
| Miss path: AVX2 host compute | 2,953.7 us per expert across 4 TP slices (10 threads); activation round trip 57.1 us |
| Prompt staging, 512 rows, every expert staged | 65 to 93 ms against 5.2 to 5.5 ms all on the card |
| MXFP4 Engram rows | bit-equal to FP32 in FP16 on 2 x 4,098 rows; 0.91 to 1.01 M rows/s against 4.2 to 4.6 M rows/s for IQ4_NL (eager) |

ESTIMATED from the miss-path rows and the tally (6 tokens x top-6 per round, independent tokens, 1.33 distinct
host experts per layer per round on average): all experts on the cards 36.1 ms per round; zero copy 96.9; DMA 69.8 to
79.8; GPU gather 84.8 to 94.8; host compute 190 to 200. No path comes within about 10 ms per round of all on the cards.
