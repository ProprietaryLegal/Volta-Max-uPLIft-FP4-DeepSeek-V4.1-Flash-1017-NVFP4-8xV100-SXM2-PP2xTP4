# Measurements

Every number here is MEASURED on 2026-11-09 unless its row or section gives another date or it is labelled
ESTIMATED (arithmetic shown) or UPSTREAM (the named source). The raw receipts are private build logs and are not
published. This page reduces them to tables and names the receipt each row comes from (paths under the build's
`logs/`). The one exception is the closeness test of 2026-11-11: its receipts, scorer output and scripts are
published in [quality/closeness-test/](../quality/closeness-test/).

The rows and sections of the two-tier mode are the all-NVFP4 build now published as
[Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100). They stay here as the dated record, and that
repository's README is the current statement of them. In rows before the mixed tally, "the accepted configuration"
and "the accepted tiers" mean the three-tier profile.

## Results log

Newest first. A later arm is one more row at the top, with its details in a section below.

| Date | Build or profile | Test | Result | Receipt |
|---|---|---|---|---|
| 2026-11-12 | `hot_iq`, final build: the speed bench on the two-slot default, then one to three prompt slots (repository only) | window 18, run 2 declared, 175 W and 1,522 MHz | bench, one request at a time: short sets 123.2 general and 97.3 legal-style tok/s, code 150 to 180 tok/s 1K to 256K, memory 2 MiB over the line on stage 1, no out-of-memory line. Slots: one 118.9 tok/s, two together 159.0 (1.34x), three together 192.3 (1.62x). Token-exact, memory PASS on every slot boot. Quality not established | `baseline/window18-a.out` |
| 2026-11-12 | `hot_iq`, R1 budget, two-slot default, fast reading, fused IQ tail, SwiGLU kernel: the two prompt-reading kernels (`kernel`, `rb4`) against the old path (`chain`, `rb2`) (repository only) | window 17, boots old, new, new, old, a 215,847-token prompt read twice per boot, 175 W | second reads 4.45 and 4.41 s faster (noise 0.50 s), first reads -8.9 and -4.2 s inside a noise floor of 11.6 s: NO MEASURABLE CHANGE by the first-read rule, the second-read gain resolved. Token-exact, peak 31,736 against 31,688 MiB. Installed. Quality not established (fast reading) | `baseline/window17-20261112t2.out` |
| 2026-11-12 | `hot_iq`, R1 tiers, one slot, fast reading, fused IQ tail, SwiGLU kernel: switches off against on (repository only) | window 16, boots off, on, on, off | token-exact with the switches off and on. A follow-up turn that adds a system message to a 30,779-token chat reused 30,752 of those tokens (99.9 %) and took 0.45 s. With the switch off it reused none and took 14.84 s. Decode speed: no measurable change. Both ACCEPT, installed in the launcher | `baseline/window16-port-20261112c.out` |
| 2026-11-12 | HOT-path decode kernel, one card | launch variants and cast fold, rule fixed before the data | 0.000 ms per round predicted for the launch variants, 0.066 ms for the cast fold, below what eight cards resolve. Closed, no eight-card run | one-card receipt (private) |
| 2026-11-12 | `hot_iq`, R1 budget, fast reading, fused IQ tail, SwiGLU kernel, 1 to 4 prompt slots (repository only) | window 15: aggregate throughput by slots and requests running together, run 2 | one slot 118.7 tok/s. Two slots, two together 160.1 (1.35x). Three slots, three together 194.0 (1.63x). Four slots, four together 198.3 (1.67x). Token-exact on every arm, memory PASS, HOT 6,215 / 5,949 / 5,663 / 5,376 of 15,360. Quality not established ([section](#prompt-slots-window-15)) | `baseline/window15-20261112.out` |
| 2026-11-12 | `hot_iq`, R1 tiers, fast reading, fused IQ tail against the torch tail (repository only) | window 14c, paired A1 B1 B2 A2 | p1 -3.449, p2 -1.333 ms per round, noise floor 2.220: NO MEASURABLE CHANGE. Second runs -1.297 ms (repeat 0.070). Token-exact. Quality not established (fast reading) | `baseline/window14c-20261112.out` |
| 2026-11-11 | `hot_iq`, R1 tiers, fast reading, fused IQ tail against the torch tail (repository only) | window 14, paired A1 B1 B2 A2 | p1 -3.167, p2 +0.085 ms per round, noise floor 2.616: NO MEASURABLE CHANGE. Second runs -1.273 ms (repeat 0.111). Token-exact, 12 MiB less per card. Quality not established (fast reading) | `baseline/window14-20261111.out` |
| 2026-11-11 | `hot_iq`, R1 tiers, fast (iq) prompt reading, one slot (repository only) | window 13b: nine long chats of 191,812 to 319,936 tokens, thinking on | 9 of 9 final answers ended normally, 7 of 7 follow-ups kept the prompt in every chat. First read 85.71 to 118.90 s near 200K tokens, 139.75 and 147.12 s near 300K. Memory FAIL (stage 1 32,062 MiB, no out-of-memory line). No verdict (native arm not run). Quality not established | `baseline/window13b-20261111.out` |
| 2026-11-11 | `hot_iq` on the mixed tally, R1 tiers, SwiGLU kernel against the chain (repository only) | window 12, paired A1 B1 B2 A2 | p1 -2.634, p2 -3.847 ms per round, noise floor 2.605: NO MEASURABLE CHANGE. Second runs -1.279 and -1.288 ms. Token-exact | `baseline/window12-swk-20261111.out` |
| 2026-11-11 | `hot_iq` on the mixed tally, R1 tiers, group-plan kernel, one slot (repository only) | speed bench rerun (169 requests), short sets and 216K tool loop on the bench boot | thinking off, decode 149.6 and 122.5 tok/s on code after 1K and 256K prompts (first bench 123.3 and 103.8), legal 84.7 and 78.5 (71.9 and 61.8), first token after a 256K prompt 207.9 s (209.6 s). Short sets general 101.1 tok/s (28.29 ms per round) and legal-style 81.1 (28.07 ms) on the rerun's boot, against 87.0 (34.73 ms) and 67.3 (34.41 ms) on the first bench's boot. On the rerun's boot the 216K tool loop's final request ended with stop reason `end_turn` after 1,146 tokens. Counting it, that request has run to the limit in 1 of 2 runs with the group-plan kernel, 2 of 4 on R1's tiers and 3 of 6 on every build of this repository so far. | `benchvolta/W9C-t3n-mix4b-r700-gpk-20261111T122408Z/cells.jsonl`, `run.json`, `baseline/W9C-t3n-mix4b-r700-gpk-loop216k-think1.jsonl`, `-g1-tput.out`, `-l1-tput.out` |
| 2026-11-11 | `hot_iq` on the mixed tally, R1 tiers, group-plan kernel against the chain (repository only, not the package as uploaded) | window 11c, the one replication, paired A1 B1 B2 A2 | decode ms per round A1 33.90, B1 29.17, B2 28.53, A2 33.88. p1 -4.514, p2 -5.364, mean -4.939, noise floor 1.135: ACCEPT. Whole request 85.5 to 85.9 tok/s (chain) against 98.3 to 101.2 (kernel). Token-exact in every compare. Report with the next row ([section](#2026-11-11-r1-tiers-and-the-group-plan-kernel)) | `baseline/window11c-20261111.out`, `dispatch/ab-W11c-*-tput.out`, `dispatch/w11c-verdict.out` |
| 2026-11-11 | `hot_iq` on the mixed tally, R1 tiers, group-plan kernel against the chain (repository only) | window 11, paired A1 B1 B2 A2 | decode ms per round A1 34.29, B1 28.54, B2 30.79, A2 34.01. p1 -5.774, p2 -2.471, mean -4.123, noise floor 3.429 (B2r - B2): NO MEASURABLE CHANGE. Whole request 84.6 to 85.9 tok/s against 93.5 to 101.6. Token-exact in every compare | `baseline/window11-20261111.out`, `dispatch/ab-W11-*-tput.out`, `dispatch/w11-verdict.out` |
| 2026-11-11 | `hot_iq` on the mixed tally, R1 tiers, group-plan kernel (repository only) | memory gate: 500K-token read and 216K tool loop | 498,099-token read 403.0 s, tool loop 7 of 7 follow-ups kept the prompt, decode mean 115.0 tok/s, no out-of-memory line, peak 31,694 MiB (stage 0) and 30,994 MiB (stage 1) against the limit 31,998: PASS. The final request ran to the 32,000-token limit inside thinking | `baseline/w11-gpkmem-20261111.out`, `baseline/W11-gpkmem-long500k.out`, `baseline/W11-gpkmem-loop216k-think1.out` |
| 2026-11-11 | `hot_iq` on the mixed tally, stage-1 reserve 7.00 GiB (R1), chain (repository only) | memory gate: 500K-token read and 216K tool loop | 498,099-token read 400.6 s, tool loop decode mean 100.4 tok/s, no out-of-memory line, peak 31,732 MiB (stage 0) and 31,456 MiB (stage 1): PASS. The final request ran to the 32,000-token limit inside thinking | `baseline/w10a-mem-20261111.out`, `baseline/W10a-mem-long500k.out`, `baseline/W10a-mem-loop216k-think1.out` |
| 2026-11-11 | `hot_iq` on the mixed tally, stage-1 reserve 7.00 GiB (R1), chain (repository only) | tiers, route shares, short sets, tool loops, closeness check | stage 1 2,459 hot / 5,221 compressed (digest `9473486077c6`), stage 0 unchanged. Decode routes on NVFP4 experts 80.28 / 84.37 / 86.39 / 85.36 % (in-sample). Short sets 81.6 / 65.4 tok/s, 31K loops 96.1 / 85.0, 216K loop decode mean 97.7 (final request ended normally). Closeness check on the replaced 12-prompt set PASS: 11 for and 1 against, 9 for and 3 against | `baseline/window10a-20261111.out`, `t3w/quality-wide-score-R1-W10Aw-W10Ap.json` |
| 2026-11-11 | `hot_iq`, a boot of the exported package (the build of this release) | closeness to the NVFP4 reference, 320 contexts and 513 MMLU questions | same next token 95.1 % (Volta sPLIt 92.2 %), KL divergence 0.0794 nats (0.1958), all 32 tokens identical 25.6 % (10.9 %), changed multiple-choice answers 1.6 % (15.4 %), accuracy 83.6 % (reference 83.8 %, Volta sPLIt 84.4 %). HOT_IQ CLOSER on all four comparisons fixed in advance, 1,475 MHz ([README](../README.md#how-close-to-nvfp4)) | [`quality/closeness-test/score-v1.json`](../quality/closeness-test/score-v1.json) and its receipts |
| 2026-11-11 | `hot_iq`, a boot of the exported package | boot check and same output as the measured build | ready after 263 s, 0 tracebacks, tier lines and both tier digests equal on all eight ranks. 12 + 5 check prompts, 1,024 tokens each, token for token identical, mean log-probability difference 0.0, 1,475 MHz | `baseline/window-cls1-20261111.out` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot (the build of this release) | speed bench, long prompts 1K to 256K, thinking off | code 123.3 tok/s at 1K to 103.8 at 256K, legal 71.9 to 61.8, prose and chat 59.6 to 67.9. First token 5.60 s at 1K, 106.4 s at 128K, 209.6 s at 256K. Greedy output byte-identical in 45 of 45 cells, 1,425 MHz ([section](#speed-bench-1k-to-256k)) | `benchvolta/W9C-t3n-mix4bh-20261110T182114Z/cells.jsonl`, `run.json` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, long prompts 1K to 256K, thinking on | code 102.3 to 136.8 tok/s, structured 107.1 to 122.5, legal 67.8 to 81.8, prose 60.3 to 67.9, chat 54.6 to 66.6. First token 5.76 s at 1K to 210.2 s at 256K, 1,425 MHz | `benchvolta/W9C-t3n-mix4bh-20261110T182114Z/cells.jsonl`, `run.json` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, prompt reading by kind | 175 to 247 tok/s at 1K, 1,030 to 1,055 at 32K, 1,221 to 1,263 at 128K, 1,243 to 1,276 at 256K, 1,425 MHz | `benchvolta/W9C-t3n-mix4bh-20261110T182114Z/cells.jsonl`, `run.json` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, short prompts by kind | greedy: code 127.2 tok/s, structured 129.5, legal 74.9, prose 65.2, chat 59.9. Sampled 58.4 to 127.5. Greedy repeats byte-identical, 1,425 MHz | `benchvolta/W9C-t3n-mix4bh-20261110T182114Z/cells.jsonl`, `run.json` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, four prompts, thinking on, sampled | 92.4 tok/s token-weighted (84.7 to 130.8 per prompt), 1,425 MHz | `benchvolta/W9C-t3n-mix4bh-20261110T182114Z/cells.jsonl`, `run.json` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot, bench boot | tool loop, 216,255-token prompt, run 2 | first read 166.22 s, 7 of 7 follow-ups kept the prompt, first token median 9.47 s, decode mean 99.8 tok/s over all seven, 93.9 over the six that ended with a tool call. The final request ran to the 32,000-token limit inside thinking with no final answer (5.44 tokens per round). With run 1 (row below): 1 of 2 runs at the limit, 1,425 MHz | `baseline/W9C-t3n-mix4bh-loop216k-think1.out`, `.jsonl` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot, bench boot | short sets | general 87.0 tok/s (34.73 ms per round), legal-style 67.3 tok/s (34.41 ms), 1,425 MHz | `baseline/W9C-t3n-mix4bh-g1-tput.out`, `-l1-` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot (the build of this release) | short sets | general 83.6 tok/s (36.25 ms per round), legal-style 64.7 tok/s (35.78 ms). In-sample ([section](#hot_iq-on-the-mixed-tally-the-build-of-this-release)) | `baseline/W9-hiq1-g1-tput.out`, `-l1-` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | tool loops at 31K tokens, source code and US Code Title 28 | follow-ups decode 90.9 tok/s (40.09 ms per round) and 87.8 tok/s (38.18 ms). In-sample | `baseline/W9-hiq1-loop-think1.out`, `-looplegal-` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | tool loop, 216,255-token prompt, run 1 | first read 168.72 s, 7 of 7 follow-ups kept the prompt, first token median 9.36 s, decode mean 90.4 tok/s, final answer ended normally (1,385 tokens). Run 2 ran to the limit (row above) | `baseline/W9-hiq1-loop216k-think1.out` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | closeness, replaced 12-prompt set | PASS: first different token 11 for, 1 against, and mean difference 8 for, 4 against, 0 undecided | `t3w/quality-wide-score-R1-W9HIw-W9HIp.json` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | long thinking chats 0, 1 and 2 | 0 of 3 at the 32,000-token limit (3,057, 1,262 and 1,592 tokens), 95 % Wilson interval 0.000 to 0.561 | `baseline/W9-hiq1-term-c0.out`, `-c1-`, `-c2-` |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot, route counters on | decode routes on compressed records | general 21.0 %, legal-style 16.7 %, source loop 15.0 %, US Code loop 15.7 % | `stats/accept-host-routes-W9-hiqstat-kinds.json` |
| 2026-11-10 | `hot_iq` on the mixed tally, three slots: NOT accepted (quality not established) | short sets, three 30K chats taking turns | general 81.9 tok/s (35.23 ms per round), legal-style 68.7 tok/s (34.42 ms). 6 of 6 follow-ups kept their own chat's prompt | `baseline/W9-hiq3b-g1-tput.out`, `-l1-`, `-rotate3.out` |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B) at the accepted tiers: NOT accepted (quality not established) | first read of chat 0 (216,245 tokens) | booted with the accepted configuration's hot, warm and host experts, then ran out of card memory on stage 0 in the second chunk of the read (a request for 108 MiB with 64.75 MiB free). The server ended by itself. INFERENCE from the code, nothing run: both reading modes peak at the same point and the variant is short by the workspace it keeps from load. The fix is built and checked on one card of the test host (MEASURED 2026-11-10: results bit-equal to the previous file at four chunk sizes, peak of the reading call 1.30 GB to 0.67 GB at 8,188 rows). It is in this snapshot's engine file (the `hot_iq` measurements on the mixed tally ran with that file), and the variant has not yet run at the accepted tiers with it ([section](#iq-records-in-prompt-reading-plan-04-part-b-not-accepted-quality-not-established)) | private boot log, private one-card receipt |
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

## Prompt slots, fast prompt reading and four kernels (2026-11-12)

MEASURED 2026-11-11 and 2026-11-12 on the eight cards. Windows 12 and 13b ran at 150 W and a clock ceiling of
1,475 MHz per card, windows 14, 14c and 15 at 175 W and 1,522 MHz (limits raised on 2026-11-11 at 22:46 UTC), windows
16 and 17 at 175 W, window 18 at 175 W and 1,522 MHz. All
on R1's tier budget and the group-plan kernel of the section below. These rows are for the engine and templates of
this repository. **The Hugging Face package as uploaded is the build of the first release**, and no figure in this
section is a measurement of it. Every window's rule was fixed before its data. Public prompts only: the three short
prompts of [harness/run/tput_prompts.py](../harness/run/tput_prompts.py) and Python standard-library source.

### Prompt slots and the final bench (window 18)

Rule: [docs/acceptance/WINDOW18-accept-rule.md](acceptance/WINDOW18-accept-rule.md). MEASURED 2026-11-12, 175 W and 1,522 MHz per card, on the installed build of this update (the switches of window 16 and the kernels of window 17 included). Four boots: the speed bench on the two-slot default ([README](../README.md#speed-bench-2026-11-12-final-build), [below](#speed-bench-2026-11-12-final-build)), then one, two and three prompt slots by window 15's procedure: each level the three short prompts, thinking off, temperature 0, 1,024 tokens out per request, started together, twice, run 2 declared as the reading before the data. Ratio = against one slot, one stream, run 2 (118.9 tok/s). The tier counts and digests of the three slot boots equal window 15's arms of the same slot count, and the bench boot's equal the two-slot arm's. Receipt `baseline/window18-a.out`. WINDOW W18 VALID.

Run 2 (the reading): aggregate tok/s (ratio), mean decode tok/s of each request, ms per decode round:

| Slots | 1 request | 2 together | 3 together |
|---|---|---|---|
| 1 | 118.9 (1.00x), 137.0, 25.08 |  |  |
| 2 | 124.2 (1.04x), 143.2, 24.90 | 159.0 (1.34x), 107.5, 27.23 |  |
| 3 | 123.5 (1.04x), 138.5, 24.87 | 164.8 (1.39x), 108.4, 27.20 | 192.3 (1.62x), 87.8, 38.68 |

Run 1 against run 2, aggregate tok/s (run 1 is the noise view, not the reading):

| Slots | Requests together | Run 1 | Run 2 |
|---|---|---|---|
| 1 | 1 | 117.5 | 118.9 |
| 2 | 1 | 110.3 | 124.2 |
| 2 | 2 | 159.3 | 159.0 |
| 3 | 1 | 117.0 | 123.5 |
| 3 | 2 | 162.5 | 164.8 |
| 3 | 3 | 190.6 | 192.3 |

Three-slot profile, run 2: two requests add 33.4 % to one, a third adds 16.7 % to two. Same token ids as each prompt alone on the same boot: 4 of 4, 8 of 8 and 14 of 14.

Memory, highest sample on any card of the stage over the whole boot, against the 31,998 MiB line:

| Boot | Stage 0 | Stage 1 | Samples | Out-of-memory lines | Verdict |
|---|---|---|---|---|---|
| bench (two slots) | 31,736 MiB (262 to the line) | 32,000 MiB (-2) | 359 | 0 | FAIL |
| 1 slot | 31,754 MiB (244 to the line) | 30,926 MiB (1,072) | 11 | 0 | PASS |
| 2 slots | 31,702 MiB (296 to the line) | 30,818 MiB (1,180) | 14 | 0 | PASS |
| 3 slots | 31,256 MiB (742 to the line) | 30,820 MiB (1,178) | 16 | 0 | PASS |

One long read and one short request (209,175-token prompt of Python standard-library source, a short request 10 s later):

| Slots | Long prompt: first token | Short request: first token | Short request: decode | Short finished before the long prompt's first token |
|---|---|---|---|---|
| 1 | 85.8 s | 76.73 s (it waits for the read) | 82.4 tok/s, 25.14 ms per round | no |
| 2 | 116.7 s | 4.73 s | 9.7 tok/s, 224.85 ms per round | no |
| 3 | 125.1 s | 9.41 s | 9.4 tok/s, 241.56 ms per round | no |

Window 15's tables below (one to four slots) were measured before the switches of window 16 and the kernels of window 17, and stay as its record.

### Prompt slots (window 15)

Rule: [docs/acceptance/WINDOW15-accept-rule.md](acceptance/WINDOW15-accept-rule.md). Four arms, each one a fresh boot with
fast prompt reading, the fused IQ tail and the SwiGLU kernel: one, two, three and four prompt slots
([templates](../engine/configs/)). Each level is the three short prompts, thinking off, temperature 0, 1,024 tokens
out per request, started together. Run 2 of each level is the reading, declared before the data. Aggregate tok/s =
all tokens of the level over the time from the first start to the last end. One request = the three prompts one
after another (one stream). Ratio = against one slot, one stream, run 2 (118.7 tok/s). Receipt
`baseline/window15-20261112.out` (TABLE W15, lines 198 to 210). Result: WINDOW W15 VALID, Xid count 0 throughout.

Run 2, the reading. Each cell: aggregate tok/s (ratio), mean decode tok/s of each request, ms per round.

| Slots | 1 request | 2 together | 3 together | 4 together |
|---|---|---|---|---|
| 1 | 118.7 (1.00x), 137.2, 25.05 | | | |
| 2 | 124.4 (1.05x), 143.5, 24.85 | 160.1 (1.35x), 109.0, 26.99 | | |
| 3 | 123.7 (1.04x), 138.6, 24.85 | 165.2 (1.39x), 108.4, 27.11 | 194.0 (1.63x), 88.2, 38.47 | |
| 4 | 123.7 (1.04x), 137.9, 24.74 | 166.9 (1.41x), 110.8, 26.96 | 193.3 (1.63x), 86.9, 38.76 | 198.3 (1.67x), 61.7, 49.83 |

Run 1, the noise view (not the reading):

| Slots | 1 request | 2 together | 3 together | 4 together |
|---|---|---|---|---|
| 1 | 106.4 (0.90x), 117.4, 28.94 | | | |
| 2 | 105.8 (0.89x), 131.5, 28.28 | 158.4 (1.33x), 108.6, 27.11 | | |
| 3 | 119.4 (1.01x), 134.0, 25.73 | 162.6 (1.37x), 107.2, 27.37 | 193.6 (1.63x), 88.1, 38.48 | |
| 4 | 111.5 (0.94x), 120.4, 28.16 | 167.3 (1.41x), 111.0, 26.92 | 193.2 (1.63x), 86.7, 38.80 | 200.7 (1.69x), 61.8, 49.72 |

Run 1 to run 2: one request +3.6 to +17.5 % (first runs of a boot are slow), two or more together -1.2 to +1.6 %.
On the four-slot arm, two requests together add 34.9 % to one, a third adds 15.9 % to two and a fourth adds 2.6 %
to three (run 2). The engine runs one request per step and the two pipeline stages hold two steps, so the second
request fills the idle stage and further requests mostly share the same two stages.

Lowest to highest decode tok/s of one request, run 2: one slot 82.3 to 180.4. Two slots: 87.9 to 193.0 alone, 81.9
to 136.0 with two together. Three slots: 91.5 to 178.0, 85.9 to 130.9, 66.6 to 103.6 with three. Four slots: 92.4
to 171.4, 85.6 to 136.0, 65.7 to 100.6, 50.5 to 77.4 with four. Receipt: the LEVEL lines of each arm.

**Same output.** Every request of every level of two or more, and the short request of the mixed case, gave the
same token ids as the same prompt alone on the same arm (run 2): 4 of 4 on one slot, 8 of 8 on two, 14 of 14 on
three, 22 of 22 on four (TOKENS lines 48, 90, 144, 192).

**Memory.** Every card sampled every 15 s, limit 31,998 MiB (card 32,510 less 512), out-of-memory lines 0 on every
arm (MEMORY lines 47, 89, 143, 191):

| Slots | Stage 0 high mark (to the limit) | Stage 1 high mark (to the limit) | CUDA graphs per rank |
|---|---|---|---|
| 1 | 31,706 MiB (292) | 30,920 MiB (1,078) | 0.10 to 0.11 GiB |
| 2 | 31,652 MiB (346) | 30,822 MiB (1,176) | 0.19 GiB |
| 3 | 31,528 MiB (470) | 30,764 MiB (1,234) | 0.27 to 0.28 GiB |
| 4 | 31,374 MiB (624) | 30,786 MiB (1,212) | 0.36 GiB |

These samples cover short requests and one 209K-token read per arm. The 500K-token read and the 216K tool loop were
not run on the slot profiles.

**The price in NVFP4 experts.** Each slot keeps its own state and CUDA graphs, so fewer experts fit at NVFP4. Tier
lines at boot (ARM lines 228 to 231), HOT experts of 7,680 per stage:

| Slots | Stage 0 HOT (digest) | Stage 1 HOT (digest) | All HOT of 15,360 | Decode routes on HOT by the tally, stage 0 / stage 1 |
|---|---|---|---|---|
| 1 | 3,833 (`8e7829184090`) | 2,382 (`7f442956514b`) | 6,215 | 88.67 / 79.46 % |
| 2 | 3,691 (`0b6440ce205f`) | 2,258 (`f67e73174ded`) | 5,949 | 87.59 / 77.95 % |
| 3 | 3,531 (`a5d4b7a514d1`) | 2,132 (`7cad74ab15b9`) | 5,663 | 86.29 / 76.31 % |
| 4 | 3,369 (`ea1385132930`) | 2,007 (`c239180fb427`) | 5,376 | 84.87 / 74.58 % |

The last column is the planner's own figure at boot: the share of the mixed tally's decode routes on HOT experts per
stage (in-sample, a prediction of the route counters, which were off in this window). The share of decode routes
that run at exactly NVFP4 precision, by kind of text, is PREDICTED (not measured) from the slope of that share per
GiB of budget on each stage, with the method's own error of about ±1.3 points. The one-slot profile of R1 (stage-1
budget 192 MiB larger than window 15's one-slot arm) is the measured anchor:

| Kind of text | One slot, R1 profile: measured (predicted) | One slot, window 15 arm | 2 slots | 3 slots | 4 slots |
|---|---|---|---|---|---|
| general | 80.3 % (80.4) | about 79.9 % | about 78.2 to 78.3 % | about 76.6 to 76.8 % | about 74.9 to 75.3 % |
| legal-style | 84.4 % (84.7) | about 84.1 % | about 82.6 to 82.7 % | about 81.1 to 81.4 % | about 79.6 to 80.0 % |
| source loop | 86.4 % (86.1) | about 85.7 % | about 84.5 to 84.6 % | about 83.4 to 83.5 % | about 82.2 to 82.5 % |
| US Code loop | 85.4 % (85.6) | about 85.1 % | about 83.7 to 83.8 % | about 82.4 to 82.5 % | about 81.0 to 81.3 % |

The prediction used HOT counts of 3,674 to 3,682 / 2,245 to 2,257 for two slots, 3,515 to 3,530 / 2,107 to 2,132 for
three and 3,354 to 3,379 / 1,969 to 2,006 for four. The boots placed as many or slightly more (table above), so the
measured shares should sit at or slightly above these ranges. No route counter ran on a slot profile.

**One long read and one short request (mixed case).** A 209,175-token public prompt (48 tokens out), then 10 s later
the first short prompt (1,024 tokens out). MIXED lines 224 to 227:

| Slots | Long prompt: first token | Short request: first token | Short request: decode | Short finished before the long prompt's first token |
|---|---|---|---|---|
| 1 | 103.3 s | 94.31 s (it waits for the read) | 81.9 tok/s, 25.18 ms per round | no |
| 2 | 123.8 s | 7.25 s | 9.3 tok/s, 234.97 ms per round | no |
| 3 | 125.2 s | 7.64 s | 9.2 tok/s, 245.55 ms per round | no |
| 4 | 122.9 s | 6.57 s | 9.4 tok/s, 244.12 ms per round | no |

With two or more slots the short request starts in about 7 s, then decodes one round per read chunk at about 9 tok/s
while the long prompt is read, and the read takes 19.6 to 21.9 s longer. Both requests end later than with one slot
(arithmetic from the table: the short request ends about 117 s after the long one is sent with one slot, 125 to 129 s
with two to four), so a short request during a long read gains nothing in this measurement.

Quality: the slot profiles hold fewer NVFP4 experts than any configuration with a closeness or long-chat test, and
none of those tests ran on them. **Quality not established.** The published default is two slots
([template](../engine/configs/hot_iq.yaml.template)). The one- and three-slot templates ship as options. The four-slot profile of this window was measured before
the prompt-reading kernels and is not shipped.

### Fast prompt reading (windows 13b and 9d)

Fast prompt reading (`nvfp4_8r_prefill_warm: iq`) reads each compressed expert from its IQ2-class record on the card
during prompt reading, in place of its NVFP4 record staged from host RAM (`native_staged`). It is the default of the
templates of this repository. Decode is the same in both modes. MEASURED 2026-11-11, window 13b, one slot, iq reading
only (receipt `baseline/window13b-20261111.out`, CHATLINE lines 81 to 89), the long-chat test of
[harness/run/termination_test.py](../harness/run/termination_test.py): nine chats of Python standard-library source,
each one a first read, six tool-call follow-ups and a final answer, thinking on.

| Chat | First read | First token | Follow-ups' first token | Final answer |
|---|---|---|---|---|
| 0 | 216,245 tokens | 114.55 s | 3.92 to 4.58 s | ended normally, 791 tokens |
| 1 | 198,307 tokens | 118.90 s | 6.26 to 7.71 s | ended normally, 845 tokens |
| 2 | 204,871 tokens | 100.12 s | 3.82 to 3.99 s | ended normally, 1,295 tokens |
| 3 | 204,814 tokens | 95.79 s | 3.98 to 6.32 s | ended normally, 1,102 tokens |
| 4 | 200,827 tokens | 90.11 s | 3.44 to 3.79 s | ended normally, 1,151 tokens |
| 5 | 198,140 tokens | 91.04 s | 3.23 to 3.92 s | ended normally, 1,013 tokens |
| 6 | 319,936 tokens | 147.12 s | 6.98 to 7.44 s | ended normally, 609 tokens |
| 7 | 303,709 tokens | 139.75 s | 3.69 to 4.68 s | ended normally, 1,042 tokens |
| 8 | 191,812 tokens | 85.71 s | 3.23 to 3.55 s | ended normally, 1,400 tokens |

Every follow-up kept the whole previous prompt (7 of 7 in each chat). Distinct 20-unit windows of the final answers
0.999 to 1.0, longest repeated span 5 to 20 units. Xid count 0. Near 200K tokens the first read took 86 to 119 s.
The native reading of the first release took 155.56 s, 161.39 s and 164.9 s on chats 1, 2 and 0
([termination test](#long-thinking-answers-the-termination-test)). That is a comparison across days and builds, not a
paired one: window 13b's native arm was not run. At 175 W and 1,522 MHz, chat 7 read in 142.1 s (139.75 s before)
and in 140.92 s with the fused IQ tail installed. Window 9d ran fast reading on a profile with a host tier, not a
template of this repository (its receipt is private): chat 3 first read 113.57 s, final answer
ended normally after 4,348 tokens, distinct 20-unit windows 0.935, longest repeated span 113 units.

What the rule decided: **no verdict.** The rule needed both arms and a score, and only the iq arm ran. Its memory
check FAILED: stage 1 sampled 32,062 MiB against the 31,998 MiB line, with no out-of-memory line. The window's
decision line read "IQ STAYS OPT-IN". The maintainer made fast reading the default over that line. The slot profiles
answer the memory finding with 192 MiB more stage-1 reserve (window 15: stage-1 high marks 30,764 to 30,920 MiB).

**Quality not established.** Fast reading changes the output from the first token: the compressed experts' part of
every prompt is computed from IQ2-class records, not from NVFP4 weights. On 2026-11-10 an earlier form of this mode
gave 93 tokens on request 0 against 48 ([IQ records in prompt reading](#iq-records-in-prompt-reading-plan-04-part-b-not-accepted-quality-not-established)).
The closeness figures of this repository were all measured with native reading. No closeness test ran with fast
reading. To serve with native reading use the `native-slots1` option ([Reproduce](../README.md#reproduce)).

### Fused IQ tail (patch G1, windows 14 and 14c)

One Triton launch now does the tail of the compressed experts' step (the weighted sum into the output), in place of
the chain of tensor operations (`nvfp4_8r_iq_tail: fused`, against `torch`). Paired arms A torch, B fused, one slot,
iq reading, three public prompts, 1,024 tokens each (receipts `baseline/window14-20261111.out` lines 111 to 113,
`baseline/window14c-20261112.out` VERDICT and SECOND-RUN lines):

| Window | p1 (B1-A1) | p2 (B2-A2) | Noise floor | Verdict by the rule | Second runs: fused against torch | Repeat across boots |
|---|---|---|---|---|---|---|
| 14 | -3.167 ms | +0.085 ms | 2.616 ms | NO MEASURABLE CHANGE | -1.273 ms per round | 0.111 ms |
| 14c | -3.449 ms | -1.333 ms | 2.220 ms | NO MEASURABLE CHANGE | -1.297 ms per round | 0.070 ms |

Token for token the same output in every compare. The fused arm idles 12 MiB lower on every card. First runs of a
boot carry 2 to 3 ms per round of noise, which hides a 1.3 ms change from the first-run rule. On second runs the
gain is resolved in both windows (about 1.3 ms per round against a cross-boot repeat of 0.07 to 0.11 ms). It was
installed on that second-run reading.

### SwiGLU kernel (window 12)

One Triton launch computes V4.1's limited SwiGLU of the NVFP4 experts in decode (`nvfp4_8r_swiglu: kernel`, against
`chain`). Window 12, paired arms, native reading, one slot (receipt `baseline/window12-swk-20261111.out` lines 80 to
89): p1 -2.634 ms, p2 -3.847 ms per decode round against a noise floor of 2.605: NO MEASURABLE CHANGE by the rule.
Second runs: -1.279 and -1.288 ms per round (28.25 and 28.18 down to 26.97 and 26.89). Token for token the same output.
It is in every template of this repository, on that second-run reading.

### Two request-path switches (window 16)

Two switches taken over from the Volta sPLIt build, exported by the launcher since 2026-11-12:
`VLLM_V41_SYSTEM_IN_PLACE=1` and `VLLM_V41_TOPK_SEG=1024` (the Volta sPLIt engine's production patches 0058 and
0059, see [engine/README.md](../engine/README.md)). The first keeps a system message that arrives in the
middle of a chat at the place where it was sent, so the prompt the server already holds still matches and is not
read again. Before, it was joined into the system block at the front of the prompt, which changed the start of the
prompt. The second sets the segment width of the decode top-k kernel to 1024 (unset means 8192, the old width). The
width does not change which tokens are chosen. With both unset the engine gives the same tokens as the build before
this change.

The acceptance run (receipt `baseline/window16-port-20261112c.out`, 175 W per card) booted the one-slot profile with
fast prompt reading, the fused IQ tail and the SwiGLU kernel four times in the order off, on, on, off. The only
difference between the arms was the two environment variables, and the run read them back from every server process
of every boot. All prompts were public or invented text, and no reply text was kept.

Token-exactness was a gate, in four steps. First, a conversion check on the CPU compared the request conversion
before and after the change: 9 of 9 checks held. Second, a kernel check on one card compared segment widths 4096,
2048 and 1024 with width 8192, with the serial kernel and with a written specification: 14 of 14 cases were equal.
Third, the token ids of two fixed public prompt sets were captured from the full model. The off arm gave the same ids
as captures taken before the change, on both sets. The on arm gave the same ids as the off arm, on both sets. Fourth,
a short set of three prompts (3,072 generated tokens in all) gave matching token ids in all three later boots against
the first.

The system-message test sent a chat whose prompt was 30,779 tokens of Python standard-library source, then a second
turn that added a system message. With the switch off the second turn reused 0 of the 30,779 tokens and took 14.84 s.
With the switch on it reused 30,752 tokens (99.9 %) and took 0.45 s.

Speed: no measurable change. Each boot ran the short set twice. The second runs gave 121.4 and 121.8 tok/s with the
switches off, and 121.2 and 121.7 with them on. In decode time per round the on arm differed from the off arm by
+0.039 and +0.005 ms on the second runs, against 0.082 ms between two boots of the same arm. The first runs of a boot
are noisier: their differences were +0.086 and +3.608 ms per round, inside a noise floor of 4.225 ms. Timed alone on
one card over four cases, the top-k kernel took a mean of about 172 microseconds per call at width 1024 against about
189 at width 8192. That difference is too small to show in whole-request speed. Verdict by the rule: both switches
ACCEPT.

Limitation: the acceptance run served one request at a time. The speed of two requests running together with the
switches on was not measured in it. Window 18 measured one to three requests together on the installed build
([prompt slots and the final bench](#prompt-slots-and-the-final-bench-window-18)).

### Two prompt-reading kernels (window 17)

MEASURED 2026-11-12, 175 W per card, on the two-slot default with fast prompt reading, the fused IQ tail and the
SwiGLU kernel (receipt `baseline/window17-20261112t2.out`: VERDICT, SECOND-RUN, MEAN READ, SAME, COMPARE and MEMPEAK
lines). Both rules were fixed before the data.

**What the two kernels do.** In prompt reading, every layer sends each token to a few experts and runs two matrix
products per expert with a SwiGLU step between them. The one-launch SwiGLU does that middle step in a single GPU
launch instead of a chain of element-wise operations. The RB 4 kernel changes how many token rows the expert
matrix kernel processes per block of weights it loads, from 16 to 32, so the same weight bytes are read from
memory half as often. Neither kernel changes the order of any sum or any rounding, so the output bits are the same
as the old path.

**How bit-equality was shown.** First on one card, on recorded inputs of three layers (layers 5, 12 and 27) at 1,
7, 512 and 8,188 tokens plus seeded random routings, 30 cases per layer for each kernel. Every output bit matched
the old path in all 90 cases for each kernel. Then on eight cards, with both kernels on, against the old path
(chain and rb2) booted in the same session: every token-exact comparison matched, on the short capture sets and on
the long prompt.

**One card, one layer call of 8,188 tokens (information, not the decision).** One-launch SwiGLU 50.3 to 51.8 ms
against 52.8 to 53.9 ms for the chain. RB 4 42.4 to 42.7 ms against 52.3 to 54.4 ms for RB 2. The allocation peak
was the same in each pair. The card was held at a 150 W limit, and in the RB test the RB 4 runs held a higher clock
(935 to 994 MHz) than the RB 2 runs (771 to 889 MHz), so part of that one-card gain is clock, not work saved.

**Eight cards, reading a 215,847-token prompt.** Four boots in the order old, new, new, old, each reading the
prompt twice (no cache reuse). Read times in seconds:

| Boot | First read | Second read |
|---|---|---|
| Old path (chain, rb2), boot 1 | 105.877 | 94.237 |
| New kernels (kernel, rb4), boot 1 | 96.972 | 89.788 |
| New kernels (kernel, rb4), boot 2 | 93.191 | 89.335 |
| Old path (chain, rb2), boot 2 | 97.432 | 93.741 |

- First reads, the rule written before the data: no measurable change. The two new-minus-old differences were
  -8.905 s and -4.241 s, but first reads in this session varied by up to 11.640 s on the same build, so the rule
  could not separate a gain from noise.
- Second reads, also declared before the data: the gain is resolved. New minus old was -4.449 s and -4.406 s
  (mean -4.428 s), against a noise floor of 0.496 s.
- Mean of all eight reads: 97.822 s for the old path, 92.322 s with the new kernels.
- Memory: the highest per-card peak during the reads was 31,736 MiB with the new kernels and 31,688 MiB with the
  old path. Card by card the new kernels used 44 to 48 MiB more on the first pair of boots.
- Profile keys: `nvfp4_8r_prefill_swiglu: kernel` and `nvfp4_8r_prefill_rb: rb4` (new), `chain` and `rb2` (old).
  Both are required, with no default. The `native-slots1` template carries `chain` and `rb2`.

**Model, not a measurement.** Scaling the one-card times by the number of expert-layer calls in one long read gave
42.0 s of expert-kernel time per read with RB 2, and 37.4 s with RB 4 if only the per-cycle gain carries over
(33.0 s if the one-card time gain carries over in full), plus about 2 s from the one-launch SwiGLU. That predicts
about 6.6 s saved by the per-cycle model and about 11 s by the time model, against 4.4 s measured, so the measured
gain sits nearer the per-cycle model: on one card the RB 4 runs held a higher clock, and the model assumes the two
pipeline stages never overlap, which would make the real saving smaller. The rest of the roughly 90 s read is
attention over the long context, the dense and shared layers, the other expert tiers and the hand-offs between the
two pipeline stages, which these kernels do not touch.

**RB 8, not shipped.** Handling 64 rows per weight load gives the same bits but is slower: the kernel needs 204
registers per thread and 41,728 bytes of shared memory per block, and on the evenly routed prompt shape its second
timed half ran up to 1.39 times the old kernel's per-cycle time.

**The library.** The RB 4 kernel runs from a compiled library, `v41qpnrb4.so`, which this repository does not
distribute, like the other compiled libraries of the engine. Its source and build script are in
[engine/kernels/prefill-rb4/](../engine/kernels/prefill-rb4/README.md). Without it, `nvfp4_8r_prefill_rb: rb2` gives
the same output bits without the second-read gain.

### HOT-path decode kernel (closed, not done)

A decode kernel for the HOT experts was costed earlier at up to 1.45 ms per round. That figure was an upper bound on
the headroom, not a prediction. Measured on one card, the candidate launch variants predicted 0.000 ms per round by a
rule fixed before the data, and the remaining cast fold 0.066 ms per round, below what an eight-card run can resolve.
No eight-card run was spent on it.

On one card, the decode expert kernels spent 1.45 ms per decode round above the weight-read floor (34.5 us per
layer over 40 layers), the most any better launch could recover. We built bit-equal launch variants (one column
tile per block instead of two, and fewer blocks that each walk several expert groups). All gave the same bits. None
of the variants was faster across the expert counts seen in decode. Folding an output cast into the previous kernel
would save 1.66 us per layer, about 0.07 ms per round, below what an eight-card bench can resolve. Not shipped.

## 2026-11-11: R1 tiers and the group-plan kernel

MEASURED 2026-11-11 on the eight cards, card clock ceiling 1,475 MHz, one request at a time. These rows are for the
engine and template of this repository. **The Hugging Face package as uploaded is the build of the first release**
(stage-1 reserve 7.5 GiB, chain), and no figure in this section is a measurement of it. The headline figures stay
those of the package until its small-file update.

### R1: stage-1 reserve 7.00 GiB

The stage-1 reserve was set at 7.5 GiB from the peaks of the two-tier calibration boot (about 6.9 GiB needed plus
0.5 GiB margin). R1 lowers it to 7.00 GiB and checks the peaks again on the `hot_iq` profile itself. The planner and
the tally are unchanged. With 0.5 GiB more budget per card stage 1 holds 2,459 hot experts and 5,221 compressed
(budget 22.03 GiB, tier digest `9473486077c6`), against 2,254 and 5,426 (21.53 GiB, `5b09c0383a80`). Stage 0 is
unchanged (3,833 and 3,847, `8e7829184090`). So 6,292 of the 15,360 routed experts run at NVFP4 precision, against
6,087. A local rerun of the planner on the private tally at this reserve gives the same two digests.

| Test | R1 (7.00 GiB, chain, 1,475 MHz) | First release (7.5 GiB, chain, 1,425 MHz) |
|---|---|---|
| Decode routes on NVFP4 experts, general | 80.28 % | 79.04 % |
| legal-style | 84.37 % | 83.26 % |
| source loop | 86.39 % | 85.02 % |
| US Code loop | 85.36 % | 84.31 % |
| Short set, general | 81.6 tok/s (35.65 ms per round) | 83.6 tok/s (36.25 ms) |
| Short set, legal-style | 65.4 tok/s (35.45 ms) | 64.7 tok/s (35.78 ms) |
| 31K tool loop, source code | 96.1 tok/s (38.48 ms) | 90.9 tok/s (40.09 ms) |
| 31K tool loop, US Code | 85.0 tok/s (39.17 ms) | 87.8 tok/s (38.18 ms) |

Route shares from the route counters on a separate boot of the same profile (in-sample, as in the
[build section](#hot_iq-on-the-mixed-tally-the-build-of-this-release)), host tier 0.000 %. The speeds are one boot
each at different clock ceilings, so they show no change beyond run noise, not a gain or a loss. Memory: the
498,099-token read and the 216K tool loop both completed with no out-of-memory line, stage-1 peak 31,456 MiB per card
against the limit 31,998. Closeness on the replaced 12-prompt set, by the rule of the
[Quality](../README.md#quality) section: first different token no earlier than Volta sPLIt's on 11 of 12 prompts (1
against), smaller mean log-probability difference on 9 (3 against): PASS. Three long thinking chats at temperature 0:
none at the 32,000-token limit. The full closeness test of 320 contexts and 513 questions was not rerun on these
tiers.

The 216K tool loop ran three times on R1's tiers that day. Its final request ended normally once (chain, decode mean
97.7 tok/s) and ran to the 32,000-token limit inside thinking twice (chain 100.4, kernel 115.0). With the first
release's two runs that is 3 of 5 runs at the limit. The speed bench rerun's boot adds one more run:
[Speed bench rerun](#speed-bench-rerun-2026-11-11-kernel-build).

### The group-plan kernel (plan 03 part B)

Each decode step of a layer sorts its expert routes into groups before the expert launches: a permutation, the group
of each launch block and the group offsets, as int32 tensors. Until now a chain of tensor operations built them
(argsort, gathers, compares, a cumulative sum and scatters). The group-plan kernel
(`engine/src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_group_plan.py`) builds the same tensors in one Triton launch, bit for bit, inside the
decode CUDA graphs. The profile key `nvfp4_8r_group_plan` chooses `chain` or `kernel`, with no default. Prompt
reading always uses the chain. At the start of each window, on one card of the eight, the acceptance test
`harness/benchmarks/nvfp4_8r/test_group_plan.py` passed (random tier arrays of the four layer kinds, 22,000 random
routings and 464 edge cases, expected values computed route by route on the CPU, 0 violations), and so did the
real-weight bit-equality check `check_layers_mt6.py` (layers 5, 12 and 27, 72 comparisons, chain against kernel, eager
and CUDA-graph replay). Receipts `dispatch/w11-test_group_plan.log`, `dispatch/w11-check_layers_mt6.log` and their
`w11c-` twins.

Two windows, each A1 B1 B2 A2 with one boot per arm and the throughput run twice per boot, decided by one rule
written before the first window's data ([rule](acceptance/WINDOW11-accept-rule.md)). Window 11 gave NO MEASURABLE
CHANGE, and window 11c, the one replication declared before its data ([replication](acceptance/WINDOW11-replication.md)),
gave ACCEPT. Both are reported.

| Arm | Window 11, ms per round (repeat) | tok/s | Window 11c, ms per round (repeat) | tok/s |
|---|---|---|---|---|
| A1 chain | 34.29 (33.86) | 84.6 (85.6) | 33.90 (33.83) | 85.5 (85.7) |
| B1 kernel | 28.54 (28.19) | 100.0 (101.2) | 29.17 (28.26) | 98.3 (101.2) |
| B2 kernel | 30.79 (28.09) | 93.5 (101.6) | 28.53 (28.23) | 100.3 (101.2) |
| A2 chain | 34.01 (33.79) | 85.3 (85.9) | 33.88 (33.81) | 85.7 (85.9) |
| p1 (B1 - A1), p2 (B2 - A2), mean | -5.774, -2.471, -4.123 | | -4.514, -5.364, -4.939 | |
| noise floor, twice it | 3.429, 6.858 | | 1.135, 2.270 | |
| verdict | NO MEASURABLE CHANGE | | ACCEPT | |

Decode ms per round is decode time over rounds (first to last streamed token). tok/s is output tokens over the whole
request. Three public prompts, 1,024 tokens each, temperature 0, thinking off, `ignore_eos`. Every kernel arm's
output was identical token for token to A1's, and the chain after the kernel patch was identical to window 10a's
captures before it, so the quality results of the chain carry over to the kernel. In window 11 one kernel run (B2,
first run) slowed during its second and third prompts (32.13 and 34.08 ms per round against 28.43 for its first).
Its cause is not known. The kernel was measured on R1's tiers only.

Memory gate on the kernel profile, fixed before its data: PASS (rows above). The 216K tool loop decoded at 115.0
tok/s against 100.4 on the chain at the same tiers. That is one run each, in different windows, both with a final
request that ran to the 32,000-token limit, so it is not a paired comparison.

## hot_iq on the mixed tally: the build of this release

MEASURED 2026-11-10 on the 8-card host, one request at a time, temperature 0, card clock ceiling 1,425 MHz (the
rows of 2026-11-09 ran at 1,522 MHz). Profile: `hot_iq`, one prompt slot, card reserves 4.5 / 7.5 GiB, prompts read
with the full-precision records (`native_staged`), the load-time allocator setting off
(`VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0`). Tiers from the mixed tally: stage 0 3,833 hot and 3,847 warm (compressed)
experts, stage 1 2,254 and 5,426, no host tier (tier digests `8e7829184090` and `5b09c0383a80`). The engine file
is this snapshot's without the tier-map change, which reads the same tiers from the Hugging Face package's map
(booted on eight cards on 2026-11-11 with the same tiers and digests and the same output token for token, row at the
top).

The full speed bench of this configuration (prompt lengths 1K to 256K, five kinds of text) is its own section
below ([Speed bench](#speed-bench-1k-to-256k)).

| Measure | Result | Receipt |
|---|---|---|
| Short prompts, general set | 83.6 tok/s (36.25 ms per round) | `baseline/W9-hiq1-g1-tput.out` |
| Short prompts, legal-style set | 64.7 tok/s (35.78 ms per round) | `baseline/W9-hiq1-l1-tput.out` |
| Tool loop at 31K tokens, source code | 90.9 tok/s (40.09 ms per round) | `baseline/W9-hiq1-loop-think1.out` |
| Tool loop at 31K tokens, US Code Title 28 | 87.8 tok/s (38.18 ms per round) | `baseline/W9-hiq1-looplegal-think1.out` |
| Tool loop at 216K tokens, run 1 | first read of 216,255 tokens 168.72 s, 7 of 7 follow-ups kept the prompt, first token median 9.36 s, decode mean 90.4 tok/s, final answer ended normally (1,385 tokens) | `baseline/W9-hiq1-loop216k-think1.out` |
| Tool loop at 216K tokens, run 2, bench boot | first read 166.22 s, 7 of 7 follow-ups kept the prompt, first token median 9.47 s, decode mean 99.8 tok/s over all seven, 93.9 over the six that ended with a tool call, the final request ran to the 32,000-token limit inside thinking with no final answer ([section](#speed-bench-1k-to-256k)) | `baseline/W9C-t3n-mix4bh-loop216k-think1.out`, `.jsonl` |
| Short sets, bench boot | general 87.0 tok/s (34.73 ms per round), legal-style 67.3 tok/s (34.41 ms) | `baseline/W9C-t3n-mix4bh-g1-tput.out`, `-l1-` |

Short-set tok/s is output tokens over the whole request time, the wait for the first token included. Tool-loop
tok/s is the mean decode rate of the follow-ups (first to last streamed token).

Decode routes on compressed records, counted by the route counters (`VLLM_NVFP4_8R_STATS=1`) on a separate boot of
the same profile: general 20.96 %, legal-style 16.74 %, source loop 14.98 %, US Code loop 15.69 % of the whole
model's decode routes. By stage: 10.38 to 14.85 % on stage 0 and 19.58 to 27.07 % on stage 1. Host tier 0.000 %.

In-sample. The mixed tally weights four kinds of text equally: one private, and three made from this repository's
public prompt sets (the general short set, the source-code tool loops at 31K and 216K tokens, and the legal-style
short set with the US Code tool loop). The rows above are on text the hot set was chosen from. Chats 1 and 2 of
the long thinking test take later sections of the public source text and are held out.

Closeness on the replaced 12-prompt set (MEASURED, greedy, thinking off, 1,024 tokens each, receipts
`t3w/quality-W9HIw.json` and `t3w/quality-W9HIp.json`, score `t3w/quality-wide-score-R1-W9HIw-W9HIp.json`). The
five prompts of the 12-prompt set on which Volta sPLIt shares no prefix with the reference
([section](#quality-on-twelve-prompts-inconclusive)) are replaced by prompts from a pool written for this check,
chosen by a selection rule fixed in advance that reads no result of this build. The rule of 8 of 12 on each part
is unchanged. First different token no earlier than Volta sPLIt's: 11 for, 1 against. Smaller mean
log-probability difference: 8 for, 4 against, 0 undecided. PASS.

Long thinking chats (MEASURED, thinking on, temperature 0, a 32,000-token limit on the final request). Chats 0, 1
and 2 of the termination test ([section](#long-thinking-answers-the-termination-test)): final answers of 3,057,
1,262 and 1,592 tokens, all ended normally, 0 of 3 at the limit (95 % Wilson interval 0.000 to 0.561). Every
follow-up kept its prompt. The rule fixed before the run: no more chats at the limit than the host-tier profile on
the same tally (not accepted, quality not established), which ran chat 0 to the limit and ended chats 1 and 2
normally. Chats 3 to 8 of the planned nine and the runs at sampling settings were not made. Receipts
`baseline/W9-hiq1-term-c0.out`, `-c1-`, `-c2-`.

Three prompt slots (`hot_iq` on the mixed tally, not accepted, quality not established). Tiers stage 0 3,346 /
4,334, stage 1 1,886 / 5,794. General 81.9 tok/s (35.23 ms per round), legal-style 68.7 tok/s (34.42 ms). Three
chats of 30,439 to 31,174 tokens taking turns: 6 of 6 follow-ups kept their own chat's prompt, first token median
8.08 s. Receipts `baseline/W9-hiq3b-g1-tput.out`, `-l1-`, `-rotate3.out`.

## Speed bench: 1K to 256K

MEASURED 2026-11-10 on the build of this release (`hot_iq` on the mixed tally, one prompt slot, reserves 4.5 / 7.5
GiB, prompts read with the full-precision records), one request at a time, card clock ceiling 1,425 MHz. Receipts
`benchvolta/W9C-t3n-mix4bh-20261110T182114Z/cells.jsonl`, `run.json` (169 of 169 requests, status DONE). The prompts and the method
are Volta sPLIt's speed bench: the run record checks each document's token count and the chat template's tokens
against Volta sPLIt's (no drift). Long prompts go to `/v1/completions` as token ids: greedy, seed 42, `ignore_eos`,
1,024 tokens, a random cache salt, thinking off, or on at reasoning effort 50. Short prompts are one chat message,
thinking off, up to 1,024 tokens with a natural stop, at temperature 0 or 1. The four thinking-on prompts run at
temperature 1, reasoning effort 50, up to 2,048 tokens. Decode = (tokens - 1) / (last token - first token), timed
by the benchmark program. In every long-prompt cell the first streamed event carried one token, so that formula holds.

### Long prompts, thinking off

| prompt | first token | prompt reading | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | 5.60 s | 175-247 tok/s | 123.3 | 130.7 | 71.9 | 59.9 | 60.4 |
| 2K | 7.02 s | 274-324 tok/s | 118.9 | 119.5 | 65.2 | 64.5 | 67.9 |
| 4K | 9.12 s | 437-454 tok/s | 120.9 | 114.9 | 62.5 | 63.5 | 66.0 |
| 8K | 13.3 s | 595-622 tok/s | 127.0 | 126.4 | 69.7 | 67.8 | 63.7 |
| 16K | 19.3 s | 812-868 tok/s | 126.6 | 123.0 | 69.7 | 60.2 | 64.5 |
| 32K | 31.4 s | 1,030-1,055 tok/s | 119.3 | 124.4 | 66.7 | 61.8 | 67.2 |
| 64K | 56.1 s | 1,161-1,188 tok/s | 116.9 | 117.6 | 68.0 | 59.6 | 66.7 |
| 128K | 106.4 s | 1,221-1,263 tok/s | 115.4 | 116.9 | 67.9 | 59.9 | 61.8 |
| 256K | 209.6 s | 1,243-1,276 tok/s | 103.8 | 110.0 | 61.8 | 60.5 | 63.3 |

90 requests, each cell the mean of 2 runs. Largest difference between the two runs of a cell 6.3 tok/s. Greedy
output byte-identical in 45 of 45 cells. First token = median of the five kinds, prompt reading = range over the
five kinds.

### Long prompts, thinking on

| prompt | first token | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 5.76 s | 131.7 | 113.1 | 81.8 | 63.8 | 63.4 |
| 2K | 7.17 s | 134.2 | 107.2 | 75.6 | 62.9 | 64.1 |
| 4K | 9.14 s | 135.8 | 121.9 | 73.9 | 61.7 | 62.4 |
| 8K | 13.2 s | 135.2 | 122.5 | 77.3 | 67.9 | 66.5 |
| 16K | 19.3 s | 136.8 | 111.7 | 71.4 | 63.6 | 66.6 |
| 32K | 31.3 s | 102.3 | 112.3 | 77.6 | 63.9 | 60.0 |
| 64K | 56.2 s | 122.0 | 120.7 | 75.5 | 65.2 | 59.4 |
| 128K | 106.8 s | 110.8 | 114.6 | 71.5 | 60.5 | 58.0 |
| 256K | 210.2 s | 110.5 | 107.1 | 67.8 | 60.3 | 54.6 |

45 requests, one run per cell.

### Prompt reading by text type

| prompt | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|
| 1K | 182 | 175 | 183 | 247 | 189 |
| 2K | 279 | 274 | 292 | 324 | 301 |
| 4K | 449 | 437 | 448 | 450 | 454 |
| 8K | 595 | 622 | 617 | 616 | 609 |
| 16K | 868 | 847 | 847 | 812 | 831 |
| 32K | 1,055 | 1,045 | 1,049 | 1,034 | 1,030 |
| 64K | 1,188 | 1,167 | 1,170 | 1,166 | 1,161 |
| 128K | 1,263 | 1,255 | 1,232 | 1,221 | 1,227 |
| 256K | 1,275 | 1,276 | 1,243 | 1,251 | 1,248 |

Prompt tokens / first-token time, thinking off, mean of 2 runs, tok/s.

### Short prompts, by text type

| kind of output | greedy (T=0) | sampled (T=1) | tokens per round, T=0 | Volta sPLIt greedy | Volta sPLIt sampled |
|---|---:|---:|---:|---:|---:|
| code | **127.2** (125.0-127.4) | 127.4 (113.8-129.5) | 4.39 | 213.1 | 196.8 |
| structured (JSON) | **129.5** (125.6-129.7) | 127.5 (126.1-129.5) | 4.45 | 209.4 | 205.4 |
| legal | **74.9** (74.9-75.0) | 68.0 (67.7-69.7) | 2.59 | 115.7 | 108.4 |
| prose | **65.2** (64.6-65.2) | 60.7 (58.2-60.8) | 2.26 | 98.0 | 87.2 |
| chat | **59.9** (59.9-60.1) | 58.4 (55.8-61.0) | 2.06 | 96.2 | 93.8 |
| thinking on, 4 prompts, token-weighted | - | **92.4** (84.7-130.8 per prompt) | - | - | 132.3 |

Up to 1,024 tokens with a natural stop, thinking off, median of 3 runs with range. Greedy repeats byte-identical
for every kind. The four
thinking-on prompts: 130.8 tok/s (215 tokens), 98.5 (544), 121.4 (368) and 84.7 (2,048). Volta sPLIt's columns are its
published bench of 2026-11-05 at a 1,522 MHz ceiling.

### Tool loop at 216K tokens, run 2

The same 216,255-token loop as run 1 (section above), thinking on, on the bench boot after the bench. Receipt
`baseline/W9C-t3n-mix4bh-loop216k-think1.jsonl`.

| Request | Prompt tokens | Kept | First token | Output tokens | Decode | Stop |
|---|---:|---:|---:|---:|---:|---|
| 0 | 216,255 | 0 | 166.22 s | 48 | 148.03 tok/s | tool_use |
| 1 | 219,315 | 216,240 | 9.47 s | 258 | 92.46 tok/s | tool_use |
| 2 | 222,441 | 219,504 | 9.10 s | 198 | 83.37 tok/s | tool_use |
| 3 | 225,633 | 222,576 | 9.26 s | 175 | 89.60 tok/s | tool_use |
| 4 | 228,647 | 225,744 | 9.52 s | 218 | 99.62 tok/s | tool_use |
| 5 | 232,230 | 228,800 | 9.89 s | 193 | 99.05 tok/s | tool_use |
| 6 | 235,478 | 232,368 | 9.68 s | 198 | 99.42 tok/s | tool_use |
| 7 | 238,424 | 235,616 | 9.22 s | 32,000 | 134.97 tok/s | max_tokens |

**The final request of the 216K tool loop ran to the limit on the bench boot.** In this tool loop (216,255-token prompt, thinking on) the first read took 166.22 s and all seven follow-ups kept the previous prompt (first token median 9.47 s, decode mean 99.8 tok/s over the seven and 93.9 tok/s over the six that ended with a tool call (83.4 to 99.6)). The final request then ran to the 32,000-token limit inside thinking and gave no final answer (stop reason `max_tokens`, 5.44 tokens per round). On the earlier boot of the same profile the same loop's final answer ended normally (1,385 tokens). So on this build that loop's final request has run to the limit in 1 of 2 runs. Chats 0, 1 and 2 of the long thinking test on the mixed tally ([above](#hot_iq-on-the-mixed-tally-the-build-of-this-release)) are separate requests of 216,245, 198,307 and 204,871 tokens and are not counted here.

### Short sets on the bench boot

General 87.0 tok/s (34.73 ms per round, 3,072 tokens in 35.31 s), legal-style 67.3 tok/s (34.41 ms, 3,072 tokens in
45.67 s), receipts `baseline/W9C-t3n-mix4bh-g1-tput.out`, `-l1-`. The earlier boot of the same profile measured 83.6
and 64.7 (section above). Both at 1,425 MHz.

## Speed bench rerun 2026-11-11 (kernel build)

One run of each cell set (169 of 169 requests, status DONE), the prompts, method and request order of the first bench. Each cell is first bench / rerun (first bench: 7.50 GiB tiers, chain dispatch, 2026-11-10, 1,425 MHz / rerun: R1 tiers (7.00 GiB), group-plan kernel, 2026-11-11, 1,475 MHz). Not paired: the two runs differ in tier map, dispatch, clock ceiling and day at once, so a cell's difference is not the kernel's alone. The paired, dispatch-only comparison is windows 11 and 11c. Low-priority single-thread file jobs ran on the host for about 10 minutes, from 12:26 to 12:38 UTC, near the start of the long-prompt part (requests 10 to 40 of 169: long-off 1K to 8K). An upload of this package's last files to the Hugging Face Hub, about 7 MB/s on average by its own log, ran beside the bench from its start until 12:59 UTC (requests 1 to 69 of 169: long-off 1K to 64K). The 216K tool loop and the short sets ran after both.

### Long prompts, thinking off

| prompt | first token | prompt reading | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | 5.60 s / 5.49 s | 175-247 / 175-257 tok/s | 123.3 / 149.6 | 130.7 / 148.0 | 71.9 / 84.7 | 59.9 / 76.0 | 60.4 / 71.8 |
| 2K | 7.02 s / 6.81 s | 274-324 / 282-333 tok/s | 118.9 / 148.6 | 119.5 / 148.3 | 65.2 / 78.9 | 64.5 / 73.7 | 67.9 / 76.3 |
| 4K | 9.12 s / 8.94 s | 437-454 / 441-464 tok/s | 120.9 / 153.0 | 114.9 / 155.6 | 62.5 / 82.4 | 63.5 / 70.4 | 66.0 / 78.4 |
| 8K | 13.3 s / 13.1 s | 595-622 / 594-634 tok/s | 127.0 / 152.9 | 126.4 / 144.7 | 69.7 / 79.3 | 67.8 / 73.2 | 63.7 / 72.8 |
| 16K | 19.3 s / 19.2 s | 812-868 / 842-880 tok/s | 126.6 / 148.2 | 123.0 / 148.5 | 69.7 / 82.8 | 60.2 / 79.5 | 64.5 / 76.1 |
| 32K | 31.4 s / 31.4 s | 1,030-1,055 / 997-1,065 tok/s | 119.3 / 153.4 | 124.4 / 147.9 | 66.7 / 81.9 | 61.8 / 74.1 | 67.2 / 79.6 |
| 64K | 56.1 s / 56.0 s | 1,161-1,188 / 1,158-1,200 tok/s | 116.9 / 144.7 | 117.6 / 148.6 | 68.0 / 80.5 | 59.6 / 73.7 | 66.7 / 79.3 |
| 128K | 106.4 s / 105.7 s | 1,221-1,263 / 1,220-1,274 tok/s | 115.4 / 130.2 | 116.9 / 141.5 | 67.9 / 73.4 | 59.9 / 74.0 | 61.8 / 79.5 |
| 256K | 209.6 s / 207.9 s | 1,243-1,276 / 1,252-1,285 tok/s | 103.8 / 122.5 | 110.0 / 139.0 | 61.8 / 78.5 | 60.5 / 66.4 | 63.3 / 75.9 |

First bench: 90 requests, largest difference between the two runs of a cell 6.3 tok/s, greedy output byte-identical in 45 of 45 cells. Rerun: 90 requests, largest difference between the two runs of a cell 23.3 tok/s, greedy output byte-identical in 45 of 45 cells.

### Long prompts, thinking on

| prompt | first token | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 5.76 s / 5.58 s | 131.7 / 161.9 | 113.1 / 146.7 | 81.8 / 96.1 | 63.8 / 77.6 | 63.4 / 76.0 |
| 2K | 7.17 s / 6.95 s | 134.2 / 164.1 | 107.2 / 143.6 | 75.6 / 99.1 | 62.9 / 82.6 | 64.1 / 80.4 |
| 4K | 9.14 s / 8.93 s | 135.8 / 156.1 | 121.9 / 128.9 | 73.9 / 91.3 | 61.7 / 74.7 | 62.4 / 80.6 |
| 8K | 13.2 s / 12.9 s | 135.2 / 169.3 | 122.5 / 118.6 | 77.3 / 91.0 | 67.9 / 78.8 | 66.5 / 73.8 |
| 16K | 19.3 s / 19.1 s | 136.8 / 164.6 | 111.7 / 145.4 | 71.4 / 80.9 | 63.6 / 76.6 | 66.6 / 76.8 |
| 32K | 31.3 s / 31.2 s | 102.3 / 143.6 | 112.3 / 139.9 | 77.6 / 91.2 | 63.9 / 78.0 | 60.0 / 73.5 |
| 64K | 56.2 s / 55.4 s | 122.0 / 165.8 | 120.7 / 94.7 | 75.5 / 90.8 | 65.2 / 75.7 | 59.4 / 76.1 |
| 128K | 106.8 s / 105.7 s | 110.8 / 154.9 | 114.6 / 132.7 | 71.5 / 91.3 | 60.5 / 79.0 | 58.0 / 73.6 |
| 256K | 210.2 s / 208.5 s | 110.5 / 149.4 | 107.1 / 127.1 | 67.8 / 79.9 | 60.3 / 70.6 | 54.6 / 66.4 |

One run per cell in both benches.

### Prompt reading by text type

| prompt | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|
| 1K | 182 / 187 | 175 / 180 | 183 / 175 | 247 / 257 | 189 / 193 |
| 2K | 279 / 285 | 274 / 282 | 292 / 301 | 324 / 333 | 301 / 308 |
| 4K | 449 / 458 | 437 / 441 | 448 / 458 | 450 / 458 | 454 / 464 |
| 8K | 595 / 594 | 622 / 634 | 617 / 631 | 616 / 625 | 609 / 617 |
| 16K | 868 / 880 | 847 / 853 | 847 / 861 | 812 / 849 | 831 / 842 |
| 32K | 1,055 / 1,065 | 1,045 / 1,033 | 1,049 / 1,055 | 1,034 / 1,043 | 1,030 / 997 |
| 64K | 1,188 / 1,200 | 1,167 / 1,161 | 1,170 / 1,158 | 1,166 / 1,171 | 1,161 / 1,170 |
| 128K | 1,263 / 1,274 | 1,255 / 1,258 | 1,232 / 1,240 | 1,221 / 1,220 | 1,227 / 1,236 |
| 256K | 1,275 / 1,283 | 1,276 / 1,285 | 1,243 / 1,252 | 1,251 / 1,261 | 1,248 / 1,255 |

Prompt tokens / first-token time, thinking off, mean of 2 runs, tok/s.

### Short prompts, by text type

| kind of output | greedy (T=0) | sampled (T=1) | tokens per round, T=0 | Volta sPLIt greedy | Volta sPLIt sampled |
|---|---:|---:|---:|---:|---:|
| code | 127.2 (125.0-127.4) / 158.3 (154.9-158.6) | 127.4 (113.8-129.5) / 149.6 (148.3-163.7) | 4.39 / 4.47 | 213.1 | 196.8 |
| structured (JSON) | 129.5 (125.6-129.7) / 160.0 (155.5-160.1) | 127.5 (126.1-129.5) / 156.0 (152.8-157.8) | 4.45 / 4.48 | 209.4 | 205.4 |
| legal | 74.9 (74.9-75.0) / 89.0 (88.8-89.1) | 68.0 (67.7-69.7) / 88.0 (85.1-93.0) | 2.59 / 2.51 | 115.7 | 108.4 |
| prose | 65.2 (64.6-65.2) / 76.3 (75.8-76.3) | 60.7 (58.2-60.8) / 74.2 (72.9-76.9) | 2.26 / 2.16 | 98.0 | 87.2 |
| chat | 59.9 (59.9-60.1) / 76.1 (76.0-76.3) | 58.4 (55.8-61.0) / 70.8 (67.3-74.3) | 2.06 / 2.13 | 96.2 | 93.8 |
| thinking on, 4 prompts, token-weighted | - | 92.4 (84.7-130.8 per prompt) / 108.0 (96.5-145.6 per prompt) | - | - | 132.3 |

Thinking-on prompts, first bench: t1-00 130.8 tok/s (215 tokens), t1-05 98.5 tok/s (544 tokens), t1-09 121.4 tok/s (368 tokens), t1-11 84.7 tok/s (2048 tokens).. Rerun: t1-00 145.6 tok/s (276 tokens), t1-05 120.3 tok/s (961 tokens), t1-09 136.4 tok/s (365 tokens), t1-11 96.5 tok/s (2048 tokens)..

### Tool loop at 216K tokens on the rerun's boot

| Request | Prompt tokens | Kept | First token | Output tokens | Decode | Stop |
|---|---:|---:|---:|---:|---:|---|
| 0 | 216,255 | 0 | 165.55 s | 48 | 204.12 tok/s | tool_use |
| 1 | 219,315 | 216,240 | 9.28 s | 252 | 107.95 tok/s | tool_use |
| 2 | 222,435 | 219,504 | 8.94 s | 213 | 102.70 tok/s | tool_use |
| 3 | 225,642 | 222,592 | 9.03 s | 192 | 95.19 tok/s | tool_use |
| 4 | 228,672 | 225,776 | 9.24 s | 209 | 113.33 tok/s | tool_use |
| 5 | 232,246 | 228,816 | 9.67 s | 196 | 98.22 tok/s | tool_use |
| 6 | 235,496 | 232,384 | 9.55 s | 227 | 124.01 tok/s | tool_use |
| 7 | 238,471 | 235,664 | 9.21 s | 1,146 | 113.24 tok/s | end_turn |

First read 165.55 s for 216,255 tokens (first bench boot 166.22 s). Follow-ups that kept the previous prompt: 7 of 7, first token median 9.24 s (9.47 s), decode mean 107.8 tok/s over the 7 (99.8). On the rerun's boot the 216K tool loop's final request ended with stop reason `end_turn` after 1,146 tokens. Counting it, that request has run to the limit in 1 of 2 runs with the group-plan kernel, 2 of 4 on R1's tiers and 3 of 6 on every build of this repository so far.

### Short sets on the bench boot

General 101.1 tok/s (28.29 ms per round) and legal-style 81.1 (28.07 ms) on the rerun's boot, against 87.0 (34.73 ms) and 67.3 (34.41 ms) on the first bench's boot.

Receipts of the rerun: `benchvolta/W9C-t3n-mix4b-r700-gpk-20261111T122408Z/cells.jsonl` and `run.json`, `baseline/W9C-t3n-mix4b-r700-gpk-loop216k-think1.jsonl`,
`baseline/W9C-t3n-mix4b-r700-gpk-g1-tput.out` and `-l1-tput.out`. Profile
`serve-nvfp4-pp2-t3n-mix4b-r700-gpk.yaml` (R1 tiers, `nvfp4_8r_group_plan: kernel`), one prompt slot, power caps
150 W per card.

## Speed bench 2026-11-12 (final build)

MEASURED 2026-11-12, 175 W and 1,522 MHz per card, window 18 ([rule](acceptance/WINDOW18-accept-rule.md)): the build of this update as published, two prompt slots (the default), fast prompt reading, the fused IQ tail, the SwiGLU kernel, the two request-path switches and the two prompt-reading kernels. One request at a time with a fresh cache salt each, so the slots do not change what it measures. 169 of 169 requests, status DONE, the prompts, method and request order of the first bench. Each cell is kernel build / final build (kernel build: one slot, native prompt reading, 2026-11-11, 1,475 MHz). Not paired: the two runs differ in slots, tier map, prompt reading, kernels, switches, clock and day at once, so a cell's difference is not one change's alone. Fast prompt reading changes the output from the first token: quality not established. The sections above it are the first release's bench and the kernel build's, kept as their record.

**Memory on the bench boot: 2 MiB over the line.** The highest sample on the cards of stage 1 was 32,000 MiB, against the 31,998 MiB line this build is held to (the card's 32,510 MiB less 512): 2 MiB over the line and 510 MiB below the card's capacity. It came in 38 of 359 samples, all while prompts of 8K to 256K tokens were served. Stage 0 peaked at 31,736 MiB. No out-of-memory line in any of the 169 requests, and all 169 finished. The rule of window 18 counts this boot as a memory FAIL. The one-, two- and three-slot boots of the same window stayed under the line.

Decode, thinking off, one request at a time, 1K to 256K: code 150 to 180 tok/s (kernel build 122 to 153), legal 86 to 106 (73 to 85). First token after a 128K prompt 54.3 s (kernel build 105.7 s), after 256K 108.4 s (207.9 s).

### Before and after

Against the previous public bench, the kernel build of 2026-11-11. Both are one request at a time. The profile
differs (one slot then, the two-slot default now), and so do the tier map, prompt reading, the kernels, the two
switches and the clock ceiling, so this compares build with build, not the effect of one change.

| | before: kernel build, 2026-11-11 | after: final build, 2026-11-12 |
|---|---:|---:|
| profile | one slot, native prompt reading, 1,475 MHz | two-slot default, fast prompt reading, fused IQ tail, SwiGLU kernel, both switches, both prompt-reading kernels, 175 W and 1,522 MHz |
| general short set | 101.1 tok/s (28.29 ms per round) | 123.2 tok/s (25.10 ms) |
| legal-style short set | 81.1 tok/s (28.07 ms) | 97.3 tok/s (24.92 ms) |
| first token after 1K / 32K / 128K / 256K | 5.49 / 31.4 / 105.7 / 207.9 s | 1.00 / 16.0 / 54.3 / 108.4 s |
| prompt reading at 256K | 1,252 to 1,285 tok/s | 2,329 to 2,430 tok/s |
| decode 1K to 256K, code, thinking off | 122 to 153 tok/s | 150 to 180 tok/s |
| decode 1K to 256K, legal, thinking off | 73 to 85 tok/s | 86 to 106 tok/s |
| 216K tool loop, first read | 165.55 s | 86.87 s |
| 216K tool loop, follow-ups: first token median, decode mean | 9.24 s, 107.8 tok/s | 3.54 s, 126.8 tok/s |

![Decode speed by prompt length on the final build, one request at a time on the two-slot default, against Volta sPLIt's published bench](charts/decode-by-length-final-light.png#gh-light-mode-only)
![Decode speed by prompt length on the final build, one request at a time on the two-slot default, against Volta sPLIt's published bench](charts/decode-by-length-final-dark.png#gh-dark-mode-only)

### Long prompts: 1K to 256K, thinking off

| prompt | first token | prompt reading | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1K | 5.49 s / 1.00 s | 175-257 / 987-1,174 tok/s | 149.6 / 150.1 | 148.0 / 169.2 | 84.7 / 105.5 | 76.0 / 81.2 | 71.8 / 88.2 |
| 2K | 6.81 s / 1.67 s | 282-333 / 1,180-1,294 tok/s | 148.6 / 167.1 | 148.3 / 188.4 | 78.9 / 97.2 | 73.7 / 82.9 | 76.3 / 82.6 |
| 4K | 8.94 s / 2.97 s | 441-464 / 1,305-1,425 tok/s | 153.0 / 180.3 | 155.6 / 162.3 | 82.4 / 85.9 | 70.4 / 80.4 | 78.4 / 81.0 |
| 8K | 13.1 s / 6.23 s | 594-634 / 1,076-1,341 tok/s | 152.9 / 177.0 | 144.7 / 166.0 | 79.3 / 87.3 | 73.2 / 85.5 | 72.8 / 78.4 |
| 16K | 19.2 s / 9.51 s | 842-880 / 1,678-1,788 tok/s | 148.2 / 155.0 | 148.5 / 170.7 | 82.8 / 93.8 | 79.5 / 85.1 | 76.1 / 89.4 |
| 32K | 31.4 s / 16.0 s | 997-1,065 / 1,896-2,149 tok/s | 153.4 / 165.1 | 147.9 / 164.5 | 81.9 / 93.4 | 74.1 / 78.2 | 79.6 / 81.0 |
| 64K | 56.0 s / 28.4 s | 1,158-1,200 / 2,196-2,355 tok/s | 144.7 / 170.3 | 148.6 / 162.8 | 80.5 / 86.9 | 73.7 / 84.4 | 79.3 / 84.7 |
| 128K | 105.7 s / 54.3 s | 1,220-1,274 / 2,295-2,448 tok/s | 130.2 / 151.4 | 141.5 / 153.3 | 73.4 / 87.3 | 74.0 / 78.4 | 79.5 / 75.5 |
| 256K | 207.9 s / 108.4 s | 1,252-1,285 / 2,329-2,430 tok/s | 122.5 / 154.6 | 139.0 / 150.0 | 78.5 / 93.5 | 66.4 / 80.8 | 75.9 / 71.2 |

Thinking off, each cell the mean of 2 runs. Kernel build: 90 requests, largest difference between the two runs of a cell 23.3 tok/s, greedy output byte-identical in 45 of 45 cells. Final build: 90 requests, largest difference 54.6 tok/s, greedy output byte-identical in 45 of 45 cells.

### Long prompts, thinking on

Each cell is kernel build / final build.

| prompt | first token | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 5.58 s / 0.96 s | 161.9 / 125.2 | 146.7 / 158.9 | 96.1 / 105.5 | 77.6 / 94.7 | 76.0 / 89.2 |
| 2K | 6.95 s / 1.62 s | 164.1 / 180.2 | 143.6 / 143.8 | 99.1 / 107.5 | 82.6 / 85.3 | 80.4 / 83.9 |
| 4K | 8.93 s / 2.91 s | 156.1 / 176.2 | 128.9 / 161.0 | 91.3 / 102.6 | 74.7 / 86.9 | 80.6 / 91.7 |
| 8K | 12.9 s / 5.97 s | 169.3 / 192.6 | 118.6 / 148.3 | 91.0 / 102.8 | 78.8 / 90.8 | 73.8 / 87.3 |
| 16K | 19.1 s / 9.29 s | 164.6 / 168.3 | 145.4 / 155.1 | 80.9 / 104.4 | 76.6 / 88.6 | 76.8 / 91.6 |
| 32K | 31.2 s / 15.5 s | 143.6 / 122.1 | 139.9 / 165.2 | 91.2 / 96.4 | 78.0 / 90.3 | 73.5 / 88.7 |
| 64K | 55.4 s / 28.1 s | 165.8 / 174.8 | 94.7 / 155.5 | 90.8 / 93.8 | 75.7 / 87.4 | 76.1 / 78.6 |
| 128K | 105.7 s / 53.9 s | 154.9 / 118.7 | 132.7 / 140.6 | 91.3 / 92.2 | 79.0 / 87.5 | 73.6 / 76.8 |
| 256K | 208.5 s / 108.8 s | 149.4 / 115.8 | 127.1 / 140.4 | 79.9 / 95.8 | 70.6 / 83.0 | 66.4 / 78.2 |

### Prompt reading by text type

Each cell is kernel build / final build.

| prompt | code | structured | legal | prose | chat |
|---|---:|---:|---:|---:|---:|
| 1K | 187 / 1,035 | 180 / 987 | 175 / 988 | 257 / 1,174 | 193 / 1,032 |
| 2K | 285 / 1,216 | 282 / 1,229 | 301 / 1,180 | 333 / 1,294 | 308 / 1,259 |
| 4K | 458 / 1,425 | 441 / 1,421 | 458 / 1,305 | 458 / 1,342 | 464 / 1,382 |
| 8K | 594 / 1,201 | 634 / 1,325 | 631 / 1,076 | 625 / 1,341 | 617 / 1,318 |
| 16K | 880 / 1,788 | 853 / 1,736 | 861 / 1,678 | 849 / 1,724 | 842 / 1,711 |
| 32K | 1,065 / 2,125 | 1,033 / 2,149 | 1,055 / 2,031 | 1,043 / 1,896 | 997 / 2,051 |
| 64K | 1,200 / 2,355 | 1,161 / 2,307 | 1,158 / 2,196 | 1,171 / 2,288 | 1,170 / 2,317 |
| 128K | 1,274 / 2,448 | 1,258 / 2,416 | 1,240 / 2,308 | 1,220 / 2,295 | 1,236 / 2,419 |
| 256K | 1,283 / 2,430 | 1,285 / 2,406 | 1,252 / 2,329 | 1,261 / 2,423 | 1,255 / 2,418 |

### Beside Volta sPLIt

Each cell is final build / Volta sPLIt's published bench of 2026-11-05 at a 1,522 MHz ceiling, thinking off, mean of 2 runs.

| prompt | first token | prompt reading | code | legal |
|---|---:|---:|---:|---:|
| 1K | 1.00 s / 1.22 s | 987-1,174 / 820-948 | 150.1 / 215.0 | 105.5 / 117.6 |
| 2K | 1.67 s / 1.95 s | 1,180-1,294 / 1,024-1,082 | 167.1 / 203.6 | 97.2 / 114.8 |
| 4K | 2.97 s / 3.26 s | 1,305-1,425 / 1,245-1,264 | 180.3 / 194.9 | 85.9 / 111.6 |
| 8K | 6.23 s / 6.94 s | 1,076-1,341 / 1,178-1,188 | 177.0 / 207.9 | 87.3 / 109.1 |
| 16K | 9.51 s / 10.4 s | 1,678-1,788 / 1,574-1,583 | 155.0 / 201.8 | 93.8 / 105.6 |
| 32K | 16.0 s / 17.2 s | 1,896-2,149 / 1,894-1,913 | 165.1 / 200.5 | 93.4 / 116.5 |
| 64K | 28.4 s / 30.9 s | 2,196-2,355 / 2,111-2,132 | 170.3 / 194.1 | 86.9 / 108.3 |
| 128K | 54.3 s / 59.2 s | 2,295-2,448 / 2,198-2,220 | 151.4 / 182.8 | 87.3 / 110.9 |
| 256K | 108.4 s / 118.7 s | 2,329-2,430 / 2,197-2,222 | 154.6 / 160.0 | 93.5 / 103.1 |

### Short prompts, by text type

| kind of output | greedy (T=0) | sampled (T=1) | tokens per round, T=0 | Volta sPLIt greedy | Volta sPLIt sampled |
|---|---:|---:|---:|---:|---:|
| code | 158.3 (154.9-158.6) / 172.9 (170.4-173.2) | 149.6 (148.3-163.7) / 173.8 (162.1-178.4) | 4.47 / 4.35 | 213.1 | 196.8 |
| structured (JSON) | 160.0 (155.5-160.1) / 191.9 (187.5-192.4) | 156.0 (152.8-157.8) / 183.7 (179.1-187.9) | 4.48 / 4.75 | 209.4 | 205.4 |
| legal | 89.0 (88.8-89.1) / 103.2 (103.2-103.2) | 88.0 (85.1-93.0) / 95.6 (90.3-102.1) | 2.51 / 2.59 | 115.7 | 108.4 |
| prose | 76.3 (75.8-76.3) / 85.9 (85.7-86.1) | 74.2 (72.9-76.9) / 82.6 (80.3-83.5) | 2.16 / 2.16 | 98.0 | 87.2 |
| chat | 76.1 (76.0-76.3) / 81.5 (81.3-81.6) | 70.8 (67.3-74.3) / 80.1 (79.1-82.6) | 2.13 / 2.03 | 96.2 | 93.8 |
| thinking on, 4 prompts, token-weighted | - | 108.0 (96.5-145.6 per prompt) / 116.1 (99.9-173.8 per prompt) | - | - | 132.3 |

Up to 1,024 tokens with a natural stop, thinking off, median of 3 runs with range. Greedy repeats were byte-identical for every kind. Volta sPLIt's columns are its published bench of 2026-11-05.

### Short sets on the bench boot

General 123.2 tok/s (25.10 ms per round) and legal-style 97.3 (24.92 ms) on the final build's bench boot (one request at a time on the two-slot default), against 101.1 (28.29 ms) and 81.1 (28.07 ms) on the kernel build's. Three prompts each, output tokens over the whole request time.

### The 216K tool loop on the final build's boot

One request at a time on the two-slot default. First read 86.87 s for 216,255 tokens (kernel build 165.55 s). Follow-ups that kept the previous prompt: 7 of 7, first token median 3.54 s (9.24 s), decode mean 126.8 tok/s over the 7 (107.8). On the final build's boot the 216K tool loop's final request ran to the 32,000-token limit (stop reason `max_tokens`) and gave no final answer. Counting it, that request has run to the limit in 2 of 3 runs with the group-plan kernel and 4 of 7 on every build of this repository so far.

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

Since 2026-11-12 the launcher `engine/configs/run_serve.sh` sets it off itself, as the measurements of `hot_iq` on
the mixed tally and every later one ran. Before, only the `hot_iq` launcher `engine/configs/serve_hot_iq.sh` set
it. Every boot of this section after the
failed first one, and the pinned-tier boot of the accepted tiers
([Long thinking answers](#long-thinking-answers-the-termination-test)), ran with the setting off. Every other
measurement on this page before the mixed tally was made with the engine's default. With the setting
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
experts. The fix is in this snapshot's engine file: the `hot_iq` measurements on the mixed tally ran with that
file, reading prompts with the full-precision records. The variant has not yet run at the accepted tiers with the
fix, so whether it then reads a long prompt there is not measured. The variant is not in this snapshot.

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
