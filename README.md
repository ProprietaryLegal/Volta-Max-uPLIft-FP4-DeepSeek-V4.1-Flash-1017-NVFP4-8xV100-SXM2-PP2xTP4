# Volta Max uPLIft FP4

## Routed experts mostly at the official weights' precision on 8x V100 from 2018 · three-tier 57 to 58 tok/s legal-style, 51 to 52 general · work in progress, slower than Volta sPLIt

[![three-tier legal-style: 57-58 tok/s](https://img.shields.io/badge/three--tier%20legal--style-57--58%20tok%2Fs-1F6FEB?style=for-the-badge)](#tokens-per-second)
[![three-tier general: 51-52 tok/s](https://img.shields.io/badge/three--tier%20general-51--52%20tok%2Fs-8250DF?style=for-the-badge)](#tokens-per-second)
[![two-tier legal-style: 28-30 tok/s](https://img.shields.io/badge/two--tier%20legal--style-28--30%20tok%2Fs-555555?style=for-the-badge)](#tokens-per-second)
[![KV pool: 2.08M tokens](https://img.shields.io/badge/KV%20pool-2.08M%20tokens-555555?style=for-the-badge)](docs/measurements.md)
[![status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-D29922?style=for-the-badge)](#what-did-not-work-or-is-not-yet-good)

Built by **Nick Mermiges, Esq.**, founder of [PLI Labs](https://proprietarylegal.ai/labs.html) · [proprietarylegal.ai](https://proprietarylegal.ai)

[![proprietarylegal.ai](https://img.shields.io/badge/proprietarylegal.ai-website-0B3D91)](https://proprietarylegal.ai) [![PLI Labs](https://img.shields.io/badge/PLI%20Labs-research-1F6FEB)](https://proprietarylegal.ai/labs.html) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![Hardware: 8x V100-SXM2-32GB](https://img.shields.io/badge/hardware-8x%20V100--SXM2--32GB-76B900)](#hardware) [![Earlier build: Volta sPLIt](https://img.shields.io/badge/earlier%20build-Volta%20sPLIt-555555)](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4)

NVIDIA's [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) pack served on
eight V100-SXM2-32GB cards (two 4-card NVLink boards, pipeline parallel 2 x tensor parallel 4). The most-used
("hot") routed experts run at the precision of DeepSeek's official weights: NVFP4 folds exactly to MXFP4 (4.25 bits
per weight), which the v100-skinny QPN kernel runs as W4A16. In the three-tier mode 3.7 to 9.3 % of routed-expert
calls run from compressed records. In the two-tier mode none do. Our earlier build,
[Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4), serves
the same model with compressed IQ2-class experts and is faster.

This repository is the first public snapshot of work in progress: the measurements, the code of the build as
measured, what worked and what did not. The unfinished prompt-reading variant of plan 04 part B is not in this
snapshot.
The build is slower than Volta sPLIt in decode speed and in prompt reading speed. We publish it now because
the method works, the numbers are measured, and the next steps are planned piece by piece. New results are added as
dated rows in the [measurement log](#measurement-log).

## Tokens per second

MEASURED on the 8-card host on 2026-11-09, the same day for all three builds, one request at a time, public prompts,
greedy (T=0), thinking off, 1,024 tokens out for the short sets and 512 for the context rows. "Two runs" are two
runs in one boot. How each row was measured: [docs/measurements.md](docs/measurements.md).

| | Volta sPLIt (earlier build) | This build, two-tier | This build, three-tier |
|---|---|---|---|
| Routed experts | IQ2-class records, all on the cards | all at official precision; hot on the cards, the rest in host RAM | official precision on the cards and in host RAM; less-used experts from the IQ2-class records |
| Short prompts, legal-style set (two runs) | 118.7 / 118.3 tok/s (21.54 / 21.62 ms per round) | 28.2 / 30.2 tok/s (87.58 / 81.79 ms) | 56.9 / 58.4 tok/s (42.16 / 41.02 ms) |
| Short prompts, general set (two runs) | 136.9 / 137.9 tok/s (21.48 / 21.57 ms) | 18.8 / 20.1 tok/s (174.64 / 163.37 ms) | 50.6 / 51.8 tok/s (59.38 / 57.95 ms) |
| Decode after 46 tokens of context | 125.9 tok/s | 33.7 tok/s | 75.9 tok/s |
| Decode after 30,795 tokens | 103.2 tok/s | 30.6 tok/s | 68.1 tok/s |
| Decode after 106,531 tokens | 101.0 tok/s | 31.5 tok/s | 60.9 tok/s |
| Decode after 257,741 tokens (two-tier: 303,004) | 98.1 tok/s | 30.2 tok/s | 66.1 tok/s |
| Prompt reading, 30K to 300K tokens (the context rows and the tool loop's first read) | 2,610 to 2,946 tok/s | 1,419 to 1,595 tok/s | 1,080 to 1,327 tok/s |
| First token, short prompt | 0.37 to 0.45 s | 0.69 s | 0.84 to 1.07 s |

A "round" is one target forward pass that checks the DSpark drafter's proposals. The table carries two measures of
tok/s. On the two "Short prompts" rows, tok/s is the output tokens divided by the whole request time, the wait for
the first token included, as the short-set program (`ab_t1_v2.py --tput` of the Volta sPLIt harness) prints it. The
ms per round beside it is decode time only (first to last streamed token) divided by the rounds. So a short-set
tok/s is lower than tokens per round divided by the decode time per round. On the "Decode after" rows and in the
tool loop, tok/s is decode only: output tokens after the first, divided by the time from the first to the last
streamed token.

The three-tier mode decodes at 1.9 to 2.7 times two-tier's tok/s (same-run pairs): about half Volta sPLIt's on
legal-style text and 37 % on general text. From 46 tokens of context to the longest prompt, decode goes from 125.9
to 98.1 tok/s on Volta sPLIt, from 33.7 to 30.2 on two-tier and from 75.9 to 66.1 on three-tier (60.9 at 106,531
tokens).

Three-tier's prompt reading is the slowest of the three. The variant that also runs the warm experts from their
compressed records during prompt reading ([plan 04](docs/plans/04-prefill-staging.md) Part B) now boots.
MEASURED 2026-11-10 (not accepted, quality not established): at reserves of 5.3 / 8.1 GiB it read a 216,245-token
prompt in 107.82 s against 164.9 s. At the accepted tiers it ran out of card memory on that read
([What did not work](#what-did-not-work-or-is-not-yet-good)).
It is not in this snapshot: see
[Long thinking answers](#long-thinking-answers-the-termination-test).

Two-tier also read a 498,099-token prompt: 274.9 s on the calibration boot (card reserves 6 / 9 GiB) and 326.6 s on
the final boot (4.5 / 7.5 GiB), with no out-of-memory error. Its KV pool holds 2,076,870 tokens at a max-model-len
of 524,288.

**The accepted mode of this build today is the host-tier three-tier profile** (`hot_iq_host`, one slot, card
reserves 4.5 / 7.5 GiB): closeness PASS on three prompts and INCONCLUSIVE on a wider set ([Quality](#quality)), 51
to 58 tok/s on short prompts, 47 to 66 tok/s in the tool loop. This page calls that profile the accepted
configuration.

### No host tier (`hot_iq`): not accepted, quality not established

Speeds MEASURED 2026-11-09; quality not established. These profiles drop the host tier: experts that would run
from host RAM run from their IQ2-class records on the cards.

| | Host tier, one slot (accepted) | No host tier, one slot | No host tier, three prompt slots |
|---|---|---|---|
| Experts per stage, hot / warm / host | stage 0: 4,107 / 3,213 / 360; stage 1: 2,695 / 4,379 / 606 | stage 0: 3,843 / 3,837 / 0; stage 1: 2,184 / 5,496 / 0 | stage 0: 3,334 / 4,346 / 0; stage 1: 1,779 / 5,901 / 0 |
| Decode routes on warm (compressed) records, by the tally | 3.74 % and 9.29 % | 4.74 % and 13.53 % | 7.23 % and 18.02 % |
| Short prompts, general set (two runs) | 50.6 / 51.8 tok/s (59.38 / 57.95 ms per round) | 86.3 / 86.2 tok/s (33.36 / 33.37 ms) | 86.8 / 87.3 tok/s (33.05 / 32.88 ms) |
| Short prompts, legal-style set (two runs) | 56.9 / 58.4 tok/s (42.16 / 41.02 ms) | 66.9 / 71.9 tok/s (35.99 / 33.27 ms) | 67.8 / 68.1 tok/s (33.14 / 32.96 ms) |
| Tool loop: first read, 216,245 tokens | 164.0 s | 169.7 s | 179.8 s |
| Tool loop: six tool-call requests | first token 8.7 to 9.4 s; 47.0 to 64.9 tok/s; 12.5 to 15.0 s each | first token 9.5 to 10.1 s; 58.8 to 105.0 tok/s; 12.3 to 13.5 s each | first token 9.9 to 10.7 s; 92.2 to 99.6 tok/s; 12.6 to 13.5 s each |
| Tool loop: final request | ended normally (6,417 tokens) | **did not finish**: 32,000 tokens of thinking, token limit, 5.66 tokens per round | **did not finish**: 32,000 tokens of thinking, token limit, 5.87 tokens per round |
| Closeness check against two-tier, three prompts | PASS | not run | INCONCLUSIVE by the rule ([Quality](#quality)) |

Verdict. The no-host-tier profiles are not accepted: their speeds are measured, their quality is not established.
The closeness check is inconclusive, and the final request of the 216K tool loop ran to the 32,000-token limit in
thinking in both no-host-tier runs (5.66 and 5.87 tokens per round, quality not established). The full count of
that loop is in the next section.

### Long thinking answers: the termination test

MEASURED 2026-11-10, one request at a time, public text, thinking on, temperature 0, a 32,000-token limit on the
final request. One chat is the 216K tool loop [below](#tool-loop-and-turn-boundary): a first read, six tool-call
requests, then the final answer. Chat 0 is that loop's own chat. Chats 1 and 2 take later sections of the same
public source text. The test asks whether the final answer ends or runs to the limit. It also measures repetition
in the answer, in memory, and writes numbers only. The test program is not in this snapshot.

| | Volta sPLIt (earlier build) | This build, accepted configuration |
|---|---|---|
| Chat 0, 216,245 tokens: final answer | ended normally, 10,013 tokens | ended normally, 6,417 tokens |
| Chat 1, 198,307 tokens: final answer | ended normally, 732 tokens | ended normally, 1,499 tokens |
| Chat 2, 204,871 tokens: final answer | ended normally, 1,812 tokens | ended normally, 1,642 tokens |
| Chats that ran to the limit | 0 of 3 | 0 of 3 |
| First read | 69.96 to 73.61 s | 155.56 to 164.9 s |
| Follow-ups: first token | 3.29 to 3.81 s | 8.76 to 9.51 s |

In this termination test 3 of 3 chats ended normally in the accepted configuration, and 3 of 3 on Volta sPLIt.
Chat 0 gave the token counts of the 2026-11-09 run again on both builds: the same context, kept-prompt and output
counts on all eight requests. What three chats do not show: with 0 of 3 at the limit, the 95 % Wilson interval for
the share of chats that run to the limit is 0.000 to 0.561, so the test bounds that rate only below 0.56.

Chat 0 has now run on six configurations of this build. It ended normally only in the accepted configuration, in
three runs. On chat 0, every configuration with a lower share of decode routes on official-precision experts ran
its final answer to the 32,000-token limit in thinking:

| Configuration | Decode routes on official-precision experts by the tally, stage 0 / stage 1 | Chat 0, final answer |
|---|---|---|
| Accepted: host tier, one slot, reserves 4.5 / 7.5 GiB | 96.26 % / 90.71 % | ended normally in three runs (6,417 tokens each). The third ran under the load setting with a profile that pins the same tiers |
| No host tier, one slot (not accepted, quality not established) | 95.26 % / 86.47 % | ran to the limit, 5.66 tokens per round |
| Host tier, one slot, reserves 5.3 / 8.1 GiB, traced (not accepted, quality not established) | 95.08 % / 88.84 % | ran to the limit, 5.21 tokens per round |
| Plan 04 part B variant, host tier, one slot, reserves 5.3 / 8.1 GiB (not accepted, quality not established) | 94.43 % / 88.27 % | ran to the limit, 5.33 tokens per round. Its chats 1 and 2 ended normally (below) |
| Host tier, three slots, reserves 5.3 / 8.1 GiB (not accepted, quality not established) | 94.16 % / 87.46 % | ran to the limit, 5.16 tokens per round (the chat had a label line: 216,255 tokens) |
| No host tier, three slots (not accepted, quality not established) | 92.77 % / 81.98 % | ran to the limit, 5.87 tokens per round |

What this does and does not show. The lower share and the limit go together on chat 0 only. Those are five boots,
and four of them ran chat 0 and no other chat. The fifth, the part B variant (not accepted, quality not
established), then ran chats 1 and 2 on a second boot of the same profile, and both ended normally (749 and 1,491
tokens). With chat 0 it stands at 1 of 3 at the limit, a 95 % Wilson interval of 0.061 to 0.792. So a lower share
does not mean a run to the limit in general, and the rate of the other four on other chats is not measured. The
cause is not established, and chance is not excluded. The engine files of 2026-11-10, which add the part B code
(switched off in the accepted profile), do not explain it at the accepted tiers: the accepted configuration gave
the same token counts on the files of 2026-11-09 and on those of 2026-11-10. The load setting named below does not
explain it either. With a profile that pins the accepted tiers under that setting (not in this snapshot), chat 0
gave the same token counts on all eight requests a third time: first read 167.64 s, final answer 6,417 tokens. A
run to the limit is also not a fault of this build alone. Volta sPLIt ran its final answer to the
32,000-token limit on a separate 31,173-token chat the same day, in a repetition loop (the longest repeated span
covers 18,185 of its 23,336 text units), and once on 2026-11-09 on the shorter loop below (5.6 tokens per round).
The run to the limit of the part B variant (quality not established) is a repetition loop too (25,652 of 26,822
units). The four earlier runs to the limit kept no measure of repetition.

Plan 04 part B at reserves of 5.3 / 8.1 GiB, two boots of one profile (MEASURED 2026-11-10, not accepted, quality
not established). Chat 0: the first read of 216,245 tokens took 107.82 s against 164.9 s in the accepted
configuration, and the seven follow-ups waited 4.02 to 5.38 s for their first token (median 4.44 s) against 8.76 to
9.43 s. Chats 1 and 2: first reads of 102.12 s and 103.0 s against 155.56 s and 161.39 s, and median waits of
4.42 s and 4.26 s. Its 12-prompt closeness check is INCONCLUSIVE ([Quality](#quality)). Its load failure is
explained and fixed. The engine's allocator setting during weight loading fills the cards, the driver error of the
failed boot is reproduced on one card under that setting, and with the setting off the variant boots on eight
cards. That the first boot failed in exactly this way is an inference.

Plan 04 part B at the accepted tiers (MEASURED 2026-11-10, not accepted, quality not established): the variant
booted with the accepted configuration's hot, warm and host experts and ran out of card memory on stage 0 in the
second chunk of its first 216,245-token read (a request for 108 MiB with 64.75 MiB free). INFERENCE, from reading
the code with nothing run: both reading modes peak at the same point, and the variant is short by the workspace it
keeps from load. A fix is planned. It is not built.

### Tool loop and turn boundary

**Agent-style tool loop.** MEASURED 2026-11-09: `harness_loop.py`, Anthropic Messages API format, a
216,245-token first prompt of public source text, thinking on, then seven follow-ups: six tool-call requests and the
final answer.

| | Volta sPLIt (earlier build) | This build, three-tier |
|---|---|---|
| First read, 216,245 tokens | 73.4 s (2,946 tok/s) | 164.0 s (about 1,319 tok/s) |
| All seven follow-ups: prompt kept from the previous request | the whole previous prompt (216,368 to 235,264 tokens) | the whole previous prompt (216,224 to 235,776 tokens) |
| Six tool-call requests: first token | 3.3 to 3.8 s (all seven follow-ups) | 8.7 to 9.4 s |
| Six tool-call requests: decode | 112 to 138 tok/s, 26.1 to 30.4 ms per round (all seven follow-ups) | 47.0 to 64.9 tok/s (60.5 to 78.2 ms per round) |
| Six tool-call requests: wall time each | about 4.6 s | 12.5 to 15.0 s |
| Final answer | included in the rows above | 6,417 tokens at 66.1 tok/s (62.8 ms per round) |

The accepted configuration ran this chat again on 2026-11-10 with the same token counts on every request. Other
configurations ran its final request to the 32,000-token limit in thinking: see
[Long thinking answers](#long-thinking-answers-the-termination-test) and
[docs/measurements.md](docs/measurements.md).

**Turn boundary.** MEASURED 2026-11-09 (`harness_loop.py` `BOUNDARY=1`): a 31,164-token chat of public text with a
three-step tool loop, thinking on, temperature 0, then a new human message. Volta sPLIt rows: 23:38 to 23:46Z.

| New human message | Build | Prompt tokens kept | First token |
|---|---|---|---|
| earlier thinking sent back unchanged | this build, three-tier | 38,432 of 38,469 | 1.11 s |
| earlier thinking stripped from the conversation | this build, three-tier | 0 of 38,396 | 33.08 s |
| earlier thinking sent back unchanged | Volta sPLIt | 38,800 of 38,845 | 0.54 s |
| earlier thinking stripped from the conversation | Volta sPLIt | 0 of 38,193 | 14.07 s |
| through our router after its change | Volta sPLIt | 69,728 of 69,772 | 1.45 s |

The engine keeps a prompt only when the new prompt reproduces the previous sequence to within 124 tokens of its end,
so the program sending the requests must send earlier thinking back unchanged. Our own request router now does so
(last row). Caution: on Volta sPLIt, re-sending this identical chat made follow-ups re-read their whole prompt. In
seven runs of the same chat, 3, 2, 2, 2, 1, 0 and 0 of 3 follow-ups kept their prompt; neither our router (the second
run) nor a 3-second pause (runs 5 and 7) made the difference. With a unique first line per run, 3 of 3 in each of
three runs (MEASURED 2026-11-09). HYPOTHESIS: the engine matches an older copy of the same chat. So every repeat of a
loop test must be a unique chat; the earlier builds' loops here ran once per boot. The final request ran to the
32,000-token limit once (the router run, 5.6 tokens per round) and reached a 3,000-token cap in 5 of 7 later runs:
long thinking on this prompt is common on Volta sPLIt at temperature 0; a 3,000-token cap does not show a runaway.

## What this is

**The question.** DeepSeek-V4.1-Flash-1017 has 384 routed experts in each of 40 layers. At the official MXFP4
precision they take about 269 GiB (ESTIMATED: 384 x 35.4M weights x 40 x 4.25 / 8 bytes). Eight 32 GB V100s have
256 GiB of HBM in total, before attention weights, KV cache and the drafter. Volta sPLIt fits the model by serving
compressed IQ2-class experts. This build asks what it costs to serve the experts at the official precision instead.

**The pack.** NVIDIA's NVFP4 pack stores the routed experts in NVFP4 (FP8 scale per 16 values). Its card says the
conversion from DeepSeek's MXFP4 was lossless. We fold it back to MXFP4 (one power-of-two scale per 32 values) and
check every value: 44,696,862,720 values bit-equal to the NVFP4 formula in FP32, relative L2 error 0 (MEASURED: all
384 experts x 3 projections x 4 TP ranks in layers 5, 12 and 27; experts 0, 191 and 383 in the other 37 layers).
The packs take 299,473,305,600 bytes, W2 padded from K=576 to 640 for the kernel.

**Two tiers.** The experts that route most often stay on the cards. The rest stay in pinned host RAM and the cards
read them over PCIe as mapped host memory. Which experts are "most used" comes from a decode routing tally of our
own use (aggregate counts per layer and expert; private, not published). After the dense weights load, each
pipeline stage measures its free HBM, keeps a per-card reserve for KV, graphs, the drafter and prompt staging, and
fills the rest with the hottest experts. Final boot (MEASURED): stage 0 keeps 5,463 experts on the cards and 2,217
in host RAM, stage 1 4,704 and 2,976 (of 7,680 per stage).

**Three tiers.** Reading an expert from host RAM costs about 1.0 to 1.4 ms per expert per layer (MEASURED, one card).
So a middle "warm" tier runs the less-used experts from the IQ2-class records Volta sPLIt already uses, which are
about half the size and fit on the cards. Hot experts run at official precision on the cards; warm experts
run from IQ2-class records on the cards in decode (their official records stay in host RAM for prompt reading);
the coldest experts run at official precision from host RAM. Boot of 2026-11-09 (MEASURED): stage 0 has 4,107 hot,
3,213 warm and 360 host experts; stage 1 2,695, 4,379 and 606. The warm tier carries 3.74 % (stage 0) and 9.29 %
(stage 1) of decode routes by the private tally (aggregates only), and the host tier 0 % by that tally. So this mode
is not "full precision": 3.7 to 9.3 % of routed-expert calls run from compressed records.

**Everything else is Volta sPLIt's.** Attention, the dense layers, the shared expert, the Engram tables, the DSpark
drafter, the PP2 x TP4 split and the switch set are served exactly as in Volta sPLIt (same engine, same GGUF for the
non-expert tensors). Only the routed-expert path is new.

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

## What runs

| | |
|---|---|
| GPUs | 8x V100-SXM2-32GB on two 4-card SXM2 boards, NVLink inside each board; four PLX switches, each on its own x16 Gen3 root port, two cards per switch at Gen3 x8 (MEASURED from sysfs) |
| Host RAM | 363 GiB available with no server running (MEASURED). Host-RAM experts take about 101 GB in two-tier and 166.8 GB of official records in three-tier (ESTIMATED from the tier counts) |
| Routed experts | [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) @ `480550d3`, folded to MXFP4 (E8M0 scale per 32) packs per layer and TP rank; three-tier warm tier: the GSQ-RCO records below |
| Other weights | as Volta sPLIt: [ISTA-DASLab GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) for attention, dense and shared-expert tensors (Q5_K 4.76 GiB, Q8_0 0.36 GiB, BF16 0.26 GiB, F32 0.15 GiB by tensor bytes, MEASURED from the file header) and the IQ4_NL Engram tables; its routed-expert records are IQ1_M 55.37 GiB, IQ2_XXS 45.68 GiB and IQ2_XS 21.95 GiB; config, tokenizer and DSpark drafter (FP8) from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| Expert kernel | v100-skinny's QPN kernel as packaged in 1Cat-vLLM ([#742](https://github.com/1CatAI/1Cat-vLLM/pull/742)): `skinny_moe_qpn_sm70`, E8M0 scale mode, W4A16, FP16 `mma.sync`; V4.1's limited SwiGLU and the weighted sum in FP32; CUDA graphs for 1 to 6 rows |
| Host tier | experts prepacked once into a page-aligned host arena, registered with CUDA and read in place by the kernel (zero copy) in decode; in prompt reading, staged into two 32-expert HBM buffers on a copy stream, one staging pair per card |
| Warm tier | Volta sPLIt's IQ expert op over the IQ2-class records kept on the cards |
| Engine | the Volta sPLIt engine (1Cat-vLLM #1277 + its patch series and production patches) plus this build: two new files and three changed files ([engine/](engine/)) |
| Layout | PP2 x TP4: layers 0-19 and the Engram lookup on one board, layers 20-39, the head and the drafter on the other |
| Card reserve | per stage, from measured peaks plus about 0.5 GiB: 4.5 GiB per card on stage 0, 7.5 GiB on stage 1 |
| Context | max-model-len 524,288; KV pool 2,076,870 tokens; one request at a time (`max-num-seqs 1`) |
| Serving profiles | [two-tier](engine/configs/serve-nvfp4-pp2.yaml), [three-tier](engine/configs/serve-nvfp4-pp2-t3.yaml) (`hot_iq_host`), [`hot_iq`](engine/configs/serve-nvfp4-pp2-t3n.yaml) (no host tier; speeds measured, quality not established, not accepted), [three prompt slots](engine/configs/serve-nvfp4-pp2-t3-a3.yaml) (`max-num-seqs 3`, reserves 5.3 / 8.1 GiB), [its one-slot control](engine/configs/serve-nvfp4-pp2-t3-a1r.yaml) (same reserves, one slot), and [no host tier with three prompt slots](engine/configs/serve-nvfp4-pp2-t3n-a3.yaml) (`hot_iq`, `max-num-seqs 3`, reserves 5.3 / 8.1 GiB; measured, not accepted). The `hot_iq` profiles are not accepted (see above). The two host-tier slot profiles are measured (rows of 2026-11-10 in the [measurement log](#measurement-log)): three slots on the accepted tiers are not accepted (quality not established), and the one-slot control ran the traced tool loop |

## What worked

All MEASURED on 2026-11-09. Single-card results come from one card of a separate 4-card test host at 150 W.

| Item | Result |
|---|---|
| Exact NVFP4 to MXFP4 fold | 44,696,862,720 values bit-equal, relative L2 0 (above) |
| Scale range on Volta | the engine's FP16 rebase accepted all 61,440 expert records (40 layers x 4 ranks x 384); widest exponent span 8, kernel limit 29 |
| QPN expert layer, one card | worst relative L2 4.78e-4 against an FP32 reference built from the raw NVFP4 weights (layers 5, 12, 27; 1, 6 and 12 rows); CUDA graphs replay bit-equal to eager; graph time 131 us per layer at 1 row, 280 to 348 us at 6 rows |
| Hot and host split | bit-equal to all experts on the card, for natural and forced host routes, decode and prompt reading |
| Three-tier layer and stage checks | passed on one card with real weights; the loader built the tiers in a real 8-card boot with no error |
| Three-tier speed | 1.9 to 2.7 times two-tier's tok/s on every decode row of the table above (same-run pairs) |
| Three-tier closeness to all-native output | PASS on the rule written before the run, three prompts ([Quality](#quality)) |
| Kept prompts in a tool loop | all seven three-tier follow-ups kept the whole previous prompt (216,224 to 235,776 tokens) |
| MXFP4 Engram from NVIDIA's pack | packed to 144-byte rows (110,595,290,400 bytes); sampled rows equal the checkpoint; decoded rows bit-equal to FP32 in FP16. Built and checked, not the default (the IQ4_NL table decodes 4 to 5 times more rows per second in eager mode) |
| Eight-card boot | two-tier and three-tier both boot in about 5 minutes and answer; KV pool 2,076,870 tokens |
| Per-stage reserve | one reserve for both stages ran out of memory on a 100K prompt; reserves set per stage from peaks sampled every 15 s (stage 0 peak 30,456 MiB, stage 1 30,304 MiB on the calibration boot) fixed it, including a 498,099-token prompt |
| Decode flat with context | two-tier 30.2 to 33.7 tok/s from 46 to 303,004 tokens |
| Cause of the general-text gap found | the host-tier profile's slow general text was host-RAM reads: without a host tier (`hot_iq`) general text decodes at 33.4 ms per round against 58.0 to 59.4 ms with it. Speed only: that profile is not accepted (its quality is not established) |
| Three prompt slots, one request at a time | three chats of 30,333 to 31,321 tokens taking turns: 6 of 6 follow-ups kept their own chat's whole prompt (Volta sPLIt with one slot, same day: 0 of 6, each follow-up re-read in 12 to 14 s). Follow-ups: first token 8.7 to 9.3 s, 86.6 to 98.7 tok/s (35.1 to 39.5 ms per round); the three final answers ended normally (3,084 / 1,530 / 3,584 tokens). Measured on the no-host-tier tiers, not accepted (quality not established); the slot mechanism is what this row establishes |
| Slots cost no decode speed | no host tier, not accepted (quality not established); with three slots configured: general 86.8 / 87.3 tok/s (33.05 / 32.88 ms per round), legal-style 67.8 / 68.1 (33.14 / 32.96), against 33.36 / 33.37 and 35.99 / 33.27 ms with one slot. The price is card memory: fewer hot experts, so warm shares of 7.23 % / 18.02 % against 4.74 % / 13.53 % |
| Long prompt with three slots | no host tier, not accepted (quality not established): a 257,706-token prompt read in 214.4 s (1,202 tok/s) with the server healthy; highest card memory 31,400 MiB (stage 0) and 30,118 MiB (stage 1) |

## What did not work or is not yet good

All MEASURED on 2026-11-09 unless a row gives another date or is marked ESTIMATED.

| Item | Result |
|---|---|
| Slower than Volta sPLIt | three-tier decodes at about half Volta sPLIt's speed on legal-style text and about 37 % on general text; two-tier at 14 to 26 % |
| Tool-loop requests | 12.5 to 15.0 s per tool-call request (six requests) on three-tier against about 4.6 s on Volta sPLIt; first token 8.7 to 9.4 s against 3.3 to 3.8 s, although the whole prompt was kept |
| Generation with thinking on | in the tool loop, with thinking on, three-tier generated at 47.0 to 64.9 tok/s on the six tool-call requests and 66.1 tok/s on the final answer. Its time per round, 60.5 to 78.2 ms, is nearer its general-text time (58.0 to 59.4 ms per round) than its legal-style time (41.0 to 42.2 ms). More drafted tokens were accepted per round, which kept its tok/s up: 3.4 to 4.3 tokens per round in the tool loop (arithmetic: tok/s x ms per round, both decode only there) against 3.15 on the general short set, which is output tokens divided by rounds (the short-set tok/s is over the whole request, so tok/s x ms per round gives less there) |
| Prompt reading | 41 to 56 % of Volta sPLIt's rate at 30,795 and 106,531 tokens. Every expert that is not hot is staged from host RAM while reading, so three-tier reads 1,080 to 1,327 tok/s, slower than two-tier. The mode that runs warm experts from their IQ2-class records while reading ([plan 04](docs/plans/04-prefill-staging.md) Part B) was built and passed its CPU tests and a one-card check. Its first eight-card boot (2026-11-10) failed while loading. That failure is explained and fixed (the engine's allocator setting during weight loading, reproduced on one card, and with the setting off the variant boots). MEASURED on that boot, 2026-11-10, one slot, reserves 5.3 / 8.1 GiB: first read of 216,245 tokens in 107.82 s against 164.9 s, and a median wait of 4.44 s before the first token of a follow-up against 8.76 to 9.43 s. On a second boot, chats 1 and 2 of the termination test: first reads 102.12 s and 103.0 s, and both final answers ended normally (749 and 1,491 tokens). Not accepted: its quality is not established. Its 12-prompt closeness check is INCONCLUSIVE, and its long-thinking request on chat 0 ran to the 32,000-token limit (1 of 3 chats at the limit, 95 % Wilson interval 0.061 to 0.792). Not in this snapshot |
| Prompt-reading variant at the accepted tiers | MEASURED 2026-11-10 (plan 04 part B, not accepted, quality not established): with the accepted configuration's hot, warm and host experts the variant ran out of card memory on stage 0 in the second chunk of its first 216,245-token read (a request for 108 MiB with 64.75 MiB free), and the server ended by itself. INFERENCE, from reading the code with nothing run: both reading modes peak at the same point, and the variant is short by the workspace it keeps from load. A fix is planned, not built. So the faster reading is measured only with fewer hot experts (reserves 5.3 / 8.1 GiB) |
| Slow first token | short prompts: 0.69 s (two-tier) and 0.84 to 1.07 s (three-tier) against 0.37 to 0.45 s. A 216K-token prompt read in full: about 136 s (two-tier) and 164.0 s (three-tier) against 73.4 to 78 s |
| Reading experts from host RAM | one host-RAM expert per token at 1 row adds about 975 us to a layer; at 6 rows each costs about 1.0 to 1.4 ms per layer (about 3.5 to 4.9 GB/s over PCIe). Other miss paths on one card: copy-engine DMA 743.7 us per expert, GPU gather 1,014.8 us, AVX2 host kernel 2,953.7 us across 4 TP slices. ESTIMATED from those and the tally: no path comes within about 10 ms per round of all experts on the cards. That is why three-tier exists |
| General text with a host tier | the hot set comes from a private routing tally (not published), and general text misses it more often. With a host tier, general text decodes about twice as slowly as legal-style text on two-tier and 17 ms per round slower on three-tier. The cause is host-RAM reads (see "What worked"); removing the host tier removes the gap but its quality is not yet established (next row) |
| No-host-tier profiles not accepted | the closeness check on the three-slot tiers is INCONCLUSIVE by the rule: speeds measured, quality not established. In the 216K tool loop the final request ran to the 32,000-token limit in thinking in both no-host-tier runs (5.66 and 5.87 tokens per round), all at temperature 0 |
| Long thinking answers | termination test, MEASURED 2026-11-10: in the accepted configuration 3 of 3 chats of 198,307 to 216,245 tokens ended normally (6,417, 1,499 and 1,642 tokens), as on Volta sPLIt (10,013, 732 and 1,812). Three chats bound the share of chats that run to the limit only below 0.56 (95 % Wilson interval 0.000 to 0.561). On chat 0, every configuration of this build with a lower share of routes on official-precision experts ran the final answer to the 32,000-token limit: five boots, none accepted, quality not established. That holds for chat 0 only. One of the five, the part B variant, ended chats 1 and 2 normally and stands at 1 of 3 at the limit (95 % Wilson interval 0.061 to 0.792). The other four ran chat 0 and no other chat. The cause is not established. Volta sPLIt itself ran to the limit in a repetition loop on a separate 31,173-token chat ([details](#long-thinking-answers-the-termination-test)) |
| Closeness on a wider prompt set | the 12-prompt closeness check of 2026-11-10 is INCONCLUSIVE for the accepted configuration (first different token: 11 for, 1 against, and mean difference: 7 for, 0 against, 5 undecided), for three slots on the accepted tiers, not accepted, quality not established (10 for, 2 against, and 7 for, 0 against, 5 undecided), and for the part B variant at reserves 5.3 / 8.1 GiB, not accepted, quality not established (10 for, 2 against, and 5 for, 1 against, 6 undecided). The rule, 8 of 12 on each part, was fixed before the run ([Quality](#quality)) |
| Memory margin | at the final reserves the highest card memory seen was 32,086 MiB (stage 0) and 31,356 MiB (stage 1) against a limit of about 32,494 MiB: about 0.4 and 1.1 GiB of margin |
| Warm records are not native | on synthetic inputs an expert's IQ2-class record is far from its official record (relative L2 0.60 to 0.69 in layers 5, 12 and 27). The closeness check on real text passed on three prompts and is INCONCLUSIVE on the 12-prompt set of 2026-11-10 |
| First boots | each failed fast and was fixed at its cause: a host mapping requested on card 0 for every worker, prompt staging buffers kept per layer instead of per card (out of memory), and one shared reserve for both stages (out of memory on a 100K prompt) |

## Next

The ranked campaign and the research notes behind it follow in the next update. Each piece is built and measured
alone, biggest expected improvement first. Done so far: three prompt slots work (no host tier; "What worked"; on the
accepted tiers they are measured but not accepted, quality not established), our router now sends earlier thinking
back unchanged, the termination test for long thinking answers exists and has run, and the load failure of the
prompt-reading variant of plan 04 is fixed. Next, first: that variant at the accepted tiers, after the planned
fix for its memory shortfall, with the termination test and the closeness capture. Then the replacement prompts
of the 12-prompt closeness set (it stays INCONCLUSIVE until then), the quality of the no-host-tier profiles, and
fused kernels for route grouping and SwiGLU. The detailed plans are in [docs/plans/](docs/plans/), and the measured research on
where the time goes is [docs/research/LANE-RESEARCH.md](docs/research/LANE-RESEARCH.md). Our target is 80 to 100
tok/s at official expert precision on these eight cards.

## Quality

**Three-tier against all-native output: PASS on three prompts.** MEASURED 2026-11-09 with
[quality_t3.py](harness/benchmarks/nvfp4_8r/quality_t3.py): the three public prompts of the Volta sPLIt harness's
general set, 1,024 tokens each, greedy, thinking off. The reference is this build's two-tier output (every routed
expert at official precision), which repeated token for token on a second capture in the same boot.

| Against the two-tier reference | Prompt 1 | Prompt 2 | Prompt 3 |
|---|---|---|---|
| Three-tier: first different token | 25 | 6 | 12 |
| Three-tier: mean absolute log-probability difference | 0.0486 | 0.0026 | 0.0327 |
| Volta sPLIt (compressed experts everywhere): first different token | 0 | 10 | 5 |
| Volta sPLIt: mean absolute log-probability difference | not defined (no common prefix) | 0.0392 | 0.1429 |
| No host tier, three slots: first different token | 25 | 45 | 5 |
| No host tier, three slots: mean absolute log-probability difference | 0.0710 | 0.0722 | 0.0796 |

The log-probability difference is that of the chosen token, averaged over the tokens before the first difference.
The rule, written before the run ([plan 01](docs/plans/01-three-tier-residency.md) section 8): three-tier must
diverge no earlier than Volta sPLIt on two of three prompts (prompts 1 and 3) and have the smaller mean difference on
two (prompts 2 and 3). Both hold. Limits: three prompts, greedy decoding, thinking off. This is a closeness check
against the all-native output, not a benchmark score.

**No host tier with three slots: INCONCLUSIVE by the same rule.** MEASURED 2026-11-09. The divergence part holds on
three of three prompts (25, 45 and 5 against 0, 10 and 5). The mean part needs the smaller mean difference on two
prompts: it is below Volta sPLIt's on prompt 3, above on prompt 2, and undecidable on prompt 1 (Volta sPLIt has no
common prefix there), so the rule cannot decide. Its mean differences are also larger than the host-tier profile's on
all three prompts.

**Twelve prompts (2026-11-10): INCONCLUSIVE, stated as such.** MEASURED on a wider set of twelve prompts invented
for this check (code 3, JSON 2, general prose 3, legal-style prose 2, reasoning 2), 1,024 tokens each, greedy,
thinking off, against the two-tier reference, which again repeated token for token on a second capture. The rule
was fixed before the run. Each of the two parts must hold on at least 8 of the 12 prompts. A part is refused when 5
prompts go against it. Anything else is INCONCLUSIVE.

| Against Volta sPLIt | First different token no earlier | Smaller mean log-probability difference | Result |
|---|---|---|---|
| Accepted configuration | 11 for, 1 against | 7 for, 0 against, 5 undecided | INCONCLUSIVE |
| Three slots on the accepted tiers (not accepted, quality not established) | 10 for, 2 against | 7 for, 0 against, 5 undecided | INCONCLUSIVE |
| Plan 04 part B variant, reserves 5.3 / 8.1 GiB (not accepted, quality not established) | 10 for, 2 against | 5 for, 1 against, 6 undecided | INCONCLUSIVE |

The first part holds for all three. The second part cannot decide, so the result of the 12-prompt set is
INCONCLUSIVE: on five prompts Volta sPLIt differs from the reference at the first token, so there is no common
prefix and no mean difference to compare with, and seven prompts cannot reach eight. The part B variant itself
differs from the reference at the first token on three prompts, one of them outside those five, so six are
undecided for it. The rule was not changed after the data. No prompt went against the accepted configuration on
the second part, and one on the first. The accepted configuration stays accepted on the three-prompt PASS above.
Three slots and the part B variant stay not accepted. Next: replace the five prompts on which Volta sPLIt shares no
prefix with the reference and score again.
The per-prompt figures are in [docs/measurements.md](docs/measurements.md). The program of
the wider set is not in this snapshot. Its receipts hold token ids and log-probabilities only.

The routed experts in two-tier are value-exact to DeepSeek's official MXFP4 experts (above). NVIDIA's own accuracy
table for the pack, measured with vLLM on 4x GB300 (UPSTREAM), is within run noise of the MXFP4 source, for example
GPQA-D 92.93 against 92.77 and AA-LCR 80.19 against 80.41. Those figures are NVIDIA's, on NVIDIA's serving stack,
not this build.

## Measurement log

One row per measurement, newest first. Later results are added at the top with their date; the tables above are
updated when a row changes them. Details: [docs/measurements.md](docs/measurements.md).

| Date | Build or profile | Test | Result | Basis |
|---|---|---|---|---|
| 2026-11-10 | IQ records in prompt reading (plan 04 part B) at the accepted tiers: NOT accepted (quality not established) | first read of chat 0 (216,245 tokens) | booted with the accepted configuration's hot, warm and host experts, then ran out of card memory on stage 0 in the second chunk of the read (a request for 108 MiB with 64.75 MiB free). The server ended by itself. INFERENCE from the code, nothing run: both reading modes peak at the same point and the variant is short by the workspace it keeps from load. A fix is planned, not built | MEASURED, boot log |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 10 for, 2 against, and mean difference 5 for, 1 against, 6 undecided ([Quality](#quality)) | MEASURED, greedy, rule (8 of 12) fixed before the run |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | termination test, chats 1 and 2 (198,307 and 204,871 tokens), second boot | both final answers ended normally (749 and 1,491 tokens). First reads 102.12 s and 103.0 s (accepted configuration: 155.56 s and 161.39 s). 7 of 7 follow-ups kept the prompt in each chat, first token median 4.42 s and 4.26 s. With chat 0 the variant stands at 1 of 3 at the limit (95 % Wilson interval 0.061 to 0.792) | MEASURED, public text, temperature 0 |
| 2026-11-10 | accepted tiers under the load setting (a profile that pins them, not in this snapshot) | termination test, chat 0 (216,245 tokens) | the same hot, warm and host experts as the accepted configuration. First read 167.64 s. The final answer ended normally with 6,417 tokens, and all eight requests gave the token counts of the two earlier runs. On this chat the setting did not change the token counts | MEASURED, public text, temperature 0 |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | termination test, chat 0 (216,245 tokens) | first read 107.82 s (accepted configuration: 164.9 s). 7 of 7 follow-ups kept the prompt, first token 4.02 to 5.38 s, median 4.44 s (accepted configuration: 8.76 to 9.43 s). Tool-call requests decoded at 49.4 to 62.6 tok/s. The final answer ran to the 32,000-token limit in thinking, a repetition loop (5.33 tokens per round) | MEASURED, public text, temperature 0 |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B): NOT accepted (quality not established) | load failure, and an eight-card boot with the allocator setting off | explained and fixed: under the engine's allocator setting during weight loading the cards fill (one card: 0.038 GiB free after the build against 7.696 GiB with the setting off), and a probe on one card gives the failed boot's driver error "invalid argument" under that setting. With the setting off the variant was ready in 349 s with no error. That the first boot failed in exactly this way is an inference. Not in this snapshot | MEASURED, one card and eight cards |
| 2026-11-10 | three slots on the accepted tiers: NOT accepted (quality not established) | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 10 for, 2 against, and mean difference 7 for, 0 against, 5 undecided ([Quality](#quality)) | MEASURED, greedy, rule (8 of 12) fixed before the run |
| 2026-11-10 | accepted configuration | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 11 for, 1 against, and mean difference 7 for, 0 against, 5 undecided, because Volta sPLIt differs from the reference at the first token on five prompts ([Quality](#quality)) | MEASURED, greedy, rule (8 of 12) fixed before the run |
| 2026-11-10 | accepted configuration | termination test, three chats of 198,307 to 216,245 tokens | 3 of 3 final answers ended normally (6,417, 1,499 and 1,642 tokens). Chat 0 gave the token counts of the 2026-11-09 run again. Three chats bound the share that runs to the limit only below 0.56 (95 % Wilson interval 0.000 to 0.561) | MEASURED, public text, temperature 0 |
| 2026-11-10 | this build, all configurations | chat 0 of the termination test (the 216K tool loop) | ended normally in the accepted configuration (three runs). Ran to the 32,000-token limit on chat 0 in the five boots with a lower share of routes on official-precision experts (none accepted, quality not established). One of the five, the part B variant, ended chats 1 and 2 normally (rows above), so this holds for chat 0 only. Cause not established ([table](#long-thinking-answers-the-termination-test)) | MEASURED, temperature 0 |
| 2026-11-10 | Volta sPLIt | termination test, the same three chats, and a 31,173-token chat | 3 of 3 final answers ended normally (10,013, 732 and 1,812 tokens), with the same bound, below 0.56. On the separate 31,173-token chat the final answer ran to the 32,000-token limit in a repetition loop | MEASURED, public text, temperature 0 |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B) | first eight-card boot | failed while loading: driver error "invalid argument" during the build of the IQ bank. Explained and fixed later the same day (rows above). Not in this snapshot | MEASURED, boot log |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | closeness, three prompts | INCONCLUSIVE by the rule | MEASURED, greedy, three public prompts |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | tool loop, 216,255-token prompt | first read 173.5 s; 7 of 7 follow-ups kept the prompt, first token median 9.62 s; the final request ran to the 32,000-token limit (5.16 tokens per round) | MEASURED, public text, temperature 0 |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | short sets, three chats, 257,741-token prompt | general 50.9 / 51.3 tok/s, legal-style 63.9 / 64.2 tok/s; 6 of 6 follow-ups kept their own chat's prompt; 257,741 tokens read in 207.1 s; highest card memory 31,704 MiB | MEASURED, public text |
| 2026-11-10 | three-tier, one slot, reserves 5.3 / 8.1 GiB, traced | tool loop, 216,245-token prompt | first read 171.9 s; 7 of 7 kept; the final request ran to the 32,000-token limit (5.21 tokens per round); host staging 86 % of prompt-reading expert time (trace; traced speeds are slowed) | MEASURED, public text, temperature 0 |
| 2026-11-09 | Volta sPLIt | short tool loop, ten runs | seven runs of the identical chat: 3, 2, 2, 2, 1, 0 and 0 of 3 follow-ups kept their prompt (run 2 through our router; a 3-second pause in runs 5 and 7 changed nothing); three runs with a unique first line: 3 of 3 each. HYPOTHESIS: the engine matches an older copy of the same chat, so every repeat of a loop test must be a unique chat. Final request: the 32,000-token limit once (the router run, 5.6 tokens per round); a 3,000-token cap reached in 5 of 7 later runs (long thinking on this prompt is common at temperature 0; the cap does not show a runaway) | MEASURED |
| 2026-11-09 | Volta sPLIt | turn boundary, 31,164-token chat | earlier thinking kept: 38,800 of 38,845 kept, first token 0.54 s; stripped: 0 of 38,193, 14.07 s; through our router after its change: 69,728 of 69,772, 1.45 s | MEASURED |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | tool loop repeat, 216K prompt | six tool-call requests 12.6 to 13.5 s each, 92.2 to 99.6 tok/s; the final request again ran to the 32,000-token limit in thinking (5.87 tokens per round) | MEASURED |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | closeness, three prompts | INCONCLUSIVE by the rule ([Quality](#quality)) | MEASURED |
| 2026-11-09 | no host tier, three slots | 257,706-token prompt | not accepted (quality not established); read in 214.4 s (1,202 tok/s); highest card memory 31,400 / 30,118 MiB | MEASURED |
| 2026-11-09 | no host tier, three slots | short sets | not accepted (quality not established): general 86.8 / 87.3 tok/s, legal-style 67.8 / 68.1 tok/s, no slowdown against one slot | MEASURED |
| 2026-11-09 | three prompt slots (no host tier) | three chats taking turns | WORK: 6 of 6 follow-ups kept their own chat's whole prompt (Volta sPLIt with one slot: 0 of 6); the no-host-tier tiers are not accepted (quality not established) | MEASURED |
| 2026-11-09 | no host tier (`hot_iq`): NOT accepted (quality not established) | short sets, tool loop | general 86.3 / 86.2 tok/s, legal-style 66.9 / 71.9 tok/s; tool-call requests 12.3 to 13.5 s each. Caution: the loop's final request did not finish (32,000 tokens of thinking, token limit); closeness check not run ([details](#tokens-per-second)) | MEASURED |
| 2026-11-09 | two-tier on the current code | short set, general | 18.3 tok/s (178.95 ms per round) against 18.8 tok/s (174.64 ms) on the first earlier run: 2.5 % per round, inside the 6.5 % within-boot noise | MEASURED |
| 2026-11-09 | three-tier against two-tier | closeness, three prompts | PASS ([Quality](#quality)) | MEASURED |
| 2026-11-09 | three-tier | turn boundary, 31,164-token chat | earlier thinking kept: first token 1.11 s; stripped: 33.08 s | MEASURED |
| 2026-11-09 | Volta sPLIt, three-tier | tool loop, 216K-token prompt, thinking on | [tool-loop table](#tokens-per-second): 12.5 to 15.0 s per request against about 4.6 s | MEASURED |
| 2026-11-09 | two-tier | 498,099-token prompt | 274.9 s (reserves 6 / 9 GiB), 326.6 s (4.5 / 7.5 GiB); no out-of-memory error | MEASURED |
| 2026-11-09 | Volta sPLIt, two-tier, three-tier | short sets, context rows, prompt reading | the [tokens-per-second table](#tokens-per-second) | MEASURED, same day |

## Hardware

Eight V100-SXM2-32GB on two NVLink quad boards, the same host and settings as the Volta sPLIt measurements of the
same day. Each of the four PCIe switches has its own x16 Gen3 root port and two cards at Gen3 x8. Volta has FP16
`mma.sync.m8n8k4` only: no FP4 or FP8 tensor cores, no `cp.async`, no `ldmatrix`. Every 4-bit format is decoded in
software into FP16 operands, so low-bit experts pay by reading fewer bytes, not by faster math. v100-skinny measured
a read ceiling of about 879 GB/s on a V100-SXM2 and its QPN kernel at 77 % of it at 1 row (UPSTREAM).

## Reproduce

1. Build the Volta sPLIt engine as its
   [Reproduce section](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#reproduce)
   describes. Then apply [engine/patches/](engine/patches/) in the engine source root (`patch -p1`; each header gives
   the sha256 of the base file and of the result) and copy the two new files from [engine/src/](engine/src/) into
   the same paths of the vllm tree. See [engine/README.md](engine/README.md).
2. Download NVIDIA's NVFP4 pack (revision `480550d3`), the ISTA-DASLab GSQ-RCO GGUF and the DeepSeek config,
   tokenizer and DSpark drafter. [harness/run/copy_weights.sh](harness/run/copy_weights.sh) checks every pack file
   against the Hugging Face sha256; the pack builder refuses to start without its `ALL-VERIFIED` log.
3. Provide a decode routing tally: routing counts per layer and expert from your own traffic, in the array layout
   that `choose_residency` and `choose_tiers` read (see the engine files). Ours derives from private use and is not
   published.
4. Build and check the packs on CPU: `build_packs` then `accept_p1` (bit equality), as in
   [harness/benchmarks/nvfp4_8r/run_p1.sh](harness/benchmarks/nvfp4_8r/run_p1.sh).
5. Lay out the run directory. The scripts were measured in this layout and find each other there, not in this
   repository's folders. `$LANE` is this build's root (with `engine/venv/`, `engine/src/`, `models/` and `packs/`),
   `$PROD` the Volta sPLIt checkout (its bench5 harness scripts). From this repository's root:

   ```bash
   export LANE=/path/to/build PROD=/path/to/volta-split
   mkdir -p "$LANE/run/state" "$LANE/run/cache" "$LANE/logs/baseline" "$LANE/engine/src/benchmarks/nvfp4_8r"
   cp engine/configs/* harness/run/* "$LANE/run/"
   cp harness/benchmarks/nvfp4_8r/* "$LANE/engine/src/benchmarks/nvfp4_8r/"
   ```

   Copy Volta sPLIt's run tools into `$LANE/run/tools/` (`env.sh` puts them on the Python path), and replace `$LANE`
   in the profiles with your path. When `LANE` or `PROD` is not set, the shell scripts that read it stop with a
   message and the Python files stop with a `KeyError` naming the variable.
6. Start with `cd "$LANE/run" && PROFILE=serve-nvfp4-pp2-t3.yaml CUDA_VISIBLE_DEVICES=... bash run_serve.sh nvfp4` (the last word names the server's log and pid files).
   The three-slot profiles (`serve-nvfp4-pp2-t3-a3.yaml`, `serve-nvfp4-pp2-t3n-a3.yaml`) also need `NVFP4_SLOTS=3`
   on that command line: `run_serve.sh` refuses a profile whose `max-num-seqs` differs from it.
7. Measure with `$LANE/run/`[measure_boot.sh](harness/run/measure_boot.sh) (short sets, long prompts, repeat),
   [ctx_tput.py](harness/run/ctx_tput.py) (context rows) and [harness_loop.py](harness/run/harness_loop.py) (tool
   loop). The legal-style prompt file is not published; any file of one public prompt per line works, saved as
   `$LANE/logs/baseline/legal-public-prompts.txt`.

## Repository layout

| Path | What it is |
|-|-|
| `docs/` | [measurements](docs/measurements.md), [plans](docs/plans/), [research](docs/research/LANE-RESEARCH.md) (where the time goes); [index](docs/README.md) |
| `engine/src/` | the two new engine files: the native expert path and the tier planner |
| `engine/patches/` | one patch per changed engine file (loader, GGUF MoE method, Engram reader), against the Volta sPLIt engine |
| `engine/configs/` | launcher, environment and serving profiles |
| `harness/run/` | measurement scripts: boots, short sets, long prompts, context rows, harness-shaped tests, card memory |
| `harness/benchmarks/nvfp4_8r/` | pack builder, acceptance checks, single-card benches and CPU tests |
| `tools/` | `check_data_boundary.sh` (pre-publication scan) |
| `.publication/SOURCE.json` | what each file came from and what was left out |

## Work with PLI Labs

[PLI Labs](https://proprietarylegal.ai/labs.html) is the open research arm of Proprietary Legal Intelligence. PLI builds
AI around an organization's own work product: retrieval over its documents, staged workflows, validation of every output
and attorney review before anything goes out. This stack is how we run that privately, on hardware we own. If you want
private, on-prem LLM serving of your own, from choosing the hardware to the measured stack and the workflows on top, we
consult on exactly that: [proprietarylegal.ai](https://proprietarylegal.ai).

## Licence

Original documentation and code are Apache-2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE). Patches against third-party
code keep that code's licence. No model weights, packs, wheels, compiled libraries, routing tallies or prompt corpora are
distributed here.

## Credits

This build stands on other people's work:

- **DeepSeek**: DeepSeek-V4.1-Flash-1017, its official MXFP4 experts, the DSpark drafter and the Engram design.
- **NVIDIA**: the DeepSeek-V4.1-Flash-1017-NVFP4 pack whose experts this build serves, and its accuracy table.
- **1CatAI / 1Cat-vLLM**: the Volta vLLM fork; [#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277) (yangzhuxinyzx),
  the DeepSeek-V4.1 package Volta sPLIt ports; [#742](https://github.com/1CatAI/1Cat-vLLM/pull/742) (Peuqui), the
  `sm70_skinny` QPN MoE backend this build calls; [#1282](https://github.com/1CatAI/1Cat-vLLM/pull/1282) (Peuqui),
  tensor-core Triton prefill, a candidate in our campaign; [#1387](https://github.com/1CatAI/1Cat-vLLM/pull/1387) (ours),
  the grouped prefill GEMM for IQ experts that the warm tier's prompt-reading work builds on.
- **Dennis Vertzyas (dnv2003), [v100-skinny](https://github.com/dnv2003/v100-skinny)**: the QPN kernels and the Volta
  bandwidth findings.
- **ISTA-DASLab**: the GSQ-RCO GGUF used for the non-expert tensors and the warm tier. **ggml**: the GGUF and IQ formats.
- **The vLLM project**: the serving engine.
- **Volta sPLIt** (PLI Labs), and every project it credits: the engine, switch set and harness underneath.

Every source, with what we took from it, is in [CREDITS.md](CREDITS.md). If you build on this repository, please credit
them as well.
