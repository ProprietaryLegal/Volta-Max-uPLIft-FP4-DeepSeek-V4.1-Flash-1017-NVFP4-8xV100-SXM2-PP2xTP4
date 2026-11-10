# Volta Max uPLIft FP4

## Volta Max uPLIft: as much of the routed work at NVIDIA's NVFP4 precision as fits on eight V100s, every routed expert on the cards in decode · 6,087 of 15,360 routed experts at exactly NVFP4 precision on the mixed tally · 79 to 85 % of decode routes in the measured sets (in-sample) · 64.7 tok/s legal-style, 83.6 general · 8x V100 from 2018 · slower than Volta sPLIt

[![decode routes at NVFP4 precision: 79-85 %](https://img.shields.io/badge/routes%20at%20NVFP4%20precision-79--85%20%25-76B900?style=for-the-badge)](#how-close-to-nvfp4)
[![Max uPLIft legal-style: 64.7 tok/s](https://img.shields.io/badge/Max%20uPLIft%20legal--style-64.7%20tok%2Fs-1F6FEB?style=for-the-badge)](#hot_iq-on-the-mixed-tally-the-build-of-this-release)
[![Max uPLIft general: 83.6 tok/s](https://img.shields.io/badge/Max%20uPLIft%20general-83.6%20tok%2Fs-8250DF?style=for-the-badge)](#hot_iq-on-the-mixed-tally-the-build-of-this-release)
[![31K tool loops: 88-91 tok/s](https://img.shields.io/badge/31K%20tool%20loops-88--91%20tok%2Fs-2EA043?style=for-the-badge)](#hot_iq-on-the-mixed-tally-the-build-of-this-release)
[![Hugging Face package: upload in progress](https://img.shields.io/badge/%F0%9F%A4%97%20package-upload%20in%20progress-FFD21E?style=for-the-badge)](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4)
[![speed bench, code 1K to 256K: 104-127 tok/s](https://img.shields.io/badge/speed%20bench%2C%20code%201K%20to%20256K-104--127%20tok%2Fs-0969DA?style=for-the-badge)](#tokens-per-second)
[![status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-D29922?style=for-the-badge)](#what-did-not-work-or-is-not-yet-good)

Built by **Nick Mermiges, Esq.**, founder of [PLI Labs](https://proprietarylegal.ai/labs.html) · [proprietarylegal.ai](https://proprietarylegal.ai)

[![proprietarylegal.ai](https://img.shields.io/badge/proprietarylegal.ai-website-0B3D91)](https://proprietarylegal.ai) [![PLI Labs](https://img.shields.io/badge/PLI%20Labs-research-1F6FEB)](https://proprietarylegal.ai/labs.html) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![Hardware: 8x V100-SXM2-32GB](https://img.shields.io/badge/hardware-8x%20V100--SXM2--32GB-76B900)](#hardware) [![Earlier build: Volta sPLIt](https://img.shields.io/badge/earlier%20build-Volta%20sPLIt-555555)](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4) [![Hugging Face: hot_iq package, upload in progress, do not download yet](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-hot__iq%20package%2C%20upload%20in%20progress-FFD21E)](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4) [![Related build: Volta Full NVFP4](https://img.shields.io/badge/related%20build-Volta%20Full%20NVFP4-555555)](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100)

**Volta Max uPLIft is the build that puts as much of the routed work at NVIDIA's NVFP4 precision as fits on eight V100
cards, with every routed expert held on the cards.** The configuration key is `hot_iq`. Every routed expert's compressed
IQ2-class record is placed on the cards first. Then experts are promoted to their NVFP4 weights in order of use per
byte, one by one, until the next one does not fit. This maximizes use, not the count. Packing the cheapest promotions
first would fit 6,377 to 6,379 experts at NVFP4 against 6,087, but less-used ones, so it could not carry more of the
work at NVFP4 (to within one expert's share on each pipeline stage, [below](#hot_iq-on-the-mixed-tally-the-build-of-this-release)). Use and work here are the mixed
tally's counts of decode routes, measured in-sample. On the mixed tally of this release that puts 6,087 of the 15,360 routed experts at exactly the
precision of NVIDIA's [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) pack,
unchanged. In the measured sets 79.0 to 85.0 % of decode routes ran on those experts (route counters, in-sample,
[below](#hot_iq-on-the-mixed-tally-the-build-of-this-release)). The other 15.0 to 21.0 % ran on ISTA-DASLab's IQ2-class
records. In decode all experts are in VRAM: no routed expert is read over PCIe. Prompt reading is different: it
reads the compressed experts at NVFP4 precision from records staged in host RAM (`native_staged`). How close the
output is to the all-NVFP4 reference is the next section.

The pack is served on eight V100-SXM2-32GB cards (two 4-card NVLink boards, pipeline parallel 2 x tensor parallel 4).
The promoted ("hot") routed experts run at the precision of DeepSeek's official weights: NVFP4 folds exactly to MXFP4 (4.25 bits
per weight), which the v100-skinny QPN kernel runs as W4A16. Two related builds serve the same model on the same
cards ([Related builds](#related-builds)). [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100) runs every routed expert at NVFP4
precision, the less-used ones from host RAM over PCIe, and is slower. Our earlier build,
[Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4), runs every routed
expert from a compressed IQ2-class record and is faster. The three-tier mode was an earlier step on the way to this
build, with part of the experts in host RAM. Its measurements are in the [history](docs/history.md).

**One download, upload in progress:** the [hot_iq package on Hugging Face](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4) (556 GB) will hold
every weight file this build serves, the tier map that says which experts are hot, and the serving profile. As of
2026-11-10 the weights are still uploading and the file list there is incomplete, so do not download it yet. A
dated line here and on the package card will say when the upload is complete. Then start it with
[engine/configs/serve_hot_iq.sh](engine/configs/serve_hot_iq.sh) as [Reproduce](#reproduce) describes.

This repository is a public record of work in progress: the measurements, the code of the build as measured, what
worked and what did not. The steps that led to this build, the three-tier mode and the first `hot_iq` profiles
on the first tally (2026-11-09 and the earlier results of 2026-11-10), are dated rows of the
[measurement log](#measurement-log), with their tables and checks in [docs/history.md](docs/history.md). They stay
as published. The unfinished prompt-reading variant of plan 04 part B is not in this snapshot.
The build is slower than Volta sPLIt in decode speed and in prompt reading speed. We publish it now because
the method works, the numbers are measured, and the next steps are planned piece by piece. New results are added as
dated rows in the [measurement log](#measurement-log).

## How close to NVFP4

**The closeness test is designed and its rules are public. Its results are not in yet.** It compares this
build's output with NVIDIA's NVFP4 pack served with every routed expert at its own NVFP4 precision (the reference,
[Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100)), with Volta sPLIt as the yardstick: tokens 2 to 32 of greedy continuations on 320 contexts of five kinds
(same next token, log-probability gap, KL divergence, first difference) and 513 MMLU questions, on the same eight
cards. Its rules and the sha256 of its scripts were fixed on 2026-11-10 at 19:54 UTC, before any capture, in
[PREREGISTRATION.txt](quality/closeness-test/PREREGISTRATION.txt). The captures run on 2026-11-10 and 2026-11-11. The results will be added here and to the measurement
log as dated rows, whatever they show.
The scripts, the item ids, the sources and the rules are in [quality/closeness-test/](quality/closeness-test/).
The check on the replaced 12-prompt set, which this build passed, is in [Quality](#quality).

## hot_iq on the mixed tally: the build of this release

The full speed bench of this build, prompts of 1K to 256K tokens and five kinds of text, is
[Tokens per second](#tokens-per-second).

MEASURED 2026-11-10 on the 8-card host, one request at a time, temperature 0, card clock ceiling 1,425 MHz (the
2026-11-09 rows of the measurement log ran at 1,522 MHz). Profile: `hot_iq`, one prompt slot, card reserves 4.5 / 7.5 GiB, prompts
read with the full-precision records (`native_staged`). Tiers from the mixed tally: 3,833 hot and 3,847 compressed
experts on stage 0, 2,254 and 5,426 on stage 1 (tier digests `8e7829184090` and `5b09c0383a80`, no host tier).
The planner keeps every compressed record on the cards, then promotes experts in order of use per byte added and
stops at the first promotion that does not fit, so less than one promotion's bytes of the budget stay unused. This
is not the largest number of NVFP4 experts that fits. A promotion costs the native record less the compressed
record it replaces, and the compressed record is IQ1_M, IQ2_XXS or IQ2_XS by layer, so the cost differs by layer
(2.32 to 2.94 MB per card). Promoting the cheapest experts first, whatever their use, would fit 3,982 or 3,983 on
stage 0 and 2,395 or 2,396 on stage 1, 6,377 to 6,379 in all against 6,087. This is arithmetic from the record
sizes in the GGUF header and the planner's budget. The logs give the budget to 0.01 GiB, and rerunning the planner
on the private tally narrows it to the 2.9 MB (stage 0) and 2.5 MB (stage 1) that give the logged tiers. Whether
one more expert of a cheaper layer would still fit after the last promotion is INCONCLUSIVE at that precision.
Ranking by use per byte and stopping at the first promotion that does not fit is the standard greedy for this
packing problem, and its total use is within one expert's use of the fractional optimum, which bounds every packing
with or without the rule of one hot and one compressed expert per layer. So at these budgets no other choice of hot
experts, the cheapest-first count included, carries more of the mixed tally's routed use at NVFP4 (an in-sample
measure, like the route shares above) than the planner's choice plus one expert's share, on each of the two stages.

| Measure | Result |
|---|---|
| Short prompts, general set | 83.6 tok/s (36.25 ms per round) |
| Short prompts, legal-style set | 64.7 tok/s (35.78 ms per round) |
| Short prompts, both sets, on the bench boot | general 87.0 tok/s (34.73 ms per round), legal-style 67.3 (34.41 ms) ([two boots](#two-boots-two-pairs-of-short-set-speeds)) |
| Tool loop at 31K tokens, source code | 90.9 tok/s (40.09 ms per round) |
| Tool loop at 31K tokens, US Code Title 28 | 87.8 tok/s (38.18 ms per round) |
| Tool loop at 216K tokens, run 1: first read | 216,255 tokens in 168.72 s (about 1,280 tok/s) |
| Tool loop at 216K tokens, run 1: seven follow-ups | 7 of 7 kept the whole previous prompt, first token median 9.36 s, decode mean 90.4 tok/s, final answer ended normally (1,385 tokens) |
| Tool loop at 216K tokens, run 2, on the bench boot | first read 166.22 s, 7 of 7 follow-ups kept the previous prompt, first token median 9.47 s, decode mean 99.8 tok/s over all seven, 93.9 over the six that ended with a tool call. The final request ran to the 32,000-token limit inside thinking with no final answer: 1 of 2 runs at the limit ([The 216K tool loop](#the-216k-tool-loop)) |
| Decode routes on compressed records (route counters) | general 21.0 %, legal-style 16.7 %, source loop 15.0 %, US Code loop 15.7 % |
| Closeness to the all-official-precision reference, replaced 12-prompt set, thinking off | PASS by the rule fixed before the run (8 of 12 on each part): first different token 11 for, 1 against, and mean difference 8 for, 4 against, 0 undecided |
| Long thinking chats 0, 1 and 2 (198,307 to 216,245 tokens), thinking on | 0 of 3 at the 32,000-token limit: final answers of 3,057, 1,262 and 1,592 tokens. 95 % Wilson interval for the share at the limit: 0.000 to 0.561 |

Short-set tok/s is output tokens over the whole request time, the wait for the first token included. Tool-loop tok/s
is decode only, the mean over the follow-ups. Every row is at the 1,425 MHz ceiling. The headline pair is run 1's
boot ([two boots](#two-boots-two-pairs-of-short-set-speeds)). The route shares are counted by the engine's route counters
(`nvfp4_8r_stats.py`, off by default) on a separate boot of the same profile.

**In-sample.** The mixed tally weights four kinds of text equally: one private, and three made from this
repository's public prompt sets (the general short set, the source-code tool loops at 31K and 216K tokens, and the
legal-style short set with the US Code tool loop). So the speeds and route shares above are in-sample: they show
how the hot set serves the text it was chosen from. Chats 1 and 2 of the long thinking test take later sections of
the public source text and are held out. The tally is not published. The package's `tier_map.json` carries only
whether each expert is hot or compressed.

**Closeness.** The reference is the same NVFP4 pack with every routed expert at official precision (the profile
published as [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100)), and the yardstick is Volta sPLIt. Per prompt, this build's first token that differs from the reference must
come no earlier than Volta sPLIt's, and its mean absolute log-probability difference must be no larger. The
replaced 12-prompt set is the 12-prompt set of the earlier checks ([history](docs/history.md#quality)) with its
five undecidable prompts replaced, as planned there, by a selection rule fixed in advance that reads no result of this build. The rule of 8 of 12 was
not changed.

**Long thinking chats.** These are chats 0, 1 and 2 of the termination test
([history](docs/history.md#long-thinking-answers-the-termination-test)). The rule fixed before the run: no more chats at the limit
than the host-tier profile on the same tally, which ran chat 0 to the limit (1 of 3, not accepted, quality not
established). The test's six further chats and its runs at sampling settings were not made, so three chats bound
the share at the limit only below 0.56.

**Three prompt slots** (`hot_iq` on the mixed tally, MEASURED 2026-11-10, not accepted, quality not established):
general 81.9 tok/s (35.23 ms per round) and legal-style 68.7 tok/s (34.42 ms) with fewer hot experts (3,346 and
1,886). In three 30K chats taking turns, 6 of 6 follow-ups kept their own chat's prompt (first token median 8.08 s).

**How it compares** (same host, [history](docs/history.md#tokens-per-second)): Volta sPLIt decoded 118 to 119 tok/s
on the legal-style set and 137 to 138 on the general set on 2026-11-09. The three-tier mode on the first tally
decoded 57 to 58 and 51 to 52 the same day, and the three-tier accepted configuration 58.4 and 51.5 in the
comparison run of 2026-11-10. Prompt by prompt against Volta sPLIt's published bench: [Beside Volta sPLIt](#beside-volta-split). [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100) states its own speeds.

**Limits.**

- The hot set is fixed. No expert is promoted during a chat, and text unlike the tally's four kinds may route more
  often to compressed records.
- The 216K tool loop's final request ran to the 32,000-token limit in 1 of 2 runs on this build
  ([The 216K tool loop](#the-216k-tool-loop)).
- Compressed records are not official precision: on synthetic inputs an expert's IQ2-class record is far from its
  official record (relative L2 0.60 to 0.69, [What did not work](#what-did-not-work-or-is-not-yet-good)).
- Prompts are read with the full-precision records of all experts. The compressed experts' records are staged from
  pinned host RAM (180.8 GB). The faster reading variant of plan 04 part B is not used: in its tests of 2026-11-10
  its first read of one long chat stopped the host or a card three times out of three. Replays on one card and on
  two cards completed without a fault. The cause is not yet known.
- The engine reads the package's `tier_map.json` in place of a tally (the tier-map change of this release). Its
  CPU check passed: the map gives the measured tiers and digests on all 40 layers, and in layers 0 and 20 (first
  tensor-parallel rank) every expert record of the package is byte-equal to the record that was measured. It has
  not yet booted on eight cards (2026-11-10): that boot check follows and will be added as a dated row.

## Tokens per second

The speed bench of the build of this release. MEASURED 2026-11-10: `hot_iq` on the mixed tally, one prompt slot,
reserves 4.5 / 7.5 GiB, one request at a time, card clock ceiling 1,425 MHz, 169 requests, all completed. The
prompts and the method are those of [Volta sPLIt's speed bench](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#long-prompts-1k-to-256k), and the run checks
each document's token count against Volta sPLIt's. Each prompt is a real document of its kind cut to exactly that
many tokens: code = vLLM's Python source, structured = LegalBench items as JSON lines, legal = a public CC BY legal
corpus, then LegalBench prompt texts, prose = War and Peace, chat = a chat-style instruction over Les Misérables.
Long prompts: greedy, 1,024 tokens out, a random cache salt so no prompt is served from the prefix cache. Decode =
(tokens - 1) / (last token - first token), timed by the benchmark program. Prompt reading = prompt tokens / first-token time. The
mixed tally includes general, source-code and legal-style text ([In-sample](#hot_iq-on-the-mixed-tally-the-build-of-this-release)), so these kinds are not held out.
To rerun it against one idle server, `harness/run/benchvolta.sh URL` (with `LANE` set to the build's root) writes the
receipts and `harness/run/bench_volta_summary.py RECEIPT_DIR` prints these tables, given the kit's public documents in
`--kitdata`, checked against their SHA-256 sums, which this repository does not carry.

![Decode speed by prompt length, Volta Max uPLIft against Volta sPLIt's published bench, for code, structured, legal and prose](docs/charts/decode-by-length-light.png#gh-light-mode-only)
![Decode speed by prompt length, Volta Max uPLIft against Volta sPLIt's published bench, for code, structured, legal and prose](docs/charts/decode-by-length-dark.png#gh-dark-mode-only)

**What holds up.** Decode falls slowly with prompt length: code 123.3 tok/s after a 1K prompt and 103.8 after
256K, legal 71.9 and 61.8, prose and chat 59.6 to 67.9 at every length (thinking off, mean of 2 runs). The greedy
output of the two runs was byte-identical in 45 of 45 cells, and the largest difference between the two runs of a
cell was 6.3 tok/s.

**What does not.** Prompt reading is slow. It rises with length and tops out at 1,243 to 1,276 tok/s at 256K. The
first token comes 5.60 s after a 1K prompt, 106.4 s after 128K and 209.6 s after 256K. Every prompt is read with
the official records of all experts, the compressed experts' records staged from pinned host RAM
([Limits](#hot_iq-on-the-mixed-tally-the-build-of-this-release)). This build is slower than Volta sPLIt in every thinking-off cell, every prompt-reading cell,
every first-token time and every short-prompt row of the bench ([below](#beside-volta-split)). With thinking on (one
run per cell in both benches), code at 4K, 8K, 16K and 256K decoded 1.9 to 8.2 tok/s faster than Volta sPLIt's
cells, and the other 41 thinking-on cells were slower.

### Long prompts: 1K to 256K

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

Thinking off, 90 requests, each cell the mean of 2 runs. First token = median of the five kinds. Prompt reading
= range over the five kinds. Thinking on (45 requests, one per cell) and prompt reading by kind are in
[docs/measurements.md](docs/measurements.md#speed-bench-1k-to-256k).

### Beside Volta sPLIt

The same prompts and method on the same host, not the same clock or day: Volta sPLIt's figures are its published
bench of 2026-11-05 at a 1,522 MHz ceiling, this build's are of 2026-11-10 at 1,425 MHz. Each cell is this build /
Volta sPLIt, thinking off, mean of 2 runs.

| prompt | first token | prompt reading | code | legal |
|---|---:|---:|---:|---:|
| 1K | 5.60 s / 1.22 s | 175-247 / 820-948 | 123.3 / 215.0 | 71.9 / 117.6 |
| 2K | 7.02 s / 1.95 s | 274-324 / 1,024-1,082 | 118.9 / 203.6 | 65.2 / 114.8 |
| 4K | 9.12 s / 3.26 s | 437-454 / 1,245-1,264 | 120.9 / 194.9 | 62.5 / 111.6 |
| 8K | 13.3 s / 6.94 s | 595-622 / 1,178-1,188 | 127.0 / 207.9 | 69.7 / 109.1 |
| 16K | 19.3 s / 10.4 s | 812-868 / 1,574-1,583 | 126.6 / 201.8 | 69.7 / 105.6 |
| 32K | 31.4 s / 17.2 s | 1,030-1,055 / 1,894-1,913 | 119.3 / 200.5 | 66.7 / 116.5 |
| 64K | 56.1 s / 30.9 s | 1,161-1,188 / 2,111-2,132 | 116.9 / 194.1 | 68.0 / 108.3 |
| 128K | 106.4 s / 59.2 s | 1,221-1,263 / 2,198-2,220 | 115.4 / 182.8 | 67.9 / 110.9 |
| 256K | 209.6 s / 118.7 s | 1,243-1,276 / 2,197-2,222 | 103.8 / 160.0 | 61.8 / 103.1 |

### Short prompts, by text type

| kind of output | greedy (T=0) | sampled (T=1) | tokens per round, T=0 | Volta sPLIt greedy | Volta sPLIt sampled |
|---|---:|---:|---:|---:|---:|
| code | **127.2** (125.0-127.4) | 127.4 (113.8-129.5) | 4.39 | 213.1 | 196.8 |
| structured (JSON) | **129.5** (125.6-129.7) | 127.5 (126.1-129.5) | 4.45 | 209.4 | 205.4 |
| legal | **74.9** (74.9-75.0) | 68.0 (67.7-69.7) | 2.59 | 115.7 | 108.4 |
| prose | **65.2** (64.6-65.2) | 60.7 (58.2-60.8) | 2.26 | 98.0 | 87.2 |
| chat | **59.9** (59.9-60.1) | 58.4 (55.8-61.0) | 2.06 | 96.2 | 93.8 |
| thinking on, 4 prompts, token-weighted | - | **92.4** (84.7-130.8 per prompt) | - | - | 132.3 |

The bench's short prompts, up to 1,024 tokens with a natural stop, thinking off, median of 3 runs with range. Greedy repeats were
byte-identical for every kind. Volta sPLIt: its published bench of 2026-11-05 at 1,522 MHz, same prompts. These are
not the short sets of the [build section](#hot_iq-on-the-mixed-tally-the-build-of-this-release), which count the wait for the first token.

### The 216K tool loop

**The final request of the 216K tool loop ran to the limit on the bench boot.** In this tool loop (216,255-token prompt, thinking on) the first read took 166.22 s and all seven follow-ups kept the previous prompt (first token median 9.47 s, decode mean 99.8 tok/s over the seven and 93.9 tok/s over the six that ended with a tool call (83.4 to 99.6)). The final request then ran to the 32,000-token limit inside thinking and gave no final answer (stop reason `max_tokens`, 5.44 tokens per round). On the earlier boot of the same profile the same loop's final answer ended normally (1,385 tokens). So on this build that loop's final request has run to the limit in 1 of 2 runs. Chats 0, 1 and 2 of the long thinking test on the mixed tally ([above](#hot_iq-on-the-mixed-tally-the-build-of-this-release)) are separate requests of 216,245, 198,307 and 204,871 tokens and are not counted here.

### Two boots, two pairs of short-set speeds

The short sets (three prompts each, output tokens over the whole request time) ran on two boots of this profile,
both at 1,425 MHz: general 83.6 tok/s and legal-style 64.7 on the boot of the tool loops and long thinking chats,
general 87.0 (34.73 ms per round) and legal-style 67.3 (34.41 ms) at the end of the speed bench. The headline and
the badges carry the first pair, the lower one.

## What this is

**The question.** DeepSeek-V4.1-Flash-1017 has 384 routed experts in each of 40 layers. At the official MXFP4
precision they take about 269 GiB (ESTIMATED: 384 x 35.4M weights x 40 x 4.25 / 8 bytes). Eight 32 GB V100s have
256 GiB of HBM in total, before attention weights, KV cache and the drafter. Volta sPLIt fits the model by serving
compressed IQ2-class experts. This build asks how much of the routed work can run at the official precision with every routed expert still on the cards.

**The pack.** NVIDIA's NVFP4 pack stores the routed experts in NVFP4 (FP8 scale per 16 values). Its card says the
conversion from DeepSeek's MXFP4 was lossless. We fold it back to MXFP4 (one power-of-two scale per 32 values) and
check the folded values against the NVFP4 formula: they are bit-equal. The check, its counts and the pack recipe are
stated in [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#what-this-is), which serves every expert from these packs. The
pack builder and its checks are also in this repository ([harness/benchmarks/nvfp4_8r/](harness/benchmarks/nvfp4_8r/)).

**Three tiers (an earlier step).** With the less-used experts in host RAM (as Volta Full NVFP4 serves them), reading an expert from host RAM costs about 1.0 to 1.4 ms per expert per layer (MEASURED, one card).
So a middle "warm" tier runs the less-used experts from the IQ2-class records Volta sPLIt already uses, which are
about half the size and fit on the cards. Hot experts run at official precision on the cards; warm experts
run from IQ2-class records on the cards in decode (their official records stay in host RAM for prompt reading);
the coldest experts run at official precision from host RAM. Boot of 2026-11-09 (MEASURED): stage 0 has 4,107 hot,
3,213 warm and 360 host experts; stage 1 2,695, 4,379 and 606. The warm tier carries 3.74 % (stage 0) and 9.29 %
(stage 1) of decode routes by the private tally (aggregates only), and the host tier 0 % by that tally. So this mode
is not "full precision": 3.7 to 9.3 % of routed-expert calls run from compressed records. Its measurements are
in the [history](docs/history.md).

**No host tier: Max uPLIft (`hot_iq`), the build of this release.** The three-tier step and Volta Full NVFP4 keep
experts in host RAM.
This build keeps every routed expert on the cards in decode: every compressed record goes on the cards first, and
experts are promoted to official precision in order of use per byte until the next one does not fit
([above](#hot_iq-on-the-mixed-tally-the-build-of-this-release)).

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
| Host RAM | 363 GiB available with no server running (MEASURED). `hot_iq` keeps the compressed experts' official records, 180.8 GB, in pinned host RAM for prompt reading, plus the Engram tables |
| Routed experts | [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) @ `480550d3`, folded to MXFP4 (E8M0 scale per 32) packs per layer and TP rank for the hot experts. Compressed experts: the GSQ-RCO records below |
| Other weights | as Volta sPLIt: [ISTA-DASLab GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) for attention, dense and shared-expert tensors (Q5_K 4.76 GiB, Q8_0 0.36 GiB, BF16 0.26 GiB, F32 0.15 GiB by tensor bytes, MEASURED from the file header) and the IQ4_NL Engram tables; its routed-expert records are IQ1_M 55.37 GiB, IQ2_XXS 45.68 GiB and IQ2_XS 21.95 GiB; config, tokenizer and DSpark drafter (FP8) from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| Expert kernel | v100-skinny's QPN kernel as packaged in 1Cat-vLLM ([#742](https://github.com/1CatAI/1Cat-vLLM/pull/742)): `skinny_moe_qpn_sm70`, E8M0 scale mode, W4A16, FP16 `mma.sync`; V4.1's limited SwiGLU and the weighted sum in FP32; CUDA graphs for 1 to 6 rows |
| Prompt reading | the compressed experts' official records in pinned host RAM, staged into two 32-expert HBM buffers on a copy stream, one staging pair per card |
| Compressed experts | Volta sPLIt's IQ expert op over the IQ2-class records kept on the cards |
| Engine | the Volta sPLIt engine (1Cat-vLLM #1277 + its patch series and production patches) plus this build: three new files (the native expert path with the tier-map reader, the tier planner and the route counters) and three changed files ([engine/](engine/)) |
| Layout | PP2 x TP4: layers 0-19 and the Engram lookup on one board, layers 20-39, the head and the drafter on the other |
| Card reserve | per stage, from measured peaks plus about 0.5 GiB: 4.5 GiB per card on stage 0, 7.5 GiB on stage 1 |
| Context | max-model-len 524,288, one request at a time (`max-num-seqs 1`) |
| Serving profiles | the build of this release: the [package profile template](engine/configs/hot_iq.yaml.template) (`hot_iq` on the mixed tally, one slot, reserves 4.5 / 7.5 GiB, tiers from the package's `tier_map.json`), started by [serve_hot_iq.sh](engine/configs/serve_hot_iq.sh). Earlier steps, whose profiles read a tally: [three-tier](engine/configs/serve-nvfp4-pp2-t3.yaml) (`hot_iq_host`), [`hot_iq`](engine/configs/serve-nvfp4-pp2-t3n.yaml) (no host tier, first tally; speeds measured, quality not established, not accepted), [three prompt slots](engine/configs/serve-nvfp4-pp2-t3-a3.yaml) (`max-num-seqs 3`, reserves 5.3 / 8.1 GiB), [its one-slot control](engine/configs/serve-nvfp4-pp2-t3-a1r.yaml) (same reserves, one slot), and [no host tier with three prompt slots](engine/configs/serve-nvfp4-pp2-t3n-a3.yaml) (`hot_iq`, `max-num-seqs 3`, reserves 5.3 / 8.1 GiB; measured, not accepted). The first-tally `hot_iq` profiles are not accepted (see above). The two host-tier slot profiles are measured (rows of 2026-11-10 in the [measurement log](#measurement-log)): three slots on the accepted tiers are not accepted (quality not established), and the one-slot control ran the traced tool loop. The all-NVFP4 profile [serve-nvfp4-pp2.yaml](engine/configs/serve-nvfp4-pp2.yaml) is Volta Full NVFP4's ([Related builds](#related-builds)). It stays here because the closeness test boots it as the reference |

## What worked

MEASURED on the dates given.

| Item | Result |
|---|---|
| `hot_iq` on the mixed tally (2026-11-10) | the build of this release: closeness PASS on the replaced 12-prompt set, 0 of 3 long thinking chats at the limit, 64.7 tok/s legal-style and 83.6 general on the short sets (in-sample, [above](#hot_iq-on-the-mixed-tally-the-build-of-this-release)) |
| The native expert path | the exact NVFP4 to MXFP4 fold, the scale range on Volta, the QPN expert layer on one card and the bit-equal hot and host split (MEASURED 2026-11-09) are shared with Volta Full NVFP4 and stated there ([What worked](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#what-worked)) |
| Dropping the host tier (2026-11-09) | the host-tier profile's slow general text was host-RAM reads: without a host tier (`hot_iq`) general text decodes at 33.4 ms per round against 58.0 to 59.4 ms with it. Speed only: that profile is not accepted (its quality is not established) |
| Earlier steps | the three-tier mode, the first-tally profiles and three prompt slots: [history](docs/history.md#what-worked) |

## What did not work or is not yet good

MEASURED on the dates given.

| Item | Result |
|---|---|
| Slower than Volta sPLIt | on the mixed tally 64.7 tok/s legal-style and 83.6 general (2026-11-10, 1,425 MHz) against Volta sPLIt's 118 to 119 and 137 to 138 (2026-11-09, 1,522 MHz) ([How it compares](#hot_iq-on-the-mixed-tally-the-build-of-this-release)). In the speed bench, slower in every thinking-off cell, every prompt-reading cell and every short-prompt row against Volta sPLIt's bench of 2026-11-05 at 1,522 MHz ([Beside Volta sPLIt](#beside-volta-split)) |
| Slow prompt reading | 1,243 to 1,276 tok/s at 256K, first token 5.60 s after a 1K prompt, 106.4 s after 128K and 209.6 s after 256K (speed bench, 2026-11-10, 1,425 MHz, [Tokens per second](#tokens-per-second)) |
| The 216K tool loop's final request | ran to the 32,000-token limit inside thinking on the bench boot, ended normally on the earlier boot: 1 of 2 runs at the limit (2026-11-10, [The 216K tool loop](#the-216k-tool-loop)) |
| Prompt reading | prompts are read with the official records of all experts, the compressed experts' records staged from pinned host RAM. The faster reading variant of plan 04 part B is not used: its first read of one long chat stopped the host or a card three times out of three on eight cards, and the cause is not yet known ([Limits](#hot_iq-on-the-mixed-tally-the-build-of-this-release)) |
| Compressed records are not official precision | on synthetic inputs an expert's IQ2-class record is far from its official record (relative L2 0.60 to 0.69 in layers 5, 12 and 27, MEASURED 2026-11-09) |
| The hot set is fixed | it is chosen once, from the mixed tally. Text unlike the tally's four kinds may route more often to compressed records |
| Earlier steps | what did not work in the three-tier mode, the first-tally profiles and the prompt-reading variant: [history](docs/history.md#what-did-not-work-or-is-not-yet-good) |

## Next

The ranked campaign and the research notes behind it follow in the next update. Each piece is built and measured
alone, biggest expected improvement first. Done so far: three prompt slots work (no host tier, "What worked"; on the
accepted tiers they are measured but not accepted, quality not established), our router now sends earlier thinking
back unchanged, the termination test for long thinking answers exists and has run, the load failure of the
prompt-reading variant of plan 04 is fixed, and `hot_iq` on the mixed tally passed its closeness check on the
replaced 12-prompt set and its long thinking chats and is the build of this release ([above](#hot_iq-on-the-mixed-tally-the-build-of-this-release)). The full speed bench
of that build has run ([Tokens per second](#tokens-per-second)). Next, first: that reading variant at the accepted tiers on eight cards,
with the fix for its memory shortfall (built, checked on one card, in this snapshot's engine file), the cause of its
faults on one long chat, and fused kernels for route grouping and SwiGLU. The detailed plans are in [docs/plans/](docs/plans/), and the measured research on
where the time goes is [docs/research/LANE-RESEARCH.md](docs/research/LANE-RESEARCH.md). Our target is 80 to 100
tok/s at official expert precision on these eight cards.

## Quality

**Max uPLIft (`hot_iq` on the mixed tally): PASS on the replaced 12-prompt set.** MEASURED 2026-11-10, greedy,
thinking off, 1,024 tokens each. The reference is the same NVFP4 pack with every routed expert at official
precision, booted from the profile published as [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100). The yardstick is Volta sPLIt.
The rule, fixed before the run: per prompt, this build's first token that differs from the reference must come no
earlier than Volta sPLIt's, and the mean absolute log-probability difference of the chosen token, averaged over the
tokens before the first difference, must be no larger. Each part must hold on at least 8 of the 12 prompts. The set
is the twelve prompts written for the earlier checks (code 3, JSON 2, general prose 3, legal-style prose 2,
reasoning 2), with the five on which Volta sPLIt shares no prefix with the reference replaced from a pool written for
this check, by a selection rule fixed in advance that reads no result of this build. First different token no
earlier: 11 for, 1 against. Smaller mean log-probability difference: 8 for, 4 against, 0 undecided. Both parts
reach 8. The per-prompt figures are in [docs/measurements.md](docs/measurements.md). The program of the wider set is
not in this snapshot. Its receipts hold token ids and log-probabilities only.

**Earlier checks.** The three-tier mode passed a three-prompt check against the same reference on 2026-11-09, and
the first scoring of the 12-prompt set on 2026-11-10 was INCONCLUSIVE for every configuration then measured
([history](docs/history.md#quality)).

The hot experts are value-exact to DeepSeek's official MXFP4 experts ([the fold](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#what-this-is)). NVIDIA's own
accuracy table for the pack, measured with vLLM on 4x GB300 (UPSTREAM), is within run noise of the MXFP4 source, for
example GPQA-D 92.93 against 92.77 and AA-LCR 80.19 against 80.41. Those figures are NVIDIA's, on NVIDIA's serving
stack, not this build.

## Measurement log

One row per measurement, newest first. Later results are added at the top with their date; the tables above are
updated when a row changes them. Details: [docs/measurements.md](docs/measurements.md). Rows of the all-NVFP4
profile (two-tier) are Volta Full NVFP4's, in its [measurement log](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100#measurement-log). In rows before
the mixed tally, "the accepted configuration" and "the accepted tiers" mean the three-tier profile.

| Date | Build or profile | Test | Result | Basis |
|---|---|---|---|---|
| 2026-11-10 | `hot_iq` on the mixed tally, one slot (the build of this release) | speed bench, long prompts 1K to 256K, thinking off | code 123.3 tok/s at 1K to 103.8 at 256K, legal 71.9 to 61.8, prose and chat 59.6 to 67.9. First token 5.60 s at 1K, 106.4 s at 128K, 209.6 s at 256K. Greedy output byte-identical in 45 of 45 cells ([Tokens per second](#tokens-per-second)) | MEASURED, public text, 2 runs per cell, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, long prompts 1K to 256K, thinking on | code 102.3 to 136.8 tok/s, structured 107.1 to 122.5, legal 67.8 to 81.8, prose 60.3 to 67.9, chat 54.6 to 66.6. First token 5.76 s at 1K to 210.2 s at 256K ([table](docs/measurements.md#speed-bench-1k-to-256k)) | MEASURED, public text, 1 run per cell, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, prompt reading by kind | 175 to 247 tok/s at 1K, 1,030 to 1,055 at 32K, 1,221 to 1,263 at 128K, 1,243 to 1,276 at 256K ([table](docs/measurements.md#speed-bench-1k-to-256k)) | MEASURED, public text, 2 runs per cell, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, short prompts by kind | greedy: code 127.2 tok/s, structured 129.5, legal 74.9, prose 65.2, chat 59.9. Sampled 58.4 to 127.5. Greedy repeats byte-identical ([table](#short-prompts-by-text-type)) | MEASURED, public text, 3 runs per cell, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | speed bench, four prompts, thinking on, sampled | 92.4 tok/s token-weighted (84.7 to 130.8 per prompt) | MEASURED, public text, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot, bench boot | tool loop, 216,255-token prompt, run 2 | first read 166.22 s, 7 of 7 follow-ups kept the prompt, first token median 9.47 s, decode mean 99.8 tok/s over all seven, 93.9 over the six that ended with a tool call. The final request ran to the 32,000-token limit inside thinking with no final answer (5.44 tokens per round). With run 1 (row below): 1 of 2 runs at the limit ([The 216K tool loop](#the-216k-tool-loop)) | MEASURED, public text, temperature 0, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot, bench boot | short sets | general 87.0 tok/s (34.73 ms per round), legal-style 67.3 (34.41). The headline carries the other boot's pair (row below) | MEASURED, public text, temperature 0, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot (the build of this release) | short sets, 31K tool loops | general 83.6 tok/s (36.25 ms per round), legal-style 64.7 (35.78), source loop 90.9 (40.09), US Code loop 87.8 (38.18). In-sample ([above](#hot_iq-on-the-mixed-tally-the-build-of-this-release)) | MEASURED, public text, temperature 0, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | tool loop, 216,255-token prompt, run 1 | first read 168.72 s, 7 of 7 follow-ups kept the prompt, first token median 9.36 s, decode mean 90.4 tok/s, final answer ended normally (1,385 tokens). Run 2 ran to the limit (row above) | MEASURED, public text, temperature 0, 1,425 MHz |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | closeness, replaced 12-prompt set | PASS: first different token 11 for, 1 against, and mean difference 8 for, 4 against, 0 undecided ([Quality](#quality)) | MEASURED, greedy, thinking off, rule (8 of 12) fixed before the run |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | long thinking chats 0, 1 and 2 of the termination test | 0 of 3 at the 32,000-token limit (final answers 3,057, 1,262 and 1,592 tokens), 95 % Wilson interval 0.000 to 0.561. The six further chats and the sampling settings were not run | MEASURED, public text, temperature 0 |
| 2026-11-10 | `hot_iq` on the mixed tally, one slot | decode routes on compressed records | general 21.0 %, legal-style 16.7 %, source loop 15.0 %, US Code loop 15.7 % | MEASURED, route counters on a separate boot |
| 2026-11-10 | `hot_iq` on the mixed tally, three slots: NOT accepted (quality not established) | short sets, three 30K chats taking turns | general 81.9 tok/s (35.23 ms per round), legal-style 68.7 (34.42). 6 of 6 follow-ups kept their own chat's prompt, first token median 8.08 s | MEASURED, public text |
| 2026-11-10 | host tier on the mixed tally, one slot: NOT accepted (quality not established) | long thinking chats 0, 1 and 2 | chat 0 ran to the 32,000-token limit, chats 1 and 2 ended normally (1 of 3) | MEASURED, public text, temperature 0 |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B) at the accepted tiers: NOT accepted (quality not established) | first read of chat 0 (216,245 tokens) | booted with the accepted configuration's hot, warm and host experts, then ran out of card memory on stage 0 in the second chunk of the read (a request for 108 MiB with 64.75 MiB free). The server ended by itself. INFERENCE from the code, nothing run: both reading modes peak at the same point and the variant is short by the workspace it keeps from load. The fix is built and checked on one card of the test host (MEASURED 2026-11-10: results bit-equal to the previous file at four chunk sizes, peak of the reading call 1.30 GB to 0.67 GB at 8,188 rows). It is in this snapshot's engine file (the `hot_iq` measurements on the mixed tally ran with that file), and the variant has not yet run at the accepted tiers with it | MEASURED, boot log and one-card receipt |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 10 for, 2 against, and mean difference 5 for, 1 against, 6 undecided ([history](docs/history.md#quality)) | MEASURED, greedy, rule (8 of 12) fixed before the run |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | termination test, chats 1 and 2 (198,307 and 204,871 tokens), second boot | both final answers ended normally (749 and 1,491 tokens). First reads 102.12 s and 103.0 s (accepted configuration: 155.56 s and 161.39 s). 7 of 7 follow-ups kept the prompt in each chat, first token median 4.42 s and 4.26 s. With chat 0 the variant stands at 1 of 3 at the limit (95 % Wilson interval 0.061 to 0.792) | MEASURED, public text, temperature 0 |
| 2026-11-10 | accepted tiers under the load setting (a profile that pins them, not in this snapshot) | termination test, chat 0 (216,245 tokens) | the same hot, warm and host experts as the accepted configuration. First read 167.64 s. The final answer ended normally with 6,417 tokens, and all eight requests gave the token counts of the two earlier runs. On this chat the setting did not change the token counts | MEASURED, public text, temperature 0 |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B), one slot, reserves 5.3 / 8.1 GiB: NOT accepted (quality not established) | termination test, chat 0 (216,245 tokens) | first read 107.82 s (accepted configuration: 164.9 s). 7 of 7 follow-ups kept the prompt, first token 4.02 to 5.38 s, median 4.44 s (accepted configuration: 8.76 to 9.43 s). Tool-call requests decoded at 49.4 to 62.6 tok/s. The final answer ran to the 32,000-token limit in thinking, a repetition loop (5.33 tokens per round) | MEASURED, public text, temperature 0 |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B): NOT accepted (quality not established) | load failure, and an eight-card boot with the allocator setting off | explained and fixed: under the engine's allocator setting during weight loading the cards fill (one card: 0.038 GiB free after the build against 7.696 GiB with the setting off), and a probe on one card gives the failed boot's driver error "invalid argument" under that setting. With the setting off the variant was ready in 349 s with no error. That the first boot failed in exactly this way is an inference. Not in this snapshot | MEASURED, one card and eight cards |
| 2026-11-10 | three slots on the accepted tiers: NOT accepted (quality not established) | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 10 for, 2 against, and mean difference 7 for, 0 against, 5 undecided ([history](docs/history.md#quality)) | MEASURED, greedy, rule (8 of 12) fixed before the run |
| 2026-11-10 | three-tier, the accepted configuration before the mixed tally | closeness, wider set | INCONCLUSIVE on the 12-prompt set: first different token 11 for, 1 against, and mean difference 7 for, 0 against, 5 undecided, because Volta sPLIt differs from the reference at the first token on five prompts ([history](docs/history.md#quality)) | MEASURED, greedy, rule (8 of 12) fixed before the run |
| 2026-11-10 | three-tier, the accepted configuration before the mixed tally | termination test, three chats of 198,307 to 216,245 tokens | 3 of 3 final answers ended normally (6,417, 1,499 and 1,642 tokens). Chat 0 gave the token counts of the 2026-11-09 run again. Three chats bound the share that runs to the limit only below 0.56 (95 % Wilson interval 0.000 to 0.561) | MEASURED, public text, temperature 0 |
| 2026-11-10 | this build, all configurations on the first tally | chat 0 of the termination test (the 216K tool loop) | ended normally in the accepted configuration (three runs). Ran to the 32,000-token limit on chat 0 in the five boots with a lower share of routes on official-precision experts (none accepted, quality not established). One of the five, the part B variant, ended chats 1 and 2 normally (rows above), so this holds for chat 0 only. Cause not established ([table](docs/history.md#long-thinking-answers-the-termination-test)) | MEASURED, temperature 0 |
| 2026-11-10 | Volta sPLIt | termination test, the same three chats, and a 31,173-token chat | 3 of 3 final answers ended normally (10,013, 732 and 1,812 tokens), with the same bound, below 0.56. On the separate 31,173-token chat the final answer ran to the 32,000-token limit in a repetition loop | MEASURED, public text, temperature 0 |
| 2026-11-10 | IQ records in prompt reading (plan 04 part B) | first eight-card boot | failed while loading: driver error "invalid argument" during the build of the IQ bank. Explained and fixed later the same day (rows above). Not in this snapshot | MEASURED, boot log |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | closeness, three prompts | INCONCLUSIVE by the rule | MEASURED, greedy, three public prompts |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | tool loop, 216,255-token prompt | first read 173.5 s; 7 of 7 follow-ups kept the prompt, first token median 9.62 s; the final request ran to the 32,000-token limit (5.16 tokens per round) | MEASURED, public text, temperature 0 |
| 2026-11-10 | three-tier, three slots: NOT accepted (quality not established) | short sets, three chats, 257,741-token prompt | general 50.9 / 51.3 tok/s, legal-style 63.9 / 64.2 tok/s; 6 of 6 follow-ups kept their own chat's prompt; 257,741 tokens read in 207.1 s; highest card memory 31,704 MiB | MEASURED, public text |
| 2026-11-10 | three-tier, one slot, reserves 5.3 / 8.1 GiB, traced | tool loop, 216,245-token prompt | first read 171.9 s; 7 of 7 kept; the final request ran to the 32,000-token limit (5.21 tokens per round); host staging 86 % of prompt-reading expert time (trace; traced speeds are slowed) | MEASURED, public text, temperature 0 |
| 2026-11-09 | Volta sPLIt | short tool loop, ten runs | seven runs of the identical chat: 3, 2, 2, 2, 1, 0 and 0 of 3 follow-ups kept their prompt (run 2 through our router; a 3-second pause in runs 5 and 7 changed nothing); three runs with a unique first line: 3 of 3 each. HYPOTHESIS: the engine matches an older copy of the same chat, so every repeat of a loop test must be a unique chat. Final request: the 32,000-token limit once (the router run, 5.6 tokens per round); a 3,000-token cap reached in 5 of 7 later runs (long thinking on this prompt is common at temperature 0; the cap does not show a runaway) | MEASURED |
| 2026-11-09 | Volta sPLIt | turn boundary, 31,164-token chat | earlier thinking kept: 38,800 of 38,845 kept, first token 0.54 s; stripped: 0 of 38,193, 14.07 s; through our router after its change: 69,728 of 69,772, 1.45 s | MEASURED |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | tool loop repeat, 216K prompt | six tool-call requests 12.6 to 13.5 s each, 92.2 to 99.6 tok/s; the final request again ran to the 32,000-token limit in thinking (5.87 tokens per round) | MEASURED |
| 2026-11-09 | no host tier, three slots: NOT accepted (quality not established) | closeness, three prompts | INCONCLUSIVE by the rule ([history](docs/history.md#quality)) | MEASURED |
| 2026-11-09 | no host tier, three slots | 257,706-token prompt | not accepted (quality not established); read in 214.4 s (1,202 tok/s); highest card memory 31,400 / 30,118 MiB | MEASURED |
| 2026-11-09 | no host tier, three slots | short sets | not accepted (quality not established): general 86.8 / 87.3 tok/s, legal-style 67.8 / 68.1 tok/s, no slowdown against one slot | MEASURED |
| 2026-11-09 | three prompt slots (no host tier) | three chats taking turns | WORK: 6 of 6 follow-ups kept their own chat's whole prompt (Volta sPLIt with one slot: 0 of 6); the no-host-tier tiers are not accepted (quality not established) | MEASURED |
| 2026-11-09 | no host tier (`hot_iq`): NOT accepted (quality not established) | short sets, tool loop | general 86.3 / 86.2 tok/s, legal-style 66.9 / 71.9 tok/s; tool-call requests 12.3 to 13.5 s each. Caution: the loop's final request did not finish (32,000 tokens of thinking, token limit); closeness check not run ([details](docs/history.md#tokens-per-second)) | MEASURED |
| 2026-11-09 | three-tier against the all-NVFP4 reference (two-tier) | closeness, three prompts | PASS ([history](docs/history.md#quality)) | MEASURED |
| 2026-11-09 | three-tier | turn boundary, 31,164-token chat | earlier thinking kept: first token 1.11 s; stripped: 33.08 s | MEASURED |
| 2026-11-09 | Volta sPLIt, three-tier | tool loop, 216K-token prompt, thinking on | [tool-loop table](docs/history.md#tokens-per-second): 12.5 to 15.0 s per request against about 4.6 s | MEASURED |
| 2026-11-09 | Volta sPLIt, three-tier | short sets, context rows, prompt reading | the [tokens-per-second table](docs/history.md#tokens-per-second) | MEASURED, same day |

## Hardware

Eight V100-SXM2-32GB on two NVLink quad boards, the same host and settings as the Volta sPLIt measurements of the
same day. Each of the four PCIe switches has its own x16 Gen3 root port and two cards at Gen3 x8. Volta has FP16
`mma.sync.m8n8k4` only: no FP4 or FP8 tensor cores, no `cp.async`, no `ldmatrix`. Every 4-bit format is decoded in
software into FP16 operands, so low-bit experts pay by reading fewer bytes, not by faster math. v100-skinny measured
a read ceiling of about 879 GB/s on a V100-SXM2 and its QPN kernel at 77 % of it at 1 row (UPSTREAM).

## Reproduce

### From the Hugging Face package (`hot_iq`)

**When the upload is complete.** As of 2026-11-10 the package is still uploading: do not download it yet. Once
complete it holds the weights, the MXFP4 expert files and the tier map, so no tally and no pack build is needed.
As of 2026-11-10 the public launcher and the tier-map reader have passed the CPU check (the tier map gives the
measured tiers and digests on all 40 layers, and every expert record of layers 0 and 20 on the first rank is
byte-equal to the measured one) but not yet an eight-card boot. The boot check follows and will be added as a
dated row.

1. Build the engine as step 1 of [From source](#from-source-with-your-own-tally) describes.
2. When the upload is complete, download and check the package (556 GB): `hf download ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4 --local-dir /path/to/pkg`,
   then `cd /path/to/pkg && sha256sum -c SHA256SUMS`. Its card: [Hugging Face](https://huggingface.co/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4) (upload in progress, do not download yet).
3. Lay out the run directory as step 5 there describes.
4. Start: `cd "$LANE/run" && CUDA_VISIBLE_DEVICES=... bash serve_hot_iq.sh /path/to/pkg 8000`. The launcher writes
   `serve-hot_iq.yaml` from the package's profile template, sets the load-time allocator setting the measurements
   ran with, and starts `run_serve.sh` with one prompt slot. At load each rank logs its tiers: stage 0 3,833 hot and
   3,847 warm experts with tier digest `8e7829184090`, stage 1 2,254 and 5,426 with `5b09c0383a80`. With a tier map
   the engine does not plan tiers: a card with less free memory after the dense weights than the measured cards had
   refuses to load and names the bytes it is short.

### From source, with your own tally

1. Build the Volta sPLIt engine as its
   [Reproduce section](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#reproduce)
   describes. Then apply [engine/patches/](engine/patches/) in the engine source root (`patch -p1`; each header gives
   the sha256 of the base file and of the result) and copy the three new files from [engine/src/](engine/src/) into
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
6. Start with `cd "$LANE/run" && PROFILE=serve-nvfp4-pp2-t3n.yaml CUDA_VISIBLE_DEVICES=... bash run_serve.sh nvfp4` (the last word names the server's log and pid files).
   That profile plans `hot_iq` tiers from your tally: no host tier, every compressed record on the cards, the
   most-used experts promoted until the next does not fit. Tiers from another tally are not the measured tiers, and
   their quality is yours to establish. The other profiles of [What runs](#what-runs) start the same way. The three-slot profiles (`serve-nvfp4-pp2-t3-a3.yaml`, `serve-nvfp4-pp2-t3n-a3.yaml`) also need `NVFP4_SLOTS=3`
   on that command line: `run_serve.sh` refuses a profile whose `max-num-seqs` differs from it.
7. Measure with `$LANE/run/`[measure_boot.sh](harness/run/measure_boot.sh) (short sets, long prompts, repeat),
   [ctx_tput.py](harness/run/ctx_tput.py) (context rows) and [harness_loop.py](harness/run/harness_loop.py) (tool
   loop). The legal-style prompt file is not published; any file of one public prompt per line works, saved as
   `$LANE/logs/baseline/legal-public-prompts.txt`.

## Repository layout

| Path | What it is |
|-|-|
| `docs/` | [measurements](docs/measurements.md), [history](docs/history.md) (the steps that led to this build), [plans](docs/plans/), [research](docs/research/LANE-RESEARCH.md) (where the time goes); [index](docs/README.md) |
| `engine/src/` | the three new engine files: the native expert path (with the tier-map reader), the tier planner and the route counters |
| `engine/patches/` | one patch per changed engine file (loader, GGUF MoE method, Engram reader), against the Volta sPLIt engine |
| `engine/configs/` | launcher, environment and serving profiles: the `hot_iq` package's profile template with its launcher `serve_hot_iq.sh`, the earlier steps' profiles, and Volta Full NVFP4's profile, the closeness reference |
| `harness/run/` | measurement scripts: boots, short sets, long prompts, context rows, harness-shaped tests, card memory |
| `harness/benchmarks/nvfp4_8r/` | pack builder, acceptance checks, single-card benches and CPU tests |
| `tools/` | `check_data_boundary.sh` (pre-publication scan) |
| `.publication/SOURCE.json` | what each file came from and what was left out |

## Related builds

Same model, same eight cards, same engine.

| Build | Routed experts | Serving profile |
|---|---|---|
| Volta Max uPLIft (this repository) | as much of the routed work at NVIDIA's NVFP4 precision as fits, in order of use per byte, every routed expert on the cards in decode, the rest from IQ2-class records | `hot_iq.yaml.template` and `serve_hot_iq.sh` in [engine/configs/](engine/configs/) |
| [Volta Full NVFP4](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100) | every routed expert at exactly NVIDIA's NVFP4 precision, the most-used on the cards and the rest served from host RAM over PCIe. Slower than this build. It is the reference of this build's closeness test | `serve-nvfp4-pp2.yaml`, in that repository and in this one's [engine/configs/](engine/configs/serve-nvfp4-pp2.yaml) |
| [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4) | every routed expert from a compressed IQ2-class record, all on the cards. Faster than this build, and the yardstick of its closeness test | in its own repository |

Volta Full NVFP4 carries the engine as released here at commit `79dc164`. Its first commit is
[`fabfca1`](https://github.com/ProprietaryLegal/Volta-Full-NVFP4-DeepSeek-V4.1-Flash-8xV100/commit/fabfca16eed9193e418affce87ba01496045dc94). This repository carries the current engine,
with the tier-map reader.

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
