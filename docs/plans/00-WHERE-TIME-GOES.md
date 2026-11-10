# 00. Where the time goes

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

State on 2026-11-09. Every number is MEASURED (receipt named), ESTIMATED (arithmetic shown) or
UPSTREAM. Paths are relative to `$LANE/`. "Owner" means the maintainer of the Volta sPLIt production server, whose measurements
these are; they are quoted in `docs/research/LANE-RESEARCH.md` §2.

## 1. Time budget of one decode round (M = 6, two-tier build)

| Part | ms/round | Label | Receipt or arithmetic |
|---|---|---|---|
| Production round, all parts | 27.3 | MEASURED | owner; 114.4 tok/s, so 3.12 tokens per round |
| of it: routed experts (IQ, all in HBM) | 7.4 | MEASURED | owner; 185 us per layer |
| of it: everything else (attention, dense, all-reduce, sampling) | 19.9 | ESTIMATED | 27.3 - 7.4 |
| Native experts, all in HBM | 16.2 | ESTIMATED | 7.4 + 8.8 (an earlier unpublished per-layer bench) |
| same, cross-check on the test host | 11.2 to 13.9 | MEASURED per layer | 280 to 348 us x 40; `logs/testhost/p3-bench-20261109T185349Z.json`, one rank, no all-reduce |
| All-native, all in HBM, whole round | 36.1 | ESTIMATED | 19.9 + 16.2; 86.5 tok/s |
| Hot/cold split, no miss | +4.06 | MEASURED per layer | graph M=6: 448.5 us split vs 347.1 us resident = +101.4 us, x 40. M=1: 194.6 vs 131.1 = +63.5 us |
| Each expert read zero-copy from host RAM | +0.90 to +1.35 per miss per layer | MEASURED | same receipt, M=6 with k misses: 1,347.6 (k=1), 2,458.6, 3,947.5, 5,708.8, 6,953.0, 8,574.0 us (k=6), minus 448.5, divided by k |
| Misses per layer per round | 1.33 (stage 0 0.96, stage 1 1.69) | ESTIMATED | simulation from the tally at hot budgets 20.37 / 16.27 GiB; tokens independent; `logs/p3-round-estimate.json` |
| Two-tier round, zero-copy misses | 96.9 | ESTIMATED | `estimate_round.py` at the same old budgets |

Other measured costs of a miss (same receipt): DMA copy to HBM 743.7 us per expert (6.6 GB/s), not
graph-safe; kernel gather from mapped host RAM 1,014.8 us per expert (4.8 GB/s); id sync to the host
11.8 us; host round trip 57.1 us; CPU compute 2.95 ms per expert per TP slice.

Reading: the native kernel is not the problem (16.2 ms). Misses are: about a millisecond each, tens
per round. The remedy is fewer bytes over PCIe per round.

## 2. Memory budget per card (two-tier build, 8-card boot)

MEASURED from `run/state/p6-boot2.log` and `p6-boot3.log` (loader line `nvfp4_8r: TP rank`), same on
all four ranks of a stage:

| Quantity | Stage 0 | Stage 1 |
|---|---|---|
| Card capacity | 31.73 GiB | 31.73 GiB |
| Free before experts | 29.30 GiB | 28.86 GiB |
| Reserve (`nvfp4_8r_reserve_bytes`) | 6.00 GiB | 6.00 GiB |
| Hot budget | 23.30 GiB | 22.86 GiB |
| Hot / cold experts | 5,132 / 2,548 | 5,034 / 2,646 |
| Free after experts | 5.40 GiB | 5.29 GiB |
| "Model loading took" | 25.09 GiB, 254 to 258 s | 26.61 GiB, 268 s |

KV pool: 2,076,870 tokens (MEASURED, `p6-boot3.log`, line "GPU KV cache size").

These supersede LANE-RESEARCH §1, which estimated 6.89 / 10.21 GiB of non-expert weights and hot
budgets of 20.37 / 16.27 GiB. The measured dense footprint is 31.73 - 29.30 = 2.43 GiB (stage 0) and
2.87 GiB (stage 1). So the cold sets are smaller than the estimate (2,548 / 2,646 against 3,194 /
4,097), and the last two rows of the table in §1 overstate the misses. They have not been recomputed.

IQ records (MEASURED from the GGUF headers, CPU only): routed-expert tensors total
132,073,390,080 bytes. Per expert per rank: 11520 x (2 x bb13 + bb2) bytes, bb = 56 (IQ1_M),
66 (IQ2_XXS) or 74 (IQ2_XS): 1,935,360 to 2,557,440 bytes by layer. All 384 experts of all 20 layers:
15,801,384,960 bytes per card (stage 0), 17,216,962,560 (stage 1).

## 3. Not measured, and how to measure it

M1. Decode ms/round of the two-tier build on 8 cards. The maintainer's baseline run (table below).

M2. Miss share at the measured budgets. `engine/src/benchmarks/nvfp4_8r/estimate_round.py` (in this
repository `harness/benchmarks/nvfp4_8r/estimate_round.py`) holds the budgets in `STAGES`
(20.37 / 16.27). Add a `--budgets S0 S1` argument (GiB), then run
it with `--budgets 23.30 22.86 --receipt logs/testhost/p3-bench-20261109T185349Z.json`. It prints
aggregates only. This replaces 1.33 and 96.9 above. CPU only (`CUDA_VISIBLE_DEVICES=`).

M3. Real misses per round and their temporal locality. No per-token routing capture exists, so
whether a miss cache helps is INCONCLUSIVE. Measure with in-graph counters, on when
`VLLM_NVFP4_8R_STATS=1`:
- In `NativeV41Moe.__init__` add device tensors: `stat` int64 [3] (calls, cold routes, cold groups),
  `last_round` int32 [cold slots] filled with -2**30, `gap_hist` int64 [7].
- In `forward`, after the compaction, with device ops only (so it works inside the captured graph):
  `is_cold = key >= nh`; `slot = where(is_cold, key - nh, 0)`; `r = stat[0]`;
  `gap = r - last_round[slot]`; `gap_hist.scatter_add_(0, bucketize(gap, [1, 2, 4, 16, 64, 256]), is_cold.long())`;
  `last_round.scatter_reduce_(0, slot, where(is_cold, r, -2**30).int(), "amax")`; add to `stat`.
- A daemon thread per worker writes every 60 s, summed over the card's layers: `stat`, `gap_hist`,
  and the mean count of cold slots with `r - last_round <= g` for g in (4, 16, 64, 256). File:
  `logs/stats/nvfp4-8r-stats-STAGE-RANK.jsonl`. Aggregates only; no expert ids.
- Read: `gap_hist` up to bucket g, divided by cold routes, is the hit rate of a cache that keeps
  every cold expert used in the last g rounds. The slot count is the cache size it needs.
- Run it on the three public prompts and, with the maintainer's consent, on a normal working session.

M4. Cost of the IQ op with few live routes (plan 01 depends on it): arm difference in plan 01 §8.

M5. Why the native experts cost 16.2 ms in the estimate but 11.2 to 13.9 ms per-layer on the test host.
An all-in-HBM arm cannot exist on 8 cards, so plan 03 step 1 (per-part bench on the test host) separates
the parts instead.

M6. Prefill speed (tok/s on a long public prompt) of the two-tier build. Plan 04 §8.

M7. Peak HBM per card during graph capture, a long prefill and decode. Plan 02 §8.

M8. Layer accuracy on real activations. The P2 and P3 references use x ~ N(0, 1) and random
routing (`make_p2_refs.py`). No captured activations exist. Not planned; the end-to-end greedy and
quality checks cover it.

## BASELINE (8 cards)

Two-tier build, boot 5 of 2026-11-09 (`run/state/p6-boot5.log`; reserve 4.5 GiB stage 0, 7.5 GiB
stage 1, patches P9 and P10), port 18043, measured as README "How to measure" with
`run/measure_boot.sh` (receipts `logs/baseline/nvfp4-final-*`). All MEASURED. Run 1 is after the
warm-up; run 2 is the within-boot repeat, taken after a 104K- and a 498K-token prompt.

| Quantity | Value | Source |
|---|---|---|
| Boot time, launch to /health 200 | 288 s | boot log, launch 20:54:05Z, ready 20:58:53Z |
| Hot / cold experts per rank, stage 0 | 5,463 / 2,217 | loader line `nvfp4_8r: TP rank` |
| Hot / cold experts per rank, stage 1 | 4,704 / 2,976 | loader line `nvfp4_8r: TP rank` |
| Free HBM after load per card, stage 0 | 3.81 GiB | loader line, "free after" |
| Free HBM after load per card, stage 1 | 6.89 GiB | loader line, "free after" |
| KV pool tokens | 2,076,870 | boot log, "GPU KV cache size" |
| General prompts (bench5 set), total, run 1 / run 2 | 174.64 / 163.37 dec-ms/round; 18.8 / 20.1 tok/s; 3.32 tok/round | `nvfp4-final-g1-tput.out`, `-g2-` |
| General p0 (essay), run 1 / run 2 | 163.84 / 153.51 dec-ms/round; 14.4 / 15.4 tok/s; 2.38 tok/round | `TPUT p0` lines |
| General p1 (explanation), run 1 / run 2 | 159.32 / 151.46 dec-ms/round; 24.0 / 25.3 tok/s; 3.89 tok/round | `TPUT p1` lines |
| General p2 (Python module), run 1 / run 2 | 212.20 / 195.28 dec-ms/round; 20.6 / 22.4 tok/s; 4.43 tok/round | `TPUT p2` lines |
| Legal-style public prompts, total, run 1 / run 2 | 87.58 / 81.79 dec-ms/round; 28.2 / 30.2 tok/s; 2.52 tok/round | `nvfp4-final-l1-tput.out`, `-l2-`; prompts `logs/baseline/legal-public-prompts.txt` |
| Within-boot noise (run 1 against run 2) | 6.5 % general, 6.6 % legal-style; run 2 is the faster both times | the two rows above |
| Prefill, 104,049-token prompt | 72.2 s, 1,441 prompt tok/s | `nvfp4-final-long100k.out` |
| Prefill, 498,099-token prompt | 326.6 s, 1,525 prompt tok/s | `nvfp4-final-long500k.out` |
| Peak card memory sampled every 15 s during those prompts | stage 0 32,086 MiB, stage 1 31,356 MiB (limit about 32,494) | `nvfp4-final-long*-mem.txt` |
| Same build at reserve 6 / 6 GiB (boot 3; unsafe: died on a 100K prompt) | general 172.82; legal-style 82.85 and 80.10 dec-ms/round | `nvfp4-r6-*` |
| Same build at reserve 6 / 9 GiB (boot 4, calibration) | general 200.41; legal-style 99.84 dec-ms/round | `nvfp4-cal-*` |
| Production, same day, same prompts (port 18042, 21:42Z, after the house restore) | general 21.48 / 21.57 dec-ms/round, 136.9 / 137.9 tok/s, 3.13 tok/round; legal-style 21.54 / 21.62 dec-ms/round, 118.7 / 118.3 tok/s, 2.67 tok/round | `logs/baseline/prod1109-{g1,l1,g2,l2}-tput.out` |
| Production, 215,901-token prompt, thinking on | read 78.0 s (2,768 prompt tok/s); decode 26.80 dec-ms/round, 115.0 tok/s | `logs/baseline/prod1109-harness-think1.jsonl`, request 0 |
| Two-tier build, 303,004-token context, legal-style memo task | 87.19 dec-ms/round, 30.2 tok/s; read 190.0 s (1,595 prompt tok/s) | `logs/baseline/nvfp4-final-ctx.jsonl` |

Readings (they supersede the estimates of section 1):
- Decode is 2 to 4 times slower than the 36.1 ms all-in-HBM estimate plus 4.06 ms, and slower than
  the 52 to 62 ms the measured split was expected to give.
- A miss costs about 2.3 ms on the 8-card server, not 0.9 to 1.35: boot 4 has 8.1 more predicted
  cold routes per round than boot 3 (ESTIMATED from tally aggregates: 23.90 against 15.82) and ran the
  legal-style
  prompts 18.3 ms/round slower (ESTIMATED: 18.3 / 8.1).
- General text misses far more often than the tally predicts. The routing tally is private and is not
  published. On the general prompts the round is twice as long as on the legal-style prompts with the same hot
  set.
- Tally aggregates at the boot 5 budgets (24.80 / 21.36 GiB): 0.841 % of stage 0 decode routes and
  1.392 % of stage 1 routes go to cold experts (MEASURED from the tally). That is 6.05 + 10.02 = 16.07
  cold routes per 6-row round (ESTIMATED: share x 720 routes per stage and round).
