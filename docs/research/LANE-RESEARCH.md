# NVFP4-8R lane research (state at 2026-11-09)

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

What the build lane measured and concluded about running NVIDIA's DeepSeek-V4.1-Flash-1017 NVFP4 pack
on the eight V100s (two 4-card NVLink boards, PP2 x TP4, 150 W per card). Every number is labelled
MEASURED, ESTIMATED or UPSTREAM. Measured numbers and their receipts are summarized in `../measurements.md`.
Volta background: the public repositories linked in `../README.md`. Build plans: `../plans/`.

Status: checks P1 to P5 passed off-production (the test host card 0 and CPU). For whether the build has
been booted on the eight cards, see `../measurements.md`. Unless a MEASURED end-to-end ms/round is
cited there, the round times below are estimates assembled from measured parts.

## 0. Measured on eight cards, 2026-11-09 (supersedes the estimates of §1 to §3 where they differ)

All MEASURED on the 8-card host unless marked; receipts under `$LANE/logs/baseline/` (`nvfp4-final-*`,
`prod1109-*`), tables in `../plans/00-WHERE-TIME-GOES.md` (BASELINE).

| Quantity | Two-tier NVFP4 build (port 18043, boot 5) | Production (port 18042, same day) |
|---|---|---|
| Hot / cold experts per rank | stage 0: 5,463 / 2,217; stage 1: 4,704 / 2,976 (reserve 4.5 / 7.5 GiB) | all on the cards (IQ) |
| KV pool at max-model-len 524,288 | 2,076,870 tokens; a 498,099-token prompt ran, 0 OOM | n/a |
| Decode, general public prompts | 163.4 to 174.6 dec-ms/round, 18.8 to 20.1 tok/s | 21.5 dec-ms/round, 137 to 138 tok/s |
| Decode, legal-style public prompts | 81.8 to 87.6 dec-ms/round, 28.2 to 30.2 tok/s | 21.5 to 21.6 dec-ms/round, 118 to 119 tok/s |
| Decode against context length (legal-style task) | 33.7 / 30.6 / 31.5 / 30.2 tok/s at 46 / 30,795 / 106,531 / 303,004 tokens: flat | 26.8 dec-ms/round, 115 tok/s at 215,901 tokens (thinking on) |
| Prefill (cold read) | 1,441 to 1,595 prompt tok/s (104K to 498K tokens) | 2,768 prompt tok/s (215,901 tokens in 78.0 s) |
| Boot, launch to ready | 288 s | house restore 21:37:30Z to 21:40:28Z |

Readings:
- The round time follows the cold routes. A cold fetch costs about 2.3 ms on the server (ESTIMATED: two boots whose
  tally-predicted cold routes differ by 8.1 per round ran legal-style text 18.3 ms/round apart), against 0.9 to 1.35 ms
  on one test host card.
- General text misses far more often than the private tally predicts: twice the round time with the same hot set.
- Decode is 4 times slower than production on legal-style text and 7 times on general text; prefill is 1.7 times slower.
- Reserve tuning is finished (P9, P10): it put more experts on the cards and did not change the speed measurably.
- Kept-prompt reuse (session prefix): the engine admits a hit only when the new prompt reproduces the previous sequence,
  generated tokens included, to within 124 tokens of its end (`KVCacheManager._v41_session_hit`,
  `VLLM_V41_RING_REACH`). The DeepSeek encoder drops the reasoning of turns before the last user message
  (`drop_thinking`), so a follow-up that is a new user turn misses by design; a tool loop inside one turn hits. A test
  that drops the reasoning (`harness/run/harness_shape.py`) re-read 216K tokens on every request on both servers (134 to 140 s
  on this build, 75 to 78 s on production). The tool-loop tests use the Anthropic Messages API format with thinking on and tool
  definitions; `harness/run/harness_loop.py` builds such requests from public text and sends each assistant
  message's thinking and tool_use blocks back inside the turn.
- Top-k indexer: the v100-skinny decode-partition pin does not exist on the V4.1 path; the segmented top-k launches
  grids sized by capacity, not live length (reading of code, magnitude unmeasured). 1Cat #1427 does not apply.

## 1. The core problem: the experts do not fit

| Quantity | Value | Label |
|---|---|---|
| Experts | 15,360 (40 layers x 384) | MEASURED (pack manifest) |
| Per expert, per TP rank | 4,874,240 bytes; 18.59 MiB across 4 ranks | MEASURED |
| All experts | 299,473,305,600 bytes (278.9 GiB) | MEASURED |
| HBM on 8 cards | 256 GiB nominal, about 31.5 GiB usable per card | MEASURED / ESTIMATED |
| Non-expert weights per card | 6.89 GiB (stage 0), 10.21 GiB (stage 1) | ESTIMATED |
| Runtime overhead per card | 4.24 GiB (stage 0), 5.02 GiB (stage 1) | ESTIMATED |
| Experts that must live off-card | at least 14.50 GiB per card stage 0 (3,194 experts), 18.60 GiB stage 1 (4,097) | ESTIMATED |
| Host RAM for the cold set | about 132 GiB | ESTIMATED |

So about 47 % of the experts must be cold (outside HBM). Which experts are cold is chosen from the
private router tally (teacher phase): least-routed first (`choose_residency` in
`engine/src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py`). At the test budgets (20.37 / 16.27 GiB hot per card) the cold
set carries 2.59 % (stage 0) and 4.47 % (stage 1) of decode routes (MEASURED from the tally,
aggregate only).

## 2. Where the time goes

Baseline: production (IQ2_XXS experts, all in HBM) runs 27.3 ms/round, about 114.4 tok/s; its routed
experts cost 7.4 ms of that (MEASURED by the owner). All-native all-hot NVFP4 would add 8.8 ms:
36.1 ms/round, about 86.5 tok/s (ESTIMATED, owner). The operator's bar: 80 to 100 tok/s at full precision is
usable.

Measured parts (the test host card 0, 150 W; bench `harness/benchmarks/nvfp4_8r/bench_p3_miss_paths.py`):

| Part | Cost | Label |
|---|---|---|
| Native expert graph, M=6 round, per layer | 280 to 348 us | MEASURED |
| Native expert graph, M=1, per layer | about 131 us | MEASURED |
| Hot/cold split overhead per layer | +63.5 us (M=1), +101.4 us (M=6); 4.06 ms/round over 40 layers | MEASURED |
| Cold expert read zero-copy from registered host memory (UVA) | 1.0 to 1.4 ms per expert | MEASURED |
| Cold expert DMA copy to HBM then compute | 743.7 us per expert (6.6 GB/s); not graph-safe; id sync 11.8 us | MEASURED |
| Cold expert gather on the SMs | 1,014.8 us per expert | MEASURED |
| Cold expert computed on CPU (AVX2) | 2.95 ms per expert, x4 TP slices | MEASURED |
| Cold experts hit per layer per round | mean 1.33 (stage 0 0.96, stage 1 1.69) | ESTIMATED (simulation from the tally; tokens independent) |
| P2 numerical error, worst relative L2 vs raw NVFP4 formula | 4.78e-4 | MEASURED |
| MXFP4 Engram table decode | 0.91 to 1.01 M rows/s vs 4.2 to 4.6 M for IQ4_NL (about +120 us per lookup) | MEASURED |

Round-time estimates (`harness/benchmarks/nvfp4_8r/estimate_round.py`; all ESTIMATED from the parts above):

| Cold-expert path | ms/round | vs all-hot 36.1 |
|---|---|---|
| Zero-copy from host memory (what the build does now) | 96.9 | +60.8 |
| DMA copy per miss | 69.8 to 79.8 | +33.7 to +43.7 |
| SM gather per miss | 84.8 to 94.8 | +48.7 to +58.7 |
| CPU compute of cold experts | 190 to 200 | +154 to +164 |

Reading: the loss is not the native kernel (8.8 ms) but cold-expert misses. A miss costs about a
millisecond because one expert slice is 4.87 MB and host-to-card transfer runs at 6 to 12 GB/s; with
1.33 misses per layer per round (ESTIMATED above) over 40 layers that is 30 to 60 ms. The fix must cut the number of
bytes crossing PCIe per round, not speed up the kernel. The simulation assumes no temporal locality;
real locality (not yet measured: no per-token routing capture, result INCONCLUSIVE) would only help a
cache.

## 3. Candidate directions, ranked by expected gain

1. Three-tier residency (mixed pack): native NVFP4 for the hot experts, the existing IQ2_XXS record
   (already fits in HBM) for the warm tail, host memory only for experts that are almost never routed.
   No PCIe traffic on the decode path. Sizing (MEASURED from the tally, aggregates only):

   | Stage | Host tier at route share 0.0 | Hot native | IQ tier | Routes served native / IQ |
   |---|---|---|---|---|
   | 0 | 360 | 3,430 | 3,890 | 93.32 % / 6.68 % |
   | 1 | 606 | 2,282 | 4,792 | 87.42 % / 12.58 % |

   Allowing 0.2 host routes per round (MEASURED from the tally): stage 0 host 744, hot 3,574, IQ 3,362;
   stage 1 host 1,035, hot 2,442, IQ 4,203.
2. Miss cache in HBM (keep recently used cold experts on-card, evict least recent): gain depends on
   temporal locality, which needs a per-token routing capture first.
3. DMA copy instead of zero-copy for misses: 17 to 27 ms/round better than zero-copy but breaks CUDA
   graph capture; needs the miss handled outside the captured graph.
4. Remove the split overhead (4.06 ms/round) by making the hot path identical to all-hot when a round
   has no miss.
5. MXFP4 Engram table (P4 passed): quality change, costs about 120 us per lookup; not a speed lever.

The detailed, step-by-step plans are in `../plans/`.

## 4. Production parity (lever gap)

MEASURED by reading production's process environment and comparing sources and libraries:

- Production runs with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, an interleaved 8-card order for PP2 x TP4,
  `VLLM_USE_V2_MODEL_RUNNER=1`, `OMP_NUM_THREADS=1`, spawn workers, and about 40 `VLLM_V41_*` levers
  (keys share, fused attention prep/post, route warps 4, Engram gather rows 1024, index tile 32 MB,
  coupled drafts, plan fuse, tiled/fused index, KV hot positions 524288, attention chain, fused
  prefill, packed KV, session prefix, route fast, expert fast, PP keep batch) plus two
  `VLLM_SM70_*` allreduce levers.
- The lane launcher originally exported only four of them (DOWN_ROWS, SESSION_PREFIX, PACKED_KV, INDEX_TILE_MB); since patch P6 (`engine/configs/run_serve.sh`) it carries production's full set.
- The lane engine source references every sampled lever in as many files as production's source, and
  the lane's four helper libraries (sparse prefill, v41idx, v41pidx, v41k) are byte-identical to
  production's. So the gap was launcher-only.
- Levers specific to the IQ expert path (`EXPERT_FAST` in `gguf_v41_moe.py`; `EXPERT_DIET` has no
  Python reference) are bypassed by the native path, which returns from `_apply_nvfp4_8r` first.
  `ROUTE_FAST` (router) applies to both. Carrying the full production set is expected to be harmless;
  the boot fails fast if not.

Action: DONE (patch P6): the launcher takes the full production lever set and the lane's library paths.

## 5. Context window

Serve profile: `max-model-len 524288`, one sequence, KV pool sized so the total is at least 1,000,000
tokens (production's pool is 1,067,729 tokens, MEASURED by the owner). To be confirmed in this
build's boot log.

## 6. Upstream review (UPSTREAM)

- Volta roofline: FP16 tensor cores only; 879 GB/s HBM read; decode is bandwidth and overhead bound.
- v100-skinny QPN kernels: the packed 4-bit expert kernel this build uses
  (`skinny_moe_qpn_sm70`).
- 1Cat PR #1277 (4x V100 TP4, IQ2_XXS experts, offload 0.33): warm round 24.9 ms; fresh-text
  53.5 ms/round, dominated by 27 cold-expert misses per round over a 12.2 GB/s uplink. Same shape
  of problem as section 2.
- 12 GPUs is ruled out: the operator set the target at 8 cards.

## 7. Levers already tried and lost on production (do not re-propose)

Expert top-p prune; async scheduling under PP2 with speculative decode; GATE_COLS/RF; SPINLOOP_EXT;
FUSE_TRANSITION; SM70_MOE_ADD_ALLREDUCE as env-only; 200 W cap; L2 warm-up; dmv13_xd; K4/K3 drafts;
prefetch / evict-first. Method for any new claim: warm-up first, ABBA paired ms/round, greedy check
against the boot reference.

## 8. Data boundary

The router tally derives from private text. It is not in this repo; only the aggregates above are.
No expert-level counts, prompts, model output or GPU UUIDs are written here.
