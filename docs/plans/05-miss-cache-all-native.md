# 05. All-native track: miss cache in HBM, and DMA for the miss

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

Read `README.md` and `00-WHERE-TIME-GOES.md` first. Paths are relative to `$LANE/`.
This plan is for residency `two_tier` (every route native). Take it only if the operator rejects IQ precision
for the warm tail (plan 01), or wants the all-native build faster as a second option. It has two
gates before any code: a locality measurement and a reading of the engine's existing cache. It is
not build-ready past those gates; the gates say what to write next.

## 1. Goal and expected gain

Today a cold expert is read from host RAM every time it is routed: +0.90 to +1.35 ms per miss per
layer (MEASURED, `logs/testhost/p3-bench-20261109T185349Z.json`). Nothing is remembered: the same
expert routed in the next round pays again.

- Part A, miss cache: load a missed expert into a spare HBM slot once; later routes to it are hits
  and cost nothing extra. Gain per round = h x (misses per round) x (0.90 to 1.35 ms), where h is
  the share of cold routes that find their expert already cached. h is not measured (INCONCLUSIVE:
  no routing capture exists). Misses per round: 1.33 per layer (ESTIMATED at the old budgets;
  `00-WHERE-TIME-GOES.md` M2 recomputes it).
- Part B, DMA for the load: 743.7 us per expert against 1,014.8 us for an in-kernel gather
  (MEASURED, same receipt): 271 us less per load, 27 %. It needs the host inside the round, which a
  FULL CUDA graph does not allow.

Example of the arithmetic only, not a prediction: if h were 0.5 and misses 53 per round (1.33 x 40),
the gain would be 0.5 x 53 x 0.90 to 1.35 = 24 to 36 ms/round.

## 2. Why it should work; what would falsify it

A cache helps only if routing has temporal locality beyond what the static tally ranking already
captures. Spare HBM spent on cache slots could instead hold more static hot experts (220 per GiB).

Falsifiers:
1. Locality (M3 of file 00): the hit rate h at a cache size that fits the spare HBM is not higher
   than the route share of the same number of extra static hot experts (step 2 computes both).
   Then make the hot set larger (plan 02) and close this plan.
2. The engine's cache cannot hold native records (step 3).
3. UPSTREAM warning: PR #1277 ran this cache with IQ records at offload 0.33 and measured 24.9 ms
   per warm round but 53.5 ms/round on fresh text, with 27 misses per round. A cache does not help
   text that keeps routing to new experts.

## 3. Upstream-overlap check

| Row | Verdict |
|---|---|
| §0.6 PR #1277, engine file `gguf_expert_cache.py` | Duplicate risk, so adopt. The engine already has a graph-safe routed-expert cache (see §4). Part A must reuse it, not write a second one |
| §0.9 route 3 (host-RAM offload) | Adopted: this is route 3 with a cache in front |
| §0.5 v100-skinny QPN | Adopted unchanged: the QPN kernel reads a slot bank as it reads the hot bank |
| LANE-RESEARCH §7 "prefetch / evict-first" | What production tested and lost (D21) was cache hints inside the on-card kernel; not used here. Predicting cold experts and copying them early was never measured: unmeasured, not lost. This plan as written loads only on a real miss; a hit-rate count can ride on step 1's stats boot. Eviction policy stays the lane's own |
| §0.7 traps | Read before step 3 |

## 4. Code map

`engine/src/vllm/model_executor/layers/quantization/gguf_expert_cache.py` (1,373 lines). Its header
says, for decode: one kernel looks the routed ids up in a device slot table and publishes them to a
host-mapped ring; on a miss it spins until the native lane (`csrc/gguf_expert_cache.cpp`) answers
with a copy plan; a second kernel performs the plan, reading pinned host memory in-kernel; a third
updates the slot table and remaps the ids. All inside the captured graph.

| Lines | Item |
|---|---|
| 53, 114, 157 | Triton kernels `_lookup_publish`, `_apply_plan`, `_remap` |
| 197 | `_native()`: the lane; `nvfp4_8r_native.py` already uses its `host_register` and `mapped_view` |
| 232-693 | `ExpertCache`: `__init__` 235, `bank` 436, `store` 445, `start` 469, `stats` 559, `resolve` 577, prefill 646-693 |
| 1357 | `create_cache(banks, num_experts, config, device)` |

Where production calls it: `gguf_moe.py` `apply`, `cache.resolve(self.cache_layer, piece_ids, topk_weights[start:stop])`
(lines 533-535). Config keys (profile `kernel-config.sm70_gguf`): `expert_cache_fraction`,
`expert_cache_offload_fraction`, `expert_cache_spare_slots`, `expert_cache_host_slack`,
`expert_cache_host_storage`.

This build: `nvfp4_8r_native.py` `NativeExpertSet.from_raw_host` 175-210, `_split_groups` 348-363,
`forward` 365-394. Graph breaks: `engine/src/vllm/compilation/breakable_cudagraph.py` (433 lines):
`eager_break_during_capture` 61-128; it does not break when the runtime mode is
`CUDAGraphMode.FULL` (line 107), which is this profile's mode. An existing decorated MoE
function: `fp8_sm70_moe.py` line 312.

## 5. Design

### Part A: the engine's cache in front of the cold bank

Types and states first. No new state machine: the slot table and its transitions (miss, plan, load,
remap, write-back) are the engine cache's, with one owner (the `ExpertCache` and its lane).

- A native record is four byte matrices per expert: q13 [1152, 2560], s13 [1152, 160],
  q2 [5120, 320], s2 [5120, 20]. Each becomes one cache bank `[slots, rows, bytes]`. The two global
  scales per expert (g1, g2, 8 bytes) stay in HBM for all experts, indexed by expert id.
- `NativeV41Moe` holds, instead of the host `cold` set, a `CachedColdSet`: the four slot banks (HBM),
  `g1`/`g2` for every cold expert, and the cache handle. One type; a layer has either a host cold set
  (today) or a cached one, chosen by a required config key `nvfp4_8r_cold: zero_copy | cache`.
- Dispatch: `cold_slot = cache.resolve(layer, cold ids)` replaces `cold_slot_of_expert[ids]`; the
  second QPN launch reads the slot banks. Hot routes must not enter the cache: pass them as ids the
  cache ignores, if it has such an id (step 3 answers), else resolve a fixed-shape id vector whose
  hot positions repeat a cold id already present.
- Slots per layer: at least 36 (a round can route 36 distinct cold experts in one layer; fewer slots
  would make a round unservable). Above 36: from step 2. HBM per slot: 4,874,240 bytes.
- Prefill keeps `forward_prefill` and the host arena; the cache's host copy and the prefill arena
  must be the same memory (one owner). If the cache needs its own host arena, prefill stages from
  that one.

### Part B: DMA for the load (variant, gated)

The load in Part A is an in-kernel gather (4.8 GB/s). A driver copy is faster (6.6 GB/s) but must be
issued by the host between two graph segments. The engine can break a capture at a decorated
function, but not in FULL mode. So Part B = switch the decode graphs to breakable capture, and run
one decorated function per layer that syncs the ids (11.8 us), copies the missing experts (743.7 us
each) into the slot banks and writes the slot map in place. Fixed cost per round, hit or miss:
40 x (11.8 + 57.1 us host round trip) = 2.76 ms (MEASURED parts), plus the cost of the mode change
itself, which is not measured. Attention ops are decorated too, so they would also break.

## 6. Build steps

1. Locality measurement: build and run M3 of `00-WHERE-TIME-GOES.md` on the two-tier boot (the maintainer's
   window; public prompts, then a working session if the maintainer allows). Patch
   `patches/C1-routing-stats.patch`, both trees (README rule 3). Receipt `logs/stats/`.
2. Decision table (CPU; aggregates only), `engine/src/benchmarks/nvfp4_8r/cache_decision.py`: for
   spare HBM of 1, 2 and 3 GiB per card, print (a) static: the tally route share of the next
   220 x GiB experts by rank; (b) cache: slots = GiB / (20 layers x 4,874,240) per layer, and h from
   M3's histogram at the largest g whose mean live-slot count fits. If (b) <= (a) everywhere: stop
   (falsifier 1) and report.
3. Reading gate. Read `gguf_expert_cache.py` whole and `csrc/gguf_expert_cache.cpp` if it is in
   `engine/src` (if only the compiled library is there, say so). Answer in writing to the maintainer:
   the `banks` entry format `create_cache` takes; whether shard names other than w1/w3/w2 and four
   shards per layer are accepted; whether row bytes are free; how `resolve` treats ids to ignore;
   the eviction rule; what `spare_slots` and `fraction` mean; whether slots >= 36 is enforced. If
   the lane is fixed to three GGUF shards and cannot be changed without its source: stop
   (falsifier 2).
4. Only after the maintainer accepts step 3's answers: write build steps for Part A in the form of plan
   01 §6 (check first, one patch per step, both trees), using the real names found.
5. Part B gate, before any Part B code: boot `two_tier` with `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and
   the graph mode the breakable wrapper needs, no other change. ABBA against the FULL boot. If the
   mode alone costs more than misses per round x 0.27 ms minus 2.76 ms, stop.

## 7. Acceptance check (written first, at step 4)

`accept_cold_cache.py` (the test host, one card), modelled on `accept_p3_layer.py`:
- Real packs of layers 5, 12, 27; P2 reference inputs; fp64 references from the raw NVFP4 formula.
  Rule: relative L2 <= 1e-2, eager and graph replay, M = 1 and 6.
- Cache behaviour from arithmetic on the test's own routing sequence (not from the cache's
  counters): replay a fixed sequence of 500 routings through one graph. The test computes, on the
  CPU with a plain LRU model of the same slot count, how many loads must happen; the cache's
  `stats()` must report that number if step 3 found the eviction rule to be LRU, else the bound
  "loads <= distinct cold experts per window". Output of every replay bit-equal to the zero-copy
  path on the same inputs (same weights: any difference is a bug).
- A round with 36 distinct cold experts in one layer must complete and be correct.
- INCONCLUSIVE (exit 3): `gpu_gate` refuses; packs or references missing; the lane library absent.

## 8. Measurement on the real server

Arms: A = `nvfp4_8r_cold: zero_copy`, B = `cache`. One arm = one boot; order A, B, B, A; per boot
`warmup.py`, then `ab_t1_v2.py --tput` twice; receipts in `logs/cache/`; score with
`pairs_arms_v2.py --metric dec --from-out logs/cache A1 B1 A2 B2`. Noise floor and counting rule:
README. Also record the cache's hit and load counts per arm (aggregates) and compare the measured h
with M3's prediction.
Output must be token-exact between arms (`ab_t1_v2.py --compare`): both are all-native.
The three public prompts repeat nothing; also run the maintainer's choice of a longer public session, or
the cache is tested only on its worst case (fresh text, as PR #1277's 53.5 ms/round).

## 9. Risks, traps, rollback, done

- The cache's miss path spins inside a kernel until the host lane answers. A stuck lane hangs the
  round with no error. Do not add a timeout; find the cause (lane thread dead, ring full).
- Slots below 36 per layer can make a round unservable. Enforce at config parse.
- HBM for slots comes out of the hot budget: account for it in `plan_stage`, or the boot overruns
  the reserve.
- PR #1277's lesson (UPSTREAM): on fresh text the cache degrades to a load per miss, about what
  zero-copy costs now, plus the lane round trip. Expect no gain there.
- Do not combine with plan 01's WARM tier before measuring each alone.
- The lost lever of LANE-RESEARCH §7 is cache hints inside the on-card kernel (D21): none here.
  Expert prediction is unmeasured, not lost; build nothing for it before its hit rate is measured.

Rollback: `nvfp4_8r_cold: zero_copy` (same code, old path); full rollback from the before-copies.

Done when: steps 1 to 3 are answered with receipts (that alone can close the plan); if built, the
check is PASS, the ABBA result is recorded with its noise floor, output is token-exact, and
production is restored and healthy.
