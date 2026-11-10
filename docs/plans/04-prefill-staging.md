# 04. Prefill staging: DMA copy now, IQ prefill for the warm tier later

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

Read `README.md` and `00-WHERE-TIME-GOES.md` first. Paths are relative to `$LANE/`.
This plan changes prefill only (M above 6, eager). Decode ms/round does not move.
Part A applies to any residency and is build-ready. Part B needs plan 01 and is an outline with a
gate: its first step decides whether it can be built.

## 1. Goal and expected gain

In prefill every non-hot expert a chunk routes to is copied from host RAM into an HBM staging bank,
32 at a time (`forward_prefill`, `_run_cold`). The copy is `torch.index_select` on the mapped host
view: a kernel on the card pulls the bytes over PCIe.

| Way to bring one expert (4,874,240 B) to HBM | us per expert | Rate | Label |
|---|---|---|---|
| Kernel gather from mapped host RAM (today) | 1,014.8 | 4.8 GB/s | MEASURED, `logs/testhost/p3-bench-20261109T185349Z.json`, `sm_gather` k=1 |
| DMA copy from pinned host RAM | 743.7 | 6.6 GB/s | MEASURED, same receipt, `dma` k=1 |

Part A: switch staging to DMA copies. Expected: 26.7 % less staging time per expert
((1,014.8 - 743.7) / 1,014.8, arithmetic on MEASURED values from the decode bench; not measured in
prefill). Staging per stage per large chunk, if the chunk routes to every non-hot expert (ESTIMATED
upper bound): two-tier 127 to 132 experts per layer x 20 layers: 2.6 to 2.7 s -> 1.9 to 2.0 s. Three-tier (plan
01) 207 to 245 per layer: 4.2 to 5.0 s -> 3.1 to 3.6 s. Whether staging dominates prefill is not
measured; step 1 measures it.

Part B: run WARM experts on their IQ records (already in HBM after plan 01) in prefill too. Staging
shrinks to the HOST tier (18 to 30 experts per layer at share 0.0, or none with `hot_iq`), and the
native host-RAM copies of WARM experts are no longer needed (about 150 GB of host RAM, ESTIMATED
from plan 01 §5.3). Cost: the IQ prefill compute, not measured in this build.

## 2. Why it should work; what would falsify it

Part A: prefill is eager, so the host knows which experts a chunk uses (`used`, line 436) and can
issue driver copies. The copy stream and double buffering already exist. The host arena is
registered with the driver (`from_raw_host`, line 200), so a copy from it needs no staging buffer.
Falsifiers: step 1 shows staging is under 10 % of prefill time; or PyTorch does not treat the
registered arena as pinned (`is_pinned()` false), so `copy_(non_blocking=True)` blocks; or the
measured prefill tok/s does not improve beyond the noise floor.

Part B falsifier: production's IQ prefill (`SharedOriginalPrefill`) cannot run on a chosen subset of
experts without its cache object (step B1).

## 3. Upstream-overlap check

| Row | Verdict |
|---|---|
| §0.6 PR #1277 (cold experts over a 12.2 GB/s uplink) | Partly: same transport limit. That PR's rate is UPSTREAM; ours is 6.6 GB/s MEASURED |
| §0.9 route 3 (host-RAM offload) | Adopted for prefill; this plan makes its copy cheaper |
| §0.9 route 1 (mixed pack) | Part B extends it to prefill |
| §0.4 Volta roofline | No kernel change |
| LANE-RESEARCH §7 "prefetch / evict-first" | What production tested and lost (D21) was cache hints inside the on-card kernel. This plan uses none. Predicting cold experts is a different lever and is unmeasured; this plan needs no prediction: a chunk's experts are known before the copy |

## 4. Code map

`engine/src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py`:

| Lines | Item |
|---|---|
| 175-210 | `NativeExpertSet.from_raw_host`: `backing`, `arena`, `host_views` (192-195, CPU views of the arena, dropped after the fill), `host_register` (200), `mapped_view` (201) |
| 111-123 | `NativeExpertSet` fields; `host_backing` |
| 217-259 | `_STAGED_FIELDS`, `PrefillStaging` (two banks, copy stream), `prefill_staging` (one per card) |
| 403-472 | `forward_prefill`: `used` list (436), batches of 32 (438-449) |
| 474-500 | `_run_cold`: line 488 is the copy to replace |

`engine/src/benchmarks/nvfp4_8r/bench_p3_miss_paths.py` lines 109-131: the two copy arms that
produced the table (`ring[slot].copy_(pinned[e], non_blocking=True)` against `torch.index_select`).
`accept_p3_layer.py`: prefill checks at M = 64 and 512 against fp64 references.
Part B: `engine/src/vllm/model_executor/layers/quantization/gguf_shared_prefill.py`
(`SharedOriginalPrefill.__init__` 239-328, `prepare` 727-834, `forward` 836-1090).

## 5. Design

### Part A

Type change: `NativeExpertSet` gains `host_views: tuple[torch.Tensor, ...] = ()`: the four CPU
views (q13, s13, q2, s2) of the registered arena, set only by `from_raw_host`. A host set always has
them; an HBM set never does. `from_raw_host` raises unless `host_views[0].is_pinned()` is true.

`_run_cold` copy, per batch, on the copy stream:
- for the code and scale fields: for each position j and host slot s of the batch,
  `buf[name][j].copy_(c.host_views[i][s], non_blocking=True)`;
- for the global-scale field (`g1` or `g2`, already in HBM): `torch.index_select` as today.
The batch's slot numbers come from the host list `used` (pass the slice as a Python list next to
the device `ids`). Events, double buffering and the staging owner (one per card) are unchanged.
No fallback to `index_select`: one path.

Timing tool, on when `VLLM_NVFP4_8R_PREFILL_TRACE=1` (prefill runs Python on every call): in
`forward_prefill` add `torch.cuda.synchronize` and wall-clock marks around the staged part and the
whole call; every 50 calls log the sums: calls, rows, staged experts, seconds staged, seconds total.
Aggregates only. Off in every timed arm (the synchronize calls slow it).

### Part B (outline)

Three partials per rank in prefill: native HOT from HBM, native HOST staged, IQ WARM from the
`IqWarmBank`. They add, as in decode (plan 01 §5.6). The IQ part should reuse production's prefill
kernels through `SharedOriginalPrefill.forward`, which reads its inputs from a cache object
(`cache.specs`, `cache.banks`, `cache.runtime.table`, `cache.cold_offsets`, `cache.group`,
`cache.prefill_tokens`). The design is a small adapter type that presents the warm bank as an
all-resident cache of W experts with identity slots, and ids mapped to warm slots with the other
routes dropped. Whether those kernels drop a negative id in prefill is not known. Do not build
before step B1 answers both questions.

Part B should call the F2
kit, production's IQ prefill kernel, which is in this engine behind `cache.prefill_moe`
(`gguf_moe.py`, after the point where the native path returns; upstream 1Cat #1387,
`VLLM_V41_PREFILL_IQ_GEMM=1`; first token -4.1 % at 32K, -4.3 % at 150K, -2.4 % at 300K, UPSTREAM).
Prefill waits on stage 0, so Part B is worth about 1.7 to 1.9 s per chunk (ESTIMATED there). The
three-tier build reads 1,080 to 1,327 prompt tok/s against 1,419 to 1,595 two-tier (MEASURED
2026-11-09, `logs/baseline/t3-T3a-ctx.jsonl`), so Part B is one of the first next steps.

## 6. Build steps

1. Measure first. Add the timing tool (patch `patches/F1-prefill-trace.patch`, both trees, README
   rule 3). In a maintainer's window boot once with the trace on and run the long-prefill probe of §8.
   Receipt `logs/prefill/trace-STAMP.json`: share of prefill time spent staging. If under 10 %,
   stop and report.
2. Write the check of §7 first; it must fail (`host_views` missing).
3. `NativeExpertSet.host_views` and the `is_pinned` assertion. Verify on the test host: build one host
   set; print `is_pinned()`. If false: INCONCLUSIVE, stop, report. Do not allocate a second pinned
   copy (it would double host RAM).
4. `_run_cold` DMA copies. Verify: §7 on the test host. Patch `patches/F2-prefill-dma.patch`.
5. Deploy to both trees; import-path proof; boot; §8.

Part B:
- B1 (reading gate). Read `gguf_shared_prefill.py` lines 239-328 and 727-1090. List every `cache.`
  and `self.` attribute `prepare` and `forward` use, and what happens to an id < 0. Write the list
  to the maintainer. If it needs cold-set planner state that has no all-resident value, or cannot drop
  ids, stop: Part B is not buildable this way.
- B2 onward: only after the maintainer accepts B1's list; write those steps then, in the form of plan 01.

## 7. Acceptance check (written first)

`accept_prefill_dma.py` (the test host, one card; modelled on `accept_p3_layer.py`'s prefill part).
- Real packs of layers 5, 12, 27; a hot/cold split that puts at least 100 experts on the host;
  inputs and fp64 references from `run_p3_prefill_refs.sh` (M = 64 and 512; raw NVFP4 formula).
- Expected: relative L2 against the fp64 reference <= 1e-2 (the P2/P3 rule, external to the code).
- Also: the output is bit-equal to the same forward before the change (`patches/` before-copy
  loaded as a second module). The staged bytes are the same, so any difference is a bug. This is a
  regression check; the reference check is the acceptance.
- Race check: run M = 512 two hundred times with a batch size of 4 (many buffer swaps); every run
  must be bit-equal to the first.
- INCONCLUSIVE (exit 3): `gpu_gate` refuses, a reference file is missing, or `is_pinned()` is false.

## 8. Measurement on the real server

Long-prefill probe, `run/tools/prefill_probe.py` (new): POST `/v1/chat/completions` on
`http://127.0.0.1:18043`, streaming, temperature 0, thinking off, `max_tokens` 16,
`stream_options.include_usage`. Prompt: a public-domain text of at least 200,000 tokens, with a
first line `Run LABEL N.` so that prefix caching (`enable-prefix-caching: true`) cannot serve
it. Record time to first token and `usage.prompt_tokens`; prefill tok/s = prompt tokens / TTFT.
Receipt `logs/prefill/probe-LABEL.jsonl`: numbers and the 16 output token ids only.

Arms: A = before, B = after; one arm = one boot; order A, B, B, A; two probes per boot (labels
`ARM` and `ARMr`). Noise floor: the larger of the within-boot differences and |A1 - A2|. A gain
counts if both pairs agree in sign and the mean difference is above twice the noise floor.
Quality: the 16 output tokens of A and B must be identical (same weights, same arithmetic). Also run
`ab_t1_v2.py --tput` once per arm and `--compare` the arms: decode must be token-exact and its
ms/round unchanged within the noise floor.

## 9. Risks, traps, rollback, done

- A non-blocking copy returns before the bytes arrive. The existing `ready` event orders compute
  after the copy; keep every copy inside `with torch.cuda.stream(copy)` and before `ready.record`.
- 64 small driver calls per batch instead of 3 gathers: if per-call overhead shows in step 4's
  timing, copy runs of adjacent slots in one call (slots are sorted; contiguous runs are common
  when a chunk uses most experts).
- The CPU views alias the registered arena. Never write through them after load.
- PCIe is shared by the four cards of a board. Per-card rates measured on one card (the test host) may
  not add up on the 8-card host; the probe on the real server decides.
- The trace's `synchronize` calls change timing: never leave the trace on in a timed arm.
- Part B changes prefill output for warm experts (IQ precision); it would need plan 01's quality
  rule, applied to a prefill-heavy prompt.

Rollback: restore `nvfp4_8r_native.py` from the before-copy to both trees; `cmp`.

Done when (Part A): the check is PASS; the probe ABBA is recorded with its noise floor; the output
tokens are identical; the maintainer has prefill tok/s before and after. Part B: B1's list delivered.
