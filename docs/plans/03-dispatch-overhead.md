# 03. Dispatch overhead: make the no-miss round cost what all-hot costs

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

Read `README.md` and `00-WHERE-TIME-GOES.md` first. Paths are relative to `$LANE/`.
Part A needs plan 01 built. Part B applies to any residency.

## 1. Goal and expected gain

A layer with a hot and a cold bank costs more than an all-hot layer even when no route is cold:

| M | All-hot graph | Split graph, no miss | Extra | Label |
|---|---|---|---|---|
| 1 | 131.1 us | 194.6 us | +63.5 us | MEASURED, `logs/testhost/p3-bench-20261109T185349Z.json` |
| 6 | 347.1 us | 448.5 us | +101.4 us | MEASURED, same receipt |

Over 40 layers at M = 6 that is 4.06 ms/round (arithmetic). The extra work is two kernel launches
on the cold bank (W13 and W2) that find no group, and about ten small tensor ops in `_split_groups`.
How the 101.4 us divides between them is not measured; step 1 measures it.

- Part A (config only): residency `hot_iq` from plan 01 has no host bank in decode. Expected gain:
  the launches' share of the 4.06 ms, at most 4.06 ms/round (`hot_iq` is not accepted: quality not established).
  ESTIMATED upper bound.
- Part B (code): compute the whole group plan in one kernel instead of a chain of tensor ops. The
  chain also runs in the all-hot path (`_compact`), so the gain can exceed the split overhead. Not
  estimated: step 1 gives the bound (index-math us per layer x 40).

## 2. Why it should work; what would falsify it

A CUDA graph replays every captured kernel. Each small op is one launch with fixed overhead, so ten
ops on 36 integers cost far more than their arithmetic. An empty-group launch still starts its
grid (min(slots, bank) groups) before each block exits.

Falsifiers:
- Step 1 shows the index math (`_split_groups` alone) under 20 us per layer at M = 6 and the two
  empty launches under 10 us together. Then neither part can pay; close the plan with the receipt.
- Part A: the ABBA difference between `hot_iq_host` and `hot_iq` is inside the noise floor.
- Part B: the fused kernel is not faster than the chain in the step 1 bench.

## 3. Upstream-overlap check

| Row | Verdict |
|---|---|
| §0.5 v100-skinny QPN | Partly. The compaction here is upstream's (`skinny_sm70_moe._apply_grouped`, lines 670-746). Before Part B, search v100-skinny and the 1Cat PRs for a device-side group-plan kernel. If one exists, adopt it instead of writing one |
| §0.6 PRs | Not adopted: none fuses the group plan |
| §0.7 traps | Read before Part B |
| LANE-RESEARCH §7 lost levers | FUSE_TRANSITION, GATE_COLS/RF and SM70_MOE_ADD_ALLREDUCE are different levers and stay lost. This plan fuses only the integer group plan of the native expert path |

## 4. Code map

`engine/src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py`:

| Lines | Item |
|---|---|
| 325-341 | `_compact`: argsort, gather, compare, cumsum, `scatter_reduce_`, `scatter_` |
| 343-346 | `_groups` (all-hot) |
| 348-363 | `_split_groups` (hot + cold): two sums, where, minimum, arange, two gathers, clamps, casts |
| 365-394 | `forward`: the launches over the plan |
| plan 01 §5.6 | `_tier_groups` (three tiers), if built |

Other: `engine/src/vllm/model_executor/layers/fused_moe/sm70/reduction.py` (lines 53-90: the house
pattern for a Triton kernel, `_mtp_weighted_reduce_kernel` and its launcher);
`engine/src/benchmarks/nvfp4_8r/accept_p2.py` (`time_graph` line 94, `gpu_gate` line 43);
`accept_p3_layer.py` (`graph_of`); `bench_p3_miss_paths.py` (the bench that produced the table);
`test_p3_split_groups.py` (kernel grid rule on CPU).

## 5. Design

### Part A: no host bank in decode

No new type. `Residency = HotIq` (plan 01 §5.1) already makes the host launch unrepresentable:
`host_in_decode` is false and the launch list has one entry. The experts that `hot_iq_host` would
put on the host (zero tally routes) become WARM. They stay native in host RAM for prefill.

### Part B: one kernel for the group plan

State: none. It is a pure function of the keys.

```python
@dataclass(frozen=True)
class GroupPlanBuffers:            # owned by one NativeV41Moe; int32; fixed size; written in place
    perm: torch.Tensor             # [S]      S = max_tokens * 6
    gids_hot: torch.Tensor         # [S]
    goff_hot: torch.Tensor         # [S + 1]
    gids_host: torch.Tensor        # [S]      present only when host_in_decode
    goff_host: torch.Tensor        # [S + 1]
```

`group_plan(key, n, nh, nc, buffers)` writes exactly what `_split_groups` / `_tier_groups` return
today, for the first n = 6M keys:
- `rank[i]` = number of j with `key[j] < key[i]`, or `key[j] == key[i]` and j < i (stable order);
  `perm[rank[i]] = i`.
- A sorted position starts a group if it is 0 or its key differs from the one before. Group index =
  number of group starts at or before it, minus 1. `goff[g]` = position of the start of group g;
  entries past the last group = n. `gids[g]` = key at that start.
- Hot launch: `gids_hot = gids if gids < nh else 0`; `goff_hot = min(goff, count of keys < nh)`.
- Host launch: rotate left by the number of hot groups; `goff_host = min(rotated goff, count of keys < nh + nc)`;
  `gids_host = clamp(rotated gids - nh, 0, nc - 1)`.

S is at most 36, so the O(S^2) counting rank is 1,296 comparisons: one small program. Vehicle:
Triton, as `reduction.py` does; no new shared library. The planner of this document is not sure of
the best Triton formulation for the scatter steps: read `reduction.py` and the Triton documentation
first, and keep the kernel to one launch. If it needs more than two launches, stop: the gain is gone.

The Python chain stays in the file as the reference path for `forward_prefill` (eager, any M).
Selection is a required constructor argument `group_plan: Literal["chain", "kernel"]` set from a
required config key `nvfp4_8r_group_plan`. No silent fallback from kernel to chain.

## 6. Build steps

1. Per-part bench, the test host one card: `engine/src/benchmarks/nvfp4_8r/bench_dispatch_parts.py`. With
   `graph_of` and `time_graph`, at M = 1 and 6, layer 5 rank 0, time graphs that contain only:
   (a) `_compact`; (b) `_groups`; (c) `_split_groups`; (d) one W13 launch on the HBM bank with all
   groups empty; (e) the same on the host bank; (f) the full all-hot forward; (g) the full split
   forward with no miss. Receipt `logs/dispatch/parts-STAMP.json`. Check (g) - (f) against the
   table in §1 (within 10 %); if not, the bench is wrong: fix it before reading the parts.
   Patch `patches/D1-bench-dispatch-parts.patch`.
2. Decide from the receipt (falsifiers of §2). Write the decision and the numbers to the maintainer.
3. Part A: after plan 01, boot arms `hot_iq_host` 0.0 and `hot_iq` (§8). No code.
4. Part B, check first: extend `test_p3_split_groups.py` into `test_group_plan.py` (§7). It must
   fail (no kernel). Patch `patches/D2-test-group-plan.patch`.
5. Part B kernel in a new file `nvfp4_8r_group_plan.py` next to `nvfp4_8r_native.py`; buffers and
   the `group_plan` argument in `NativeV41Moe`; config key in `NativePackConfig` and
   `run/serve-nvfp4-pp2.yaml` (`chain` first). Deploy to both trees (README rule 3). Patch
   `patches/D3-group-plan-kernel.patch`.
6. Rerun step 1's bench with the kernel: new part (h) = kernel alone, (i) = full forward with the
   kernel. Then §8.

## 7. Acceptance check (written first)

`test_group_plan.py` (the test host, one card; `gpu_gate` first).
- Inputs: 20,000 random routings at M = 1..6 over random hot / host / warm maps, plus edge cases:
  all hot; none hot; all one expert; every route a different expert; all warm; host only.
- Expected values, computed route by route from the maps on the CPU, without any compaction code:
  under the kernel's grid rule (group g covers `perm[goff[g]:goff[g+1]]`, skipped when empty), every
  hot route is covered exactly once by the hot launch with its hot slot; every host route exactly
  once by the host launch with its cold slot; no warm route is covered; every `gids` value is inside
  its bank.
- Pass rule: zero violations. FAIL on any. INCONCLUSIVE (exit 3) if `gpu_gate` refuses or Triton
  cannot compile on the card (name the error).
- Then a layer check: `forward` output with `kernel` bit-equal to `chain` on the P2 reference
  inputs of layers 5, 12, 27, eager and graph replay, M = 1 and 6. This is a regression check; the
  coverage check above is the acceptance.

## 8. Measurement

- Per layer (the test host): parts receipt before and after; report us per layer at M = 1 and 6.
- Real server, Part A: arms A = `hot_iq_host` 0.0, B = `hot_iq`. Part B: A = `chain`, B = `kernel`.
  One arm = one boot; order A, B, B, A; per boot `warmup.py` then `ab_t1_v2.py --tput` twice;
  receipts in `logs/dispatch/`; score with `pairs_arms_v2.py --metric dec --from-out logs/dispatch A1 B1 A2 B2`.
  Noise floor and counting rule: README.
- Output: Part B must be token-exact (`ab_t1_v2.py --compare`). Part A changes which experts are
  warm, so compare with plan 01's quality rule instead; the two arms differ only on experts with
  zero tally routes, so expect near-identical output and report the first divergence if any.

## 9. Risks, traps, rollback, done

- The first Triton call compiles. Compile before graph capture (a warm call at construction), or
  the capture fails. The cache directory is `run/cache/triton` (`run/env.sh`).
- Integer overflow and dtype: the QPN op takes int32 `perm`, `gids`, `goff`. Keep int32 end to end.
- The kernel must write every element of every buffer for the first n entries and `goff[n]`;
  a stale tail from a larger M would be read as a group.
- Do not remove the host launch for `two_tier` or `hot_iq_host`: correctness needs it when a route
  is on the host.
- The IQ op's own fixed cost (plan 01) is outside this plan: the deployed `v41k.so` has no matching
  source in `engine/src`.

Rollback: Part A: set `nvfp4_8r_residency` back. Part B: `nvfp4_8r_group_plan: chain`; full
rollback from the before-copies in `patches/`.

Done when: the parts receipt exists and the decision is recorded; each part that was built has a
PASS check and an ABBA result above the noise floor (or is recorded as "no measurable change" and
switched off); production restored and healthy.
