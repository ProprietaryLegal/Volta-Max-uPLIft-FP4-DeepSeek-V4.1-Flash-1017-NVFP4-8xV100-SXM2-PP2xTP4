# 02. Reserve and sizing: put the spare HBM to work

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

Read `README.md` and `00-WHERE-TIME-GOES.md` first. Paths are relative to `$LANE/`.
This plan is independent of plan 01 and can be done before or after it.

## 0. Status 2026-11-09: measured and set by the maintainer; no speed gain; what is left

The maintainer did the core of this plan during the first 8-card window (results in docs/measurements.md. Receipts:
`logs/baseline/nvfp4-cal-*`, `run/state/p6-boot3.log` to `p6-boot5.log`). The text
below section 0 is the plan as written before that and is kept for its method.

- The falsifier of section 2 held for stage 1. With 6 GiB on both stages the server died of CUDA
  out-of-memory in `forward_prefill` on a 100K-token prompt (stage 1 cards; MEASURED, boot 3).
- The reserve is now per stage (patch P9: `nvfp4_8r_reserve_bytes` is a map {first layer of the
  stage: bytes}; `NativePackConfig.reserve_for(layers)`). "One value for both stages" in section 5
  no longer applies.
- Calibration boot (stage 0 6 GiB, stage 1 9 GiB), card memory sampled every 15 s during 100K- and
  500K-token prompts (MEASURED): highest use 30,456 MiB on stage 0 and 30,304 MiB on stage 1; the
  card limit is about 32,494 MiB. Need = reserve minus headroom: about 4.0 GiB (stage 0) and
  6.9 GiB (stage 1). Readings taken after a request are not peaks: the engine frees memory again.
- Set (patch P10): stage 0 4.5 GiB, stage 1 7.5 GiB (need plus about 0.5 GiB). Stage 0 gained
  331 hot experts per card, stage 1 lost 330: decode speed is unchanged against 6 / 6 GiB
  (MEASURED: 174.6 against 172.8 dec-ms/round on the general prompts). The gain is that long
  prompts no longer crash; a 498,099-token prompt runs.
- Sampling rule: `nvidia-smi` at most once per 15 s on the 8-card host (house rule). Do not use the `-l 1`
  sampler of section 5; use `run/memwatch.sh`.

What is left of this plan:
1. Find what stage 1 holds that stage 0 does not (about 2.9 GiB more at peak). Candidates: drafter,
   head, their graph pools. Use `torch.cuda.memory_summary()` per worker at ready and after a long
   prompt, behind an environment switch, aggregates only.
2. The prefill working buffers scale with `max-num-batched-tokens` (8,192; about 1.2 GB ESTIMATED in
   section 2). Halving the chunk should release about 0.6 GiB per card for experts at some cost in
   prefill speed. One paired boot decides it.
3. The stage-wide minimum of free HBM (section 5) is still unbuilt; plan 01 step 7 builds it.
4. Margin policy: section 5 says 1 GiB; the maintainer used about 0.5 GiB with a verification run. If a
   long prompt ever dies of out-of-memory, raise the stage's value by 0.5 GiB and re-verify.

## 1. Goal and expected gain

The loader keeps `nvfp4_8r_reserve_bytes` = 6 GiB per card free for everything that allocates after
the experts (KV pool, CUDA graphs, drafter, prefill buffers). The value was set without a
measurement (`patches/P6-serve-reserve.patch`). Goal: measure the real peak and set the reserve to
peak plus a stated margin.

Each GiB released per card buys (arithmetic from MEASURED record sizes):

| Build | Per GiB per card | Arithmetic |
|---|---|---|
| Two-tier | 220 experts move from host RAM to HBM | 2^30 / 4,874,240 |
| Three-tier (plan 01) | 381 (stage 0) or 408 (stage 1) experts move from IQ to native | 2^30 / (4,874,240 - mean IQ record 2,057,472 or 2,241,792) |

Two-tier decode gain in ms/round: not known. It is (misses removed per round) x 0.90 to 1.35 ms
(MEASURED per miss). Step 2 computes the misses at each candidate reserve from the tally, as an
aggregate. Three-tier gain is quality (fewer IQ routes), not speed.

How much can be released: not known. After load the cards have 5.40 / 5.29 GiB free (MEASURED,
`run/state/p6-boot3.log`). What the later allocations take is the measurement of this plan.

## 2. Why it should work; what would falsify it

Known later allocations per card:

| Item | Bytes | Label |
|---|---|---|
| KV pool | 268,435,456 | MEASURED: `kv-cache-memory-bytes` in `run/serve-nvfp4-pp2.yaml` |
| Prefill staging, one per card | 311,951,360 | ESTIMATED: 2 banks x 32 experts x 4,874,240 (`PrefillStaging`, patch P8) |
| `forward_prefill` buffers at 8,192 rows | about 1.2 GB | ESTIMATED: y13 113 MB + inter 63 MB + y2 503 MB + out 168 MB + FP32 g, u, product 340 MB |
| CUDA graph pools (M 1..6), drafter, attention prefill buffers | unknown | measure |

The known items sum to about 1.8 GB of a 6.4 GB reserve. If the unknown items are small, 2 to 3 GiB
can be released. Evidence that the margin is not huge: boot 2 died with CUDA out of memory in
`forward_prefill` when staging was one per layer (20 x 312 MB = 6.2 GB; `run/state/p6-boot2.log`;
fixed by `patches/P8-prefill-staging-one-per-card.patch`).

Falsifier: the measured peak leaves less than the margin free at reserve 6 GiB. Then nothing can be
released; record the numbers and close the plan.

## 3. Upstream-overlap check

| Row | Verdict |
|---|---|
| §0.6 PR #1277 (offload fraction as a fixed 0.33) | Not adopted: this build sizes from measured free HBM, not a fraction |
| §0.9 route 3 (host-RAM offload) | Partly: same mechanism, smaller cold set |
| §0.7 traps | Read before step 3 |

No duplicate upstream work.

## 4. Code map

| File | Lines | Item |
|---|---|---|
| `run/serve-nvfp4-pp2.yaml` | key `nvfp4_8r_reserve_bytes` | the value to change |
| `engine/src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py` | 537-568 | `NativePackConfig` (reserve parsed: positive int, no default) |
| same | 589-603 | `plan_stage`: `budget = free_bytes - cfg.reserve_bytes` |
| same | 503-530 | `choose_residency`: stage-wide ranking by count |
| same | 220-259, 450-453 | prefill staging and `forward_prefill` buffers |
| `engine/src/vllm/model_executor/model_loader/gguf_loader.py` | 89-141 | `_attach_nvfp4_8r`: reads free HBM per rank, logs "free after" |
| `engine/src/benchmarks/nvfp4_8r/test_residency.py` | | pattern for a tally aggregate check |

## 5. Design

Types and states first.

- The reserve stays one required positive int in the profile. No default, no automatic tuning at
  boot: a boot must be reproducible from its profile.
- Stage-wide budget. Today each rank plans from its own free HBM. The four ranks of a stage must
  hold the same expert set (required by plan 01, harmless for two tiers). Take the minimum free
  bytes over the stage's ranks (`dist.all_gather_object` over `get_tp_group().cpu_group`) and plan
  from it. If plan 01 step 7 is already built, this exists; do not build it twice.
- New measuring tool, not part of serving: `run/tools/hbm_peak.sh`. It samples
  `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 1` into
  `logs/reserve/hbm-LABEL.csv`. Index, used and total only: no UUID column.
  `engine/src/benchmarks/nvfp4_8r/hbm_peak_report.py` reads the CSV and prints, per card, the
  minimum free MiB and the time it occurred.
- Margin: 1 GiB per card. This is a policy value (the maintainer may change it), not a measurement. It
  covers allocator fragmentation and requests larger than the test.
- New reserve = 6 GiB - (minimum free over all cards and phases) + margin, rounded up to 64 MiB.
  One value for both stages (the smaller stage sets it).

Per-layer hot-set sizing: nothing to gain in the two-tier build. All records have the same size, so
ranking the whole stage by count (what `choose_residency` does) already maximizes the routes served
from HBM. For three tiers the per-layer IQ record size matters; plan 01 §5.3 handles it.

Checked and rejected, with arithmetic:
- Moving the PP boundary. The stages differ by 0.44 GiB of free HBM (29.30 against 28.86): 97
  experts. Not worth a profile change.
- Dropping the W2 K padding (576 -> 640). It would save 174,080 bytes per record (3.6 %), about 183
  more hot experts per card. It needs a QPN split-K that fits K = 576, which is kernel work in
  v100-skinny (UPSTREAM, LANE-RESEARCH.md section 6 (upstream review)). Not planned here.

## 6. Build steps

1. Write the check of §7 and `hbm_peak_report.py`. Run the check on a made-up CSV whose minimum is
   known by hand: it must fail before the report script exists. Patch
   `patches/R1-hbm-peak-tools.patch`.
2. Tally aggregate (CPU, `CUDA_VISIBLE_DEVICES=`): extend `estimate_round.py` with `--budgets`
   (see `00-WHERE-TIME-GOES.md` M2) and run it for budgets 23.30 / 22.86 plus 1, 2 and 3 GiB.
   Record the estimated misses per layer per round and ms/round for each. Aggregates only.
3. Measuring boot (the maintainer's window), profile unchanged (reserve 6 GiB). Start `hbm_peak.sh` before the
   boot. Phases, each marked in the CSV by a label line: boot and graph capture; `warmup.py`; one
   long prefill (a public text of at least 200,000 tokens, `max_tokens` 16, thinking off); the
   `--tput` run. Stop the sampler. Run the report.
4. Compute the new reserve by the rule of §5. Edit `run/serve-nvfp4-pp2.yaml`. Patch
   `patches/R2-serve-reserve-measured.patch`. If the stage-wide minimum is not built yet, build it
   now in `_attach_nvfp4_8r` and deploy to both trees (README rule 3). Patch
   `patches/R3-stage-wide-budget.patch`.
5. Verification boot with the new reserve: repeat the phases of step 3 with the sampler running.
6. Measure (§8).

## 7. Acceptance check (written first)

`engine/src/benchmarks/nvfp4_8r/check_reserve.py --csv BEFORE.csv --csv-after AFTER.csv --boot-log LOG --margin-mib 1024`

Checks, with expected values from outside the code:
- The loader lines in the boot log show four equal hot/cold counts per stage (text of the log).
- Hot count per stage agrees with `floor(hot budget / 4,874,240)`, capped at 7,680, for the budget
  printed in the same line (arithmetic). The line prints GiB to two decimals, so allow one expert
  either way (0.005 GiB is 1.1 records).
- In the after-CSV every card's minimum free is at least margin / 2 = 512 MiB (real observation of
  the real server).
- The boot log has no "out of memory" line.

PASS needs all four. FAIL if any is false. INCONCLUSIVE (exit 3) if a CSV lacks one of the four
phase labels, covers fewer than 8 cards, or the boot log has no loader line: the run did not
exercise what the reserve protects.

## 8. Measurement

- Before / after are two boots, so use ABBA across boots: A = reserve 6 GiB, B = new reserve. Per
  boot: `warmup.py`, then `ab_t1_v2.py --tput` twice (labels `ARM` and `ARMr`), receipts in
  `logs/reserve/`. Score with `pairs_arms_v2.py --metric dec --from-out logs/reserve A1 B1 A2 B2`.
  Noise floor and counting rule: README "How to measure".
- Greedy check: `ab_t1_v2.py --compare` A against B. For the two-tier build the output must be
  token-exact (the same native weights serve every route from either tier). A difference is a FAIL
  of this plan, not noise.
- Prefill: time to first token of the 200,000-token request in both arms (fewer cold experts
  should make it faster). Report tok/s = prompt tokens / TTFT.
- Compare the measured gain with step 2's estimate and write both in the report.

## 9. Risks, traps, rollback, done

- `nvidia-smi` "used" includes memory PyTorch has cached but not in use. So the measured minimum
  free understates what is really available. That errs on the safe side; do not correct for it.
- A longer prompt than the test, or a second sequence, can allocate more. `max-num-seqs` is 1 and
  the prefill chunk is capped by `max-num-batched-tokens: 8192`; if either changes, redo step 3.
- Host RAM falls as the reserve falls (fewer cold experts), so no host check is needed.
- Do not lower the reserve below the rule "because it booted". A boot that passes and a long
  prefill that dies an hour later is the failure this plan exists to prevent.
- Never run the sampler with a UUID column, and never paste `run/lib.sh`.
- Lost levers (LANE-RESEARCH §7) are not touched by this plan.

Rollback: restore `nvfp4_8r_reserve_bytes: 6442450944` (`patches/P6-serve-reserve.patch` state).

Done when: the check is PASS on the verification boot; the ABBA result and the greedy compare are
in `logs/reserve/`; the maintainer has the new reserve, the peak table and the measured gain.
