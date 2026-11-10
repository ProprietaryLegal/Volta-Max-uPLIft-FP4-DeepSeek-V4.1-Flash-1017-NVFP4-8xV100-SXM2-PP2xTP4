# NVFP4-8R build-and-test plans

> Public copy of a build record written on 2026-11-09. Host names, private paths, names and figures from our private traffic were replaced or removed (see `.publication/SOURCE.json`). "Production" means the earlier Volta sPLIt build. `$LANE` is this build's root and `$PROD` the Volta sPLIt checkout. Paths under `logs/` name private receipts that are not published; `research/kernels/` and `research/volta/` are private copies of public sources (links in [docs/README.md](../README.md)).

Plans for speeding up the NVFP4-8R build: NVIDIA's DeepSeek-V4.1-Flash-1017 NVFP4 pack served on
the eight V100 cards (PP2 x TP4, port 18043), next to production (port 18042). Written 2026-11-09
so that each plan can be built without further questions. Background:
`../research/LANE-RESEARCH.md` and `../measurements.md`.

## Files

| File | What |
|---|---|
| `00-WHERE-TIME-GOES.md` | Time and memory budget of a decode round; what is not measured; the 8-card BASELINE table |
| `01-three-tier-residency.md` | Native hot / IQ warm / native host. Built, booted and measured on 2026-11-09 (arm T3a; results in `../measurements.md`) |
| `02-reserve-and-sizing.md` | Per-stage HBM reserve from a measured peak. Done by the maintainer on 2026-11-09 (its section 0); the remainder is listed there |
| `03-dispatch-overhead.md` | Remove the hot/cold split overhead and the per-layer index math |
| `04-prefill-staging.md` | Prefill: stage non-hot experts by DMA copy; later, IQ prefill for the warm tier |
| `05-miss-cache-all-native.md` | All-native track: miss cache in HBM filled by DMA between rounds; same-round DMA through breakable graphs |

## Ranking

The ranked campaign, which orders these plans by expected gain, follows in the next update. Until then a plan
file's own "expected gain" is superseded wherever `../measurements.md` has a measurement.

## Shared rules

1. Work only in `$LANE/` (`engine/`, `run/`, `patches/`, `logs/`). Never write under
   `$PROD/`. Never touch production's files or processes.
2. The cards belong to production unless a measurement window is open. Stop and restore production only
   with the host's own stop and restore commands, and only inside such a window.
3. A Python file changed in `engine/src/vllm/` must also be copied to
   `engine/venv/lib/python3.12/site-packages/vllm/` (the server imports from there). Run `cmp` on the
   two copies and prove the import path:
   `CUDA_VISIBLE_DEVICES= engine/venv/bin/python -c "import MODULE as m; print(m.__file__)"`.
4. One patch file per change under `patches/` (`diff -u` of before and after), plus the before-copy.
5. Fail fast. No fallbacks, no retries, no raised timeouts, no blanket `except`. A check that cannot
   decide prints INCONCLUSIVE (exit 3) and names the missing input. It never passes by default.
6. Types first: parse config once at the loader into a type that carries every invariant. One owner
   per resource.
7. An acceptance check is written before the code and seen failing. Its expected values come from
   outside the implementation: the raw NVFP4 formula, the real weights, arithmetic, or the real
   server. Not from the code under test, not from mocks.
8. Data boundary. The router tally (`engine/private/p16-tally.npz`) derives from private text. Code
   may read it; nothing per-expert leaves memory. Logs, receipts and docs carry only counts of
   experts and shares of routes. No prompts, model output, private names or GPU UUIDs anywhere.
   Tests use the three public prompts of `ab_t1_v2.py`.
9. Label every number MEASURED (with receipt path), ESTIMATED (with arithmetic) or UPSTREAM.
10. The test host tests: 150 W cap, check for foreign compute apps first, `accept_p2.gpu_gate` refuses
    otherwise (exit 3). CPU tests on the 8-card host run with `CUDA_VISIBLE_DEVICES=` (empty).
11. Do not re-propose the lost levers of LANE-RESEARCH §7. One correction: "prefetch / evict-first" there
    means cache hints inside the on-card kernel (production test D21). Predicting which cold experts the next
    layers need was never measured; it is not a lost lever.
12. Record every output path and its size, checked by `ls -la`.

## How to measure

Harness (read-only): `$PROD/1cat-1277/bench5/warmup.py`, `ab_t1_v2.py`, `pairs_arms_v2.py`.

1. Warm-up: `warmup.py http://127.0.0.1:18043` after every boot, before any timing.
2. Throughput: `ab_t1_v2.py --tput http://127.0.0.1:18043 3 OUT.jsonl | tee DIR/ab-LABEL-tput.out`.
   Temperature 0, thinking off, `ignore_eos`, 1,024 tokens, three public prompts. Read
   `dec-ms/round` (decode only: first to last streamed token) and tok/s from the `TPUT` lines.
3. Paired ABBA: boot or switch arms in the order A, B, B, A. Score with
   `pairs_arms_v2.py --metric dec --from-out DIR A1 B1 A2 B2`.
4. Noise floor: repeat the throughput run inside one boot, and compare the two A arms. The noise
   floor is the larger difference. A gain counts only if both pairs agree in sign and the mean
   difference is above twice the noise floor. Otherwise report "no measurable change".
5. Greedy check: `ab_t1_v2.py --compare A.jsonl B.jsonl` is token-exact. Use it when a change must
   not alter output (plans 02, 03, 04-A, 05). Plan 01 changes output by design and has its own
   quality rule.
6. Receipts go under `$LANE/logs/PLAN/`. Keep the boot log path with them.
7. Lane scripts (`$LANE/run/`): `arm_measure.sh LABEL short|full [LONG]` measures one booted arm (short sets twice; `full` adds the context curve and the tool loop).
8. `ctx_tput.py URL CHARS OUT.jsonl [MAX_TOKENS]`: decode speed and read time at one context length (public filler text, legal-style task).
9. `harness_loop.py URL CHARS N OUT.jsonl [THINK] [TAIL_CHARS] [CHATS] [BOUNDARY]`: the harness-shaped tool loop; CHATS > 1 rotates chats (kept-prompt HITS), BOUNDARY=1 adds the keep / strip pair at a new human message.

## Glossary

- Round: one decode step of the target model. It verifies up to 6 rows (5 draft tokens plus one) and
  yields 3.12 tokens on average (114.4 tok/s x 27.3 ms, MEASURED on production).
- M: rows in a step. Decode graphs are captured for M = 1 to 6. More rows is prefill (eager).
- Route: one (token, expert) pair. Each row routes to 6 of 384 experts per layer: 36 routes at M = 6.
- Stage: half of the model under PP2. Stage 0 = layers 0 to 19, stage 1 = layers 20 to 39. Each
  stage runs on four cards (TP ranks 0 to 3).
- Native record: one expert's MXFP4 weights for one TP rank, 4,874,240 bytes. Folded value-exact
  from the NVFP4 checkpoint. Run by the QPN kernel `skinny_moe_qpn_sm70`.
- IQ record: the same expert in production's GGUF quantization (IQ1_M, IQ2_XXS or IQ2_XS by layer),
  1.94 to 2.56 MB per TP rank. Run by the v41k kernels through
  `torch.ops.vllm.gguf_v41_original_expert_dp4a`.
- Hot / warm / host (cold): where an expert's record lives. Hot = native in HBM. Warm = IQ in HBM.
  Host or cold = native in registered host RAM.
- Zero-copy: a kernel reads registered host RAM directly over PCIe (UVA). DMA copy: the driver
  copies host RAM to HBM first.
- Miss: a routed expert that is not in HBM.
- Reserve: HBM per card kept free after the dense weights load (`nvfp4_8r_reserve_bytes`).
- Tally: private per-expert route counts used to rank experts. Only aggregates may be written down.
- ABBA: arm order A, B, B, A, so that drift cancels.
- HBM: the card's memory. PP / TP: pipeline / tensor parallelism.
