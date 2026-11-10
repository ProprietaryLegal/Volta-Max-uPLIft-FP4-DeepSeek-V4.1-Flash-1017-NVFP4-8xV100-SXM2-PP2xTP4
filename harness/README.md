# Harness

## `run/`: measuring a booted server

| Script | What it does |
|---|---|
| `measure_boot.sh URL LABEL [LONG]` | the standard measurement of one boot: warm-up and general set, legal-style set, optional long prompts (`LONG=1`: 100K then 250K; `LONG=2`: 100K then 500K for builds that change card memory use), then both sets again |
| `baseline.sh URL LABEL [N]` | the Volta sPLIt harness's `warmup.py` and `ab_t1_v2.py --tput` (general set) |
| `tput_prompts.py URL PROMPTS.txt OUT.jsonl` | the same measurement over a prompt file, one prompt per line |
| `ctx_tput.py` | decode speed and read time after a given amount of public context |
| `longprompt.py` | one long public prompt with a short answer, for prompt reading and card memory |
| `memwatch.sh OUT SECONDS` | card memory every 15 s (index and MiB only) |
| `harness_shape.py` | one chat shaped like an agent conversation: long prompt, follow-ups that add the answer and new text |
| `harness_loop.py` | a tool loop in the Anthropic Messages API format; `CHATS>1` rotates chats, `BOUNDARY=1` tests a new human message with and without earlier thinking |
| `arm_measure.sh LABEL short|full [LONG]` | one arm: `measure_boot.sh`, then for `full` the context rows and the tool loop |
| `swap_arm.sh PREV NEXT PROFILE [SLOTS]` | stops one lane server by its recorded process group (only with no request running) and boots the next |
| `copy_weights.sh` | copies the NVIDIA pack from `$WEIGHTS_SRC` and checks every file against the Hugging Face sha256; prints `ALL-VERIFIED` |

All of them send one request at a time, use public text only, record numbers only (no prompt or generated text) and
stop at the first failure. They call the Volta sPLIt harness scripts (`warmup.py`, `ab_t1_v2.py`) under
`$PROD/1cat-1277/bench5/`; get them from the Volta sPLIt repository's `harness/`. Shell scripts and Python files
read `LANE` and `PROD` from the environment and stop if either is unset; set both, do not edit the files. The
legal-style prompt file named by `measure_boot.sh` is not published.

## `benchmarks/nvfp4_8r/`: building and checking

A Python package; place it at `benchmarks/nvfp4_8r/` in the engine source tree and run modules with
`python -m benchmarks.nvfp4_8r.NAME` (NAME is the module, for example `accept_p1`).

Runner layout. The CPU reference runners (`run_p2_refs.sh`, `run_p3_prefill_refs.sh`, `run_f_prefill_refs.sh`) run
from `$LANE` with `engine/venv/bin/python` and `PYTHONPATH=engine/pypath`; on the 8-card host `engine/pypath` holds
one link, `benchmarks -> ../src/benchmarks`, so the package imports while vLLM comes from the venv. They write
references under `$LANE/p2/` and logs under `$LANE/logs/`. The `*_testhost.sh` runners run from `$W`, a work
directory on a one-card test host holding `venv/` (the engine venv), `pypath/` (the same role as `engine/pypath`),
`packs/` (the packs of layers 5, 12 and 27, ranks 0 and 3, with `packs/manifest.json`), `p2/` (the references from
the CPU runners) and, for the three-tier runners, `t3/` (`iq-manifest.json`, the IQ slices, the `t3-refs-L*.npz`
references and `v41k.so`); they write to `cache/`, `logs/` and `bench/` under `$W`.

| Group | Files |
|---|---|
| Packs (CPU) | `probe_p1.py`, `build_packs.py`, `accept_p1.py`, `probe_completed_layer.py`, `run_p1.sh`, `scan_p2_scales.py`, `copy_gguf.sh`, `copy_p5_assets.sh`, `check_p5_paths.py` |
| References (CPU, FP64 from the raw weights) | `make_p2_refs.py`, `run_p2_refs.sh`, `run_p3_prefill_refs.sh`, `make_p4_refs.py`, `make_t3_refs.py`, `make_t3_iq_slices.py`, `make_f_prefill_refs.py`, `run_f_prefill_refs.sh` |
| Single-card gates (test host) | `accept_p2.py`, `accept_p3_layer.py`, `accept_p4.py`, `accept_t3_layer.py`, `accept_t3_stage.py`, `falsify_t3_iq_drop.py`, `falsify_f_iq_prefill.py`, and their `run_*_testhost.sh` runners |
| Benches and estimates | `bench_p3_miss_paths.py`, `cpu_moe.py`, `cpu_mxfp4_moe.c`, `estimate_round.py` |
| CPU tests | `test_p3_loader_wiring.py`, `test_p3_split_groups.py`, `test_residency.py`, `test_t3_dispatch.py`, `test_t3_iq_bank.py`, `test_t3_quality.py`, `test_t3_stage_plan.py`, `test_t3_tiers.py` |
| Engram and quality | `pack_engram_mxfp4.py`, `check_engram_mxfp4_file.py`, `sparse_tables.py`, `quality_t3.py` |

Every single-card gate checks the card first (power cap, no other compute process, enough free host memory) and
exits INCONCLUSIVE rather than running under the wrong conditions. Scripts that need the routing tally read it from
the path you give; the tally and anything derived per expert from it (route lists, reference routes) are not
published.
