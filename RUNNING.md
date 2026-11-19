# Running Volta Max uPLIft

How to start the `hot_iq` build on eight V100-SXM2 32 GB cards, check that it booted as measured, stop it, build it
from source with your own routing tally, and measure it. Every command here is the one the repository's scripts and
READMEs document; nothing in this file was run for it. The figures the build gives are in the
[README](README.md#results); [engine/README.md](engine/README.md) describes every engine file, patch and template.

Placeholders: `<pkg>` is where you download the Hugging Face package, `<build>` this build's root on your host (the
`LANE` of the scripts), `<volta-split>` a checkout of the Volta sPLIt repository (the `PROD` of the scripts), `<engine>`
the engine source tree. Use absolute paths.

## 1. The engine

The engine is the Volta sPLIt engine: [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) with the DeepSeek-V4.1-Flash
package of [#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277) and Volta sPLIt's patch series and production patches
through 0059, built in its own venv (Python 3.12) with its compiled libraries (`v41k.so`, `v41idx.so`, `v41pidx.so` and
the sparse prefill library).

1. Build that engine as the
   [Volta sPLIt Reproduce section](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4#reproduce)
   describes, with its production patches through 0059 (0058 and 0059 carry the two request-path switches).
2. Apply this repository's five patches in the engine source root. Each patch header gives the sha256 of the base file
   and of the result.

   ```bash
   cd <engine>
   for p in <this-repo>/engine/patches/*.patch; do patch -p1 < "$p"; done
   ```

3. Copy the new files of [engine/src/](engine/src/) into the same paths of the engine's `vllm` tree.
4. Build the RB 4 prompt-reading library, or do without it. The builder is CPU only and touches no card; run it from
   [engine/kernels/prefill-rb4/](engine/kernels/prefill-rb4/README.md) with the engine's Python:

   ```bash
   cp <engine>/csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu <engine>/csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh csrc/
   CUDA_HOME=<cuda-12.8> CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=7.0 python -I build_qpn_rb4.py
   ```

   It refuses a source file whose sha256 is not the one it expects, and prints the library's path and sha256 last.
   Copy the library as `v41qpnrb4.so` beside `nvfp4_8r_qpn_rb.py` in the engine tree. That module pins the sha256 of the
   measured build and refuses any other file, so set `LIBRARY_SHA256` in it to your file's sha256. Without the library,
   change `nvfp4_8r_prefill_rb: rb4` to `rb2` in the template you use: the same output bits, without window 17's
   second-read gain.

## 2. The cards and the package

Eight V100-SXM2 32 GB cards on two NVLink boards of four. Tensor parallelism stays inside a board; the hand-off between
the two pipeline stages goes over PCIe. The host needs room for the compressed experts' NVFP4 records in pinned RAM
(180.8 GB) plus the Engram tables, see [Requirements](README.md#requirements).

Download the package (556 GB) and check every file:

```bash
hf download ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-hot_iq-8xV100-SXM2-PP2xTP4 --local-dir <pkg>
cd <pkg> && sha256sum -c SHA256SUMS
```

The package holds the MXFP4 expert files, the GGUF, the drafter, the config and tokenizer, one serving template per
option in `serve/` and one tier map per option in `tier_maps/`. A copy downloaded before 2026-11-13 has old templates:
the package as first uploaded (2026-11-10, `serve/hot_iq.yaml.template` and `tier_map.json` only) lacks the keys
`nvfp4_8r_group_plan`, `nvfp4_8r_swiglu` and `nvfp4_8r_iq_tail`, and templates from before 2026-11-12 lack
`nvfp4_8r_prefill_swiglu` and `nvfp4_8r_prefill_rb`. Download `serve/` and `tier_maps/` again (small files).

## 3. The run directory

The scripts were measured in one layout and find each other there, not in this repository's folders. `LANE` is this
build's root (with `engine/venv/`, `engine/src/`, `models/` and `packs/`), `PROD` the Volta sPLIt checkout (its bench
harness scripts). From this repository's root:

```bash
export LANE=<build> PROD=<volta-split>
mkdir -p "$LANE/run/state" "$LANE/run/cache" "$LANE/logs/baseline" "$LANE/engine/src/benchmarks/nvfp4_8r"
cp engine/configs/* harness/run/* "$LANE/run/"
cp harness/benchmarks/nvfp4_8r/* "$LANE/engine/src/benchmarks/nvfp4_8r/"
```

Copy Volta sPLIt's run tools into `$LANE/run/tools/` (`env.sh` puts them on the Python path), and replace `$LANE` in the
profiles with your path. When `LANE` or `PROD` is not set, the shell scripts stop with a message and the Python files
stop with a `KeyError` naming the variable. Set both; do not edit the files.

## 4. The options and the switches

`NVFP4_OPTION` picks the serving option. Each one is the package's `serve/hot_iq-OPTION.yaml.template` with its tier map
`tier_maps/OPTION.json`.

| Option | Prompt slots | Prompt reading | Notes |
|---|---:|---|---|
| `slots2` | 2 | fast (compressed experts read from their IQ records), RB 4 and the one-launch SwiGLU | the default; quality against native reading not established |
| `slots1` | 1 | fast | |
| `slots3` | 3 | fast | |
| `native-slots1` | 1 | native (every expert read from its NVFP4 record), `chain` and `rb2` | R1's tiers; the reading of the closeness figures |

The routed-expert keys of every template are required and have no default; a template without one is refused at
boot: `nvfp4_8r_packs`, `nvfp4_8r_tier_map`, `nvfp4_8r_reserve_bytes`, `nvfp4_8r_residency`, `nvfp4_8r_prefill_warm`,
`nvfp4_8r_group_plan`, `nvfp4_8r_swiglu`, `nvfp4_8r_prefill_swiglu`, `nvfp4_8r_prefill_rb` and `nvfp4_8r_iq_tail`.
[engine/configs/hot_iq-slots2.yaml.template](engine/configs/hot_iq-slots2.yaml.template) is the default's template.

[run_serve.sh](engine/configs/run_serve.sh) sets the switches the measurements ran with. You do not set them yourself:

```bash
VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0   # load-time allocator setting; under the default scope an eight-card boot died at load
VLLM_V41_SYSTEM_IN_PLACE=1           # a system message sent mid-chat stays where it was sent
VLLM_V41_TOPK_SEG=1024               # segment width of the decode top-k kernel (unset: 8192)
VLLM_V41_PREFIX_LOG=1                # one numbers-only cache log line per admitted request (two to four slots)
VLLM_V41_SNAPSHOT_HIT=1              # a follow-up with a changed previous reply resumes from a slot snapshot (two to four slots)
VLLM_V41_MULTI_SEQ=1                 # the engine's multi-slot mode, with its prompt budget (two to four slots)
```

plus the engine's production switch set, listed in the script. On one slot it unsets the cache log line and the
snapshot hit, which the engine accepts only in its multi-slot mode. The engine prints "Unknown vLLM environment variable
detected: VLLM_V41_SNAPSHOT_HIT" once per boot; that is expected.

## 5. Start and check

```bash
cd "$LANE/run"
NVFP4_OPTION=slots2 CUDA_VISIBLE_DEVICES=<first board's four indices>,<second board's four> \
  bash serve_hot_iq.sh <pkg> 8000
```

[serve_hot_iq.sh](engine/configs/serve_hot_iq.sh) writes the option's template, with the package path and the port
(default 8000, loopback) filled in, to `$LANE/run/serve-hot_iq.yaml`, then starts `run_serve.sh` with that profile and
the template's slot count. The launcher refuses a template, tier map or kernel key that does not match the option. The
server's log and pid are `$LANE/run/state/hot_iq.log` and `.pid`.

Card order. Give the first board's four indices, then the second board's. On a host with more than one NUMA node, pair
each card with the one four places later across two PCIe host bridges: a direct peer transfer between the two cards
behind one host bridge hung on our host. Our list is `0,4,1,5,2,7,3,8`.

Check the boot log. At load each rank logs its tiers, which must be these:

| Option | First stage | Second stage |
|---|---|---|
| `slots2` | 3,691 hot, 3,989 compressed, digest `0b6440ce205f` | 2,245 hot, 5,435 compressed, digest `5b2a5e7e493d` |
| `slots1` | digest `8e7829184090` | digest `c15146b996eb` |
| `slots3` | digest `a5d4b7a514d1` | digest `2be289273a5e` |

With a tier map the engine does not plan tiers. A card with less free memory after the dense weights than the measured
cards had refuses to load and names the bytes it is short. The engine also refuses a tier map whose digest differs, and
the first release's `tier_map.json`.

The server is vLLM's OpenAI-compatible API server with the Anthropic Messages endpoint, the `deepseek_v41` tool-call
parser and the `deepseek_v4` reasoning parser; it admits 524,288 tokens a request.

## 6. Stop

```bash
kill -TERM -- -$(cat "$LANE/run/state/hot_iq.pid")
```

The pid file holds the server's process group, so this stops every worker.

## 7. From source, with your own tally

The package's tier maps carry the result of a private routing tally. To choose your own hot experts:

1. Build the engine as in step 1.
2. Download NVIDIA's NVFP4 pack (`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`, revision `480550d3`), the ISTA-DASLab
   GSQ-RCO GGUF, and DeepSeek's config, tokenizer and DSpark drafter.
   [harness/run/copy_weights.sh](harness/run/copy_weights.sh) copies the pack from `$WEIGHTS_SRC` and checks every file
   against its Hugging Face sha256; it prints `ALL-VERIFIED`. The pack builder refuses to start without that log.
3. Provide a decode routing tally: routing counts per layer and expert from your own traffic, in the array layout that
   `choose_residency` and `choose_tiers` read (see the engine files).
4. Build and check the packs on CPU, `build_packs` then `accept_p1` (bit equality), as
   [harness/benchmarks/nvfp4_8r/run_p1.sh](harness/benchmarks/nvfp4_8r/run_p1.sh) runs them. The package's modules run
   from the engine source tree as `python -m benchmarks.nvfp4_8r.NAME`.
5. Lay out the run directory as in step 3, then start a profile that plans tiers from your tally:

   ```bash
   cd "$LANE/run" && PROFILE=serve-nvfp4-pp2-t3n.yaml CUDA_VISIBLE_DEVICES=<card order> bash run_serve.sh nvfp4
   ```

   The last word names the server's log and pid files. That profile plans `hot_iq` tiers: no host tier, every
   compressed record on the cards, the most-used experts promoted until the next does not fit. A profile with more than
   one slot (`max-num-seqs` 2, 3 or 4) also needs `NVFP4_SLOTS` set to that number, for example `NVFP4_SLOTS=3` for
   `serve-nvfp4-pp2-t3-a3.yaml` and `serve-nvfp4-pp2-t3n-a3.yaml`; `run_serve.sh` refuses a profile whose
   `max-num-seqs` differs from it. Stop it with `kill -TERM -- -$(cat "$LANE/run/state/nvfp4.pid")`.

Tiers from another tally are not the measured tiers, and their quality is yours to establish.

## 8. Measure

The scripts send one request at a time, use public text only, record numbers only (no prompt or generated text) and
stop at the first failure. They call the Volta sPLIt harness scripts (`warmup.py`, `ab_t1_v2.py`) from
`$PROD`; get them from the Volta sPLIt repository's `harness/`. [harness/README.md](harness/README.md) lists every script.

| Script | What it measures |
|---|---|
| `measure_boot.sh URL LABEL [LONG]` | warm-up and the general set, the legal-style set, optional long prompts (`LONG=1`: 100K then 250K; `LONG=2`: 100K then 500K, for a build that changes card memory use), then both sets again |
| `tput_prompts.py URL PROMPTS.txt OUT.jsonl` | the same measurement over a prompt file, one prompt per line |
| `ctx_tput.py URL CHARS OUT [MAX_TOKENS]` | decode speed and read time after a given amount of public context |
| `longprompt.py URL CHARS [MAX_TOKENS [SKIP]]` | one long public prompt (Python standard-library source) with a short answer, for prompt reading and card memory |
| `harness_loop.py URL CHARS N_REQ OUT [THINK] [TAIL] [CHATS] [BOUNDARY]` | a tool loop in the Anthropic Messages format; `CHATS>1` rotates chats, `BOUNDARY=1` tests a new human message with and without earlier thinking |
| `memwatch.sh OUT SECONDS` | card memory every 15 s, index and MiB only |

For example, after a boot on port 8000:

```bash
cd "$LANE/run"
bash measure_boot.sh http://127.0.0.1:8000 my-boot 1
```

The legal-style prompt file is not published. Any file of one public prompt per line works, saved as
`$LANE/logs/baseline/legal-public-prompts.txt`. The 1K to 256K bench of the README uses Volta sPLIt's speed bench, its prompts and method; the method and every
table are in [docs/measurements.md](docs/measurements.md#speed-bench-1k-to-256k).

The closeness test against NVIDIA's NVFP4 pack has its own scripts in [quality/closeness-test/](quality/closeness-test/).
Its README gives the order: each script's self-test, then fetching and building the items, capturing a reference arm
and a `hot_iq` arm from two servers, scoring, and the tables and charts:

```bash
python build_items.py --self-test
python capture.py --self-test
python score.py --self-test
python report.py --self-test
python build_items.py fetch --data data
python build_items.py build --data data --tokenizer tokenizer.json --out plan.json --ids ids.json
```

The capture and scoring commands, with their arguments, are in
[quality/closeness-test/README.md](quality/closeness-test/README.md). The reference arm is NVIDIA's NVFP4 pack served on
the same cards with [engine/configs/serve-nvfp4-pp2.yaml](engine/configs/serve-nvfp4-pp2.yaml).

## 9. Checks without a card

The pack builder's checks, the configuration tests and the tier tests run on CPU from the engine source tree, for
example:

```bash
python -m benchmarks.nvfp4_8r.test_tally_mix
python -m benchmarks.nvfp4_8r.test_route_stats
```

`test_residency` needs a routing tally and is not runnable from this repository alone. The single-card gates
(`accept_*`, `falsify_*`, `test_group_plan`, `test_prefill_rb` and their `run_*_testhost.sh` runners) need one card and
check its conditions first, exiting INCONCLUSIVE rather than running under the wrong ones.

Before a commit, the repository's data-boundary scan refuses weights, archives, corpora, private directories, files over
1 MiB and credential patterns:

```bash
bash tools/check_data_boundary.sh --staged      # the index
bash tools/check_data_boundary.sh --tree HEAD   # every file in a commit
```
