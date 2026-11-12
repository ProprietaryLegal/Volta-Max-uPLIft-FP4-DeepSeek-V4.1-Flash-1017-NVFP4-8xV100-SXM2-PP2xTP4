# Engine

This build runs on the engine of [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4)
(1Cat-vLLM #1277 with Volta sPLIt's patch series and production patches). It adds nine files and changes three.

The files here are those the `hot_iq` measurements on the mixed tally ran with (2026-11-10), with the tier-map
change of this release added: a profile can name a tier map (`nvfp4_8r_tier_map`, the Hugging Face package's
`tier_map.json`) in place of a tally, and the native path reads the package's MXFP4 files, which hold every expert
in expert-id order. The measurements dated 2026-11-09 and the earlier ones of 2026-11-10 ran on the engine file of
the first public commit, which is in this repository's history.

Since 2026-11-11 the native path also carries the group-plan kernel (plan 03 part B, MEASURED on eight cards on
2026-11-11, [docs/measurements.md](../docs/measurements.md#2026-11-11-r1-tiers-and-the-group-plan-kernel)). Every
profile must name `nvfp4_8r_group_plan`: `kernel` builds each decode step's route plan in one Triton launch, and
`chain` builds the same int32 tensors, bit for bit, with the chain of tensor operations every earlier measurement
ran with. There is no default. The kernel is the setting of the package template here, and `chain` is the fallback
and the setting of every other profile in `configs/`. The Hugging Face package as uploaded (2026-11-10) has no
such key.

Since 2026-11-12 ([docs/measurements.md](../docs/measurements.md#prompt-slots-fast-prompt-reading-and-four-kernels-2026-11-12)) five more files are in. Every
profile must also name `nvfp4_8r_swiglu` (`chain` or `kernel`), and a profile with a compressed tier
`nvfp4_8r_iq_tail` (`torch` or `fused`). There is no default for either. Since window 17 every profile also names
`nvfp4_8r_prefill_swiglu` (`chain` or `kernel`) and `nvfp4_8r_prefill_rb` (`rb2` or `rb4`), with no default. The
templates of the two-slot default and its options carry `kernel`, `fused`, the two prompt-reading kernels
(`nvfp4_8r_prefill_swiglu: kernel`, `nvfp4_8r_prefill_rb: rb4`) and fast prompt reading
(`nvfp4_8r_prefill_warm: iq`, quality not established). The `native-slots1` template carries `chain` and `rb2`. `serve_hot_iq.sh` picks the option with `NVFP4_OPTION` and no longer takes `NVFP4_GROUP_PLAN`. The
Hugging Face package as uploaded has none of these keys and does not boot on this engine until its small-file
update.

## New files (`src/`)

| File | What it does |
|---|---|
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py` | reads the MXFP4 packs, prepacks experts for the QPN kernel (into HBM, or into a page-aligned pinned host arena registered with CUDA), chooses residency per stage from free HBM and the tally, and runs the routed experts: one grouped QPN launch per projection and bank, V4.1's limited SwiGLU and the weighted sum in FP32, CUDA-graph safe for 1 to 6 rows; prompt chunks stage host experts into two shared HBM buffers per card. `VLLM_NVFP4_8R_LOAD_TRACE=1` logs card and allocator memory during the stage build (counts and bytes only). It was added after the measurements of 2026-11-09 and is off by default. The prompt-reading step allocates and frees its tensors in the order of the memory fix (checked on one card, [CHANGELOG](../CHANGELOG.md)). With a tier map it reads the v2 pack format (`nvfp4-8r-mxfp4-v2-id-order`: one file per layer and rank, all 384 records in expert-id order) and takes its tiers from the map. `VLLM_NVFP4_8R_STATS=1` switches on the route counters (the row after next). `nvfp4_8r_group_plan` (required, `chain` or `kernel`) chooses how decode builds the route plan of its expert launches. Prompt reading always uses the chain |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_tiers.py` | the three-tier planner (pure numpy): hot, warm and host tiers per stage from the tally and the HBM budget; logs counts of experts and shares of routes only. The tier-map reader: parses a tier map whole (format, layers, ascending ids, stated counts, a sha256 digest per stage), serves `hot_iq` only, and refuses to load when the stage's record budget cannot hold the map's hot experts or the digest differs |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_group_plan.py` | the group-plan kernel (plan 03 part B): one Triton launch that turns a decode step's expert routes into the permutation, launch-block groups and group offsets of the QPN launches (all-resident, hot and host, and three-tier layers), the same int32 tensors as the chain bit for bit, CUDA-graph safe, compiled when the layer is built. Checked by `harness/benchmarks/nvfp4_8r/test_group_plan.py` and `check_layers_mt6.py` |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_swiglu.py` | the SwiGLU kernel (window 12): one Triton launch computes V4.1's limited SwiGLU of the NVFP4 experts in decode, in place of a chain of tensor operations. Profile key `nvfp4_8r_swiglu` (required, `chain` or `kernel`). Token for token the same output as the chain in window 12. Checked by `harness/benchmarks/nvfp4_8r/test_swiglu.py`. Since window 17 it also holds the one-launch SwiGLU of prompt reading (profile key `nvfp4_8r_prefill_swiglu`, required, `chain` or `kernel`), bit-equal to the chain, checked by `harness/benchmarks/nvfp4_8r/test_prefill_swiglu.py` |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_iq_tail.py` | the fused IQ tail (patch G1, windows 14 and 14c): one Triton launch does the weighted sum of the compressed experts' step in decode. Profile key `nvfp4_8r_iq_tail` (required with a compressed tier, `torch` or `fused`). Token for token the same output as `torch` in both windows. Checked by `harness/benchmarks/nvfp4_8r/test_iq_tail.py` |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_qpn_rb.py` | the RB 4 prompt-reading launcher (window 17): with `nvfp4_8r_prefill_rb: rb4` the QPN expert kernel of prompt reading handles 32 token rows per weight load instead of 16, same output bits. It loads the compiled library `v41qpnrb4.so` from its own folder and refuses a missing file or one whose sha256 differs from `LIBRARY_SHA256`. The library is not in this repository: [kernels/prefill-rb4/](kernels/prefill-rb4/README.md) has its source and build script. With `rb2` the module loads nothing. Checked by `harness/benchmarks/nvfp4_8r/test_prefill_rb.py` |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_warm_prefill.py` | fast prompt reading (`nvfp4_8r_prefill_warm: iq`, windows 13b and 9d): prompts read the compressed experts from their IQ2-class records on the cards with the Volta sPLIt engine's IQ prefill kernels, in place of staging their NVFP4 records from host RAM. It changes the output from the first token: quality not established |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_tally_mix.py` | pure functions, no file, no device and no data: mixes decode tallies so that each kind of text has the same say in each layer, and sizes a host tier by a route budget |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_stats.py` | decode route counters per layer and tensor-parallel rank, off unless `VLLM_NVFP4_8R_STATS=1`: device tensors updated inside the decode CUDA graphs, snapshots written beside the tally on a label change. The route shares of the `hot_iq` measurements come from them. A tier-map profile refuses the switch (it has no tally directory to write to) |

## Changed files (`patches/`)

Each patch is a unified diff against the Volta sPLIt engine file as deployed on 2026-11-09. Its header gives the
sha256 of that base file and of the result; the assembler checked that applying the patch to the base reproduces this
build's file exactly. Apply in the engine source root with `patch -p1`.

The two request-path switches of window 16, `VLLM_V41_SYSTEM_IN_PLACE` and `VLLM_V41_TOPK_SEG`, are not patches
of this repository. They are the Volta sPLIt engine's own production patches `0058-system-notes-in-place.patch`
and `0059-topk-segment-width.patch` (in that repository under `engine/production/`, public since 2026-11-11).
Build the engine with them. This build's `run_serve.sh` exports both switches.

| Patch | What changes |
|---|---|
| `patches/gguf_loader.patch` | admits `nvfp4_8r_packs`, either `nvfp4_8r_tally` or `nvfp4_8r_tier_map` (never both, the tier map with `hot_iq` only, and its log line names the map), `nvfp4_8r_reserve_bytes` (a map from a stage's first layer to bytes), `nvfp4_8r_residency`, `nvfp4_8r_host_route_share` and `nvfp4_8r_prefill_warm` (`native_staged`, or `iq` for fast prompt reading), parsed once and all required. It drops routed-expert tensors from the GGUF weight map before any read. After load, each stage measures free HBM, subtracts its reserve and attaches the native expert layers (two-tier or three-tier) on each worker's own card |
| `patches/gguf_moe.patch` | the GGUF MoE method hands routed experts to the native path: 1 to 6 rows through the graph path, other row counts through the prompt path |
| `patches/deepseek_v41_engram_io.patch` | a reader for MXFP4 Engram rows (144 bytes) next to IQ4_NL; the table format follows the file given in `engram_path` (`.json` manifest for MXFP4, `.gguf` for IQ4_NL) |

## Configs (`configs/`)

| File | What it is |
|---|---|
| `configs/run_serve.sh` | launcher: refuses if a server already listens or a GPU has a compute process, sources `env.sh`, sets Volta sPLIt's production switch set, since 2026-11-12 also the two request-path switches of window 16 (`VLLM_V41_SYSTEM_IN_PLACE=1`, `VLLM_V41_TOPK_SEG=1024`) and the load-time allocator setting (`VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0`), checks the profile's `max-num-seqs` against `NVFP4_SLOTS` (1 to 4), starts the server in its own process group |
| `configs/env.sh` | environment and import check |
| `configs/serve-nvfp4-pp2.yaml` | two-tier profile |
| `configs/serve-nvfp4-pp2-t3.yaml` | three-tier profile (`hot_iq_host`, host route share 0.0) |
| `configs/hot_iq.yaml.template` | the profile of the default option, a copy of `hot_iq-slots2.yaml.template`, with `${PKG}` for the package path. The package as uploaded carries the first release's file as `serve/hot_iq.yaml.template` (one slot, reserves 4.5 / 7.5 GiB, none of the new keys) until its small-file update |
| `configs/hot_iq-slots1.yaml.template` to `hot_iq-slots3.yaml.template` | the options of 2026-11-12: one to three prompt slots (window 15 measured one to four), each with its tier map (`${PKG}/tier_maps/slotsN.json`), fast prompt reading, `nvfp4_8r_swiglu: kernel`, `nvfp4_8r_iq_tail: fused`, `nvfp4_8r_prefill_swiglu: kernel` and `nvfp4_8r_prefill_rb: rb4`. Reserves: the one-slot fast-reading reserves plus 160 MiB per extra slot on both stages and 192 MiB on stage 1. Quality not established |
| `configs/hot_iq-native-slots1.yaml.template` | the `native-slots1` option: one slot, native reading (`native_staged`), the R1 tiers (`${PKG}/tier_maps/native-slots1.json`), `chain` and `rb2` for prompt reading |
| `configs/serve_hot_iq.sh` | its launcher: `serve_hot_iq.sh PKG [PORT]` writes `serve-hot_iq.yaml` from the package's `serve/hot_iq-OPTION.yaml.template` with `tier_maps/OPTION.json` and starts `run_serve.sh` with that option's slot count. `NVFP4_OPTION` picks the option: `slots2` (default), `slots1`, `slots3` or `native-slots1`. It refuses a template whose slot count, tier map or kernel keys do not match the option (checked on the CPU with a stand-in for `run_serve.sh`, seven cases, 2026-11-12) |
| `configs/serve-nvfp4-pp2-t3n.yaml` | three-tier with no host tier (`hot_iq`) on the first tally: not accepted (quality not established) |
| `configs/serve-nvfp4-pp2-t3-a3.yaml` | three-tier with three prompt slots (`max-num-seqs 3`, reserves 5.3 / 8.1 GiB): measured, not accepted (quality not established) |
| `configs/serve-nvfp4-pp2-t3-a1r.yaml` | the three-slot profile's one-slot control: same reserves, `max-num-seqs 1` |
| `configs/serve-nvfp4-pp2-t3n-a3.yaml` | no host tier (`hot_iq`) on the first tally with three prompt slots (`max-num-seqs 3`, reserves 5.3 / 8.1 GiB): not accepted (quality not established) |

Paths are written with `$LANE` (this build's root, with `engine/`, `models/`, `packs/` and `run/` under it). The
shell scripts and Python files read it from the environment and stop when it is not set; in the YAML profiles
replace it with your path. The launcher and profiles run from `$LANE/run/` (the copy commands are in the README's
Reproduce section). The profiles name the
private tally file (`engine/private/p16-tally.npz`); point `nvfp4_8r_tally` at your own. The package template needs
no tally: it names the package's tier map.

The compiled libraries the launcher names (`v41k.so`, `v41idx.so`, `v41pidx.so`, the sparse prefill library) are the
Volta sPLIt builds. Build them from its source. The RB 4 prompt-reading library `v41qpnrb4.so` is not in this
repository either: build it from [kernels/prefill-rb4/](kernels/prefill-rb4/README.md) into the folder of
`nvfp4_8r_qpn_rb.py` and set `LIBRARY_SHA256` in that module to your build's sha256, or run with
`nvfp4_8r_prefill_rb: rb2` (same output bits, without the second-read gain).
