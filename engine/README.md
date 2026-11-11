# Engine

This build runs on the engine of [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4)
(1Cat-vLLM #1277 with Volta sPLIt's patch series and production patches). It adds three files and changes three.

The files here are those the `hot_iq` measurements on the mixed tally ran with (2026-11-10), with the tier-map
change of this release added: a profile can name a tier map (`nvfp4_8r_tier_map`, the Hugging Face package's
`tier_map.json`) in place of a tally, and the native path reads the package's MXFP4 files, which hold every expert
in expert-id order. The measurements dated 2026-11-09 and the earlier ones of 2026-11-10 ran on the engine file of
the first public commit, which is in this repository's history.

## New files (`src/`)

| File | What it does |
|---|---|
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py` | reads the MXFP4 packs, prepacks experts for the QPN kernel (into HBM, or into a page-aligned pinned host arena registered with CUDA), chooses residency per stage from free HBM and the tally, and runs the routed experts: one grouped QPN launch per projection and bank, V4.1's limited SwiGLU and the weighted sum in FP32, CUDA-graph safe for 1 to 6 rows; prompt chunks stage host experts into two shared HBM buffers per card. `VLLM_NVFP4_8R_LOAD_TRACE=1` logs card and allocator memory during the stage build (counts and bytes only). It was added after the measurements of 2026-11-09 and is off by default. The prompt-reading step allocates and frees its tensors in the order of the memory fix (checked on one card, [CHANGELOG](../CHANGELOG.md)). With a tier map it reads the v2 pack format (`nvfp4-8r-mxfp4-v2-id-order`: one file per layer and rank, all 384 records in expert-id order) and takes its tiers from the map. `VLLM_NVFP4_8R_STATS=1` switches on the route counters (next row) |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_tiers.py` | the three-tier planner (pure numpy): hot, warm and host tiers per stage from the tally and the HBM budget; logs counts of experts and shares of routes only. The tier-map reader: parses a tier map whole (format, layers, ascending ids, stated counts, a sha256 digest per stage), serves `hot_iq` only, and refuses to load when the stage's record budget cannot hold the map's hot experts or the digest differs |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_stats.py` | decode route counters per layer and tensor-parallel rank, off unless `VLLM_NVFP4_8R_STATS=1`: device tensors updated inside the decode CUDA graphs, snapshots written beside the tally on a label change. The route shares of the `hot_iq` measurements come from them. A tier-map profile refuses the switch (it has no tally directory to write to) |

## Changed files (`patches/`)

Each patch is a unified diff against the Volta sPLIt engine file as deployed on 2026-11-09. Its header gives the
sha256 of that base file and of the result; the assembler checked that applying the patch to the base reproduces this
build's file exactly. Apply in the engine source root with `patch -p1`.

| Patch | What changes |
|---|---|
| `patches/gguf_loader.patch` | admits `nvfp4_8r_packs`, either `nvfp4_8r_tally` or `nvfp4_8r_tier_map` (never both, the tier map with `hot_iq` only, and its log line names the map), `nvfp4_8r_reserve_bytes` (a map from a stage's first layer to bytes), `nvfp4_8r_residency`, `nvfp4_8r_host_route_share` and `nvfp4_8r_prefill_warm` (`native_staged` in this snapshot), parsed once and all required; drops routed-expert tensors from the GGUF weight map before any read; after load, each stage measures free HBM, subtracts its reserve and attaches the native expert layers (two-tier or three-tier) on each worker's own card |
| `patches/gguf_moe.patch` | the GGUF MoE method hands routed experts to the native path: 1 to 6 rows through the graph path, other row counts through the prompt path |
| `patches/deepseek_v41_engram_io.patch` | a reader for MXFP4 Engram rows (144 bytes) next to IQ4_NL; the table format follows the file given in `engram_path` (`.json` manifest for MXFP4, `.gguf` for IQ4_NL) |

## Configs (`configs/`)

| File | What it is |
|---|---|
| `configs/run_serve.sh` | launcher: refuses if a server already listens or a GPU has a compute process, sources `env.sh`, sets Volta sPLIt's production switch set, checks the profile's `max-num-seqs` against `NVFP4_SLOTS`, starts the server in its own process group |
| `configs/env.sh` | environment and import check |
| `configs/serve-nvfp4-pp2.yaml` | two-tier profile |
| `configs/serve-nvfp4-pp2-t3.yaml` | three-tier profile (`hot_iq_host`, host route share 0.0) |
| `configs/hot_iq.yaml.template` | the build of this release: the profile of the Hugging Face package (`hot_iq` on the mixed tally, one slot, reserves 4.5 / 7.5 GiB, tiers from the package's `tier_map.json`), with `${PKG}` for the package path. The package carries the same file as `serve/hot_iq.yaml.template` |
| `configs/serve_hot_iq.sh` | its launcher (CPU check passed 2026-11-10, eight-card boot check passed 2026-11-11): `serve_hot_iq.sh PKG [PORT]` writes `serve-hot_iq.yaml` from the package's template, sets the load-time allocator setting the `hot_iq` measurements ran with (`VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0`) and starts `run_serve.sh` with one prompt slot |
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
Volta sPLIt builds; build them from its source.
