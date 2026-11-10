# Engine

This build runs on the engine of [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4)
(1Cat-vLLM #1277 with Volta sPLIt's patch series and production patches). It adds two files and changes three.

## New files (`src/`)

| File | What it does |
|---|---|
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_native.py` | reads the MXFP4 packs, prepacks experts for the QPN kernel (into HBM, or into a page-aligned pinned host arena registered with CUDA), chooses residency per stage from free HBM and the tally, and runs the routed experts: one grouped QPN launch per projection and bank, V4.1's limited SwiGLU and the weighted sum in FP32, CUDA-graph safe for 1 to 6 rows; prompt chunks stage host experts into two shared HBM buffers per card. `VLLM_NVFP4_8R_LOAD_TRACE=1` logs card and allocator memory during the stage build (counts and bytes only). It was added after the measurements and is off by default |
| `src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_tiers.py` | the three-tier planner (pure numpy): hot, warm and host tiers per stage from the tally and the HBM budget; logs counts of experts and shares of routes only |

## Changed files (`patches/`)

Each patch is a unified diff against the Volta sPLIt engine file as deployed on 2026-11-09. Its header gives the
sha256 of that base file and of the result; the assembler checked that applying the patch to the base reproduces this
build's file exactly. Apply in the engine source root with `patch -p1`.

| Patch | What changes |
|---|---|
| `patches/gguf_loader.patch` | admits `nvfp4_8r_packs`, `nvfp4_8r_tally`, `nvfp4_8r_reserve_bytes` (a map from a stage's first layer to bytes), `nvfp4_8r_residency`, `nvfp4_8r_host_route_share` and `nvfp4_8r_prefill_warm` (`native_staged` in this snapshot), parsed once and all required; drops routed-expert tensors from the GGUF weight map before any read; after load, each stage measures free HBM, subtracts its reserve and attaches the native expert layers (two-tier or three-tier) on each worker's own card |
| `patches/gguf_moe.patch` | the GGUF MoE method hands routed experts to the native path: 1 to 6 rows through the graph path, other row counts through the prompt path |
| `patches/deepseek_v41_engram_io.patch` | a reader for MXFP4 Engram rows (144 bytes) next to IQ4_NL; the table format follows the file given in `engram_path` (`.json` manifest for MXFP4, `.gguf` for IQ4_NL) |

## Configs (`configs/`)

| File | What it is |
|---|---|
| `configs/run_serve.sh` | launcher: refuses if a server already listens or a GPU has a compute process, sources `env.sh`, sets Volta sPLIt's production switch set, checks the profile's `max-num-seqs` against `NVFP4_SLOTS`, starts the server in its own process group |
| `configs/env.sh` | environment and import check |
| `configs/serve-nvfp4-pp2.yaml` | two-tier profile |
| `configs/serve-nvfp4-pp2-t3.yaml` | three-tier profile (`hot_iq_host`, host route share 0.0) |
| `configs/serve-nvfp4-pp2-t3n.yaml` | three-tier with no host tier (`hot_iq`): not accepted (quality not established) |
| `configs/serve-nvfp4-pp2-t3-a3.yaml` | three-tier with three prompt slots (`max-num-seqs 3`, reserves 5.3 / 8.1 GiB): measured, not accepted (quality not established) |
| `configs/serve-nvfp4-pp2-t3-a1r.yaml` | the three-slot profile's one-slot control: same reserves, `max-num-seqs 1` |
| `configs/serve-nvfp4-pp2-t3n-a3.yaml` | no host tier (`hot_iq`) with three prompt slots (`max-num-seqs 3`, reserves 5.3 / 8.1 GiB): not accepted (quality not established) |

Paths are written with `$LANE` (this build's root, with `engine/`, `models/`, `packs/` and `run/` under it). The
shell scripts and Python files read it from the environment and stop when it is not set; in the YAML profiles
replace it with your path. The launcher and profiles run from `$LANE/run/` (the copy commands are in the README's
Reproduce section). The profiles name the
private tally file (`engine/private/p16-tally.npz`); point `nvfp4_8r_tally` at your own.

The compiled libraries the launcher names (`v41k.so`, `v41idx.so`, `v41pidx.so`, the sparse prefill library) are the
Volta sPLIt builds; build them from its source.
