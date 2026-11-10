# Credits

This build is PLI Labs' work: the exact NVFP4 to MXFP4 pack conversion and its checks, the native expert path, the
two-tier and three-tier residency, the per-stage reserve, the harness scripts and the measurements. It runs on the
engine of our earlier build, [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4),
and stands on the projects below. Each is credited where its code or ideas are used. If you build on this repository,
please credit them too. If we have missed or misdescribed anyone's contribution, please open an issue and we will
correct it. Pull request authors were checked on GitHub on 2026-11-09.

## Model and weights

- **DeepSeek** ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  MIT): the model, its official MXFP4 routed experts (the precision this build serves), the DSpark speculative drafter
  (served as released, FP8), the config and tokenizer, and the Engram design.
- **NVIDIA** ([nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4),
  MIT): the NVFP4 pack whose routed experts this build folds to MXFP4 and serves, revision `480550d3`; its MXFP4 Engram
  tables, which this build can also read; and the accuracy table quoted in the README.
- **ISTA-DASLab** ([DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
  MIT): the GSQ-RCO quantization. This build takes the non-expert tensors and the IQ4_NL Engram from it, as Volta sPLIt
  does, and its IQ routed-expert records form the three-tier mode's warm tier.
- **ggml** ([ggml-org/ggml](https://github.com/ggml-org/ggml), MIT): the GGUF container and the IQ quantization formats.

## Engine and kernels

- **1CatAI** ([1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), Apache-2.0): the SM70 vLLM fork.
  - **yangzhuxinyzx**: [#1277](https://github.com/1CatAI/1Cat-vLLM/pull/1277), the DeepSeek-V4.1-Flash package for 4x
    V100 that Volta sPLIt ports to two boards. Its loader, GGUF expert method and Engram reader are the files this build
    changes.
  - **Peuqui**: [#742](https://github.com/1CatAI/1Cat-vLLM/pull/742), the `sm70_skinny` QPN MoE backend for NVFP4 and
    MXFP4 (`skinny_moe_qpn_sm70`, `qpn_prepack`, the E8M0 rebase), which the native expert path calls; and
    [#1282](https://github.com/1CatAI/1Cat-vLLM/pull/1282), opt-in tensor-core Triton prefill routes, a candidate in
    our campaign (not used yet).
  - Every author Volta sPLIt credits for its patch series: see its
    [CREDITS.md](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4/blob/main/CREDITS.md).
  - [#1387](https://github.com/1CatAI/1Cat-vLLM/pull/1387) is ours (the grouped prefill GEMM for IQ routed experts); the
    warm tier's prompt-reading work builds on it.
- **Dennis Vertzyas** ([dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny), MIT kernels and docs,
  Apache-2.0 fork patches): the QPN W4A16 kernels and their nibble-interleaved prepack, which reach 1Cat-vLLM through
  #742, and the Volta bandwidth findings (the ~879 GB/s read ceiling). No v100-skinny source file is copied into this
  repository; the engine carries the kernels as packaged by 1Cat-vLLM.
- **vLLM** ([vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0): the serving engine underneath.

## Our earlier build

- **Volta sPLIt** (PLI Labs, Apache-2.0): the PP2 x TP4 port, the production switch set, the IQ expert op used by the
  warm tier, the A/B harness scripts (`warmup.py`, `ab_t1_v2.py`) that the measurement scripts call, and the measuring
  method. Its credits (DeepSeek, ISTA-DASLab, 1CatAI and the authors of its patch series, vLLM, Phil Tompkins,
  TensorFold, sudoingX, Dennis Vertzyas) apply to everything this build runs on.
- **NVIDIA** compute-sanitizer and the CUDA toolkit, used throughout.
