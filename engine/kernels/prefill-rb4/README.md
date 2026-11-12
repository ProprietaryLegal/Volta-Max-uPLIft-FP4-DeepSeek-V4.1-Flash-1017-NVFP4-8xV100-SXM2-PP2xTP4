# Prefill RB 4 library: source and build

The engine key `nvfp4_8r_prefill_rb: rb4` runs a compiled library, `v41qpnrb4.so`, which the engine loads from the
folder of `engine/src/vllm/model_executor/layers/fused_moe/experts/nvfp4_8r_qpn_rb.py`. The library is not in this
repository, like the engine's other compiled libraries. This folder holds what it is built from.

- `qpn_rb4_append.cu`: our launcher, appended after the unchanged text of the engine's
  `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` (sha256 `31d6ea4f...`). That file is 1Cat-vLLM's `sm70_skinny` QPN
  MoE kernel (Apache-2.0), whose kernels come from dnv2003/v100-skinny (MIT, notice in
  [LICENSES/MIT-v100-skinny.txt](../../../LICENSES/MIT-v100-skinny.txt)). It is not copied here.
- `build_qpn_rb4.py`: joins the two files into `gen/nvfp4_8r_qpn_rb.cu` and builds it as a torch
  extension for sm_70. Its docstring gives the commands.

What the launcher changes: the installed op runs a prompt-reading launch (more than 8 rows per expert on average)
at RB 2, so each block re-reads and re-dequantizes its expert's weight tile once per 16 rows. The launcher keeps the
installed op's checks and grid, accepts only what the prompt-reading path uses (SPLITK 8, MXFP4 E8M0 scales, m8n8k4
fragments), and runs those launches at RB 4: four 8-row blocks per weight load. Every other launch runs at RB 1, as
installed. Each row keeps its accumulators and K order at any RB, so the output bits are the installed op's. The
one-card test `harness/benchmarks/nvfp4_8r/test_prefill_rb.py` checks that with `torch.equal` on layers 5, 12 and 27,
and window 17 checked it token for token on eight cards (docs/measurements.md). Decode does not use this library.

The library of the measurements was built from these two files on the same torch (2.10.0+cu128) and vLLM `_C` as the
engine. The published `qpn_rb4_append.cu` differs from that build input in one comment line only. The engine module
`nvfp4_8r_qpn_rb.py` pins that library's sha256 (`2b893e7d...`) and refuses any other file, so a library you build
loads only after you set `LIBRARY_SHA256` in that module to your file's sha256. Without the library, set
`nvfp4_8r_prefill_rb: rb2` in the profile: the same output bits, without the second-read gain.
