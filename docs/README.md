# Documents

| File | What it is |
|---|---|
| [measurements.md](measurements.md) | every published number, with how it was measured |
| [plans/README.md](plans/README.md) | the build-and-test plans and their shared rules |
| [plans/00-WHERE-TIME-GOES.md](plans/00-WHERE-TIME-GOES.md) | time and memory budget of a decode round |
| [plans/01-three-tier-residency.md](plans/01-three-tier-residency.md) | the three-tier mode (built) |
| [plans/02-reserve-and-sizing.md](plans/02-reserve-and-sizing.md) | card reserve and expert budget |
| [plans/03-dispatch-overhead.md](plans/03-dispatch-overhead.md) | the cost of the split and its index math |
| [plans/04-prefill-staging.md](plans/04-prefill-staging.md) | prompt reading: staging and IQ prefill for the warm tier |
| [plans/05-miss-cache-all-native.md](plans/05-miss-cache-all-native.md) | the all-native track |
| [research/LANE-RESEARCH.md](research/LANE-RESEARCH.md) | what this build measured: where the time goes |

The ranked campaign and the other research notes follow in the next update.

The plans and research files are public copies of the build records. Each starts with a note on what was replaced.
They were written as working records, so they cite private receipts (`logs/...`), private
notes and the earlier build's records; those are not published. Their measured numbers are collected in
[measurements.md](measurements.md).

Where the plans and research cite `research/kernels/` or `research/volta/`, they mean private copies of public
sources. Read the originals instead:

- [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny): the QPN kernels, their prepack and the Volta
  bandwidth results.
- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM): the SM70 kernels and pull requests (#742, #1277, #1282,
  #1387 and the others cited).
- [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4): the
  production patches (the `v41k`, index and prefill kernels), the switch set and the harness.
- NVIDIA's Volta architecture whitepaper and the CUDA PTX ISA, for the sm_70 instruction limits.
