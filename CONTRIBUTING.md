# Contributing

Bug reports, measurements on other V100 hosts, corrections to our numbers or credits, and pull requests are all welcome.

## Reporting a bug

Open an issue on [Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4](https://github.com/ProprietaryLegal/Volta-Max-uPLIft-FP4-DeepSeek-V4.1-Flash-1017-NVFP4-8xV100-SXM2-PP2xTP4/issues). Include the residency mode
(`two_tier`, `hot_iq_host` or `hot_iq`), the per-stage reserves in your profile, the loader lines that report the tier
counts and free memory after load, the engine log lines around the failure, and your GPU, clock and power settings.
Please remove addresses, host names and anything from your prompts that you would not post publicly.

Problems in the underlying engine belong upstream: [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), or our
earlier build, [Volta sPLIt](https://github.com/ProprietaryLegal/Volta-sPLIt-DeepSeek-V4.1-Flash-1017-8xV100-SXM2-PP2xTP4).

## Reporting a measurement

Report ms per round and tok/s the way [docs/measurements.md](docs/measurements.md) describes (T=0, prompt sizes, two runs
in one boot), the GPU clock cap and power limit, and what else ran on the machine. Token counts and timings only, never
prompt or generated text, and never a routing tally or per-expert list from your own traffic.

A number goes into the README or `docs/` only if it is MEASURED, with its date, configuration and conditions, or labelled
ESTIMATED with the arithmetic shown.

## Pull requests

- One change per pull request, with the reason in the description.
- A change that affects serving needs a before-and-after on V100 hardware (bit equality against the reference for the
  expert path at least), or a clear statement that it has none yet; we can run it on ours.
- Keep the engine's licence and headers on any file you modify; add new sources to [CREDITS.md](CREDITS.md) and [NOTICE](NOTICE).
- Do not add prompts, outputs, logs, tallies or caches from real use.

By contributing you agree that your contribution is licensed under Apache-2.0, like the rest of the original code.
