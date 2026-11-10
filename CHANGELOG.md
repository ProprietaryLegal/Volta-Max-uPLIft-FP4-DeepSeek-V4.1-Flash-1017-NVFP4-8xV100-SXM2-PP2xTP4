# Changelog

Dates are UTC. Every number is MEASURED on the 8-card host unless labelled otherwise; details in
[docs/measurements.md](docs/measurements.md).

## 2026-11-10: first public release (work in progress)

- **Long thinking answers.** On the 216K tool loop the final request ran to the 32,000-token limit in thinking in both
  later accepted-mode runs (one slot with other reserves, traced; three slots). With the earlier runs that is four of
  five runs of this build, two of three in the accepted mode; Volta sPLIt ended normally (10,013 tokens). The cause
  is not established; the test keeps no output text. The accepted mode's acceptance rests on the three-prompt
  closeness check.
- **Three prompt slots on the accepted tiers** (not accepted: quality not established; closeness INCONCLUSIVE by
  the rule). Speeds, not accepted (quality not established): general 50.9 / 51.3 tok/s and legal-style 63.9 / 64.2
  tok/s on short prompts, and a 257,741-token prompt read in 207.1 s (quality not established). Three chats taking
  turns kept 6 of 6 follow-ups. Highest card memory 31,704 MiB.
- **Prompt reading trace** (accepted mode, one slot, traced): host staging is 86 % of the expert time in prompt
  reading (0.1485 of 0.1731 s per layer call, 162.5 experts staged per call).
- **IQ records in prompt reading** (plan 04 part B): built and passed its one-card checks; its first eight-card boot
  failed while loading (driver error "invalid argument" during the build of the IQ bank); cause not established; not
  in this snapshot.
- **Open:** first, a termination test for long thinking answers; then a larger prompt set for the closeness check,
  finding the load fault of the IQ-records mode, and the quality of the no-host-tier profiles. The ranked campaign
  and the research notes follow in the next update.

## 2026-11-09: measurements of the first day

- **Packs.** NVIDIA's NVFP4 routed experts folded to MXFP4 packs per layer and TP rank, 299,473,305,600 bytes;
  44,696,862,720 values checked bit-equal to the NVFP4 formula in FP32.
- **Native expert path.** The v100-skinny QPN kernel (through 1Cat-vLLM #742) runs the experts as W4A16 with CUDA graphs
  for 1 to 6 rows; worst relative L2 4.78e-4 against an FP32 reference on one card.
- **Two-tier residency.** Hot experts on the cards, the rest zero-copy from pinned host RAM, chosen per stage at load
  from measured free memory and a private routing tally. Fixes found by the first 8-card boots: the host mapping now
  uses each worker's own card; prompt staging buffers are one pair per card; the card reserve is set per stage
  (4.5 GiB on stage 0, 7.5 GiB on stage 1) from peaks sampled every 15 s.
- **Two-tier speed.** 28.2 to 30.2 tok/s legal-style and 18.8 to 20.1 general; 30.2 to 33.7 tok/s from 46 to 303,004
  tokens of context; prompt reading 1,419 to 1,595 tok/s (30K to 303K tokens); a 498,099-token prompt read without error.
- **Three-tier residency.** A warm tier runs less-used experts from the IQ2-class records Volta sPLIt uses. 56.9 to 58.4
  tok/s legal-style, 50.6 to 51.8 general, 60.9 to 75.9 tok/s from 46 to 257,741 tokens of context; prompt reading 1,080
  to 1,327 tok/s. The warm tier carries 3.74 % and 9.29 % of decode routes by the tally.
- **MXFP4 Engram reader** for NVIDIA's Engram tables (built and checked; not the default).
- **Measured but not adopted:** copy-engine DMA, GPU gather and AVX2 host compute for host-RAM experts. None comes
  within about 10 ms per round of all experts on the cards (ESTIMATED from single-card measurements).
- **Quality.** Three-tier against two-tier on three public prompts (greedy, thinking off): PASS on the rule written
  before the run; it diverges later than Volta sPLIt on two prompts and has the smaller mean log-probability
  difference on two.
- **Agent-style tool loop** (216K-token prompt, thinking on): all seven three-tier follow-ups kept
  the whole previous prompt; 12.5 to 15.0 s per tool-call request against about 4.6 s on Volta sPLIt; generation
  47.0 to 66.1 tok/s.
- **Three prompt slots** work (measured on the no-host-tier tiers): three chats taking turns, 6 of 6 follow-ups kept
  their own chat's prompt, with no decode slowdown.
- **No host tier (`hot_iq`)**: NOT accepted, quality not established: 86 to 87 tok/s general and 67 to 72 legal-style
  on short prompts. The closeness check is inconclusive (three-slot tiers). In the 216K tool loop its final thinking
  answer ran to the 32,000-token limit in both runs. With the two later accepted-mode runs (2026-11-10, above) that
  is four of five runs of this build, two of three in the accepted mode. The cause is not established. Volta sPLIt's
  final request also ran to that limit in one of three runs with the 32,000-token limit of a shorter loop. The
  accepted mode is the host-tier three-tier profile.
- **Turn boundary on Volta sPLIt** (31,164-token chat, thinking on, temperature 0): earlier thinking sent back
  unchanged kept 38,800 of 38,845 tokens (first token 0.54 s); stripped, 0 of 38,193 (14.07 s); through our router
  after its change, 69,728 of 69,772 (1.45 s).
