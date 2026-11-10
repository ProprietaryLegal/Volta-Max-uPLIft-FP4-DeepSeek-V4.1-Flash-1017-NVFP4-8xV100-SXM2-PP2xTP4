# Changelog

Dates are UTC. Every number is MEASURED on the 8-card host unless labelled otherwise; details in
[docs/measurements.md](docs/measurements.md).

## 2026-11-10: load setting on the accepted tiers, part B on two more chats and on twelve prompts

Measured after the entry below, the same day. One request at a time, public text, temperature 0.

- **The accepted tiers under the load setting.** A profile that pins the accepted tiers with the load-time
  allocator setting off booted with the same hot, warm and host experts and the same tier digests. Chat 0 of the
  termination test ended normally with 6,417 tokens and gave the token counts of the two earlier runs on all eight
  requests (first read 167.64 s). On that chat the setting did not change the token counts. The profile is not in
  this snapshot, and the launcher is unchanged.
- **Part B on chats 1 and 2** (plan 04 part B, reserves 5.3 / 8.1 GiB, not accepted, quality not established): both
  final answers ended normally (749 and 1,491 tokens). First reads 102.12 s and 103.0 s against 155.56 s and
  161.39 s, and a median wait of 4.42 s and 4.26 s before a follow-up's first token. With chat 0 the variant stands
  at 1 of 3 at the limit (95 % Wilson interval 0.061 to 0.792).
- **Correction to the entry below.** It says every configuration with a lower share ran chat 0 to the limit,
  "five boots, each on chat 0 only". The lower share and the limit go together on chat 0 only. One of the five
  configurations, the part B variant, has now ended chats 1 and 2 normally. The other four have run chat 0 and no
  other chat.
- **Part B on twelve prompts** (not accepted, quality not established): INCONCLUSIVE by the rule. First different
  token 10 for, 2 against, and mean difference 5 for, 1 against, 6 undecided.
- **Part B at the accepted tiers** (not accepted, quality not established): the variant ran out of card memory on
  stage 0 in the second chunk of its first 216,245-token read (a request for 108 MiB with 64.75 MiB free).
  INFERENCE from the code, nothing run: both reading modes peak at the same point and the variant is short by the
  workspace it keeps from load. A fix is planned, not built.
- **Open:** first, the part B variant at the accepted tiers after the planned fix for its memory shortfall. Then
  the replacement prompts of the 12-prompt closeness set (INCONCLUSIVE until then).

## 2026-11-10: termination test, twelve-prompt closeness check, part B boots

Measured after the first release, the same day. One request at a time, public text, temperature 0.

- **Termination test for long thinking answers.** It exists and has run. In the accepted configuration (host tier,
  one slot, reserves 4.5 / 7.5 GiB) 3 of 3 chats of 198,307 to 216,245 tokens ended normally (6,417, 1,499 and 1,642
  tokens), as on Volta sPLIt (10,013, 732 and 1,812). Three chats bound the share that runs to the limit only below
  0.56 (95 % Wilson interval 0.000 to 0.561). Chat 0 gave the token counts of the 2026-11-09 run again.
- **Chat 0 on the other configurations.** Every configuration of this build with a lower share of routes on
  official-precision experts ran chat 0's final answer to the 32,000-token limit: five boots, each on chat 0 only,
  none accepted, quality not established. The cause is not established. Volta sPLIt itself ran to the limit in a
  repetition loop on a separate 31,173-token chat. This replaces the count in the entry below ("four of five runs
  of this build, two of three in the accepted mode"): the two later runs it calls accepted-mode runs had other
  reserves or three slots, and the accepted configuration has ended chat 0 in both of its runs.
- **Closeness on a wider set.** The 12-prompt closeness check is INCONCLUSIVE, with the rule (8 of 12 on each
  part) fixed before the run. Accepted configuration: first different token 11 for, 1 against, and mean difference
  7 for, 0 against, 5 undecided, because Volta sPLIt differs from the reference at the first token on five prompts.
  Three slots on the accepted tiers: 10 for, 2 against, and 7 for, 0 against, 5 undecided. They stay not accepted
  (quality not established).
- **IQ records in prompt reading** (plan 04 part B, not accepted, quality not established): the load failure is
  explained and fixed (the engine's allocator setting during weight loading, reproduced on one card, and with the
  setting off the variant boots). On that boot the first read of 216,245 tokens took 107.82 s against 164.9 s, and
  a follow-up waited 4.44 s (median) for its first token against 8.76 to 9.43 s. Its long-thinking request ran to
  the 32,000-token limit. Not in this snapshot. The launcher in this snapshot is unchanged.
- **Open:** first, the part B variant at the accepted reserves, with the termination test and the closeness
  capture. Then replacement of the five undecided prompts of the 12-prompt closeness set (INCONCLUSIVE until then).

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
