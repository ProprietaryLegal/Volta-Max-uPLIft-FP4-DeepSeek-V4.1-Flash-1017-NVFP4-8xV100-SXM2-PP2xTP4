# Window 11 (plan 03 Part B, the group-plan kernel): acceptance rule (scrubbed public copy)

This is a scrubbed public copy of the rule that decided whether the group-plan kernel becomes the default. The
private file was last changed on 2026-11-11 at 04:57:31 UTC, ten seconds before window 11 started (04:57:41 UTC), so
it was written before any window-11 data existed. The same file, unchanged, decided window 11c (see
[WINDOW11-replication.md](WINDOW11-replication.md)). Hosts, paths, ports, internal names and file hashes are left
out. Every rule is given in full. The window script and the scorers it names (`ab_t1_v2.py`, `pairs_arms_v2.py`,
`quality_t3_wide.py`) are part of the private harness and are not published.

## Arms

- **A (chain)**: the `hot_iq` profile on the mixed tally with the stage-1 reserve at 7.00 GiB (plan R1), with
  `nvfp4_8r_group_plan: chain`, the route plan built by the chain of tensor operations as before.
- **B (kernel)**: the same profile with `nvfp4_8r_group_plan: kernel` and two header comment lines. The window script
  makes this file and checks that this is the only difference.
- Engine: one `nvfp4_8r_native.py` and one `nvfp4_8r_group_plan.py` for all four arms (the files published in
  `engine/src/`). The source tree and the installed copy must hold the same bytes.
- Order: A1, B1, B2, A2. One arm is one boot. Each boot gets a warm-up, then the throughput run twice. The second run
  carries the label suffix `r`.

## How to measure (quoted from the lane's plans index, "How to measure", placeholders in capitals)

> 2. Throughput: `ab_t1_v2.py --tput http://127.0.0.1:PORT 3 OUT.jsonl | tee DIR/ab-LABEL-tput.out`.
>    Temperature 0, thinking off, `ignore_eos`, 1,024 tokens, three public prompts. Read
>    `dec-ms/round` (decode only: first to last streamed token) and tok/s from the `TPUT` lines.
> 3. Paired ABBA: boot or switch arms in the order A, B, B, A. Score with
>    `pairs_arms_v2.py --metric dec --from-out DIR A1 B1 A2 B2`.
> 4. Noise floor: repeat the throughput run inside one boot, and compare the two A arms. The noise
>    floor is the larger difference. A gain counts only if both pairs agree in sign and the mean
>    difference is above twice the noise floor. Otherwise report "no measurable change".
> 5. Greedy check: `ab_t1_v2.py --compare A.jsonl B.jsonl` is token-exact. Use it when a change must
>    not alter output (plans 02, 03, 04-A, 05).

Plan 03 section 8 adds: "Part B must be token-exact (`ab_t1_v2.py --compare`)". Its "Done when" clause says that a
part with no result above the noise floor "is recorded as 'no measurable change' and switched off".

## Figures (decode ms per round, from `pairs_arms_v2.py --metric dec`, its T0 lines)

- p1 = mean(B1 - A1) and p2 = mean(B2 - A2), from `pairs_arms_v2.py ... A1 B1 A2 B2` on the first throughput run of
  each boot.
- Mean difference d = (p1 + p2) / 2. A negative value means the kernel arm is faster.
- **Noise floor N.** The rule's reading of "repeat the throughput run inside one boot, and compare the two A arms. The
  noise floor is the larger difference": N is the largest absolute mean among these five.
  - The within-boot repeats A1r - A1, B1r - B1, B2r - B2 and A2r - A2.
  - The two A arms, A2 - A1.

  The plans index names one repeat. The window has four, and taking the largest is the stricter choice. The script
  prints all five, so another reading can be applied to the same receipts.

## Token-exact compares (both must hold)

1. **Chain after the kernel patch against the build before it.** Window 10a's two captures of the same profile,
   taken before the patch, and A1's two captures of the same prompts must both report SAME.
2. **Kernel against chain.** Two checks.
   - A1's and B1's two captures both report SAME.
   - `ab_t1_v2.py --compare` of B1's and B2's first throughput receipt against A1's prints `MATCH` on token ids.

A2's throughput receipt is also compared against A1's. A difference there means the chain is not repeatable across
boots, and the compares cannot decide (INCONCLUSIVE).

## Verdict

- **ACCEPT** (the kernel arm can be the default): both token-exact compares hold, p1 < 0 and p2 < 0, and -d > 2N.
- **NO MEASURABLE CHANGE** (the kernel stays off and the chain stays): both compares hold, but not all of the ACCEPT
  conditions are met, and the NOT ACCEPTED conditions are not met either. That covers a mixed sign, or |d| <= 2N.
- **NOT ACCEPTED** (the kernel is slower): both compares hold, p1 > 0 and p2 > 0, and d > 2N.
- **INCONCLUSIVE**: a receipt is missing or unreadable, or `pairs_arms_v2.py` refuses one, or A2 differs from A1
  token-wise.
- **Kernel arm refused**: a kernel-against-chain compare differs. The script stops there and gives no speed verdict.

The script prints a `VERDICT` line computed by this rule from the printed figures. The receipts and this rule decide.
The line is a convenience.
