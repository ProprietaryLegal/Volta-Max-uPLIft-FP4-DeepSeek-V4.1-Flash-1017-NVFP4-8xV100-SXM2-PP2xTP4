# Window px rule: one log line per request on the prompt-cache decision

Public copy of the rule written on 2026-11-12 (before any GPU data), its amendment written before any GPU data, and
the result of the first run with the rule for the rerun, written after that run and before the rerun's data. The
private file and this copy differ in names and paths (private host, agent, folder and receipt names became
descriptions) and in two notices to the maintainer, left out. Nothing else changed. The second run never started,
because a request was running on the server. The third run, under the rerun rule below, was accepted. Results: the
[README](../../README.md#update-2026-11-13-the-prompt-cache-survives-a-changed-reply) and
[docs/measurements.md](../measurements.md#one-log-line-per-request-window-px). The engine change is
[engine/patches/v41_prefix_log.patch](../../engine/patches/v41_prefix_log.patch).

Subject: engine switch `VLLM_V41_PREFIX_LOG=1`. Unset or 0 = off. The line carries numbers and a reason word only. No
text, no token ids, no hash of content.

Before the window (CPU only, default serving): the change's own check and its install check pass against the real
installed engine bytes. A refusal there ends the job, no window.

Window (default stopped by the host's launcher, stop and restore lines written): apply, boot PXoff (switch unset) and
PXon (switch 1) on the test port, same profile as the default, two slots, run the public 31 requests of the
prompt-reading chunk-size window on each, stop, then decide.

INSTALL (code stays in the engine, the default exports the switch) only if every line holds:
1. Xid count unchanged from start to end and the kernel log readable. Otherwise FAULT: nothing stopped, started or
   rolled back.
2. PXoff output token-identical on 31 of 31 to the unchanged engine's receipt of the same request program.
3. PXon output token-identical on 31 of 31 to PXoff.
4. PXon's log has exactly one decision line per request (31), PXoff's log has none.
5. Every decision line matches the numbers-only pattern (field names, integers, one reason word from the change's
   fixed list, the request id). Any other character class in a line = REJECT.
6. For each request the cached-token figure in the line equals the `cached` value the request program's receipt recorded from the
   usage block (external to the new code), and for the six requests with a known arithmetic answer it equals that
   answer: fresh requests 0, the three follow-ups 24,640, the long follow-up 216,064.
7. Card memory: both arms at or under 31,998 MiB, and PXon not more than 32 MiB above PXoff on any card.
8. Speed, same window: PXon 24K read (mean of three) not more than 5 percent above PXoff, and PXon pair decode
   median at least 0.95 x PXoff. If PXoff itself is a slow boot (24K read mean above 12.0 s, the fast boots of
   2026-11-12 read it in 10.1 to 10.2 s), speed is INCONCLUSIVE: the code may stay with the switch OFF and the
   default unchanged, and that is reported. INCONCLUSIVE is not a pass.

Any failure of 2 to 7: REJECT, byte-exact rollback, default booted, nothing installed. Rollback not PASS: nothing is
started, the maintainer decides.

## Amendment (still before any GPU data), after reading the change

The change's own window is better than the one sketched above, and it is adopted: the OFF arm is the UNPATCHED
engine, the ON arm is the patched engine with the switch on (the state the default would run), and its own request program
(ten public requests) carries answers worked out from the code's rules beforehand. Its acceptance file governs the
window verdict. The lines above map as follows:
- Lines 1, 3, 4, 6, 7 (limit): covered by its sections 2 and 4 (a) to (e).
- Line 2 falls away: the OFF arm is unpatched, so there is nothing of the patch in it to compare.
- Line 8 (speed in the window) is WITHDRAWN: an A/A run of the same day showed that two boots of the same build can
  differ by 40 to 55 percent in reading time, so a window speed bar would decide nothing.

Added on top of an ACCEPT before the switch goes onto the default (all three must hold, otherwise the patch is rolled
back and the default stays as it is):
A. Numbers only. Every line in the ON arm's line file matches, after the logger prefix, `VLLM_V41_PREFIX_LOG request`,
   the request id (`[A-Za-z0-9_-]+`), then only pairs of a lower-case field name and an integer, plus
   `decision hit|cold` and `reason` with one of the 13 codes. A regex of the maintainer's own, not the change's parser.
B. Cost, measured on CPU with the real changed file (no boot involved): one admission of a 500,000-token prompt
   against two 500,000-token slot records and four history records takes under 100 ms with the switch on. With the
   switch off the added statements take under 1 ms per request.
C. Card memory in the window: ON not more than 32 MiB above OFF on any card (from the two sampler files).

Install step if all hold: apply, `export VLLM_V41_PREFIX_LOG=1` in the launcher for the default, boot, then the boot
probe (first token of the 22K read under 11 s, else report a slow boot) and one decision line per probe request in
the default's log.

## First run: result and what follows

Result: VERDICT REJECT. The patch was rolled back byte-exact and the default serves again. Nothing installed. The
REJECT stands for the first run and is not re-labelled.
- Held: identity 10 of 10, one line per request (10) on the ON arm and none on the OFF arm, "active" exactly once,
  memory PASS on both arms, Xid 0, check A (10 of 10 lines numbers-only) and check C (ON at or below OFF on every card).
- Failed: 4 of the expectation checks, all the same one: "processed within [previous prompt, previous prompt +
  output]". Logged processed against previous prompt + 64: 6,066 against 6,064, 7,129 against 7,128, 5,729 against
  5,728, 60,067 against 60,064. One more (5,063 against 5,064) passed.
- Reading (HYPOTHESIS until shown from the code): the line prints the record's `processed` as the engine stored it
  (`request.num_computed_tokens` at release). On the real engine the drafter's verified rows can carry the count 1 to
  3 past the last kept output token. The CPU scenario drove fixed accept counts that never overshoot. If so the
  expectation's upper bound was wrong and the line is right.

Rule for the rerun, fixed before its data:
1. The change's author must show FROM THE CODE (file and line) what the largest possible
   `processed - (prompt + output)` is at release, and write that bound into the acceptance file and the request program before
   the run. A bound fitted to the first run's four numbers without the code path is not accepted. If the code shows
   the count should NOT pass prompt + output, the line or the engine has a defect and that is reported instead.
2. The rerun is the install sequence, to spare a second full outage: stop the default, apply, boot the ON arm
   (patched, switch 1) on the test port, the same ten-request program, stop the arm. Identity is judged against the
   first run's OFF receipt (unpatched engine, output is boot-independent: 31 of 31 on the same day's A/A).
3. ACCEPT of the rerun needs: identity 10 of 10 against that receipt, 10 lines, each matching every expectation
   including the corrected bound, "active" once, memory PASS and no card above the OFF arm + 32 MiB, Xid unchanged,
   check A.
4. On ACCEPT the patch STAYS and the default boots with `VLLM_V41_PREFIX_LOG=1` (one export line in the launcher,
   backed up), then the boot probe and one decision line per probe request in the default's log. On anything else:
   byte-exact rollback, default booted unchanged, and the line is closed as not installed until the maintainer says
   otherwise.
