# Closeness test: run decisions (scrubbed public copy)

This is a scrubbed public copy of the closeness test's run decisions. It is not one of the preregistered files, and
its bytes are not in [PREREGISTRATION.txt](PREREGISTRATION.txt). Hosts, paths, ports, the key id and internal names
are left out. Every decision is given in full.

Decisions 1 to 7 were fixed on 2026-11-10 at about 20:05 UTC, after the rules and the script hashes were
preregistered (19:54 UTC) and before any capture. They choose among options the design left open. None changes a
scoring rule, a constant or a script.

1. **Multiple-choice set.** MMLU, 513 questions, as designed.
2. **Sample.** 320 contexts and 513 questions, as designed.
3. **Thinking on.** Not run. Every capture is thinking off, and the results say so as a limit of the test.
4. **Capture budgets.** Reference 55 minutes, not the 45 of the design, because the upper estimate from a dry run
   was 53.6 minutes and a stop at the budget would cost the reference repeat (the noise floor), which runs last.
   `hot_iq` 30 minutes. Volta sPLIt 15 minutes. No continuation on a fresh boot is planned. An arm that stops at its
   budget is scored on what it holds.
5. **Key.** The key that hashes the token ids in the receipts is deleted after the final score is written and
   checked. The receipts are published. The scorer still checks them by the key's id.
6. **Route share on the new contexts.** Not measured. The results use the in-sample shares (79 to 85 % of decode
   routes, [README](../../README.md#how-close-to-nvfp4)) and say so.
7. **Earlier receipts.** Receipts from runs before this test that hold raw generated token ids stay local and are
   not published.

Decisions 8 and 9 are two amendments, each made during a capture and before any score.

Decision 8 was made on 2026-11-10 at 21:36 UTC, during the first capture and before any score.

8. **Volta sPLIt continuation.** The Volta sPLIt arm gets one continuation part on the same production boot, with a
   75 minute budget for that part. Part 1 keeps its 15 minute budget and stops there. The reason: while it waits for
   the production server to be idle, the capture program pauses 5 s between its two idle checks before every
   request, so the arm ran at 5.74 s per item (first progress line, 20 of 833 items, no waits for other traffic),
   not the 0.7 s the 10 minute estimate assumed. The 15 minutes only limited how long the test shares the production
   server, and the capture yields to real use before every request. What was seen when this was decided: that
   timing line only. No reference or `hot_iq` capture existed yet, so no comparison could be computed. No script,
   rule or constant changed. Decision 4 stands for the other two arms: no continuation on a fresh boot.

Decision 9 was made on 2026-11-11 at 02:05 UTC, during the `hot_iq` capture and before any score.

9. **`hot_iq` continuation.** The `hot_iq` arm gets one continuation part on the same boot of the package profile,
   with a 20 minute budget for that part. Part 1 keeps its 30 minute budget and stops there. The reason: the arm ran
   at 2.98 s per item (progress line at 100 of 833 items, projected 2,478 s against the 1,800 s budget), not the
   1.6 s the 22 minute estimate assumed. What was seen when this was decided: the timing lines only. The reference
   capture had not started, so no comparison could be computed. No script, rule or constant changed. The capture
   run was paused, not changed, between part 1 and the switch to the reference, so that part 2 ran on the same
   boot. Decision 4 stands: no continuation on a fresh boot. If the reference stopped at its 55 minute budget, it
   would be scored on what it holds. It did not stop: it completed.

A continuation part re-captures the first 10 contexts, and the scorer refuses it unless they are identical to part 1
([README](README.md)).
