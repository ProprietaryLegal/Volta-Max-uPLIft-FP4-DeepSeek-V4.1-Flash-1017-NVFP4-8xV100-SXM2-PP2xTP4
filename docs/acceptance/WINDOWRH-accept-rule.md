# Window rh rule: the snapshot hit (first named the replay hit)

Public copy of the rule written on 2026-11-13 before the change existed, with its six amendments, each written before
the data it governs (Amendment 6 is about the finish step after the accepted run). The private file and this copy
differ in names and paths (private host, agent, folder, script and receipt names became descriptions) and in one part
left out: the conduct section's host operating steps (which other work is paused, where stop lines are written, who
is told). Nothing else changed. Results: the
[README](../../README.md#update-2026-11-13-the-prompt-cache-survives-a-changed-reply) and
[docs/measurements.md](../measurements.md#snapshot-hit-window-rh). The engine change is
[engine/patches/v41_snapshot_hit.patch](../../engine/patches/v41_snapshot_hit.patch).

The change's own acceptance file may add detail but may not loosen any line here. Verdict words: ACCEPT, REJECT,
INCONCLUSIVE. Only ACCEPT installs. REJECT and INCONCLUSIVE roll back byte-exact and the default returns as it was.

## What the change must do

Today a request gets its cached prefix only when the block pool's hit P lies within 124 tokens of the slot's processed
position. One token that differs further back (for example the model's previous reply not coming back token for token)
makes the server read the whole prompt again (observed 2026-11-12: prompt 218,510, pool hit 212,720, processed
218,437, reason mismatch_beyond_rewind, hit 0).

After the change, when a free slot's record is the same conversation (the request's tokens contain the whole prompt
that record was prefilled with) and the pool hit P is behind that slot's reach, the request is admitted with a
shorter hit P' = P minus R rows (aligned as the engine requires), and the R rows are read again so that the state the
rings no longer hold is rebuilt. R is a constant derived from the code (the same cone argument as
VLLM_V41_PREFILL_SKIP_ROWS) and fixed in the change's design file before the window. Everything else decides as today.

## Scope limits (fixed)

- A request that does not contain the record's whole prompt is NOT given a replay hit (it would take a slot from
  another conversation). It stays cold and gets its own reason word.
- No new device memory. No change to the slot choice of cold requests. No change to an ordinary hit.
- Behind a switch (unset or 0 = today's behaviour, byte-for-byte decisions). The window tests the switch on.

## Arms and cases (install-shaped, switch-on arm only, test port, eight cards, two slots, the default's profile)

Greedy, fixed max_tokens, public text only. Every case k uses its own cache salt for the case and a different salt
for its cold reference, so the reference can never hit.

- I. Identity (nothing else changes): the request sequence of window px's accepted run, replayed. Every request's
  tokens equal the stored receipt of that run, and every log line's decision, reason, hit and slot equal that
  receipt's line.
- C. Replay cases, at least 8: prompt X read cold with a long reply G (more than 124 tokens past the prompt), then
  prompt2 = X + G[:d] + Y. Divergence depths processed minus (len(X)+d): about 200, 1,000 and 6,000. Lengths of X:
  about 8K, 40K and at least 100K (one case). At least one case with Y longer than 2 R. At least one case where d = 0
  (the reply differs from its first token). At least one case sent as a real chat (two turns through
  /v1/chat/completions or /v1/messages, the assistant turn returned altered), as a chat program would send it.
- F. Follow-on, at least 2 of the C cases: one more ordinary turn after the replay hit (prompt2 + its reply + Z).
- N. Negative, at least 2: a request that shares only a leading part of a record's prompt (shorter than that prompt)
  while that slot is free. And one request whose replay slot is busy.
- K. Control, at least 2: an ordinary within-reach hit (short reply, then the next turn), with its own cold reference.

## Checks (all decided from the receipts and the server log, numbers and token ids only)

- R1 decision. Every C case logs decision hit with the new replay reason, in the slot that held X, and
  hit = the P' the design file's formula gives for that case (computed by the checker from the case's own numbers,
  not read from the log). Tokens read again = prompt minus hit <= (prompt minus first differing index) + R + 2 grains.
- R2 exact output. Every C case's output token ids equal its cold reference's, all positions. Every F turn logs an
  ordinary hit and its tokens equal its cold reference's.
- R3 identity. Block I holds for every request.
- R4 negatives. The partial-prompt request logs cold with the new reason word, takes the slot a cold request would
  take today, and its tokens equal its reference. The busy-slot request logs slot_busy as today.
- R5 control. Every K case's tokens equal its cold reference's. If a K case differs, the comparison "hit against
  cold" is not a valid yardstick on this engine: the verdict is INCONCLUSIVE, whatever R2 shows.
- R6 time. For the case with X of at least 100K: time to first token of prompt2 <= 0.25 x its cold reference's.
- R7 memory and health. Peak memory per card is no higher than the installed receipt's bound (window px's memory
  check). Xid 0 before and after. Boot probe as usual. Switch-off decisions equal today's on the CPU checker (all
  scenarios).
- R8 abort safety (CPU checker, real scheduler code): a request freed inside its replay rows leaves no record that a
  later ordinary hit may use.

ACCEPT = R1, R2, R3, R4, R5, R6, R7, R8 all hold. REJECT = any of R1, R2, R3, R4, R6, R7, R8 fails with R5 holding.
INCONCLUSIVE = R5 fails, or a check could not be computed.

## Conduct

One window. The default is stopped only when no request runs: wait, never force. An Xid or host reset is the crash
rule: nothing started, nothing stopped, report. No second window without telling the maintainer first. No request
text, no model text and no traffic figures in any receipt that leaves the host.

## Amendment 1 (still before any data): snapshot hit replaces the replay rows

The change's author refuted the replay-rows design from the code and the serving log: layers 1-19 keep window rings
too (19 and 20 rings per stage), so the chain is 39 ring layers, and the KV source layers 2, 8, 14 and 20 would write
wrong compressed rows and index keys inside the replayed span into permanent state. No R repairs that. The change is
therefore the "snapshot hit": each rank copies its slot's ring rows to host memory right after the forward that ends
a request's prompt, and copies them back into the same ring tensors when a later request of the same conversation is
admitted at the record's prompt end. This amendment replaces the lines above that name R or P'. All other lines stand.

- Hit formula: h = floor to the grain of min(P, N_rec), where N_rec is the record's prompt length and P the pool hit.
  Admitted only if N_rec - 124 <= h, the slot is free, the request contains the record's whole prompt, and h is not
  below the position from which the snapshot's rows are exact. Otherwise today's decision.
- R1 becomes: every C case logs decision hit with the new snapshot reason, in the slot that held X, with hit = h
  computed by the checker from the case's own numbers. Tokens read again = prompt minus hit
  <= (prompt minus N_rec) + 2 grains.
- Block C: "Y longer than 2 R" becomes "Y longer than 6,144 tokens" (the new part crosses a prefill-skip boundary).
- Memory: no new device memory (R7 unchanged). New host memory at most 100 MB in all, stated and measured.
- R9 ring validity (CPU, before the window). The slot table carries, for the live rings and for the snapshot, the
  interval of positions whose ring rows are exact. Every admission with the switch on is decided from those
  intervals. A property test over random request sequences (hits, short and long replies, aborts, preemptions,
  requests ending before the previous one's end, draft overshoot up to 4) compares each admission with an
  independent 256-slot ring simulator written from the arithmetic, not from the engine code: no admitted hit may read
  a row the simulator marks stale, and no hit that today's rule admits and the simulator marks clean may be refused.
  This includes the case found in today's engine (a request that ends before the previous request's end, then a
  third hit below the first one's reach): with the switch on it is refused with its own reason word, and with the
  switch off today's decisions stand byte for byte.
- R10 worker agreement. The worker's own check of the same intervals may never fire in the window or in the CPU
  checker's scenarios (a firing is a REJECT). It exists so a disagreement stops loudly, never reads stale rows.
- R11 snapshot cost. The first snapshot's copy time is logged per rank (number only): at most 100 ms on every rank.
  Ordinary hit cases in block I show no change in decisions or tokens (R3).
- Remote drafter: the switch refuses to start with the remote drafter backend configured (it keeps 256 rows).
- ACCEPT = R1 to R11 all hold. REJECT and INCONCLUSIVE as above, with R9, R10, R11 on the REJECT list.

## Amendment 2 (before any window data, answers the builder's two questions)

A2.1 (R3). Block I compares decision, hit and slot against the stored receipt on all ten lines. The reason word is
compared on nine lines. For the edited-turn request (it shares 5,992 of the 7,928 prompt tokens of its slot's record)
the reason must be exactly `partial_prompt` with the switch on. This is one named table entry in the checker, no
pattern. With the switch off, all ten reason words equal the receipt (R7 switch-off line is unchanged).

A2.2 (R7 memory). The bound "each card's peak in window px's OFF-arm memory file + 32 MiB" applies to the block I
segment only. Block I runs FIRST after boot, and its memory reading is taken and written to its own file before any
block C request is sent (reserved memory does not shrink, so a later reading cannot be attributed). For the whole
run the bound is the house limit: no card above 31,998 MiB and no out-of-memory line in any rank log. The patch's
own claim (no device memory added) is still checked: the block I reading is the evidence for it. A whole-run reading
above the block I bound but inside the house limit is reported as a number, not a verdict.

A2.3 (record). Touched files are v1/core/kv_cache_manager.py, v1/core/sched/v41_multi_seq.py,
models/deepseek_v41/slots.py and v1/worker/gpu/model_states/deepseek_v41.py (slots.py, not model.py).

A2.2a (before any window data). "Its own file" in A2.2 is met by the one sampler file split by time: the request
program waits one full sampler interval after block I's last request and records that moment. The block I reading is
every sample up to that moment, and it is INCONCLUSIVE if no sample falls between the last request's end and that
moment.

A2.2b (before any window data). The package as resealed implements A2.2 as first written, not the A2.2a allowance:
the request program runs block I alone, the window script then waits for two sampler lines written after block I
ended and copies the sampler file as it stands to its own file before any later request is sent. The memory check
decides on that file and is INCONCLUSIVE if its last sample does not lie between block I's end and the first later
request. The block I scope and the edited-turn word `partial_prompt` are fixed. The change's own acceptance file
predates this on that point and is superseded here. The maintainer ran the change's CPU checks on the final bytes and
resealed its checksums.

## Amendment 3 (after the first window, before any data of a second)

Record of the first window: verdict INCONCLUSIVE by R5 (control k1 differed from its cold reference), patch rolled
back byte for byte, default booted unchanged, Xid 0. R1, R3, R4, R6, R7 to R11 held. R2 differed in c3 and c8, six
other cases and both F turns equal. That verdict stands. No reading of the first window counts toward a later
window's verdict.

Measured afterwards on the default (no patch, public text, greedy, four private diagnosis receipts):
- cold against cold: identical, every time.
- an ordinary hit is not token-identical to a cold read in general. On the c3 second prompt, ordinary hits at 6000,
  7984, 8000 and 8016 gave one output, hits at 7920, 8080 and 8192 gave the cold output, a hit at 7600 a third.
  Every first difference is at a token whose top-two logit gap in the cold read is small (0.125, 0.19, 0.47).
- the ordinary hit at 8000 on the c3 second prompt equals the first window's c3 snapshot output, all 64 tokens. The
  ordinary hit at 7888 on the c8 second prompt equals its c8 snapshot output, all 64 tokens.

So "equal to a cold read" is not a property of the engine, with or without the patch. R2 and R5 as written tested
the engine, not the patch. The yardstick that isolates the patch is an ordinary hit at the same position.

A3.1 (R2 reference). For every C case and the chat case, the reference is an ORDINARY hit at the case's own hit
position h, on the same second prompt (the same token ids), in the same boot. It is sent as its own chain on its own
salt: prompt X with a short reply (the record stays within reach), then prompt2. The checker reads the chain's log
line: decision hit, today's ordinary reason word, hit = the case's logged hit, and not the snapshot reason. If that
line is anything else, or the chain's first reply is not a prefix of the case's first reply, R2 could not be
computed: INCONCLUSIVE. R2 holds when the case's output token ids equal the reference chain's at every position.

A3.2 (F reference). Each F turn's reference is the same third prompt sent on the reference chain (possible only when
A3.1 holds for that case, the third prompt contains the second reply). Both lines log an ordinary hit at the same
position. Tokens equal at every position.

A3.3 (R5 control, replaces the old R5). (a) At least two prompts are each read cold twice on different salts: tokens
identical. (b) At least one ordinary-hit chain is run twice on different salts: same hit position, tokens identical.
If (a) or (b) fails the engine is not deterministic and there is no yardstick: INCONCLUSIVE.

A3.4 (cold references). The cold references of C, F and the ordinary K follow-ups are still sent. The compare
prints, per case, equal or the first differing index. No verdict depends on them. R4 keeps its references as they
are (both sides of those comparisons are cold reads). R6 keeps its cold reference (it is a time, not tokens).

A3.5 (unchanged). R1, R3, R4, R6, R7 to R11 and Amendments 1 and 2. ACCEPT = R1, R2 (A3.1), F (A3.2), R3, R4,
R5 (A3.3), R6 to R11 all hold. REJECT = any of them but R5 fails while R5 holds. INCONCLUSIVE = R5 fails, or a check
could not be computed.

A3.6 (before the window). The CPU plan check drives the real scheduler through the new request plan and must show
every reference chain taking an ordinary hit at its case's h. The engine patch bytes are those of the first window,
unchanged. The second window uses a new single-use tag.

## Amendment 4 (after the second window, before any data of a third)

The second window ended INCONCLUSIVE, rolled back byte-exact. R1, R2 on seven of eight cases, F, R3, R4, R5, R6,
R10, R11 and the memory check held. One check could not be computed: the reference chain of case c7 (a 20-token
reply, then the second prompt) took today's ordinary hit at 99984, not at the case's h = 100000, so A3.1 had no
reference at the same position. That the three outputs of c7 (snapshot, cold, ordinary hit at 99984) were equal does
not decide R2 and is not used.

A4.1 (engine fact read from the second window's lines, to be confirmed in the scheduler source on CPU before the plan
is changed). The block pool holds the full blocks of the tokens a finished record had at the START of its last step:
prompt + reply - k, where k is the number of reply tokens the last step added, 1 <= k <= K, K = the profile's
num_speculative_tokens + 1 = 6. The plan's formula took k = 1. The c7 reference ended with k of 5 or 6 (processed
100019, pool hit 99984).

A4.2 (the plan). The reference chain's first reply length r of every case, the chat case included, must put the
ordinary hit at the case's h for EVERY k from 1 to K, and be admitted by today's ordinary-hit window for every
processed length that record can have. The CPU plan check of A3.6 must show this per case and per k. A case for
which no such r exists stops the plan (it is not sent with a weaker r). The same per-k reading applies to every
other place where the plan predicts a pool hit after a finished reply. If the source reading of A4.1 gives a
different range of k or a different rule, the builder reports that before changing anything and the plan follows
the source, not this paragraph.

A4.3 (unchanged). Every rule, reference and verdict of Amendments 1 to 3. A reference chain that logs a hit at any
other position than h stays INCONCLUSIVE. Every rule stays as written.

A4.4 (before the window). The engine patch bytes are those of the first two windows, unchanged. The third window
uses a new single-use tag. The fallback server is verified before the default is stopped.

## Amendment 5 (after the third run stopped in its CPU pre-checks, before any window data of it or later)

The third run never stopped the default: its cost pre-check failed one line ("[off] the switch unset adds under 5 ms
over base") and the script stopped, exit 1. Nothing was stopped, applied or sent. Its tag is used.

Reading of that failure. The cost check times each arm (on, off, base) ONCE, by wall clock, at nice 19 on the host
that is serving the default. In the failing run the admission-log step took 35.08 ms (on) and 31.00 ms (off)
against 18.27 ms (base). In the four earlier runs of the same bytes that step took 17.64 to 19.36 ms in off and base
alike, and off minus base lay between -1.1 and +1.5 ms. One wall-clock sample cannot tell a scheduling delay from
code cost.

A5.1 (measurement, thresholds unchanged). The cost check runs the three arms five times, interleaved (on, off,
base, then again), each run in its own process as now. The decision, reason and hit of every run must be the
expected ones (any run that differs fails the check). The time compared against each threshold is, per arm, the
smallest total of its five runs. The thresholds stay: on under 100 ms, on minus base under 100 ms, off minus base
under 5 ms. All fifteen readings are printed.

A5.2 (unchanged). Everything else in Amendments 1 to 4. The engine patch bytes are unchanged. The next window uses
a new single-use tag. The fallback server is verified again before the default is stopped.

## Amendment 6 (after the fourth window's ACCEPT, about the finish step only)

The fourth window ended ACCEPT, exit 0: R1, R2, F, R3, R4, R5, R6, R10, R11 and the memory check hold, compare 193
readings ok and 0 BAD, Xid 0. Every reference chain took its ordinary hit at its case's h.

The finish script then failed its check A (decision lines 76 ok, 0 bad, switch lines 33 ok, 1 bad), rolled the patch
back byte-exact and booted the default unchanged, as written. The one line it did not know is the engine's own
start-up warning, "(APIServer pid=N) WARNING date time [envs.py:N] Unknown vLLM environment variable detected:
VLLM_V41_SNAPSHOT_HIT": fixed words and numbers, no text of any request. The same line is in the second window's
receipt. Check A had never run on a real receipt before (it only runs after an ACCEPT).

A6.1. Check A's list of allowed switch lines gains exactly that line, matched as a whole line. Nothing else in
check A changes. Tested on the fourth and second windows' receipts (76 0 34 0 each) and on two made-up lines that end
in the variable's name (both refused).

A6.2. The verdict of the fourth window stands, no window is repeated. The install is: verify the fallback server and
an idle default, stop the default, apply the change (the same sealed bytes that the fourth window ran, checked
against their hashes), then the finish script, which repeats check A on the fourth window's receipts, adds the export
line, boots the default and checks the switch (scheduler 1, eight worker ranks) and the boot probe. Any failure on
the way is handled as the finish script's header says. A refused apply boots the default unchanged.
