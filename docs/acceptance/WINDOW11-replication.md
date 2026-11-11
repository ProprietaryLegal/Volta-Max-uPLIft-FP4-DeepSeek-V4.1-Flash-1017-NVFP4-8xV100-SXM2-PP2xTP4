# Window 11b/11c: one replication of window 11, and the install gate (scrubbed public copy)

This is a scrubbed public copy of the note that fixed the one replication of window 11 and, later, the memory gate
for installing the kernel. Hosts, paths, internal names and file hashes are left out. Every rule and figure is given
in full. The private file was last changed on 2026-11-11 at 06:23:46 UTC, when the last section was added. Each
section gives the time it was written. No separate timestamp shows that the first two sections were unchanged
between 05:41 and 06:23 UTC.

## What window 11 showed

- Token-exact held everywhere: the chain after the kernel patch was SAME as the build before it (window 10a's
  captures), the kernel was SAME as the chain on both capture sets, and the throughput compares of B1, B2 and A2
  against A1 all printed MATCH. No GPU fault in the kernel log.
- Decode ms per round: A1 34.29, A1r 33.84, B1 28.54, B1r 28.19, B2 30.79, B2r 28.09, A2 34.01, A2r 33.79.
- p1 -5.774, p2 -2.471, mean -4.123. Noise floor 3.429 (B2r - B2), twice that 6.858.
- VERDICT by the [acceptance rule](WINDOW11-accept-rule.md): NO MEASURABLE CHANGE. The kernel stays off on this
  window's evidence.
- Observation, not a verdict: inside B2's first run prompt 0 ran at 28.43, prompt 1 at 32.13 and prompt 2 at 34.08 ms
  per round. The repeat on the same boot ran 28.01 / 28.16 / 28.19. Seven of eight runs were within 0.5 ms of their
  arm's other runs. The cause of the one slow run is not known (no kernel-log line, no arm-log line, and the Hugging
  Face upload was running on the host throughout).

## Rule for the replication (written 05:41 UTC, after window 11 and before any replication data)

- Same window script and same acceptance rule, both unchanged. The only addition is a side recorder outside the
  script (`vmstat -t 5`) so a slow run can be matched to host load.
- The install decision follows the replication's VERDICT line alone, by the unchanged rule. Window 11's figures are
  not pooled into it.
- This is the only replication. If it is again NO MEASURABLE CHANGE the kernel stays off, and the next step is
  finding what makes a kernel-arm run slow, not a third identical window.
- Both windows are recorded and, if anything is published, both are published.

## Note (05:42 UTC, before any replication data)

The first start under the tag `b` stopped in preflight: the recorder's file name matched the script's single-use
check for that tag's receipts. Nothing booted and no figure exists. The replication ran under the tag `c`. Everything
above stands, reading "11c" for "11b".

## Window 11c result and the install gate (written 06:23 UTC, before the memory arm's data)

- Window 11c: token-exact in every compare, no GPU fault. Decode ms per round A1 33.90, A1r 33.83, B1 29.17, B1r
  28.26, B2 28.53, B2r 28.23, A2 33.88, A2r 33.81. p1 -4.514, p2 -5.364, mean -4.939. Noise floor 1.135, twice that
  2.270. VERDICT ACCEPT by the unchanged rule.
- Install gate, fixed here before its data: the kernel profile must pass plan R1's memory gate unchanged (the same
  memory-arm script as R1 with the profile and names changed). The 500K-token read and the 216K tool loop both exit
  0, `/health` returns 200 after each, the arm log has no out-of-memory line, and the peak memory of every card of
  both stages is at or below 31,998 MiB. PASS: the `hot_iq` launch option boots the kernel profile. FAIL: it stays on
  the chain profile and the kernel profile stays an opt-in file.

## Result of the install gate (added to this public copy, not part of the private note)

PASS on 2026-11-11 (memory arm started 06:23:46 UTC): 500K-token read (498,099 prompt tokens) exit 0, 216K tool loop
exit 0, `/health` 200 after each, no out-of-memory line, peak 31,694 MiB on stage 0 and 30,994 MiB on stage 1.
