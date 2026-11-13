# Window m32 rule: stage-1 reserve +32 MiB

Public copy of the rule written on 2026-11-12 before any data of the window existed. The private file and this copy
differ in names and paths (private file and profile names became descriptions) and in the opening request, which is
said here in one sentence: raise the stage-1 reserve by 32 MiB if possible, otherwise leave it. Nothing else changed.
Results: the [README](../../README.md#update-2026-11-13-the-prompt-cache-survives-a-changed-reply) and
[docs/measurements.md](../measurements.md#stage-1-reserve-32-mib-window-m32).

Finding it answers: window 18's bench boot of the two-slot default sampled 32,000 MiB on the stage-1 cards against
the build's line of 31,998 MiB (0 out-of-memory lines). That mark was first reached 9 minutes into the bench and
came back through most of it (the bench boot's memory samples).

Arm M32a: the default's profile with one value changed: `nvfp4_8r_reserve_bytes` 20: 7517241344 -> 7550795776
(+33,554,432 B). Two slots, iq reading, on the test port.

Test: the code kind of the speed bench (`bench_volta_run.py --kinds code`, 37 requests, prompts 1K to 256K, the same
cells that open the full bench), with the 15 s memory sampler. Read by the memory verdict of window 15's scripts at
31,998 MiB.

INSTALL (make m32 the default: the launcher's default entry, then a boot) only if ALL hold:
1. the boot is READY, eight ranks on iq reading, stage 0 tiers 3691 / 3989 as now (the stage-0 reserve is unchanged).
2. the bench part exits 0.
3. MEMORY PASS: no sample above 31,998 MiB on any card, 0 out-of-memory lines.
4. the Xid count is unchanged and /health is 200 at the end.

Anything else, INCONCLUSIVE included: NOT installed, the default is booted back unchanged (otherwise leave it).
FAULT (a new Xid line, a dead arm, a request error on an arm that answers): nothing is stopped or started.

What a pass does and does not show: the code-kind cells are 37 of the bench's 169. The other kinds are not rerun.
Tokens will differ slightly from the published two-slot tier map (stage 1 holds a few fewer NVFP4 experts).
Public pages are not changed by this window: they describe the profile they were measured with.
