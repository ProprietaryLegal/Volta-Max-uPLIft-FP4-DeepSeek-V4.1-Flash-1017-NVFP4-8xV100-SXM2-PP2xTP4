# Window 18 rule: the final speed bench and the concurrency rows

Public copy of the rule fixed on 2026-11-12 before any window-18 data existed. The private file and this copy differ
in names and paths (private host, agent and file names became descriptions, private script names became what the
scripts do) and in one part left out: the private file's last section, how the host's private service control stops
the served model before the window and restores it after. Nothing else changed. Results: the
[README](../../README.md#speed-bench-2026-11-12-final-build) and
[docs/measurements.md](../measurements.md#prompt-slots-and-the-final-bench-window-18).

## 1. What the window does and does not do

**This window measures. It installs nothing.** It changes no profile, no launcher, no switch and no default. It
measures the build the host's launcher names as the default when the window starts. Its outputs are the speed tables
of the earlier bench (now on that build) and one table of aggregate throughput with one, two and three requests
running together. The window is VALID when the readings below were taken as this file says. It is INCONCLUSIVE when a
reading it needs could not be taken or a check below fails. A FAULT ends it.

## 2. What is measured, read at run time

The launcher's default entry gives the profile PROF, its slot count SLOTS, the residency and the prompt-reading mode.
Two engine changes may land before the window (window 16 adds switches to the launcher, window 17 adds two keys to
every profile and may rename PROF). So no hash of the launcher or of any profile is pinned. The window reads them,
prints them and writes them once to a provenance file: the default entry, the sha256 of the launcher, of every profile
booted and of every window file, a digest of the engine's `nvfp4_8r_*` expert files in the source tree and in the
installed environment, the launcher's export lines and the loader keys of PROF.

PROF must pass the profile check the launcher itself makes (slot count, residency and reading mode) and name one port.
The engine's `nvfp4_8r_*.py` files and `v41_multi_seq.py` must be the same bytes in the source tree and in the
installed environment.

## 3. Arms, in this order

Each arm is a fresh boot from eight empty cards, stopped before the next.

| Arm | Profile | Slots | What runs |
|---|---|---|---|
| B | PROF | SLOTS | the speed bench of the earlier windows: (1) [harness/run/benchvolta.sh](../../harness/run/benchvolta.sh) (169 requests, one at a time), then [harness/run/bench_volta_summary.py](../../harness/run/bench_volta_summary.py), (2) the 216K tool loop, thinking on, (3) production's warm-up and the two short sets |
| S1 | the one-slot sibling of PROF (or PROF when SLOTS = 1) | 1 | window 15's procedure |
| S2 | PROF when SLOTS = 2 (the case on 2026-11-12), else its two-slot sibling | 2 | window 15's procedure |
| S3 | the three-slot sibling of PROF | 3 | window 15's procedure |

On 2026-11-12 PROF was the two-slot default ([template](../../engine/configs/hot_iq-slots2.yaml.template)), so S1 and S3
are the one- and three-slot profiles of window 15 ([one slot](../../engine/configs/hot_iq-slots1.yaml.template),
[three slots](../../engine/configs/hot_iq-slots3.yaml.template)). A slot profile other than PROF must pass the same
profile check with its own slot count and a sibling check: it differs from PROF in `max-num-seqs` (n) and in the two
stage reserves, each by (n - SLOTS) x 160 MiB, and in nothing else. A key that a later install adds to both files
equally is allowed. A partial rerun may name a subset of the slot arms in this order, or the bench or the slot arms
alone, under a new tag.

Window 15's procedure on arm Sn ([window 15 rule](WINDOW15-accept-rule.md)): boot (tier lines and CUDA graph size
printed), memory sampler, production's warm-up, level 1 (the three public prompts of
[harness/run/tput_prompts.py](../../harness/run/tput_prompts.py), one after another, thinking off, temperature 0,
`ignore_eos`, 1,024 tokens out) twice, levels k = 2 to n (k requests started together, request i on prompt p(i mod 3))
twice each, then the mixed case once (a 200K-token public prompt, Python standard-library source, 800,000 characters,
48 tokens out, and 10 s later the short request p0), sampler off, token compare, stop.

The bench arm also runs the memory sampler from its boot to its last request. Its tier lines and graph size are
printed. The bench runs on the default's slot count because that is what the public figures describe. Its requests go
one at a time with a fresh cache salt each, so slots do not change what it measures.

## 4. Readings

**Bench.** The tables of the bench summary, the same summary as the earlier bench (long prompts 1K to 256K, thinking
on, prompt reading by type, short prompts). The 216K loop's per-request and loop lines. The speed lines of the two
short sets.

**Concurrency.** As the window 15 rule, section 4. **The declared reading is run 2 of each level.** Run 1 is printed
beside it for noise and never used for the table.

Per request (from its own stream): tokens, first-token time, decode tok/s = tokens after the first chunk / time from
the first chunk to the last, ms per round = that time / (chunks - 1).

Per level: aggregate tok/s = all tokens of the level / time from the first request's start to the last request's end
(level 1: the sum of its three token counts over the sum of their wall times). Per-request decode tok/s and ms per
round: mean, lowest and highest. Ratio = the level's aggregate tok/s / S1's level-1 aggregate tok/s (run 2).

The table: rows S1 to S3, columns 1 to 3 requests. The KEY ROW is the largest arm's levels (S3: 1, 2 and 3 together)
with the change at each step. Mixed case, per arm: the short request's first-token time, decode tok/s, ms per round,
whether it finished before the long prompt's first token, the long prompt's token count and first-token time.

## 5. Noise known before the window

- First runs of a boot carry 2 to 3 ms per round of noise. Second runs carry about 0.1 ms per round (windows 14 and
  14c, 2026-11-12). Hence run 2 is the declared reading of every level.
- The bench runs each long cell twice (rep 0, rep 1) and each short cell three times, as the earlier bench did. Its
  summary prints the largest difference between the two runs of a cell. On 2026-11-11 that was 23.3 tok/s on a cell.
  No bench figure is a single first run of the boot: the bench's own warm-up request comes first.
- Window 15 measured run 2 against run 1 on the same procedure (its NOISE lines).

## 6. Checks

(a) **Memory, per arm (B, S1, S2, S3).** Every sample on every card at most 31,998 MiB, and 0 out-of-memory lines in
the arm's log ([harness/run/w15_verdict.py](../../harness/run/w15_verdict.py) `mem`). PASS, FAIL or INCONCLUSIVE (no
sample, or the sampler stopped early). A FAIL is a finding: it does not stop the window and is printed on the arm's
line as "memory FAIL". Window 13b's long reads in fast-reading mode peaked at 32,062 MiB on stage 1, so a FAIL on the
bench's 256K cells is possible and is reported as measured.

(x) **Xid.** The kernel log's Xid count must not move from the window's start. It is read before and after every
request batch and every boot and stop. A move is a FAULT: the state is printed, nothing is stopped, the cards are not
queried again and the job starts nothing. An Xid 79 leaves a marker after which the stop command refuses.

(b) **Bench exits.** As the earlier bench treats them: 0 done (VALID), 4 INCONCLUSIVE, 5 prompt drift (the prompts'
token counts differ from Volta sPLIt's sweep, so the tables are not the same prompts: INCONCLUSIVE). After 0, 4 or 5
the receipts and the summary are kept and the window goes on to the slot arms. Any other exit (3 is a server error) is
a FAULT: the window stops with the bench arm serving, no slot arm starts and the default is NOT restored. A summary
that cannot be written stops the window the same way.

(t) **Token-exact, slot arms.** On each slot arm, every request of every level k >= 2 and the mixed-case short request
must give the same token ids as the same prompt in L1 run 2 of that arm, and L1 run 1 must equal L1 run 2. Each arm
reads MATCH, DIFFER (with the count and the first differing position) or INCONCLUSIVE. The throughput readings stand
either way. **The window is VALID only with MATCH on every slot arm**, because the public page states that requests
running together give the same output.

(c) **Completeness.** Every request of a level returns 1,024 tokens with its token ids. Otherwise that level reads
INCONCLUSIVE and its cell shows it. A request error on an arm that answers is a FAULT (the arm is left serving).

(r) **Rounds.** The chunk count is checked against the server's `vllm:spec_decode_num_drafts_total` change over the
level: within 2 % of each other, the ms-per-round figures stand. Otherwise that level's ms per round reads INCONCLUSIVE
(the aggregate tok/s does not depend on it).

## 7. The last lines, and VALID

`BENCH W18 SUMMARY FILE PATH`, `BENCH W18 B: ...` (profile, slots, reading, bench exit and its read, tiers, graphs,
memory, parts), `TABLE W18 (run 2, the declared reading)` and its rows, `TABLE W18 run 1`, `KEY ROW W18 S3 ...`,
`NOISE W18`, `MIXED W18 SN` per slot arm, `ARM W18 SN` per slot arm, then `WINDOW W18 VALID` or
`WINDOW W18 INCONCLUSIVE: REASONS`.

VALID needs all of: bench exit 0, all three bench parts done, a memory read of PASS or FAIL on every arm, every slot
arm's levels complete (c), and tokens MATCH on every slot arm (t). Anything else is INCONCLUSIVE with the reasons
listed. A FAULT ends the window with the readings as they stand. The window script exits 0 when it ran to the end
(VALID or INCONCLUSIVE), and only then is the default booted back.

The window's request script, [harness/run/w18_tput.py](../../harness/run/w18_tput.py), is derived from window 15's
([harness/run/w15_tput.py](../../harness/run/w15_tput.py)) with window 18's arm names and a table of up to three
columns. The scripts that ran the window boot and stop the arms through this
host's private service control and are not published.
