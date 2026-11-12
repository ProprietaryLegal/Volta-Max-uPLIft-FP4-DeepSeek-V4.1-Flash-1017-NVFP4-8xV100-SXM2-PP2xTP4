# Window 15 rule: aggregate throughput of one to four prompt slots

Public copy of the rule fixed on 2026-11-12 before any window-15 data existed. The private file and this copy differ
only in names and paths: private host, agent and file names became descriptions, and a private script name became
what the script does. Nothing else changed. Results: [docs/measurements.md](../measurements.md#prompt-slots-window-15).

The question it answers: with two boards, are two or four slots the better choice? It needs aggregate throughput.

## 1. What the window decides and what it does not

The window measures. It installs nothing and picks no slot count. Its main output is one table of aggregate decode
throughput, rows = slots on the arm, columns = requests running together. The maintainer chooses the slot count from
that table, the memory line of each arm and the price in HOT experts. The window is VALID when the readings below
were taken as this file says. It is INCONCLUSIVE when a reading it needs could not be taken. A FAULT ends it.

## 2. Arms

Each arm is a fresh boot from eight empty cards with fast (iq) prompt reading, then stopped before the next.
Order: S1, S2, S4, S3 (three slots last, so it is the arm to cut).

| Arm | Profile | Slots |
|---|---|---|
| S1 | the one-slot profile with the stage-1 change ([template](../../engine/configs/hot_iq-slots1.yaml.template)) | 1 |
| S2 | [two slots](../../engine/configs/hot_iq-slots2.yaml.template) | 2 |
| S4 | four slots (its template no longer ships since the prompt-reading kernels of 2026-11-12) | 4 |
| S3 | [three slots](../../engine/configs/hot_iq-slots3.yaml.template) | 3 |

All four carry `nvfp4_8r_iq_tail: fused` and the stage-1 change (192 MiB more stage-1 reserve than the one-slot
profile with fast reading, whose stage 1 had sampled 32,062 MiB, above the 31,998 MiB line). S1 is the one-slot arm of
the table, so every row shares the same stage-1 margin. Each extra slot adds 160 MiB of reserve on both stages for its
CUDA graphs. The templates linked above carry the tier map instead of the tally, and the public port.

## 3. Measurements on each arm, in this order

1. Boot. The tier lines are printed (HOT counts per stage and digests, aggregates). On S1 the stage-0 counts and digest
   must equal R1's (3,833 / 3,847, digest 8e7829184090): the stage-0 reserve is unchanged, so anything else means the
   arm is not the installed build, and the window is INCONCLUSIVE and stops. The CUDA graph size line is printed.
2. The memory sampler starts (one sample per 15 s on every card) and runs until the arm's last request has ended.
3. A warm-up request.
4. Level 1: the three public short prompts (`TPUT_PROMPTS` p0 p1 p2, as in
   [harness/run/tput_prompts.py](../../harness/run/tput_prompts.py)), one after another, thinking off, temperature 0,
   `ignore_eos`, 1,024 tokens out. Run twice: L1 run 1, then L1 run 2.
5. Levels k = 2 up to the arm's slot count: k requests started together (a thread barrier, the start skew is
   printed), request i on prompt p(i mod 3), same settings. Each level twice: run 1, then run 2.
6. The mixed case, once: a 200K-token public prompt (Python standard-library source, 800,000 characters, the
   [longprompt.py](../../harness/run/longprompt.py) text, thinking off, 48 tokens out) is sent. 10 s later the short
   request p0 (1,024 tokens) is sent.
7. The sampler stops. The Xid count is read. The arm is stopped.

## 4. Readings

**The declared reading is run 2 of each level.** First runs of a boot carry 2 to 3 ms per round of noise, second runs
about 0.1 (measured at the fused-tail install, windows 14 and 14c). Run 1 is printed beside it and is never used for
the table.

Per request (from its own stream): tokens, first-token time, decode tok/s = tokens after the first chunk / time from
the first chunk to the last, ms per round = that time / (chunks - 1). One chunk is one engine step of that request.

Per level:
- aggregate tok/s = all tokens of the level / time from the first request's start to the last request's end. Level 1
  is the sum of its three token counts over the sum of their wall times (one stream).
- per-request decode tok/s and ms per round: mean, lowest and highest over the level's requests.
- ratio = the level's aggregate tok/s / S1's level-1 aggregate tok/s (run 2). This is "the ratio to the one-slot single
  stream".

The table: rows S1 to S4, columns 1 to 4 requests, each cell "aggregate tok/s (ratio), per-request decode tok/s mean,
ms per round mean". A cell for more requests than the arm has slots is empty. A second table gives run 1 for noise.

Mixed case, per arm: the short request's first-token time, decode tok/s and ms per round, whether it finished before the
long prompt's first token, the long prompt's token count and first-token time.

## 5. Checks

(a) **Memory.** Every sample on every card at most 31,998 MiB, and 0 out-of-memory lines in the arm's log
([w15_verdict.py](../../harness/run/w15_verdict.py) `mem`). Read per arm: PASS, FAIL or INCONCLUSIVE (no sample, or the
sampler stopped early). A FAIL does not stop the window. It is printed on that arm's ARM line as "memory FAIL": that
slot count does not fit as staged.

(x) **Xid.** The kernel log's Xid count must not move from the window's start. A move is a FAULT: the script prints
the state and exits, nothing is stopped, and the cards are not queried again.

(t) **Token-exact.** On each arm, every request of every level k >= 2 and the mixed-case short request must give the
same token ids as the same prompt in L1 run 2 of that arm. L1 run 1 against L1 run 2 is checked too. Each arm reads
MATCH or DIFFER with the count and the first differing position. A DIFFER is a reported finding. It does not stop the
window and does not change the throughput readings. The code predicts MATCH: each step runs one request with the same
kernels (`v41_multi_seq.py:12-14` of the engine).

(c) **Completeness.** Every request of a level returns 1,024 tokens with its token ids. Otherwise that level reads
INCONCLUSIVE and its cell shows it. A request error on an arm that answers is a FAULT (the arm is left serving).

(r) **Rounds.** The chunk count is checked against the server's `vllm:spec_decode_num_drafts_total` change over the
level: within 2 % of each other, the ms-per-round figures stand. Otherwise that level's ms per round reads INCONCLUSIVE
(the aggregate tok/s does not depend on it).

(n) **Noise.** For each level, the run-1 and run-2 aggregate tok/s are printed with their difference. Run 2 is the
reading whatever the difference.

## 6. The last lines

`TABLE W15` (section 4), `MIXED W15` per arm, one `ARM` line per arm (slots, tiers, graph size, memory read, token-exact
read, levels complete), and `WINDOW W15 VALID` or `WINDOW W15 INCONCLUSIVE: REASONS`. VALID needs every booted arm's
levels complete (c), a memory read of PASS or FAIL (not INCONCLUSIVE) and S1's stage-0 tiers equal to R1's. Nothing is
installed or changed.

The request script and the verdict are published: [harness/run/w15_tput.py](../../harness/run/w15_tput.py) and
[harness/run/w15_verdict.py](../../harness/run/w15_verdict.py). The script that ran the window boots and stops the
arms through this host's private service control and is not published. It runs the steps above in this order.
