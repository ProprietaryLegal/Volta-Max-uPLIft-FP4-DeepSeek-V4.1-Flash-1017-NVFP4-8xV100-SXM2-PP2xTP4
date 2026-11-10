# How close to NVFP4: the test

This folder holds everything needed to repeat the closeness test of the `hot_iq` build against NVIDIA's NVFP4 pack:
the item builder, the capture client, the scorer and the report maker, the item ids, and the rules, which were fixed
before any result was seen.

## What is compared

| Build | Role |
|---|---|
| NVFP4 reference | NVIDIA's NVFP4 pack served with every routed expert at its own NVFP4 precision (the two-tier mode). This is a quantized model itself, not the unquantized model. |
| `hot_iq` | the build of this release: hot experts at exactly NVIDIA's NVFP4 values (folded to MXFP4 bit for bit), the others as IQ records |
| Volta sPLIt | the production IQ2 build, the yardstick |
| Reference repeat | the reference run a second time on the first 50 contexts and 50 questions: the noise floor |

All three run on the same eight Tesla V100 cards with the same engine, one request at a time, temperature 0,
thinking off, and a fresh cache salt on every request so that no prompt is served from the prefix cache.

## Why the first token is only a control

When this engine reads a prompt it computes every expert from its full NVFP4 record, for `hot_iq` too. The IQ
records are used only while decoding. So the first generated token of `hot_iq` comes from the same weights as the
reference, and a one-token answer cannot show any difference. The test therefore compares tokens 2 to 32 of a
greedy continuation, while the continuation still equals the reference's (after the first different token the two
builds read different text and the comparison would no longer be like for like). Token 1 is reported separately as
a control. The multiple-choice questions ask for a line `Answer: X`, so the letter is a decoded token, and the
report counts how many letters still came at token 1.

## Measures

For each of 320 contexts (code, prose, legal-style, structured and chat text, 64 each, 96 to 224 tokens):

- **Same next token**: the share of compared tokens where the build's greedy token equals the reference's.
- **Log-probability gap**: the mean of |log p(build) minus log p(reference)| for the reference's token, in nats.
  When that token is not in the build's top 20, the build's 20th log-probability is used, which can only make the
  gap smaller than it is (the share of such tokens is reported).
- **KL divergence**: KL(reference to build) in nats over the tokens in both top-20 lists plus one bucket for all
  other tokens. Grouping tokens can only lower a KL divergence, so this is a lower bound of the full-vocabulary value.
- **First difference**: the first token at which the greedy continuation differs from the reference's, and the share
  of contexts still identical after 1, 2, 3, 5, 9, 17 and 32 tokens.

`hot_iq` against Volta sPLIt is also compared on exactly the same tokens: for each context, the tokens where both
builds still match the reference. The headline is the share of Volta sPLIt's distance that `hot_iq` removes,
1 minus (hot_iq distance divided by Volta sPLIt distance), with its interval.

For 513 MMLU questions (9 from each of 57 subjects): accuracy of each build, its difference from the reference's,
and how often its answer differs from the reference's answer.

Intervals are 95% percentile bootstrap intervals, 2,000 resamples, seed 20261110, resampling whole contexts within
their text kind, or whole questions.

## Rules fixed before the data

- Every measure above is reported whatever it shows, for every build captured. A worse result is published as
  measured.
- The noise floor is ZERO when the reference repeat is identical, token for token and log-probability for
  log-probability (within 1e-6), on all 50 contexts and 50 questions. Otherwise the floor is the repeat's own figures.
- A build is **indistinguishable from the reference at this sample size** on a measure when that measure's 95%
  interval reaches the floor's (with a ZERO floor: when its interval includes 0). Otherwise it **differs from the
  reference**. This does not say how large the difference is. The figures do.
- `hot_iq` is **CLOSER** than Volta sPLIt on a measure when the 95% interval of (Volta sPLIt minus hot_iq) on the
  shared tokens lies above 0, **FARTHER** when it lies below 0, and **INCONCLUSIVE** otherwise. The same rule decides
  the answer-differs rate on the questions.
- Accuracy is **not distinguishable from the reference at this sample size** when the interval of its difference
  includes 0.
- A comparison with fewer than 300 contexts or 500 questions is INCONCLUSIVE. A receipt made on another item plan or
  key, a receipt filed under the wrong build, or a continuation part that does not reproduce the first 10 contexts
  exactly makes the whole score refuse.
- Capture stops at its time budget and says so (`BUDGET_STOP`). A stopped capture is scored on what it holds, under
  the minimums above.

The scripts that apply these rules are `closeness_lib.py` and `score.py`. Their sha256 at the time the rules were
fixed are in `PREREGISTRATION.txt`.

## What the receipts hold

One line per item: hashes of the token ids (HMAC-SHA256 with a random key, cut to 16 hex digits), the 20
log-probabilities of every generated position, the parsed answer letter for questions, and counts. Generated text
and raw token ids are never written. The key is deleted after scoring, so the hashes cannot be turned back into
tokens. The scorer still runs on the receipts with `--key-id`, which checks that every receipt was made with the
same key.

## Repeat it

```bash
python build_items.py --self-test
python capture.py --self-test
python score.py --self-test
python report.py --self-test
python build_items.py fetch --data data
python build_items.py build --data data --tokenizer tokenizer.json --out plan.json --ids ids.json
sha256sum plan.json ids.json    # plan 19f597f3...4d341, ids the same as items/plan-v1.ids.json
python capture.py --plan plan.json --url http://HOST:PORT --arm REF1 --arm-kind reference --passes main,floor \
  --budget-min 45 --key KEYFILE --tokenizer tokenizer.json --out receipts --part 1
python score.py --plan plan.json --key KEYFILE --ref receipts/REF1.part1.jsonl --hot receipts/HOT1.part1.jsonl \
  --volta receipts/VOL1.part1.jsonl --out score.json
python report.py tables --score score.json
python report.py charts --score score.json --out charts
```

`--dry` on any script prints what it would do and, for capture, the request plan and time estimate. A tokenizer
other than the model's gives different windows and a different plan hash.

See `SOURCES.md` for every source and licence.
