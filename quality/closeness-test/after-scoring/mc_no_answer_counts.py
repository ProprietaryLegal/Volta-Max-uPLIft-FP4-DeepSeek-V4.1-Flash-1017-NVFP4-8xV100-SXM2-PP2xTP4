#!/usr/bin/env python3
"""Multiple-choice "no answer" counts, made after scoring and outside the preregistered rules.

    python3 -I mc_no_answer_counts.py            # reads ../receipts and ../score-v1.json next to this folder

Reads only the answer letters of the main-pass multiple-choice lines of the receipts (bridge lines left out, the
parts of an arm joined), checks every receipt's sha256 against the scorer's "inputs" in score-v1.json and the totals
against its figures, then prints:
  (a) whether the reference and hot_iq gave no answer on the same questions,
  (b) how many of Volta sPLIt's changed answers fall on the questions the reference did not answer,
  (c) each build's changed-answer share on the questions the reference answered.
"Changed" is the scorer's own definition: the build's answer is not equal to the reference's (no answer counts as an
answer of its own). Fails on the first mismatch.
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ARMS = {"reference": ["CLS1-REF.part1.jsonl"], "hot_iq": ["CLS1-HOT.part1.jsonl", "CLS1-HOT.part2.jsonl"],
        "volta_split": ["CLS1-VOL.part1.jsonl", "CLS1-VOL.part2.jsonl"]}


def answers(files: list[str]) -> dict:
    out = {}
    for f in files:
        for line in open(HERE / "receipts" / f, encoding="utf-8"):
            d = json.loads(line)
            if d.get("type") == "item" and d["kind"] == "mc" and d["pass"] == "main":
                if d["item"] in out:
                    raise SystemExit(f"{f}: item {d['item']} twice in the main pass")
                out[d["item"]] = d["answer"]
    return out


score = json.load(open(HERE / "score-v1.json", encoding="utf-8"))
for arm, files in ARMS.items():
    want = [x["sha256"] for x in score["inputs"][arm]]
    got = [hashlib.sha256((HERE / "receipts" / f).read_bytes()).hexdigest() for f in files]
    if got != want:
        raise SystemExit(f"{arm}: receipts {got} are not the scored inputs {want}")
a = {arm: answers(files) for arm, files in ARMS.items()}
ids = sorted(a["reference"])
if any(sorted(x) != ids for x in a.values()):
    raise SystemExit("the arms do not hold the same questions")
none = {arm: {i for i in ids if a[arm][i] is None} for arm in a}
changed = {arm: {i for i in ids if a[arm][i] != a["reference"][i]} for arm in ("hot_iq", "volta_split")}
for arm in ("hot_iq", "volta_split"):
    mc = score["arms"][arm]["mc"]
    if (len(none[arm]) != mc["no_answer"] or len(none["reference"]) != mc["reference_no_answer"]
            or abs(len(changed[arm]) / len(ids) - mc["answer_differs_from_reference"]["value"]) > 1e-12):
        raise SystemExit(f"{arm}: counts do not match score-v1.json")
answered = [i for i in ids if a["reference"][i] is not None]
print(f"questions {len(ids)}, no answer: reference {len(none['reference'])}, hot_iq {len(none['hot_iq'])}, "
      f"Volta sPLIt {len(none['volta_split'])}")
print(f"(a) the reference's and hot_iq's no-answer questions are the same: {none['reference'] == none['hot_iq']} "
      f"(in both {len(none['reference'] & none['hot_iq'])})")
print(f"(b) Volta sPLIt's changed answers on the reference's no-answer questions: "
      f"{len(changed['volta_split'] & none['reference'])} of {len(changed['volta_split'])}")
for arm in ("hot_iq", "volta_split"):
    n = sum(1 for i in answered if a[arm][i] != a["reference"][i])
    print(f"(c) {arm}: changed on the {len(answered)} questions the reference answered: {n} = {100 * n / len(answered):.1f} %")
