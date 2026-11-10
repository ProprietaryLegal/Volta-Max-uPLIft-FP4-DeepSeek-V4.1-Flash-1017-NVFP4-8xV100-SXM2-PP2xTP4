"""Plan 01 section 8 "Quality": the score mode of quality_t3.py, run from the
command line on receipts in a directory.

The receipts are hand-made (6 tokens per prompt) and every expected value is
arithmetic written out here, not taken from the script:

  reference N2      p0 ids 10..15  logprob -0.1 -0.2 -0.3 -0.4 -0.5 -0.6
                    p1 ids 20..25  logprob -1.0 x 6
                    p2 ids 30..35  logprob -0.5 x 6
  T3  p0 token 4 differs           first divergence 4
         |d| = .01 .02 .03 .00     mean 0.06 / 4 = 0.015
      p1 no token differs          first divergence 6 (= tokens)
         |d| = .1 .1 0 0 .3 0      mean 0.5 / 6
      p2 token 0 differs           first divergence 0, no common prefix: no mean
  P   p0 token 2 differs           first divergence 2
         |d| = .1 .2               mean 0.3 / 2 = 0.15
      p1 token 5 differs           first divergence 5
         |d| = .5 .5 0 0 0         mean 1.0 / 5 = 0.2
      p2 token 3 differs           first divergence 3
         |d| = .1 .1 .1            mean 0.3 / 3 = 0.1

  Rule (plan): PASS if T3's index >= P's on at least two prompts and T3's mean
  is below P's on at least two.
    index  4>=2 yes, 6>=5 yes, 0>=3 no         -> holds
    mean   .015<.15 yes, .0833<.2 yes, p2 none -> holds        => PASS
  Arms swapped (P's receipt scored as the T3 arm):
    index  2>=4 no, 5>=6 no                    -> fails         => FAIL
  T3m: ids of N2 except p0 token 4; |d| = 1.0 everywhere on p0 and p1, 0.05 on p2
    index  4>=2, 6>=5, 6>=3                    -> holds
    mean   1.0<.15 no, 1.0<.2 no, .05<.1 yes   -> fails         => FAIL
  T3u and Pu both differ at token 0 on p1 and p2 (p0 as T3 and P):
    index  4>=2, 0>=0, 0>=0                    -> holds
    mean   yes, none, none                     -> undecided     => INCONCLUSIVE
  INCONCLUSIVE also when: the N2, T3 or P receipt is absent; no second N2
  capture is given (self-determinism not checked); the second N2 capture
  differs in a token (N2c: p1 token 3; floor first divergence 3, mean
  |d| over 3 tokens = (0 + 0 + .3) / 3 = 0.1); a receipt is for other prompts.

Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("quality_t3.py")
SHA = {"p0": "a" * 64, "p1": "b" * 64, "p2": "c" * 64}
N2 = {
    "p0": ([10, 11, 12, 13, 14, 15], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]),
    "p1": ([20, 21, 22, 23, 24, 25], [-1.0] * 6),
    "p2": ([30, 31, 32, 33, 34, 35], [-0.5] * 6),
}
T3 = {
    "p0": ([10, 11, 12, 13, 99, 15], [-0.11, -0.22, -0.27, -0.4, -3.0, -3.0]),
    "p1": ([20, 21, 22, 23, 24, 25], [-1.1, -0.9, -1.0, -1.0, -1.3, -1.0]),
    "p2": ([77, 31, 32, 33, 34, 35], [-0.5] * 6),
}
P = {
    "p0": ([10, 11, 98, 13, 14, 15], [-0.2, -0.4, -9.0, -9.0, -9.0, -9.0]),
    "p1": ([20, 21, 22, 23, 24, 97], [-1.5, -1.5, -1.0, -1.0, -1.0, -7.0]),
    "p2": ([30, 31, 32, 96, 34, 35], [-0.6, -0.6, -0.6, -8.0, -8.0, -8.0]),
}
T3M = {
    "p0": ([10, 11, 12, 13, 99, 15], [-1.1, -1.2, -1.3, -1.4, -0.5, -0.6]),
    "p1": ([20, 21, 22, 23, 24, 25], [-2.0] * 6),
    "p2": ([30, 31, 32, 33, 34, 35], [-0.55] * 6),
}
T3U = {"p0": T3["p0"], "p1": ([88, 21, 22, 23, 24, 25], [-1.0] * 6), "p2": T3["p2"]}
PU = {"p0": P["p0"], "p1": ([87, 21, 22, 23, 24, 25], [-1.0] * 6),
      "p2": ([86, 31, 32, 33, 34, 35], [-0.5] * 6)}
N2C = {"p0": N2["p0"], "p1": ([20, 21, 22, 55, 24, 25], [-1.0, -1.0, -1.3, -1.0, -1.0, -1.0]),
       "p2": N2["p2"]}


def write(directory: Path, arm: str, prompts: dict, sha: dict = SHA) -> None:
    body = {"check": "T3-quality-capture", "arm": arm, "prompts": [
        {"id": pid, "prompt_sha256": sha[pid], "token_ids": ids, "logprobs": lps,
         "top_ids": [[t] for t in ids], "top_logprobs": [[v] for v in lps]}
        for pid, (ids, lps) in prompts.items()
    ]}
    (directory / f"quality-{arm}.json").write_text(json.dumps(body) + "\n")


def score(directory: Path, *arms: str) -> tuple[int, dict | None]:
    run = subprocess.run([sys.executable, str(SCRIPT), "score", *arms, "--dir", str(directory)],
                         capture_output=True, text=True)
    if run.returncode not in (0, 1, 3):
        return run.returncode, None
    return run.returncode, json.loads(run.stdout)


def close(got: object, want: float | None) -> bool:
    if want is None:
        return got is None
    return isinstance(got, float) and abs(got - want) <= 1e-12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True, help="directory for the test receipts")
    a = parser.parse_args()
    d = Path(tempfile.mkdtemp(prefix="t3-quality-", dir=a.work))
    for arm, prompts in (("N2", N2), ("N2b", N2), ("N2c", N2C), ("T3", T3), ("P", P),
                         ("T3m", T3M), ("T3u", T3U), ("Pu", PU)):
        write(d, arm, prompts)
    write(d, "Pother", P, {**SHA, "p1": "d" * 64})
    fails: list[str] = []

    def figures(out: dict, arm: str) -> list[tuple]:
        return [(row["prompt"], row["tokens"], row["first_divergence"], row["mean_abs_dlogprob"])
                for row in out["arms"][arm]]

    def check(name: str, arms: tuple[str, ...], code: int, status: str) -> dict | None:
        got, out = score(d, *arms)
        if got != code or out is None or out.get("status") != status:
            fails.append(f"{name}: exit {got} status {None if out is None else out.get('status')}, "
                         f"want exit {code} {status}")
            return None
        return out

    out = check("pass", ("N2", "T3", "P", "N2b"), 0, "PASS")
    if out is not None:
        want = {"T3": [("p0", 6, 4, 0.015), ("p1", 6, 6, 0.5 / 6), ("p2", 6, 0, None)],
                "P": [("p0", 6, 2, 0.15), ("p1", 6, 5, 0.2), ("p2", 6, 3, 0.1)]}
        for arm, rows in want.items():
            got = figures(out, arm)
            if len(got) != 3 or any(
                g[:3] != w[:3] or not close(g[3], w[3]) for g, w in zip(got, rows)
            ):
                fails.append(f"pass: figures of {arm} are {got}, want {rows}")
        rule = out["rule"]
        if (rule["index_ge"], rule["mean_below"]) != ([True, True, False], [True, True, None]):
            fails.append(f"pass: votes {rule['index_ge']} {rule['mean_below']}")
        floor = out["floor"]
        if floor["self_deterministic"] is not True or any(
            (row["first_divergence"], row["mean_abs_dlogprob"]) != (6, 0.0)
            for row in floor["per_prompt"]
        ):
            fails.append(f"pass: floor of an identical second capture is {floor}")
        saved = d / "quality-score-N2-T3-P.json"
        if not saved.is_file() or json.loads(saved.read_text()) != out:
            fails.append("pass: the score receipt is absent or differs from what was printed")

    out = check("swapped", ("N2", "P", "T3", "N2b"), 1, "FAIL")
    if out is not None and out["rule"]["index_ge"] != [False, False, True]:
        fails.append(f"swapped: index votes {out['rule']['index_ge']}")

    out = check("mean", ("N2", "T3m", "P", "N2b"), 1, "FAIL")
    if out is not None:
        rule = out["rule"]
        if (rule["index_ge"], rule["mean_below"]) != ([True, True, True], [False, False, True]):
            fails.append(f"mean: votes {rule['index_ge']} {rule['mean_below']}")
        got = figures(out, "T3m")
        want_m = [("p0", 6, 4, 1.0), ("p1", 6, 6, 1.0), ("p2", 6, 6, 0.05)]
        if any(g[:3] != w[:3] or abs(g[3] - w[3]) > 1e-9 for g, w in zip(got, want_m)):
            fails.append(f"mean: figures of T3m are {got}")

    out = check("undecided", ("N2", "T3u", "Pu", "N2b"), 3, "INCONCLUSIVE")
    if out is not None and out["rule"]["mean_below"] != [True, None, None]:
        fails.append(f"undecided: mean votes {out['rule']['mean_below']}")

    out = check("no second N2", ("N2", "T3", "P"), 3, "INCONCLUSIVE")
    if out is not None and (out["floor"] is not None or len(figures(out, "T3")) != 3):
        fails.append("no second N2: the figures must still be printed and the floor must be null")

    out = check("N2 not deterministic", ("N2", "T3", "P", "N2c"), 3, "INCONCLUSIVE")
    if out is not None:
        floor = out["floor"]
        row = floor["per_prompt"][1]
        if floor["self_deterministic"] is not False or row["first_divergence"] != 3 or not close(
            row["mean_abs_dlogprob"], 0.1
        ):
            fails.append(f"N2 not deterministic: floor {floor}")

    for name, arms in (("N2 absent", ("N2x", "T3", "P", "N2b")), ("T3 absent", ("N2", "T3x", "P", "N2b")),
                       ("P absent", ("N2", "T3", "Px", "N2b")),
                       ("other prompts", ("N2", "T3", "Pother", "N2b"))):
        check(name, arms, 3, "INCONCLUSIVE")

    status = "FAIL" if fails else "PASS"
    print(status, json.dumps({"check": "T3-quality-score-test", "failures": fails}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
