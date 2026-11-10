#!/usr/bin/env python3
"""Plan 01 section 8 "Quality": how far an arm's greedy output is from N2's.

  quality_t3.py capture URL ARM         three requests, one at a time
  quality_t3.py score N2 T3 P [N2B]     N2B = a second capture of N2

capture sends the three public prompts of ab_t1_v2.py (TPUT_PROMPTS, read from
that file as text) with the request its --tput mode sends (temperature 0,
max_tokens 1024, ignore_eos, thinking off, streamed, return_token_ids) plus
logprobs, top_logprobs 5 and return_tokens_as_token_ids (the server then names
logprob tokens by id, so nothing textual is kept). Before each request
vllm:num_requests_running must be 0, otherwise INCONCLUSIVE. It writes
logs/t3/quality-ARM.json: token ids and logprobs only, no text. An existing
receipt is never overwritten.

score compares arms T3 and P with the reference N2, per prompt:
  first_divergence   first index where the token differs from N2's
                     (= tokens when none differs)
  mean_abs_dlogprob  mean |logprob - N2's logprob| of the chosen token over
                     the tokens before that index (null when there are none)
Rule: PASS if T3's first_divergence >= P's on at least two prompts and T3's
mean is below P's on at least two; FAIL if either is refused by two prompts.
INCONCLUSIVE if a receipt is absent or unusable, if N2 is not shown to be
self-deterministic (N2B absent, or N2B differs from N2 in a token), or if
fewer than two prompts can decide the mean rule. The figures of N2B against
N2 are printed as the self-determinism floor. The result is printed as JSON
and saved as logs/t3/quality-score-N2-T3-P.json.

Exit 0 PASS (capture: receipt written), 1 FAIL, 3 INCONCLUSIVE.
"""

from __future__ import annotations

import os
import argparse
import ast
import datetime
import hashlib
import json
import math
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RECEIPTS = Path(os.environ["LANE"] + "/logs/t3")
HARNESS = Path(os.environ["PROD"] + "/1cat-1277/bench5/ab_t1_v2.py")  # read as text, never imported
MAX_TOKENS = 1024
TOP = 5
RUNNING = "vllm:num_requests_running"
NO_LOGPROB = -9999.0  # what the server reports for a token it has no logprob for
ARM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
TOKEN_ID = re.compile(r"token_id:(\d+)")


class Inconclusive(Exception):
    """The check cannot decide; the message names what is missing."""


@dataclass(frozen=True)
class PromptCapture:
    id: str
    prompt_sha256: str
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    top_ids: tuple[tuple[int, ...], ...]
    top_logprobs: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        if n == 0 or not (len(self.logprobs) == len(self.top_ids) == len(self.top_logprobs) == n):
            raise ValueError(f"{self.id}: token ids, logprobs and top lists differ in length")
        if any(len(i) != len(v) or not i for i, v in zip(self.top_ids, self.top_logprobs)):
            raise ValueError(f"{self.id}: a top list is empty or its ids and logprobs differ")
        if not all(math.isfinite(v) for v in self.logprobs):
            raise ValueError(f"{self.id}: a logprob is not finite")

    def to_json(self) -> dict:
        return {"id": self.id, "prompt_sha256": self.prompt_sha256,
                "tok_sha": hashlib.sha256(json.dumps(list(self.token_ids)).encode()).hexdigest()[:16],
                "token_ids": list(self.token_ids), "logprobs": list(self.logprobs),
                "top_ids": [list(t) for t in self.top_ids],
                "top_logprobs": [list(t) for t in self.top_logprobs]}


@dataclass(frozen=True)
class Capture:
    arm: str
    prompts: tuple[PromptCapture, ...]

    @classmethod
    def from_receipt(cls, directory: Path, arm: str) -> Capture:
        """Parse quality-ARM.json once; ValueError if it is not a capture of arm."""
        path = receipt_path(directory, arm)
        if not path.is_file():
            raise Inconclusive(f"receipt absent: {path}")
        body = json.loads(path.read_text())
        if not isinstance(body, dict) or body.get("arm") != arm or not isinstance(
            body.get("prompts"), list
        ):
            raise ValueError(f"{path} is not a capture of arm {arm}")
        prompts = []
        for item in body["prompts"]:
            keys = ("id", "prompt_sha256", "token_ids", "logprobs", "top_ids", "top_logprobs")
            if not isinstance(item, dict) or any(k not in item for k in keys):
                raise ValueError(f"{path}: a prompt lacks one of {keys}")
            prompts.append(PromptCapture(
                str(item["id"]), str(item["prompt_sha256"]),
                tuple(int(t) for t in item["token_ids"]),
                tuple(float(v) for v in item["logprobs"]),
                tuple(tuple(int(t) for t in top) for top in item["top_ids"]),
                tuple(tuple(float(v) for v in top) for top in item["top_logprobs"]),
            ))
        if not prompts:
            raise ValueError(f"{path} holds no prompt")
        return cls(arm, tuple(prompts))


def receipt_path(directory: Path, arm: str) -> Path:
    if ARM.fullmatch(arm) is None:
        raise ValueError(f"arm label {arm!r}: letters, digits, - and _ only")
    return directory / f"quality-{arm}.json"


# ---------------------------------------------------------------- score (pure)

@dataclass(frozen=True)
class Agreement:
    """One arm against the reference on one prompt."""

    prompt: str
    tokens: int  # tokens compared (the shorter of the two outputs)
    first_divergence: int  # == tokens when no token differs
    mean_abs_dlogprob: float | None  # None: the outputs differ at token 0

    def to_json(self) -> dict:
        return {"prompt": self.prompt, "tokens": self.tokens,
                "first_divergence": self.first_divergence,
                "diverged": self.first_divergence < self.tokens,
                "mean_abs_dlogprob": self.mean_abs_dlogprob}


def agreement(ref: PromptCapture, arm: PromptCapture) -> Agreement:
    if (ref.id, ref.prompt_sha256) != (arm.id, arm.prompt_sha256):
        raise ValueError(f"prompt {arm.id} is not the reference's prompt {ref.id}")
    tokens = min(len(ref.token_ids), len(arm.token_ids))
    first = next((i for i in range(tokens) if ref.token_ids[i] != arm.token_ids[i]), tokens)
    mean = None
    if first > 0:
        mean = math.fsum(abs(arm.logprobs[i] - ref.logprobs[i]) for i in range(first)) / first
    return Agreement(ref.id, tokens, first, mean)


def against(ref: Capture, arm: Capture) -> list[Agreement]:
    if len(ref.prompts) != len(arm.prompts):
        raise ValueError(f"{arm.arm} holds {len(arm.prompts)} prompts, {ref.arm} {len(ref.prompts)}")
    return [agreement(r, x) for r, x in zip(ref.prompts, arm.prompts)]


def at_least_two(votes: list[bool | None]) -> bool | None:
    """True / False when two prompts agree on it, None when they cannot decide."""
    if votes.count(True) >= 2:
        return True
    if votes.count(False) >= 2:
        return False
    return None


def score(directory: Path, n2: str, t3: str, p: str, n2b: str | None) -> dict:
    result: dict = {"check": "T3-quality-score", "status": "INCONCLUSIVE", "reference": n2,
                    "reasons": [], "arms": {}, "floor": None, "rule": None}
    reasons: list[str] = result["reasons"]
    captures: dict[str, Capture] = {}
    for arm in (n2, t3, p) + (() if n2b is None else (n2b,)):
        try:
            captures[arm] = Capture.from_receipt(directory, arm)
        except (Inconclusive, ValueError) as exc:
            reasons.append(str(exc))
    if reasons:
        return result
    try:
        rows = {arm: against(captures[n2], captures[arm]) for arm in (t3, p)}
        floor = None if n2b is None else against(captures[n2], captures[n2b])
    except ValueError as exc:
        reasons.append(str(exc))
        return result
    result["arms"] = {arm: [row.to_json() for row in rows[arm]] for arm in (t3, p)}
    index_ge: list[bool | None] = [
        a.first_divergence >= b.first_divergence for a, b in zip(rows[t3], rows[p])
    ]
    mean_below: list[bool | None] = [
        None if a.mean_abs_dlogprob is None or b.mean_abs_dlogprob is None
        else a.mean_abs_dlogprob < b.mean_abs_dlogprob
        for a, b in zip(rows[t3], rows[p])
    ]
    index_holds, mean_holds = at_least_two(index_ge), at_least_two(mean_below)
    result["rule"] = {"index_ge": index_ge, "mean_below": mean_below,
                      "index_holds": index_holds, "mean_holds": mean_holds}
    if floor is None:
        reasons.append(f"{n2} self-determinism not checked: give a second capture of {n2}")
        return result
    deterministic = all(row.first_divergence == row.tokens for row in floor)
    result["floor"] = {"arm": n2b, "self_deterministic": deterministic,
                       "per_prompt": [row.to_json() for row in floor]}
    if not deterministic:
        reasons.append(f"{n2} is not self-deterministic: {n2b} differs from it in a token")
        return result
    if index_holds is False:
        reasons.append(f"{t3}'s first divergence is before {p}'s on two prompts or more")
    if mean_holds is False:
        reasons.append(f"{t3}'s mean logprob difference is not below {p}'s on two prompts or more")
    if index_holds is False or mean_holds is False:
        result["status"] = "FAIL"
    elif index_holds is True and mean_holds is True:
        result["status"] = "PASS"
    else:
        reasons.append("fewer than two prompts decide a rule (no common prefix with the reference)")
    return result


# -------------------------------------------------------------- capture (I/O)

def harness_constants() -> tuple[str, tuple[str, ...]]:
    """MODEL and TPUT_PROMPTS of ab_t1_v2.py, read from its source text."""
    if not HARNESS.is_file():
        raise Inconclusive(f"input absent: {HARNESS}")
    found = {}
    for node in ast.parse(HARNESS.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("MODEL", "TPUT_PROMPTS")):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    model, prompts = found["MODEL"], found["TPUT_PROMPTS"]
    if not isinstance(model, str) or len(prompts) != 3 or not all(isinstance(p, str) for p in prompts):
        raise ValueError(f"{HARNESS}: MODEL / TPUT_PROMPTS are not a name and three prompts")
    return model, tuple(prompts)


def requests_running(url: str) -> float:
    text = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    values = [float(line.split()[-1]) for line in text.splitlines()
              if line.startswith(RUNNING + "{") or line.startswith(RUNNING + " ")]
    if not values:
        raise Inconclusive(f"{RUNNING} is not in {url}/metrics")
    return sum(values)


def token_id(token: object) -> int:
    match = TOKEN_ID.fullmatch(token) if isinstance(token, str) else None
    if match is None:  # the token itself is text: never echoed
        raise RuntimeError("the server did not name a logprob token by id")
    return int(match.group(1))


def capture_prompt(url: str, model: str, pid: str, prompt: str) -> PromptCapture:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": MAX_TOKENS, "ignore_eos": True, "chat_template_kwargs": {"thinking": False},
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True,
            "logprobs": True, "top_logprobs": TOP, "return_tokens_as_token_ids": True}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    ids: list[int] = []
    chosen: list[tuple[int, float]] = []
    top_ids: list[tuple[int, ...]] = []
    top_lps: list[tuple[float, ...]] = []
    usage, finish = None, None
    with urllib.request.urlopen(request, timeout=900) as reply:
        for raw in reply:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("error"):
                raise RuntimeError(f"stream error: {str(obj['error'])[:200]}")
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:  # the delta text is never kept
                ids.extend(int(t) for t in choice.get("token_ids") or [])
                for entry in (choice.get("logprobs") or {}).get("content") or []:
                    chosen.append((token_id(entry["token"]), float(entry["logprob"])))
                    top = entry["top_logprobs"]
                    top_ids.append(tuple(token_id(t["token"]) for t in top))
                    top_lps.append(tuple(float(t["logprob"]) for t in top))
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if usage is None:
        raise RuntimeError(f"{pid}: the stream carried no usage")
    counts = (len(ids), len(chosen), int(usage["completion_tokens"]))
    if counts != (MAX_TOKENS,) * 3 or finish != "length":
        raise RuntimeError(f"{pid}: token ids / logprobs / completion_tokens {counts}, finish "
                           f"{finish!r}; want {MAX_TOKENS} each and 'length'")
    if [t for t, _ in chosen] != ids:
        raise RuntimeError(f"{pid}: the logprob tokens are not the streamed token ids")
    if any(v <= NO_LOGPROB for _, v in chosen):
        raise RuntimeError(f"{pid}: the server gave no logprob for a chosen token")
    if any(len(t) != TOP for t in top_ids):
        raise RuntimeError(f"{pid}: a token does not carry {TOP} top logprobs")
    if any(t not in top for t, top in zip(ids, top_ids)):
        raise RuntimeError(f"{pid}: a greedy token is not among its own top {TOP}")
    return PromptCapture(pid, hashlib.sha256(prompt.encode()).hexdigest(), tuple(ids),
                         tuple(v for _, v in chosen), tuple(top_ids), tuple(top_lps))


def capture(url: str, arm: str, directory: Path) -> dict:
    path = receipt_path(directory, arm)
    if path.exists():
        raise FileExistsError(f"{path} exists: choose another arm label")
    model, prompts = harness_constants()
    done = []
    for i, prompt in enumerate(prompts):
        running = requests_running(url)
        if running != 0:
            raise Inconclusive(f"{RUNNING} is {running:g} before request p{i}: the server is not idle")
        done.append(capture_prompt(url, model, f"p{i}", prompt))
    receipt = {
        "check": "T3-quality-capture", "arm": arm, "url": url, "model": model,
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "request": {"temperature": 0, "max_tokens": MAX_TOKENS, "ignore_eos": True,
                    "thinking": False, "stream": True, "return_token_ids": True,
                    "logprobs": True, "top_logprobs": TOP, "return_tokens_as_token_ids": True},
        "prompts": [p.to_json() for p in done],
    }
    with path.open("x") as out:
        out.write(json.dumps(receipt) + "\n")
    return {"check": "T3-quality-capture", "status": "WRITTEN", "arm": arm, "receipt": str(path),
            "bytes": path.stat().st_size,
            "prompts": [{"id": p.id, "tokens": len(p.token_ids), "tok_sha": p.to_json()["tok_sha"],
                         "mean_logprob": math.fsum(p.logprobs) / len(p.logprobs)} for p in done]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("url")
    cap.add_argument("arm")
    sco = sub.add_parser("score")
    sco.add_argument("n2")
    sco.add_argument("t3")
    sco.add_argument("p")
    sco.add_argument("n2b", nargs="?")
    for mode in (cap, sco):
        mode.add_argument("--dir", type=Path, default=RECEIPTS, help="directory of the receipts")
    a = parser.parse_args()
    if not a.dir.is_dir():
        parser.error(f"--dir {a.dir} is not a directory")
    if a.mode == "capture":
        try:
            print(json.dumps(capture(a.url.rstrip("/"), a.arm, a.dir), indent=1))
        except Inconclusive as exc:
            print(json.dumps({"check": "T3-quality-capture", "status": "INCONCLUSIVE",
                              "arm": a.arm, "reason": str(exc)}, indent=1))
            return 3
        return 0
    result = score(a.dir, a.n2, a.t3, a.p, a.n2b)
    text = json.dumps(result, indent=1) + "\n"
    (a.dir / f"quality-score-{a.n2}-{a.t3}-{a.p}.json").write_text(text)
    sys.stdout.write(text)
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
