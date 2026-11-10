"""Shared definitions of the NVFP4 closeness test.

The plan format, the keyed token hash, the receipt format, the request constants, the time model and the
pure metric functions live here. Nothing in this module talks to a server or the network.

Why decode positions: on this engine a forward pass of 1 to 6 rows runs the routed experts through the decode
path (on the hot_iq build: hot experts from NVFP4 records, the rest from compressed IQ records), and a pass of
more rows runs the prompt-reading path, which on every NVFP4 build computes every routed expert from its NVFP4
record. The first token after a prompt comes from the prompt-reading pass. So the token at position 0 cannot show
the compressed records, and the test compares positions 1 to G-1 of a greedy continuation, at every position where
the text so far is identical in both builds. Position 0 is reported separately as a control.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PLAN_FORMAT = "nvfp4-closeness-plan-v1"
RECEIPT_FORMAT = "nvfp4-closeness-receipt-v1"
SCORE_FORMAT = "nvfp4-closeness-score-v1"

KINDS = ("code", "prose", "legal", "structured", "chat")
LETTERS = "ABCD"
TOP_K = 20  # the engines' max_logprobs (vLLM default 20); every position carries exactly 20
CONT_TOKENS = 32  # G: greedy tokens per context (ignore_eos, so every capture has exactly G)
MC_MAX_TOKENS = 8  # room for "Answer: X" and an end token
TEMPLATE_OVERHEAD = 8  # tokens the chat template adds around a user message (estimate; receipts hold the real count)
FLOOR_CONTEXTS = 50  # first 50 of the context order (10 per kind) are captured twice on the reference
FLOOR_MC = 50  # first 50 of the MC order (50 different subjects)
BRIDGE_CONTEXTS = 10  # a continuation part re-captures these first and must reproduce them exactly
MIN_CONTEXTS = 300  # fewer contexts compared in an arm: its context verdicts are INCONCLUSIVE
MIN_MC = 500  # fewer MC items compared in an arm: its MC verdicts are INCONCLUSIVE
NEAR_TIE_NATS = 0.5  # a disagreement is "near-tie" when the reference's top two are within this
IDENTICAL_LP_TOL = 1e-6  # two captures are identical when every hash is equal and every logprob within this
SURVIVAL_AT = (0, 1, 2, 4, 8, 16, 31)  # report the share of contexts identical through these positions
BOOT_B = 2000
BOOT_SEED = 20261110
PASSES = ("floor", "bridge", "main")

CONTEXT_INSTRUCTION = ("Continue the following text from exactly where it stops. Write only the continuation.\n\n")
MC_INSTRUCTION = ("The following is a multiple choice question about {subject}. Reply with the line "
                  "\"Answer: X\", where X is the letter of the correct option, and nothing else.\n\n")
MC_ANSWER = re.compile(r"Answer\s*:\s*\**\s*\(?\s*([ABCD])(?![A-Za-z])")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# ------------------------------------------------------------------------------------------- keyed token hash

def read_key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError(f"{path}: a key is 32 bytes, this one is {len(key)}")
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"{path}: the key must be readable by its owner only (chmod 600)")
    return key


def key_id(key: bytes) -> str:
    return sha256_bytes(b"key-id:" + key)[:16]


def tok_hash(key: bytes, token_id: int) -> str:
    """64-bit keyed hash of one token id. Without the key the receipts cannot be turned back into text."""
    if not 0 <= token_id < 2**32:
        raise ValueError(f"token id {token_id} out of range")
    return hmac.new(key, token_id.to_bytes(4, "big"), hashlib.sha256).hexdigest()[:16]


# ------------------------------------------------------------------------------------------- plan

@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    doc: dict

    @classmethod
    def load(cls, path: Path) -> "Plan":
        raw = path.read_bytes()
        doc = json.loads(raw)
        if doc.get("format") != PLAN_FORMAT:
            raise ValueError(f"{path}: format {doc.get('format')!r}, want {PLAN_FORMAT}")
        ids = [c["id"] for c in doc["contexts"]] + [m["id"] for m in doc["mc"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate item ids")
        order = doc["order"]
        for name, pool in (("main_contexts", "contexts"), ("floor_contexts", "contexts"),
                           ("bridge_contexts", "contexts"), ("main_mc", "mc"), ("floor_mc", "mc")):
            known = {x["id"] for x in doc[pool]}
            if not set(order[name]) <= known:
                raise ValueError(f"{path}: order {name} names items that are not in {pool}")
        if order["floor_contexts"] != order["main_contexts"][:FLOOR_CONTEXTS]:
            raise ValueError(f"{path}: the floor contexts must be the first {FLOOR_CONTEXTS} of the main order")
        if order["floor_mc"] != order["main_mc"][:FLOOR_MC]:
            raise ValueError(f"{path}: the floor MC items must be the first {FLOOR_MC} of the main order")
        if order["bridge_contexts"] != order["main_contexts"][:BRIDGE_CONTEXTS]:
            raise ValueError(f"{path}: the bridge contexts must be the first {BRIDGE_CONTEXTS} of the main order")
        if sorted(order["main_contexts"]) != sorted(known_ids(doc["contexts"])):
            raise ValueError(f"{path}: the main context order must hold every context once")
        if sorted(order["main_mc"]) != sorted(known_ids(doc["mc"])):
            raise ValueError(f"{path}: the main MC order must hold every MC item once")
        for c in doc["contexts"]:
            if "text" in c and sha256_text(c["text"]) != c["text_sha256"]:
                raise ValueError(f"{path}: context {c['id']} text does not match its sha256")
        for m in doc["mc"]:
            if "prompt" in m and sha256_text(m["prompt"]) != m["prompt_sha256"]:
                raise ValueError(f"{path}: MC item {m['id']} prompt does not match its sha256")
        return cls(path, sha256_bytes(raw), doc)

    def contexts(self) -> dict[str, dict]:
        return {c["id"]: c for c in self.doc["contexts"]}

    def mc(self) -> dict[str, dict]:
        return {m["id"]: m for m in self.doc["mc"]}


def known_ids(items: Iterable[dict]) -> list[str]:
    return [x["id"] for x in items]


def context_message(item: dict) -> str:
    """The user message of a context item: chat items are sent as they are, text items with the instruction."""
    return item["text"] if item["kind"] == "chat" else CONTEXT_INSTRUCTION + item["text"]


def mc_message(subject: str, question: str, choices: Sequence[str]) -> str:
    if len(choices) != 4:
        raise ValueError(f"an MMLU item has 4 choices, got {len(choices)}")
    body = question.strip() + "\n" + "\n".join(f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(choices))
    return MC_INSTRUCTION.format(subject=subject.replace("_", " ")) + body


def parse_mc_answer(text: str) -> int | None:
    """Index of the letter in the first "Answer: X" of the generated text, or None (no answer)."""
    m = MC_ANSWER.search(text)
    return None if m is None else LETTERS.index(m.group(1))


# ------------------------------------------------------------------------------------------- time model

@dataclass(frozen=True)
class TimeModel:
    """Per-request time of one arm. Sources: logs of the lane (see DESIGN.md section 6)."""

    ttft_at_70: float  # first-token seconds on a prompt of about 70 tokens (measured)
    ttft_at_1k: float  # first-token seconds at 1,000 tokens
    ms_round: tuple[float, float]  # decode round, central and upper
    tok_round: tuple[float, float]  # tokens per decode round (with the drafter), central and upper
    overhead_s: float = 0.08  # metrics reads and HTTP

    def ttft(self, tokens: int, upper: bool) -> float:
        x = max(0.0, (tokens - 70) / 930.0)
        x = min(x, 1.0) if not upper else min(math.sqrt(x), 1.0)
        return self.ttft_at_70 + (self.ttft_at_1k - self.ttft_at_70) * x

    def request(self, prompt_tokens: int, out_tokens: int, upper: bool) -> float:
        i = 1 if upper else 0
        rounds = math.ceil(max(0, out_tokens - 1) / self.tok_round[i])
        return self.overhead_s + self.ttft(prompt_tokens, upper) + rounds * self.ms_round[i] / 1000.0


TIME_MODELS = {
    # two-tier NVFP4 reference: first token 0.62 to 0.70 s on the short sets, 80 to 215 ms per round (1522 MHz,
    # 2026-11-09); the clock ceiling is now 1425 MHz, so rounds are taken 7 % slower; about 5 s at 1K tokens.
    "reference": TimeModel(0.70, 5.0, (140.0, 230.0), (2.9, 2.4)),
    # hot_iq one slot (window 9b, 1425 MHz): first token 0.87 to 1.04 s, 35 to 40 ms per round.
    "hot_iq": TimeModel(1.00, 5.0, (37.0, 40.0), (2.9, 2.4)),
    # Volta sPLIt (production, 2026-11-09): first token 0.37 to 0.45 s, 22.5 ms per round, reads 2,600 tok/s.
    "volta_split": TimeModel(0.45, 0.85, (23.0, 25.0), (2.9, 2.4)),
}


# ------------------------------------------------------------------------------------------- captures

@dataclass(frozen=True)
class Pos:
    """One generated position: keyed hashes and logprobs only."""

    tok: str
    lp: float
    top: tuple[str, ...]
    toplp: tuple[float, ...]


@dataclass(frozen=True)
class ItemCapture:
    item: str
    pass_: str
    prompt_sha256: str
    prompt_tokens: int
    positions: tuple[Pos, ...]
    answer: int | None  # MC only
    answer_pos: int | None  # MC only: index of the letter token
    seq_sha: str  # keyed hash of the whole token sequence

    @classmethod
    def from_json(cls, d: dict) -> "ItemCapture":
        n = len(d["tok"])
        if not (len(d["lp"]) == len(d["top"]) == len(d["toplp"]) == n) or n == 0:
            raise ValueError(f"{d['item']}: hash and logprob lists differ in length")
        pos = []
        for i in range(n):
            if len(d["top"][i]) != TOP_K or len(d["toplp"][i]) != TOP_K:
                raise ValueError(f"{d['item']} position {i}: top list is not {TOP_K} long")
            pos.append(Pos(d["tok"][i], float(d["lp"][i]), tuple(d["top"][i]), tuple(float(v) for v in d["toplp"][i])))
        return cls(d["item"], d["pass"], d["prompt_sha256"], int(d["prompt_tokens"]), tuple(pos),
                   d.get("answer"), d.get("answer_pos"), d["seq_sha"])


def identical(a: ItemCapture, b: ItemCapture) -> bool:
    if a.prompt_sha256 != b.prompt_sha256 or len(a.positions) != len(b.positions):
        return False
    for x, y in zip(a.positions, b.positions):
        if x.tok != y.tok or x.top != y.top:
            return False
        if abs(x.lp - y.lp) > IDENTICAL_LP_TOL or any(abs(u - v) > IDENTICAL_LP_TOL for u, v in zip(x.toplp, y.toplp)):
            return False
    return a.answer == b.answer


# ------------------------------------------------------------------------------------------- metrics (pure)

def kl_both_top(ref: Pos, arm: Pos) -> tuple[float, float]:
    """KL(reference || arm) in nats over the tokens in BOTH top lists plus one bucket for every other token.

    Grouping tokens can only lower a KL divergence (data-processing inequality), so this is a lower bound on the
    full-vocabulary KL. Returns (kl, reference mass on the shared tokens)."""
    arm_lp = dict(zip(arm.top, arm.toplp))
    p_shared, q_shared, kl = 0.0, 0.0, 0.0
    for t, lp in zip(ref.top, ref.toplp):
        if t in arm_lp:
            p, q = math.exp(lp), math.exp(arm_lp[t])
            p_shared += p
            q_shared += q
            kl += p * (lp - arm_lp[t])
    p_rest, q_rest = max(1.0 - p_shared, 1e-12), max(1.0 - q_shared, 1e-12)
    kl += p_rest * math.log(p_rest / q_rest)
    return max(kl, 0.0), p_shared


def abs_dlp(ref: Pos, arm: Pos) -> tuple[float, bool]:
    """|arm logprob - reference logprob| of the reference's chosen token. When that token is not in the arm's
    top list its arm logprob is at most the arm's last listed one, which is used (censored, a lower bound)."""
    arm_lp = dict(zip(arm.top, arm.toplp))
    if ref.tok in arm_lp:
        return abs(arm_lp[ref.tok] - ref.lp), False
    return abs(min(arm.toplp) - ref.lp), True


@dataclass
class ContextResult:
    """One context, one arm against the reference."""

    item: str
    kind: str
    first_div: int  # first position whose token differs; G when none differs
    pos0_agree: bool
    pos0_absdlp: float
    pos0_kl: float
    n: int = 0  # decode positions compared: 1 .. min(first_div, G - 1)
    disagree: int = 0
    near_tie: int = 0
    sum_absdlp: float = 0.0
    censored: int = 0
    sum_kl: float = 0.0
    sum_shared: float = 0.0

    def to_json(self) -> dict:
        return dict(self.__dict__)


def compare_context(item: str, kind: str, ref: ItemCapture, arm: ItemCapture, upto: int | None = None) -> ContextResult:
    """Compare an arm with the reference on one context. upto limits the decode positions compared to
    1 .. upto (used for the positions both other arms share); None means 1 .. min(first_div, G - 1)."""
    if ref.prompt_sha256 != arm.prompt_sha256:
        raise ValueError(f"{item}: the two captures were made on different prompts")
    g = len(ref.positions)
    if len(arm.positions) != g:
        raise ValueError(f"{item}: {len(arm.positions)} positions, reference {g}")
    first = next((i for i in range(g) if ref.positions[i].tok != arm.positions[i].tok), g)
    d0, _ = abs_dlp(ref.positions[0], arm.positions[0])
    k0, _ = kl_both_top(ref.positions[0], arm.positions[0])
    res = ContextResult(item, kind, first, first > 0, d0, k0)
    last = min(first, g - 1) if upto is None else min(upto, first, g - 1)
    for p in range(1, last + 1):
        r, a = ref.positions[p], arm.positions[p]
        res.n += 1
        if r.tok != a.tok:
            res.disagree += 1
            if r.toplp[0] - r.toplp[1] < NEAR_TIE_NATS:
                res.near_tie += 1
        d, cens = abs_dlp(r, a)
        res.sum_absdlp += d
        res.censored += int(cens)
        k, shared = kl_both_top(r, a)
        res.sum_kl += k
        res.sum_shared += shared
    return res


# ------------------------------------------------------------------------------------------- bootstrap

def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted list (numpy's default method)."""
    if not sorted_vals:
        raise ValueError("percentile of nothing")
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def strata_indices(strata: Sequence[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    return [groups[k] for k in sorted(groups)]


def bootstrap(stat, n_items: int, strata: Sequence[str] | None, b: int = BOOT_B, seed: int = BOOT_SEED):
    """Percentile 95 % interval of stat(indices) over b resamples of the items (whole contexts or MC items,
    with replacement, within each stratum when strata are given). stat may return None (undefined on a
    resample); such resamples are counted and the interval is None when more than 1 % are undefined."""
    import numpy as np

    rng = np.random.default_rng(seed)
    groups = strata_indices(strata) if strata is not None else [list(range(n_items))]
    point = stat(np.arange(n_items))
    vals, undefined = [], 0
    for _ in range(b):
        idx = np.concatenate([rng.choice(np.asarray(g), size=len(g), replace=True) for g in groups])
        v = stat(idx)
        if v is None:
            undefined += 1
        else:
            vals.append(float(v))
    vals.sort()
    if not vals or undefined > b // 100:
        return point, None, undefined
    return point, (percentile(vals, 0.025), percentile(vals, 0.975)), undefined


def read_receipts(paths: Sequence[Path], plan: Plan, key_ident: str) -> tuple[dict, list[dict]]:
    """Every item line of an arm's part files, keyed by (pass, item). Headers must name this plan and key, one
    arm; an item captured in two parts must be identical in both. Returns (items, headers)."""
    items: dict[tuple[str, str], ItemCapture] = {}
    headers = []
    arm = None
    for path in paths:
        lines = path.read_text().splitlines()
        if not lines:
            raise ValueError(f"{path}: empty receipt")
        head = json.loads(lines[0])
        if head.get("type") != "header" or head.get("format") != RECEIPT_FORMAT:
            raise ValueError(f"{path}: first line is not a {RECEIPT_FORMAT} header")
        if head["plan_sha256"] != plan.sha256:
            raise ValueError(f"{path}: made on plan {head['plan_sha256'][:16]}, this plan is {plan.sha256[:16]}")
        if head["key_id"] != key_ident:
            raise ValueError(f"{path}: made with key {head['key_id']}, this key is {key_ident}")
        if arm is not None and head["arm"] != arm:
            raise ValueError(f"{path}: arm {head['arm']}, earlier parts {arm}")
        arm = head["arm"]
        headers.append(head)
        for line in lines[1:]:
            d = json.loads(line)
            if d["type"] != "item":
                continue
            cap = ItemCapture.from_json(d)
            key = (cap.pass_, cap.item)
            if key in items and not identical(items[key], cap):
                raise ValueError(f"{path}: {key} differs from its capture in an earlier part")
            items.setdefault(key, cap)
    return items, headers
