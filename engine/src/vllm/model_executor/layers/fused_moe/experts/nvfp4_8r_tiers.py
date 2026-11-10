# SPDX-License-Identifier: Apache-2.0
"""NVFP4-8R three-tier residency: which record of an expert a card holds.

Pure (numpy only, no torch, no CUDA). One tier per expert of a pipeline
stage, identical on the stage's four TP ranks:

  HOT   native MXFP4 record in HBM          decode: QPN kernel
  WARM  production IQ record in HBM         decode: production IQ op
        (plus the native record in host RAM for prefill)
  HOST  native MXFP4 record in host RAM     decode: QPN kernel, zero-copy

``choose_tiers`` ranks by the private decode tally. Tier arrays and counts
stay in memory: callers log counts of experts and shares of routes only.
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

import numpy as np

NUM_EXPERTS = 384
EXPERT_BYTES = 4_874_240  # one native record of one TP rank
# Bytes of one 256-weight block per GGML type of the routed experts.
IQ_BLOCK_BYTES = {29: 56, 16: 66, 17: 74}  # IQ1_M, IQ2_XXS, IQ2_XS
# Blocks of one expert on one TP4 rank: gate and up are 576 rows x 20 blocks
# each, down (row split) is 1280 rows x 9 blocks.
_BLOCKS = 576 * 20
assert _BLOCKS == 1280 * 9 == 11520

RESIDENCY_KEY = "nvfp4_8r_residency"
HOST_SHARE_KEY = "nvfp4_8r_host_route_share"
MAX_HOST_ROUTE_SHARE = 0.05


class Tier(enum.IntEnum):
    HOT = 0
    WARM = 1
    HOST = 2


@dataclass(frozen=True)
class TwoTier:
    """Hot native experts in HBM, every other one zero-copy from host RAM."""


@dataclass(frozen=True)
class HotIq:
    """HOT and WARM only: no decode route leaves HBM."""


@dataclass(frozen=True)
class HotIqHost:
    """HOT, WARM, and a HOST tier holding at most ``route_share`` of the
    stage's decode routes (0.0: exactly the experts never routed to)."""

    route_share: float

    def __post_init__(self) -> None:
        s = self.route_share
        if type(s) is not float or not math.isfinite(s) or not 0.0 <= s <= MAX_HOST_ROUTE_SHARE:
            raise ValueError(
                f"{HOST_SHARE_KEY} must be a float in [0, {MAX_HOST_ROUTE_SHARE}], got {s!r}"
            )


Residency = TwoTier | HotIq | HotIqHost
WarmResidency = HotIq | HotIqHost

PREFILL_WARM_KEY = "nvfp4_8r_prefill_warm"


class PrefillWarm(enum.Enum):
    """Which record of a WARM expert prefill computes with (plan 04 part B)."""

    # Its native record, staged from host RAM per chunk (the build before B).
    NATIVE_STAGED = "native_staged"
    # Its IQ record in HBM, through production's IQ prefill: nothing staged.
    IQ = "iq"


def check_prefill_warm(mode: object, residency: Residency) -> PrefillWarm:
    """``mode`` if it is a legal prefill mode for ``residency``."""
    if type(mode) is not PrefillWarm:
        raise ValueError(f"{PREFILL_WARM_KEY}: {mode!r} is not a PrefillWarm")
    if mode is PrefillWarm.IQ and isinstance(residency, TwoTier):
        raise ValueError(
            f"{PREFILL_WARM_KEY} iq needs a WARM tier: {RESIDENCY_KEY} two_tier has none"
        )
    return mode


def parse_prefill_warm(extra: Mapping[str, object], residency: Residency) -> PrefillWarm:
    """The WARM prefill mode of model-loader-extra-config; no default."""
    names = [m.value for m in PrefillWarm]
    if PREFILL_WARM_KEY not in extra:
        raise ValueError(f"{PREFILL_WARM_KEY} is required ({' or '.join(names)})")
    raw = extra[PREFILL_WARM_KEY]
    if type(raw) is not str or raw not in names:
        raise ValueError(f"{PREFILL_WARM_KEY} must be {' or '.join(names)}, got {raw!r}")
    return check_prefill_warm(PrefillWarm(raw), residency)


def parse_residency(extra: Mapping[str, object]) -> Residency:
    """The residency of model-loader-extra-config; no defaults."""
    if RESIDENCY_KEY not in extra:
        raise ValueError(f"{RESIDENCY_KEY} is required (two_tier, hot_iq or hot_iq_host)")
    name = extra[RESIDENCY_KEY]
    has_share = HOST_SHARE_KEY in extra
    if name == "hot_iq_host":
        if not has_share:
            raise ValueError(f"{RESIDENCY_KEY} hot_iq_host needs {HOST_SHARE_KEY}")
        share = extra[HOST_SHARE_KEY]
        if type(share) is int:  # YAML writes 0 for 0.0
            share = float(share)
        return HotIqHost(share)  # type: ignore[arg-type]
    if name not in ("two_tier", "hot_iq"):
        raise ValueError(
            f"{RESIDENCY_KEY} must be two_tier, hot_iq or hot_iq_host, got {name!r}"
        )
    if has_share:
        raise ValueError(f"{HOST_SHARE_KEY} is only valid with {RESIDENCY_KEY} hot_iq_host")
    return TwoTier() if name == "two_tier" else HotIq()


def iq_record_bytes(type13: int, type2: int) -> int:
    """HBM bytes of one expert's IQ record on one TP4 rank (gate, up, down)."""
    for t in (type13, type2):
        if t not in IQ_BLOCK_BYTES:
            raise ValueError(f"routed-expert GGML type {t!r} is not IQ1_M/IQ2_XXS/IQ2_XS")
    return _BLOCKS * (2 * IQ_BLOCK_BYTES[type13] + IQ_BLOCK_BYTES[type2])


def _check_tier_array(tier: object, where: str) -> np.ndarray:
    if (
        not isinstance(tier, np.ndarray)
        or tier.dtype != np.int8
        or tier.shape != (NUM_EXPERTS,)
        or bool(((tier < 0) | (tier > 2)).any())
    ):
        raise ValueError(f"{where}: a tier array is int8 [{NUM_EXPERTS}] of Tier values")
    return tier


@dataclass(frozen=True)
class SlotMaps:
    """int32 [384] slot of each expert in each bank, -1 where absent. Slots
    are ranks in ascending expert id. ``cold`` is the host-RAM native set
    (every non-HOT expert); ``host`` is its HOST subset, same slot numbers."""

    hot: np.ndarray
    cold: np.ndarray
    host: np.ndarray
    iq: np.ndarray


def slot_maps(tier: np.ndarray) -> SlotMaps:
    tier = _check_tier_array(tier, "slot_maps")

    def ranks(mask: np.ndarray) -> np.ndarray:
        out = np.full(NUM_EXPERTS, -1, np.int32)
        out[mask] = np.arange(int(mask.sum()), dtype=np.int32)
        return out

    cold = ranks(tier != Tier.HOT)
    host = np.where(tier == Tier.HOST, cold, -1).astype(np.int32)
    return SlotMaps(ranks(tier == Tier.HOT), cold, host, ranks(tier == Tier.WARM))


@dataclass(frozen=True, eq=False)
class TierPlan:
    """The tiers of one pipeline stage (the same on its four TP ranks).

    ``hbm_bytes`` and ``digest`` are derived, so they cannot disagree with
    the arrays. A plan exists only for a residency with a WARM tier.
    """

    layers: range
    tiers: dict[int, np.ndarray]  # layer -> int8 [384] of Tier values
    iq_bytes: dict[int, int]  # layer -> IQ record bytes per expert per rank
    budget_bytes: int
    residency: WarmResidency

    def __post_init__(self) -> None:
        if not isinstance(self.residency, (HotIq, HotIqHost)):
            raise ValueError(f"a tier plan needs a WARM tier, got {self.residency!r}")
        if self.layers.step != 1 or len(self.layers) == 0:
            raise ValueError(f"stage layers must be a contiguous run, got {self.layers}")
        want = list(self.layers)
        if sorted(self.tiers) != want or sorted(self.iq_bytes) != want:
            raise ValueError(f"tier plan must cover exactly layers {self.layers}")
        frozen = {}
        for layer in self.layers:
            tier = _check_tier_array(self.tiers[layer], f"layer {layer}").copy()
            tier.setflags(write=False)
            frozen[layer] = tier
            iq = self.iq_bytes[layer]
            if type(iq) is not int or not 0 < iq < EXPERT_BYTES:
                raise ValueError(f"layer {layer}: IQ record bytes {iq!r}")
            if not bool((tier == Tier.HOT).any()):
                raise ValueError(f"layer {layer} has no HOT expert: the budget is too small")
            if not bool((tier == Tier.WARM).any()):
                raise ValueError(f"layer {layer} has no WARM expert")
            if isinstance(self.residency, HotIq) and bool((tier == Tier.HOST).any()):
                raise ValueError(f"layer {layer}: a hot_iq plan holds a HOST expert")
        object.__setattr__(self, "tiers", frozen)
        object.__setattr__(self, "iq_bytes", dict(self.iq_bytes))
        if self.hbm_bytes > self.budget_bytes:
            raise ValueError(
                f"tier plan uses {self.hbm_bytes} bytes of HBM, budget {self.budget_bytes}"
            )

    @cached_property
    def hbm_bytes(self) -> int:
        """Record bytes the plan puts on each card of the stage."""
        return sum(
            int((self.tiers[layer] == Tier.HOT).sum()) * EXPERT_BYTES
            + int((self.tiers[layer] == Tier.WARM).sum()) * self.iq_bytes[layer]
            for layer in self.layers
        )

    @cached_property
    def digest(self) -> str:
        """sha256 of the tier arrays in layer order: equal on all four ranks."""
        h = hashlib.sha256()
        for layer in self.layers:
            h.update(self.tiers[layer].tobytes())
        return h.hexdigest()

    def count(self, tier: Tier) -> int:
        return sum(int((self.tiers[layer] == tier).sum()) for layer in self.layers)


def choose_tiers(
    counts: np.ndarray,
    layers: range,
    iq_bytes: Mapping[int, int],
    budget_bytes: int,
    residency: WarmResidency,
) -> TierPlan:
    """Tiers of one stage from the decode tally and the per-card HBM budget.

    1. HOST: none for HotIq. For HotIqHost(s): the stage's experts ascending
       by (count, layer, expert), the longest prefix whose summed count is at
       most s x the stage's total count (s = 0: exactly the zero-count ones).
    2. Every other expert starts WARM; the budget must hold all their IQ records.
    3. Promote WARM to HOT in descending count / (native bytes - IQ bytes),
       ties ascending (layer, expert), while the next promotion fits; stop at
       the first that does not.
    """
    if not isinstance(residency, (HotIq, HotIqHost)):
        raise ValueError(f"choose_tiers needs a residency with a WARM tier, got {residency!r}")
    if counts.ndim != 2 or counts.shape[1] != NUM_EXPERTS or counts.dtype.kind not in "iu":
        raise ValueError(f"decode tally must be integer [layers, {NUM_EXPERTS}]")
    if layers.step != 1 or len(layers) == 0 or layers.start < 0 or layers.stop > counts.shape[0]:
        raise ValueError(f"stage layers {layers} are not a run inside the tally")
    if sorted(iq_bytes) != list(layers):
        raise ValueError(f"IQ record bytes must cover exactly layers {layers}")
    stage = [
        (int(counts[layer, e]), layer, e) for layer in layers for e in range(NUM_EXPERTS)
    ]
    if any(c < 0 for c, _, _ in stage):
        raise ValueError("decode tally has a negative count")
    tiers = {layer: np.full(NUM_EXPERTS, Tier.WARM, np.int8) for layer in layers}
    if isinstance(residency, HotIqHost):
        limit = Fraction(residency.route_share) * sum(c for c, _, _ in stage)
        running = 0
        for count, layer, e in sorted(stage):
            running += count
            if running > limit:
                break
            tiers[layer][e] = Tier.HOST
    need = sum(
        int((tiers[layer] == Tier.WARM).sum()) * iq_bytes[layer] for layer in layers
    )
    if need > budget_bytes:
        raise ValueError(
            f"budget {budget_bytes} cannot hold the IQ tier of layers "
            f"{layers.start}-{layers.stop - 1} ({need} bytes)"
        )
    cost = {layer: EXPERT_BYTES - iq_bytes[layer] for layer in layers}
    if any(c <= 0 for c in cost.values()):
        raise ValueError("an IQ record is not smaller than a native record")
    candidates = sorted(
        ((count, layer, e) for count, layer, e in stage if tiers[layer][e] == Tier.WARM),
        key=lambda c: (-Fraction(c[0], cost[c[1]]), c[1], c[2]),
    )
    for _, layer, e in candidates:
        if need + cost[layer] > budget_bytes:
            break
        need += cost[layer]
        tiers[layer][e] = Tier.HOT
    return TierPlan(layers, tiers, dict(iq_bytes), budget_bytes, residency)
