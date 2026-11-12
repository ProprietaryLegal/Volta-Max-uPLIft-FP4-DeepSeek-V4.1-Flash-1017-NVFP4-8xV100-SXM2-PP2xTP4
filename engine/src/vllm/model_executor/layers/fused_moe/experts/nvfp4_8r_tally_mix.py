"""Mixing decode tallies and sizing the HOST tier by a route budget (campaign
card 8.2). Pure functions: arrays in, arrays and numbers out; no file, no
device, no log line.

A tally is integer counts [layers, 384] of decode routes per expert. The
existing one derives from one kind of text, so experts other text routes to
can land in the HOST tier and cost a host fetch in the decode round. A mix
gives every kind of text the same say in each layer: each tally is first
normalised per layer (so a tally with more rounds does not outweigh one with
fewer), then the tallies are added with the caller's weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

MIX_SCALE = 1 << 40  # counts of a mixed layer add up to this (before rounding)
DECODE_ROUTES_PER_LAYER = 36  # six rows of six routes: a full decode round
SHARE_DECIMALS = 6


def _check_tally(tally: object, where: str) -> np.ndarray:
    if (
        not isinstance(tally, np.ndarray)
        or tally.ndim != 2
        or tally.shape[1] != tiers.NUM_EXPERTS
        or tally.dtype.kind not in "iu"
    ):
        raise ValueError(f"{where}: a tally is integer [layers, {tiers.NUM_EXPERTS}]")
    if bool((tally < 0).any()):
        raise ValueError(f"{where}: a tally has a negative count")
    if bool((tally.sum(1) <= 0).any()):
        raise ValueError(f"{where}: a layer of the tally has no route")
    return tally


def mix_tallies(tallies: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """int64 [layers, 384]: sum over tallies of weight x (count / the tally's
    layer total), scaled so that a layer adds up to MIX_SCALE. Weights are
    required, one per tally, each finite and positive; only their ratios
    matter. An expert some tally routes to never rounds to zero."""
    if len(tallies) == 0 or len(tallies) != len(weights):
        raise ValueError(f"{len(tallies)} tallies need as many weights, got {len(weights)}")
    for w in weights:
        if type(w) not in (int, float) or not math.isfinite(w) or w <= 0:
            raise ValueError(f"a mix weight is a finite positive number, got {w!r}")
    checked = [_check_tally(t, f"tally {i}") for i, t in enumerate(tallies)]
    if len({t.shape for t in checked}) != 1:
        raise ValueError(f"the tallies cover different layers: {[t.shape for t in checked]}")
    total = float(sum(weights))
    mix = np.zeros(checked[0].shape, np.float64)
    for tally, w in zip(checked, weights):
        mix += (w / total) * (tally / tally.sum(1, keepdims=True, dtype=np.float64))
    out = np.rint(mix * MIX_SCALE).astype(np.int64)
    if bool(((out == 0) != (mix == 0)).any()):
        raise ValueError("a routed expert rounded to zero: the tallies are too large for MIX_SCALE")
    return out


def host_route_share(host_routes_per_round: float, stage_layers: int) -> float:
    """The largest nvfp4_8r_host_route_share, to SHARE_DECIMALS places, at
    which a stage of ``stage_layers`` layers sends at most
    ``host_routes_per_round`` routes of a full decode round (36 per layer) to
    HOST experts by the tally it is planned from."""
    if type(stage_layers) is not int or stage_layers <= 0:
        raise ValueError(f"a stage has a positive number of layers, got {stage_layers!r}")
    if (
        type(host_routes_per_round) not in (int, float)
        or not math.isfinite(host_routes_per_round)
        or host_routes_per_round < 0
    ):
        raise ValueError(f"a route budget is a finite number >= 0, got {host_routes_per_round!r}")
    unit = 10**SHARE_DECIMALS
    share = math.floor(
        host_routes_per_round * unit / (DECODE_ROUTES_PER_LAYER * stage_layers)
    ) / unit
    if share > tiers.MAX_HOST_ROUTE_SHARE:
        raise ValueError(
            f"{host_routes_per_round} routes per round is a share of {share}, above the "
            f"engine's limit {tiers.MAX_HOST_ROUTE_SHARE}"
        )
    return share


def tier_route_shares(
    tally: np.ndarray, plan: Mapping[int, np.ndarray], layers: range
) -> tuple[float, float, float]:
    """Mean over the stage's layers of the share of the tally's routes in that
    layer that go to HOT, WARM and HOST experts of ``plan`` (layer -> tier
    array). Every layer runs once per round, so the layers weigh equally."""
    _check_tally(tally, "tally")
    if sorted(plan) != list(layers) or layers.stop > tally.shape[0]:
        raise ValueError(f"the plan and the tally do not both cover layers {layers}")
    shares = np.zeros(3, np.float64)
    for layer in layers:
        row = tally[layer]
        for t in (tiers.Tier.HOT, tiers.Tier.WARM, tiers.Tier.HOST):
            shares[t] += int(row[plan[layer] == t].sum()) / int(row.sum())
    shares /= len(layers)
    return float(shares[0]), float(shares[1]), float(shares[2])


def host_routes_per_round(host_share: float, stage_layers: int) -> float:
    """HOST routes of one full decode round in a stage whose layers send
    ``host_share`` of their routes to HOST experts."""
    return host_share * DECODE_ROUTES_PER_LAYER * stage_layers
