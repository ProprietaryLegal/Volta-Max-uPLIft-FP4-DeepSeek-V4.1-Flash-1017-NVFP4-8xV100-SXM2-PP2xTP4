"""Rank 4 acceptance (campaign card 8.2): HOST routes per decode round.

Reads the counter snapshots a stats boot wrote (VLLM_NVFP4_8R_STATS=1) and
fails unless, on every text kind, at most 1.0 routes per decode round go to
a HOST expert, summed over the model's stages.

The limit comes from card 8.2, not from the engine: the goal is that general
and thinking-on text decode within one host fetch (2.3 ms/round) of
legal-style text, so a round may pay for at most one HOST route. Today's
tiers are expected to fail it on general text (card A4: 6.7 to 9.3, a
hypothesis until the stats boot measures it).

A text kind K is the difference of the snapshots labelled ``K-start`` and
``K-end`` (run/stats_mark.py writes the marks while the server is idle).

  PASS          every kind: HOST routes per round <= 1.0
  FAIL          a kind is above 1.0
  INCONCLUSIVE  (exit 3) a snapshot is absent; a pair comes from two boots;
                the four ranks of a stage disagree; layers or stages
                disagree on the number of rounds (a mark was taken while the
                server was working); or a kind has no rounds.

Prints and saves aggregates only: counts of rounds and routes, shares, and
per-round means. No per-expert value leaves the snapshot files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_stats as rs

LIMIT = 1.0  # HOST routes per decode round, whole model (card 8.2)
RANKS = 4
CALLS, ROUTES, HOST_ROUTES, HOST_GROUPS, WARM_ROUTES = range(5)


class Inconclusive(RuntimeError):
    pass


def stage_delta(directory: Path, kind: str, first_layer: int) -> dict[str, np.ndarray]:
    """The counters one stage added between K-start and K-end (rank 0's, after
    checking that the four ranks agree)."""
    per_rank = []
    for rank in range(RANKS):
        pair = []
        for end in ("start", "end"):
            path = rs.snapshot_path(directory, f"{kind}-{end}", first_layer, rank)
            if not path.is_file():
                raise Inconclusive(f"missing input: {path}")
            pair.append(rs.Snapshot.load(path))
        a, b = pair
        if a.boot != b.boot:
            raise Inconclusive(f"{kind} stage {first_layer} rank {rank}: start and end are from two boots")
        if not np.array_equal(a.tier, b.tier) or not np.array_equal(a.layers, b.layers):
            raise Inconclusive(f"{kind} stage {first_layer} rank {rank}: tiers differ between marks")
        delta = {k: getattr(b, k) - getattr(a, k) for k in ("stat", "gap_hist", "routes")}
        if any(bool((d < 0).any()) for d in delta.values()):
            raise Inconclusive(f"{kind} stage {first_layer} rank {rank}: a counter went down")
        delta["tier"], delta["recent_end"] = b.tier, b.recent
        per_rank.append(delta)
    for rank in range(1, RANKS):
        for key in ("stat", "gap_hist", "routes", "tier"):
            if not np.array_equal(per_rank[0][key], per_rank[rank][key]):
                raise Inconclusive(
                    f"{kind} stage {first_layer}: rank {rank} and rank 0 disagree on {key}"
                )
    return per_rank[0]


def kind_result(directory: Path, kind: str, stages: list[int]) -> dict:
    out: dict = {"stages": {}}
    rounds_seen = set()
    host_per_round = 0.0
    for first_layer in stages:
        d = stage_delta(directory, kind, first_layer)
        stat = d["stat"]
        calls = set(stat[:, CALLS].tolist())
        if len(calls) != 1:
            raise Inconclusive(f"{kind} stage {first_layer}: its layers count different rounds")
        rounds = calls.pop()
        if rounds <= 0:
            raise Inconclusive(f"{kind} stage {first_layer}: no decode round between the marks")
        # The per-expert counts must add up to the route counter and the tier
        # sums to the tier counters: the snapshot is consistent with itself.
        tier = d["tier"]
        by_tier = [int(d["routes"][tier == t].sum()) for t in (0, 1, 2)]
        totals = stat.sum(0)
        if sum(by_tier) != int(totals[ROUTES]) or by_tier[2] != int(totals[HOST_ROUTES]) \
                or by_tier[1] != int(totals[WARM_ROUTES]):
            raise Inconclusive(f"{kind} stage {first_layer}: route counts and tier counters disagree")
        rounds_seen.add(rounds)
        layers = stat.shape[0]
        host = int(totals[HOST_ROUTES])
        gaps = d["gap_hist"].sum(0)
        out["stages"][f"{first_layer}"] = {
            "layers": layers,
            "experts": {"hot": int((tier == 0).sum()), "warm": int((tier == 1).sum()),
                        "host": int((tier == 2).sum())},
            "rounds": rounds,
            "routes_per_round": round(int(totals[ROUTES]) / rounds, 3),
            "rows_per_round": round(int(totals[ROUTES]) / rounds / layers / 6, 3),
            "host_routes_per_round": round(host / rounds, 4),
            "host_experts_fetched_per_round": round(int(totals[HOST_GROUPS]) / rounds, 4),
            "route_share": {"hot": round(by_tier[0] / int(totals[ROUTES]), 6),
                            "warm": round(by_tier[1] / int(totals[ROUTES]), 6),
                            "host": round(by_tier[2] / int(totals[ROUTES]), 6)},
            # Plan 00 M3: hit rate of a cache holding every HOST expert used
            # in the last g rounds, and how many experts it would hold now.
            "host_cache_hit_rate": {
                str(g): (round(int(gaps[: rs.GAP_EDGES.index(g) + 1].sum()) / host, 4) if host else None)
                for g in rs.RECENT
            },
            "host_cache_experts_at_end": {
                str(g): int(d["recent_end"][:, i].sum()) for i, g in enumerate(rs.RECENT)
            },
        }
        host_per_round += host / rounds
    if len(rounds_seen) != 1:
        raise Inconclusive(f"{kind}: the stages count different rounds: {sorted(rounds_seen)}")
    out["host_routes_per_round"] = round(host_per_round, 4)
    out["within_limit"] = host_per_round <= LIMIT
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", required=True,
                        help="first layer of every pipeline stage, e.g. 0 20")
    parser.add_argument("--kinds", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    res: dict = {"check": "rank4-host-routes", "limit_host_routes_per_round": LIMIT,
                 "stages": a.stages, "kinds": {}}
    try:
        for kind in a.kinds:
            res["kinds"][kind] = kind_result(a.stats_dir, kind, a.stages)
    except Inconclusive as exc:
        res["verdict"], res["inconclusive"] = "INCONCLUSIVE", str(exc)
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        print(f"INCONCLUSIVE: {exc}")
        return 3
    over = [f"{k}: {v['host_routes_per_round']} HOST routes per round, limit {LIMIT}"
            for k, v in res["kinds"].items() if not v["within_limit"]]
    res["verdict"], res["fails"] = ("FAIL" if over else "PASS"), over
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    for kind, v in res["kinds"].items():
        print(kind, json.dumps({"host_routes_per_round": v["host_routes_per_round"], **{
            f"stage{s}": {k: st[k] for k in ("rounds", "rows_per_round", "host_routes_per_round",
                                             "route_share")}
            for s, st in v["stages"].items()}}))
    print(res["verdict"], json.dumps(over))
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
