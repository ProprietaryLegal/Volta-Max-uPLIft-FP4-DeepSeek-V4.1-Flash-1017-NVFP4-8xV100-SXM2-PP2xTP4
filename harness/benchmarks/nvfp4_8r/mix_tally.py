"""Rank 4 tally tool (campaign card 8.2). CPU only; run on the 8-card host.

  from-stats  One tally file from the counter snapshots of a stats boot:
              the routes per expert that the named text kinds added between
              their K-start and K-end marks (decode calls only), summed over
              the kinds. Every check of accept_host_routes applies (four
              ranks agree, marks outside a round).
  mix         Mix named tallies (each normalised per layer; the weights are
              required, one per tally), write the mixed tally, and plan the
              tiers the server would choose from it with
              nvfp4_8r_host_route_share set by the rule: the largest share,
              to six places, at which the mix sends at most
              --host-routes-per-round routes of a full decode round (36 per
              layer) to HOST experts in a stage. Prints, for today's tiers
              (the profile as it is) and for the mix's tiers, the tier counts
              and, per text kind, the share of its routes in HOT / WARM /
              HOST and the HOST routes per round that predicts per stage.
              The planner is the deployed one, reading the written file.

Rule checked by ``mix`` (exit 1 if not met): by the mixed tally itself each
stage is at or under the budget. Reported, not a rule here: kinds whose
predicted HOST routes per round, summed over the stages, exceed the
acceptance limit of accept_host_routes (1.0); the stats boot of the mix arm
decides that.

Data boundary: tally files hold per-expert counts derived from text and are
written only beside the profile's tally (its private directory). This tool
prints and saves counts of experts, shares of routes, per-round means and
12-hex digest prefixes only. Exit 0 / 1 / 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tally_mix as tm
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

from . import accept_host_routes as ahr

LIMIT = ahr.LIMIT


class Inconclusive(RuntimeError):
    pass


def read_tally(path: Path) -> np.ndarray:
    if not path.is_file():
        raise Inconclusive(f"missing input: {path}")
    with np.load(path, allow_pickle=False) as data:
        phases = [str(p) for p in data["phases"]]
        if native.TALLY_PHASE not in phases:
            raise Inconclusive(f"{path} has no {native.TALLY_PHASE} phase")
        return np.asarray(data["counts"][phases.index(native.TALLY_PHASE)])


def write_tally(path: Path, counts: np.ndarray, meta: dict, private: Path) -> None:
    if path.resolve().parent != private.resolve():
        raise Inconclusive(f"a tally is written only into {private}, not {path.parent}")
    if path.exists():
        raise Inconclusive(f"{path} exists; name a new file")
    with open(path, "xb") as f:
        np.savez(f, phases=np.asarray([native.TALLY_PHASE]), counts=counts[None].astype(np.int64),
                 meta=np.asarray(json.dumps(meta)))


def pairs(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not value or name in out:
            raise Inconclusive(f"{what} is NAME=VALUE, once per name, got {item!r}")
        out[name] = value
    return out


def from_stats(a) -> int:
    private = a.stats_dir.resolve().parent
    counts = None
    rounds = {}
    for kind in a.kinds:
        rows = []
        for first in a.stages:
            d = ahr.stage_delta(a.stats_dir, kind, first)
            calls = set(d["stat"][:, ahr.CALLS].tolist())
            if len(calls) != 1 or min(calls) <= 0:
                raise ahr.Inconclusive(f"{kind} stage {first}: no single count of rounds between the marks")
            rounds[f"{kind}/{first}"] = calls.pop()
            rows.append(d["routes"])
        kind_counts = np.concatenate(rows)
        counts = kind_counts if counts is None else counts + kind_counts
    assert counts is not None
    meta = {"source": "stats boot counters", "kinds": a.kinds, "rounds": rounds}
    write_tally(a.write, counts, meta, private)
    res = {"check": "rank4-tally-from-stats", "kinds": a.kinds, "rounds": rounds, "layers": int(counts.shape[0]),
           "routes": int(counts.sum()), "experts_never_routed": int((counts == 0).sum()),
           "file": str(a.write), "bytes": a.write.stat().st_size}
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0


def mix(a) -> int:
    if not a.profile.is_file():
        raise Inconclusive(f"missing input: {a.profile}")
    doc = yaml.safe_load(a.profile.read_text())
    extra = json.loads(json.dumps(doc["model-loader-extra-config"]))  # string keys, as the server passes them
    tokens = doc["kernel-config"]["sm70_gguf"]["expert_cache_prefill_tokens"]
    cfg_today = native.NativePackConfig.from_extra(extra)
    private = cfg_today.tally.parent
    free = {int(k): int(v) for k, v in pairs(a.free, "--free").items()}
    firsts = sorted(cfg_today.reserve_by_stage)
    if sorted(free) != firsts:
        raise Inconclusive(f"--free names stages {sorted(free)}, the profile has {firsts}")
    files = {name: Path(p) for name, p in pairs(a.tally, "--tally").items()}
    weights = {name: float(w) for name, w in pairs(a.weight, "--weight").items()}
    if sorted(weights) != sorted(files):
        raise Inconclusive(f"--weight names {sorted(weights)}, --tally names {sorted(files)}")
    names = sorted(files)
    tallies = {name: read_tally(files[name]) for name in names}
    n_layers = {t.shape[0] for t in tallies.values()}
    if len(n_layers) != 1:
        raise Inconclusive(f"the tallies cover different layer counts: {sorted(n_layers)}")
    total_layers = n_layers.pop()
    stages = {first: range(first, nxt) for first, nxt in zip(firsts, [*firsts[1:], total_layers])}
    mixed = tm.mix_tallies([tallies[n] for n in names], [weights[n] for n in names])
    shares = {tm.host_route_share(a.host_routes_per_round, len(r)) for r in stages.values()}
    if len(shares) != 1:
        raise Inconclusive(f"the stages need different shares {sorted(shares)}; the engine takes one")
    share = shares.pop()
    write_tally(a.write, mixed, {"source": "mix", "tallies": names, "weights": weights,
                                 "scale": tm.MIX_SCALE}, private)
    cfg_mix = native.NativePackConfig.from_extra({
        **extra, "nvfp4_8r_tally": str(a.write), tiers.RESIDENCY_KEY: "hot_iq_host",
        tiers.HOST_SHARE_KEY: share,
        # A tier plan does not depend on the IQ tail; the key is required with a WARM tier.
        native.iqt.IQ_TAIL_KEY: "torch",
    })

    from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index

    index = gguf_tensor_index(gguf_shard_paths(Path(doc["model"])))
    tensor_names = {layer: {"w1": f"blk.{layer}.ffn_gate_exps.weight", "w3": f"blk.{layer}.ffn_up_exps.weight",
                            "w2": f"blk.{layer}.ffn_down_exps.weight"} for layer in range(total_layers)}
    sources = native.iq_layer_sources(index, tensor_names, 0)
    kinds = {**tallies, "mix": mixed}
    res: dict = {
        "check": "rank4-tally-mix", "profile": str(a.profile), "free_bytes": free,
        "tallies": {n: {"file": str(files[n]), "weight": weights[n], "routes": int(tallies[n].sum())}
                    for n in names},
        "rule": {"host_routes_per_round_per_stage": a.host_routes_per_round,
                 "routes_per_layer_per_round": tm.DECODE_ROUTES_PER_LAYER,
                 tiers.HOST_SHARE_KEY: share},
        "mixed_tally": {"file": str(a.write), "bytes": a.write.stat().st_size},
        "accept_limit_host_routes_per_round": LIMIT, "plans": {},
    }
    fails: list[str] = []
    for label, cfg in (("today", cfg_today), ("mix", cfg_mix)):
        entry: dict = {"residency": str(cfg.residency), "stages": {}, "host_routes_per_round": {}}
        totals = dict.fromkeys(kinds, 0.0)
        for first, layers in stages.items():
            src = {layer: sources[layer] for layer in layers}
            stage = native.plan_tiers(cfg, list(layers), free[first],
                                      {k: s.record_bytes for k, s in src.items()},
                                      native.warm_prefill_bytes(cfg, src, tokens))
            plan = stage.plan
            per_kind = {}
            for kind, tally in kinds.items():
                hot, warm, host = tm.tier_route_shares(tally, plan.tiers, layers)
                per_round = tm.host_routes_per_round(host, len(layers))
                totals[kind] += per_round
                per_kind[kind] = {"route_share": {"hot": round(hot, 6), "warm": round(warm, 6),
                                                  "host": round(host, 6)},
                                  "host_routes_per_round": round(per_round, 4)}
                if label == "mix" and kind == "mix" and per_round > a.host_routes_per_round:
                    fails.append(f"layers {layers.start}-{layers.stop - 1}: the mix predicts {per_round:.4f} "
                                 f"HOST routes per round, budget {a.host_routes_per_round}")
            entry["stages"][str(first)] = {
                "layers": len(layers), "budget_bytes": plan.budget_bytes, "record_bytes": plan.hbm_bytes,
                "experts": {"hot": plan.count(tiers.Tier.HOT), "warm": plan.count(tiers.Tier.WARM),
                            "host": plan.count(tiers.Tier.HOST)},
                "digest12": plan.digest[:12], "kinds": per_kind,
            }
        entry["host_routes_per_round"] = {k: round(v, 4) for k, v in totals.items()}
        entry["kinds_over_accept_limit"] = sorted(k for k, v in totals.items() if k != "mix" and v > LIMIT)
        res["plans"][label] = entry
    res["fails"], res["verdict"] = fails, ("FAIL" if fails else "PASS")
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(f"rule: {tiers.HOST_SHARE_KEY} = {share} (at most {a.host_routes_per_round} HOST routes per round "
          f"per stage at {tm.DECODE_ROUTES_PER_LAYER} routes per layer)")
    for label, entry in res["plans"].items():
        for first, st in entry["stages"].items():
            e = st["experts"]
            print(f"{label:5s} stage {first:>2s}: {e['hot']} hot / {e['warm']} warm / {e['host']} host, "
                  f"digest {st['digest12']}")
            for kind, v in st["kinds"].items():
                s = v["route_share"]
                print(f"        {kind:12s} route share {s['hot']:.4f} / {s['warm']:.4f} / {s['host']:.6f}; "
                      f"HOST routes per round {v['host_routes_per_round']}")
        print(f"{label:5s} HOST routes per round, all stages: {json.dumps(entry['host_routes_per_round'])}; "
              f"over the acceptance limit {LIMIT}: {entry['kinds_over_accept_limit']}")
    print(res["verdict"], json.dumps(fails))
    return 1 if fails else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("from-stats")
    s.add_argument("--stats-dir", type=Path, required=True)
    s.add_argument("--stages", type=int, nargs="+", required=True)
    s.add_argument("--kinds", nargs="+", required=True)
    s.add_argument("--write", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("mix")
    m.add_argument("--profile", type=Path, required=True)
    m.add_argument("--free", nargs="+", required=True, metavar="FIRST_LAYER=BYTES",
                   help="the planner's free bytes per stage, from a boot's loader lines")
    m.add_argument("--tally", action="append", required=True, metavar="NAME=FILE")
    m.add_argument("--weight", action="append", required=True, metavar="NAME=WEIGHT")
    m.add_argument("--host-routes-per-round", type=float, required=True)
    m.add_argument("--write", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    try:
        return from_stats(a) if a.command == "from-stats" else mix(a)
    except (Inconclusive, ahr.Inconclusive) as exc:
        print(f"INCONCLUSIVE: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
