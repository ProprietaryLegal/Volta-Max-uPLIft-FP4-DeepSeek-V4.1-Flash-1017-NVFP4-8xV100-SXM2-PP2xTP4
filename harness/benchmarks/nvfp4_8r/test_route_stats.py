"""Rank 4 counters (campaign card 8.2), CPU test.

CPU only. The device kernels of ``forward`` (the QPN op, the Triton reduce,
production's IQ op) have no CPU path and are replaced by stand-ins, so this
tests the Python side: what the counters count, that the switch changes
nothing else, and the snapshot files. It does not test that the counter
operations capture and replay in a CUDA graph: that needs a card.

  a  Counters. With VLLM_NVFP4_8R_STATS=1, three layers with different tier
     maps run 700 decode calls (M = 1..6, seeded routing that reaches HOST
     experts rarely, sometimes or often by layer). Every counter must equal a reference kept route by
     route in plain Python from the tier map (calls, routes, HOST routes,
     distinct HOST experts per call, WARM routes, routes per expert, rounds
     since last use in the seven buckets, experts used in the last 4, 16, 64,
     256 rounds). The reference must fill every bucket, else INCONCLUSIVE.
     A snapshot saved and loaded returns the same arrays.
  b  Switch. With the switch off a layer has no counters, and ``forward``
     (M = 1..6) and ``forward_prefill`` (M = 7, 64) give the same call log
     and result bits as the module before this change (--reference); with
     the switch on, the same call log and bits again. The reference must
     repeat itself.
  c  Writer. A mark already in the directory at start is not answered; a new
     mark is answered once, with one file named for the label, stage and
     rank; a label with other characters is refused.
  d  Acceptance script, as it will be used. Two stages of 20 layers, four
     ranks each, 5 % of each layer's experts HOST. Kind "uniform": uniform
     routing; kind "nohost": routing that avoids HOST experts. The script
     must exit 1 on "uniform" with HOST routes per round equal to this
     test's own count of them, exit 0 on "nohost" (0 per round), and exit 3
     when a snapshot is absent, when one rank's file disagrees, and when a
     mark falls inside a round.

Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

BASE = "vllm.model_executor.layers.fused_moe.experts.nvfp4_8r_native"
HIDDEN, TOP_K, EXPERTS = 5120, 6, 384
HOT, WARM, HOST = 0, 1, 2
EDGES = (1, 2, 4, 16, 64, 256)
RECENT = (4, 16, 64, 256)


class Inconclusive(RuntimeError):
    pass


def load(path: Path, want_sha: str, tag: str, stats: str | None):
    """The module file under its own name, read with the switch set to ``stats``."""
    if not path.is_file():
        raise Inconclusive(f"missing input: {path}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(want_sha) < 16 or not sha.startswith(want_sha):
        raise Inconclusive(f"{path} is {sha[:16]}, not {want_sha[:16]}")
    env = dict(os.environ)
    env.pop("VLLM_NVFP4_8R_STATS", None)
    if stats is not None:
        env["VLLM_NVFP4_8R_STATS"] = stats
    spec = importlib.util.spec_from_file_location(f"{BASE}_{tag}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with mock.patch.dict(os.environ, env, clear=True):
        spec.loader.exec_module(module)
    return module


def placeholder_set(nat, n: int):
    from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import QpnFragmentOrder

    z = torch.zeros((n, 1), dtype=torch.uint8)
    g = torch.ones(n, dtype=torch.float32)
    return nat.NativeExpertSet(z, z, z, z, g, g, QpnFragmentOrder.M8N8K4)


def three_tier(nat, tier: np.ndarray):
    n_hot, n_warm = int((tier == HOT).sum()), int((tier == WARM).sum())
    # Real shapes without memory: the stand-ins never read the bank.
    w13 = torch.empty((1, 576, 20 * 74), dtype=torch.uint8).expand(n_warm, -1, -1)
    w2 = torch.empty((1, 1280, 9 * 74), dtype=torch.uint8).expand(n_warm, -1, -1)
    # --reference is the module before plan 03 part B and has no group_plan
    # (nor swiglu, CAMPAIGN 8.4); a module that has them gets the chains (the
    # behaviour this test pins).
    plan = {"group_plan": nat.gplan.GroupPlan.CHAIN} if hasattr(nat, "gplan") else {}
    if hasattr(nat, "swig"):
        plan["swiglu"] = nat.swig.Swiglu.CHAIN
    if hasattr(nat, "iqt"):
        plan["iq_tail"] = nat.iqt.IqTail.TORCH
    # D3 (prefill SwiGLU): a module whose swiglu module names the prefill key takes it as a required keyword.
    if hasattr(nat, "swig") and hasattr(nat.swig, "PREFILL_SWIGLU_KEY"):
        plan["prefill_swiglu"] = nat.swig.Swiglu.CHAIN
    # RB 4 patch: a module that imports nvfp4_8r_qpn_rb takes prefill_rb as a required keyword.
    if hasattr(nat, "qrb"):
        plan["prefill_rb"] = nat.qrb.PrefillRb.RB2
    return nat.NativeV41Moe.three_tier(
        placeholder_set(nat, n_hot), placeholder_set(nat, EXPERTS - n_hot),
        nat.IqWarmBank(w13, w13, w2, 17, 17), tier, max_tokens=6, prefill=None, **plan,
    )


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:12]


class StandIn:
    """The device kernels. ``write`` False: touch nothing, log nothing."""

    def __init__(self, write: bool) -> None:
        self.write = write
        self.log: list = []

    def qpn(self, xin, q, sc, g, perm, gids, goff, top_k, y, slot_major, m, split, one, order):
        if not self.write:
            return
        routes = perm[: int(goff.max())].long()
        src = xin if xin.size(0) == y.size(0) else xin.repeat_interleave(top_k, 0)
        rows = src[routes]
        self.log.append(("qpn", digest(rows), tuple(q.shape), digest(perm), digest(gids),
                         digest(goff), tuple(y.shape), slot_major, m, repr(split), order))
        ramp = torch.linspace(-1.0, 1.0, y.size(1))
        y[routes] = (rows.float().mean(1, keepdim=True) * 8.0 + ramp).to(y.dtype)

    def reduce(self, y2, weights, out):
        if not self.write:
            return
        # Rows of routes no launch wrote are multiplied by weight 0.0 here.
        m, k = weights.shape
        live = (weights != 0).reshape(-1)
        self.log.append(("reduce", digest(y2[live]), digest(weights), tuple(out.shape)))
        out.copy_((torch.where(live[:, None], y2.float(), 0.0).view(m, k, -1)
                   * weights[:, :, None]).sum(1))

    def iq(self, x, iq_ids, weights, w1, w3, w2, type13, type2, zero, limit):
        if not self.write:
            return torch.zeros((x.size(0), HIDDEN), dtype=torch.float16)
        self.log.append(("iq", digest(x), digest(iq_ids), digest(weights), type13, type2, zero, limit))
        share = ((iq_ids >= 0).float() * weights).sum(1, keepdim=True)
        return (x.float() * share * 0.125).to(torch.float16)

    def over(self, nat):
        return (mock.patch.object(torch.ops._C, "skinny_moe_qpn_sm70", self.qpn),
                mock.patch.object(nat, "_mtp_weighted_reduce", self.reduce),
                mock.patch.object(torch.ops.vllm, "gguf_v41_original_expert_dp4a", self.iq))


def run(nat, moe, stand: StandIn, call: str, x, ids, w):
    a, b, c = stand.over(nat)
    with a, b, c:
        return getattr(moe, call)(x, ids, w)


def tier_map(rng, p) -> np.ndarray:
    tier = rng.choice(3, EXPERTS, p=p).astype(np.int8)
    tier[:3] = (HOT, WARM, HOST)  # every tier is present
    return tier


def draw(rng, m: int, pools: list[np.ndarray], p_first: float) -> np.ndarray:
    """[m, 6] distinct experts per row; each route from pools[0] with
    probability ``p_first``, else from pools[1]."""
    rows = []
    for _ in range(m):
        row: list[int] = []
        while len(row) < TOP_K:
            pool = pools[0] if rng.random() < p_first else pools[1]
            e = int(rng.choice(pool))
            if e not in row:
                row.append(e)
        rows.append(row)
    return np.asarray(rows, np.int32)


def tensors(rng, ids: np.ndarray):
    m = ids.shape[0]
    w = rng.random((m, TOP_K)) + 0.05
    return (torch.from_numpy(rng.standard_normal((m, HIDDEN)).astype(np.float16)),
            torch.from_numpy(ids), torch.from_numpy((w / w.sum(1, keepdims=True)).astype(np.float32)))


class Reference:
    """The counters of one layer, kept route by route."""

    def __init__(self, tier: np.ndarray) -> None:
        self.tier = tier
        self.stat = [0, 0, 0, 0, 0]
        self.hist = [0] * 7
        self.routes = np.zeros(EXPERTS, np.int64)
        self.last: dict[int, int] = {}

    def count(self, ids: np.ndarray) -> None:
        r = self.stat[0]
        flat = [int(e) for e in ids.reshape(-1)]
        hosts = [e for e in flat if self.tier[e] == HOST]
        for e in hosts:
            if e not in self.last:
                self.hist[6] += 1
            else:
                gap = r - self.last[e]
                self.hist[next((i for i, edge in enumerate(EDGES) if gap <= edge), 6)] += 1
        for e in set(hosts):
            self.last[e] = r
        for e in flat:
            self.routes[e] += 1
        self.stat[0] += 1
        self.stat[1] += len(flat)
        self.stat[2] += len(hosts)
        self.stat[3] += len(set(hosts))
        self.stat[4] += sum(self.tier[e] == WARM for e in flat)

    def recent(self) -> list[int]:
        r = self.stat[0]
        return [sum(1 for last in self.last.values() if r - last <= g) for g in RECENT]


def part_a(on, rs, fails: list, unsure: list) -> dict:
    rng = np.random.default_rng(8204)
    tiers = [tier_map(rng, p) for p in ((0.55, 0.40, 0.05), (0.30, 0.60, 0.10), (0.80, 0.18, 0.02))]
    layers = [three_tier(on, t) for t in tiers]
    refs = [Reference(t) for t in tiers]
    stand = StandIn(False)
    for moe in layers:
        if moe.stats is None:
            fails.append("a: the switch is on and a layer has no counters")
            return {}
    for _ in range(700):
        m = int(rng.integers(1, 7))
        # HOST experts are reached rarely, sometimes and often, so every bucket fills.
        for moe, ref, tier, p_host in zip(layers, refs, tiers, (0.004, 0.05, 0.3)):
            ids = draw(rng, m, [np.flatnonzero(tier == HOST), np.flatnonzero(tier != HOST)], p_host)
            run(on, moe, stand, "forward", *tensors(rng, ids))
            ref.count(ids)
    filled = np.sum([r.hist for r in refs], 0)
    if (filled == 0).any():
        unsure.append(f"a: the reference left a bucket empty: {filled.tolist()}")
    for i, (moe, ref) in enumerate(zip(layers, refs)):
        got = moe.stats.read()
        want = {"stat": ref.stat, "gap_hist": ref.hist, "recent": ref.recent(), "routes": ref.routes}
        for key, value in want.items():
            if got[key].tolist() != np.asarray(value).tolist():
                shown = "" if key == "routes" else f": {got[key].tolist()} against {list(value)}"
                fails.append(f"a: layer {i} {key} differs from the reference{shown}")
    snap = rs.take_snapshot("a-end", "boot", 7, 2, {7 + i: m.stats for i, m in enumerate(layers)},
                            {7 + i: t for i, t in enumerate(tiers)})
    with tempfile.TemporaryDirectory() as tmp:
        back = rs.Snapshot.load(snap.save(Path(tmp)))
    same = all(np.array_equal(getattr(snap, k), getattr(back, k))
               for k in ("layers", "stat", "gap_hist", "recent", "routes", "tier"))
    if not same or (back.label, back.boot, back.first_layer, back.rank) != ("a-end", "boot", 7, 2):
        fails.append("a: a saved snapshot does not load back equal")
    return {"calls": 700, "layers": 3, "host_routes": [r.stat[2] for r in refs],
            "reference_buckets": filled.tolist()}


def part_b(on, off, ref, fails: list, unsure: list) -> dict:
    rng = np.random.default_rng(8205)
    tier = tier_map(rng, (0.55, 0.40, 0.05))
    hot_warm = tier.copy()
    hot_warm[tier == HOST] = WARM  # forward_prefill on CPU: no HOST route (those need a card)
    cases = 0
    for call, ms, t in (("forward", range(1, 7), tier), ("forward_prefill", (7, 64), hot_warm)):
        for m in ms:
            ids = draw(rng, m, [np.arange(EXPERTS), np.arange(EXPERTS)], 0.5)
            args = tensors(rng, ids)
            got = []
            for nat in (ref, ref, off, on):
                moe = three_tier(nat, t)
                if call == "forward_prefill":
                    # WARM prefill from IQ records: a handle that is the IQ stand-in.
                    stand = StandIn(True)
                    handle = mock.Mock()
                    handle.partial = lambda x, i, w, s=stand: s.iq(x, i, w, None, None, None, 17, 17, 0, 0)
                    moe.warm = dataclasses.replace(moe.warm, prefill=handle)
                else:
                    stand = StandIn(True)
                out = run(nat, moe, stand, call, *args).clone()
                got.append((stand.log, out))
                if nat is off and moe.stats is not None:
                    fails.append("b: the switch is off and a layer has counters")
                if nat is on and moe.stats is None:
                    fails.append("b: the switch is on and a layer has no counters")
            (la, oa), (lb, ob), (lo, oo), (ln, on_out) = got
            cases += 1
            if la != lb or not torch.equal(oa, ob) or not bool(torch.isfinite(oa).all()) \
                    or float(oa.abs().sum()) == 0.0:
                unsure.append(f"b: {call} M={m}: the reference does not repeat itself")
            elif lo != la or not torch.equal(oo, oa):
                fails.append(f"b: {call} M={m}: switch off differs from the module before")
            elif ln != la or not torch.equal(on_out, oa):
                fails.append(f"b: {call} M={m}: switch on changes the calls or the result")
    return {"cases": cases}


def part_c(on, rs, fails: list) -> dict:
    rng = np.random.default_rng(8206)
    tier = tier_map(rng, (0.55, 0.40, 0.05))
    moe = three_tier(on, tier)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "stats"
        d.mkdir()
        (d / rs.MARK_FILE).write_text("old-boot\n")
        writer = rs.StatsWriter(d, 20, 3, {20: moe.stats}, {20: tier})
        if writer.poll() is not None or list(d.glob("snap-*")):
            fails.append("c: a mark from before the start was answered")
        (d / rs.MARK_FILE).write_text("general-start\n")
        first, second = writer.poll(), writer.poll()
        names = sorted(p.name for p in d.glob("snap-*"))
        if first is None or second is not None or names != ["snap-general-start-s20-r3.npz"]:
            fails.append(f"c: one new mark gave files {names}")
        (d / rs.MARK_FILE).write_text("Bad Label\n")
        try:
            writer.poll()
        except ValueError:
            pass
        else:
            fails.append("c: a label with other characters was accepted")
    return {"files": names}


def part_d(on, rs, accept_cmd: list[str], fails: list, unsure: list) -> dict:
    rng = np.random.default_rng(8207)
    stages = (0, 20)
    tiers = {s: [tier_map(rng, (0.55, 0.40, 0.05)) for _ in range(20)] for s in stages}
    built = {(s, r): [three_tier(on, t) for t in tiers[s]] for s in stages for r in range(4)}
    stand = StandIn(False)
    out: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        def mark(label: str) -> None:
            for (s, r), layers in built.items():
                rs.take_snapshot(label, f"boot-{s}-{r}", s, r,
                                 {s + i: m.stats for i, m in enumerate(layers)},
                                 {s + i: t for i, t in enumerate(tiers[s])}).save(d)

        want = {}
        for kind, rounds in (("uniform", 30), ("nohost", 20)):
            mark(f"{kind}-start")
            host = 0
            for _ in range(rounds):
                m = int(rng.integers(1, 7))
                for s in stages:
                    for i, tier in enumerate(tiers[s]):
                        pool = np.arange(EXPERTS) if kind == "uniform" else np.flatnonzero(tier != HOST)
                        ids = draw(rng, m, [pool, pool], 0.5)
                        host += int((tier[ids] == HOST).sum())
                        args = tensors(rng, ids)
                        for r in range(4):
                            run(on, built[(s, r)][i], stand, "forward", *args)
            mark(f"{kind}-end")
            want[kind] = round(host / rounds, 4)

        def accept(kinds: list[str], directory: Path) -> tuple[int, dict]:
            receipt = directory / "receipt.json"
            proc = subprocess.run(
                [*accept_cmd, "--stats-dir", str(directory), "--stages", "0", "20", "--kinds", *kinds,
                 "--output", str(receipt)], capture_output=True, text=True)
            if not receipt.is_file():
                raise Inconclusive(f"d: the acceptance script wrote no receipt: {proc.stderr[-300:]}")
            return proc.returncode, json.loads(receipt.read_text())

        code, res = accept(["uniform"], d)
        got = res["kinds"]["uniform"]["host_routes_per_round"] if code == 1 else None
        out["uniform"] = {"exit": code, "host_routes_per_round": got, "counted_here": want["uniform"]}
        if code != 1 or got != want["uniform"] or want["uniform"] <= 1.0:
            fails.append(f"d: uniform routing: exit {code}, {got} against {want['uniform']} counted here")
        code, res = accept(["nohost"], d)
        got = res["kinds"]["nohost"]["host_routes_per_round"] if code == 0 else None
        out["nohost"] = {"exit": code, "host_routes_per_round": got}
        if code != 0 or got != 0.0 or want["nohost"] != 0.0:
            fails.append(f"d: routing that avoids HOST: exit {code}, {got}")
        code, _ = accept(["uniform", "absent"], d)
        out["absent_kind_exit"] = code
        if code != 3:
            fails.append(f"d: an absent snapshot gave exit {code}, want 3")
        # One rank's file disagrees: add a call's worth to rank 2 of stage 20.
        path = rs.snapshot_path(d, "nohost-end", 20, 2)
        snap = rs.Snapshot.load(path)
        stat = snap.stat.copy()
        stat[:, 0] += 1
        rs.Snapshot(snap.label, snap.boot, snap.first_layer, snap.rank, snap.unix_time, snap.layers,
                    stat, snap.gap_hist, snap.recent, snap.routes, snap.tier).save(d)
        code, _ = accept(["nohost"], d)
        out["rank_disagrees_exit"] = code
        if code != 3:
            fails.append(f"d: a rank that disagrees gave exit {code}, want 3")
        snap.save(d)
        # A mark inside a round: stage 0 has run a call that stage 20 has not.
        mark("split-start")
        m = 3
        for s in stages:
            for i, tier in enumerate(tiers[s]):
                args = tensors(rng, draw(rng, m, [np.arange(EXPERTS)] * 2, 0.5))
                for r in range(4):
                    run(on, built[(s, r)][i], stand, "forward", *args)
        for i, tier in enumerate(tiers[0]):
            args = tensors(rng, draw(rng, m, [np.arange(EXPERTS)] * 2, 0.5))
            for r in range(4):
                run(on, built[(0, r)][i], stand, "forward", *args)
        mark("split-end")
        code, _ = accept(["split"], d)
        out["mark_inside_round_exit"] = code
        if code != 3:
            fails.append(f"d: a mark inside a round gave exit {code}, want 3")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha", required=True)
    parser.add_argument("--accept-cmd", required=True,
                        help="the command that runs accept_host_routes, without its arguments (one string)")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    fails: list[str] = []
    unsure: list[str] = []
    res: dict = {"check": "rank4-route-stats-cpu"}
    try:
        from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_stats as rs

        on = load(a.module, a.sha, "on", "1")
        off = load(a.module, a.sha, "off", None)
        ref = load(a.reference, a.reference_sha, "ref", None)
        if on._STATS is not True or off._STATS is not False:
            fails.append("the switch was not read from the environment")
        try:
            load(a.module, a.sha, "bad", "yes")
        except ValueError:
            pass
        else:
            fails.append("VLLM_NVFP4_8R_STATS=yes was accepted")
        res["a"] = part_a(on, rs, fails, unsure)
        res["b"] = part_b(on, off, ref, fails, unsure)
        res["c"] = part_c(on, rs, fails)
        res["d"] = part_d(on, rs, shlex.split(a.accept_cmd), fails, unsure)
    except Inconclusive as exc:
        unsure.append(str(exc))
    res["inconclusive"], res["fails"] = unsure, fails
    res["verdict"] = "INCONCLUSIVE" if unsure else ("FAIL" if fails else "PASS")
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 3 if unsure else (1 if fails else 0)


if __name__ == "__main__":
    sys.exit(main())
