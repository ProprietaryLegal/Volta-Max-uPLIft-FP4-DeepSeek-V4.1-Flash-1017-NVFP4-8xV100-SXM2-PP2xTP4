"""Plan 03 Part B acceptance (the test host, one card): the fused group-plan kernel.

Written before the kernel (plan 03 section 7). It must fail while
nvfp4_8r_group_plan does not exist.

Part 1, coverage (the acceptance). Random tier arrays of the four layer
kinds the engine builds (all resident; two-tier hot/host; three-tier with a
HOST tier; three-tier without one), 20,000 random routings at M = 1..6 on
production-sized objects (max_tokens 6), 2,000 more at M = 7..16 on
max_tokens 16 objects, and the edge cases: all hot, none hot, all one expert,
every route a different expert, all warm, host only. The M values are
shuffled, so a small M often follows a larger one (a stale tail would show).
Before every call the kernel's buffers are filled with garbage, so an element
the kernel does not write shows as a violation.

Expected values come route by route from the tier array on the CPU, with no
compaction code: under the QPN grid rule (skinny_moe_qpn_sm70: grid covers
min(S, E) groups; group g covers perm[goff[g]:goff[g+1]] and is skipped when
empty) every HOT route is covered exactly once by the HBM launch at its HOT
slot, every HOST route exactly once by the host launch at its host-RAM slot,
no WARM route is covered, and every gids value the grid visits is inside its
bank. perm must be a permutation of 0..n-1 and goff must be non-decreasing
inside 0..n.

Part 2, equality with the chain (the specification of the old path):
every launch tensor of the kernel object is bit-equal to the chain object's
(_groups / _split_groups / _tier_groups) for the same routing.

Part 3, layer regression (plan 03 section 7, last item): forward output of
`kernel` bit-equal to `chain` on the P2 reference inputs (layers 5, 12, 27;
all-resident and two-tier hot/host from the P1 manifest), eager and CUDA graph
replay, M = 1 and 6. A three-tier forward on real weights needs the four-rank
IQ peer bank and is not run here (its launches are covered by parts 1 and 2).

Checking this test with deliberately broken kernels: do it at plan level only
(parts 1 and 2, which compare plan tensors). Never run part 3, or any product
forward, on a broken kernel's plan: a stale goff[n] makes the product QPN
kernels read out of bounds (the test host card 0, Xid 31, 2026-11-11).

Pass rule: zero violations. Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE (Triton
cannot compile on the card; the error is named). The gate (no compute
process, power limit, MemAvailable) is the launcher's.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import QpnFragmentOrder

HOT, WARM, HOST = int(tiers.Tier.HOT), int(tiers.Tier.WARM), int(tiers.Tier.HOST)
KINDS = ("resident", "two_tier", "three_tier_host", "three_tier_nohost")
N_MAIN, N_WIDE = 20_000, 2_000
IQ_TYPE = 16  # IQ2_XXS; the placeholder bank is never read


class Violation(AssertionError):
    pass


# -- objects -------------------------------------------------------------------
def placeholder_set(n: int, dev: torch.device) -> nat.NativeExpertSet:
    z = torch.zeros((n, 1), dtype=torch.uint8, device=dev)
    g = torch.ones(n, dtype=torch.float32, device=dev)
    return nat.NativeExpertSet(z, z, z, z, g, g, QpnFragmentOrder.M8N8K4)


def placeholder_iq(n: int, dev: torch.device) -> nat.IqWarmBank:
    """Right shapes, no memory: stride-0 views of one zero record."""
    banks = []
    for role in ("w1", "w3", "w2"):
        _, rows, blocks = nat._IQ_ROWS[role]
        one = torch.zeros((1, rows, blocks * tiers.IQ_BLOCK_BYTES[IQ_TYPE]),
                          dtype=torch.uint8, device=dev)
        banks.append(one.expand(n, -1, -1))
    return nat.IqWarmBank(banks[0], banks[1], banks[2], IQ_TYPE, IQ_TYPE)


def build(gp, kind: str, tier: np.ndarray, max_tokens: int, mode, dev: torch.device):
    maps = tiers.slot_maps(tier)
    n_hot = int((tier == HOT).sum())
    to = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
    hot_set = placeholder_set(n_hot, dev)
    if kind == "resident":
        return nat.NativeV41Moe(hot_set, to(maps.hot), max_tokens, group_plan=mode)
    cold_set = placeholder_set(int((tier != HOT).sum()), dev)
    if kind == "two_tier":
        return nat.NativeV41Moe(hot_set, to(maps.hot), max_tokens, cold=cold_set,
                                cold_slot_of_expert=to(maps.cold), group_plan=mode)
    iq = placeholder_iq(int((tier == WARM).sum()), dev)
    return nat.NativeV41Moe.three_tier(hot_set, cold_set, iq, tier, max_tokens, None,
                                       group_plan=mode)


def random_tier(kind: str, rng: np.random.Generator) -> np.ndarray:
    """A tier array of ``kind``; the counts run from 1 to the most the kind allows."""
    perm = rng.permutation(nat.NUM_EXPERTS)
    tier = np.full(nat.NUM_EXPERTS, HOT, np.int8)
    if kind == "resident":
        return tier
    if kind == "two_tier":
        n_host = int(rng.choice([1, 2, 6, int(rng.integers(1, 384)), 383]))
        tier[perm[:n_host]] = HOST
        return tier
    n_hot = int(rng.choice([1, int(rng.integers(1, 383)), 382 if kind == "three_tier_host" else 383]))
    rest = perm[n_hot:]
    if kind == "three_tier_nohost":
        tier[rest] = WARM
        return tier
    n_warm = int(rng.integers(1, len(rest)))  # leaves at least one HOST
    tier[rest[:n_warm]] = WARM
    tier[rest[n_warm:]] = HOST
    return tier


def random_routing(tier: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """[m, 6] int32, six distinct experts a row; three styles so groups repeat."""
    style = rng.integers(3)
    if style == 0:  # uniform over all experts
        rows = [rng.choice(nat.NUM_EXPERTS, 6, replace=False) for _ in range(m)]
    elif style == 1:  # a small shared pool: rows share experts
        pool = rng.choice(nat.NUM_EXPERTS, int(rng.integers(6, 40)), replace=False)
        rows = [rng.choice(pool, 6, replace=False) for _ in range(m)]
    else:  # skewed toward one tier present in the layer
        present = [t for t in (HOT, WARM, HOST) if (tier == t).any()]
        favour = np.flatnonzero(tier == present[rng.integers(len(present))])
        rows = []
        for _ in range(m):
            take = min(int(rng.integers(0, 7)), len(favour))
            a = list(rng.choice(favour, take, replace=False))
            others = [e for e in rng.permutation(nat.NUM_EXPERTS) if e not in a]
            rows.append(np.array(a + others[: 6 - take]))
    return np.stack(rows).astype(np.int32)


def edge_routings(tier: np.ndarray, m: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The plan's edge cases that this tier array allows. "all one expert"
    repeats one id in every slot (not a top-k routing; the plan must still be
    right for repeated keys)."""
    out = {}
    by = {t: np.flatnonzero(tier == t) for t in (HOT, WARM, HOST)}
    n = 6 * m

    def rows_from(pool: np.ndarray) -> np.ndarray | None:
        if len(pool) < 6:
            return None
        return np.stack([rng.choice(pool, 6, replace=False) for _ in range(m)]).astype(np.int32)

    for name, pool in (("all_hot", by[HOT]), ("all_warm", by[WARM]), ("host_only", by[HOST]),
                       ("none_hot", np.flatnonzero(tier != HOT))):
        r = rows_from(pool)
        if r is not None:
            out[name] = r
    for t, name in ((HOT, "one_expert_hot"), (WARM, "one_expert_warm"), (HOST, "one_expert_host")):
        if len(by[t]):
            out[name] = np.full((m, 6), int(rng.choice(by[t])), np.int32)
    out["all_different"] = rng.permutation(nat.NUM_EXPERTS)[:n].reshape(m, 6).astype(np.int32)
    return out


# -- checks --------------------------------------------------------------------
def poison(moe, rng: np.random.Generator) -> None:
    b = moe._planner.buffers
    tensors = [b.perm, b.gids_hot, b.goff_hot]
    if b.host is not None:
        tensors += [b.host.gids, b.host.goff]
    tensors += [b.scratch_key, b.scratch_start]  # read only after the kernel writes them
    for t in tensors:
        t.copy_(torch.from_numpy(rng.integers(-1000, 1000, t.numel()).astype(np.int32)))


def coverage(tier: np.ndarray, kind: str, ids: np.ndarray, launches) -> None:
    """Plan 03 section 7, route by route from the tier array."""
    maps = tiers.slot_maps(tier)
    flat = ids.reshape(-1)
    n = flat.size
    want_banks = 1 if kind == "resident" else (
        1 if kind == "three_tier_nohost" else 2)
    if len(launches) != want_banks:
        raise Violation(f"{len(launches)} launches, want {want_banks}")
    seen: dict[int, tuple[int, int]] = {}
    for which, (bank, perm, gids, goff) in enumerate(launches):
        perm, gids, goff = (t.cpu().numpy() for t in (perm, gids, goff))
        if perm.dtype != np.int32 or gids.dtype != np.int32 or goff.dtype != np.int32:
            raise Violation(f"launch {which}: dtypes {perm.dtype} {gids.dtype} {goff.dtype}")
        if perm.shape != (n,) or gids.shape != (n,) or goff.shape != (n + 1,):
            raise Violation(f"launch {which}: shapes {perm.shape} {gids.shape} {goff.shape}")
        if sorted(perm.tolist()) != list(range(n)):
            raise Violation(f"launch {which}: perm is not a permutation of 0..{n - 1}")
        if (goff < 0).any() or (goff > n).any() or (np.diff(goff) < 0).any():
            raise Violation(f"launch {which}: goff not non-decreasing in 0..{n}: {goff}")
        grid = min(n, bank.num_slots)
        for g in range(grid):
            if not 0 <= gids[g] < bank.num_slots:
                raise Violation(f"launch {which}: gids[{g}] = {gids[g]} outside bank of {bank.num_slots}")
            for r in range(goff[g], goff[g + 1]):
                slot = int(perm[r])
                if slot in seen:
                    raise Violation(f"slot {slot} covered twice")
                seen[slot] = (which, int(gids[g]))
    for slot, e in enumerate(flat.tolist()):
        t = int(tier[e])
        if t == HOT:
            want = (0, int(maps.hot[e]))
        elif t == HOST or (t == WARM and kind == "two_tier"):
            want = (1, int(maps.cold[e]))
        else:
            want = None
        if seen.get(slot) != want:
            raise Violation(f"slot {slot} expert {e} tier {t}: covered {seen.get(slot)}, want {want}")


def equal_launches(chain_l, kernel_l) -> None:
    if len(chain_l) != len(kernel_l):
        raise Violation(f"chain has {len(chain_l)} launches, kernel {len(kernel_l)}")
    for which, (c, k) in enumerate(zip(chain_l, kernel_l)):
        if c[0] is not k[0] and c[0].num_slots != k[0].num_slots:
            raise Violation(f"launch {which}: different banks")
        for name, a, b in zip(("perm", "gids", "goff"), c[1:], k[1:]):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
                raise Violation(f"launch {which} {name}: chain {a.cpu().tolist()} "
                                f"kernel {b.cpu().tolist()}")


def run_coverage(gp, dev: torch.device, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    res = {"routings": 0, "edge_cases": 0, "maps": 0, "by_kind": {k: 0 for k in KINDS},
           "violations": 0, "first_violations": []}
    jobs = []  # (max_tokens, n routings, M range)
    for max_tokens, count, ms in ((6, N_MAIN, range(1, 7)), (16, N_WIDE, range(7, 17))):
        jobs.append((max_tokens, count, list(ms)))
    for max_tokens, count, ms in jobs:
        n_maps = 40 if max_tokens == 6 else 8
        per_map = count // n_maps
        for j in range(n_maps):
            kind = KINDS[j % len(KINDS)]
            tier = random_tier(kind, rng)
            chain = build(gp, kind, tier, max_tokens, gp.GroupPlan.CHAIN, dev)
            kernel = build(gp, kind, tier, max_tokens, gp.GroupPlan.KERNEL, dev)
            res["maps"] += 1
            cases = [("random", random_routing(tier, int(rng.choice(ms)), rng))
                     for _ in range(per_map)]
            for m in (ms[0], ms[-1]):
                cases += list(edge_routings(tier, m, rng).items())
            order = rng.permutation(len(cases))
            for idx in order:
                name, ids = cases[idx]
                ids_dev = torch.from_numpy(ids).to(dev)
                poison(kernel, rng)
                k_l = kernel._launches(ids_dev)
                c_l = chain._launches(ids_dev)
                try:
                    coverage(tier, kind, ids, k_l)
                    equal_launches(c_l, k_l)
                except Violation as exc:
                    res["violations"] += 1
                    if len(res["first_violations"]) < 10:
                        res["first_violations"].append(
                            {"kind": kind, "case": name, "M": int(ids.shape[0]), "error": str(exc)[:600]})
                if name == "random":
                    res["routings"] += 1
                else:
                    res["edge_cases"] += 1
                res["by_kind"][kind] += 1
            del chain, kernel
    return res


# -- layer regression ------------------------------------------------------------
def graph_out(moe, x, i, t):
    sx, si, st = x.clone(), i.clone(), t.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            moe.forward(sx, si, st)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = moe.forward(sx, si, st)
    graph.replay()
    torch.cuda.synchronize()
    got = out.clone()
    del graph
    return got


def run_layers(gp, packs: Path, refs_dir: Path, dev: torch.device) -> dict:
    res = {}
    for layer in (5, 12, 27):
        with np.load(refs_dir / f"refs-L{layer:02d}.npz", allow_pickle=False) as data:
            refs = {k: data[k] for k in data.files}
        split = nat.read_manifest(packs)[layer][0]
        raw = torch.from_numpy(nat.read_rank_experts(packs, layer, 0))
        hs, cs = nat.split_maps(split["hot"], split["cold"], dev)
        full_set = nat.NativeExpertSet.from_raw(raw, dev)
        hot_set = nat.NativeExpertSet.from_raw(raw[split["hot"]], dev)
        cold_set = nat.NativeExpertSet.from_raw_host(raw[split["cold"]], dev)
        del raw
        ident = torch.arange(384, dtype=torch.int32, device=dev)
        row = {}
        for kind in ("resident", "two_tier"):
            objs = {}
            for mode in (gp.GroupPlan.CHAIN, gp.GroupPlan.KERNEL):
                if kind == "resident":
                    objs[mode] = nat.NativeV41Moe(full_set, ident, 16, group_plan=mode)
                else:
                    objs[mode] = nat.NativeV41Moe(hot_set, hs, 16, cold=cold_set,
                                                  cold_slot_of_expert=cs, group_plan=mode)
            checks = []
            for s in (0, 1):
                for m in (1, 6):
                    x = torch.from_numpy(refs[f"x_m6_s{s}"][:m]).to(dev)
                    i = torch.from_numpy(refs[f"ids_m6_s{s}"][:m]).to(dev)
                    t = torch.from_numpy(refs[f"tw_m6_s{s}"][:m]).to(dev)
                    eager = {mo: o.forward(x, i, t).clone() for mo, o in objs.items()}
                    graph = {mo: graph_out(o, x, i, t) for mo, o in objs.items()}
                    checks.append({
                        "s": s, "M": m,
                        "eager_bit_equal": bool(torch.equal(eager[gp.GroupPlan.CHAIN],
                                                            eager[gp.GroupPlan.KERNEL])),
                        "graph_bit_equal": bool(torch.equal(graph[gp.GroupPlan.CHAIN],
                                                            graph[gp.GroupPlan.KERNEL])),
                        "graph_equals_eager": bool(torch.equal(graph[gp.GroupPlan.KERNEL],
                                                               eager[gp.GroupPlan.KERNEL])),
                        "finite": bool(torch.isfinite(eager[gp.GroupPlan.KERNEL]).all()),
                    })
            row[kind] = checks
            del objs
        res[f"L{layer:02d}"] = row
        del full_set, hot_set, cold_set
        torch.cuda.empty_cache()
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs-dir", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    res: dict = {"check": "plan03-group-plan", "gate": json.loads(a.gate_json), "seed": a.seed}
    name = "vllm.model_executor.layers.fused_moe.experts.nvfp4_8r_group_plan"
    try:
        gp = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        res["status"] = "FAIL"
        res["reason"] = f"no kernel: {exc}"
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({"status": "FAIL", "reason": res["reason"]}))
        return 1
    import triton

    dev = torch.device("cuda", 0)
    res["device"] = torch.cuda.get_device_name(dev)
    res["triton"] = triton.__version__
    try:
        probe = build(gp, "two_tier", random_tier("two_tier", np.random.default_rng(0)), 6,
                      gp.GroupPlan.KERNEL, dev)
        res["compiled_max_tokens_6"] = {
            str(m): dict(zip(("block", "warps", "regs", "spills"), v))
            for m, v in probe._planner.compiled.items()}
        del probe
    except (triton.compiler.errors.CompilationError, triton.runtime.errors.OutOfResources) as exc:
        res["status"] = "INCONCLUSIVE"
        res["reason"] = f"Triton cannot compile the kernel on this card: {type(exc).__name__}: {exc}"
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({"status": "INCONCLUSIVE", "reason": res["reason"][:600]}))
        return 3
    res["coverage"] = run_coverage(gp, dev, a.seed)
    cov = res["coverage"]
    if cov["violations"] != 0 or cov["routings"] != N_MAIN + N_WIDE:
        # A wrong plan can make the product kernels read out of bounds (Xid 31,
        # 2026-11-11): no real forward runs on a plan that failed here.
        res["status"] = "FAIL"
        res["layers"] = "not run: plan-level violations"
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({"status": "FAIL", "routings": cov["routings"], "violations": cov["violations"],
                          "layers": res["layers"], "first": cov["first_violations"][:2]}))
        return 1
    res["layers"] = run_layers(gp, a.packs, a.refs_dir, dev)
    layer_fail = [f"{lay}.{kind}.s{c['s']}.M{c['M']}" for lay, row in res["layers"].items()
                  for kind, checks in row.items() for c in checks
                  if not (c["eager_bit_equal"] and c["graph_bit_equal"]
                          and c["graph_equals_eager"] and c["finite"])]
    res["layer_failures"] = layer_fail
    ok = cov["violations"] == 0 and cov["routings"] == N_MAIN + N_WIDE and not layer_fail
    res["status"] = "PASS" if ok else "FAIL"
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": res["status"], "routings": cov["routings"],
                      "edge_cases": cov["edge_cases"], "violations": cov["violations"],
                      "layer_failures": layer_fail, "first": cov["first_violations"][:2]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
