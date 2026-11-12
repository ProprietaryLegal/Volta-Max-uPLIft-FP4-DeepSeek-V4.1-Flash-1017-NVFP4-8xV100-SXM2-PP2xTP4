"""Plan 06 acceptance (the test host, one card): the fused IQ tail.

Written before the build (plan 06 section 7). It must fail while
nvfp4_8r_iq_tail does not exist. Three parts, run in this order. A later part
runs only when every earlier part has zero violations: the layer part runs
product kernels (QPN, the IQ op) on the new plan's tensors, and a wrong plan
or a wrong IQ slot would make them read out of bounds.

Part 1, plan (no product kernel runs). Random tier arrays of the two
three-tier kinds (with and without a HOST tier); for each a ``torch`` layer
and a ``fused`` layer on placeholder banks, both with the group-plan kernel.
Random routings at M = 1..6 (max_tokens 6) and M = 7..16 (max_tokens 16),
shuffled so a small M follows a larger one, and test_group_plan's edge cases.
Before every call the fused planner's buffers hold garbage. Checked:
  - iq[i] of the fused plan equals the IQ slot of route i's expert computed
    on the CPU from the tier array by its definition (WARM experts take the
    slots 0, 1, ... in ascending expert id; every other expert -1);
  - the entries of the iq buffer past n are still the garbage;
  - the fused plan's launches pass test_group_plan.coverage (route by route
    from the tier array, no compaction code) and equal the torch layer's
    launches bit for bit.

Part 2, the tail kernel alone. Random FP16 rows y2, FP32 weights, IQ slots
and an FP16 partial at M = 1..6: every finite FP16 bit pattern (signed zeros,
subnormals, the largest values) and a second set of small normal values whose
six products are of like size, so that every add rounds. The output buffer
holds garbage before the call. Expected value: the installed sequence,
_mtp_weighted_reduce(y2, torch.where(iq < 0, w, 0.0)) + part.float(), compared
on the FP32 bit patterns (a -0.0 against a +0.0 is a violation).

Part 3, layer. The real packs and IQ slices of one layer (run once each for
5, 12 and 27): the product ``torch`` layer (build_layer) and a ``fused``
layer on the same banks. For M = 6 at 0, 1, 3, 6, 12, 16, 36 live IQ routes,
M = 1 at 0, 1, 3, 6, and rows sharing 1, 2 and 6 WARM experts: the fused
plan's iq equals the chain's ``_iq_routes`` ids, and the FP32 forward outputs
are bit-equal, eager and in a CUDA graph replay. One process has no TP peers:
bench_iq_live_routes.self_peer stands in for them (its module text).

Never run this test on a deliberately broken kernel (test_group_plan's note:
the test host card 0, Xid 31, 2026-11-11).

Pass rule: zero violations in every part. Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE
(Triton cannot compile on the card; the error is named). The gate (no compute
process, power limit, MemAvailable) is the launcher's.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
from vllm.model_executor.layers.fused_moe.sm70.reduction import _mtp_weighted_reduce

from . import bench_iq_live_routes as live
from . import iq_live_core as core
from . import test_group_plan as tgp

HOT, WARM, HOST = tgp.HOT, tgp.WARM, tgp.HOST
KINDS = ("three_tier_host", "three_tier_nohost")
N_MAIN, N_WIDE = 6_000, 600
TAIL_TRIALS = 150            # per M and per value set
GARBAGE = 0x5A5A5A5A


def expected_iq_slots(tier: np.ndarray) -> np.ndarray:
    """IQ slot of every expert by the definition (ascending id), -1 if not WARM."""
    warm = tier == WARM
    return np.where(warm, np.cumsum(warm) - 1, -1).astype(np.int32)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype == torch.float32 and a.shape == b.shape and bool(
        torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def three_tier(iqt, tier: np.ndarray, max_tokens: int, tail, swiglu, dev: torch.device):
    n_hot = int((tier == HOT).sum())
    return nat.NativeV41Moe.three_tier(
        tgp.placeholder_set(n_hot, dev), tgp.placeholder_set(nat.NUM_EXPERTS - n_hot, dev),
        tgp.placeholder_iq(int((tier == WARM).sum()), dev), tier, max_tokens, None,
        group_plan=gplan.GroupPlan.KERNEL, swiglu=swiglu, prefill_swiglu=nat.swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2, iq_tail=tail)


# -- part 1 ----------------------------------------------------------------------
def run_plan(iqt, swiglu, dev: torch.device, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    res = {"routings": 0, "edge_cases": 0, "maps": 0, "by_kind": {k: 0 for k in KINDS},
           "violations": 0, "first_violations": []}
    for max_tokens, count, ms in ((6, N_MAIN, list(range(1, 7))), (16, N_WIDE, list(range(7, 17)))):
        n_maps = 24 if max_tokens == 6 else 6
        for j in range(n_maps):
            kind = KINDS[j % len(KINDS)]
            tier = tgp.random_tier(kind, rng)
            want_slot = expected_iq_slots(tier)
            ref = three_tier(iqt, tier, max_tokens, iqt.IqTail.TORCH, swiglu, dev)
            fused = three_tier(iqt, tier, max_tokens, iqt.IqTail.FUSED, swiglu, dev)
            if not isinstance(fused._planner.keys, gplan.TierKeys) or fused._planner.buffers.iq is None \
                    or ref._planner.buffers.iq is not None or fused._iq_tail is None \
                    or ref._iq_tail is not None:
                raise RuntimeError("the two layers are not the torch and the fused object")
            if not np.array_equal(fused.warm.iq_slot_of_expert.cpu().numpy(), want_slot):
                raise RuntimeError("the layer's IQ slot map is not the definition's")
            res["maps"] += 1
            cases = [("random", tgp.random_routing(tier, int(rng.choice(ms)), rng))
                     for _ in range(count // n_maps)]
            for m in (ms[0], ms[-1]):
                cases += list(tgp.edge_routings(tier, m, rng).items())
            buf = fused._planner.buffers
            for idx in rng.permutation(len(cases)):
                name, ids = cases[idx]
                n = ids.size
                ids_dev = torch.from_numpy(ids).to(dev)
                tgp.poison(fused, rng)
                buf.iq.fill_(GARBAGE)
                launches, iq = fused._route_plan(ids_dev)
                ref_launches, ref_iq = ref._route_plan(ids_dev)
                try:
                    if ref_iq is not None:
                        raise tgp.Violation("the torch layer's plan carries IQ slots")
                    if iq is None or iq.dtype != torch.int32 or tuple(iq.shape) != (n,):
                        raise tgp.Violation(f"iq is {None if iq is None else (iq.dtype, tuple(iq.shape))}")
                    got = buf.iq.cpu().numpy()
                    want = want_slot[ids.reshape(-1)]
                    if not np.array_equal(got[:n], want):
                        raise tgp.Violation(f"iq {got[:n].tolist()} want {want.tolist()}")
                    if (got[n:] != GARBAGE).any():
                        raise tgp.Violation("the kernel wrote iq entries past n")
                    tgp.coverage(tier, kind, ids, launches)
                    tgp.equal_launches(ref_launches, launches)
                except tgp.Violation as exc:
                    res["violations"] += 1
                    if len(res["first_violations"]) < 10:
                        res["first_violations"].append(
                            {"kind": kind, "case": name, "M": int(ids.shape[0]), "error": str(exc)[:600]})
                res["routings" if name == "random" else "edge_cases"] += 1
                res["by_kind"][kind] += 1
            del ref, fused
    res["pass"] = res["violations"] == 0 and res["routings"] > 0 and res["edge_cases"] > 0
    return res


# -- part 2 ----------------------------------------------------------------------
def finite_fp16(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Uniform over the finite FP16 bit patterns (exponent field below 31)."""
    bits = rng.integers(0, 1 << 16, size=shape, dtype=np.uint16)
    bad = (bits & 0x7C00) == 0x7C00
    bits[bad] &= 0xBFFF  # clear one exponent bit: finite
    return bits.view(np.float16)


def run_tail(iqt, dev: torch.device, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    res = {"cases": 0, "violations": 0, "first_violations": [], "by_set": {}}
    for m in range(1, 7):
        s = m * nat.TOP_K
        for value_set in ("all_finite_bits", "like_sized", "edge_weights"):
            for trial in range(TAIL_TRIALS):
                if value_set == "all_finite_bits":
                    y2 = finite_fp16(rng, (s, nat.HIDDEN))
                    part = finite_fp16(rng, (m, nat.HIDDEN))
                    w = rng.random((m, nat.TOP_K), dtype=np.float32)
                elif value_set == "like_sized":
                    y2 = rng.normal(0.0, 0.05, (s, nat.HIDDEN)).astype(np.float16)
                    part = rng.normal(0.0, 0.05, (m, nat.HIDDEN)).astype(np.float16)
                    w = rng.dirichlet(np.ones(nat.TOP_K), m).astype(np.float32)
                else:
                    y2 = rng.normal(0.0, 1.0, (s, nat.HIDDEN)).astype(np.float16)
                    part = finite_fp16(rng, (m, nat.HIDDEN))
                    w = rng.choice(np.array([0.0, -0.0, 1.0, -1.0, 1e-30, 3.0e4, 0.1666667],
                                            np.float32), (m, nat.TOP_K))
                density = rng.choice([0.0, 1.0, rng.random()])
                iq = np.where(rng.random(s) < density, rng.integers(0, 384, s), -1).astype(np.int32)
                if trial % 7 == 0:
                    iq[iq < 0] = np.int32(-(2 ** 31))  # any negative id is "not WARM"
                ty2, tw, tiq, tpart = (torch.from_numpy(np.ascontiguousarray(a)).to(dev)
                                       for a in (y2, w, iq, part))
                ref = torch.empty((m, nat.HIDDEN), dtype=torch.float32, device=dev)
                _mtp_weighted_reduce(ty2, torch.where(tiq.view(m, nat.TOP_K) < 0, tw, 0.0).contiguous(), ref)
                want = ref + tpart.float()
                out = torch.full((m, nat.HIDDEN), float("nan"), dtype=torch.float32, device=dev)
                iqt.iq_tail_into(ty2, tw, tiq.view(m, nat.TOP_K) if trial % 2 else tiq, tpart, out)
                torch.cuda.synchronize()
                res["cases"] += 1
                res["by_set"][value_set] = res["by_set"].get(value_set, 0) + 1
                if not bits_equal(out, want):
                    res["violations"] += 1
                    if len(res["first_violations"]) < 10:
                        diff = (out.view(torch.int32) != want.view(torch.int32))
                        j = int(torch.nonzero(diff.reshape(-1))[0])
                        res["first_violations"].append(
                            {"M": m, "set": value_set, "trial": trial, "differing": int(diff.sum()),
                             "first_index": j,
                             "got_bits": int(out.view(torch.int32).reshape(-1)[j]),
                             "want_bits": int(want.view(torch.int32).reshape(-1)[j])})
    res["pass"] = res["violations"] == 0 and res["cases"] == 6 * 3 * TAIL_TRIALS
    return res


# -- part 3 ----------------------------------------------------------------------
def run_layer(iqt, swiglu, a, dev: torch.device) -> dict:
    with np.load(a.slices / f"t3-refs-L{a.layer:02d}.npz", allow_pickle=False) as data:
        refs = {k: data[k] for k in ("x_s0", "tw_s0", "meta")}
    meta = json.loads(bytes(refs["meta"]).decode())
    if meta["layer"] != a.layer or meta["m"] != 6:
        raise ValueError(f"the reference file is for {meta}")
    shards = [
        nat.IqShardSource.from_slice_file(
            a.slices / f"iq-L{a.layer:02d}-r{live.RANK}-{role}.bin", role, kind)
        for role, kind in (("w1", meta["type13"]), ("w3", meta["type3"]), ("w2", meta["type2"]))
    ]
    tier = core.bench_tier()
    cfg = nat.NativePackConfig(
        a.packs, nat.TallySource(Path("/nonexistent/the-tally-stays-on-the-server")),
        {0: 1}, tiers.HotIq(), tiers.PrefillWarm.NATIVE_STAGED, gplan.GroupPlan.KERNEL,
        nat.swig.Swiglu(a.swiglu), iqt.IqTail.TORCH, prefill_swiglu=nat.swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)
    banks = live.self_peer(dev)
    ref = nat.build_layer(cfg, a.layer, live.RANK, tier, nat.IqLayerSource(*shards), dev,
                          max_tokens=live.MAX_TOKENS, stage_prefill=None)
    fused = nat.NativeV41Moe.three_tier(
        ref.experts, ref.cold, ref.warm.iq, tier, live.MAX_TOKENS, None,
        group_plan=gplan.GroupPlan.KERNEL, swiglu=swiglu, prefill_swiglu=nat.swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2, iq_tail=iqt.IqTail.FUSED)
    if ref._iq_tail is not None or fused._iq_tail is None:
        raise RuntimeError("the two layers are not the torch and the fused object")
    x6 = torch.from_numpy(refs["x_s0"]).to(dev)
    t6 = torch.from_numpy(refs["tw_s0"]).to(dev)
    seed0 = 1000 * a.layer
    cases = [(f"M{m}_k{k}", core.live_routes(k, m, tier, seed0 + 10 * m + k), k, k)
             for m, ks in ((6, core.K_VALUES), (1, (0, 1, 3, 6))) for k in ks]
    cases += [(f"M6_shared{j}", core.shared_routes(j, 6, tier, seed0 + 500 + j), 6 * j, j)
              for j in (1, 2, 6)]
    # M = 1 after M = 6 once more, so a stale tail of the larger call would show.
    cases.append(("M1_k3_again", core.live_routes(3, 1, tier, seed0 + 13), 3, 3))
    res = {"cases": {}, "violations": 0, "first_violations": []}

    def bad(name: str, what: str) -> None:
        res["violations"] += 1
        if len(res["first_violations"]) < 10:
            res["first_violations"].append({"case": name, "error": what})

    for name, ids, live_n, distinct in cases:
        routing = live.Routing(ref, banks, x6, t6, ids, live_n, distinct)
        row = {"live_routes": live_n}
        _, iq = fused._route_plan(routing.si)
        row["plan_iq_equals_chain"] = bool(torch.equal(iq, routing.iq_ids.reshape(-1)))
        want = routing.b().clone()
        got = fused.forward(routing.sx, routing.si, routing.st).clone()
        again = routing.b().clone()
        row["eager_bit_equal"] = bits_equal(got, want) and bits_equal(again, want)
        g_ref, o_ref = live.capture(routing.b)
        g_fused, o_fused = live.capture(lambda: fused.forward(routing.sx, routing.si, routing.st))
        for _ in range(3):
            g_ref.replay()
        torch.cuda.synchronize()
        want_graph = o_ref.clone()
        for _ in range(3):
            g_fused.replay()
        torch.cuda.synchronize()
        got_graph = o_fused.clone()
        row["graph_bit_equal"] = bits_equal(got_graph, want_graph)
        row["graph_equals_eager"] = bits_equal(want_graph, want)
        row["nonzero"] = int((want != 0).sum())
        row["finite"] = bool(torch.isfinite(want).all())
        del g_ref, g_fused, o_ref, o_fused
        for key in ("plan_iq_equals_chain", "eager_bit_equal", "graph_bit_equal",
                    "graph_equals_eager", "finite"):
            if not row[key]:
                bad(name, key)
        if row["nonzero"] == 0:
            bad(name, "the forward output is all zero")
        res["cases"][name] = row
        del routing
    res["pass"] = res["violations"] == 0 and len(res["cases"]) == len(cases)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--slices", type=Path, required=True)
    p.add_argument("--swiglu", choices=("chain", "kernel"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    res: dict = {"check": "iq_tail", "layer": a.layer, "seed": a.seed, "swiglu": a.swiglu,
                 "gate": json.loads(a.gate_json)}

    def finish(status: str, code: int) -> int:
        res["status"] = status
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({"status": status, "layer": a.layer, "reason": res.get("reason", "")[:600],
                          **{k: {"pass": v.get("pass"), "violations": v.get("violations"),
                                 "first": v.get("first_violations", [])[:2]}
                             for k, v in res.items() if k in ("plan", "tail", "layers")}})[:1400])
        return code

    name = "vllm.model_executor.layers.fused_moe.experts.nvfp4_8r_iq_tail"
    try:
        iqt = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        res["reason"] = f"no kernel: {exc}"
        return finish("FAIL", 1)
    import triton

    dev = torch.device("cuda", 0)
    swiglu = nat.swig.Swiglu(a.swiglu)
    res.update({
        "device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
        "triton": triton.__version__,
        "sha256": {n: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                   for n, mod in (("native", nat), ("group_plan", gplan), ("iq_tail", iqt))},
    })
    try:
        probe = three_tier(iqt, core.bench_tier(), 6, iqt.IqTail.FUSED, swiglu, dev)
        res["plan_variants"] = {
            str(m): dict(zip(("block", "warps", "regs", "spills"), v))
            for m, v in probe._planner.compiled.items()}
        del probe
    except (triton.compiler.errors.CompilationError, triton.runtime.errors.OutOfResources) as exc:
        res["reason"] = f"Triton cannot compile a kernel on this card: {type(exc).__name__}: {exc}"
        return finish("INCONCLUSIVE", 3)
    res["plan"] = run_plan(iqt, swiglu, dev, a.seed)
    if not res["plan"]["pass"]:
        return finish("FAIL", 1)
    res["tail"] = run_tail(iqt, dev, a.seed + 1)
    if not res["tail"]["pass"]:
        return finish("FAIL", 1)
    res["layers"] = run_layer(iqt, swiglu, a, dev)
    return finish("PASS" if res["layers"]["pass"] else "FAIL", 0 if res["layers"]["pass"] else 1)


if __name__ == "__main__":
    sys.exit(main())
