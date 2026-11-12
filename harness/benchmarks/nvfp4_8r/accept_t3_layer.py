"""A3 (plan 01, the test host, one card): the three-tier layer on real weights.

Four processes = four TP ranks on ONE card, gloo group. Each rank builds the
layer with the product functions (attach_iq_peer, build_layer, forward) from
the real packs of layers 5/12/27 and its IQ slices (make_t3_iq_slices), and
runs the same sequence of calls. The sum of the four ranks' outputs is
compared with fp64 references made on CPU outside the engine (make_t3_refs):
native routes from the raw NVFP4 formula, warm routes from
gguf.quants.dequantize, same limited SwiGLU.

  c1 routed experts HOT, the rest WARM: rel L2 <= 1e-2 (the P2 rule).
  c2 routed experts WARM, the rest HOT: rel L2 e0 recorded; rule <= 2e-2
     (Q8 arithmetic). e0 above it is INCONCLUSIVE: the tolerance derivation
     is wrong and is not loosened here.
  c3 mixed (HOT if id % 4 < 2, WARM if id % 4 == 2, HOST if id % 4 == 3):
     rel L2 <= 2e-2.
  c4 tokens with no WARM route: after NaN blocks of the output's size were
     freed, the IQ partial is exactly zero there and the output is finite.
  c5 CUDA graph per M = 1..6: replay with sample A, then sample B (another
     tier mix); replay B is bit-equal to eager B.
  c6 the native partial of c3 is bit-equal to the all-resident object
     (today's path) run with the WARM routes' weights set to 0.

Every call that runs the IQ op is followed by a device synchronize and a
group collective: here nothing else bounds rank skew on the peer bank (the
server's MoE all-reduce does).
Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE (the missing input is named).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .falsify_t3_iq_drop import HIDDEN, TOP_K, Inconclusive, gate, nan_prefill

HOT, WARM, HOST = 0, 1, 2
RULE_NATIVE = 1e-2
RULE_IQ = 2e-2


def total(out: torch.Tensor) -> np.ndarray:
    """Sum of the four ranks' outputs (gloo); doubles as the barrier."""
    torch.cuda.synchronize()
    acc = out.double().cpu()
    dist.all_reduce(acc)
    return acc.numpy()


def agree(flag: bool) -> bool:
    t = torch.tensor([int(flag)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def step() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def rel(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def reference(nat: np.ndarray, iq: np.ndarray, ids: np.ndarray, tier: np.ndarray, m: int):
    """Slot by slot, the reference of the tier the slot runs in."""
    warm = (tier[ids[:m]] == WARM)[..., None]
    both = np.where(warm, iq.reshape(6, TOP_K, HIDDEN)[:m], nat.reshape(6, TOP_K, HIDDEN)[:m])
    return both.sum(1)


def run_layer(native, tiers, cfg, refs_path: Path, slices: Path, rank: int, dev) -> dict:
    if not refs_path.is_file():
        raise Inconclusive(f"references absent: {refs_path}")
    with np.load(refs_path, allow_pickle=False) as data:
        refs = {k: data[k] for k in data.files}
    meta = json.loads(bytes(refs["meta"]).decode())
    layer, samples = meta["layer"], meta["samples"]
    for temp in ("hot", "cold"):
        if not (cfg.packs / f"L{layer:02d}-r{rank}-{temp}.bin").is_file():
            raise Inconclusive(f"pack absent: L{layer:02d}-r{rank}-{temp}.bin in {cfg.packs}")
    shards = []
    for role, kind in (("w1", meta["type13"]), ("w3", meta["type3"]), ("w2", meta["type2"])):
        path = slices / f"iq-L{layer:02d}-r{rank}-{role}.bin"
        if not path.is_file():
            raise Inconclusive(f"IQ slice absent: {path}")
        shards.append(native.IqShardSource.from_slice_file(path, role, kind))
    source = native.IqLayerSource(*shards)

    routed = np.zeros(384, bool)
    for s in range(samples):
        routed[refs[f"ids_s{s}"].ravel()] = True
    ident = np.arange(384)
    tier_maps = {
        "c1": np.where(routed, HOT, WARM).astype(np.int8),
        "c2": np.where(routed, WARM, HOT).astype(np.int8),
        "c3": np.select([ident % 4 < 2, ident % 4 == 2], [HOT, WARM], HOST).astype(np.int8),
    }
    res: dict = {"layer": layer, "type13": meta["type13"], "type2": meta["type2"],
                 "routed_experts": int(routed.sum())}

    def sample(s: int, m: int):
        return (torch.from_numpy(refs[f"x_s{s}"][:m]).to(dev),
                torch.from_numpy(refs[f"ids_s{s}"][:m]).to(dev),
                torch.from_numpy(refs[f"tw_s{s}"][:m]).to(dev))

    def accuracy(moe, tier: np.ndarray) -> dict:
        errs = []
        finite = True
        for s in range(samples):
            for m in (1, 6):
                out = moe.forward(*sample(s, m))
                finite &= bool(torch.isfinite(out).all())
                errs.append(rel(total(out), reference(refs[f"nat_s{s}"], refs[f"iq_s{s}"],
                                                      refs[f"ids_s{s}"], tier, m)))
        return {"worst_rel_l2": max(errs), "best_rel_l2": min(errs), "n": len(errs),
                "finite": agree(finite)}

    # c1, and c4 where no route at all is WARM.
    moe = native.build_layer(cfg, layer, rank, tier_maps["c1"], source, dev, max_tokens=6,
                             stage_prefill=None)
    res["c1"] = accuracy(moe, tier_maps["c1"])
    zero, finite, effective = True, True, True
    for s in range(samples):
        for m in (1, 6):
            x, ids, tw = sample(s, m)
            iq_ids, _ = moe._iq_routes(ids, tw)
            effective &= agree(nan_prefill(m, dev))
            part = moe._iq_partial(x, iq_ids, tw)
            zero &= bool((part == 0).all())
            step()
            effective &= agree(nan_prefill(m, dev))
            out = moe.forward(x, ids, tw)
            finite &= bool(torch.isfinite(out).all())
            step()
    res["c4_no_warm_route"] = {"iq_partial_exact_zero": agree(zero), "finite": agree(finite),
                               "nan_prefill_effective": effective}
    del moe
    torch.cuda.empty_cache()

    # c2: every routed expert WARM.
    moe = native.build_layer(cfg, layer, rank, tier_maps["c2"], source, dev, max_tokens=6,
                             stage_prefill=None)
    res["c2"] = accuracy(moe, tier_maps["c2"])
    del moe
    torch.cuda.empty_cache()

    # c3 mixed; c4 on its tokens without a WARM route; c5 graphs; c6 native partial.
    tier = tier_maps["c3"]
    moe = native.build_layer(cfg, layer, rank, tier, source, dev, max_tokens=6,
                             stage_prefill=None)
    res["c3"] = accuracy(moe, tier)
    zero, effective, tokens = True, True, 0
    for s in range(samples):
        x, ids, tw = sample(s, 6)
        dry = torch.from_numpy((tier[refs[f"ids_s{s}"]] != WARM).all(1)).to(dev)
        tokens += int(dry.sum())
        iq_ids, _ = moe._iq_routes(ids, tw)
        effective &= agree(nan_prefill(6, dev))
        part = moe._iq_partial(x, iq_ids, tw)
        zero &= bool((part[dry] == 0).all())
        step()
    res["c4_mixed_step"] = {"tokens_without_warm_route": tokens,
                            "iq_partial_exact_zero": agree(zero),
                            "nan_prefill_effective": effective}
    raw = torch.from_numpy(native.read_rank_experts(cfg.packs, layer, rank))
    full = native.NativeV41Moe(native.NativeExpertSet.from_raw(raw, dev),
                               torch.arange(384, dtype=torch.int32, device=dev), max_tokens=6,
                               group_plan=gplan.GroupPlan.CHAIN, swiglu=swig.Swiglu.CHAIN, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=native.qrb.PrefillRb.RB2)
    del raw
    equal = True
    for s in range(samples):
        for m in (1, 6):
            x, ids, tw = sample(s, m)
            moe.forward(x, ids, tw)
            step()
            partial = moe.out[:m].clone()
            keep = torch.from_numpy(tier[refs[f"ids_s{s}"][:m]] != WARM).to(dev)
            equal &= bool(torch.equal(partial, full.forward(x, ids, tw * keep)))
    res["c6_native_partial_bit_equal"] = agree(equal)
    del full
    graphs = {}
    for m in range(1, 7):
        sx, si, st = (t.clone() for t in sample(0, m))
        for _ in range(3):
            moe.forward(sx, si, st)
            step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = moe.forward(sx, si, st)
        for s in (0, 1):
            x, ids, tw = sample(s, m)
            sx.copy_(x), si.copy_(ids), st.copy_(tw)
            graph.replay()
            step()
        replayed = static_out.clone()
        eager = moe.forward(*sample(1, m)).clone()
        step()
        graphs[str(m)] = agree(bool(torch.equal(replayed, eager)))
        del graph
    res["c5_graph_replay_bit_equal"] = graphs
    del moe
    torch.cuda.empty_cache()
    return res


def verdict(layers: list[dict]) -> tuple[str, list[str]]:
    fails, unsure = [], []
    for r in layers:
        tag = f"L{r['layer']}"
        if not r["c1"]["worst_rel_l2"] <= RULE_NATIVE:
            fails.append(f"{tag} c1 rel L2 {r['c1']['worst_rel_l2']:.3e} > {RULE_NATIVE}")
        if not r["c2"]["worst_rel_l2"] <= RULE_IQ:
            unsure.append(f"{tag} c2 e0 {r['c2']['worst_rel_l2']:.3e} > {RULE_IQ}: "
                          "the tolerance derivation is wrong")
        if not r["c3"]["worst_rel_l2"] <= RULE_IQ:
            fails.append(f"{tag} c3 rel L2 {r['c3']['worst_rel_l2']:.3e} > {RULE_IQ}")
        for case in ("c1", "c2", "c3"):
            if not r[case]["finite"]:
                fails.append(f"{tag} {case} output is not finite")
        for case in ("c4_no_warm_route", "c4_mixed_step"):
            if not r[case]["nan_prefill_effective"]:
                unsure.append(f"{tag} {case}: the allocator did not hand the NaN block back")
            if not r[case]["iq_partial_exact_zero"]:
                fails.append(f"{tag} {case}: IQ partial is not exactly zero")
        if not r["c4_no_warm_route"]["finite"]:
            fails.append(f"{tag} c4: output is not finite")
        for m, ok in r["c5_graph_replay_bit_equal"].items():
            if not ok:
                fails.append(f"{tag} c5 graph M{m}: replay B != eager B")
        if not r["c6_native_partial_bit_equal"]:
            fails.append(f"{tag} c6: native partial != all-resident with WARM weights 0")
    if fails:
        return "FAIL", fails + unsure
    if unsure:
        return "INCONCLUSIVE", unsure
    return "PASS", []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--packs", type=Path)
    parser.add_argument("--slices", type=Path)
    parser.add_argument("--refs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    if a.gate_only:
        try:
            print(json.dumps(gate(a.card)))
        except Inconclusive as exc:
            print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
            return 3
        return 0
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{a.port}",
                            rank=a.rank, world_size=4)
    dev = torch.device("cuda", 0)
    # build_layer reads only cfg.packs; the tally and the reserve are the
    # loader's inputs and are not used here (the tally never leaves the 8-card host).
    cfg = native.NativePackConfig(a.packs, Path("/nonexistent/tally-stays-local"),
                                  {0: 1}, tiers.HotIqHost(0.0),
                                  tiers.PrefillWarm.NATIVE_STAGED, gplan.GroupPlan.CHAIN, swig.Swiglu.CHAIN,
                                  native.iqt.IqTail.TORCH, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=native.qrb.PrefillRb.RB2)
    try:
        if not (a.packs / "manifest.json").is_file():
            raise Inconclusive(f"pack manifest absent in {a.packs}")
        try:
            native.attach_iq_peer(dist.group.WORLD, a.rank, dev)
        except (RuntimeError, OSError) as exc:
            raise Inconclusive(f"the IPC peer bank cannot be mapped: {exc}") from exc
        layers = [run_layer(native, tiers, cfg, r, a.slices, a.rank, dev) for r in a.refs]
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    status, notes = verdict(layers)
    if a.rank == 0:
        a.output.write_text(json.dumps({"check": "A3", "status": status, "failures": notes,
                                        "rules": {"native": RULE_NATIVE, "iq": RULE_IQ},
                                        "layers": layers}, indent=1) + "\n")
    print(json.dumps({"rank": a.rank, "status": status, "failures": notes}))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[status]


if __name__ == "__main__":
    sys.exit(main())
