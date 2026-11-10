"""Plan 01 step 7 (the test host, one card): the stage build the loader calls.

Four processes = four TP ranks on ONE card, gloo group, real pack of layer 5
and its IQ slices. The loader's three-tier path is
  stage_min_free -> plan_tiers -> build_stage (digest check, peer bank, layers).
plan_tiers needs the private tally and is checked on the 8-card host (test_t3_stage_plan);
here the plan is a synthetic one-layer TierPlan (HOT if id % 4 < 2, WARM if
id % 4 == 2, HOST if id % 4 == 3) and the two collectives run for real:

  s1 stage_min_free: rank r reports 1000 + r bytes; every rank must get 1000.
  s2 rank symmetry: rank 3 holds a plan with one expert in another tier.
     build_stage must raise ValueError on all four ranks, before the peer bank
     is registered (no rank may go on with a plan the others do not share).
  s3 equal plans: build_stage returns the layer; the sum of the four ranks'
     outputs matches the fp64 reference (make_t3_refs: raw NVFP4 for native
     routes, gguf dequantize for WARM routes) within the c3 rule 2e-2.

Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE (the missing input is named).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from .accept_t3_layer import HOST, HOT, RULE_IQ, WARM, agree, reference, rel, total
from .falsify_t3_iq_drop import Inconclusive, gate

LAYER = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--packs", type=Path)
    parser.add_argument("--slices", type=Path)
    parser.add_argument("--refs", type=Path)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    if a.gate_only:
        try:
            print(json.dumps(gate(a.card)))
        except Inconclusive as exc:
            print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
            return 3
        return 0
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
    from vllm.model_executor.layers.quantization import gguf_v41_moe as iq_op

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{a.port}",
                            rank=a.rank, world_size=4)
    dev = torch.device("cuda", 0)
    group = dist.group.WORLD
    res: dict = {"layer": LAYER}
    try:
        for need in (a.refs, a.packs / "manifest.json",
                     *(a.packs / f"L{LAYER:02d}-r{a.rank}-{t}.bin" for t in ("hot", "cold"))):
            if not need.is_file():
                raise Inconclusive(f"input absent: {need}")
        with np.load(a.refs, allow_pickle=False) as data:
            refs = {k: data[k] for k in data.files}
        meta = json.loads(bytes(refs["meta"]).decode())
        if meta["layer"] != LAYER:
            raise Inconclusive(f"references are for layer {meta['layer']}, not {LAYER}")
        shards = []
        for role, kind in (("w1", meta["type13"]), ("w3", meta["type3"]), ("w2", meta["type2"])):
            path = a.slices / f"iq-L{LAYER:02d}-r{a.rank}-{role}.bin"
            if not path.is_file():
                raise Inconclusive(f"IQ slice absent: {path}")
            shards.append(native.IqShardSource.from_slice_file(path, role, kind))
        sources = {LAYER: native.IqLayerSource(*shards)}
        cfg = native.NativePackConfig(a.packs, Path("/nonexistent/tally-stays-local"),
                                      {0: 1}, tiers.HotIqHost(0.0),
                                      tiers.PrefillWarm.NATIVE_STAGED)
        # build_stage takes the TP group; native_staged uses its cpu_group only.
        tp_group = SimpleNamespace(cpu_group=group)

        # s1: arithmetic expectation, min(1000..1003) = 1000.
        res["s1_min_free"] = native.stage_min_free(group, 1000 + a.rank)

        ident = np.arange(384)
        tier = np.select([ident % 4 < 2, ident % 4 == 2], [HOT, WARM], HOST).astype(np.int8)

        def plan_of(t: np.ndarray):
            return tiers.TierPlan(range(LAYER, LAYER + 1), {LAYER: t},
                                  {LAYER: sources[LAYER].record_bytes}, 1 << 40,
                                  tiers.HotIqHost(0.0))

        # s2: rank 3 moves expert 2 from WARM to HOT.
        odd = tier.copy()
        if a.rank == 3:
            odd[2] = HOT
        raised = False
        try:
            native.build_stage(cfg, a.rank, plan_of(odd), sources, dev, tp_group,
                               max_tokens=6, prefill_tokens=4096)
        except ValueError:
            raised = True
        res["s2_all_ranks_refused"] = agree(raised)
        res["s2_peer_bank_untouched"] = agree(iq_op._PEER.get(dev) is None)

        # s3: equal plans build and compute.
        try:
            built = native.build_stage(cfg, a.rank, plan_of(tier), sources, dev, tp_group,
                                       max_tokens=6, prefill_tokens=4096)
        except (RuntimeError, OSError) as exc:
            raise Inconclusive(f"the stage could not be built: {exc}") from exc
        moe = built[LAYER]
        errs, finite = [], True
        for s in range(meta["samples"]):
            for m in (1, 6):
                out = moe.forward(torch.from_numpy(refs[f"x_s{s}"][:m]).to(dev),
                                  torch.from_numpy(refs[f"ids_s{s}"][:m]).to(dev),
                                  torch.from_numpy(refs[f"tw_s{s}"][:m]).to(dev))
                finite &= bool(torch.isfinite(out).all())
                errs.append(rel(total(out), reference(refs[f"nat_s{s}"], refs[f"iq_s{s}"],
                                                      refs[f"ids_s{s}"], tier, m)))
        res["s3"] = {"keys": sorted(built), "worst_rel_l2": max(errs), "n": len(errs),
                     "finite": agree(finite)}
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    fails = []
    if res["s1_min_free"] != 1000:
        fails.append(f"s1: least free {res['s1_min_free']}, want 1000")
    if not res["s2_all_ranks_refused"]:
        fails.append("s2: a rank accepted a plan the others do not share")
    if not res["s2_peer_bank_untouched"]:
        fails.append("s2: the peer bank was registered before the digest check")
    if res["s3"]["keys"] != [LAYER] or not res["s3"]["finite"]:
        fails.append("s3: stage build returned the wrong layers or a non-finite output")
    if not res["s3"]["worst_rel_l2"] <= RULE_IQ:
        fails.append(f"s3: rel L2 {res['s3']['worst_rel_l2']:.3e} > {RULE_IQ}")
    status = "FAIL" if fails else "PASS"
    if a.rank == 0:
        a.output.write_text(json.dumps({"check": "T3-step7-stage-build", "status": status,
                                        "failures": fails, "rule_rel_l2": RULE_IQ, **res},
                                       indent=1) + "\n")
    print(json.dumps({"rank": a.rank, "status": status, "failures": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
