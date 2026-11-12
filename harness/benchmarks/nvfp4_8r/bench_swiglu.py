"""CAMPAIGN 8.4 decision bench (the test host, one card): decode SwiGLU chain
against the one-launch kernel. The rule is in the agent folder
(DECISION-RULE-swiglu.md), written before this bench's data.

Layers 5, 12, 27 rank 0, real packs, max_tokens 6 objects in production's
hot_iq shape (manifest hot set HOT, the rest WARM, no HOST), group plan
kernel (as installed). The IQ op is replaced in both arms by a fixed zero
partial (the real op needs four ranks' CUDA-IPC peer bank). M = 1 and 6,
routes and inputs refs ids/x/tw_m6_s0[:M].

Parts per layer and M (us):
  j            the SwiGLU chain alone on the chain object's y13 / inter
  k            the kernel alone on the kernel object's y13 / inter
  fwd_chain    three-tier forward, swiglu chain
  fwd_kernel   the same, swiglu kernel
  fwd_chain_again, fwd_kernel_again  drift check (information)
Graph: as bench_dispatch_parts.timed (one graph per part, 50 warm replays,
five time_graph medians, settled within 3 %). Eager: 20 warm calls, then
five repeats of 200 calls timed with CUDA events, median per call.
Before timing each layer and M, the two forwards' outputs are checked
bit-equal (int32 view).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import triton

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

from .bench_dispatch_parts import timed
from .test_group_plan import placeholder_iq
from .test_swiglu import bits_equal, chain_into, zero_iq

MAX_TOKENS = 6
EAGER_WARM, EAGER_REPS, EAGER_OUTER = 20, 200, 5
D6_RULE, D1_RULE = -15.0, 1.03
LAYERS = (5, 12, 27)
RULE_PARTS = ("fwd_chain", "fwd_kernel")


def eager(fn) -> dict[str, object]:
    for _ in range(EAGER_WARM):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(EAGER_OUTER):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(EAGER_REPS):
            fn()
        b.record()
        b.synchronize()
        vals.append(round(a.elapsed_time(b) * 1000.0 / EAGER_REPS, 2))
    return {"median_us": round(statistics.median(vals), 2), "values_us": vals}


def objects(packs: Path, layer: int, dev: torch.device):
    split = nat.read_manifest(packs)[layer][0]
    raw = torch.from_numpy(nat.read_rank_experts(packs, layer, 0))
    hot_set = nat.NativeExpertSet.from_raw(raw[split["hot"]], dev)
    cold_set = nat.NativeExpertSet.from_raw_host(raw[split["cold"]], dev)
    del raw
    tier = np.full(nat.NUM_EXPERTS, int(tiers.Tier.WARM), np.int8)
    tier[split["hot"]] = int(tiers.Tier.HOT)
    iq = placeholder_iq(int((tier == int(tiers.Tier.WARM)).sum()), dev)
    out = {}
    for mode in (swig.Swiglu.CHAIN, swig.Swiglu.KERNEL):
        moe = nat.NativeV41Moe.three_tier(hot_set, cold_set, iq, tier, MAX_TOKENS, None,
                                          group_plan=gplan.GroupPlan.KERNEL, swiglu=mode, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2,
                                          iq_tail=nat.iqt.IqTail.TORCH)
        zero_iq(moe, dev)
        out[mode] = moe
    assert out[swig.Swiglu.KERNEL]._swiglu_kernel is not None
    assert out[swig.Swiglu.CHAIN]._swiglu_kernel is None
    return out, len(split["hot"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs-dir", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    res: dict = {"bench": "swiglu_8.4", "gate": json.loads(a.gate_json), "max_tokens": MAX_TOKENS,
                 "torch": torch.__version__, "triton": triton.__version__,
                 "device": torch.cuda.get_device_name(dev),
                 "swiglu_sha256": hashlib.sha256(Path(swig.__file__).read_bytes()).hexdigest(),
                 "native_sha256": hashlib.sha256(Path(nat.__file__).read_bytes()).hexdigest(),
                 "layers": {}}
    for layer in LAYERS:
        refs = dict(np.load(a.refs_dir / f"refs-L{layer:02d}.npz", allow_pickle=False))
        objs, n_hot = objects(a.packs, layer, dev)
        c, k = objs[swig.Swiglu.CHAIN], objs[swig.Swiglu.KERNEL]
        row: dict = {"n_hot": n_hot}
        for m in (1, 6):
            sx = torch.from_numpy(refs["x_m6_s0"][:m]).to(dev).clone()
            si = torch.from_numpy(refs["ids_m6_s0"][:m]).to(dev).clone()
            st = torch.from_numpy(refs["tw_m6_s0"][:m]).to(dev).clone()
            if not bits_equal(c.forward(sx, si, st).clone(), k.forward(sx, si, st).clone()):
                raise AssertionError(f"L{layer} M{m}: the two forwards differ before timing")
            s = m * nat.TOP_K
            yc, ic, yk, ik = c.y13[:s], c.inter[:s], k.y13[:s], k.inter[:s]
            parts = {
                "j": lambda: chain_into(yc, ic),
                "k": lambda: k._swiglu_kernel(yk, ik),
                "fwd_chain": lambda: c.forward(sx, si, st),
                "fwd_kernel": lambda: k.forward(sx, si, st),
                "fwd_chain_again": lambda: c.forward(sx, si, st),
                "fwd_kernel_again": lambda: k.forward(sx, si, st),
            }
            graph, eag = {}, {}
            for name, fn in parts.items():
                graph[name] = timed(fn)
                print(json.dumps({f"L{layer:02d}": {f"M{m}": {name: graph[name]["median_us"]}}}), flush=True)
            for name, fn in parts.items():
                eag[name] = eager(fn)
            row[f"M{m}"] = {"graph": graph, "eager": eag}
        res["layers"][f"L{layer:02d}"] = row
        del objs, c, k
        torch.cuda.empty_cache()

    L = res["layers"]
    g = lambda l, m, part: L[l][f"M{m}"]["graph"][part]["median_us"]  # noqa: E731
    names = [f"L{x:02d}" for x in LAYERS]
    d6 = {l: round(g(l, 6, "fwd_kernel") - g(l, 6, "fwd_chain"), 2) for l in names}
    d1 = {l: round(g(l, 1, "fwd_kernel") - g(l, 1, "fwd_chain"), 2) for l in names}
    unsettled = [f"{l}.M{m}.{p}" for l in names for m in (1, 6) for p in RULE_PARTS
                 if not L[l][f"M{m}"]["graph"][p]["settled"]]
    build = all(v <= D6_RULE for v in d6.values()) and all(v <= D1_RULE for v in d1.values())
    verdict = "INCONCLUSIVE (unsettled parts used)" if unsettled else ("BUILD" if build else "NO BUILD")
    mean6 = statistics.mean(d6.values())
    res["rule"] = {"d6_us": d6, "d1_us": d1, "d6_rule_max": D6_RULE, "d1_rule_max": D1_RULE,
                   "unsettled_used": unsettled, "verdict": verdict,
                   "predicted_ms_per_round_40_layers": round(mean6 * 40 / 1000, 3),
                   "d6_min_max": [min(d6.values()), max(d6.values())],
                   "j_minus_k": {l: {f"M{m}": round(g(l, m, "j") - g(l, m, "k"), 2) for m in (1, 6)}
                                 for l in names},
                   "again_d": {l: {f"M{m}": round(g(l, m, "fwd_kernel_again") - g(l, m, "fwd_chain_again"), 2)
                                   for m in (1, 6)} for l in names}}
    res["all_unsettled"] = [f"{l}.M{m}.{p}" for l in names for m in (1, 6)
                            for p, v in L[l][f"M{m}"]["graph"].items() if not v["settled"]]
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"verdict": verdict, "rule": {k: v for k, v in res["rule"].items()},
                      "medians": {l: {f"M{m}": {p: v["median_us"] for p, v in L[l][f"M{m}"]["graph"].items()}
                                      for m in (1, 6)} for l in names}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
