"""Three-tier route plan alone, chain against kernel (measurement, no rule).

hot_iq shape: HOT experts in the HBM native set, every other expert WARM (IQ
record on the card), no HOST tier, so decode has one QPN launch
(host_in_decode False). Objects at max_tokens 6 with placeholder banks: the
plan reads only the slot maps and the routes, never the weights.

Shapes: A = layer 5 rank 0 manifest split (hot set HOT, cold set WARM);
B = 200 HOT / 184 WARM (the stage-0 average of an earlier mix4b boot,
3,998 HOT over 20 layers), HOT = the first 200 of the manifest hot list.
Routes: refs ids_m6_s0[:M]. Parts per shape and M: chain = _tier_groups,
kernel = the planner's one launch. Timing as bench_dispatch_parts (graph of the
part alone, 50 warm replays, five time_graph medians, settled within 3 %).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import triton

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

from .bench_dispatch_parts import timed
from .test_group_plan import placeholder_iq, placeholder_set

LAYER, RANK, MAX_TOKENS = 5, 0, 6
HOT, WARM = int(tiers.Tier.HOT), int(tiers.Tier.WARM)


def three_tier(tier: np.ndarray, mode, dev):
    n_hot = int((tier == HOT).sum())
    return nat.NativeV41Moe.three_tier(
        placeholder_set(n_hot, dev), placeholder_set(nat.NUM_EXPERTS - n_hot, dev),
        placeholder_iq(int((tier == WARM).sum()), dev), tier, MAX_TOKENS, None, group_plan=mode)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    hot_list = sorted(nat.read_manifest(a.packs)[LAYER][RANK]["hot"])
    shapes = {}
    for name, hot in (("A_manifest", hot_list), ("B_200_hot", hot_list[:200])):
        tier = np.full(nat.NUM_EXPERTS, WARM, np.int8)
        tier[hot] = HOT
        shapes[name] = tier
    refs = dict(np.load(a.refs, allow_pickle=False))
    res: dict = {"gate": json.loads(a.gate_json), "max_tokens": MAX_TOKENS, "torch": torch.__version__,
                 "triton": triton.__version__, "device": torch.cuda.get_device_name(dev), "shapes": {}}
    for name, tier in shapes.items():
        chain = three_tier(tier, gplan.GroupPlan.CHAIN, dev)
        kern = three_tier(tier, gplan.GroupPlan.KERNEL, dev)
        assert chain.warm is not None and not chain.warm.host_in_decode
        row = {"n_hot": int((tier == HOT).sum()), "n_warm": int((tier == WARM).sum()),
               "compiled": {str(k): dict(zip(("block", "warps", "regs", "spills"), v))
                            for k, v in kern._planner.compiled.items()}}
        for m in (1, 6):
            si = torch.from_numpy(refs["ids_m6_s0"][:m]).to(dev).clone()
            # same launch tensors, checked once before timing
            lc, lk = chain._tier_groups(si), kern._launches(si)
            n = m * nat.TOP_K
            if len(lc) != 1 or len(lk) != 1:
                raise AssertionError(f"{name} M{m}: launches {len(lc)} / {len(lk)}")
            for (_, pc, gc, oc), (_, pk, gk, ok) in zip(lc, lk):
                if not (torch.equal(pc[:n], pk[:n]) and torch.equal(gc[:n], gk[:n])
                        and torch.equal(oc[:n + 1], ok[:n + 1])):
                    raise AssertionError(f"{name} M{m}: kernel plan differs from the chain")
            row[f"M{m}"] = {"chain_tier_groups": timed(lambda: chain._tier_groups(si)),
                            "kernel_plan": timed(lambda: kern._planner.plan(si))}
        res["shapes"][name] = row
        del chain, kern
        torch.cuda.empty_cache()
    unsettled = [f"{s}.{m}.{k}" for s, r in res["shapes"].items() for m in ("M1", "M6")
                 for k, v in r[m].items() if not v["settled"]]
    res["unsettled_parts"] = unsettled
    res["status"] = "BENCH_DONE"
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": "BENCH_DONE", "us": {s: {m: {k: v["median_us"] for k, v in r[m].items()}
                                                         for m in ("M1", "M6")} for s, r in res["shapes"].items()},
                      "unsettled": unsettled}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
