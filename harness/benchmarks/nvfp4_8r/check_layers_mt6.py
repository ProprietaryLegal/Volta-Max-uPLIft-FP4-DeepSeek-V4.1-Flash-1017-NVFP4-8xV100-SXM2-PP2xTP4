"""Real-weight bit-equality of the group-plan kernel on production-sized objects.

test_group_plan's layer regression builds max_tokens 16 objects. This runs the
same comparison on max_tokens 6 objects, as the loader builds them: L05, L12,
L27 rank 0, resident (all 384 experts) and two-tier (manifest hot/cold), chain
against kernel, M = 1..6, refs samples s0 and s1, eager and CUDA-graph replay.

PASS only if every kernel output is bit-equal to the chain's (eager and graph),
finite, and graph equals eager. Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat

from .test_group_plan import graph_out

MAX_TOKENS = 6


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs-dir", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    res: dict = {"check": "layers_max_tokens_6", "gate": json.loads(a.gate_json),
                 "max_tokens": MAX_TOKENS, "layers": {}}
    fails: list[str] = []
    for layer in (5, 12, 27):
        with np.load(a.refs_dir / f"refs-L{layer:02d}.npz", allow_pickle=False) as data:
            refs = {k: data[k] for k in data.files}
        split = nat.read_manifest(a.packs)[layer][0]
        raw = torch.from_numpy(nat.read_rank_experts(a.packs, layer, 0))
        hs, cs = nat.split_maps(split["hot"], split["cold"], dev)
        full_set = nat.NativeExpertSet.from_raw(raw, dev)
        hot_set = nat.NativeExpertSet.from_raw(raw[split["hot"]], dev)
        cold_set = nat.NativeExpertSet.from_raw_host(raw[split["cold"]], dev)
        del raw
        ident = torch.arange(384, dtype=torch.int32, device=dev)
        row = {}
        for kind in ("resident", "two_tier"):
            objs = {}
            for mode in (gplan.GroupPlan.CHAIN, gplan.GroupPlan.KERNEL):
                if kind == "resident":
                    objs[mode] = nat.NativeV41Moe(full_set, ident, MAX_TOKENS, group_plan=mode)
                else:
                    objs[mode] = nat.NativeV41Moe(hot_set, hs, MAX_TOKENS, cold=cold_set,
                                                  cold_slot_of_expert=cs, group_plan=mode)
            checks = []
            for s in (0, 1):
                for m in range(1, MAX_TOKENS + 1):
                    x = torch.from_numpy(refs[f"x_m6_s{s}"][:m]).to(dev)
                    i = torch.from_numpy(refs[f"ids_m6_s{s}"][:m]).to(dev)
                    t = torch.from_numpy(refs[f"tw_m6_s{s}"][:m]).to(dev)
                    eager = {mo: o.forward(x, i, t).clone() for mo, o in objs.items()}
                    graph = {mo: graph_out(o, x, i, t) for mo, o in objs.items()}
                    c = {
                        "s": s, "M": m,
                        "eager_bit_equal": bool(torch.equal(eager[gplan.GroupPlan.CHAIN],
                                                            eager[gplan.GroupPlan.KERNEL])),
                        "graph_bit_equal": bool(torch.equal(graph[gplan.GroupPlan.CHAIN],
                                                            graph[gplan.GroupPlan.KERNEL])),
                        "graph_equals_eager": bool(torch.equal(graph[gplan.GroupPlan.KERNEL],
                                                               eager[gplan.GroupPlan.KERNEL])),
                        "finite": bool(torch.isfinite(eager[gplan.GroupPlan.KERNEL]).all()),
                    }
                    checks.append(c)
                    if not all(v for k, v in c.items() if k not in ("s", "M")):
                        fails.append(f"L{layer:02d} {kind} s{s} M{m}: {c}")
            row[kind] = checks
            del objs
        res["layers"][f"L{layer:02d}"] = row
        del full_set, hot_set, cold_set
        torch.cuda.empty_cache()
    res["failures"] = fails
    res["status"] = "FAIL" if fails else "PASS"
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    n = sum(len(v) for r in res["layers"].values() for v in r.values())
    print(json.dumps({"status": res["status"], "comparisons": n, "failures": fails[:5]}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
