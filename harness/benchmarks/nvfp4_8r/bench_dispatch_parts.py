"""Plan 03 step 1: where the split layer's no-miss overhead goes (the test host, one card).

Layer 5 rank 0, real packs, P1 manifest hot/cold split, the reference routes of
bench_p3_miss_paths with every cold route replaced by a hot one (k = 0, the
same seeds), at M = 1 and M = 6. Each part is captured alone into a CUDA graph
and timed with accept_p2.time_graph (GPU time of one replay, the host queue
kept ahead by a sleep kernel). Parts:

  a  _compact on the split key (argsort, gather, compare, cumsum, scatters)
  b  _groups (all-hot plan: map gather + _compact + casts)
  c  _split_groups (hot + host plan: the index math of a split layer)
  d  one W13 launch on the HBM bank, every group empty
  d2 one W2 launch on the HBM bank, every group empty
  e  one W13 launch on the host bank, every group empty
  e2 one W2 launch on the host bank, every group empty
  f  full all-hot forward (all 384 experts resident)
  g  full split forward, no route on the host bank
  f_again  (f) timed again after (g)
  h  the group-plan kernel alone, split layer (replaces c)
  h_hot  the group-plan kernel alone, all-resident layer (replaces b)
  i  full split forward with the kernel, no route on the host bank (vs g)
  i_hot  full all-hot forward with the kernel (vs f)

"Every group empty" is what the host launches get in a no-miss split round:
goff = n everywhere, gids = 0. Each part is timed OUTER times (each one
time_graph median of REPS replays) after WARM_REPLAYS untimed replays; the
receipt keeps every value, the median, and "settled" (every repeat within 3 %
of the median). The first run without warm-up (parts-20261111T010637Z) showed
(f) at M = 1 drifting 213 -> 153 us over its repeats.

Acceptance rules, written before the data (plan 03 sections 2 and 6.1):
- bench check: (g) - (f) within 10 % of the receipt
  logs/testhost/p3-bench-20261109T185349Z.json (+63.5 us at M = 1, +101.4 us
  at M = 6). FAIL means the bench does not measure what the plan assumed.
- falsifier: (c) < 20 us at M = 6 and (e) + (e2) < 10 us at M = 6 closes the
  plan (neither part can pay).
The gate (no compute process on the card, power limit, MemAvailable) is run by
the launcher with one nvidia-smi call and passed in as --gate-json.
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
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native

from .accept_p2 import time_graph
from .bench_p3_miss_paths import routes

LAYER, RANK = 5, 0
TABLE_US = {1: 63.5, 6: 101.4}  # (g) - (f), p3-bench-20261109T185349Z.json
CHECK_TOL = 0.10
OUTER = 5
WARM_REPLAYS = 50
SETTLED_TOL = 0.03


def capture(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def timed(fn) -> dict[str, object]:
    graph = capture(fn)
    for _ in range(WARM_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    values = [round(time_graph(graph), 2) for _ in range(OUTER)]
    del graph
    med = statistics.median(values)
    settled = all(abs(v - med) <= SETTLED_TOL * med for v in values)
    return {"median_us": round(med, 2), "values_us": values, "settled": settled}


def empty_launch(moe: native.NativeV41Moe, bank: native.NativeExpertSet, w2: bool,
                 x: torch.Tensor, m: int):
    s = m * native.TOP_K
    dev = x.device
    perm = torch.arange(s, dtype=torch.int32, device=dev)
    gids = torch.zeros(s, dtype=torch.int32, device=dev)
    goff = torch.full((s + 1,), s, dtype=torch.int32, device=dev)
    if w2:
        args = (moe.inter[:s], bank.q2, bank.s2, bank.g2, perm, gids, goff, native.TOP_K,
                moe.y2[:s], True, m, moe.split2, 1, int(bank.order))
    else:
        args = (x, bank.q13, bank.s13, bank.g1, perm, gids, goff, native.TOP_K,
                moe.y13[:s], False, m, moe.split13, 1, int(bank.order))
    return lambda: torch.ops._C.skinny_moe_qpn_sm70(*args)


def parts_for(full: native.NativeV41Moe, moe: native.NativeV41Moe,
              full_k: native.NativeV41Moe, moe_k: native.NativeV41Moe, x, i, t, m: int) -> dict:
    sx, si, st = x.clone(), i.clone(), t.clone()
    flat = si.reshape(-1).long()
    nh = moe.experts.num_slots
    hot = moe.slot_of_expert[flat].long()
    key = torch.where(hot >= 0, hot, nh + moe.cold_slot_of_expert[flat].long())
    assert moe.cold is not None
    return {
        "a": lambda: native.NativeV41Moe._compact(key),
        "b": lambda: full._groups(si),
        "c": lambda: moe._split_groups(si),
        "d": empty_launch(moe, moe.experts, False, sx, m),
        "d2": empty_launch(moe, moe.experts, True, sx, m),
        "e": empty_launch(moe, moe.cold, False, sx, m),
        "e2": empty_launch(moe, moe.cold, True, sx, m),
        "f": lambda: full.forward(sx, si, st),
        "g": lambda: moe.forward(sx, si, st),
        "f_again": lambda: full.forward(sx, si, st),
        "h": lambda: moe_k._planner.plan(si),
        "h_hot": lambda: full_k._planner.plan(si),
        "i": lambda: moe_k.forward(sx, si, st),
        "i_hot": lambda: full_k.forward(sx, si, st),
    }


def build(packs: Path, dev: torch.device, max_tokens: int):
    split = native.read_manifest(packs)[LAYER][RANK]
    hot, cold = split["hot"], split["cold"]
    raw = native.read_rank_experts(packs, LAYER, RANK)
    full_set = native.NativeExpertSet.from_raw(torch.from_numpy(raw), dev)
    ident = torch.arange(384, dtype=torch.int32, device=dev)
    hs, cs = native.split_maps(hot, cold, dev)
    cold_set = native.NativeExpertSet.from_raw_host(torch.from_numpy(raw[cold]), dev)
    hot_set = native.NativeExpertSet.from_raw(torch.from_numpy(raw[hot]), dev)
    objs = []
    for mode in (gplan.GroupPlan.CHAIN, gplan.GroupPlan.KERNEL):
        objs.append(native.NativeV41Moe(full_set, ident, max_tokens, group_plan=mode))
        objs.append(native.NativeV41Moe(hot_set, hs, max_tokens, cold=cold_set,
                                        cold_slot_of_expert=cs, group_plan=mode))
    full, moe, full_k, moe_k = objs
    return full, moe, full_k, moe_k, hot, cold


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    gate = json.loads(a.gate_json)
    dev = torch.device("cuda", 0)
    refs = dict(np.load(a.refs, allow_pickle=False))
    full, moe, full_k, moe_k, hot, cold = build(a.packs, dev, a.max_tokens)
    res: dict = {
        "max_tokens": a.max_tokens,
        "compiled": {name: {str(m): dict(zip(("block", "warps", "regs", "spills"), v))
                            for m, v in o._planner.compiled.items()}
                     for name, o in (("resident", full_k), ("split", moe_k))},
        "gate": gate, "layer": LAYER, "rank": RANK, "n_hot": len(hot), "n_cold": len(cold),
        "device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
        "triton": triton.__version__, "outer": OUTER,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "parts_us": {},
    }
    x6 = torch.from_numpy(refs["x_m6_s0"]).to(dev)
    t6 = torch.from_numpy(refs["tw_m6_s0"]).to(dev)
    for m in (1, 6):
        ids, _ = routes(refs["ids_m6_s0"], hot, cold, m, 0, 100 * m)
        i = torch.from_numpy(ids).to(dev)
        assert bool((moe.slot_of_expert[i.long()] >= 0).all()), "a route left the hot bank"
        row = {}
        for name, fn in parts_for(full, moe, full_k, moe_k, x6[:m], i, t6[:m], m).items():
            row[name] = timed(fn)
            print(json.dumps({f"M{m}": {name: row[name]["median_us"]}}), flush=True)
        res["parts_us"][f"M{m}"] = row
    check = {}
    for m in (1, 6):
        row = res["parts_us"][f"M{m}"]
        diff = row["g"]["median_us"] - row["f"]["median_us"]
        ok = abs(diff - TABLE_US[m]) <= CHECK_TOL * TABLE_US[m]
        check[f"M{m}"] = {"g_minus_f_us": round(diff, 2), "table_us": TABLE_US[m],
                          "f_settled": row["f"]["settled"], "g_settled": row["g"]["settled"],
                          "verdict": "PASS" if ok else "FAIL"}
    res["bench_check"] = check
    res["kernel_us"] = {
        f"M{m}": {
            "split_plan_chain_c": res["parts_us"][f"M{m}"]["c"]["median_us"],
            "split_plan_kernel_h": res["parts_us"][f"M{m}"]["h"]["median_us"],
            "resident_plan_chain_b": res["parts_us"][f"M{m}"]["b"]["median_us"],
            "resident_plan_kernel_h_hot": res["parts_us"][f"M{m}"]["h_hot"]["median_us"],
            "split_forward_chain_g": res["parts_us"][f"M{m}"]["g"]["median_us"],
            "split_forward_kernel_i": res["parts_us"][f"M{m}"]["i"]["median_us"],
            "resident_forward_chain_f": res["parts_us"][f"M{m}"]["f"]["median_us"],
            "resident_forward_kernel_i_hot": res["parts_us"][f"M{m}"]["i_hot"]["median_us"],
        }
        for m in (1, 6)
    }
    m6 = res["parts_us"]["M6"]
    index_math = m6["c"]["median_us"]
    two_empty = m6["e"]["median_us"] + m6["e2"]["median_us"]
    closes = index_math < 20.0 and two_empty < 10.0
    res["falsifier"] = {
        "index_math_M6_us": index_math, "two_empty_host_launches_M6_us": round(two_empty, 2),
        "rule": "close if index_math_M6 < 20 us and two_empty_M6 < 10 us",
        "verdict": "CLOSE" if closes else "GO",
    }
    res["unsettled_parts"] = [f"{mk}.{k}" for mk, row in res["parts_us"].items()
                              for k, v in row.items() if not v["settled"]]
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": "BENCH_DONE", "bench_check": check, "falsifier": res["falsifier"],
                      "kernel_us": res["kernel_us"],
                      "unsettled_parts": res["unsettled_parts"]}))
    return 0 if all(v["verdict"] == "PASS" and v["f_settled"] for v in check.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
