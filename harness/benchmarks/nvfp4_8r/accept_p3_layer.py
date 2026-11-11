"""P3 layer gate (the test host, one card): cold experts in pinned host RAM.

For layers 5/12/27 (rank 0), the P1 manifest's cold experts live in pinned
host memory read zero-copy by the QPN kernels; hot experts stay in HBM.
Pass rules, all required:
- split (hot HBM + cold host) output is bit-equal to the all-resident P2
  module on every route: the reference routes and the same routes with slot 0
  of every token forced to a cold expert (seeded draw from the manifest);
- split output vs the raw-NVFP4 fp32 reference: rel L2 < 1e-2;
- CUDA graphs of the split module at M = 1..6 replay bit-equal to the
  all-resident eager output on a no-cold and a forced-cold route;
- prefill (forward_prefill, M = 64 and 512): split output bit-equal to the
  all-resident forward_prefill on natural and forced-cold routes (forced
  routes stage several cold batches), and rel L2 < 1e-2 vs the raw-NVFP4
  fp32 prefill reference (make_p2_refs --ms 64 512).
Logged, not gated: cold-route fraction and eager/graph us, split vs resident.
This is the layer-level half of PLAN P3; the truncated-model greedy check
needs the model integration and runs after it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native

from .accept_p2 import Inconclusive, gpu_gate, rel_l2, time_eager, time_graph

RANK = 0


def forced(ids: torch.Tensor, cold: list[int], seed: int) -> torch.Tensor:
    """Slot 0 of every token -> a cold expert not already routed to it."""
    gen = np.random.default_rng(seed)
    out = ids.clone().cpu()
    for row in range(out.size(0)):
        options = [e for e in cold if e not in out[row].tolist()]
        out[row, 0] = int(gen.choice(options))
    return out.to(ids.device)


def graph_of(moe, x, i, t):
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
    return graph, (sx, si, st), out


def run_layer(packs: Path, refs_path: Path, prefill_path: Path, dev: torch.device) -> dict:
    with np.load(refs_path, allow_pickle=False) as data:
        refs = {k: data[k] for k in data.files}
    meta = json.loads(bytes(refs["meta"]).decode())
    layer, samples = meta["layer"], meta["samples"]
    split = native.read_manifest(packs)[layer][RANK]
    raw = native.read_rank_experts(packs, layer, RANK)
    full = native.NativeV41Moe(
        native.NativeExpertSet.from_raw(torch.from_numpy(raw), dev),
        torch.arange(384, dtype=torch.int32, device=dev), max_tokens=16, group_plan=gplan.GroupPlan.CHAIN,
    )
    hot_slot, cold_slot = native.split_maps(split["hot"], split["cold"], dev)
    hot = native.NativeExpertSet.from_raw(torch.from_numpy(raw[split["hot"]]), dev)
    cold = native.NativeExpertSet.from_raw_host(torch.from_numpy(raw[split["cold"]]), dev)
    del raw
    moe = native.NativeV41Moe(hot, hot_slot, max_tokens=16, cold=cold,
                              cold_slot_of_expert=cold_slot, group_plan=gplan.GroupPlan.CHAIN)
    cold_set = set(split["cold"])
    res: dict = {"layer": layer, "hot": len(split["hot"]), "cold": len(split["cold"]),
                 "bit_equal": {}, "worst_rel_l2": {}, "timing_us": {}, "graphs": {}}
    for m in meta["ms"]:
        eq_nat, eq_forced, worst = True, True, 0.0
        cold_routes = [0, 0]
        t_split, t_full, t_split_f, t_full_f = [], [], [], []
        for s in range(samples):
            x = torch.from_numpy(refs[f"x_m{m}_s{s}"]).to(dev)
            i = torch.from_numpy(refs[f"ids_m{m}_s{s}"]).to(dev)
            t = torch.from_numpy(refs[f"tw_m{m}_s{s}"]).to(dev)
            f = forced(i, split["cold"], 7 * layer + 100 * m + s)
            cold_routes[0] += sum(int(e) in cold_set for e in i.flatten().tolist())
            cold_routes[1] += sum(int(e) in cold_set for e in f.flatten().tolist())
            a = moe.forward(x, i, t).clone()
            eq_nat &= bool(torch.equal(a, full.forward(x, i, t)))
            worst = max(worst, rel_l2(a, refs[f"ref_r{RANK}_m{m}_s{s}"]))
            b = moe.forward(x, f, t).clone()
            eq_forced &= bool(torch.equal(b, full.forward(x, f, t)))
            t_split.append(time_eager(moe, (x, i, t)))
            t_full.append(time_eager(full, (x, i, t)))
            t_split_f.append(time_eager(moe, (x, f, t)))
            t_full_f.append(time_eager(full, (x, f, t)))
        slots = samples * m * native.TOP_K
        res["bit_equal"][str(m)] = {"natural": eq_nat, "forced_cold": eq_forced}
        res["worst_rel_l2"][str(m)] = worst
        res["timing_us"][str(m)] = {
            "cold_route_fraction_natural": cold_routes[0] / slots,
            "cold_route_fraction_forced": cold_routes[1] / slots,
            "split_natural": round(statistics.mean(t_split), 2),
            "resident_natural": round(statistics.mean(t_full), 2),
            "split_forced": round(statistics.mean(t_split_f), 2),
            "resident_forced": round(statistics.mean(t_full_f), 2),
        }
    x6 = torch.from_numpy(refs["x_m6_s0"]).to(dev)
    i6 = torch.from_numpy(refs["ids_m6_s0"]).to(dev)
    t6 = torch.from_numpy(refs["tw_m6_s0"]).to(dev)
    f6 = forced(i6, split["cold"], 99 + layer)
    for m in range(1, 7):
        graph, (sx, si, st), out = graph_of(moe, x6[:m], i6[:m], t6[:m])
        equal = []
        for ids in (i6, f6):
            sx.copy_(x6[:m]), si.copy_(ids[:m]), st.copy_(t6[:m])
            graph.replay()
            torch.cuda.synchronize()
            replayed = out.clone()
            equal.append(bool(torch.equal(replayed, full.forward(x6[:m], ids[:m], t6[:m]))))
        res["graphs"][str(m)] = {
            "replay_bit_equal": equal, "graph_us_forced": round(time_graph(graph), 2)
        }
        del graph
    res["prefill"] = run_prefill(moe, full, prefill_path, split["cold"], layer, dev)
    return res


def run_prefill(moe, full, path: Path, cold: list[int], layer: int, dev) -> dict:
    with np.load(path, allow_pickle=False) as data:
        refs = {k: data[k] for k in data.files}
    meta = json.loads(bytes(refs["meta"]).decode())
    if meta["layer"] != layer or RANK not in meta["ranks"]:
        raise ValueError(f"prefill refs {path} do not cover L{layer} r{RANK}")
    out: dict = {}
    for m in meta["ms"]:
        entry = {"bit_equal_natural": True, "bit_equal_forced": True, "worst_rel_l2": 0.0,
                 "split_us": [], "resident_us": [], "cold_experts_routed_forced": []}
        for s in range(meta["samples"]):
            x = torch.from_numpy(refs[f"x_m{m}_s{s}"]).to(dev)
            i = torch.from_numpy(refs[f"ids_m{m}_s{s}"]).to(dev)
            t = torch.from_numpy(refs[f"tw_m{m}_s{s}"]).to(dev)
            f = forced(i, cold, 3 * layer + m + s)
            a = moe.forward_prefill(x, i, t)
            entry["bit_equal_natural"] &= bool(torch.equal(a, full.forward_prefill(x, i, t)))
            entry["worst_rel_l2"] = max(entry["worst_rel_l2"],
                                        rel_l2(a, refs[f"ref_r{RANK}_m{m}_s{s}"]))
            b = moe.forward_prefill(x, f, t)
            entry["bit_equal_forced"] &= bool(torch.equal(b, full.forward_prefill(x, f, t)))
            entry["cold_experts_routed_forced"].append(
                len({int(e) for e in f.flatten().tolist()} & set(cold)))
            for key, mod, ids in (("split_us", moe, f), ("resident_us", full, f)):
                torch.cuda.synchronize()
                a0, b0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a0.record()
                mod.forward_prefill(x, ids, t)
                b0.record()
                b0.synchronize()
                entry[key].append(round(a0.elapsed_time(b0) * 1000.0, 1))
        out[str(m)] = entry
    return out


def verdict(layers: list[dict]) -> list[str]:
    fails = []
    for r in layers:
        for m, e in r["bit_equal"].items():
            if not (e["natural"] and e["forced_cold"]):
                fails.append(f"L{r['layer']} M{m} split != resident {e}")
        for m, v in r["worst_rel_l2"].items():
            if not v < 1e-2:
                fails.append(f"L{r['layer']} M{m} rel L2 {v}")
        for m, g in r["graphs"].items():
            if not all(g["replay_bit_equal"]):
                fails.append(f"L{r['layer']} graph M{m} replay != resident")
        for m, e in r["prefill"].items():
            if not (e["bit_equal_natural"] and e["bit_equal_forced"]):
                fails.append(f"L{r['layer']} prefill M{m} split != resident")
            if not e["worst_rel_l2"] < 1e-2:
                fails.append(f"L{r['layer']} prefill M{m} rel L2 {e['worst_rel_l2']}")
    return fails


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs", type=Path, nargs="+", required=True)
    p.add_argument("--prefill-refs", type=Path, nargs="+", required=True)
    p.add_argument("--card", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        gate = gpu_gate(a.card)
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    dev = torch.device("cuda", 0)
    if len(a.refs) != len(a.prefill_refs):
        raise SystemExit("one prefill reference file per decode reference file")
    layers = [run_layer(a.packs, r, pf, dev) for r, pf in zip(a.refs, a.prefill_refs)]
    fails = verdict(layers)
    receipt = {"gate": gate, "layers": layers,
               "status": "P3_LAYER_PASS" if not fails else "P3_LAYER_FAIL",
               "failures": fails, "p3_accepted": False}
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"status": receipt["status"], "failures": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
