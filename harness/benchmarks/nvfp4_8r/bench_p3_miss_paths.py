"""P3 speed bench (the test host, one card + its host CPU): cold-expert miss paths.

Measures both arms of plan step P3, before any engine build.
Layer 5 rank 0 (real weights, P1 manifest split, real reference routes with k
distinct cold experts forced in; k = 0..6) at M = 1 and M = 6:
- t_res(M,k): all-resident CUDA-graph time; t_zc(M,k): current zero-copy split
  graph time; overhead of the split with zero cold routes = t_zc(M,0)-t_res(M,0);
- arm (a1) copy engine: host->HBM DMA of k prepacked experts (pinned) + the
  device->host sync a host-issued copy needs to learn the ids;
- arm (a2) SM gather (graph-safe): index_select of k experts from the
  registered host arena into an HBM ring, as forward_prefill stages them;
- arm (b) host compute: AVX2 CPU kernel (cpu_mxfp4_moe.c) runs k cold experts
  x 4 TP slices (all four ranks' cold work shares one host CPU), one routed
  row each; validated first against the raw-NVFP4 fp32 references; plus the
  activation round trip (d2h x, sync, h2d y).
Composed per-layer critical paths are reported with lower/upper bounds (the
parts are MEASURED, the composition is an ESTIMATE).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native

from . import cpu_moe
from .accept_p2 import Inconclusive, gpu_gate, time_graph
from .accept_p3_layer import graph_of

LAYER, RANK, KS = 5, 0, range(0, 7)


def routes(base: np.ndarray, hot: list[int], cold: list[int], m: int, k: int, seed: int):
    gen = np.random.default_rng(seed)
    ids = base[:m].copy()
    hot_set = set(hot)
    for row in ids:  # k = 0 baseline: hot experts only
        for j, e in enumerate(row):
            if int(e) not in hot_set:
                row[j] = gen.choice([h for h in hot if h not in row])
    picks = gen.choice(cold, k, replace=False)
    for i, c in enumerate(picks):  # k distinct cold experts
        r, j = (i, 0) if m > 1 else (0, i)
        ids[r, j] = c
    return ids, [int(c) for c in picks]


def event_us(fn, reps: int = 20) -> float:
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return round(statistics.median(times), 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--card", type=int, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        gate = gpu_gate(a.card)
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    dev = torch.device("cuda", 0)
    refs = dict(np.load(a.refs, allow_pickle=False))
    split = native.read_manifest(a.packs)[LAYER][RANK]
    hot, cold = split["hot"], split["cold"]
    raw = native.read_rank_experts(a.packs, LAYER, RANK)
    full = native.NativeV41Moe(native.NativeExpertSet.from_raw(torch.from_numpy(raw), dev),
                               torch.arange(384, dtype=torch.int32, device=dev), 16)
    hs, cs = native.split_maps(hot, cold, dev)
    cold_set = native.NativeExpertSet.from_raw_host(torch.from_numpy(raw[cold]), dev)
    moe = native.NativeV41Moe(native.NativeExpertSet.from_raw(torch.from_numpy(raw[hot]), dev),
                              hs, 16, cold=cold_set, cold_slot_of_expert=cs)
    res: dict = {"gate": gate, "layer": LAYER, "rank": RANK, "graph_us": {}, "copy_us": {},
                 "cpu": {}, "roundtrip_us": {}}
    x6 = torch.from_numpy(refs["x_m6_s0"]).to(dev)
    t6 = torch.from_numpy(refs["tw_m6_s0"]).to(dev)
    for m in (1, 6):
        for k in KS:
            ids, _ = routes(refs["ids_m6_s0"], hot, cold, m, k, 100 * m + k)
            i = torch.from_numpy(ids).to(dev)
            out = {}
            for name, mod in (("res", full), ("zc", moe)):
                graph, (sx, si, st), _ = graph_of(mod, x6[:m], i, t6[:m])
                out[name] = round(time_graph(graph), 1)
                del graph
            res["graph_us"][f"M{m}_k{k}"] = out
            print(json.dumps({f"M{m}_k{k}": out}), flush=True)
    # (a1) copy engine: pinned prepacked experts -> HBM ring, plus the id sync.
    parts = (cold_set.q13, cold_set.s13, cold_set.q2, cold_set.s2)
    per = [t[0].numel() for t in parts]
    pinned = torch.empty((len(cold), sum(per)), dtype=torch.uint8, pin_memory=True)
    for j, t in enumerate(parts):
        off = sum(per[:j])
        pinned[:, off : off + per[j]].copy_(t.cpu().view(len(cold), -1))
    ring = torch.empty((6, sum(per)), dtype=torch.uint8, device=dev)
    ids_dev = torch.zeros((6, 6), dtype=torch.int32, device=dev)
    ids_host = torch.empty((6, 6), dtype=torch.int32, pin_memory=True)
    for k in range(1, 7):
        sel = list(range(k))

        def dma(sel=sel):
            for slot, e in enumerate(sel):
                ring[slot].copy_(pinned[e], non_blocking=True)

        def gather(k=k):
            idx = torch.arange(k, device=dev)
            for t in parts:
                torch.index_select(t, 0, idx)

        res["copy_us"][f"k{k}"] = {"dma": event_us(dma), "sm_gather": event_us(gather)}
    sync = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids_host.copy_(ids_dev, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        sync.append((time.perf_counter() - t0) * 1e6)
    res["copy_us"]["id_sync"] = round(statistics.median(sync), 1)
    # (b) host compute: validate, then time k experts x 4 slices, one row each.
    dll = cpu_moe.build(a.build)
    cpu = cpu_moe.CpuMoe(dll)
    for m in (1, 6):
        x = refs[f"x_m{m}_s0"].astype(np.float32)
        got = cpu.layer(raw, x, refs[f"ids_m{m}_s0"], refs[f"tw_m{m}_s0"])
        ref = refs[f"ref_r0_m{m}_s0"]
        res["cpu"][f"rel_l2_M{m}"] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    if not max(res["cpu"][f"rel_l2_M{m}"] for m in (1, 6)) < 1e-2:
        raise SystemExit(f"CPU kernel disagrees with the references: {res['cpu']}")
    slices = [raw] + [native.read_rank_experts(a.packs, layer, r)
                      for layer, r in ((5, 3), (12, 0), (12, 3), (27, 0), (27, 3))]
    x = refs["x_m6_s0"].astype(np.float32)
    outbuf = np.zeros((6, 5120), np.float32)
    one = np.ones(1, np.float32)
    cursor = 0
    for k in range(1, 7):
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            for j in range(k):
                for s in range(4):  # four TP slices, distinct memory each rep
                    src = slices[(cursor + s) % len(slices)]
                    cpu.expert(src[(cursor * 7 + j) % 384], x, np.array([j]), one, outbuf)
                cursor += 1
            times.append((time.perf_counter() - t0) * 1e6)
        res["cpu"][f"k{k}_x4slices_us"] = round(statistics.median(times), 1)
    res["cpu"]["threads"] = a.threads
    # activation round trip for (b): d2h x (fp16 [6,5120]), sync, h2d y (fp32).
    xh = torch.empty((6, 5120), dtype=torch.float16, pin_memory=True)
    yh = torch.zeros((6, 5120), dtype=torch.float32, pin_memory=True)
    rt = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xh.copy_(x6, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        yd = yh.to(dev, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        rt.append((time.perf_counter() - t0) * 1e6)
    del yd
    res["roundtrip_us"]["d2h_sync_h2d_sync"] = round(statistics.median(rt), 1)
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": "BENCH_DONE", "copy_us": res["copy_us"], "cpu": res["cpu"],
                      "roundtrip_us": res["roundtrip_us"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
