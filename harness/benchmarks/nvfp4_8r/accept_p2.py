"""P2 acceptance (the test host, one card): native V4.1 expert method on real packs.

Pass rules (step P2 of this build), all required:
- rel L2 < 1e-2 vs the raw-NVFP4 fp32 reference (make_p2_refs) at M = 1, 6, 12
  for every sample of layers 5/12/27 and every reference rank;
- CUDA graphs capture at M = 1..6 and replay bit-equal to eager on two routes;
- eager per-layer GPU time at M = 6 <= 1.10 x an earlier unpublished per-layer bench's all-QPN D for the
  same layer (MEASURED there: L5 395.01, L12 400.77, L27 366.72 us).
The GPU gate refuses (INCONCLUSIVE, exit 3) unless the card has no compute app,
its power limit is <= 150 W and MemAvailable >= 60 GiB. It never changes caps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native

REF_D_US = {
    5: {1: 169.09, 6: 395.01, 12: 566.4},
    12: {1: 173.57, 6: 400.77, 12: 581.38},
    27: {1: 174.34, 6: 366.72, 12: 519.81},
}
TIME_RULE = 1.10
REL_L2_RULE = 1e-2
SLEEP_CYCLES = 20_000_000
REPS = 7


class Inconclusive(RuntimeError):
    pass


def gpu_gate(card: int) -> dict[str, object]:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    if apps:
        raise Inconclusive(f"foreign compute apps present: {apps}")
    power = float(
        subprocess.run(
            ["nvidia-smi", "-i", str(card), "--query-gpu=power.limit",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    )
    if power > 150.0:
        raise Inconclusive(f"card {card} power limit {power} W > 150 W rule")
    avail = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if avail < 60 * 2**20:
        raise Inconclusive(f"MemAvailable {avail} kB < 60 GiB")
    return {"card": card, "power_limit_w": power, "mem_available_kb": avail}


def rel_l2(out: torch.Tensor, ref: np.ndarray) -> float:
    r = torch.from_numpy(ref).to(out.device, torch.float64)
    return float((out.double() - r).norm() / r.norm())


def sample(refs: dict[str, np.ndarray], m: int, s: int, dev: torch.device):
    return (
        torch.from_numpy(refs[f"x_m{m}_s{s}"]).to(dev),
        torch.from_numpy(refs[f"ids_m{m}_s{s}"]).to(dev),
        torch.from_numpy(refs[f"tw_m{m}_s{s}"]).to(dev),
    )


def time_eager(moe: native.NativeV41Moe, args) -> float:
    times = []
    for _ in range(REPS):
        torch.cuda._sleep(SLEEP_CYCLES)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        moe.forward(*args)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(times)


def time_graph(graph: torch.cuda.CUDAGraph) -> float:
    times = []
    for _ in range(REPS):
        torch.cuda._sleep(SLEEP_CYCLES)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(times)


def graph_check(moe: native.NativeV41Moe, refs, dev) -> dict[str, object]:
    """Capture at M = 1..6; replay two different routes bit-equal to eager."""
    out: dict[str, object] = {}
    x6a, i6a, t6a = sample(refs, 6, 0, dev)
    x6b, i6b, t6b = sample(refs, 6, 1, dev)
    for m in range(1, 7):
        sx, si, st = x6a[:m].clone(), i6a[:m].clone(), t6a[:m].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                moe.forward(sx, si, st)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = moe.forward(sx, si, st)
        equal = []
        for x, i, t in ((x6a, i6a, t6a), (x6b, i6b, t6b)):
            sx.copy_(x[:m]), si.copy_(i[:m]), st.copy_(t[:m])
            graph.replay()
            torch.cuda.synchronize()
            # The output may alias a persistent buffer: copy before eager runs.
            replayed = static_out.clone()
            eager = moe.forward(x[:m].clone(), i[:m].clone(), t[:m].clone()).clone()
            equal.append(bool(torch.equal(replayed, eager)))
        out[str(m)] = {"replay_bit_equal": equal, "graph_us": round(time_graph(graph), 2)}
        del graph
    return out


def run_layer(packs: Path, refs_path: Path, dev: torch.device) -> dict[str, object]:
    with np.load(refs_path, allow_pickle=False) as data:
        refs = {k: data[k] for k in data.files}
    meta = json.loads(bytes(refs["meta"]).decode())
    layer, ranks, samples = meta["layer"], meta["ranks"], meta["samples"]
    result: dict[str, object] = {"layer": layer, "ranks": {}}
    for rank in ranks:
        t0 = time.monotonic()
        raw = native.read_rank_experts(packs, layer, rank)
        experts = native.NativeExpertSet.from_raw(torch.from_numpy(raw), dev)
        del raw
        moe = native.NativeV41Moe(
            experts, torch.arange(384, dtype=torch.int32, device=dev), max_tokens=16
        )
        load_s = time.monotonic() - t0
        worst = {}
        for m in meta["ms"]:
            worst[str(m)] = max(
                rel_l2(moe.forward(*sample(refs, m, s, dev)), refs[f"ref_r{rank}_m{m}_s{s}"])
                for s in range(samples)
            )
        entry: dict[str, object] = {"load_s": round(load_s, 1), "worst_rel_l2": worst}
        if rank == ranks[0]:
            entry["eager_us"] = {
                str(m): round(statistics.mean(
                    time_eager(moe, sample(refs, m, s, dev)) for s in range(samples)
                ), 2)
                for m in meta["ms"]
            }
            entry["graphs"] = graph_check(moe, refs, dev)
        result["ranks"][str(rank)] = entry  # type: ignore[index]
        del moe, experts
        torch.cuda.empty_cache()
    return result


def verdict(layers: list[dict]) -> tuple[bool, list[str]]:
    fails = []
    for res in layers:
        layer = res["layer"]
        for rank, e in res["ranks"].items():
            for m, v in e["worst_rel_l2"].items():
                if not v < REL_L2_RULE:
                    fails.append(f"L{layer} r{rank} M{m} rel L2 {v}")
            if "eager_us" in e:
                us, limit = e["eager_us"]["6"], TIME_RULE * REF_D_US[layer][6]
                if not us <= limit:
                    fails.append(f"L{layer} M6 {us} us > {limit:.1f} us")
                for m, g in e["graphs"].items():
                    if not all(g["replay_bit_equal"]):
                        fails.append(f"L{layer} graph M{m} replay != eager")
    return not fails, fails


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs", type=Path, nargs="+", required=True)
    p.add_argument("--card", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        gate = gpu_gate(a.card)
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    dev = torch.device("cuda", 0)
    layers = [run_layer(a.packs, r, dev) for r in a.refs]
    ok, fails = verdict(layers)
    receipt = {"gate": gate, "layers": layers, "ref_d_us": REF_D_US,
               "status": "P2_PASS" if ok else "P2_FAIL", "failures": fails}
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"status": receipt["status"], "failures": fails}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
