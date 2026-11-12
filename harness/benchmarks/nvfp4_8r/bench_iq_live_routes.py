"""Card 8.5 (rank 7): what the compressed-expert (IQ) op costs at few live
routes. One process, one card, one layer per run, real packs and IQ slices.

The layer is the product three-tier object (build_layer, group plan
"kernel", no HOST tier) with 96 WARM experts (id % 4 == 2) and 288 HOT ones.
Each row of six routes goes to six distinct experts; exactly k of the 6 M
routes go to WARM experts (iq_live_core.live_routes, seeded). Parts, each
captured alone in a CUDA graph and timed like bench_dispatch_parts (3 eager
calls, capture, 50 warm replays, 5 repeats of the median of 7 replays, a
sleep kernel before each replay):

  a   _iq_partial alone (the IQ op: q8_plan, gate_up_fast, peer_gather,
      down_rows_fast), k live routes, the others -1
  b   the installed forward
  c   the same forward with the IQ op call replaced by a constant FP16 zero
      tensor (.float() and the add stay)
  a2  (a) captured and timed again after (b) and (c)   [informative]

Informative parts at M = 6: r = _iq_routes alone, t = .float() + add alone,
s<j> = (a) with every row on the same six experts, j of them WARM (6 j live
routes on j distinct experts). M = 1 runs a, b, c at k = 0, 1, 3, 6.

One process has no TP peers. self_peer() registers four buffers on this
card as the op's peer bank and marks the three absent ranks as arrived, so
peer_gather copies this rank's slot to three buffers and does not wait. The
cross-card copy and the wait for peers are in no number here; neither
depends on k (the copy length is fixed by M). After the first eager call of
a routing the own slot is copied into the other three slots, so the down
kernel reads the same kind of data it reads in a four-rank run.

Checks made on every routing before it is timed (a failure raises):
  the count of live routes is k, the plan's count of distinct live experts
  is the expected one, the op's output is finite (all zero at k = 0), the
  forward's output is bit-equal to (c)'s output plus the op's output, and a
  graph replay of (a) is bit-equal to the eager call.

The decision (DECISION-RULE.md, written before any data) needs the three
layers' receipts: decide_iq_live_routes.py reads them. --profile adds, after
the timing receipt is on disk, per-kernel GPU times inside replays
(torch.profiler, as diag_kernel_times) to <output stem>.prof.json.
Exit 0 when the receipt is written, 1 on a failed check.
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
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
from vllm.model_executor.layers.quantization import gguf_v41_moe as v41

from . import iq_live_core as core

RANK = 0
MAX_TOKENS = 6
OUTER = 5
WARM_REPLAYS = 50
REPS = 7
SLEEP_CYCLES = 20_000_000
PROFILE_REPLAYS = 30
ARRIVED = 0x7FFFFFFF  # a flag value no epoch this run reaches


def self_peer(dev: torch.device) -> list[torch.Tensor]:
    """The IQ op's peer bank for one process on one card (see the module text)."""
    native._require_iq_decode(RANK, dev)
    banks = [torch.zeros(v41.ROWS_BYTES, dtype=torch.uint8, device=dev) for _ in range(4)]
    v41.register_rows_peer(dev, RANK, [b.data_ptr() for b in banks], banks[RANK])
    flags = banks[RANK][v41.ROWS_FLAGS : v41.ROWS_FLAGS + 16].view(torch.int32)
    for r in range(4):
        if r != RANK:
            flags[r] = ARRIVED
    torch.cuda.synchronize(dev)
    native._IQ_PEER_CARDS.add(dev)
    return banks


def fill_peer_slots(banks: list[torch.Tensor], m: int) -> None:
    """Copy this rank's slot (just written by an eager call) into the slots of
    the three absent ranks of the own bank."""
    stride = (m * native.TOP_K * 648 + 255) // 256 * 256
    own = banks[RANK]
    for r in range(4):
        if r != RANK:
            own[r * stride : (r + 1) * stride].copy_(own[RANK * stride : (RANK + 1) * stride])
    torch.cuda.synchronize()


def time_graph(graph: torch.cuda.CUDAGraph) -> float:
    """accept_p2.time_graph: GPU time of one replay, us, median of REPS."""
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


def capture(fn):
    """(graph, the captured call's output), as bench_dispatch_parts.capture."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def timed(fn, want: torch.Tensor | None = None) -> dict:
    graph, out = capture(fn)
    for _ in range(WARM_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    if want is not None and not torch.equal(out, want):
        raise RuntimeError("a graph replay is not bit-equal to the eager call")
    values = [time_graph(graph) for _ in range(OUTER)]
    del graph, out
    return core.summarize(values)


def profile_graph(fn) -> dict:
    """diag_kernel_times.profile_graph: per-kernel GPU time inside replays."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    graph, out = capture(fn)
    for _ in range(WARM_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILE_REPLAYS):
            torch.cuda._sleep(SLEEP_CYCLES)
            graph.replay()
        torch.cuda.synchronize()
    del graph, out
    evs = sorted((e for e in prof.events() if e.device_type == DeviceType.CUDA
                  and e.time_range.end > e.time_range.start),
                 key=lambda e: e.time_range.start)
    replays: list[list] = []
    cur = None
    for e in evs:
        if "spin" in e.name or "sleep" in e.name.lower():
            cur = []
            replays.append(cur)
        elif cur is not None:
            cur.append(e)
    replays = [r for r in replays if r]
    if len(replays) != PROFILE_REPLAYS:
        raise RuntimeError(f"found {len(replays)} replays in the trace, want {PROFILE_REPLAYS}")
    counts = {len(r) for r in replays}
    if len(counts) != 1:
        raise RuntimeError(f"replays hold different kernel counts: {sorted(counts)}")
    kernels = []
    for pos in range(len(replays[0])):
        durs = [r[pos].time_range.end - r[pos].time_range.start for r in replays]
        gaps = [r[pos].time_range.start - r[pos - 1].time_range.end for r in replays] if pos else [0]
        kernels.append({"pos": pos, "name": replays[0][pos].name[:100],
                        "median_us": round(statistics.median(durs), 2),
                        "gap_before_median_us": round(statistics.median(gaps), 2)})
    spans = [r[-1].time_range.end - r[0].time_range.start for r in replays]
    sums = [sum(e.time_range.end - e.time_range.start for e in r) for r in replays]
    return {"kernels": kernels, "span_median_us": round(statistics.median(spans), 2),
            "kernel_sum_median_us": round(statistics.median(sums), 2),
            "gaps_median_us": round(statistics.median(s - k for s, k in zip(spans, sums)), 2),
            "n_kernels": len(replays[0])}


class Routing:
    """One (M, ids) routing on the layer: static inputs and its checks."""

    def __init__(self, moe, banks, x6, t6, ids: np.ndarray, live: int, distinct: int):
        dev = x6.device
        self.moe, self.m = moe, ids.shape[0]
        self.sx, self.st = x6[: self.m].clone(), t6[: self.m].clone()
        self.si = torch.from_numpy(ids).to(dev)
        self.iq_ids, _ = moe._iq_routes(self.si, self.st)
        self.iq_ids = self.iq_ids.clone()
        self.zero16 = torch.zeros((self.m, native.HIDDEN), dtype=torch.float16, device=dev)
        got = int((self.iq_ids >= 0).sum())
        if got != live:
            raise RuntimeError(f"{got} live routes on the card, want {live}")
        self.a()  # writes this rank's slot; the other three slots still hold an older routing
        torch.cuda.synchronize()
        units = int(v41._plan(dev)[0])
        if units != distinct:
            raise RuntimeError(f"the plan holds {units} distinct live experts, want {distinct}")
        fill_peer_slots(banks, self.m)
        self.part = self.a().clone()
        again = self.a().clone()
        if not torch.equal(again, self.part):
            raise RuntimeError("the IQ op is not repeatable on the same inputs")
        if not bool(torch.isfinite(self.part).all()):
            raise RuntimeError("the IQ op's output is not finite")
        if live == 0 and bool((self.part != 0).any()):
            raise RuntimeError("the IQ op's output is not zero with no live route")
        if live > 0 and not bool((self.part != 0).any()):
            raise RuntimeError("the IQ op's output is all zero with live routes")
        whole = self.b().clone()
        without = self.c().clone()
        if not torch.equal(whole, without + self.part.float()):
            raise RuntimeError("forward is not bit-equal to (c) plus the IQ op's output")
        self.checks = {"live_routes": got, "plan_distinct_experts": units,
                       "iq_output_nonzero": int((self.part != 0).sum()),
                       "forward_equals_c_plus_a": True, "repeatable": True}

    def a(self):
        return self.moe._iq_partial(self.sx, self.iq_ids, self.st)

    def b(self):
        return self.moe.forward(self.sx, self.si, self.st)

    def c(self):
        self.moe._iq_partial = lambda x, i, w: self.zero16
        try:
            return self.moe.forward(self.sx, self.si, self.st)
        finally:
            del self.moe._iq_partial

    def r(self):
        return self.moe._iq_routes(self.si, self.st)

    def t(self):
        return self.moe.out[: self.m] + self.zero16.float()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--slices", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    gate = json.loads(a.gate_json)
    dev = torch.device("cuda", 0)
    with np.load(a.slices / f"t3-refs-L{a.layer:02d}.npz", allow_pickle=False) as data:
        refs = {k: data[k] for k in ("x_s0", "tw_s0", "meta")}
    meta = json.loads(bytes(refs["meta"]).decode())
    if meta["layer"] != a.layer or meta["m"] != 6:
        raise ValueError(f"the reference file is for {meta}")
    shards = [
        native.IqShardSource.from_slice_file(
            a.slices / f"iq-L{a.layer:02d}-r{RANK}-{role}.bin", role, kind)
        for role, kind in (("w1", meta["type13"]), ("w3", meta["type3"]), ("w2", meta["type2"]))
    ]
    source = native.IqLayerSource(*shards)
    tier = core.bench_tier()
    # build_layer reads only cfg.packs; the tally never leaves the server host.
    cfg = native.NativePackConfig(
        a.packs, native.TallySource(Path("/nonexistent/the-tally-stays-on-the-server")),
        {0: 1}, tiers.HotIq(), tiers.PrefillWarm.NATIVE_STAGED, gplan.GroupPlan.KERNEL,
        native.swig.Swiglu.CHAIN, native.iqt.IqTail.TORCH, prefill_swiglu=native.swig.Swiglu.CHAIN, prefill_rb=native.qrb.PrefillRb.RB2)
    banks = self_peer(dev)
    moe = native.build_layer(cfg, a.layer, RANK, tier, source, dev, max_tokens=MAX_TOKENS,
                             stage_prefill=None)
    if moe.warm is None or moe.warm.host_in_decode or moe._planner is None:
        raise RuntimeError("the layer is not the hot_iq object with the group-plan kernel")
    x6 = torch.from_numpy(refs["x_s0"]).to(dev)
    t6 = torch.from_numpy(refs["tw_s0"]).to(dev)
    diet = int(torch.ops.v41k.expert_diet()) if hasattr(torch.ops.v41k, "expert_diet") else None
    res: dict = {
        "bench": "bench_iq_live_routes", "layer": a.layer, "rank": RANK,
        "max_tokens": MAX_TOKENS, "type13": source.type13, "type2": source.type2,
        "n_hot": int((tier == core.HOT).sum()), "n_warm": int((tier == core.WARM).sum()),
        "group_plan": moe.group_plan.value, "gate": gate,
        "device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
        "triton": triton.__version__, "outer": OUTER, "reps": REPS,
        "warm_replays": WARM_REPLAYS,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "v41_moe_sha256": hashlib.sha256(Path(v41.__file__).read_bytes()).hexdigest(),
        "v41k_library": v41.default_v41k_library(),
        "v41k_sha256": hashlib.sha256(Path(v41.default_v41k_library()).read_bytes()).hexdigest(),
        "expert_diet_flags": diet,
        "peer": "one process: three absent ranks marked arrived, their slots a copy of this rank's",
        "parts_us": {}, "checks": {}, "informative_us": {},
    }
    seed0 = 1000 * a.layer

    def run(tag: str, routing: Routing, names: tuple[str, ...]) -> dict:
        row = {}
        for name in names:
            fn = getattr(routing, name.rstrip("2"))
            row[name] = timed(fn, routing.part if name.startswith("a") else None)
            print(json.dumps({tag: {name: row[name]["median_us"],
                                    "settled": row[name]["settled"]}}), flush=True)
        return row

    for m, ks in ((6, core.K_VALUES), (1, (0, 1, 3, 6))):
        rows, checks = {}, {}
        for k in ks:
            routing = Routing(moe, banks, x6, t6, core.live_routes(k, m, tier, seed0 + 10 * m + k),
                              k, k)
            checks[str(k)] = routing.checks
            rows[str(k)] = run(f"M{m} k{k}", routing, ("a", "b", "c", "a2"))
            del routing
        res["parts_us"][f"M{m}"] = rows
        res["checks"][f"M{m}"] = checks

    info: dict = {}
    for j in (1, 2, 6):
        routing = Routing(moe, banks, x6, t6, core.shared_routes(j, 6, tier, seed0 + 500 + j),
                          6 * j, j)
        info[f"s{j}"] = {"live_routes": 6 * j, "distinct_experts": j,
                         **run(f"M6 shared j{j}", routing, ("a",))}
        del routing
    routing = Routing(moe, banks, x6, t6, core.live_routes(6, 6, tier, seed0 + 66), 6, 6)
    info["r_t"] = run("M6 k6", routing, ("r", "t"))
    del routing
    res["informative_us"] = info

    parts = {int(k): v for k, v in res["parts_us"]["M6"].items()}
    res["layer_verdict"] = core.layer_verdict(parts)
    res["unsettled_parts"] = [f"{mk}.k{k}.{n}" for mk, rows in res["parts_us"].items()
                              for k, row in rows.items() for n, v in row.items()
                              if not v["settled"]]
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": "TIMING_WRITTEN", "layer": a.layer,
                      "layer_verdict": res["layer_verdict"],
                      "unsettled_parts": res["unsettled_parts"]}), flush=True)

    if a.profile:
        prof: dict = {"layer": a.layer, "gate": gate, "replays": PROFILE_REPLAYS, "graphs": {}}
        for m, ks in ((6, core.K_VALUES), (1, (0, 1, 6))):
            for k in ks:
                routing = Routing(moe, banks, x6, t6,
                                  core.live_routes(k, m, tier, seed0 + 10 * m + k), k, k)
                names = ("a", "b") if (m, k) in ((6, 0), (6, 6), (6, 36)) else ("a",)
                for name in names:
                    prof["graphs"][f"M{m}_k{k}_{name}"] = profile_graph(getattr(routing, name))
                del routing
        out = a.output.with_suffix(".prof.json")
        out.write_text(json.dumps(prof, indent=1) + "\n")
        print(json.dumps({"status": "PROFILE_WRITTEN", "file": str(out)}), flush=True)
    print(json.dumps({"status": "BENCH_DONE", "layer": a.layer,
                      "passes": res["layer_verdict"]["passes"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
