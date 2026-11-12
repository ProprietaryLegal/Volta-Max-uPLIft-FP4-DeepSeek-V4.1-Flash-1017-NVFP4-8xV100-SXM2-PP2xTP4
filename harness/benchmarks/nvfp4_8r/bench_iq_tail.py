"""Plan 06 one-card speed bench: the three-tier decode forward with the
torch tail (installed) against the fused tail, one layer per run.

Same layer, inputs, routings and timing as bench_iq_live_routes (card 8.5):
real packs and IQ slices, 96 WARM experts (id % 4 == 2), M = 6 at 0, 1, 3, 6,
12, 16, 36 live IQ routes and M = 1 at 0, 1, 3, 6; one captured graph per
part, 50 warm replays, 5 repeats of the median of 7 replays with a sleep
kernel before each. The ``torch`` layer is the product object (build_layer);
the ``fused`` layer is built on the same banks. Per (M, k), after a check
that the two forwards are bit-equal on that routing, four graphs are captured
and timed in the order torch, fused, fused, torch (t1, f1, f2, t2). One
throw-away graph is timed first: the first graph of a process runs slow
(card 8.5's diagnosis).

Informative, at M = 6 and k = 6 only: the same four with the caller's
``.to(float16)`` after the forward (tc1, fc1, fc2, tc2;
gguf_moe._apply_nvfp4_8r converts the result at once), and with --profile the
per-kernel GPU times of both forwards.

The rule is plan 06 section 7 (iq_tail_core, written before this bench's
data); decide_iq_tail.py reads the three layers' receipts. Exit 0 when the
receipt is written, 1 on a failed check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import triton

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_iq_tail as iqt
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
from vllm.model_executor.layers.quantization import gguf_v41_moe as v41

from . import bench_iq_live_routes as live
from . import iq_live_core as core
from . import iq_tail_core as rule

ORDER = (("t1", "torch"), ("f1", "fused"), ("f2", "fused"), ("t2", "torch"))


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(
        torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--slices", type=Path, required=True)
    p.add_argument("--swiglu", choices=("chain", "kernel"), required=True)
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
            a.slices / f"iq-L{a.layer:02d}-r{live.RANK}-{role}.bin", role, kind)
        for role, kind in (("w1", meta["type13"]), ("w3", meta["type3"]), ("w2", meta["type2"]))
    ]
    tier = core.bench_tier()
    # build_layer reads only cfg.packs; the tally never leaves the server host.
    cfg = native.NativePackConfig(
        a.packs, native.TallySource(Path("/nonexistent/the-tally-stays-on-the-server")),
        {0: 1}, tiers.HotIq(), tiers.PrefillWarm.NATIVE_STAGED, gplan.GroupPlan.KERNEL,
        native.swig.Swiglu(a.swiglu), iqt.IqTail.TORCH, prefill_swiglu=native.swig.Swiglu.CHAIN, prefill_rb=native.qrb.PrefillRb.RB2)
    banks = live.self_peer(dev)
    ref = native.build_layer(cfg, a.layer, live.RANK, tier, native.IqLayerSource(*shards), dev,
                             max_tokens=live.MAX_TOKENS, stage_prefill=None)
    fused = native.NativeV41Moe.three_tier(
        ref.experts, ref.cold, ref.warm.iq, tier, live.MAX_TOKENS, None,
        group_plan=gplan.GroupPlan.KERNEL, swiglu=native.swig.Swiglu(a.swiglu), prefill_swiglu=native.swig.Swiglu.CHAIN, prefill_rb=native.qrb.PrefillRb.RB2,
        iq_tail=iqt.IqTail.FUSED)
    if ref.warm is None or ref.warm.host_in_decode or ref._iq_tail is not None \
            or fused._iq_tail is None or fused.swiglu is not ref.swiglu:
        raise RuntimeError("the two layers are not the hot_iq torch and fused objects")
    x6 = torch.from_numpy(refs["x_s0"]).to(dev)
    t6 = torch.from_numpy(refs["tw_s0"]).to(dev)
    res: dict = {
        "bench": "bench_iq_tail", "layer": a.layer, "rank": live.RANK,
        "max_tokens": live.MAX_TOKENS, "swiglu": a.swiglu, "group_plan": ref.group_plan.value,
        "n_hot": int((tier == core.HOT).sum()), "n_warm": int((tier == core.WARM).sum()),
        "gate": gate, "device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
        "triton": triton.__version__, "outer": live.OUTER, "reps": live.REPS,
        "warm_replays": live.WARM_REPLAYS, "order": [name for name, _ in ORDER],
        "sha256": {name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                   for name, mod in (("native", native), ("group_plan", gplan), ("iq_tail", iqt),
                                     ("v41_moe", v41))},
        "v41k_sha256": hashlib.sha256(Path(v41.default_v41k_library()).read_bytes()).hexdigest(),
        "peer": "one process: three absent ranks marked arrived, their slots a copy of this rank's",
        "parts_us": {}, "informative_us": {},
    }
    seed0 = 1000 * a.layer

    def forwards(routing: live.Routing) -> dict:
        return {"torch": routing.b,
                "fused": lambda: fused.forward(routing.sx, routing.si, routing.st)}

    def checked(routing: live.Routing) -> torch.Tensor:
        fns = forwards(routing)
        want = fns["torch"]().clone()
        got = fns["fused"]().clone()
        if not bits_equal(got, want):
            raise RuntimeError("the fused forward is not bit-equal to the torch forward")
        return want

    warm = live.Routing(ref, banks, x6, t6, core.live_routes(6, 6, tier, seed0 + 66), 6, 6)
    res["throwaway_first_graph_us"] = live.timed(warm.b, checked(warm))["median_us"]
    del warm

    for m, ks in ((6, core.K_VALUES), (1, (0, 1, 3, 6))):
        rows = {}
        for k in ks:
            routing = live.Routing(ref, banks, x6, t6,
                                   core.live_routes(k, m, tier, seed0 + 10 * m + k), k, k)
            want = checked(routing)
            fns = forwards(routing)
            row = {name: live.timed(fns[arm], want) for name, arm in ORDER}
            rows[str(k)] = row
            print(json.dumps({f"M{m} k{k}": {n: v["median_us"] for n, v in row.items()},
                              "settled": all(v["settled"] for v in row.values())}), flush=True)
            del routing
        res["parts_us"][f"M{m}"] = rows

    routing = live.Routing(ref, banks, x6, t6, core.live_routes(6, 6, tier, seed0 + 16), 6, 6)
    want16 = checked(routing).to(torch.float16)
    fns = forwards(routing)
    cast = {"torch": lambda: fns["torch"]().to(torch.float16),
            "fused": lambda: fns["fused"]().to(torch.float16)}
    if not torch.equal(cast["fused"](), want16):
        raise RuntimeError("the fused forward's FP16 result differs")
    res["informative_us"]["M6_k6_with_caller_cast"] = {
        name.replace("1", "c1").replace("2", "c2"): live.timed(cast[arm], want16)
        for name, arm in ORDER}
    print(json.dumps({"M6 k6 with cast": {n: v["median_us"] for n, v in
                                          res["informative_us"]["M6_k6_with_caller_cast"].items()}}),
          flush=True)

    medians = {mk: {int(k): {n: v["median_us"] for n, v in row.items()} for k, row in rows.items()}
               for mk, rows in res["parts_us"].items()}
    res["layer_verdict"] = rule.layer_tail_verdict(medians)
    res["unsettled_parts"] = [f"{mk}.k{k}.{n}" for mk, rows in res["parts_us"].items()
                              for k, row in rows.items() for n, v in row.items()
                              if not v["settled"]]
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    lv = res["layer_verdict"]
    print(json.dumps({"status": "TIMING_WRITTEN", "layer": a.layer, "saved_k6_us": lv["saved_k6_us"],
                      "settled": lv["settled"], "s2_slower": lv["s2_slower"],
                      "unsettled_parts": res["unsettled_parts"]}), flush=True)

    if a.profile:
        prof = {"layer": a.layer, "gate": gate, "replays": live.PROFILE_REPLAYS, "graphs": {}}
        for k in (0, 6):
            r = live.Routing(ref, banks, x6, t6, core.live_routes(k, 6, tier, seed0 + 60 + k), k, k)
            checked(r)
            for arm, fn in forwards(r).items():
                prof["graphs"][f"M6_k{k}_{arm}"] = live.profile_graph(fn)
            del r
        out = a.output.with_suffix(".prof.json")
        out.write_text(json.dumps(prof, indent=1) + "\n")
        print(json.dumps({"status": "PROFILE_WRITTEN", "file": str(out)}), flush=True)
    print(json.dumps({"status": "BENCH_DONE", "layer": a.layer, "saved_k6_us": lv["saved_k6_us"],
                      "settled": lv["settled"], "s1_holds": lv["s1_holds"],
                      "s2_slower": len(lv["s2_slower"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
