"""Plan 04 part B, step B1 falsifier, run on the deployed kernels (the test host,
one card).

Question: can production's IQ prefill (SharedOriginalPrefill.forward with
VLLM_V41_PREFILL_IQ_GEMM=1: transcode, the v41prefmoe GEMM kit, the row-split
activation and the accumulate kernels) serve a chosen subset of a layer's
experts held in its own bank, with every other route dropped (id -1)?

Four processes = four TP ranks on ONE card, gloo group. Each rank reads the
WARM experts' IQ records of five layers (IqWarmBank.from_source, the deployed
reader, from make_t3_iq_slices files) and hands SharedOriginalPrefill a cache
object that says: W experts per layer, all resident, slot = position in the
bank. The adapter is inline here; the product type is written after this run.
The NCCL turn that orders the ranks' row writes is replaced by a device
synchronize and a gloo barrier (four ranks cannot share one card in NCCL).

Expected values are outside the engine (make_f_prefill_refs): ref_iq is the
fp64 sum of the WARM routes only, from gguf.quants.dequantize. Per layer and
M in 7, 512, 8,188:
  a  the four ranks' outputs against ref_iq: rel L2 <= 2e-2 (plan 01 c2 rule);
     each rank writes only its own 1,280 output rows;
  b  rows with no WARM route are exactly 0.0;
  c  with every route dropped the output is exactly 0.0;
  d  call a again after c: bit-equal to the first.

Exit 0: the path serves a subset (the falsifier does not hold). 1: it holds.
3: INCONCLUSIVE (the missing input is named).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from .falsify_t3_iq_drop import Inconclusive, gate

RULE = 2e-2
HIDDEN, INTER, TOP_K, WARM = 5120, 2304, 6, 1
MS = (7, 512, 8188)
PREFILL_TOKENS = 4096  # run/serve-nvfp4-pp2.yaml: expert_cache_prefill_tokens


class GlooTurn:
    """Stands for pynccl's all_reduce turn: every rank's stream work so far
    is complete on every rank when it returns."""

    def __init__(self, dev: torch.device) -> None:
        self.dev = dev

    def all_reduce(self, _fence: torch.Tensor) -> None:
        torch.cuda.synchronize(self.dev)
        dist.barrier()


def agree(flag: bool) -> bool:
    t = torch.tensor([int(flag)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def summed(value: float) -> float:
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t)
    return float(t.item())


def load(refs_paths: list[Path]) -> list[dict]:
    out = []
    for path in refs_paths:
        if not path.is_file():
            raise Inconclusive(f"references absent: {path}")
        with np.load(path, allow_pickle=False) as data:
            refs = {k: data[k] for k in ("tier", "ids", "tw", "x", "ref_iq", "meta")}
        refs["meta"] = json.loads(bytes(refs["meta"]).decode())
        out.append(refs)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--slices", type=Path, nargs="+")
    parser.add_argument("--refs", type=Path, nargs="+")
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
    from vllm.model_executor.layers.quantization import gguf_shared_layout, gguf_shared_prefill
    from vllm.model_executor.layers.quantization.gguf_shared_runtime import (
        SHARDS,
        SharedExpertLayerSpec,
        resident_tables,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{a.port}",
                            rank=a.rank, world_size=4)
    dev = torch.device("cuda", 0)
    try:
        if not gguf_shared_layout.DOWN_ROWS or not gguf_shared_prefill._PREFILL_IQ:
            raise Inconclusive("VLLM_V41_DOWN_ROWS=1 and VLLM_V41_PREFILL_IQ_GEMM=1 are not set")
        layers = load(a.refs)
        specs, banks = [], {}
        for i, refs in enumerate(layers):
            meta = refs["meta"]
            warm = np.flatnonzero(refs["tier"] == WARM).tolist()
            shards = []
            for role, kind in (("w1", meta["type13"]), ("w3", meta["type3"]), ("w2", meta["type2"])):
                name = f"iq-L{meta['layer']:02d}-r{a.rank}-{role}.bin"
                found = [d / name for d in a.slices if (d / name).is_file()]
                if not found:
                    raise Inconclusive(f"IQ slice absent: {name}")
                shards.append(native.IqShardSource.from_slice_file(found[0], role, kind))
            bank = native.IqWarmBank.from_source(native.IqLayerSource(*shards), warm, dev)
            for shard in SHARDS:
                banks[i, shard] = getattr(bank, shard)
            specs.append(SharedExpertLayerSpec(
                HIDDEN, INTER, len(warm), len(warm), (bank.type13, bank.type13, bank.type2)))
            slot = np.full(384, -1, np.int32)
            slot[warm] = np.arange(len(warm), dtype=np.int32)
            refs["iq_ids"] = slot[refs["ids"]]
        table, cold = resident_tables(specs, dev)
        cache = SimpleNamespace(
            experts=max(s.experts for s in specs), device=dev, specs=tuple(specs),
            budget=SimpleNamespace(all_resident=True), rank=a.rank,
            group=SimpleNamespace(cpu_group=dist.group.WORLD,
                                  device_communicator=SimpleNamespace(pynccl_comm=GlooTurn(dev))),
            prefill_tokens=PREFILL_TOKENS, _barrier=dist.barrier,
            runtime=SimpleNamespace(table=table, cold_offsets=cold, banks=banks),
            layers=[r["meta"]["layer"] for r in layers],
        )
        tm = gguf_shared_prefill.SharedOriginalPrefill(cache)
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    own = slice(a.rank * 1280, (a.rank + 1) * 1280)
    other = torch.ones(HIDDEN, dtype=torch.bool, device=dev)
    other[own] = False
    results, holds = [], []
    for i, refs in enumerate(layers):
        meta = refs["meta"]
        res: dict = {"layer": meta["layer"], "type13": meta["type13"], "type2": meta["type2"],
                     "warm_experts": specs[i].experts, "cases": []}
        for m in MS:
            x = torch.from_numpy(refs["x"][:m]).to(dev)
            tw = torch.from_numpy(refs["tw"][:m]).to(dev)
            ids = torch.from_numpy(np.ascontiguousarray(refs["iq_ids"][:m])).to(dev)
            none = torch.full_like(ids, -1)
            ref = torch.from_numpy(refs["ref_iq"][:m, own]).to(dev).double()
            no_warm = torch.from_numpy((refs["iq_ids"][:m] < 0).all(1)).to(dev)

            first = tm.forward(i, x, ids, tw).clone()
            torch.cuda.synchronize(dev)
            dist.barrier()
            dropped = tm.forward(i, x, none, tw).clone()
            torch.cuda.synchronize(dev)
            dist.barrier()
            started = time.monotonic()
            again = tm.forward(i, x, ids, tw)
            torch.cuda.synchronize(dev)
            seconds = time.monotonic() - started
            dist.barrier()
            num = summed(float((first[:, own].double() - ref).square().sum()))
            den = summed(float(ref.square().sum()))
            case = {
                "m": m,
                "warm_routes": int((refs["iq_ids"][:m] >= 0).sum()),
                "rows_without_warm_route": int(no_warm.sum()),
                "rel_l2": (num / den) ** 0.5,
                "other_rows_zero": agree(bool((first[:, other] == 0).all())),
                "no_warm_rows_exact_zero": agree(bool((first[no_warm] == 0).all())),
                "all_dropped_exact_zero": agree(bool((dropped == 0).all())),
                "finite": agree(bool(torch.isfinite(first).all())),
                "repeat_bit_equal": agree(bool(torch.equal(first, again))),
                "seconds_four_ranks_one_card": seconds,
            }
            res["cases"].append(case)
            tag = f"L{meta['layer']} M={m}"
            if not case["rel_l2"] <= RULE:
                holds.append(f"{tag}: rel L2 {case['rel_l2']:.3e} vs the live-only reference > {RULE}")
            for key in ("other_rows_zero", "no_warm_rows_exact_zero", "all_dropped_exact_zero",
                        "finite", "repeat_bit_equal"):
                if not case[key]:
                    holds.append(f"{tag}: {key} is false")
            del first, dropped, again, x, tw, ids, none, ref
        results.append(res)
    status = "FALSIFIER_HOLDS" if holds else "FALSIFIER_DOES_NOT_HOLD"
    if a.rank == 0:
        a.output.write_text(json.dumps({
            "check": "plan04-B1-falsifier", "status": status, "notes": holds, "rule_rel_l2": RULE,
            "prefill_tokens": PREFILL_TOKENS, "layers": results,
        }, indent=1) + "\n")
    print(json.dumps({"rank": a.rank, "status": status, "notes": holds}))
    return 1 if holds else 0


if __name__ == "__main__":
    sys.exit(main())
