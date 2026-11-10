"""Falsifier 1 of plan 01, run on the deployed v41k.so (the test host, one card).

Question: does torch.ops.vllm.gguf_v41_original_expert_dp4a, as the server
runs it (EXPERT_FAST, PLAN_FUSE, DOWN_ROWS, EXPERT_DIET=3, peer bank), drop a
route whose id is negative, and leave no unwritten output row?

Four processes = four TP ranks on ONE card, gloo group, the CUDA-IPC peer
bank registered as plan 5.5 describes (inline here: the product function is
written after this run). Each rank holds its own slice of the IQ bank
(make_t3_iq_slices). Expected values are outside the op:
- fp64 references from gguf.quants.dequantize (make_t3_refs): the sum of the
  four ranks' outputs must match the sum over the LIVE routes only;
- a token whose six routes are all dropped, and a step with every route
  dropped, must give exactly 0.0 on every rank, after NaN-filled blocks of
  the output's size were freed to the allocator;
- the same partial-drop call after a different history must be bit-equal
  (a dropped route's stale bank row is never read).
Every rank runs the same sequence; each call is followed by a device
synchronize and a group collective, because nothing else bounds rank skew
here (the server's MoE all-reduce does).

Exit 0: the op drops them (falsifier 1 does not hold). 1: falsifier 1 holds.
3: INCONCLUSIVE (the missing input is named).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

RULE = 2e-2  # plan 01 section 7, c2: Q8 arithmetic
HIDDEN, TOP_K = 5120, 6
MIN_MEM_KB = 120 * 2**20  # the maintainer's rule: MemAvailable >= 120 GiB before loading


class Inconclusive(RuntimeError):
    pass


def gate(card: int) -> dict:
    """The accept_p2.gpu_gate rules with the maintainer's limits: no compute app on
    the host, the card's power limit <= 150 W, MemAvailable >= 120 GiB. The two
    nvidia-smi calls are 15 s apart. Never changes a cap."""
    import subprocess
    import time

    def smi(*args: str) -> str:
        return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True).stdout

    apps = smi("--query-compute-apps=pid").split()
    if apps:
        raise Inconclusive(f"compute apps present on the host: {len(apps)}")
    time.sleep(15)
    power = float(smi("-i", str(card), "--query-gpu=power.limit").strip())
    if power > 150.0:
        raise Inconclusive(f"card {card} power limit {power} W > 150 W rule")
    avail = next(
        int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if avail < MIN_MEM_KB:
        raise Inconclusive(f"MemAvailable {avail} kB < 120 GiB")
    return {"card": card, "power_limit_w": power, "mem_available_kb": avail}


def attach_peer_inline(rank: int, dev: torch.device):
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.model_executor.layers.quantization import gguf_v41_moe as moe
    from vllm.model_executor.layers.quantization.gguf_shared_prefill import _DeviceView

    if not moe._expert_major_ready():
        raise Inconclusive("v41k library not loaded (VLLM_V41K_LIB / VLLM_V41_EM)")
    if not (moe._EXPERT_FAST and moe._PLAN_FUSE):
        raise Inconclusive("VLLM_V41_EXPERT_FAST=1 and VLLM_V41_PLAN_FUSE=1 are not set")
    pointers = CustomAllreduce.create_shared_buffer(moe.ROWS_BYTES, dist.group.WORLD)
    own = torch.as_tensor(_DeviceView(pointers[rank], (moe.ROWS_BYTES,), "|u1"), device=dev)
    moe.register_rows_peer(dev, rank, pointers, own)
    torch.cuda.synchronize(dev)
    dist.barrier()
    if moe._PEER.get(dev) is None:
        raise Inconclusive("the IPC peer bank is not registered for this device")
    return pointers, own


def load_bank(slices: Path, layer: int, rank: int, experts: list[int], dev: torch.device):
    from vllm.model_executor.layers.quantization.gguf_native import empty_guarded_weight

    manifest_path = slices / "iq-manifest.json"
    if not manifest_path.is_file():
        raise Inconclusive(f"IQ slice manifest absent: {manifest_path}")
    entry = json.loads(manifest_path.read_text())["layers"][str(layer)]
    banks = {}
    for role in ("w1", "w3", "w2"):
        name = f"iq-L{layer:02d}-r{rank}-{role}.bin"
        path = slices / name
        if not path.is_file():
            raise Inconclusive(f"IQ slice absent: {path}")
        shape = tuple(entry["files"][name]["shape"])
        data = np.memmap(path, np.uint8, "r", shape=shape)
        picked = torch.from_numpy(np.ascontiguousarray(data[experts]))
        bank = empty_guarded_weight(tuple(picked.shape), entry[f"type_{role}"], dev)
        bank.copy_(picked)
        banks[role] = bank
    return banks, entry["type_w1"], entry["type_w2"]


def nan_prefill(m: int, dev: torch.device) -> bool:
    """Free NaN blocks of the op output's size; true if the allocator hands one back."""
    blocks = [torch.full((m, HIDDEN), float("nan"), dtype=torch.float16, device=dev)
              for _ in range(8)]
    del blocks
    probe = torch.empty((m, HIDDEN), dtype=torch.float16, device=dev)
    reused = bool(probe.isnan().all())
    del probe
    return reused


def run_layer(refs_path: Path, slices: Path, rank: int, dev: torch.device) -> dict:
    if not refs_path.is_file():
        raise Inconclusive(f"references absent: {refs_path}")
    with np.load(refs_path, allow_pickle=False) as data:
        refs = {k: data[k] for k in data.files}
    meta = json.loads(bytes(refs["meta"]).decode())
    layer, samples = meta["layer"], meta["samples"]
    experts = sorted({int(e) for s in range(samples) for e in refs[f"ids_s{s}"].ravel()})
    slot = {e: i for i, e in enumerate(experts)}
    banks, type13, type2 = load_bank(slices, layer, rank, experts, dev)
    if (type13, type2) != (meta["type13"], meta["type2"]):
        raise Inconclusive(f"L{layer}: slice types {type13}/{type2} != reference types")
    own_rows = slice(rank * 1280, (rank + 1) * 1280)

    def step() -> None:
        torch.cuda.synchronize(dev)
        dist.barrier()

    def op(x, ids, tw):
        return torch.ops.vllm.gguf_v41_original_expert_dp4a(
            x, ids, tw, banks["w1"], banks["w3"], banks["w2"], type13, type2, 0, 10.0
        )

    def total(out: torch.Tensor) -> np.ndarray:
        """Sum of the four ranks' outputs (gloo); doubles as the barrier."""
        torch.cuda.synchronize(dev)
        acc = out.double().cpu()
        dist.all_reduce(acc)
        return acc.numpy()

    def agree(flag: bool) -> bool:
        """True only if the flag is true on all four ranks."""
        t = torch.tensor([int(flag)])
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def rel(got: np.ndarray, ref: np.ndarray) -> float:
        return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))

    res = {"layer": layer, "type13": type13, "type2": type2, "bank_experts": len(experts),
           "all_live_rel_l2": [], "partial_live_rel_l2": [], "partial_vs_all_routes_rel_l2": [],
           "token_drop_rel_l2": [], "other_rows_zero": True, "token_drop_exact_zero": True,
           "all_dropped_exact_zero": True, "finite": True, "nan_prefill_effective": True,
           "partial_repeat_bit_equal": True, "graph_replay_bit_equal": True, "calls": 0}
    for s in range(samples):
        ids_np, tw_np = refs[f"ids_s{s}"], refs[f"tw_s{s}"]
        iq = refs[f"iq_s{s}"].reshape(6, TOP_K, HIDDEN)
        for m in (range(1, 7) if s == 0 else (1, 6)):
            x = torch.from_numpy(refs[f"x_s{s}"][:m]).to(dev)
            tw = torch.from_numpy(tw_np[:m]).to(dev)
            slots = torch.tensor([[slot[int(e)] for e in row] for row in ids_np[:m]],
                                 dtype=torch.int32, device=dev)
            odd = torch.from_numpy((ids_np[:m] % 2 == 1)).to(dev)
            token0 = slots.clone()
            token0[0] = -1
            none = torch.full_like(slots, -1)
            ref_all = iq[:m].sum(1)
            ref_even = (iq[:m] * (ids_np[:m] % 2 == 0)[..., None]).sum(1)

            def local(out: torch.Tensor) -> None:
                other = torch.ones(HIDDEN, dtype=torch.bool, device=dev)
                other[own_rows] = False
                res["other_rows_zero"] &= bool((out[:, other] == 0).all())
                res["finite"] &= bool(torch.isfinite(out).all())

            # A: every route live.
            out = op(x, slots, tw)
            local(out)
            res["all_live_rel_l2"].append(rel(total(out), ref_all))
            # B: routes to odd expert ids dropped.
            partial = torch.where(odd, -1, slots).to(torch.int32)
            out_b = op(x, partial, tw).clone()
            local(out_b)
            got = total(out_b)
            if np.linalg.norm(ref_even) > 0:
                res["partial_live_rel_l2"].append(rel(got, ref_even))
            res["partial_vs_all_routes_rel_l2"].append(rel(got, ref_all))
            # C: token 0 loses all six routes (other tokens keep theirs).
            res["nan_prefill_effective"] &= agree(nan_prefill(m, dev))
            out = op(x, token0, tw)
            local(out)
            res["token_drop_exact_zero"] &= bool((out[0] == 0).all())
            got = total(out)
            if m > 1:
                res["token_drop_rel_l2"].append(rel(got[1:], ref_all[1:]))
            # D: every route dropped.
            res["nan_prefill_effective"] &= agree(nan_prefill(m, dev))
            out = op(x, none, tw)
            res["all_dropped_exact_zero"] &= bool((out == 0).all())
            res["finite"] &= bool(torch.isfinite(out).all())
            step()
            # E: B again after a different history (stale bank rows of dropped routes).
            out = op(x, partial, tw)
            res["partial_repeat_bit_equal"] &= bool(torch.equal(out, out_b))
            step()
            res["calls"] += 5
            # G: the server runs the op inside a CUDA graph.
            sx, si, st = x.clone(), slots.clone(), tw.clone()
            for _ in range(3):
                op(sx, si, st)
                step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = op(sx, si, st)
            for ids in (slots, partial, token0, none, partial):
                si.copy_(ids)
                graph.replay()
                step()
                replayed = static_out.clone()
                if ids is none:
                    res["all_dropped_exact_zero"] &= bool((replayed == 0).all())
                if ids is token0:
                    res["token_drop_exact_zero"] &= bool((replayed[0] == 0).all())
                eager = op(x, ids, tw)
                res["graph_replay_bit_equal"] &= bool(torch.equal(replayed, eager))
                step()
            res["calls"] += 13
            del graph
    for key in ("other_rows_zero", "token_drop_exact_zero", "all_dropped_exact_zero", "finite",
                "partial_repeat_bit_equal", "graph_replay_bit_equal"):
        res[key] = agree(res[key])
    for key in ("all_live_rel_l2", "partial_live_rel_l2", "partial_vs_all_routes_rel_l2",
                "token_drop_rel_l2"):
        res[key] = {"worst": max(res[key]), "best": min(res[key]), "n": len(res[key])}
    del banks
    torch.cuda.empty_cache()
    return res


def verdict(layers: list[dict]) -> tuple[str, list[str]]:
    holds, unsure = [], []
    for r in layers:
        tag = f"L{r['layer']}"
        if not r["nan_prefill_effective"]:
            unsure.append(f"{tag}: the allocator did not hand the NaN block back")
        if r["all_live_rel_l2"]["worst"] > RULE:
            unsure.append(f"{tag}: all-live rel L2 {r['all_live_rel_l2']['worst']:.3e} > {RULE} "
                          "(the tolerance derivation or the harness is wrong)")
        if r["partial_live_rel_l2"]["worst"] > RULE:
            holds.append(f"{tag}: dropped routes still contribute "
                         f"(rel L2 {r['partial_live_rel_l2']['worst']:.3e} vs live-only reference)")
        if r["token_drop_rel_l2"]["worst"] > RULE:
            holds.append(f"{tag}: token-drop rel L2 {r['token_drop_rel_l2']['worst']:.3e}")
        for key in ("other_rows_zero", "token_drop_exact_zero", "all_dropped_exact_zero",
                    "finite", "partial_repeat_bit_equal", "graph_replay_bit_equal"):
            if not r[key]:
                holds.append(f"{tag}: {key} is false")
    if holds:
        return "FALSIFIER_1_HOLDS", holds + unsure
    if unsure:
        return "INCONCLUSIVE", unsure
    return "FALSIFIER_1_DOES_NOT_HOLD", []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--slices", type=Path)
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
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{a.port}",
                            rank=a.rank, world_size=4)
    dev = torch.device("cuda", 0)
    try:
        bank = attach_peer_inline(a.rank, dev)
        layers = [run_layer(r, a.slices, a.rank, dev) for r in a.refs]
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    status, notes = verdict(layers)
    if a.rank == 0:
        a.output.write_text(json.dumps({"check": "plan01-falsifier-1", "status": status,
                                        "notes": notes, "rule_rel_l2": RULE,
                                        "layers": layers}, indent=1) + "\n")
    print(json.dumps({"rank": a.rank, "status": status, "notes": notes}))
    del bank
    return {"FALSIFIER_1_DOES_NOT_HOLD": 0, "FALSIFIER_1_HOLDS": 1, "INCONCLUSIVE": 3}[status]


if __name__ == "__main__":
    sys.exit(main())
