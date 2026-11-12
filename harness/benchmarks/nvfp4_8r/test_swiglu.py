"""Acceptance of the one-launch decode SwiGLU (CAMPAIGN 8.4) on one card.

Written before any data (agent folder DECISION-RULE-swiglu.md, order step 1).

1. Exhaustive: every pair of FP16 bit patterns (gate, up), 2^32 pairs, laid
   out as the layer's y13 rows ([S, 1152]: gate in columns 0..575, up in
   576..1151). The chain (the three lines of NativeV41Moe.forward) and the
   kernel (nvfp4_8r_swiglu.swiglu_into) write into inter-shaped buffers
   ([S, 640]) pre-filled with a sentinel. PASS needs every one of the 576
   written columns bit-equal (a NaN counts as equal only when both sides are
   NaN; how many such pairs differ in payload is reported) and columns 576..639
   still the sentinel. Information only: production's op
   torch.ops._C.deepseek_v41_swiglu_sm70_out on contiguous copies, against the
   chain.
2. Layers: L05, L12, L27 rank 0 real packs, max_tokens 6 objects: resident and
   two-tier (manifest split) with group plan chain and kernel, and three-tier
   (manifest hot HOT, the rest WARM, placeholder IQ bank) with group plan
   kernel and the IQ op replaced by a fixed zero partial in both arms (the real
   op needs four ranks' CUDA-IPC peer bank). Each pair differs only in swiglu
   (chain against kernel) and shares its expert sets. Refs samples s0 and s1,
   M = 1..6, eager and CUDA-graph replay: the forward output (int32 view) and
   the layer's inter rows are bit-equal between the arms, the kernel arm's
   graph output equals its eager output, and the output is finite.

Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

from .test_group_plan import graph_out, placeholder_iq

MAX_TOKENS = 6
ROWS = 65536                  # rows of one exhaustive chunk
SENTINEL = 0x7BFF             # FP16 65504: never a SwiGLU output (|out| <= 100)
INTER, Y13, OUT = swig.INTER, swig.Y13_COLS, swig.OUT_COLS


def chain_into(y13: torch.Tensor, inter: torch.Tensor) -> None:
    """The reference: NativeV41Moe.forward's chain, line for line."""
    g = y13[:, :nat.TP_INTER].float().clamp(max=nat.SWIGLU_LIMIT)
    u = y13[:, nat.TP_INTER:].float().clamp(-nat.SWIGLU_LIMIT, nat.SWIGLU_LIMIT)
    inter[:, :nat.TP_INTER].copy_(torch.nn.functional.silu(g) * u)


def compare_bits(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int, torch.Tensor]:
    """(differing, both NaN, both NaN with other bits, mask of differing)."""
    ab, bb = a.view(torch.int16), b.view(torch.int16)
    both_nan = torch.isnan(a) & torch.isnan(b)
    diff = (ab != bb) & ~both_nan
    return int(diff.sum()), int(both_nan.sum()), int((both_nan & (ab != bb)).sum()), diff


def exhaustive(dev: torch.device) -> dict:
    allh = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.float16)
    total = 1 << 32
    per = ROWS * INTER
    y13 = torch.empty((ROWS, Y13), dtype=torch.float16, device=dev)
    ic = torch.empty((ROWS, OUT), dtype=torch.float16, device=dev)
    ik = torch.empty_like(ic)
    ip = torch.empty((ROWS, INTER), dtype=torch.float16, device=dev)
    sentinel = torch.tensor([SENTINEL], dtype=torch.int16).view(torch.float16).item()
    prod = getattr(torch.ops._C, "deepseek_v41_swiglu_sm70_out", None)
    res = {"pairs": 0, "differ": 0, "both_nan": 0, "both_nan_other_bits": 0, "pad_touched": 0,
           "examples": [], "production_op": "unavailable" if prod is None else
           {"differ": 0, "both_nan_other_bits": 0, "examples": []}}
    t0 = time.time()
    start = 0
    while start < total:
        p = torch.arange(start, start + per, dtype=torch.int64, device=dev) % total
        gate, up = allh[p & 0xFFFF].view(ROWS, INTER), allh[p >> 16].view(ROWS, INTER)
        y13[:, :INTER].copy_(gate)
        y13[:, INTER:].copy_(up)
        ic.fill_(sentinel)
        ik.fill_(sentinel)
        chain_into(y13, ic)
        swig.swiglu_into(y13, ik)
        n = min(per, total - start)
        flat_c, flat_k = ic[:, :INTER].reshape(-1)[:n], ik[:, :INTER].reshape(-1)[:n]
        d, bn, bno, mask = compare_bits(flat_c, flat_k)
        res["pairs"] += n
        res["differ"] += d
        res["both_nan"] += bn
        res["both_nan_other_bits"] += bno
        res["pad_touched"] += int((ik[:, INTER:].view(torch.int16) != SENTINEL).sum())
        if d and len(res["examples"]) < 10:
            for j in torch.nonzero(mask)[: 10 - len(res["examples"])].flatten().tolist():
                pj = start + j
                res["examples"].append({"gate_bits": pj & 0xFFFF, "up_bits": pj >> 16,
                                        "chain_bits": int(flat_c[j:j + 1].view(torch.int16).item()) & 0xFFFF,
                                        "kernel_bits": int(flat_k[j:j + 1].view(torch.int16).item()) & 0xFFFF})
        if prod is not None:
            prod(ip, y13[:, :INTER].contiguous(), y13[:, INTER:].contiguous())
            dp, _, bnop, maskp = compare_bits(flat_c, ip.reshape(-1)[:n])
            res["production_op"]["differ"] += dp
            res["production_op"]["both_nan_other_bits"] += bnop
            if dp and len(res["production_op"]["examples"]) < 5:
                for j in torch.nonzero(maskp)[:5 - len(res["production_op"]["examples"])].flatten().tolist():
                    pj = start + j
                    res["production_op"]["examples"].append(
                        {"gate_bits": pj & 0xFFFF, "up_bits": pj >> 16,
                         "chain_bits": int(flat_c[j:j + 1].view(torch.int16).item()) & 0xFFFF,
                         "op_bits": int(ip.reshape(-1)[j:j + 1].view(torch.int16).item()) & 0xFFFF})
        start += per
    torch.cuda.synchronize()
    res["seconds"] = round(time.time() - t0, 1)
    res["pass"] = res["pairs"] == total and res["differ"] == 0 and res["pad_touched"] == 0
    return res


def zero_iq(moe: nat.NativeV41Moe, dev: torch.device) -> None:
    """Both arms: the IQ op's partial is a fixed zero (see the docstring)."""
    zero = torch.zeros((MAX_TOKENS, nat.HIDDEN), dtype=torch.float16, device=dev)
    moe._iq_partial = lambda x, iq_ids, tw: zero[: x.size(0)]


def layer_pairs(packs: Path, layer: int, dev: torch.device):
    split = nat.read_manifest(packs)[layer][0]
    raw = torch.from_numpy(nat.read_rank_experts(packs, layer, 0))
    hs, cs = nat.split_maps(split["hot"], split["cold"], dev)
    full_set = nat.NativeExpertSet.from_raw(raw, dev)
    hot_set = nat.NativeExpertSet.from_raw(raw[split["hot"]], dev)
    cold_set = nat.NativeExpertSet.from_raw_host(raw[split["cold"]], dev)
    del raw
    ident = torch.arange(384, dtype=torch.int32, device=dev)
    tier = np.full(nat.NUM_EXPERTS, int(tiers.Tier.WARM), np.int8)
    tier[split["hot"]] = int(tiers.Tier.HOT)
    iq = placeholder_iq(int((tier == int(tiers.Tier.WARM)).sum()), dev)

    def make(kind: str, plan, mode):
        if kind == "resident":
            return nat.NativeV41Moe(full_set, ident, MAX_TOKENS, group_plan=plan, swiglu=mode, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)
        if kind == "two_tier":
            return nat.NativeV41Moe(hot_set, hs, MAX_TOKENS, cold=cold_set, cold_slot_of_expert=cs,
                                    group_plan=plan, swiglu=mode, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)
        moe = nat.NativeV41Moe.three_tier(hot_set, cold_set, iq, tier, MAX_TOKENS, None,
                                          group_plan=plan, swiglu=mode, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2,
                                          iq_tail=nat.iqt.IqTail.TORCH)
        zero_iq(moe, dev)
        return moe

    for kind, plan in (("resident", gplan.GroupPlan.CHAIN), ("resident", gplan.GroupPlan.KERNEL),
                       ("two_tier", gplan.GroupPlan.CHAIN), ("two_tier", gplan.GroupPlan.KERNEL),
                       ("three_tier", gplan.GroupPlan.KERNEL)):
        c, k = make(kind, plan, swig.Swiglu.CHAIN), make(kind, plan, swig.Swiglu.KERNEL)
        if c._swiglu_kernel is not None or k._swiglu_kernel is None:
            raise AssertionError(f"{kind}: the arms do not differ in swiglu")
        yield f"{kind}/{plan.value}", c, k


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


def run_layers(packs: Path, refs_dir: Path, dev: torch.device) -> tuple[dict, list[str]]:
    res, fails = {}, []
    for layer in (5, 12, 27):
        with np.load(refs_dir / f"refs-L{layer:02d}.npz", allow_pickle=False) as data:
            refs = {k: data[k] for k in data.files}
        row = {}
        for name, c, k in layer_pairs(packs, layer, dev):
            checks = []
            for s in (0, 1):
                for m in range(1, MAX_TOKENS + 1):
                    x = torch.from_numpy(refs[f"x_m6_s{s}"][:m]).to(dev)
                    i = torch.from_numpy(refs[f"ids_m6_s{s}"][:m]).to(dev)
                    t = torch.from_numpy(refs[f"tw_m6_s{s}"][:m]).to(dev)
                    ec = c.forward(x, i, t).clone()
                    ek = k.forward(x, i, t).clone()
                    rows = m * nat.TOP_K
                    inter_eq = bool(torch.equal(c.inter[:rows].view(torch.int16),
                                                k.inter[:rows].view(torch.int16)))
                    gc, gk = graph_out(c, x, i, t), graph_out(k, x, i, t)
                    chk = {"s": s, "M": m,
                           "eager_bit_equal": bits_equal(ec, ek),
                           "inter_bit_equal": inter_eq,
                           "graph_bit_equal": bits_equal(gc, gk),
                           "graph_equals_eager": bits_equal(gk, ek),
                           "finite": bool(torch.isfinite(ek).all())}
                    checks.append(chk)
                    if not all(v for key, v in chk.items() if key not in ("s", "M")):
                        fails.append(f"L{layer:02d} {name} s{s} M{m}: {chk}")
            row[name] = checks
            del c, k
        res[f"L{layer:02d}"] = row
        torch.cuda.empty_cache()
    return res, fails


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs-dir", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    import triton

    res: dict = {"check": "swiglu_kernel", "gate": json.loads(a.gate_json),
                 "torch": torch.__version__, "triton": triton.__version__,
                 "device": torch.cuda.get_device_name(dev),
                 "swiglu_sha256": hashlib.sha256(Path(swig.__file__).read_bytes()).hexdigest(),
                 "native_sha256": hashlib.sha256(Path(nat.__file__).read_bytes()).hexdigest()}
    fails: list[str] = []
    ex = exhaustive(dev)
    res["exhaustive"] = ex
    print(json.dumps({"exhaustive": {k: v for k, v in ex.items() if k != "examples"}}), flush=True)
    if not ex["pass"]:
        fails.append(f"exhaustive: {ex['differ']} differing pairs, pad touched {ex['pad_touched']}, "
                     f"pairs {ex['pairs']}, first {ex['examples'][:3]}")
    layers, lf = run_layers(a.packs, a.refs_dir, dev)
    res["layers"] = layers
    fails += lf
    res["failures"] = fails
    res["comparisons"] = sum(len(v) for r in layers.values() for v in r.values())
    res["status"] = "FAIL" if fails else "PASS"
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": res["status"], "exhaustive_pairs": ex["pairs"],
                      "exhaustive_differ": ex["differ"], "both_nan": ex["both_nan"],
                      "production_op": ex["production_op"] if isinstance(ex["production_op"], str)
                      else {k: v for k, v in ex["production_op"].items() if k != "examples"},
                      "layer_comparisons": res["comparisons"], "failures": fails[:5]}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
