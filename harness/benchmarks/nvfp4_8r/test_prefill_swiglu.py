"""D3 acceptance on one card (rules 3 to 5 of the D3 method): forward_prefill with
prefill_swiglu = kernel against the same records with chain.

One process, one layer of the f-refs set (5, 12 or 27), rank 0 records from the
pack, the f-refs synthetic tier (190 HOT, 101 WARM, 93 HOST). Object kinds, each
built twice over the SAME expert sets, once with chain and once with kernel:
  res  all 384 experts in HBM;
  ns   two-tier, native_staged: HOT in HBM, every other expert staged from host RAM;
  t3   three-tier, the installed iq reading's native part: HOT in HBM, HOST staged,
       WARM routes weight 0.0 in the native reduce; the IQ prefill handle is a stub
       whose partial is zero (the real one is a four-rank collective; its input
       and output do not touch the SwiGLU). The IQ bank is a stub holding only its
       slot count and a device tensor for the constructor's device check:
       forward_prefill never reads it.
Inputs: f-refs x / ids / tw at M = 1, 7, 512, 8188; seeds 1, 2, 3 (uniform
distinct top-6 ids, weights normalized to sum 1, x = the f-refs rows) at M = 8188
and 513.

Rule 3: torch.equal(kernel output, chain output) on every case; control: chain
twice is bit-equal (else that case is INCONCLUSIVE); f-refs outputs finite
(seeded: recorded). Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE.
Rule 4, information: kernel outputs of res summed over ranks 0..3 (each rank's
records built in turn on this card) against ref_native on rows with no WARM
route, rel L2 (the native rule is <= 1e-2).
Rule 5, information: res at M = 8188, chain against kernel, five calls each after
a warm-up of at least 2 s in the call pattern, the SM clock read by a sleep
kernel before each call; and the call's allocation peak above its start.
Nothing private is written: digests, counts, timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

REF_MS = (1, 7, 512, 8188)
SEEDS = (1, 2, 3)
SEED_MS = (8188, 513)
CLOCK_CYCLES = 2_000_000
REPS = 5
WARM_S = 2.0


class ZeroIqPrefill:
    """Stands in for IqLayerPrefill: a zero IQ partial of the right shape,
    made for one (stub) bank."""

    def __init__(self, bank) -> None:
        self.bank = bank

    def serves(self, bank) -> bool:
        return bank is self.bank

    def partial(self, x: torch.Tensor, iq_ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if iq_ids.shape != (x.size(0), nat.TOP_K):
            raise ValueError(f"iq_ids {tuple(iq_ids.shape)} for {x.size(0)} rows")
        return torch.zeros((x.size(0), nat.HIDDEN), dtype=torch.float32, device=x.device)


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def clock_mhz() -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    torch.cuda._sleep(CLOCK_CYCLES)
    b.record()
    b.synchronize()
    return CLOCK_CYCLES / (a.elapsed_time(b) * 1000.0)


def seeded(seed: int, m: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.rand((m, nat.NUM_EXPERTS), generator=g).topk(nat.TOP_K, dim=1).indices.to(torch.int32)
    w = torch.rand((m, nat.TOP_K), generator=g, dtype=torch.float32) + 1e-3
    w = w / w.sum(1, keepdim=True)
    return ids.to(dev).contiguous(), w.to(dev).contiguous()


def rel(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for need in (a.refs, a.packs / "manifest.json",
                 *(a.packs / f"L{a.layer:02d}-r{r}-{t}.bin" for r in range(4) for t in ("hot", "cold"))):
        if not need.is_file():
            print(f"INCONCLUSIVE: input absent: {need}")
            return 3
    dev = torch.device("cuda", 0)
    with np.load(a.refs, allow_pickle=False) as d:
        refs = {k: d[k] for k in d.files}
    meta = json.loads(bytes(refs["meta"]).decode())
    if meta["layer"] != a.layer or meta["iq_only"]:
        print(f"INCONCLUSIVE: {a.refs} is layer {meta['layer']}, iq_only {meta['iq_only']}")
        return 3
    tier = refs["tier"]
    hot_ids = [int(e) for e in np.flatnonzero(tier == tiers.Tier.HOT)]
    cold_ids = [int(e) for e in np.flatnonzero(tier != tiers.Tier.HOT)]
    n_warm = int((tier == tiers.Tier.WARM).sum())
    x_all = torch.from_numpy(refs["x"]).to(dev).contiguous()
    res: dict = {"check": "prefill_swiglu", "layer": a.layer, "gate": json.loads(a.gate_json),
                 "experts": {"hot": len(hot_ids), "warm": n_warm, "host": len(cold_ids) - n_warm},
                 "modules_under_test": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes())
                                        .hexdigest()[:16] for m in (nat, swig)},
                 "device": torch.cuda.get_device_name(dev)}

    def build(rank: int, kinds: tuple[str, ...]):
        raw = torch.from_numpy(nat.read_rank_experts(a.packs, a.layer, rank))
        objs: dict[str, dict[str, nat.NativeV41Moe]] = {}
        full = nat.NativeExpertSet.from_raw(raw, dev) if "res" in kinds else None
        if "ns" in kinds or "t3" in kinds:
            hot = nat.NativeExpertSet.from_raw(raw[hot_ids], dev)
            cold = nat.NativeExpertSet.from_raw_host(raw[cold_ids], dev)
            hs, cs = nat.split_maps(hot_ids, cold_ids, dev)
        del raw
        ident = torch.arange(nat.NUM_EXPERTS, dtype=torch.int32, device=dev)
        for mode in swig.Swiglu:
            kw = dict(group_plan=gplan.GroupPlan.CHAIN, swiglu=swig.Swiglu.CHAIN, prefill_swiglu=mode, prefill_rb=nat.qrb.PrefillRb.RB2)
            if "res" in kinds:
                objs.setdefault("res", {})[mode.value] = nat.NativeV41Moe(full, ident, 6, **kw)
            if "ns" in kinds:
                objs.setdefault("ns", {})[mode.value] = nat.NativeV41Moe(
                    hot, hs, 6, cold=cold, cold_slot_of_expert=cs, **kw)
            if "t3" in kinds:
                bank = SimpleNamespace(num_slots=n_warm, w1=torch.empty(0, device=dev))
                objs.setdefault("t3", {})[mode.value] = nat.NativeV41Moe.three_tier(
                    hot, cold, bank, tier, 6, ZeroIqPrefill(bank),
                    iq_tail=nat.iqt.IqTail.TORCH, **kw)
        for kind, pair in objs.items():
            for mode, moe in pair.items():
                if moe.prefill_swiglu is not swig.Swiglu(mode):
                    raise RuntimeError(f"{kind}/{mode} holds {moe.prefill_swiglu}")
        return objs

    inputs = []
    for m in REF_MS:
        inputs.append((f"ref-M{m}", True, x_all[:m],
                       torch.from_numpy(np.ascontiguousarray(refs["ids"][:m])).to(dev),
                       torch.from_numpy(np.ascontiguousarray(refs["tw"][:m])).to(dev)))
    for s in SEEDS:
        for m in SEED_MS:
            ids, w = seeded(s, m, dev)
            inputs.append((f"seed{s}-M{m}", False, x_all[:m], ids, w))

    objs = build(0, ("res", "ns", "t3"))
    cases, n_fail, n_inconc = [], 0, 0
    for kind in ("res", "ns", "t3"):
        for name, is_ref, x, ids, w in inputs:
            c1 = objs[kind]["chain"].forward_prefill(x, ids, w)
            c2 = objs[kind]["chain"].forward_prefill(x, ids, w)
            k1 = objs[kind]["kernel"].forward_prefill(x, ids, w)
            k2 = objs[kind]["kernel"].forward_prefill(x, ids, w)
            torch.cuda.synchronize(dev)
            control = torch.equal(c1, c2)
            equal = torch.equal(k1, c1)
            finite = bool(torch.isfinite(c1).all()) and bool(torch.isfinite(k1).all())
            if not control:
                verdict = "INCONCLUSIVE"
                n_inconc += 1
            elif not equal or (is_ref and not finite):
                verdict = "FAIL"
                n_fail += 1
            else:
                verdict = "PASS"
            cases.append({"kind": kind, "input": name, "rows": int(x.size(0)), "verdict": verdict,
                          "bit_equal": equal, "chain_repeat_equal": control,
                          "kernel_repeat_equal": torch.equal(k1, k2), "finite": finite,
                          "digest_chain": digest(c1), "digest_kernel": digest(k1)})
            del c1, c2, k1, k2
    res["cases"] = cases

    # Rule 5 (information): timing and the allocation peak, res, M = 8188.
    x, ids, w = inputs[3][2], inputs[3][3], inputs[3][4]
    timing = {}
    for mode in ("chain", "kernel", "chain", "kernel"):
        moe = objs["res"][mode]
        fn = lambda: moe.forward_prefill(x, ids, w)  # noqa: E731
        t0, n = time.monotonic(), 0
        while time.monotonic() - t0 < WARM_S or n < 2:
            fn()
            torch.cuda.synchronize(dev)
            n += 1
        vals, mhz = [], []
        for _ in range(REPS):
            mhz.append(round(clock_mhz(), 1))
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            out = fn()
            e1.record()
            e1.synchronize()
            vals.append(round(e0.elapsed_time(e1), 3))
            del out
        torch.cuda.synchronize(dev)
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        out = fn()
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev) - base
        del out
        timing.setdefault(mode, []).append({"warm_calls": n, "values_ms": vals,
                                            "median_ms": statistics.median(vals), "mhz_before_each": mhz,
                                            "alloc_peak_above_start_bytes": int(peak)})
    res["timing_res_M8188"] = timing
    del objs
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()

    # Rule 4 (information): four ranks' kernel outputs, res, against ref_native.
    no_warm = (tier[refs["ids"]] != tiers.Tier.WARM).all(1)
    total = {m: np.zeros((m, nat.HIDDEN), np.float64) for m in (512, 8188)}
    for rank in range(4):
        moe = build(rank, ("res",))["res"]["kernel"]
        for m in total:
            out = moe.forward_prefill(x_all[:m], inputs[REF_MS.index(m)][3], inputs[REF_MS.index(m)][4])
            total[m] += out.double().cpu().numpy()
            del out
        del moe
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
    ref = refs["ref_native"].astype(np.float64)
    res["native_rule_info"] = {
        str(m): {"rows_without_warm_route": int(no_warm[:m].sum()),
                 "rel_l2": rel(total[m][no_warm[:m]], ref[:m][no_warm[:m]]), "limit": 1e-2}
        for m in total}

    status = "FAIL" if n_fail else ("INCONCLUSIVE" if n_inconc else "PASS")
    res.update({"status": status, "cases_total": len(cases), "cases_fail": n_fail, "cases_inconclusive": n_inconc})
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": status, "layer": a.layer, "cases": len(cases), "fail": n_fail,
                      "inconclusive": n_inconc,
                      "native_rel_l2": {k: round(v["rel_l2"], 6) for k, v in res["native_rule_info"].items()},
                      "median_ms": {k: [t["median_ms"] for t in v] for k, v in timing.items()},
                      "peak_bytes": {k: [t["alloc_peak_above_start_bytes"] for t in v] for k, v in timing.items()}}))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[status]


if __name__ == "__main__":
    sys.exit(main())
