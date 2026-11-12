"""RB 4 patch acceptance on one card (section 2 of the RB 4 method): forward_prefill with prefill_rb = rb4
against the same records with rb2.

One process, one layer of the f-refs set (5, 12 or 27), rank 0 records from the pack, the f-refs synthetic tier
(190 HOT, 101 WARM, 93 HOST). Object kinds as test_prefill_swiglu.py (res: all 384 in HBM; ns: two-tier,
native_staged; t3: three-tier, the iq reading's native part with a zero IQ partial stub). Each kind is built three
times over the SAME expert sets: A = (prefill_swiglu chain, prefill_rb rb2), today's code; R = (chain, rb4);
B = (kernel, rb4), window 17's arm B. Inputs: f-refs x / ids / tw at M = 1, 7, 512, 8188; seeds 1, 2, 3 (uniform
distinct top-6 ids) at M = 8188 and 513. ns at small M stages host-RAM batches of few rows, which the library runs
at RB 1 like the installed op.

Rule: torch.equal(R, A) and torch.equal(B, A) on every case; control: A twice bit-equal (else that case is
INCONCLUSIVE). Build control: on the res HOT launch of f-refs M = 8188, the library at rb 2 and at rb 4 against the
installed op, W13 and W2, into buffers pre-filled with the same sentinel bits, compared as int16 over the whole
buffer (rb 2 differing makes the run INCONCLUSIVE, rb 4 differing is a FAIL). Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE.
Information: res at M = 8188, A against R, five calls each after a warm-up of at least 2 s in the call pattern,
the SM clock read by a sleep kernel before each call, and the call's allocation peak above its start.
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
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_qpn_rb as qrb
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

REF_MS = (1, 7, 512, 8188)
SEEDS = (1, 2, 3)
SEED_MS = (8188, 513)
CLOCK_CYCLES = 2_000_000
REPS = 5
WARM_S = 2.0
ARMS = {"A": (swig.Swiglu.CHAIN, qrb.PrefillRb.RB2), "R": (swig.Swiglu.CHAIN, qrb.PrefillRb.RB4),
        "B": (swig.Swiglu.KERNEL, qrb.PrefillRb.RB4)}
SENTINEL = -12345  # int16 bits of an FP16 NaN-free pattern no launch writes by chance


class ZeroIqPrefill:
    """Stands in for IqLayerPrefill: a zero IQ partial of the right shape, made for one (stub) bank."""

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


def build_control(moe: nat.NativeV41Moe, x: torch.Tensor, ids: torch.Tensor) -> dict:
    """The res HOT launch of forward_prefill, by the installed op and by the library at rb 2 and 4."""
    m, dev, e = x.size(0), x.device, moe.experts
    s = m * nat.TOP_K
    flat = ids.reshape(-1).long()
    hot = moe.slot_of_expert[flat].long()
    nh = e.num_slots
    hot_key = torch.where(hot >= 0, hot, nh)
    perm, gids, goff, _, _ = moe._compact(hot_key)
    count = (hot_key < nh).sum()
    launch = (perm, torch.where(gids < nh, gids, 0).to(torch.int32), torch.minimum(goff, count).to(torch.int32))
    out: dict = {}
    inter = torch.zeros((s, nat.W2_K), dtype=torch.float16, device=dev)
    for names, slot_major, split, cols in ((("q13", "s13", "g1"), False, moe.split13, 2 * nat.TP_INTER),
                                           (("q2", "s2", "g2"), True, moe.split2, nat.HIDDEN)):
        xin = x if not slot_major else inter
        ys = {}
        for who in ("installed", "lib_rb2", "lib_rb4"):
            y = torch.full((s, cols), SENTINEL, dtype=torch.int16, device=dev).view(torch.float16)
            args = (xin, *(getattr(e, n) for n in names), *launch, nat.TOP_K, y, slot_major, m, split, 1,
                    int(e.order))
            if who == "installed":
                torch.ops._C.skinny_moe_qpn_sm70(*args)
            else:
                torch.ops.nvfp4_8r_qpn_rb.moe_prefill(*args, 2 if who == "lib_rb2" else 4)
            ys[who] = y
        torch.cuda.synchronize(dev)
        ref = ys["installed"].view(torch.int16)
        key = "w13" if not slot_major else "w2"
        out[key] = {who: bool(torch.equal(ys[who].view(torch.int16), ref)) for who in ("lib_rb2", "lib_rb4")}
        out[key]["written_rows"] = int((ref != SENTINEL).any(1).sum())
        if not slot_major:
            y = ys["installed"]
            g13 = y[:, :nat.TP_INTER].float().clamp(max=nat.SWIGLU_LIMIT)
            u13 = y[:, nat.TP_INTER:].float().clamp(-nat.SWIGLU_LIMIT, nat.SWIGLU_LIMIT)
            # Unwritten rows hold the sentinel's finite FP16 value; the W2 launches read only their groups' rows.
            inter[:, :nat.TP_INTER].copy_(torch.nn.functional.silu(g13) * u13)
    out["split"] = [moe.split13, moe.split2]
    return out


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
    res: dict = {"check": "prefill_rb", "layer": a.layer, "gate": json.loads(a.gate_json),
                 "experts": {"hot": len(hot_ids), "warm": n_warm, "host": len(cold_ids) - n_warm},
                 "modules_under_test": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes())
                                        .hexdigest()[:16] for m in (nat, qrb, swig)},
                 "library_sha256": hashlib.sha256(qrb.LIBRARY.read_bytes()).hexdigest()[:16],
                 "device": torch.cuda.get_device_name(dev)}

    raw = torch.from_numpy(nat.read_rank_experts(a.packs, a.layer, 0))
    full = nat.NativeExpertSet.from_raw(raw, dev)
    hot = nat.NativeExpertSet.from_raw(raw[hot_ids], dev)
    cold = nat.NativeExpertSet.from_raw_host(raw[cold_ids], dev)
    hs, cs = nat.split_maps(hot_ids, cold_ids, dev)
    del raw
    ident = torch.arange(nat.NUM_EXPERTS, dtype=torch.int32, device=dev)
    objs: dict[str, dict[str, nat.NativeV41Moe]] = {"res": {}, "ns": {}, "t3": {}}
    for arm, (sw, rb) in ARMS.items():
        kw = dict(group_plan=gplan.GroupPlan.CHAIN, swiglu=swig.Swiglu.CHAIN, prefill_swiglu=sw, prefill_rb=rb)
        objs["res"][arm] = nat.NativeV41Moe(full, ident, 6, **kw)
        objs["ns"][arm] = nat.NativeV41Moe(hot, hs, 6, cold=cold, cold_slot_of_expert=cs, **kw)
        bank = SimpleNamespace(num_slots=n_warm, w1=torch.empty(0, device=dev))
        objs["t3"][arm] = nat.NativeV41Moe.three_tier(hot, cold, bank, tier, 6, ZeroIqPrefill(bank),
                                                      iq_tail=nat.iqt.IqTail.TORCH, **kw)
    for kind, arms in objs.items():
        for arm, moe in arms.items():
            if (moe.prefill_swiglu, moe.prefill_rb) != ARMS[arm]:
                raise RuntimeError(f"{kind}/{arm} holds {moe.prefill_swiglu}, {moe.prefill_rb}")

    inputs = []
    for m in REF_MS:
        inputs.append((f"ref-M{m}", True, x_all[:m],
                       torch.from_numpy(np.ascontiguousarray(refs["ids"][:m])).to(dev),
                       torch.from_numpy(np.ascontiguousarray(refs["tw"][:m])).to(dev)))
    for s in SEEDS:
        for m in SEED_MS:
            ids, w = seeded(s, m, dev)
            inputs.append((f"seed{s}-M{m}", False, x_all[:m], ids, w))

    ctl = build_control(objs["res"]["A"], inputs[3][2], inputs[3][3])
    res["build_control_ref_M8188"] = ctl
    ctl_rb2 = ctl["w13"]["lib_rb2"] and ctl["w2"]["lib_rb2"]
    ctl_rb4 = ctl["w13"]["lib_rb4"] and ctl["w2"]["lib_rb4"]

    cases, n_fail, n_inconc = [], 0, 0
    for kind in ("res", "ns", "t3"):
        for name, is_ref, x, ids, w in inputs:
            a1 = objs[kind]["A"].forward_prefill(x, ids, w)
            a2 = objs[kind]["A"].forward_prefill(x, ids, w)
            r1 = objs[kind]["R"].forward_prefill(x, ids, w)
            b1 = objs[kind]["B"].forward_prefill(x, ids, w)
            torch.cuda.synchronize(dev)
            control = torch.equal(a1, a2)
            eq_r, eq_b = torch.equal(r1, a1), torch.equal(b1, a1)
            finite = bool(torch.isfinite(a1).all())
            if not control:
                verdict = "INCONCLUSIVE"
                n_inconc += 1
            elif not (eq_r and eq_b) or (is_ref and not finite):
                verdict = "FAIL"
                n_fail += 1
            else:
                verdict = "PASS"
            cases.append({"kind": kind, "input": name, "rows": int(x.size(0)), "verdict": verdict,
                          "rb4_equal": eq_r, "kernel_rb4_equal": eq_b, "rb2_repeat_equal": control,
                          "finite": finite, "digest_A": digest(a1), "digest_R": digest(r1), "digest_B": digest(b1)})
            del a1, a2, r1, b1
    res["cases"] = cases

    # Information: timing and the allocation peak, res, M = 8188, A against R.
    x, ids, w = inputs[3][2], inputs[3][3], inputs[3][4]
    timing = {}
    for arm in ("A", "R", "A", "R"):
        moe = objs["res"][arm]
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
        timing.setdefault(arm, []).append({"warm_calls": n, "values_ms": vals,
                                           "median_ms": statistics.median(vals), "mhz_before_each": mhz,
                                           "alloc_peak_above_start_bytes": int(peak)})
    res["timing_res_M8188"] = timing

    if not ctl_rb2:
        status = "INCONCLUSIVE"
    elif n_fail or not ctl_rb4:
        status = "FAIL"
    elif n_inconc:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    res.update({"status": status, "cases_total": len(cases), "cases_fail": n_fail, "cases_inconclusive": n_inconc})
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": status, "layer": a.layer, "cases": len(cases), "fail": n_fail,
                      "inconclusive": n_inconc, "build_control": {k: v for k, v in ctl.items() if k != "split"},
                      "median_ms": {k: [t["median_ms"] for t in v] for k, v in timing.items()},
                      "peak_bytes": {k: [t["alloc_peak_above_start_bytes"] for t in v] for k, v in timing.items()}}))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[status]


if __name__ == "__main__":
    sys.exit(main())
