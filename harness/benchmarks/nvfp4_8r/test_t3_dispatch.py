"""A2 (plan 01, CPU): decode dispatch of the three-tier NativeV41Moe.

Expected values come from the tier map route by route, without the product's
compaction: route slot s with expert e must be computed
- exactly once by the hot launch at e's hot slot when e is HOT,
- exactly once by the host launch at e's cold slot when e is HOST,
- by no launch when e is WARM; then iq_ids[s] is e's warm slot and the native
  reduce weight of s is exactly 0. Every other route has iq_ids -1 and keeps
  its router weight.
Slots are ranks in ascending expert id (plan 5.1). A launch is emulated with
skinny_moe_qpn_sm70's grid rule (test_p3_split_groups): grid = min(route
slots, bank slots) groups; group g covers perm[goff[g]:goff[g+1]] and is
skipped when empty; every group id the grid can read must be a bank slot.
Weights are placeholders: this does not replace the GPU gate (accept_t3_layer).

Exit 0 PASS, 1 FAIL. Receipt: --output JSON with status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HOT, WARM, HOST = 0, 1, 2
BB = {29: 56, 16: 66, 17: 74}


def placeholder_set(nat, n: int):
    from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
        QpnFragmentOrder,
    )

    z = torch.zeros((n, 1), dtype=torch.uint8)
    g = torch.ones(n, dtype=torch.float32)
    return nat.NativeExpertSet(z, z, z, z, g, g, QpnFragmentOrder.M8N8K4)


def placeholder_bank(nat, n: int, type13: int, type2: int):
    # Real shapes, untouched memory: the dispatch never reads the bank.
    return nat.IqWarmBank(
        torch.empty((n, 576, 20 * BB[type13]), dtype=torch.uint8),
        torch.empty((n, 576, 20 * BB[type13]), dtype=torch.uint8),
        torch.empty((n, 1280, 9 * BB[type2]), dtype=torch.uint8),
        type13, type2,
    )


def emulate(launch, n_slots: int) -> list[tuple[int, int]]:
    """(route slot, bank slot) pairs one kernel launch computes."""
    bank, perm, gids, goff = launch
    for t in (perm, gids, goff):
        if t.dtype != torch.int32:
            raise AssertionError(f"launch tensor dtype {t.dtype}, kernel takes int32")
    if perm.numel() != n_slots or gids.numel() != n_slots or goff.numel() != n_slots + 1:
        raise AssertionError("launch tensor sizes")
    done = []
    for g in range(min(n_slots, bank.num_slots)):
        if not 0 <= int(gids[g]) < bank.num_slots:
            raise AssertionError(f"group {g} id {int(gids[g])} outside the bank")
        beg, end = int(goff[g]), int(goff[g + 1])
        if not (0 <= beg <= n_slots and 0 <= end <= n_slots):
            raise AssertionError(f"group {g} offsets {beg}..{end} outside the routes")
        for r in range(beg, end):
            done.append((int(perm[r]), int(gids[g])))
    return done


def check(moe, tier: np.ndarray, ids: torch.Tensor, weights: torch.Tensor) -> None:
    hot = {e: i for i, e in enumerate(e for e in range(384) if tier[e] == HOT)}
    cold = {e: i for i, e in enumerate(e for e in range(384) if tier[e] != HOT)}
    warm = {e: i for i, e in enumerate(e for e in range(384) if tier[e] == WARM)}
    has_host = bool((tier == HOST).any())
    launches = moe._tier_groups(ids)
    if len(launches) != (2 if has_host else 1):
        raise AssertionError(f"{len(launches)} launches, host tier present: {has_host}")
    if launches[0][0] is not moe.experts or (has_host and launches[1][0] is not moe.cold):
        raise AssertionError("launch banks are not (hot set, host set)")
    n = ids.numel()
    seen: dict[int, tuple[int, int]] = {}
    for which, launch in enumerate(launches):
        for slot, bank_slot in emulate(launch, n):
            if slot in seen:
                raise AssertionError(f"route {slot} computed twice")
            seen[slot] = (which, bank_slot)
    iq_ids, w_native = moe._iq_routes(ids, weights)
    if iq_ids.dtype != torch.int32 or iq_ids.shape != ids.shape:
        raise AssertionError("iq_ids must be int32 [M, 6]")
    if w_native.dtype != torch.float32 or w_native.shape != weights.shape:
        raise AssertionError("native weights must be fp32 [M, 6]")
    flat, flat_w = ids.reshape(-1).tolist(), weights.reshape(-1).tolist()
    got_iq, got_w = iq_ids.reshape(-1).tolist(), w_native.reshape(-1).tolist()
    for slot, e in enumerate(flat):
        t = int(tier[e])
        want = {HOT: (0, hot.get(e)), HOST: (1, cold.get(e)), WARM: None}[t]
        if seen.get(slot) != want:
            raise AssertionError(f"route {slot} tier {t}: launches gave {seen.get(slot)}, want {want}")
        want_iq = warm[e] if t == WARM else -1
        if got_iq[slot] != want_iq:
            raise AssertionError(f"route {slot} tier {t}: iq id {got_iq[slot]}, want {want_iq}")
        want_w = 0.0 if t == WARM else flat_w[slot]
        if got_w[slot] != want_w:
            raise AssertionError(f"route {slot} tier {t}: native weight {got_w[slot]}, want {want_w}")


def tier_maps(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    maps = []
    for name, p in (("mixed", (0.5, 0.3, 0.2)), ("mostly_hot", (0.9, 0.05, 0.05)),
                    ("hot_iq", (0.6, 0.4, 0.0)), ("mostly_warm", (0.1, 0.8, 0.1)),
                    ("mostly_host", (0.1, 0.1, 0.8))):
        tier = rng.choice(3, 384, p=p).astype(np.int8)
        tier[rng.integers(384)] = HOT  # a layer has at least one HOT ...
        tier[(int(np.flatnonzero(tier == HOT)[0]) + 1) % 384] = WARM  # ... and one WARM
        maps.append((name, tier))
    one = np.full(384, HOST, np.int8)
    one[17], one[300] = HOT, WARM
    maps.append(("one_hot_one_warm", one))
    one = np.full(384, WARM, np.int8)
    one[383] = HOT
    maps.append(("one_hot_rest_warm", one))
    one = np.full(384, HOT, np.int8)
    one[0] = WARM
    maps.append(("one_warm_rest_hot", one))
    return maps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat

    rng = np.random.default_rng(20261109)
    cases: dict[str, int] = {}
    fails: list[str] = []
    for name, tier in tier_maps(rng):
        n_hot, n_warm = int((tier == HOT).sum()), int((tier == WARM).sum())
        type13, type2 = [(29, 29), (16, 16), (17, 17), (17, 29)][len(cases) % 4]
        moe = nat.NativeV41Moe.three_tier(
            placeholder_set(nat, n_hot), placeholder_set(nat, 384 - n_hot),
            placeholder_bank(nat, n_warm, type13, type2), tier, max_tokens=6,
            prefill=None,
        )
        pools = {
            "random": list(range(384)),
            "all_warm": np.flatnonzero(tier == WARM).tolist(),
            "no_warm": np.flatnonzero(tier != WARM).tolist(),
            "all_hot": np.flatnonzero(tier == HOT).tolist(),
            "all_host": np.flatnonzero(tier == HOST).tolist(),
        }
        count = 0
        try:
            for mode, pool in pools.items():
                if len(pool) < 6:
                    continue  # no 6 distinct experts of this kind to route to
                for m in range(1, 7):
                    for _ in range(12 if mode != "random" else 30):
                        ids = torch.tensor(
                            np.stack([rng.choice(pool, 6, replace=False) for _ in range(m)]),
                            dtype=torch.int32,
                        )
                        w = rng.random((m, 6)) + 0.1
                        weights = torch.tensor(w / w.sum(1, keepdims=True), dtype=torch.float32)
                        check(moe, tier, ids, weights)
                        count += 1
        except AssertionError as exc:
            fails.append(f"{name}: {exc}")
        cases[name] = count
    total = sum(cases.values())
    if total < 2000:
        fails.append(f"only {total} routings ran, the check needs 2,000")
    status = "PASS" if not fails else "FAIL"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"check": "A2", "status": status, "routings": cases,
                                    "total": total, "failures": fails}, indent=1) + "\n")
    print(status, f"{total} routings", *fails, sep="\n")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
