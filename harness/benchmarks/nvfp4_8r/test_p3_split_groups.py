"""CPU unit test of NativeV41Moe's hot/cold group construction.

Expected values come from the routing definition, not the implementation:
slot s (token s // 6) must be computed exactly once, by the bank its expert
belongs to (hot HBM or cold host), at that bank's slot for the expert. The
emulated launch follows skinny_moe_qpn_sm70's grid rule (csrc .cu:695-696,
235-239): grid.y = min(S, E) groups, a group with goff[g+1] <= goff[g] exits.
Weights are irrelevant here and are tiny placeholders; this does not replace
the GPU gates (accept_p2, accept_p3_layer).
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    QpnFragmentOrder,
)


def placeholder_set(n: int) -> nat.NativeExpertSet:
    z = torch.zeros((n, 1), dtype=torch.uint8)
    g = torch.ones(n, dtype=torch.float32)
    return nat.NativeExpertSet(z, z, z, z, g, g, QpnFragmentOrder.M8N8K4)


def emulate(launch, n_slots: int) -> list[tuple[int, int]]:
    """(slot, expert) pairs one kernel launch computes."""
    bank, perm, gids, goff = launch
    done = []
    for g in range(min(n_slots, bank.num_slots)):
        beg, end = int(goff[g]), int(goff[g + 1])
        for r in range(beg, end):
            done.append((int(perm[r]), int(gids[g])))
    return done


def check(moe, hot_slot, cold_slot, ids: torch.Tensor) -> None:
    launches = moe._split_groups(ids)
    n = ids.numel()
    seen: dict[int, tuple[int, int]] = {}
    for which, launch in enumerate(launches):
        for slot, expert in emulate(launch, n):
            if slot in seen:
                raise AssertionError(f"slot {slot} computed twice")
            seen[slot] = (which, expert)
    flat = ids.reshape(-1).tolist()
    for slot, e in enumerate(flat):
        expected = (0, int(hot_slot[e])) if hot_slot[e] >= 0 else (1, int(cold_slot[e]))
        if seen.get(slot) != expected:
            raise AssertionError(f"slot {slot} expert {e}: {seen.get(slot)} != {expected}")


def main() -> int:
    rng = np.random.default_rng(0)
    cases = 0
    for n_cold in (1, 36, 158, 383):
        cold = sorted(rng.choice(384, n_cold, replace=False).tolist())
        hot = [e for e in range(384) if e not in set(cold)]
        dev = torch.device("cpu")
        hot_slot, cold_slot = nat.split_maps(hot, cold, dev)
        moe = nat.NativeV41Moe(placeholder_set(len(hot)), hot_slot, 16,
                               cold=placeholder_set(len(cold)), cold_slot_of_expert=cold_slot,
                               group_plan=gplan.GroupPlan.CHAIN, swiglu=swig.Swiglu.CHAIN, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)
        for m in range(1, 17):
            for mode in ("random", "all_hot", "all_cold", "one_cold"):
                pool = {"random": list(range(384)), "all_hot": hot,
                        "all_cold": cold, "one_cold": hot}[mode]
                if len(pool) < 6:
                    continue  # no 6 distinct experts to route to
                ids = torch.tensor(
                    np.stack([rng.choice(pool, 6, replace=False) for _ in range(m)]),
                    dtype=torch.int32,
                )
                if mode == "one_cold":
                    ids[rng.integers(m), 0] = int(rng.choice(cold))
                check(moe, hot_slot, cold_slot, ids)
                cases += 1
    print(f"PASS {cases} routing cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
