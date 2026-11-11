# SPDX-License-Identifier: Apache-2.0
"""NVFP4-8R decode group plan in one Triton launch (plan 03 Part B).

NativeV41Moe's QPN launches take a plan of the routes: ``perm`` (the routes
sorted by their bank slot, stable), and per launch ``gids`` / ``goff`` (group
g covers perm[goff[g]:goff[g+1]] in slot gids[g] of the bank; the kernel's
grid visits min(S, E) groups and skips the empty ones). The reference path
builds it with a chain of about ten small tensor ops (``_compact`` and
``_groups`` / ``_split_groups`` / ``_tier_groups`` in nvfp4_8r_native.py),
each one graph node with its own launch cost. This module writes the same
int32 tensors, bit for bit, in one program:

- key[i] of route i: its HOT slot; else n_hot + its slot in the second map
  (the host-RAM set for a two-tier layer, the HOST slot for a three-tier
  layer); else the sentinel n_hot + n_second (a WARM route).
- rank[i] = #{j: key[j] < key[i]} + #{j < i: key[j] == key[i]}, perm[rank[i]] = i.
- Route i starts a group when no earlier route has its key. The group's index
  is the number of distinct smaller keys; goff = that group's first sorted
  position (n past the last group), gids = its key (0 past the last group).
- HBM launch: gids_hot = gids if < n_hot else 0; goff_hot = min(goff, #HOT).
- Host launch: both rotated left by the number of HOT groups;
  goff_host = min(goff, #native routes), gids_host = clamp(gids - n_hot, 0,
  n_second - 1).

n = 6M <= 96 (max_tokens <= 16), so one program holds a lane per route
(BLOCK = next power of two >= n + 1). The O(n^2) counts run as two runtime
loops over j < n that read key[j] and start[j] back from two small scratch
buffers (one bar.sync after each is written); perm, goff and gids are then
written by scatter from the group-starting routes plus a fill of the
disjoint tail. The code size does not grow with BLOCK. An earlier build
counted with [BLOCK x BLOCK] broadcasts: fully unrolled, 448 KB of code at
M = 6, and after the forward's weight stream its cold instruction fetch
cost about 49 us a call on the test host (plan 03 item 2,
logs/dispatch/diag-icache6-20261111T020845Z.json). Every element of the
first n entries and goff[n] is written on every call, so a stale tail from a
larger M is never read. Nothing syncs with the host and the buffers are
fixed, so the plan captures into a CUDA graph; ``GroupPlanner`` compiles and
runs every M's variant at construction, before any capture.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

GROUP_PLAN_KEY = "nvfp4_8r_group_plan"
TOP_K = 6
NUM_EXPERTS = 384
MAX_TOKENS = 16  # the QPN op's M limit (skinny_sm70_moe._QPN_MAX_M)


class GroupPlan(enum.Enum):
    """How a NativeV41Moe builds its decode group plan."""

    # The reference chain of tensor ops (also the prefill path, any M).
    CHAIN = "chain"
    # One Triton launch (this module).
    KERNEL = "kernel"


def check_group_plan(mode: object) -> GroupPlan:
    if type(mode) is not GroupPlan:
        raise ValueError(f"{GROUP_PLAN_KEY}: {mode!r} is not a GroupPlan")
    return mode


def parse_group_plan(extra: Mapping[str, object]) -> GroupPlan:
    """The group plan of model-loader-extra-config; no default."""
    names = [m.value for m in GroupPlan]
    if GROUP_PLAN_KEY not in extra:
        raise ValueError(f"{GROUP_PLAN_KEY} is required ({' or '.join(names)})")
    raw = extra[GROUP_PLAN_KEY]
    if type(raw) is not str or raw not in names:
        raise ValueError(f"{GROUP_PLAN_KEY} must be {' or '.join(names)}, got {raw!r}")
    return GroupPlan(raw)


@dataclass(frozen=True)
class ResidentKeys:
    """Every expert in the HBM set: key = HOT slot; one launch."""

    hot_slot: torch.Tensor  # int32 [384], no -1


@dataclass(frozen=True)
class SplitKeys:
    """HOT slots first, then the second map's slots, then one sentinel for
    an expert in neither (WARM). ``host_launch``: the layer runs a second
    QPN launch over its host-RAM set."""

    hot_slot: torch.Tensor  # int32 [384], -1 where not HOT
    second_slot: torch.Tensor  # int32 [384], -1 where absent
    n_hot: int
    n_second: int
    host_launch: bool


PlanKeys = ResidentKeys | SplitKeys


@dataclass(frozen=True)
class HostPlanBuffers:
    gids: torch.Tensor  # int32 [S]
    goff: torch.Tensor  # int32 [S + 1]


@dataclass(frozen=True)
class GroupPlanBuffers:
    """Owned by one GroupPlanner; int32, fixed size S = max_tokens * 6,
    written in place."""

    perm: torch.Tensor  # [S]
    gids_hot: torch.Tensor  # [S]
    goff_hot: torch.Tensor  # [S + 1]
    host: HostPlanBuffers | None
    scratch_key: torch.Tensor  # [S], the kernel's own: key[j] for its loops
    scratch_start: torch.Tensor  # [S], the kernel's own: start flag of route j

    @classmethod
    def allocate(cls, slots: int, host: bool, device: torch.device) -> GroupPlanBuffers:
        def z(n: int) -> torch.Tensor:
            return torch.zeros(n, dtype=torch.int32, device=device)

        return cls(z(slots), z(slots), z(slots + 1),
                   HostPlanBuffers(z(slots), z(slots + 1)) if host else None,
                   z(slots), z(slots))


@dataclass(frozen=True)
class PlanViews:
    """One call's plan: views of the first n (goff: n + 1) buffer entries."""

    perm: torch.Tensor
    gids_hot: torch.Tensor
    goff_hot: torch.Tensor
    host: tuple[torch.Tensor, torch.Tensor] | None  # (gids, goff) of the host launch


@triton.jit(do_not_specialize=["n", "n_hot", "n_second"])
def _group_plan_kernel(
    ids_ptr, hot_ptr, second_ptr,
    perm_ptr, gids_h_ptr, goff_h_ptr, gids_c_ptr, goff_c_ptr,
    skey_ptr, sstart_ptr,
    n, n_hot, n_second,
    SPLIT: tl.constexpr, HOST: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    valid = offs < n
    expert = tl.load(ids_ptr + offs, mask=valid, other=0)
    hot = tl.load(hot_ptr + expert, mask=valid, other=0)
    if SPLIT:
        second = tl.load(second_ptr + expert, mask=valid, other=0)
        key = tl.where(hot >= 0, hot, tl.where(second >= 0, n_hot + second, n_hot + n_second))
    else:
        key = hot
    tl.store(skey_ptr + offs, key, mask=valid)
    tl.debug_barrier()
    less = tl.zeros([BLOCK], dtype=tl.int32)
    dup_before = tl.zeros([BLOCK], dtype=tl.int32)
    for j in range(0, n):
        kj = tl.load(skey_ptr + j)
        less += (kj < key).to(tl.int32)
        dup_before += ((kj == key) & (j < offs)).to(tl.int32)
    start = valid & (dup_before == 0)
    tl.store(sstart_ptr + offs, start.to(tl.int32), mask=valid)
    tl.debug_barrier()
    gidx = tl.zeros([BLOCK], dtype=tl.int32)
    for j in range(0, n):
        kj = tl.load(skey_ptr + j)
        sj = tl.load(sstart_ptr + j)
        gidx += ((sj != 0) & (kj < key)).to(tl.int32)

    # perm[rank[i]] = i (rank is a permutation of 0..n-1).
    tl.store(perm_ptr + less + dup_before, offs, mask=valid)

    # Group q < n_groups starts at the route with gidx == q: goff = its sorted
    # position (less), gids = its key. Past the last group: goff = n, gids = 0.
    # The scatter (q < n_groups) and the fill (q >= n_groups) never overlap.
    n_groups = tl.sum(start.to(tl.int32), axis=0)
    count_hot = tl.sum((valid & (key < n_hot)).to(tl.int32), axis=0)
    tail = offs >= n_groups
    tl.store(goff_h_ptr + gidx, tl.minimum(less, count_hot), mask=start)
    tl.store(goff_h_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32) + count_hot,
             mask=tail & (offs <= n))
    tl.store(gids_h_ptr + gidx, tl.where(key < n_hot, key, 0), mask=start)
    tl.store(gids_h_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=tail & valid)

    if HOST:
        # The host launch's tables are the full ones rotated left by the HOT
        # groups: entry t reads group t + groups_hot, index clamped to n for
        # goff and to n - 1 for gids.
        count_native = tl.sum((valid & (key < n_hot + n_second)).to(tl.int32), axis=0)
        groups_hot = tl.sum((start & (key < n_hot)).to(tl.int32), axis=0)
        t = gidx - groups_hot
        mine = start & (t >= 0)
        tl.store(goff_c_ptr + t, tl.minimum(less, count_native), mask=mine)
        tl.store(gids_c_ptr + t, tl.minimum(tl.maximum(key - n_hot, 0), n_second - 1), mask=mine)
        tail_c = offs >= n_groups - groups_hot
        tl.store(goff_c_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32) + count_native,
                 mask=tail_c & (offs <= n))
        # A tail entry's gids index min(t + groups_hot, n - 1) is a real group
        # only when every route is its own group (n_groups == n): then it is
        # group n - 1, whose key is the largest.
        last = tl.max(tl.where(valid, key, -1), axis=0)
        fill = tl.where(n_groups == n, tl.minimum(tl.maximum(last - n_hot, 0), n_second - 1), 0)
        tl.store(gids_c_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32) + fill, mask=tail_c & valid)


def _block_for(n: int) -> int:
    return triton.next_power_of_2(n + 1)


def _warps_for(block: int) -> int:
    """One lane a route: 1 warp up to 32 lanes, 2 at 64, 4 at 128."""
    return max(1, block // 32)


def _check_map(name: str, m: torch.Tensor, device: torch.device) -> None:
    if m.shape != (NUM_EXPERTS,) or m.dtype != torch.int32 or not m.is_contiguous():
        raise ValueError(f"{name} must be a contiguous int32 [384] map")
    if m.device != device:
        raise ValueError(f"{name} is on {m.device}, the planner on {device}")


class GroupPlanner:
    """The decode group plan of one NativeV41Moe layer, one launch a call."""

    def __init__(self, keys: PlanKeys, max_tokens: int, device: torch.device):
        if device.type != "cuda":
            raise ValueError(f"the group-plan kernel runs on CUDA, not {device}")
        if not 0 < max_tokens <= MAX_TOKENS:
            raise ValueError(f"max_tokens must be 1..{MAX_TOKENS}")
        _check_map("hot slot map", keys.hot_slot, device)
        if isinstance(keys, SplitKeys):
            _check_map("second slot map", keys.second_slot, device)
            if keys.n_hot < 1 or keys.n_second < 1:
                raise ValueError(f"a split layer needs both banks ({keys.n_hot}, {keys.n_second})")
            self._second, self._n_hot, self._n_second = keys.second_slot, keys.n_hot, keys.n_second
            self._split, host = True, keys.host_launch
        elif isinstance(keys, ResidentKeys):
            # No second map: the kernel never reads it; pass the hot map.
            self._second, self._n_hot, self._n_second = keys.hot_slot, NUM_EXPERTS, 1
            self._split, host = False, False
        else:
            raise TypeError(f"plan keys must be ResidentKeys or SplitKeys, got {type(keys)}")
        self.keys = keys
        self.max_tokens = max_tokens
        self.device = device
        slots = max_tokens * TOP_K
        self.buffers = GroupPlanBuffers.allocate(slots, host, device)
        # Compile (and run once) every M's variant before any CUDA graph capture.
        # {M: (BLOCK, warps, registers, spills)} of the compiled variants.
        self.compiled: dict[int, tuple[int, int, int, int]] = {}
        warm = torch.zeros((max_tokens, TOP_K), dtype=torch.int32, device=device)
        for m in range(1, max_tokens + 1):
            self.plan(warm[:m])
        torch.cuda.synchronize(device)

    def plan(self, ids: torch.Tensor) -> PlanViews:
        """ids int32 [M, 6] (global expert ids, contiguous) -> this call's plan."""
        m = ids.size(0) if ids.dim() == 2 else -1
        if ids.shape != (m, TOP_K) or not 0 < m <= self.max_tokens:
            raise ValueError(f"bad routing shape {tuple(ids.shape)} for max_tokens {self.max_tokens}")
        if ids.dtype != torch.int32 or not ids.is_contiguous() or ids.device != self.device:
            raise ValueError(f"ids must be contiguous int32 on {self.device}, got {ids.dtype} "
                             f"on {ids.device}")
        n = m * TOP_K
        b = self.buffers
        host = b.host
        gids_c = host.gids if host is not None else b.gids_hot
        goff_c = host.goff if host is not None else b.goff_hot
        block = _block_for(n)
        warps = _warps_for(block)
        compiled = _group_plan_kernel[(1,)](
            ids, self.keys.hot_slot, self._second,
            b.perm, b.gids_hot, b.goff_hot, gids_c, goff_c,
            b.scratch_key, b.scratch_start,
            n, self._n_hot, self._n_second,
            SPLIT=self._split, HOST=host is not None, BLOCK=block,
            num_warps=warps,
        )
        if m not in self.compiled:
            self.compiled[m] = (block, warps, int(compiled.n_regs), int(compiled.n_spills))
        return PlanViews(
            b.perm[:n], b.gids_hot[:n], b.goff_hot[: n + 1],
            (host.gids[:n], host.goff[: n + 1]) if host is not None else None,
        )
