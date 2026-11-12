# SPDX-License-Identifier: Apache-2.0
"""NVFP4-8R three-tier decode tail in one Triton launch (plan 06).

A three-tier NativeV41Moe layer adds two partials: the native one (the QPN
rows y2 of the HOT and HOST routes, weighted by the router) and the IQ one
(production's IQ op over the WARM routes). The reference tail is a chain:

    iq_ids = iq_slot_of_expert[topk_ids.long()]            # 2 launches
    w_native = torch.where(iq_ids < 0, topk_weights, 0.0)  # 3 launches
    _mtp_weighted_reduce(y2, w_native, out)                # 1 launch
    part = IQ op(x, iq_ids, topk_weights)                  # 4 launches
    return out + part.float()                              # 2 launches

With ``fused`` the group-plan kernel writes iq_ids in its own launch
(nvfp4_8r_group_plan.TierKeys) and this module's kernel does the rest of the
tail in the reduce's launch: the same FP32 sum over the six slots in slot
order with the weight of a WARM route replaced by 0.0, then one FP32 add of
the IQ partial, stored into the layer's FP32 buffer. Seven launches fewer,
the same bits (test_iq_tail.py). The IQ op itself is not changed.

Nothing syncs with the host and the buffers are fixed, so the launch captures
into a CUDA graph; ``IqTailKernel`` compiles and runs it at construction,
before any capture.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping

import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
from vllm.triton_utils import tl, triton

IQ_TAIL_KEY = "nvfp4_8r_iq_tail"
TOP_K = 6
HIDDEN = 5120
BLOCK = 256     # the upstream reduce's (fused_moe/sm70/reduction.py)
NUM_WARPS = 4


class IqTail(enum.Enum):
    """How a three-tier NativeV41Moe's decode forward joins its two partials."""

    # The reference chain of torch ops around the reduce and the IQ op.
    TORCH = "torch"
    # IQ slots from the group-plan kernel, one launch for the rest (this module).
    FUSED = "fused"


def check_iq_tail(
    mode: object, residency: tiers.Residency, group_plan: gplan.GroupPlan
) -> IqTail | None:
    """``mode`` if it is legal for ``residency`` and ``group_plan``: None
    exactly for two_tier (no IQ tail), an IqTail otherwise, and FUSED only
    with the group-plan kernel (the chain plan writes no IQ slots)."""
    plan = gplan.check_group_plan(group_plan)
    if isinstance(residency, tiers.TwoTier):
        if mode is not None:
            raise ValueError(
                f"{IQ_TAIL_KEY} is only valid with a WARM tier: {tiers.RESIDENCY_KEY} two_tier has none"
            )
        return None
    if not isinstance(residency, (tiers.HotIq, tiers.HotIqHost)):
        raise ValueError(f"{tiers.RESIDENCY_KEY}: {residency!r} is not a residency")
    if type(mode) is not IqTail:
        raise ValueError(f"{IQ_TAIL_KEY}: {mode!r} is not an IqTail")
    if mode is IqTail.FUSED and plan is not gplan.GroupPlan.KERNEL:
        raise ValueError(f"{IQ_TAIL_KEY} fused needs {gplan.GROUP_PLAN_KEY} kernel")
    return mode


def parse_iq_tail(
    extra: Mapping[str, object], residency: tiers.Residency, group_plan: gplan.GroupPlan
) -> IqTail | None:
    """The IQ tail of model-loader-extra-config; no default. Required with
    hot_iq and hot_iq_host, refused with two_tier."""
    names = [m.value for m in IqTail]
    if isinstance(residency, tiers.TwoTier):
        if IQ_TAIL_KEY in extra:
            raise ValueError(
                f"{IQ_TAIL_KEY} is only valid with {tiers.RESIDENCY_KEY} hot_iq or hot_iq_host"
            )
        return check_iq_tail(None, residency, group_plan)
    if IQ_TAIL_KEY not in extra:
        raise ValueError(f"{IQ_TAIL_KEY} is required ({' or '.join(names)})")
    raw = extra[IQ_TAIL_KEY]
    if type(raw) is not str or raw not in names:
        raise ValueError(f"{IQ_TAIL_KEY} must be {' or '.join(names)}, got {raw!r}")
    return check_iq_tail(IqTail(raw), residency, group_plan)


@triton.jit
def _iq_tail_kernel(
    y2_ptr,
    weights_ptr,
    iq_ptr,
    part_ptr,
    out_ptr,
    HIDDEN_C: tl.constexpr,
    TOP_K_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offsets < HIDDEN_C
    acc = tl.zeros((BLOCK_C,), tl.float32)
    # The loop of _mtp_weighted_reduce_kernel, with torch.where(iq < 0, w, 0.0)
    # as the weight: a WARM route's y2 row (older values) is multiplied by 0.0.
    for slot in tl.static_range(0, TOP_K_C):
        route = token * TOP_K_C + slot
        values = tl.load(y2_ptr + route * HIDDEN_C + offsets, mask=mask, other=0.0)
        weight = tl.load(weights_ptr + route)
        iq = tl.load(iq_ptr + route)
        weight = tl.where(iq < 0, weight, 0.0)
        acc += values.to(tl.float32) * weight
    part = tl.load(part_ptr + token * HIDDEN_C + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + token * HIDDEN_C + offsets, acc + part.to(tl.float32), mask=mask)


def _want(name: str, t: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype,
          device: torch.device) -> None:
    if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous() or t.device != device:
        raise ValueError(
            f"{name} must be contiguous {dtype} {shape} on {device}, got {t.dtype} "
            f"{tuple(t.shape)} on {t.device}{'' if t.is_contiguous() else ' (not contiguous)'}"
        )


def iq_tail_into(
    y2: torch.Tensor,
    topk_weights: torch.Tensor,
    iq_ids: torch.Tensor,
    part: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """out[t] = sum over the six slots of float32(y2[6 t + slot]) * w, with
    w = topk_weights[t, slot] where iq_ids[t, slot] < 0 and 0.0 elsewhere,
    plus float32(part[t]). One launch, M >= 1.

    y2 fp16 [6 M, 5120], topk_weights fp32 [M, 6], iq_ids int32 [M, 6] or
    [6 M], part fp16 [M, 5120], out fp32 [M, 5120], all contiguous."""
    m = out.size(0)
    if m < 1:
        raise ValueError("out has no rows")
    dev = out.device
    _want("out", out, (m, HIDDEN), torch.float32, dev)
    _want("y2", y2, (m * TOP_K, HIDDEN), torch.float16, dev)
    _want("topk_weights", topk_weights, (m, TOP_K), torch.float32, dev)
    _want("part", part, (m, HIDDEN), torch.float16, dev)
    if iq_ids.numel() != m * TOP_K or iq_ids.dtype != torch.int32 or not iq_ids.is_contiguous() \
            or iq_ids.device != dev:
        raise ValueError(
            f"iq_ids must be contiguous int32 with {m * TOP_K} entries on {dev}, got "
            f"{iq_ids.dtype} {tuple(iq_ids.shape)} on {iq_ids.device}"
        )
    _iq_tail_kernel[(m, triton.cdiv(HIDDEN, BLOCK))](
        y2, topk_weights, iq_ids, part, out,
        HIDDEN_C=HIDDEN, TOP_K_C=TOP_K, BLOCK_C=BLOCK,
        num_warps=NUM_WARPS,
    )


class IqTailKernel:
    """The fused decode tail of one three-tier NativeV41Moe layer."""

    def __init__(self, max_tokens: int, device: torch.device):
        if device.type != "cuda":
            raise ValueError(f"the IQ tail kernel runs on CUDA, not {device}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens {max_tokens} must be positive")
        self.max_tokens = max_tokens
        # Compile (and run once) before any CUDA graph capture, on scratch
        # buffers: the layer's own buffers are not touched.
        s = max_tokens * TOP_K
        y2 = torch.zeros((s, HIDDEN), dtype=torch.float16, device=device)
        w = torch.zeros((max_tokens, TOP_K), dtype=torch.float32, device=device)
        iq = torch.full((s,), -1, dtype=torch.int32, device=device)
        part = torch.zeros((max_tokens, HIDDEN), dtype=torch.float16, device=device)
        out = torch.zeros((max_tokens, HIDDEN), dtype=torch.float32, device=device)
        for m in sorted({1, max_tokens}):
            iq_tail_into(y2[: m * TOP_K], w[:m], iq[: m * TOP_K], part[:m], out[:m])
        torch.cuda.synchronize(device)

    def __call__(
        self,
        y2: torch.Tensor,
        topk_weights: torch.Tensor,
        iq_ids: torch.Tensor,
        part: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if out.size(0) > self.max_tokens:
            raise ValueError(f"{out.size(0)} rows, the layer holds {self.max_tokens}")
        iq_tail_into(y2, topk_weights, iq_ids, part, out)
