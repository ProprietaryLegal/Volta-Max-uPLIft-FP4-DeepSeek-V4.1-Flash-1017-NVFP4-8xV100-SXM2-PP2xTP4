# SPDX-License-Identifier: Apache-2.0
"""NVFP4-8R decode SwiGLU in one Triton launch (plan 03 Part B, CAMPAIGN 8.4).

NativeV41Moe.forward's reference SwiGLU is a chain of torch ops on the w13
output y13 fp16 [S, 1152] (gate in columns 0..575, up in 576..1151):

    g = y13[:, :576].float().clamp(max=10.0)
    u = y13[:, 576:].float().clamp(-10.0, 10.0)
    inter[:, :576].copy_(silu(g) * u)        # inter fp16 [S, 640]

each op its own launch (six to seven graph nodes). This module writes the same
576 columns of ``inter``, bit for bit, in one launch: FP16 -> FP32, the same
clamps (NaN passes through, as torch.clamp's), silu as g / (1 + exp(-g)) with
libdevice's expf and round-to-nearest add and divide, a round-to-nearest
multiply, and one round-to-nearest cast to FP16. Columns 576..639 of ``inter``
(W2's zero K padding) are never written.

Production's one-launch op torch.ops.vllm.deepseek_v41_swiglu
(models/deepseek_v41/activations.py, kernel v41_shared_swiglu) has the same
dtypes and limits, but takes contiguous gate / up / out matrices, so on this
layout it would need two copies in and one copy out. Nothing syncs with the
host and the buffers are fixed, so the launch captures into a CUDA graph;
``SwigluKernel`` compiles and runs it at construction, before any capture.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping

import torch

from vllm.triton_utils import tl, triton

from triton.language.extra import libdevice

SWIGLU_KEY = "nvfp4_8r_swiglu"
# D3: the same choice for forward_prefill's SwiGLU (any M); required, no default.
PREFILL_SWIGLU_KEY = "nvfp4_8r_prefill_swiglu"
INTER = 576       # TP_INTER of nvfp4_8r_native: one rank's intermediate width
Y13_COLS = 2 * INTER
OUT_COLS = 640    # W2_K of nvfp4_8r_native: inter's row width (K padded)
LIMIT = 10.0      # SWIGLU_LIMIT of nvfp4_8r_native
BLOCK = 1024
NUM_WARPS = 4


class Swiglu(enum.Enum):
    """How a NativeV41Moe computes the limited SwiGLU: decode forward under
    nvfp4_8r_swiglu, forward_prefill under nvfp4_8r_prefill_swiglu."""

    # The reference chain of torch ops.
    CHAIN = "chain"
    # One Triton launch (this module).
    KERNEL = "kernel"


def check_swiglu(mode: object) -> Swiglu:
    if type(mode) is not Swiglu:
        raise ValueError(f"{SWIGLU_KEY}: {mode!r} is not a Swiglu")
    return mode


def _parse(extra: Mapping[str, object], key: str) -> Swiglu:
    names = [m.value for m in Swiglu]
    if key not in extra:
        raise ValueError(f"{key} is required ({' or '.join(names)})")
    raw = extra[key]
    if type(raw) is not str or raw not in names:
        raise ValueError(f"{key} must be {' or '.join(names)}, got {raw!r}")
    return Swiglu(raw)


def parse_swiglu(extra: Mapping[str, object]) -> Swiglu:
    """The decode SwiGLU mode of model-loader-extra-config; no default."""
    return _parse(extra, SWIGLU_KEY)


def parse_prefill_swiglu(extra: Mapping[str, object]) -> Swiglu:
    """The prefill SwiGLU mode of model-loader-extra-config; no default."""
    return _parse(extra, PREFILL_SWIGLU_KEY)


@triton.jit
def _swiglu_kernel(
    y13_ptr,
    inter_ptr,
    n,
    INTER_C: tl.constexpr,
    Y13_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    LIMIT_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < n
    row = offs // INTER_C
    col = offs - row * INTER_C
    src = y13_ptr + row * Y13_STRIDE + col
    g = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(src + INTER_C, mask=mask, other=0.0).to(tl.float32)
    # Comparisons, not min / max: a NaN input stays NaN, as in torch.clamp.
    g = tl.where(g > LIMIT_C, LIMIT_C, g)
    u = tl.where(u > LIMIT_C, LIMIT_C, u)
    u = tl.where(u < -LIMIT_C, -LIMIT_C, u)
    # torch's silu: x / (1 + exp(-x)) in FP32, IEEE add and divide, no fusion.
    act = libdevice.div_rn(g, libdevice.add_rn(1.0, libdevice.exp(-g)))
    val = libdevice.mul_rn(act, u)
    tl.store(inter_ptr + row * OUT_STRIDE + col, val.to(tl.float16), mask=mask)


def _check(y13: torch.Tensor, inter: torch.Tensor) -> int:
    s = y13.size(0)
    if (y13.dim() != 2 or y13.shape[1] != Y13_COLS or y13.dtype != torch.float16
            or y13.stride() != (Y13_COLS, 1)):
        raise ValueError(f"y13 must be fp16 [S, {Y13_COLS}] row-major, got {tuple(y13.shape)} "
                         f"{y13.dtype} stride {y13.stride()}")
    if (inter.dim() != 2 or tuple(inter.shape) != (s, OUT_COLS) or inter.dtype != torch.float16
            or inter.stride() != (OUT_COLS, 1)):
        raise ValueError(f"inter must be fp16 [{s}, {OUT_COLS}] row-major, got {tuple(inter.shape)} "
                         f"{inter.dtype} stride {inter.stride()}")
    if inter.device != y13.device:
        raise ValueError(f"y13 on {y13.device}, inter on {inter.device}")
    return s


def swiglu_into(y13: torch.Tensor, inter: torch.Tensor) -> None:
    """inter[:, :576] = the limited SwiGLU of y13 (one launch, any S >= 1)."""
    s = _check(y13, inter)
    if s < 1:
        raise ValueError("y13 has no rows")
    if s * Y13_COLS >= 2**31:
        raise ValueError(f"{s} rows: element offsets would pass int32")
    n = s * INTER
    _swiglu_kernel[(triton.cdiv(n, BLOCK),)](
        y13, inter, n,
        INTER_C=INTER, Y13_STRIDE=Y13_COLS, OUT_STRIDE=OUT_COLS, LIMIT_C=LIMIT, BLOCK_C=BLOCK,
        num_warps=NUM_WARPS,
    )


class SwigluKernel:
    """The decode SwiGLU of one NativeV41Moe layer, one launch a call."""

    def __init__(self, max_tokens: int, top_k: int, device: torch.device):
        if device.type != "cuda":
            raise ValueError(f"the SwiGLU kernel runs on CUDA, not {device}")
        if max_tokens < 1 or top_k < 1:
            raise ValueError(f"max_tokens {max_tokens} and top_k {top_k} must be positive")
        self.max_rows = max_tokens * top_k
        # Compile (and run once) before any CUDA graph capture, on scratch
        # buffers: the layer's own buffers are not touched.
        y13 = torch.zeros((self.max_rows, Y13_COLS), dtype=torch.float16, device=device)
        inter = torch.zeros((self.max_rows, OUT_COLS), dtype=torch.float16, device=device)
        for s in (1, self.max_rows):
            swiglu_into(y13[:s], inter[:s])
        torch.cuda.synchronize(device)

    def __call__(self, y13: torch.Tensor, inter: torch.Tensor) -> None:
        if y13.size(0) > self.max_rows:
            raise ValueError(f"{y13.size(0)} rows, the layer holds {self.max_rows}")
        swiglu_into(y13, inter)
