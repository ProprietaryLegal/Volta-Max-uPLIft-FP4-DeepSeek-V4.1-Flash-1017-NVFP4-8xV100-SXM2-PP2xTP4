# SPDX-License-Identifier: Apache-2.0
"""NVFP4-8R prefill QPN launches with four 8-row blocks per weight load (RB 4).

forward_prefill runs its QPN projections (the HOT launch and each staged batch
of host-RAM experts) through the installed ``torch.ops._C.skinny_moe_qpn_sm70``.
That op runs a prefill launch (more than 8 rows per expert on average) at RB 2:
each CTA re-reads and re-dequantizes its expert's weight tile once per 16 rows.
Under skewed routing that re-read dominates (measured on one card before the patch).

``v41qpnrb4.so`` (beside this module) is the same kernel source
(skinny_moe_qpn_sm70.cu, sha256 31d6ea4f..., template text unchanged) with one
appended launcher: the installed m8n8k4 launcher's checks and grid for SPLITK 8
and MXFP4 scales; where the installed op runs RB 2 it runs RB 4, and every other
launch runs RB 1 as installed. Every row keeps its accumulators and K order at
any RB (the source's own statement, proven by the one-card tests), so the output
bits are the installed op's.

The engine key ``nvfp4_8r_prefill_rb`` is required, with no default:
``rb2`` calls the installed op (today's code path), ``rb4`` this library.
Decode is not touched.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

PREFILL_RB_KEY = "nvfp4_8r_prefill_rb"
# The library's stem must never be a module name of this package: Python's
# finder tries extension modules before source, so a nvfp4_8r_qpn_rb.so here
# would be imported in place of this file (seen on a first install, 2026-11-12). The
# name follows the package's other kernel libraries (v41prefmoe.so, k2attn.so).
LIBRARY = Path(__file__).with_name("v41qpnrb4.so")
# sha256 of the library built by engine/kernels/prefill-rb4/build_qpn_rb4.py
# (built as nvfp4_8r_qpn_rb.so, installed under the name above; same bytes).
LIBRARY_SHA256 = "2b893e7da391694ea96862e4cd1978785667e46cc375a7d389f5727cc5d63c48"
OP_NAMESPACE = "nvfp4_8r_qpn_rb"


class PrefillRb(enum.Enum):
    """Which op runs forward_prefill's QPN launches (nvfp4_8r_prefill_rb)."""

    # The installed torch.ops._C.skinny_moe_qpn_sm70 (RB 2 in prefill).
    RB2 = "rb2"
    # v41qpnrb4.so's moe_prefill at RB 4 (this module).
    RB4 = "rb4"


def check_prefill_rb(mode: object) -> PrefillRb:
    if type(mode) is not PrefillRb:
        raise ValueError(f"{PREFILL_RB_KEY}: {mode!r} is not a PrefillRb")
    return mode


def parse_prefill_rb(extra: Mapping[str, object]) -> PrefillRb:
    """The prefill QPN choice of model-loader-extra-config; no default."""
    names = [m.value for m in PrefillRb]
    if PREFILL_RB_KEY not in extra:
        raise ValueError(f"{PREFILL_RB_KEY} is required ({' or '.join(names)})")
    raw = extra[PREFILL_RB_KEY]
    if type(raw) is not str or raw not in names:
        raise ValueError(f"{PREFILL_RB_KEY} must be {' or '.join(names)}, got {raw!r}")
    return PrefillRb(raw)


def load_library() -> None:
    """Load v41qpnrb4.so once; refuse a missing or different file."""
    if hasattr(torch.ops, OP_NAMESPACE) and hasattr(getattr(torch.ops, OP_NAMESPACE), "moe_prefill"):
        return
    if not LIBRARY.is_file():
        raise RuntimeError(f"{PREFILL_RB_KEY} rb4 needs {LIBRARY}; it does not exist")
    got = hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    if got != LIBRARY_SHA256:
        raise RuntimeError(f"{LIBRARY} is sha256 {got}, this module pins {LIBRARY_SHA256}")
    torch.ops.load_library(str(LIBRARY))
    if not hasattr(getattr(torch.ops, OP_NAMESPACE), "moe_prefill"):
        raise RuntimeError(f"{LIBRARY} loaded but registered no {OP_NAMESPACE}.moe_prefill")


def prefill_qpn(mode: PrefillRb, *args) -> None:
    """One QPN launch of forward_prefill, with the installed op's arguments
    (x, codes, scales, gscales, perm, gids, goff, topk, y, x_slot_major,
    num_tokens, splitk, scale_mode, fragment_order)."""
    if mode is PrefillRb.RB2:
        torch.ops._C.skinny_moe_qpn_sm70(*args)
    elif mode is PrefillRb.RB4:
        torch.ops.nvfp4_8r_qpn_rb.moe_prefill(*args, 4)
    else:
        raise ValueError(f"{PREFILL_RB_KEY}: {mode!r} is not a PrefillRb")
