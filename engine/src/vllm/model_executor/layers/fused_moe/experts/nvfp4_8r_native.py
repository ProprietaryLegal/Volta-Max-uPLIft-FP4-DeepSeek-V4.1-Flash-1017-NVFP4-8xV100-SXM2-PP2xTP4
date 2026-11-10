# SPDX-License-Identifier: Apache-2.0
"""NVFP4-8R native V4.1 routed experts: MXFP4 E8M0/32 packs on SM70 QPN.

Packs come from benchmarks.nvfp4_8r.build_packs: per (layer, TP rank) a hot
and a cold file, experts in ascending id order, each expert one fixed record:
  w1 codes [576, 2560] | w1 E8M0 [576, 160] | w3 codes | w3 E8M0 |
  w2 codes [5120, 320] | w2 E8M0 [5120, 20]
W2's K is zero-padded 576 -> 640 (codes zero; the padding scale bytes are
rewritten to the expert's last real scale so they never widen the rebase span).

The forward is the grouped QPN path (skinny_moe_qpn_sm70, E8M0 scale mode)
with V4.1's limited SwiGLU (min(g, 10), clamp(u, +-10)) in FP32 and an FP32
weighted reduce. All shapes are fixed per M and nothing syncs with the host,
so it captures into a CUDA graph.
"""

from __future__ import annotations

import json
import math
import os
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    _QPN_MAX_M,
    QpnFragmentOrder,
    grouped_splitk,
    qpn_fragment_order,
    qpn_prepack,
    rebase_e8m0_for_fp16,
)
from vllm.model_executor.layers.fused_moe.sm70.reduction import _mtp_weighted_reduce
from vllm.model_executor.layers.quantization.gguf_expert_cache import (
    _native as _cache_native,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_8r_warm_prefill import (
        IqLayerPrefill,
        IqStagePrefill,
    )

logger = init_logger(__name__)

HIDDEN = 5120
TP_INTER = 576
W2_K = 640
TOP_K = 6
NUM_EXPERTS = 384
SWIGLU_LIMIT = 10.0
PACK_FORMAT = "nvfp4-8r-mxfp4-v1"

_W13_CODES = TP_INTER * HIDDEN // 2
_W13_SCALES = TP_INTER * HIDDEN // 32
_W2_CODES = HIDDEN * W2_K // 2
_W2_SCALES = HIDDEN * W2_K // 32
EXPERT_BYTES = 2 * (_W13_CODES + _W13_SCALES) + _W2_CODES + _W2_SCALES
assert EXPERT_BYTES == 4_874_240

# -- prefill timing tool (plan 04 step 1) -------------------------------------
PREFILL_TRACE_ENV = "VLLM_NVFP4_8R_PREFILL_TRACE"
PREFILL_TRACE_WINDOW = 50


@dataclass
class PrefillTrace:
    """Sums over this process's forward_prefill calls, logged and reset every
    PREFILL_TRACE_WINDOW calls. Aggregates only: calls, rows, experts staged
    from host RAM, seconds in the staged launches, in the IQ part and in the
    whole call. The marks synchronize the device, so a traced run is a little
    slower than an untraced one (the first staging copy cannot overlap the
    hot launch)."""

    calls: int = 0
    rows: int = 0
    staged_experts: int = 0
    seconds_staged: float = 0.0
    seconds_iq: float = 0.0
    seconds_total: float = 0.0

    def add(
        self, rows: int, staged_experts: int, seconds_staged: float,
        seconds_iq: float, seconds_total: float,
    ) -> None:
        self.calls += 1
        self.rows += rows
        self.staged_experts += staged_experts
        self.seconds_staged += seconds_staged
        self.seconds_iq += seconds_iq
        self.seconds_total += seconds_total
        if self.calls == PREFILL_TRACE_WINDOW:
            logger.info(
                "nvfp4_8r prefill trace: calls=%d rows=%d staged_experts=%d "
                "seconds_staged=%.9f seconds_iq=%.9f seconds_total=%.9f",
                self.calls, self.rows, self.staged_experts, self.seconds_staged,
                self.seconds_iq, self.seconds_total,
            )
            self.calls = self.rows = self.staged_experts = 0
            self.seconds_staged = self.seconds_iq = self.seconds_total = 0.0


def _prefill_trace_from_env(raw: str | None) -> PrefillTrace | None:
    if raw is None or raw == "0":
        return None
    if raw == "1":
        return PrefillTrace()
    raise ValueError(f"{PREFILL_TRACE_ENV} must be 0 or 1, got {raw!r}")


_PREFILL_TRACE = _prefill_trace_from_env(os.environ.get(PREFILL_TRACE_ENV))

LOAD_TRACE_ENV = "VLLM_NVFP4_8R_LOAD_TRACE"


def _load_trace_from_env(raw: str | None) -> bool:
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{LOAD_TRACE_ENV} must be 0 or 1, got {raw!r}")


_LOAD_TRACE = _load_trace_from_env(os.environ.get(LOAD_TRACE_ENV))


def _load_trace(device: torch.device, rank: int, at: str, tier) -> None:
    """One line of card and torch allocator state during the stage build
    (LOAD_TRACE_ENV=1 only). Counts and bytes only; ``tier`` is the layer's
    tier array or None outside a layer."""
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    allocated = stats["allocated_bytes.all.current"]
    reserved = stats["reserved_bytes.all.current"]
    counts = [0, 0, 0] if tier is None else [
        int((tier == t).sum())
        for t in (tiers.Tier.HOT, tiers.Tier.WARM, tiers.Tier.HOST)
    ]
    # card_used = allocated (tensors) + cached (torch holds, no tensor uses)
    # + outside_torch (the context, CUDA-IPC banks, any other process).
    logger.info(
        "nvfp4_8r load trace: rank=%d at=%s hot=%d warm=%d host=%d card_free=%d "
        "card_used=%d allocated=%d cached=%d outside_torch=%d inactive_split=%d "
        "alloc_retries=%d ooms=%d",
        rank, at, *counts, free, total - free, allocated, reserved - allocated,
        total - free - reserved, stats["inactive_split_bytes.all.current"],
        stats["num_alloc_retries"], stats["num_ooms"],
    )


def _mark(device: torch.device) -> float:
    """The clock once everything queued on ``device`` has run."""
    torch.cuda.synchronize(device)
    return time.perf_counter()


def read_manifest(packs: Path) -> list[list[dict[str, list[int]]]]:
    manifest = json.loads((packs / "manifest.json").read_text())
    if manifest["format"] != PACK_FORMAT:
        raise ValueError(f"unknown pack format {manifest['format']!r}")
    return manifest["layers"]


def read_rank_experts(packs: Path, layer: int, rank: int) -> np.ndarray:
    """All 384 expert records of one (layer, rank), indexed by expert id."""
    split = read_manifest(packs)[layer][rank]
    out = np.empty((NUM_EXPERTS, EXPERT_BYTES), np.uint8)
    seen = 0
    for temp in ("hot", "cold"):
        ids = split[temp]
        path = packs / f"L{layer:02d}-r{rank}-{temp}.bin"
        data = np.fromfile(path, np.uint8)
        if data.size != len(ids) * EXPERT_BYTES:
            raise ValueError(f"{path}: {data.size} bytes for {len(ids)} experts")
        out[np.asarray(ids, np.int64)] = data.reshape(len(ids), EXPERT_BYTES)
        seen += len(ids)
    if seen != NUM_EXPERTS or sorted(split["hot"] + split["cold"]) != list(
        range(NUM_EXPERTS)
    ):
        raise ValueError(f"L{layer} r{rank}: hot+cold is not a partition of 384")
    return out


def _split_records(raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
    e = raw.size(0)
    if raw.dtype != torch.uint8 or raw.shape != (e, EXPERT_BYTES):
        raise ValueError(f"expert records must be uint8 [E, {EXPERT_BYTES}]")
    o = 0
    parts = []
    for size, shape in (
        (_W13_CODES, (TP_INTER, HIDDEN // 2)),
        (_W13_SCALES, (TP_INTER, HIDDEN // 32)),
        (_W13_CODES, (TP_INTER, HIDDEN // 2)),
        (_W13_SCALES, (TP_INTER, HIDDEN // 32)),
        (_W2_CODES, (HIDDEN, W2_K // 2)),
        (_W2_SCALES, (HIDDEN, W2_K // 32)),
    ):
        parts.append(raw[:, o : o + size].view(e, *shape))
        o += size
    return tuple(parts)


def _cuda_index(device: torch.device) -> int:
    """The card a set lives on. An index-less ``cuda`` device means "the
    current card" and differs per TP worker: the caller resolves it once."""
    if device.type != "cuda" or device.index is None:
        raise ValueError(f"nvfp4_8r needs an indexed CUDA device, got {device}")
    return device.index


@dataclass(frozen=True)
class NativeExpertSet:
    """Device-resident experts in QPN fragment order with rebased E8M0."""

    q13: torch.Tensor  # [E, 1152, 2560] uint8
    s13: torch.Tensor  # [E, 1152 * 160] uint8
    q2: torch.Tensor  # [E, 5120, 320] uint8
    s2: torch.Tensor  # [E, 5120 * 20] uint8
    g1: torch.Tensor  # [E] float32
    g2: torch.Tensor  # [E] float32
    order: QpnFragmentOrder
    # Pinned host tensors behind UVA views (from_raw_host); empty in HBM.
    host_backing: tuple[torch.Tensor, ...] = ()

    @staticmethod
    def _prepack(raw: torch.Tensor, device: torch.device, order: QpnFragmentOrder):
        """Prepack E expert records on ``device``: (q13, s13, q2, s2, g1, g2)."""
        w1c, w1s, w3c, w3s, w2c, w2s = _split_records(raw)
        if bool((w2c[..., TP_INTER // 2 :] != 0).any()):
            raise ValueError("W2 K padding codes are not zero")
        q13 = torch.cat((w1c, w3c), 1).to(device)
        s13 = torch.cat((w1s, w3s), 1).to(device)
        q2 = w2c.contiguous().to(device)
        s2 = w2s.contiguous().to(device)
        # Padding columns carry zero codes: any scale is exact there.
        s2[..., TP_INTER // 32 :] = s2[..., TP_INTER // 32 - 1 : TP_INTER // 32]
        g1 = rebase_e8m0_for_fp16(s13).to(device)
        g2 = rebase_e8m0_for_fp16(s2).to(device)
        for e in range(q13.size(0)):
            for q, sc in ((q13, s13), (q2, s2)):
                qc, qs = qpn_prepack(q[e], sc[e], 32, order)
                q[e].view(-1).copy_(qc)
                sc[e].view(-1).copy_(qs)
        return q13, s13, q2, s2, g1, g2

    @staticmethod
    def _shapes(n: int) -> list[tuple[int, ...]]:
        return [(n, 2 * TP_INTER, HIDDEN // 2), (n, 2 * TP_INTER, HIDDEN // 32),
                (n, HIDDEN, W2_K // 2), (n, HIDDEN, W2_K // 32)]

    @classmethod
    def _fill(cls, raw, device, order, dests, g1, g2, chunk: int) -> None:
        # Bounded load transients: at most ``chunk`` records staged in HBM.
        for i in range(0, raw.size(0), chunk):
            parts = cls._prepack(raw[i : i + chunk], device, order)
            for dest, part in zip(dests, parts[:4]):
                dest[i : i + part.size(0)].copy_(part)
            g1[i : i + chunk].copy_(parts[4])
            g2[i : i + chunk].copy_(parts[5])
            del parts

    @classmethod
    def from_raw(
        cls, raw: torch.Tensor, device: torch.device, chunk: int = 32
    ) -> NativeExpertSet:
        """HBM-resident set from host records, prepacked ``chunk`` at a time."""
        order = qpn_fragment_order(_cuda_index(device))
        n = raw.size(0)
        dests = [torch.empty(sh, dtype=torch.uint8, device=device) for sh in cls._shapes(n)]
        g1 = torch.empty(n, dtype=torch.float32, device=device)
        g2 = torch.empty(n, dtype=torch.float32, device=device)
        cls._fill(raw, device, order, dests, g1, g2, chunk)
        return cls(*dests, g1, g2, order)

    @classmethod
    def from_raw_host(
        cls, raw: torch.Tensor, device: torch.device, chunk: int = 32
    ) -> NativeExpertSet:
        """Registered-host set the QPN kernels read in place over PCIe. Same
        mechanism as the engine's expert cache (host_register + mapped_view,
        never a copy); only the per-expert global scales stay in HBM. Records
        are prepacked on the device ``chunk`` at a time and copied out."""
        order = qpn_fragment_order(_cuda_index(device))
        n = raw.size(0)
        shapes = cls._shapes(n)
        page = 4096
        spans = [(math.prod(sh) + page - 1) // page * page for sh in shapes]
        total = sum(spans)
        backing = torch.empty(total + page, dtype=torch.uint8)
        start = (-backing.data_ptr()) % page
        arena = backing[start : start + total]
        host_views, offset = [], 0
        for sh, span in zip(shapes, spans):
            host_views.append(arena[offset : offset + math.prod(sh)].view(sh))
            offset += span
        g1 = torch.empty(n, dtype=torch.float32, device=device)
        g2 = torch.empty(n, dtype=torch.float32, device=device)
        cls._fill(raw, device, order, host_views, g1, g2, chunk)
        lane = _cache_native()
        lane.host_register(arena.data_ptr(), total)
        mapped = lane.mapped_view(arena, _cuda_index(device))
        if mapped.data_ptr() != arena.data_ptr():
            raise RuntimeError("mapped host arena does not alias host memory (no UVA)")
        views, offset = [], 0
        for sh, span in zip(shapes, spans):
            views.append(mapped[offset : offset + math.prod(sh)].view(sh))
            offset += span
        out = cls(*views, g1, g2, order, (backing, arena))
        weakref.finalize(out, lane.host_unregister, arena.data_ptr())
        return out

    @property
    def num_slots(self) -> int:
        return self.q13.size(0)


_STAGED_FIELDS = ("q13", "s13", "q2", "s2", "g1", "g2")


@dataclass(frozen=True)
class PrefillStaging:
    """Two HBM banks of ``batch`` expert records plus the stream that fills
    them. One per card: prefill is eager and runs one layer at a time, and
    every layer's records have the same shape, so the layers of a card share
    it. (One per layer cost 2 x 32 x 4.87 MB x 20 layers = 5.8 GiB a card.)"""

    banks: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    copy_stream: torch.cuda.Stream

    def batch(self) -> int:
        return self.banks[0]["q13"].size(0)


_PREFILL_STAGING: dict[int, PrefillStaging] = {}


def prefill_staging(
    cold: NativeExpertSet, device: torch.device, batch: int
) -> PrefillStaging:
    """The staging of ``device``'s card, created on first use."""
    index = _cuda_index(device)
    staging = _PREFILL_STAGING.get(index)
    if staging is None:
        bank = lambda: {  # noqa: E731
            name: torch.empty((batch, *getattr(cold, name).shape[1:]),
                              dtype=getattr(cold, name).dtype, device=device)
            for name in _STAGED_FIELDS
        }
        staging = PrefillStaging((bank(), bank()), torch.cuda.Stream(device))
        _PREFILL_STAGING[index] = staging
    if staging.batch() != batch:
        raise ValueError(
            f"card {index} staging holds {staging.batch()} experts, asked {batch}"
        )
    for name in _STAGED_FIELDS:
        have = staging.banks[0][name].shape[1:]
        if have != getattr(cold, name).shape[1:]:
            raise ValueError(f"card {index} staging {name} {tuple(have)} != cold set")
    return staging


# -- WARM tier: production's IQ records (plan 01) ---------------------------
# One TP4 rank's rows of a projection, row-split (VLLM_V41_DOWN_ROWS=1):
# role -> (rows of all four ranks, rows of one rank, 256-weight blocks per row).
_IQ_ROWS = {"w1": (2304, 576, 20), "w3": (2304, 576, 20), "w2": (5120, 1280, 9)}
_IQ_FILES: dict[str, int] = {}  # slice-file descriptors, open for the process


@dataclass(frozen=True)
class IqShardSource:
    """Where one rank's rows of one projection of a layer's 384 IQ experts
    sit in a file: expert e's rows start at ``offset + e * expert_stride``."""

    role: str
    ggml_type: int
    fd: int
    offset: int
    expert_stride: int

    def __post_init__(self) -> None:
        if self.role not in _IQ_ROWS or self.ggml_type not in tiers.IQ_BLOCK_BYTES:
            raise ValueError(f"IQ shard {self.role!r} of GGML type {self.ggml_type!r}")
        if self.fd < 0 or self.offset < 0 or self.expert_stride < self.rows * self.row_bytes:
            raise ValueError(f"IQ shard {self.role}: bad file position")

    @property
    def rows(self) -> int:
        return _IQ_ROWS[self.role][1]

    @property
    def row_bytes(self) -> int:
        return _IQ_ROWS[self.role][2] * tiers.IQ_BLOCK_BYTES[self.ggml_type]

    @classmethod
    def from_gguf_tensor(cls, tensor, role: str, rank: int) -> IqShardSource:
        """This rank's rows inside the GGUF copy's [384, rows, row bytes] tensor."""
        from vllm.model_executor.layers.quantization.gguf_expert_storage import _source

        if role not in _IQ_ROWS or not 0 <= rank < 4:
            raise ValueError(f"IQ shard {role!r} for TP rank {rank}")
        all_rows, local_rows, blocks = _IQ_ROWS[role]
        kind = int(tensor.tensor_type)
        if kind not in tiers.IQ_BLOCK_BYTES:
            raise ValueError(f"routed-expert GGML type {kind} is not IQ1_M/IQ2_XXS/IQ2_XS")
        width = blocks * tiers.IQ_BLOCK_BYTES[kind]
        if tuple(tensor.data.shape) != (NUM_EXPERTS, all_rows, width):
            raise ValueError(
                f"{role} expert tensor is {tuple(tensor.data.shape)}, "
                f"want {(NUM_EXPERTS, all_rows, width)}"
            )
        source = _source(tensor)
        if source is None:
            raise ValueError(f"{role} expert tensor is not a file-backed memory map")
        fd, offset = source
        return cls(role, kind, fd, offset + rank * local_rows * width, all_rows * width)

    @classmethod
    def from_slice_file(cls, path: Path, role: str, ggml_type: int) -> IqShardSource:
        """A file holding exactly this rank's [384, rows, row bytes] slice."""
        import os

        key = str(path)
        if key not in _IQ_FILES:
            _IQ_FILES[key] = os.open(key, os.O_RDONLY)
        shard = cls(role, ggml_type, _IQ_FILES[key], 0, 0x7FFFFFFFFFFFFFFF)
        stride = shard.rows * shard.row_bytes
        if os.fstat(shard.fd).st_size != NUM_EXPERTS * stride:
            raise ValueError(f"{path}: size is not {NUM_EXPERTS} x {stride} bytes")
        return cls(role, ggml_type, shard.fd, 0, stride)

    def read(self, experts: list[int], out: np.ndarray) -> None:
        """The rows of ``experts`` back to back into ``out`` [n, rows, row bytes]."""
        from vllm.model_executor.layers.quantization.gguf_expert_storage import _read

        if out.shape != (len(experts), self.rows, self.row_bytes) or out.dtype != np.uint8:
            raise ValueError(f"IQ read buffer {out.shape} for {len(experts)} experts")
        length = self.rows * self.row_bytes
        _read(self.fd, [(self.offset + e * self.expert_stride, length) for e in experts], out)


@dataclass(frozen=True)
class IqLayerSource:
    """The three IQ shards of one layer for one TP rank."""

    w1: IqShardSource
    w3: IqShardSource
    w2: IqShardSource

    def __post_init__(self) -> None:
        if (self.w1.role, self.w3.role, self.w2.role) != ("w1", "w3", "w2"):
            raise ValueError("IQ layer source takes its shards as w1, w3, w2")
        if self.w1.ggml_type != self.w3.ggml_type:
            raise ValueError(
                f"gate type {self.w1.ggml_type} != up type {self.w3.ggml_type}: "
                "the IQ op takes one type for both"
            )

    @property
    def type13(self) -> int:
        return self.w1.ggml_type

    @property
    def type2(self) -> int:
        return self.w2.ggml_type

    @property
    def record_bytes(self) -> int:
        return tiers.iq_record_bytes(self.type13, self.type2)


@dataclass(frozen=True)
class IqWarmBank:
    """One layer's WARM experts on one card in production's device layout:
    w1, w3 uint8 [W, 576, 20 x block bytes], w2 uint8 [W, 1280, 9 x block
    bytes]; slot = rank of the expert among the layer's WARM ids."""

    w1: torch.Tensor
    w3: torch.Tensor
    w2: torch.Tensor
    type13: int
    type2: int

    def __post_init__(self) -> None:
        n = self.w1.shape[0] if self.w1.ndim == 3 else 0
        for role, bank, kind in (
            ("w1", self.w1, self.type13), ("w3", self.w3, self.type13), ("w2", self.w2, self.type2)
        ):
            if kind not in tiers.IQ_BLOCK_BYTES:
                raise ValueError(f"IQ bank {role}: GGML type {kind!r}")
            _, rows, blocks = _IQ_ROWS[role]
            want = (n, rows, blocks * tiers.IQ_BLOCK_BYTES[kind])
            if n < 1 or tuple(bank.shape) != want or bank.dtype != torch.uint8:
                raise ValueError(f"IQ bank {role} is {tuple(bank.shape)} {bank.dtype}, want {want}")
            if bank.device != self.w1.device:
                raise ValueError("IQ banks of one layer sit on different devices")

    @property
    def num_slots(self) -> int:
        return int(self.w1.shape[0])

    @classmethod
    def from_source(
        cls, source: IqLayerSource, experts: list[int], device: torch.device
    ) -> IqWarmBank:
        """Read the IQ records of ``experts`` (ascending ids) onto ``device``."""
        from vllm.model_executor.layers.quantization.gguf_native import empty_guarded_weight

        if not experts or experts != sorted(set(experts)) or not (
            0 <= experts[0] and experts[-1] < NUM_EXPERTS
        ):
            raise ValueError("warm experts must be distinct ascending ids in 0..383")
        banks = []
        for shard in (source.w1, source.w3, source.w2):
            shape = (len(experts), shard.rows, shard.row_bytes)
            host = torch.empty(shape, dtype=torch.uint8, pin_memory=device.type == "cuda")
            shard.read(experts, host.numpy())
            bank = empty_guarded_weight(shape, shard.ggml_type, device)
            bank.copy_(host)
            banks.append(bank)
        return cls(banks[0], banks[1], banks[2], source.type13, source.type2)


def split_maps(
    hot: list[int], cold: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """int32 [384] slot maps of a hot/cold partition, -1 where absent."""
    if sorted(hot + cold) != list(range(NUM_EXPERTS)):
        raise ValueError("hot and cold experts must partition 0..383")
    maps = []
    for ids in (hot, cold):
        m = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32)
        m[torch.tensor(ids, dtype=torch.long)] = torch.arange(len(ids), dtype=torch.int32)
        maps.append(m.to(device))
    return maps[0], maps[1]


@dataclass(frozen=True)
class WarmTier:
    """The WARM side of a three-tier layer (plan 01): the IQ bank and the
    two extra slot maps, int32 [384], -1 where the expert is absent.
    ``host_slot_of_expert`` is the cold-set slot of each HOST expert;
    ``host_in_decode`` is true when the layer has one (a second QPN launch
    per projection over the host-RAM set). ``prefill`` is the layer's handle
    on production's IQ prefill over this bank (plan 04 part B), or None when
    prefill runs a WARM expert from its native record staged from host RAM."""

    iq: IqWarmBank
    iq_slot_of_expert: torch.Tensor
    host_slot_of_expert: torch.Tensor
    host_in_decode: bool
    prefill: IqLayerPrefill | None


class NativeV41Moe:
    """Rank-local V4.1 routed-expert MoE over a NativeExpertSet.

    ``slot_of_expert`` [384] int32 maps a global expert id to its slot in the
    set. Without ``cold`` every routed id must map to a slot (all resident).
    With ``cold`` (a host-resident set, P3) and its ``cold_slot_of_expert``
    the two maps must partition the 384 ids; one compaction sorts hot slots
    first, and each projection runs the grouped kernel twice into the same
    slot rows: once over the HBM bank, once over the host bank with its
    groups rotated to the front (the kernel's grid covers min(S, E) groups).

    With ``warm`` (plan 01, built by ``three_tier``) the cold set holds every
    non-HOT expert for prefill, but decode runs a WARM expert's routes in
    production's IQ op from the HBM bank; only HOST routes use the cold set.
    """

    def __init__(
        self,
        experts: NativeExpertSet,
        slot_of_expert: torch.Tensor,
        max_tokens: int,
        cold: NativeExpertSet | None = None,
        cold_slot_of_expert: torch.Tensor | None = None,
        warm: WarmTier | None = None,
    ):
        if (cold is None) != (cold_slot_of_expert is None):
            raise ValueError("cold set and cold slot map come together")
        if warm is not None and cold is None:
            raise ValueError("a WARM tier needs the host-RAM native set for prefill")
        if not 0 < max_tokens <= _QPN_MAX_M:
            raise ValueError(f"max_tokens must be 1..{_QPN_MAX_M}")
        if slot_of_expert.shape != (NUM_EXPERTS,) or slot_of_expert.dtype != torch.int32:
            raise ValueError("slot_of_expert must be int32 [384]")
        dev = experts.q13.device
        self.experts = experts
        self.slot_of_expert = slot_of_expert.to(dev)
        self.max_tokens = max_tokens
        self.cold = cold
        self.cold_slot_of_expert = None
        if cold is not None:
            assert cold_slot_of_expert is not None
            if cold.order != experts.order:
                raise ValueError("hot and cold sets use different fragment orders")
            both = (self.slot_of_expert >= 0).int() + (cold_slot_of_expert.to(dev) >= 0).int()
            if not bool((both == 1).all()):
                raise ValueError("hot and cold slot maps must partition the experts")
            self.cold_slot_of_expert = cold_slot_of_expert.to(dev)
        self.warm = None
        if warm is not None:
            assert self.cold_slot_of_expert is not None
            # Registers torch.ops.vllm.gguf_v41_original_expert_dp4a.
            from vllm.model_executor.layers.quantization import gguf_v41_moe  # noqa: F401

            iq_slot, host_slot = warm.iq_slot_of_expert.to(dev), warm.host_slot_of_expert.to(dev)
            for name, m in (("iq", iq_slot), ("host", host_slot)):
                if m.shape != (NUM_EXPERTS,) or m.dtype != torch.int32:
                    raise ValueError(f"{name} slot map must be int32 [384]")
            is_warm, is_host = iq_slot >= 0, host_slot >= 0
            not_hot = self.cold_slot_of_expert >= 0
            if bool((is_warm & is_host).any()) or not bool(((is_warm | is_host) == not_hot).all()):
                raise ValueError("WARM and HOST must partition the non-HOT experts")
            if not bool((host_slot[is_host] == self.cold_slot_of_expert[is_host]).all()):
                raise ValueError("a HOST expert's slot is its slot in the host-RAM set")
            n_warm = int(is_warm.sum())
            if n_warm != warm.iq.num_slots or sorted(iq_slot[is_warm].tolist()) != list(range(n_warm)):
                raise ValueError("the IQ slot map does not cover the IQ bank's slots")
            if warm.host_in_decode != bool(is_host.any()):
                raise ValueError("host_in_decode must say whether the layer has a HOST expert")
            if warm.iq.w1.device != dev:
                raise ValueError("the IQ bank and the hot set sit on different devices")
            if warm.prefill is not None and not warm.prefill.serves(warm.iq):
                raise ValueError("the IQ prefill handle was made for another bank")
            self.warm = WarmTier(
                warm.iq, iq_slot, host_slot, warm.host_in_decode, warm.prefill
            )
        s = max_tokens * TOP_K
        # Three-tier: the QPN launches never write a WARM route's slot rows, so
        # those rows keep older values; zeros make the first ones finite and
        # the reduce multiplies them by 0.0.
        alloc = torch.empty if warm is None else torch.zeros
        self.y13 = alloc((s, 2 * TP_INTER), dtype=torch.float16, device=dev)
        # Columns 576..639 stay zero forever: they meet W2's zero padding.
        self.inter = torch.zeros((s, W2_K), dtype=torch.float16, device=dev)
        self.y2 = alloc((s, HIDDEN), dtype=torch.float16, device=dev)
        self.out = torch.empty((max_tokens, HIDDEN), dtype=torch.float32, device=dev)
        self.split13 = grouped_splitk(HIDDEN, experts.order)
        self.split2 = grouped_splitk(W2_K, experts.order)

    @classmethod
    def three_tier(
        cls,
        experts: NativeExpertSet,
        cold: NativeExpertSet,
        iq: IqWarmBank,
        tier: np.ndarray,
        max_tokens: int,
        prefill: IqLayerPrefill | None,
    ) -> NativeV41Moe:
        """The layer of one tier array (int8 [384] of tiers.Tier): ``experts``
        holds the HOT experts, ``cold`` every non-HOT one, ``iq`` the WARM
        ones, each in ascending expert id. All four slot maps come from the
        one array, so they cannot disagree. ``prefill``: see WarmTier."""
        maps = tiers.slot_maps(tier)
        counts = [int((tier == t).sum()) for t in (tiers.Tier.HOT, tiers.Tier.WARM, tiers.Tier.HOST)]
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError("a three-tier layer needs a HOT and a WARM expert")
        if (experts.num_slots, cold.num_slots, iq.num_slots) != (
            counts[0], counts[1] + counts[2], counts[1]
        ):
            raise ValueError(
                f"banks hold {experts.num_slots} hot / {cold.num_slots} host-RAM / "
                f"{iq.num_slots} IQ experts; the tier array has {counts} HOT/WARM/HOST"
            )
        dev = experts.q13.device
        to = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        return cls(
            experts, to(maps.hot), max_tokens, cold=cold, cold_slot_of_expert=to(maps.cold),
            warm=WarmTier(iq, to(maps.iq), to(maps.host), counts[2] > 0, prefill),
        )

    @staticmethod
    def _compact(flat: torch.Tensor):
        # Same device-side compaction as Nvfp4SkinnySm70Experts._apply_grouped.
        n = flat.numel()
        dev = flat.device
        perm = torch.argsort(flat, stable=True).to(torch.int32)
        sorted_e = flat[perm.long()]
        new_group = torch.ones(n, dtype=torch.bool, device=dev)
        new_group[1:] = sorted_e[1:] != sorted_e[:-1]
        gidx = torch.cumsum(new_group, 0) - 1
        goff = torch.full((n + 1,), n, dtype=torch.int64, device=dev)
        goff.scatter_reduce_(
            0, gidx, torch.arange(n, dtype=torch.int64, device=dev), reduce="amin"
        )
        gids = torch.zeros(n, dtype=torch.int64, device=dev)
        gids.scatter_(0, gidx, sorted_e)
        return perm, gids, goff, sorted_e, new_group

    def _groups(self, ids: torch.Tensor):
        flat = self.slot_of_expert[ids.reshape(-1).long()].long()
        perm, gids, goff, _, _ = self._compact(flat)
        return [(self.experts, perm, gids.to(torch.int32), goff.to(torch.int32))]

    def _split_groups(self, ids: torch.Tensor):
        assert self.cold is not None and self.cold_slot_of_expert is not None
        flat_ids = ids.reshape(-1).long()
        hot = self.slot_of_expert[flat_ids].long()
        nh = self.experts.num_slots
        key = torch.where(hot >= 0, hot, nh + self.cold_slot_of_expert[flat_ids].long())
        perm, gids, goff, sorted_e, new_group = self._compact(key)
        n = key.numel()
        count_hot = (key < nh).sum()
        groups_hot = (new_group & (sorted_e < nh)).sum()
        gids_h = torch.where(gids < nh, gids, 0).to(torch.int32)
        goff_h = torch.minimum(goff, count_hot).to(torch.int32)
        idx = (torch.arange(n + 1, device=key.device) + groups_hot).clamp(max=n)
        goff_c = goff[idx].to(torch.int32)
        gids_c = (gids[idx[:n].clamp(max=n - 1)] - nh).clamp(min=0).to(torch.int32)
        return [(self.experts, perm, gids_h, goff_h), (self.cold, perm, gids_c, goff_c)]

    def _tier_groups(self, ids: torch.Tensor):
        """QPN launches of a three-tier layer: the HOT routes over the HBM set
        and, if the layer has a HOST tier, the HOST routes over the host-RAM
        set. One compaction keys HOT slots first, then HOST slots, then one
        sentinel for every WARM route; clamping the group offsets to the count
        of HOT (or of native) routes empties the sentinel group and every
        later one, and the kernel skips empty groups."""
        assert self.warm is not None and self.cold is not None
        flat = ids.reshape(-1).long()
        n = flat.numel()
        nh, nc = self.experts.num_slots, self.cold.num_slots
        hot = self.slot_of_expert[flat].long()
        host = self.warm.host_slot_of_expert[flat].long()
        key = torch.where(hot >= 0, hot, torch.where(host >= 0, nh + host, nh + nc))
        perm, gids, goff, sorted_e, new_group = self._compact(key)
        count_hot = (key < nh).sum()
        gids_h = torch.where(gids < nh, gids, 0).to(torch.int32)
        goff_h = torch.minimum(goff, count_hot).to(torch.int32)
        launches = [(self.experts, perm, gids_h, goff_h)]
        if self.warm.host_in_decode:
            count_native = (key < nh + nc).sum()
            groups_hot = (new_group & (sorted_e < nh)).sum()
            idx = (torch.arange(n + 1, device=key.device) + groups_hot).clamp(max=n)
            goff_c = torch.minimum(goff[idx], count_native).to(torch.int32)
            gids_c = (gids[idx[:n].clamp(max=n - 1)] - nh).clamp(min=0, max=nc - 1).to(torch.int32)
            launches.append((self.cold, perm, gids_c, goff_c))
        return launches

    def _iq_routes(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(IQ slot of each route or -1, router weights with WARM routes 0.0)."""
        assert self.warm is not None
        iq_ids = self.warm.iq_slot_of_expert[topk_ids.long()]
        return iq_ids, torch.where(iq_ids < 0, topk_weights, 0.0)

    def _iq_partial(
        self, x: torch.Tensor, iq_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """Production's IQ op over the WARM routes; it drops ids < 0."""
        assert self.warm is not None
        iq = self.warm.iq
        return torch.ops.vllm.gguf_v41_original_expert_dp4a(
            x, iq_ids, topk_weights, iq.w1, iq.w3, iq.w2, iq.type13, iq.type2, 0, SWIGLU_LIMIT
        )

    def forward(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        """x fp16 [M, 5120] -> this rank's FP32 partial [M, 5120] (a buffer
        view; a new tensor for a three-tier layer)."""
        m = x.size(0)
        if not 0 < m <= self.max_tokens or topk_ids.shape != (m, TOP_K):
            raise ValueError(f"bad routing shape {tuple(topk_ids.shape)} for M={m}")
        if x.dtype != torch.float16 or topk_weights.dtype != torch.float32:
            raise ValueError("x must be fp16 and topk_weights fp32")
        s = m * TOP_K
        if self.warm is not None:
            launches = self._tier_groups(topk_ids)
        elif self.cold is None:
            launches = self._groups(topk_ids)
        else:
            launches = self._split_groups(topk_ids)
        y13, inter, y2, out = self.y13[:s], self.inter[:s], self.y2[:s], self.out[:m]
        # Each launch writes only the slot rows of its own groups.
        for e, perm, gids, goff in launches:
            torch.ops._C.skinny_moe_qpn_sm70(
                x, e.q13, e.s13, e.g1, perm, gids, goff, TOP_K, y13, False, m,
                self.split13, 1, int(e.order),
            )
        g = y13[:, :TP_INTER].float().clamp(max=SWIGLU_LIMIT)
        u = y13[:, TP_INTER:].float().clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        inter[:, :TP_INTER].copy_(torch.nn.functional.silu(g) * u)
        for e, perm, gids, goff in launches:
            torch.ops._C.skinny_moe_qpn_sm70(
                inter, e.q2, e.s2, e.g2, perm, gids, goff, TOP_K, y2, True, m,
                self.split2, 1, int(e.order),
            )
        if self.warm is not None:
            # Both partials are additive: the native one is this rank's K
            # slice of every native route on all rows, the IQ one the full-K
            # result of every WARM route on this rank's rows.
            iq_ids, w_native = self._iq_routes(topk_ids, topk_weights)
            _mtp_weighted_reduce(y2, w_native.contiguous(), out)
            return out + self._iq_partial(x, iq_ids, topk_weights).float()
        _mtp_weighted_reduce(y2, topk_weights.contiguous(), out)
        return out

    # -- prefill (eager, any M) -------------------------------------------
    def _stage_buffers(self, batch: int) -> PrefillStaging:
        """This card's prefill staging (shared by every layer on the card)."""
        c = self.cold
        assert c is not None
        return prefill_staging(c, self.experts.q13.device, batch)

    def forward_prefill(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        batch_experts: int = 32,
    ) -> torch.Tensor:
        """Any-M eager path (prefill chunks; never graph-captured).

        Hot experts run from HBM. The cold experts this chunk routes to are
        staged from host RAM into HBM in batches of ``batch_experts`` by a
        copy stream, double-buffered so batch i + 1 copies while batch i
        computes; each cold expert crosses PCIe once per chunk (W13 bytes in
        the first pass, W2 bytes in the second). Returns a new FP32 tensor.

        With the WARM tier's IQ prefill handle (plan 04 part B) only HOST
        experts are staged: a WARM route gets router weight 0.0 in the native
        reduce and runs from its IQ record in HBM instead, as in decode. The
        four TP ranks then call this in step (the IQ part takes NCCL turns).
        """
        m = x.size(0)
        if topk_ids.shape != (m, TOP_K) or x.dtype != torch.float16:
            raise ValueError("prefill takes fp16 x [M, 5120] and top-6 ids")
        if topk_weights.dtype != torch.float32:
            raise ValueError("topk_weights must be fp32")
        dev, s = x.device, m * TOP_K
        trace = _PREFILL_TRACE
        started = _mark(dev) if trace is not None else 0.0
        staged_experts, seconds_staged, seconds_iq = 0, 0.0, 0.0
        iq_prefill = self.warm.prefill if self.warm is not None else None
        flat = topk_ids.reshape(-1).long()
        nh = self.experts.num_slots
        hot = self.slot_of_expert[flat].long()
        hot_key = torch.where(hot >= 0, hot, nh)
        perm, gids, goff, _, _ = self._compact(hot_key)
        count = (hot_key < nh).sum()
        hot_launch = (perm, torch.where(gids < nh, gids, 0).to(torch.int32),
                      torch.minimum(goff, count).to(torch.int32))
        cold_launches = []
        if self.cold is not None:
            assert self.cold_slot_of_expert is not None
            if iq_prefill is None:
                cold = self.cold_slot_of_expert[flat].long()
            else:  # a HOST expert's slot is its slot in the host-RAM set
                assert self.warm is not None
                cold = self.warm.host_slot_of_expert[flat].long()
            used = torch.unique(cold[cold >= 0]).tolist()  # host sync: eager path
            staged_experts = len(used)
            local = torch.full((self.cold.num_slots,), -1, dtype=torch.int64, device=dev)
            for i in range(0, len(used), batch_experts):
                ids = torch.tensor(used[i : i + batch_experts], dtype=torch.int64, device=dev)
                local.fill_(-1)
                local[ids] = torch.arange(ids.numel(), device=dev)
                lk = torch.where(cold >= 0, local[cold.clamp(min=0)], -1)
                key = torch.where(lk >= 0, lk, batch_experts)
                bperm, bgids, bgoff, _, _ = self._compact(key)
                bcount = (key < batch_experts).sum()
                cold_launches.append((
                    ids, bperm, torch.where(bgids < batch_experts, bgids, 0).to(torch.int32),
                    torch.minimum(bgoff, bcount).to(torch.int32),
                ))
        y13 = torch.empty((s, 2 * TP_INTER), dtype=torch.float16, device=dev)
        inter = torch.zeros((s, W2_K), dtype=torch.float16, device=dev)
        # IQ prefill: no launch writes a WARM route's y2 row; zeros keep it
        # finite and the reduce multiplies it by 0.0. (Its y13 and inter rows
        # are never read: the W2 launches read only their own groups' rows.)
        y2 = (torch.empty if iq_prefill is None else torch.zeros)(
            (s, HIDDEN), dtype=torch.float16, device=dev
        )
        out = torch.empty((m, HIDDEN), dtype=torch.float32, device=dev)
        e = self.experts
        for names, xin, y, slot_major, split in (
            (("q13", "s13", "g1"), x, y13, False, self.split13),
            (("q2", "s2", "g2"), inter, y2, True, self.split2),
        ):
            q, sc, g = names
            torch.ops._C.skinny_moe_qpn_sm70(
                xin, getattr(e, q), getattr(e, sc), getattr(e, g), *hot_launch,
                TOP_K, y, slot_major, m, split, 1, int(e.order),
            )
            if cold_launches:
                mark = _mark(dev) if trace is not None else 0.0
                self._run_cold(cold_launches, names, xin, y, slot_major, split, m, batch_experts)
                if trace is not None:
                    seconds_staged += _mark(dev) - mark
            if names[0] == "q13":
                g13 = y13[:, :TP_INTER].float().clamp(max=SWIGLU_LIMIT)
                u13 = y13[:, TP_INTER:].float().clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
                inter[:, :TP_INTER].copy_(torch.nn.functional.silu(g13) * u13)
                del g13, u13
        if iq_prefill is None:
            _mtp_weighted_reduce(y2, topk_weights.contiguous(), out)
        else:
            # Additive partials, as in decode: the native one is this rank's
            # K slice of every HOT and HOST route on all rows, the IQ one the
            # full-K result of every WARM route on this rank's rows.
            iq_ids, w_native = self._iq_routes(topk_ids, topk_weights)
            _mtp_weighted_reduce(y2, w_native.contiguous(), out)
            del y13, inter, y2  # not held across the IQ part's own workspace
            mark = _mark(dev) if trace is not None else 0.0
            out.add_(iq_prefill.partial(x, iq_ids, topk_weights))
            if trace is not None:
                seconds_iq = _mark(dev) - mark
        if trace is not None:
            trace.add(m, staged_experts, seconds_staged, seconds_iq, _mark(dev) - started)
        return out

    def _run_cold(self, launches, names, xin, y, slot_major, split, m, batch):
        c = self.cold
        assert c is not None
        staging = self._stage_buffers(batch)
        bufs = staging.banks
        compute, copy = torch.cuda.current_stream(), staging.copy_stream
        free = [torch.cuda.Event(), torch.cuda.Event()]
        for ev in free:
            ev.record(compute)
        for i, (ids, perm, gids, goff) in enumerate(launches):
            buf = bufs[i % 2]
            copy.wait_event(free[i % 2])  # the previous batch in this buffer is done
            with torch.cuda.stream(copy):
                for name in names:
                    torch.index_select(getattr(c, name), 0, ids, out=buf[name][: ids.numel()])
                ready = torch.cuda.Event()
                ready.record(copy)
            compute.wait_event(ready)
            q, sc, g = (buf[n] for n in names)
            torch.ops._C.skinny_moe_qpn_sm70(
                xin, q, sc, g, perm, gids, goff, TOP_K, y, slot_major, m, split, 1,
                int(c.order),
            )
            free[i % 2] = torch.cuda.Event()
            free[i % 2].record(compute)
            # Index tensors are used on the copy stream: keep them alive.
            ids.record_stream(copy)


def choose_residency(
    decode_counts: np.ndarray, layers: range, hot_budget_bytes: int
) -> dict[int, tuple[list[int], list[int]]]:
    """Per-layer (hot, cold) expert ids for one pipeline stage's TP rank.

    The stage's least-routed experts by the decode tally go to host RAM until
    the hot experts fit ``hot_budget_bytes`` per card (one TP slice each).
    Ties break by (layer, expert) so the choice is deterministic. Counts stay
    in memory; only ids leave this function.
    """
    if decode_counts.ndim != 2 or decode_counts.shape[1] != NUM_EXPERTS:
        raise ValueError(f"decode tally must be [layers, 384], got {decode_counts.shape}")
    if hot_budget_bytes <= 0:
        raise ValueError("hot budget must be positive")
    total = len(layers) * NUM_EXPERTS
    max_hot = hot_budget_bytes // EXPERT_BYTES
    n_cold = max(0, total - max_hot)
    ranked = sorted(
        (int(decode_counts[layer, e]), layer, e) for layer in layers for e in range(NUM_EXPERTS)
    )
    cold = {(layer, e) for _, layer, e in ranked[:n_cold]}
    return {
        layer: (
            [e for e in range(NUM_EXPERTS) if (layer, e) not in cold],
            [e for e in range(NUM_EXPERTS) if (layer, e) in cold],
        )
        for layer in layers
    }


EXTRA_KEYS = (
    "nvfp4_8r_packs", "nvfp4_8r_tally", "nvfp4_8r_reserve_bytes", tiers.RESIDENCY_KEY,
    tiers.PREFILL_WARM_KEY,
)
# Valid only beside EXTRA_KEYS, and only for the residency that needs them.
CONDITIONAL_KEYS = (tiers.HOST_SHARE_KEY,)
TALLY_PHASE = "teacher"  # the P16 phase that stands for decode routing


def _parse_reserve(raw: object) -> dict[int, int]:
    """{first layer of a stage: positive bytes}; keys may arrive as text."""
    bad = ValueError(
        "nvfp4_8r_reserve_bytes must map each stage's first layer to a positive "
        f"int of bytes, got {raw!r}"
    )
    if not isinstance(raw, dict) or not raw:
        raise bad
    out: dict[int, int] = {}
    for key, value in raw.items():
        if type(key) is str and key.isdecimal():
            key = int(key)
        if type(key) is not int or key < 0 or type(value) is not int or value <= 0:
            raise bad
        if key in out:
            raise bad
        out[key] = value
    return out


@dataclass(frozen=True)
class NativePackConfig:
    """model-loader-extra-config for native experts, parsed once at the loader.

    ``reserve_by_stage`` is the per-card HBM kept free after dense weights
    load (KV cache, CUDA graphs, drafter, prefill staging, transients), one
    value per pipeline stage keyed by the stage's first layer: the stages
    need different head-room (the last stage also holds the drafter and the
    head). The hot expert budget is the measured free HBM minus this, never
    a constant. ``residency`` says which tiers a stage's experts are split
    into (plan 01): two_tier is the hot/cold build, the other two add the
    production IQ record as a WARM tier in HBM. ``prefill_warm`` says which
    record of a WARM expert prefill computes with (plan 04 part B); iq with
    two_tier cannot be constructed.
    """

    packs: Path
    tally: Path
    reserve_by_stage: dict[int, int]
    residency: tiers.Residency
    prefill_warm: tiers.PrefillWarm

    def __post_init__(self) -> None:
        tiers.check_prefill_warm(self.prefill_warm, self.residency)

    def reserve_for(self, layers: range) -> int:
        if layers.start not in self.reserve_by_stage:
            raise ValueError(
                f"nvfp4_8r_reserve_bytes has no entry for the stage starting at "
                f"layer {layers.start} (has {sorted(self.reserve_by_stage)})"
            )
        return self.reserve_by_stage[layers.start]

    @classmethod
    def from_extra(cls, extra: dict) -> NativePackConfig | None:
        present = [k for k in EXTRA_KEYS if k in extra]
        if not present:
            stray = [k for k in CONDITIONAL_KEYS if k in extra]
            if stray:
                raise ValueError(f"{stray} need the nvfp4_8r pack keys {list(EXTRA_KEYS)}")
            return None
        if len(present) != len(EXTRA_KEYS):
            missing = sorted(set(EXTRA_KEYS) - set(present))
            raise ValueError(f"nvfp4_8r native experts need {missing} as well")
        reserve = _parse_reserve(extra["nvfp4_8r_reserve_bytes"])
        packs = Path(extra["nvfp4_8r_packs"])
        if not (packs / "manifest.json").is_file():
            raise ValueError(f"no pack manifest in {packs}")
        read_manifest(packs)
        tally = Path(extra["nvfp4_8r_tally"])
        if not tally.is_file():
            raise ValueError(f"no routing tally at {tally}")
        residency = tiers.parse_residency(extra)
        return cls(packs, tally, reserve, residency,
                   tiers.parse_prefill_warm(extra, residency))

    def decode_counts(self) -> np.ndarray:
        with np.load(self.tally, allow_pickle=False) as data:
            phases = [str(p) for p in data["phases"]]
            return np.asarray(data["counts"][phases.index(TALLY_PHASE)])


def is_routed_expert_name(name: str) -> bool:
    """The V4.1 adapter's own routed-expert split (shared experts excluded)."""
    return ".ffn.experts." in name


@dataclass(frozen=True)
class StagePlan:
    free_bytes: int
    reserve_bytes: int
    hot_budget_bytes: int
    layers: range
    residency: dict[int, tuple[list[int], list[int]]]


def plan_stage(
    cfg: NativePackConfig, layer_ids: list[int], free_bytes: int
) -> StagePlan:
    """Residency for one stage's TP rank from the measured free HBM."""
    ordered = sorted(layer_ids)
    if not ordered or ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError(f"stage MoE layers are not one contiguous run: {ordered}")
    layers = range(ordered[0], ordered[-1] + 1)
    reserve = cfg.reserve_for(layers)
    budget = free_bytes - reserve
    if budget < EXPERT_BYTES:
        raise ValueError(
            f"free HBM {free_bytes} minus reserve {reserve} holds no expert"
        )
    return StagePlan(free_bytes, reserve, budget, layers,
                     choose_residency(cfg.decode_counts(), layers, budget))


def two_tier_array(hot: list[int], cold: list[int]) -> np.ndarray:
    """The tier array of a hot/cold partition: cold experts are HOST."""
    if sorted(hot + cold) != list(range(NUM_EXPERTS)):
        raise ValueError("hot and cold experts must partition 0..383")
    tier = np.full(NUM_EXPERTS, tiers.Tier.HOST, np.int8)
    tier[hot] = tiers.Tier.HOT
    return tier


# Cards whose slot of the IQ op's peer bank this process registered.
_IQ_PEER_CARDS: set[torch.device] = set()


def _require_iq_decode(rank: int, device: torch.device) -> None:
    """What the IQ op needs before a card's row bank is registered."""
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization import gguf_shared_layout
    from vllm.model_executor.layers.quantization import gguf_v41_moe as moe

    if device.type != "cuda" or device.index is None:
        raise ValueError(f"the IQ peer bank needs an indexed CUDA device, got {device}")
    if not 0 <= rank < 4:
        raise ValueError(f"TP rank {rank} is outside the TP4 group")
    if device in _IQ_PEER_CARDS or moe._PEER.get(device) is not None:
        raise ValueError(f"the IQ peer bank of {device} is already registered")
    wrong = [
        name
        for name, ok in (
            ("VLLM_V41_DOWN_ROWS=1", gguf_shared_layout.DOWN_ROWS),
            ("VLLM_V41_EM=1", envs.VLLM_V41_EM == "1"),
            ("VLLM_V41_EXPERT_FAST=1", moe._EXPERT_FAST),
            ("VLLM_V41_PLAN_FUSE=1", moe._PLAN_FUSE),
            ("VLLM_V41_ROWS_TILE=4,8", envs.VLLM_V41_ROWS_TILE == "4,8"),
        )
        if not ok
    ]
    if wrong:
        raise ValueError(f"a WARM tier needs {wrong} in the environment")
    if not moe._expert_major_ready():
        raise ValueError(f"a WARM tier needs the v41k library at {moe.default_v41k_library()}")


def attach_iq_peer(cpu_group, rank: int, device: torch.device) -> None:
    """Map and register this card's slot of the CUDA-IPC row bank that the IQ
    op's peer gather reads (gate_up_fast writes a rank's intermediate rows
    there). Production registers it in SharedOriginalPrefill._init_rows, which
    the native-staged build never constructs. A collective: the stage's four
    TP ranks call it together, once per card, before the first IQ op."""
    import torch.distributed as dist

    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.model_executor.layers.quantization import gguf_v41_moe as moe
    from vllm.model_executor.layers.quantization.gguf_shared_prefill import _DeviceView

    _require_iq_decode(rank, device)
    pointers = CustomAllreduce.create_shared_buffer(moe.ROWS_BYTES, cpu_group)
    own = torch.as_tensor(_DeviceView(pointers[rank], (moe.ROWS_BYTES,), "|u1"), device=device)
    moe.register_rows_peer(device, rank, pointers, own)
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)  # every rank's flags are zero before any gather
    if moe._PEER.get(device) is None:
        raise RuntimeError(f"the IQ peer bank of {device} did not register")
    _IQ_PEER_CARDS.add(device)


def warm_prefill_bytes(
    cfg: NativePackConfig, sources: dict[int, IqLayerSource], prefill_tokens: int
) -> int:
    """HBM a card keeps for WARM prefill beyond the records: nothing when
    WARM experts are staged from host RAM, production's IQ prefill workspace
    (two row banks and a canonical tile) when they run from IQ records."""
    if cfg.prefill_warm is tiers.PrefillWarm.NATIVE_STAGED:
        return 0
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_warm_prefill

    return nvfp4_8r_warm_prefill.iq_prefill_bytes(
        [(s.type13, s.type2) for s in sources.values()], prefill_tokens
    )


def attach_iq_prefill(
    plan: tiers.TierPlan,
    sources: dict[int, IqLayerSource],
    rank: int,
    device: torch.device,
    group,
    prefill_tokens: int,
) -> IqStagePrefill:
    """Production's IQ prefill over the stage's WARM tier on this card (plan
    04 part B). It registers the decode row bank itself, so it stands in for
    attach_iq_peer. A collective of the stage's four TP ranks; ``group`` is
    their TP group (its cpu_group and its NCCL communicator)."""
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_warm_prefill

    _require_iq_decode(rank, device)
    shapes = [
        nvfp4_8r_warm_prefill.WarmLayerShape(
            layer, int((plan.tiers[layer] == tiers.Tier.WARM).sum()),
            sources[layer].type13, sources[layer].type2,
        )
        for layer in plan.layers
    ]
    stage = nvfp4_8r_warm_prefill.IqStagePrefill(shapes, rank, device, group, prefill_tokens)
    _IQ_PEER_CARDS.add(device)
    return stage


def build_layer(
    cfg: NativePackConfig,
    layer: int,
    rank: int,
    tier: np.ndarray,
    iq_source: IqLayerSource | None,
    device: torch.device,
    max_tokens: int,
    stage_prefill: IqStagePrefill | None,
) -> NativeV41Moe:
    """One layer's TP slice from its tier array (int8 [384] of tiers.Tier).

    HOT experts go to HBM and every other one to registered host RAM (the
    native record, for prefill and for HOST decode routes). A layer with WARM
    experts also gets their IQ records in HBM from ``iq_source`` and is the
    three-tier object; a layer without them is the hot/cold object.
    ``stage_prefill`` is the card's IQ prefill exactly when the config runs
    WARM prefill on IQ records and the layer has WARM experts.
    """
    ids = np.arange(NUM_EXPERTS)
    maps = tiers.slot_maps(tier)  # validates the array
    hot = ids[tier == tiers.Tier.HOT].tolist()
    cold = ids[tier != tiers.Tier.HOT].tolist()
    warm = ids[tier == tiers.Tier.WARM].tolist()
    if not hot:
        raise ValueError(f"L{layer}: no HOT expert; the budget cannot hold this stage")
    if bool(warm) != (iq_source is not None):
        raise ValueError(f"L{layer}: {len(warm)} WARM experts but IQ source {iq_source!r}")
    wants_iq_prefill = bool(warm) and cfg.prefill_warm is tiers.PrefillWarm.IQ
    if wants_iq_prefill != (stage_prefill is not None):
        raise ValueError(
            f"L{layer}: {len(warm)} WARM experts under {cfg.prefill_warm}, "
            f"IQ prefill {'given' if stage_prefill is not None else 'absent'}"
        )
    raw = torch.from_numpy(read_rank_experts(cfg.packs, layer, rank))
    # Same order of device allocations as the hot/cold build before plan 01.
    hot_slot = torch.from_numpy(maps.hot).to(device)
    cold_slot = torch.from_numpy(maps.cold).to(device)
    hot_set = NativeExpertSet.from_raw(raw[hot], device)
    if not cold:
        return NativeV41Moe(hot_set, hot_slot, max_tokens)
    cold_set = NativeExpertSet.from_raw_host(raw[cold], device)
    if iq_source is None:
        return NativeV41Moe(hot_set, hot_slot, max_tokens, cold=cold_set,
                            cold_slot_of_expert=cold_slot)
    if device not in _IQ_PEER_CARDS:
        raise ValueError(f"L{layer}: attach_iq_peer has not run for {device}")
    bank = IqWarmBank.from_source(iq_source, warm, device)
    handle = None if stage_prefill is None else stage_prefill.layer(layer, bank)
    return NativeV41Moe.three_tier(hot_set, cold_set, bank, tier, max_tokens, handle)


# -- three-tier stage (plan 01): what the loader hands over ------------------
_IQ_PROJECTIONS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def routed_expert_shards(name_map: dict[str, str], layer_of) -> dict[int, dict[str, str]]:
    """{layer: {w1|w3|w2: raw GGUF tensor name}} of a stage's weight map,
    read before the loader drops the routed experts. ``layer_of`` maps the
    expert module's name (``model.layers.N.ffn.experts``) to its layer id."""
    found: dict[int, dict[str, str]] = {}
    for raw, name in name_map.items():
        if not is_routed_expert_name(name):
            continue
        prefix, projection, leaf = name.rsplit(".", 2)
        if leaf != "weight" or projection not in _IQ_PROJECTIONS:
            raise ValueError(f"unexpected routed-expert tensor {name}")
        shards = found.setdefault(int(layer_of(prefix)), {})
        role = _IQ_PROJECTIONS[projection]
        if role in shards:
            raise ValueError(f"two {projection} tensors for {prefix}")
        shards[role] = raw
    for layer, shards in found.items():
        if sorted(shards) != ["w1", "w2", "w3"]:
            raise ValueError(f"layer {layer} has routed-expert shards {sorted(shards)} only")
    return found


def iq_layer_sources(
    tensors, names: dict[int, dict[str, str]], rank: int
) -> dict[int, IqLayerSource]:
    """This TP rank's IQ sources per layer from the GGUF tensor index."""
    return {
        layer: IqLayerSource(*(
            IqShardSource.from_gguf_tensor(tensors[shards[role]], role, rank)
            for role in ("w1", "w3", "w2")
        ))
        for layer, shards in names.items()
    }


def stage_min_free(cpu_group, free_bytes: int) -> int:
    """The least free HBM over the stage's TP ranks (a collective)."""
    import torch.distributed as dist

    frees: list = [None] * dist.get_world_size(group=cpu_group)
    dist.all_gather_object(frees, int(free_bytes), group=cpu_group)
    return min(frees)


@dataclass(frozen=True)
class TierStagePlan:
    free_bytes: int  # the least free HBM over the stage's four ranks
    reserve_bytes: int
    plan: tiers.TierPlan
    # Share of the stage's decode-tally routes in HOT, WARM, HOST.
    route_share: tuple[float, float, float]
    prefill_bytes: int  # warm_prefill_bytes, taken out of the record budget


def plan_tiers(
    cfg: NativePackConfig,
    layer_ids: list[int],
    free_bytes: int,
    iq_bytes: dict[int, int],
    prefill_bytes: int,
) -> TierStagePlan:
    """Tiers of one stage from the least free HBM over its ranks. The record
    budget is that minus the stage's reserve minus ``prefill_bytes`` (the
    config's warm_prefill_bytes)."""
    if isinstance(cfg.residency, tiers.TwoTier):
        raise ValueError("two_tier has no tier plan; use plan_stage")
    if type(prefill_bytes) is not int or prefill_bytes < 0 or (
        (prefill_bytes > 0) != (cfg.prefill_warm is tiers.PrefillWarm.IQ)
    ):
        raise ValueError(f"WARM prefill bytes {prefill_bytes!r} under {cfg.prefill_warm}")
    ordered = sorted(layer_ids)
    if not ordered or ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError(f"stage MoE layers are not one contiguous run: {ordered}")
    layers = range(ordered[0], ordered[-1] + 1)
    reserve = cfg.reserve_for(layers)
    counts = cfg.decode_counts()
    plan = tiers.choose_tiers(
        counts, layers, iq_bytes, free_bytes - reserve - prefill_bytes, cfg.residency
    )
    total = int(counts[layers.start : layers.stop].sum())
    if total <= 0:
        raise ValueError(f"the decode tally has no routes in layers {layers}")
    share = tuple(
        sum(int(counts[layer][plan.tiers[layer] == t].sum()) for layer in layers) / total
        for t in (tiers.Tier.HOT, tiers.Tier.WARM, tiers.Tier.HOST)
    )
    return TierStagePlan(free_bytes, reserve, plan, share, prefill_bytes)  # type: ignore[arg-type]


def require_one_digest(digests: list[str]) -> None:
    """Refuse a stage whose four TP ranks did not choose the same tiers: the
    IQ op's peer gather would read rows of another route set, silently."""
    if len(digests) != 4 or len(set(digests)) != 1:
        raise ValueError(
            f"the stage's TP ranks chose different tiers: {len(set(digests))} "
            f"distinct digests from {len(digests)} ranks"
        )


def build_stage(
    cfg: NativePackConfig,
    rank: int,
    plan: tiers.TierPlan,
    sources: dict[int, IqLayerSource],
    device: torch.device,
    group,
    max_tokens: int,
    prefill_tokens: int,
) -> dict[int, NativeV41Moe]:
    """The three-tier layers of one stage on one TP rank. A collective: the
    four ranks call it together (digest check, then the peer bank or, when
    WARM prefill runs on IQ records, production's IQ prefill). ``group`` is
    the stage's TP group; ``prefill_tokens`` the prefill route group
    (expert_cache_prefill_tokens), which only the IQ prefill uses."""
    import torch.distributed as dist

    cpu_group = group.cpu_group
    if sorted(sources) != list(plan.layers):
        raise ValueError(f"IQ sources cover {sorted(sources)}, the plan {plan.layers}")
    for layer in plan.layers:
        if sources[layer].record_bytes != plan.iq_bytes[layer]:
            raise ValueError(f"L{layer}: the plan was sized for another IQ record")
    digests: list = [None] * dist.get_world_size(group=cpu_group)
    dist.all_gather_object(digests, plan.digest, group=cpu_group)
    require_one_digest(digests)
    if _LOAD_TRACE:
        _load_trace(device, rank, "enter", None)
    if cfg.prefill_warm is tiers.PrefillWarm.IQ:
        stage_prefill = attach_iq_prefill(plan, sources, rank, device, group, prefill_tokens)
    else:
        attach_iq_peer(cpu_group, rank, device)
        stage_prefill = None
    if _LOAD_TRACE:
        _load_trace(device, rank, "attached", None)
    built: dict[int, NativeV41Moe] = {}
    for layer in plan.layers:
        built[layer] = build_layer(
            cfg, layer, rank, plan.tiers[layer], sources[layer], device, max_tokens,
            stage_prefill,
        )
        if _LOAD_TRACE:
            _load_trace(device, rank, f"L{layer}", plan.tiers[layer])
    return built
