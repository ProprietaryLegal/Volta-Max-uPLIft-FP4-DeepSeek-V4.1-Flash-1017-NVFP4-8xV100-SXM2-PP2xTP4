# SPDX-License-Identifier: Apache-2.0
"""NVFP4-8R plan 04 part B: prefill of the WARM tier on its IQ records.

Production's IQ prefill (``SharedOriginalPrefill`` with
VLLM_V41_PREFILL_IQ_GEMM=1: transcode of a tile of experts into canonical
banks, the v41prefmoe grouped GEMM, the row-split activation, the accumulate
kernel) serves a "cache" of a layer's experts. Here it is handed a view that
holds only the WARM experts of each layer of one pipeline stage: all
resident, slot = position in the layer's ``IqWarmBank``. A route to a HOT or
HOST expert arrives as id -1; it sorts in front of expert 0, no kernel
bound covers it, and it adds nothing (plan 04 B1 falsifier, run on the
deployed kernels).

One ``IqStagePrefill`` per card owns production's object and its HBM: two
parity banks of intermediate rows that all four TP ranks write (CUDA IPC,
ordered by an NCCL turn per tile and route group), one canonical tile
workspace, and the decode row bank that ``attach_iq_peer`` registers in the
native-staged build. Building it is a collective of the stage's four ranks.
A layer's handle is given out only against that layer's IQ bank, so a
handle without records does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from . import nvfp4_8r_tiers as tiers

HIDDEN = 5120
INTERMEDIATE = 2304
TOP_K = 6
_TILE_EXPERTS = 32  # SharedOriginalPrefill's tile


@dataclass(frozen=True)
class WarmLayerShape:
    """What the IQ prefill must know of a layer before its records exist."""

    layer: int  # model layer id
    experts: int  # WARM experts = slots of the layer's IQ bank
    type13: int  # GGML type of gate and up
    type2: int  # GGML type of down

    def __post_init__(self) -> None:
        if self.layer < 0 or not 0 < self.experts <= tiers.NUM_EXPERTS:
            raise ValueError(f"L{self.layer}: {self.experts} WARM experts")
        for kind in (self.type13, self.type2):
            if kind not in tiers.IQ_BLOCK_BYTES:
                raise ValueError(f"L{self.layer}: GGML type {kind!r} is not an IQ record")


def iq_prefill_bytes(kinds: list[tuple[int, int]], prefill_tokens: int) -> int:
    """HBM bytes an ``IqStagePrefill`` keeps for good on its card beyond the
    decode row bank (which the native-staged build holds too), from the
    (gate|up, down) GGML types of the stage's layers: the two parity row
    banks, [prefill_tokens x 6 routes, 2304] FP16 each, and one canonical
    workspace of a 32-expert tile in the stage's largest gate|up and down
    formats (an upper bound when every layer has fewer than 32 WARM experts)."""
    from vllm.model_executor.layers.quantization.gguf_expert_cache import _canonical

    if not kinds or type(prefill_tokens) is not int or prefill_tokens < 1:
        raise ValueError("IQ prefill bytes need a layer and a positive route group")
    rows = 2 * prefill_tokens * TOP_K * INTERMEDIATE * 2
    gu = max(sum(_canonical(t13, INTERMEDIATE // 2, HIDDEN)[3:]) for t13, _ in kinds)
    dn = max(sum(_canonical(t2, HIDDEN // 4, INTERMEDIATE)[3:]) for _, t2 in kinds)
    return rows + _TILE_EXPERTS * (gu + dn)


class _AllResident:
    all_resident = True


@dataclass(frozen=True)
class _Runtime:
    table: torch.Tensor  # int32 [layers, max W]: slot of an expert, identity
    cold_offsets: torch.Tensor  # int64, indexed and unused when all resident
    banks: dict[tuple[int, str], torch.Tensor]  # (layer index, w1|w3|w2) -> bytes


class _CacheView:
    """Exactly what ``SharedOriginalPrefill`` reads of its cache when every
    expert is resident and the down projection is row-split (B1 list)."""

    def __init__(self, specs, layers, rank, device, group, prefill_tokens, runtime):
        self.specs = specs
        self.experts = max(spec.experts for spec in specs)
        self.layers = layers
        self.rank = rank
        self.device = device
        self.group = group
        self.prefill_tokens = prefill_tokens
        self.budget = _AllResident()
        self.runtime = runtime

    def _barrier(self) -> None:
        dist.barrier(group=self.group.cpu_group)


# Cards whose IQ prefill this process built.
_STAGES: dict[torch.device, "IqStagePrefill"] = {}


class IqStagePrefill:
    """Production's IQ prefill over the WARM tier of one stage on one card.

    ``group`` is the stage's TP4 group: ``cpu_group`` for the collectives of
    construction and ``device_communicator.pynccl_comm.all_reduce`` as the
    turn that orders the ranks' row writes. The four ranks build it together,
    with the same shapes (the tier digest check comes first).
    """

    def __init__(
        self,
        shapes: list[WarmLayerShape],
        rank: int,
        device: torch.device,
        group,
        prefill_tokens: int,
    ) -> None:
        from vllm.model_executor.layers.quantization import (
            gguf_shared_layout,
            gguf_shared_prefill,
        )
        from vllm.model_executor.layers.quantization import gguf_v41_moe as moe
        from vllm.model_executor.layers.quantization.gguf_shared_runtime import (
            SharedExpertLayerSpec,
            resident_tables,
        )

        if device.type != "cuda" or device.index is None:
            raise ValueError(f"the IQ prefill needs an indexed CUDA device, got {device}")
        if not 0 <= rank < 4:
            raise ValueError(f"TP rank {rank} is outside the TP4 group")
        if not shapes or len({s.layer for s in shapes}) != len(shapes):
            raise ValueError("the IQ prefill needs the stage's layers, each once")
        if type(prefill_tokens) is not int or prefill_tokens < 1:
            raise ValueError(f"prefill route group of {prefill_tokens!r} tokens")
        wrong = [
            name
            for name, ok in (
                ("VLLM_V41_DOWN_ROWS=1", gguf_shared_layout.DOWN_ROWS),
                ("VLLM_V41_PREFILL_IQ_GEMM=1", gguf_shared_prefill._PREFILL_IQ),
            )
            if not ok
        ]
        if wrong:
            raise ValueError(f"WARM prefill on IQ records needs {wrong} in the environment")
        if device in _STAGES or moe._PEER.get(device) is not None:
            raise ValueError(f"the IQ row banks of {device} are already registered")
        specs = tuple(
            SharedExpertLayerSpec(
                HIDDEN, INTERMEDIATE, s.experts, s.experts, (s.type13, s.type13, s.type2)
            )
            for s in shapes
        )
        table, cold = resident_tables(specs, device)
        self.shapes = tuple(shapes)
        self.device = device
        self.persistent_bytes = iq_prefill_bytes(
            [(s.type13, s.type2) for s in shapes], prefill_tokens
        )
        self._index = {s.layer: i for i, s in enumerate(shapes)}
        self._banks: dict[tuple[int, str], torch.Tensor] = {}
        view = _CacheView(
            specs, [s.layer for s in shapes], rank, device, group, prefill_tokens,
            _Runtime(table, cold, self._banks),
        )
        # A collective: three CUDA-IPC buffers and a barrier, on all four ranks.
        self._tm = gguf_shared_prefill.SharedOriginalPrefill(view, _TILE_EXPERTS)
        if moe._PEER.get(device) is None:
            raise RuntimeError(f"the IQ decode row bank of {device} did not register")
        _STAGES[device] = self

    def layer(self, layer: int, bank) -> IqLayerPrefill:
        """The handle of ``layer``, against its WARM records (an IqWarmBank
        on this card, slot = rank of the expert among the layer's WARM ids)."""
        if layer not in self._index:
            raise ValueError(f"L{layer} is not a layer of this stage's IQ prefill")
        index = self._index[layer]
        shape = self.shapes[index]
        if (index, "w1") in self._banks:
            raise ValueError(f"L{layer}: the IQ prefill already has this layer's records")
        if (bank.num_slots, bank.type13, bank.type2) != (
            shape.experts, shape.type13, shape.type2
        ) or bank.w1.device != self.device:
            raise ValueError(
                f"L{layer}: IQ bank of {bank.num_slots} experts, types "
                f"{bank.type13}/{bank.type2} on {bank.w1.device}; the prefill was "
                f"built for {shape} on {self.device}"
            )
        for shard in ("w1", "w3", "w2"):
            self._banks[index, shard] = getattr(bank, shard)
        return IqLayerPrefill(self, index)


@dataclass(frozen=True)
class IqLayerPrefill:
    """One layer's WARM prefill on its IQ records."""

    stage: IqStagePrefill
    index: int

    def serves(self, bank) -> bool:
        """Whether this handle was given out against ``bank``."""
        return self.stage._banks.get((self.index, "w1")) is bank.w1

    def partial(
        self, x: torch.Tensor, iq_ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """x fp16 [M, 5120], IQ slot of each route or -1 [M, 6], router
        weights fp32 [M, 6] -> FP16 [M, 5120]: the full-K sum of the WARM
        routes on this rank's 1,280 output rows, zeros elsewhere. The four
        ranks call it in step (it takes an NCCL turn per tile and group)."""
        return self.stage._tm.forward(self.index, x, iq_ids, weights)
