"""CPU-only, exact NVFP4 -> MXFP4 TP4 pack conversion for the isolated lane.

No rounding is admitted: every used code must have an exact representation.
The routing tally stays outside packs; only hot/cold expert IDs are emitted.
"""

from __future__ import annotations

import os
import argparse
import json
import multiprocessing
import struct
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
ROOT = Path(os.environ["LANE"])
EXPERT_BYTES = 4_874_240
MAG = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


@dataclass(frozen=True)
class Nvfp4Projection:
    codes: U8
    block_scales: F32
    global_scale: np.float32

    def __post_init__(self) -> None:
        if self.codes.dtype != np.uint8 or self.codes.ndim != 2:
            raise ValueError("codes must be a uint8 matrix")
        rows, half_k = self.codes.shape
        if self.block_scales.shape != (rows, half_k // 8):
            raise ValueError("NVFP4 requires one scale per 16 values")
        if not np.isfinite(self.block_scales).all():
            raise ValueError("nonfinite block scale")
        if (self.block_scales < 0).any():
            raise ValueError("negative block scales are unsupported")
        if not np.isfinite(self.global_scale) or self.global_scale <= 0:
            raise ValueError("global scale must be finite and positive")


class Source:
    def __init__(self, root: Path, verification_log: Path):
        if not root.is_relative_to(ROOT / "models"):
            raise ValueError("converter requires the isolated verified NVMe checkpoint")
        lines = verification_log.read_text().splitlines()
        if "ALL-VERIFIED 84" not in lines:
            raise RuntimeError("INCONCLUSIVE: checkpoint copy is not ALL-VERIFIED")
        self.root = root
        self.index: dict[str, str] = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.headers: dict[str, tuple[int, dict[str, Any]]] = {}

    def tensor(self, name: str) -> tuple[U8, str, tuple[int, ...]]:
        shard = self.index[name]
        path = self.root / shard
        if shard not in self.headers:
            with path.open("rb") as source:
                length = struct.unpack("<Q", source.read(8))[0]
                self.headers[shard] = (length + 8, json.loads(source.read(length)))
        base, header = self.headers[shard]
        info = header[name]
        begin, end = info["data_offsets"]
        return (
            np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                offset=base + begin,
                shape=(end - begin,),
            ),
            str(info["dtype"]),
            tuple(info["shape"]),
        )

    def projection(self, layer: int, expert: int, projection: str) -> Nvfp4Projection:
        prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
        code, code_type, shape = self.tensor(prefix + ".weight")
        block, block_type, block_shape = self.tensor(prefix + ".weight_scale")
        glob, glob_type, _ = self.tensor(prefix + ".weight_scale_2")
        rows, cols = (5120, 2304) if projection == "w2" else (2304, 5120)
        if code_type not in ("U8", "I8") or shape != (rows, cols // 2):
            raise ValueError(f"invalid codes: {prefix} {code_type} {shape}")
        if block_type != "F8_E4M3" or block_shape != (rows, cols // 16):
            raise ValueError(
                f"invalid block scale: {prefix} {block_type} {block_shape}"
            )
        if glob_type != "F32" or glob.size != 4:
            raise ValueError(f"invalid global scale: {prefix}")
        # Build the E4M3FN lookup once per projection with scalar arithmetic.
        table = np.array(
            [
                (
                    (b & 7) / 512
                    if (b & 120) == 0
                    else (1 + (b & 7) / 8) * 2.0 ** (((b >> 3) & 15) - 7)
                )
                * (-1 if b & 128 else 1)
                for b in range(256)
            ],
            dtype=np.float32,
        )
        if np.any((block & 127) == 127):
            raise ValueError(f"NaN E4M3FN code: {prefix}")
        return Nvfp4Projection(
            code.reshape(shape), table[block].reshape(block_shape), glob.view("<f4")[0]
        )


def fold(source: Nvfp4Projection) -> tuple[U8, U8]:
    """Recover an exact E8M0/32 representation or raise at the first loss."""
    rows, half_k = source.codes.shape
    # One maximum per 16 values, before choosing a common exponent for 32.
    byte_max = np.maximum(MAG[np.arange(256) & 7], MAG[(np.arange(256) >> 4) & 7])
    max16 = byte_max[source.codes].reshape(rows, -1, 8).max(axis=2)
    max16 = (max16 * source.block_scales) * source.global_scale
    max32 = max16.reshape(rows, -1, 2).max(axis=2)
    nonzero = max32 > 0
    exponent = np.zeros(max32.shape, np.int32)
    exponent[nonzero] = np.ceil(np.log2(max32[nonzero].astype(np.float64) / 6)).astype(
        np.int32
    )
    if np.any((exponent < -127) | (exponent > 127)):
        raise ValueError("exact E8M0 exponent is outside the finite range")
    # Per-block nibble maps are much smaller than dense projection weights.
    exponents16 = np.repeat(exponent, 2, axis=1)
    value = (MAG[None, None, :] * source.block_scales[..., None]) * source.global_scale
    target = np.ldexp(value, -exponents16[..., None])
    mapped = np.searchsorted(MAG, target).clip(0, 7).astype(np.uint8)
    restored = np.ldexp(MAG[mapped], exponents16[..., None])
    exact = restored == value
    code = source.codes.reshape(rows, -1, 8)
    low, high = code & 7, (code >> 4) & 7
    if not (
        np.take_along_axis(exact, low, axis=2).all()
        and np.take_along_axis(exact, high, axis=2).all()
    ):
        raise ValueError("NVFP4 block cannot fold exactly to MXFP4 E8M0/32")
    lo = np.take_along_axis(mapped, low, axis=2) | (code & 8)
    hi = np.take_along_axis(mapped, high, axis=2) | ((code >> 4) & 8)
    return (lo | (hi << 4)).reshape(rows, half_k), (exponent + 127).astype(np.uint8)


def partition(tally: Path) -> list[list[dict[str, list[int]]]]:
    """Choose the globally coldest experts; serialize IDs, never counts."""
    with np.load(tally, allow_pickle=False) as data:
        phases = list(data["phases"])
        counts = data["counts"]
        if counts.shape != (len(phases), 40, 384):
            raise ValueError(f"invalid P16 tally geometry: {counts.shape}")
        decode = counts[phases.index("teacher")]
        if not np.issubdtype(decode.dtype, np.integer) or np.any(decode < 0):
            raise ValueError("P16 decode tally must contain nonnegative integer counts")
        # Each expert occupies four rank slices. A single ownership split
        # applies to all ranks, including both pipeline stages.
        number_cold = round(80 * 2**30 / (4 * EXPERT_BYTES))
        ordered = sorted(
            (int(decode[layer, expert]), layer, expert)
            for layer in range(40)
            for expert in range(384)
        )
        cold = {(layer, expert) for _, layer, expert in ordered[:number_cold]}
    return [
        [
            {
                "hot": [e for e in range(384) if (layer, e) not in cold],
                "cold": [e for e in range(384) if (layer, e) in cold],
            }
            for _ in range(4)
        ]
        for layer in range(40)
    ]


@dataclass(frozen=True)
class WorkerCount:
    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("worker count must be positive")

    @classmethod
    def parse(cls, text: str) -> WorkerCount:
        return cls(int(text))


@dataclass(frozen=True)
class LayerWork:
    layer: int
    weights: Path
    verification_log: Path
    output: Path
    rank_splits: list[dict[str, list[int]]]

    def __post_init__(self) -> None:
        if not 0 <= self.layer < 40 or len(self.rank_splits) != 4:
            raise ValueError("one layer task must own exactly four TP rank slices")


@dataclass(frozen=True)
class LayerResult:
    layer: int
    seconds: float


def build_layer(work: LayerWork) -> LayerResult:
    source = Source(work.weights, work.verification_log)
    layer, output, splits = work.layer, work.output, work.rank_splits
    started = time.monotonic()
    # A single worker owns every output for this layer.
    handles = {
        (rank, temp): (output / f"L{layer:02d}-r{rank}-{temp}.bin.part").open("xb")
        for rank in range(4)
        for temp in ("hot", "cold")
    }
    try:
        for expert in range(384):
            folded = {
                proj: fold(source.projection(layer, expert, proj))
                for proj in ("w1", "w3", "w2")
            }
            for rank in range(4):
                temp = "hot" if expert in splits[rank]["hot"] else "cold"
                destination = handles[rank, temp]
                begin, end = rank * 576, (rank + 1) * 576
                for proj in ("w1", "w3"):
                    codes, scales = folded[proj]
                    destination.write(codes[begin:end].tobytes())
                    destination.write(scales[begin:end].tobytes())
                codes, scales = folded["w2"]
                padded_codes = np.zeros((5120, 320), np.uint8)
                padded_scales = np.full((5120, 20), 127, np.uint8)
                padded_codes[:, :288] = codes[:, begin // 2 : end // 2]
                padded_scales[:, :18] = scales[:, begin // 32 : end // 32]
                destination.write(padded_codes.tobytes())
                destination.write(padded_scales.tobytes())
    finally:
        for handle in handles.values():
            handle.close()
    for rank, temp in handles:
        path = output / f"L{layer:02d}-r{rank}-{temp}.bin.part"
        expected = len(splits[rank][temp]) * EXPERT_BYTES
        if path.stat().st_size != expected:
            raise ValueError(f"incomplete layer output {path}")
        path.rename(path.with_suffix(""))
    return LayerResult(layer, time.monotonic() - started)


def layer_worker(work: LayerWork) -> None:
    completed = build_layer(work)
    print(
        json.dumps(
            {
                "layer": completed.layer,
                "experts": 384,
                "bytes": 384 * 4 * EXPERT_BYTES,
                "seconds": completed.seconds,
            }
        ),
        flush=True,
    )


def build(
    weights: Path, log: Path, tally: Path, output: Path, workers: WorkerCount
) -> None:
    weights = weights.resolve()
    Source(weights, log)
    splits = partition(tally)
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("pack output must stay inside the isolated lane")
    output.mkdir(parents=True, exist_ok=False)
    queued = deque(
        LayerWork(layer, weights, log, output, splits[layer]) for layer in range(40)
    )
    context = multiprocessing.get_context("spawn")
    # No worker pool: pools may silently replace crashed processes. Each layer
    # transitions queued -> running -> completed or failed exactly once.
    active: dict[int, tuple[multiprocessing.process.BaseProcess, LayerWork]] = {}
    try:
        while queued or active:
            while queued and len(active) < workers.value:
                work = queued.popleft()
                new_process = context.Process(target=layer_worker, args=(work,))
                new_process.start()
                active[new_process.sentinel] = new_process, work
            for sentinel in wait(list(active)):
                if not isinstance(sentinel, int):
                    raise TypeError("process wait returned a non-sentinel handle")
                process, work = active[sentinel]
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"Layer {work.layer} worker failed: exit={process.exitcode}"
                    )
                del active[sentinel]
                process.close()
    finally:
        # These process objects are owned solely by this converter. On failure,
        # terminate remaining workers and preserve their unadmitted .part files.
        for process, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
    (output / "manifest.json").write_text(
        json.dumps({"format": "nvfp4-8r-mxfp4-v1", "layers": splits}) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    parser.add_argument("--tally", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=WorkerCount.parse, required=True)
    args = parser.parse_args()
    build(args.weights, args.verification_log, args.tally, args.output, args.workers)
