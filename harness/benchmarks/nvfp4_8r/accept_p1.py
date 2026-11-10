"""P1 acceptance: compare persisted packs to the raw NVFP4 formula on CPU.

Run before implementing the converter; a missing manifest is a failing gate.
This oracle never calls the converter or the engine's dequantization routines.
"""

from __future__ import annotations

import os
import argparse
import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
GEOMETRY = {"w1": (2304, 5120), "w3": (2304, 5120), "w2": (5120, 2304)}


class Checkpoint:
    """Read-only safetensors boundary; no torch import or CUDA initialization."""

    def __init__(self, root: Path, verification_log: Path):
        if "ALL-VERIFIED 84" not in verification_log.read_text().splitlines():
            raise RuntimeError("INCONCLUSIVE: checkpoint copy is not ALL-VERIFIED")
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.index: dict[str, str] = index["weight_map"]
        self.headers: dict[str, tuple[int, dict[str, Any]]] = {}

    def tensor(self, name: str) -> tuple[U8, str]:
        shard = self.index[name]
        path = self.root / shard
        if shard not in self.headers:
            with path.open("rb") as source:
                size = struct.unpack("<Q", source.read(8))[0]
                self.headers[shard] = (8 + size, json.loads(source.read(size)))
        offset, header = self.headers[shard]
        info = header[name]
        begin, end = info["data_offsets"]
        raw = np.memmap(
            path, mode="r", dtype=np.uint8, offset=offset + begin, shape=(end - begin,)
        )
        return raw, str(info["dtype"])

    def projection(self, layer: int, expert: int, projection: str) -> F32:
        prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
        rows, cols = GEOMETRY[projection]
        codes, code_type = self.tensor(prefix + ".weight")
        scales, scale_type = self.tensor(prefix + ".weight_scale")
        global_bytes, global_type = self.tensor(prefix + ".weight_scale_2")
        if code_type not in ("U8", "I8"):
            raise ValueError(f"unexpected code dtype {code_type}")
        if scale_type != "F8_E4M3" or global_type != "F32":
            raise ValueError(f"unexpected scale types {scale_type}/{global_type}")
        if codes.size != rows * cols // 2 or scales.size != rows * cols // 16:
            raise ValueError(f"invalid projection geometry: {prefix}")
        if global_bytes.size != 4:
            raise ValueError(f"global scale must be scalar: {prefix}")
        global_scale = global_bytes.view("<f4")[0]
        # E4M3FN: sign bit, bias 7, three fraction bits, subnormal 2^-9.
        exp = ((scales >> 3) & 15).astype(np.int32)
        mantissa = (scales & 7).astype(np.float32)
        block = np.where(
            exp == 0, mantissa * np.float32(2**-9), np.ldexp(1 + mantissa / 8, exp - 7)
        )
        block = np.where(scales & 128, -block, block).astype(np.float32)
        if np.any((scales & 127) == 127) or not np.isfinite(global_scale):
            raise ValueError(f"nonfinite NVFP4 scale: {prefix}")
        nib = np.empty(codes.size * 2, dtype=np.uint8)
        nib[0::2], nib[1::2] = codes & 15, codes >> 4
        magnitudes = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
        values = magnitudes[nib & 7]
        values = np.where(nib & 8, -values, values)
        # Preserve the formula's FP32 multiplication order.
        values = values.reshape(-1, 16) * block[:, None]
        return np.asarray(values * global_scale, dtype=np.float32).reshape(rows, cols)


def packed_projection(raw: U8, rows: int, cols: int) -> F32:
    code_count = rows * cols // 2
    code = raw[:code_count]
    scale = raw[code_count:]
    if scale.size != rows * cols // 32 or np.any(scale == 255):
        raise ValueError("invalid E8M0 pack scales")
    low, high = code & 15, code >> 4
    table = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
    )
    values = np.empty(code_count * 2, np.float32)
    values[::2], values[1::2] = table[low], table[high]
    return np.ldexp(
        values.reshape(-1, 32), scale.astype(np.int32)[:, None] - 127
    ).reshape(rows, cols)


@dataclass(frozen=True)
class LayerAcceptance:
    values_checked: int
    experts_checked: int
    pack_bytes: int
    cold_bytes: int


def accept_layer(
    checkpoint: Checkpoint,
    packs: Path,
    layer: int,
    ranks: list[dict[str, list[int]]],
) -> LayerAcceptance:
    if not 0 <= layer < 40:
        raise ValueError("layer must be in the checkpoint's 0..39 range")
    checked, total_bytes, cold_bytes = 0, 0, 0
    full_layers = (5, 12, 27)
    if len(ranks) != 4:
        raise ValueError(f"L{layer}: expected TP4")
    samples = range(384) if layer in full_layers else (0, 191, 383)
    for rank, split in enumerate(ranks):
        if set(split) != {"hot", "cold"}:
            raise ValueError("pack partition metadata may contain only expert IDs")
        if split != ranks[0]:
            raise ValueError("hot/cold ownership must agree across TP ranks")
        ids = split["hot"] + split["cold"]
        if sorted(ids) != list(range(384)):
            raise ValueError(f"L{layer} rank{rank}: invalid expert partition")
        for temperature in ("hot", "cold"):
            path = packs / f"L{layer:02d}-r{rank}-{temperature}.bin"
            expected_size = len(split[temperature]) * 4_874_240
            if path.stat().st_size != expected_size:
                raise ValueError(f"invalid pack size: {path}")
            total_bytes += expected_size
            if temperature == "cold":
                cold_bytes += expected_size
    for expert in samples:
        for projection, (rows, cols) in GEOMETRY.items():
            expected = checkpoint.projection(layer, expert, projection)
            for rank, split in enumerate(ranks):
                temperature = "hot" if expert in split["hot"] else "cold"
                position = split[temperature].index(expert)
                if projection == "w1":
                    offset, nr, nk = 0, 576, 5120
                    want = expected[rank * 576 : (rank + 1) * 576]
                elif projection == "w3":
                    offset, nr, nk = 1_566_720, 576, 5120
                    want = expected[rank * 576 : (rank + 1) * 576]
                else:
                    offset, nr, nk = 3_133_440, 5120, 640
                    want = expected[:, rank * 576 : (rank + 1) * 576]
                count = nr * nk * 17 // 32
                raw = np.memmap(
                    packs / f"L{layer:02d}-r{rank}-{temperature}.bin",
                    mode="r",
                    dtype=np.uint8,
                    offset=position * 4_874_240 + offset,
                    shape=(count,),
                )
                actual = packed_projection(raw, nr, nk)
                if projection == "w2":
                    if np.any(actual[:, 576:] != 0):
                        raise AssertionError("W2 padding is not zero")
                    actual = actual[:, :576]
                if not np.array_equal(actual.view(np.uint32), want.view(np.uint32)):
                    unequal = np.count_nonzero(
                        actual.view(np.uint32) != want.view(np.uint32)
                    )
                    raise AssertionError(
                        f"P1 FAIL L{layer} E{expert} {projection} R{rank}: "
                        f"{unequal} unequal FP32 bit patterns"
                    )
                checked += actual.size
    return LayerAcceptance(checked, len(samples), total_bytes, cold_bytes)


def accept(
    weights: Path, verification_log: Path, tally: Path, packs: Path, output: Path
) -> None:
    started = time.monotonic()
    manifest = json.loads((packs / "manifest.json").read_text())
    if manifest["format"] != "nvfp4-8r-mxfp4-v1":
        raise ValueError("unknown pack format")
    if set(manifest) != {"format", "layers"}:
        raise ValueError("pack metadata must contain geometry and expert IDs only")
    if len(manifest["layers"]) != 40:
        raise ValueError("P1 requires all 40 layers")
    with np.load(tally, allow_pickle=False) as routing:
        phases = list(routing["phases"])
        counts = routing["counts"]
        if counts.shape != (len(phases), 40, 384):
            raise ValueError("invalid real P16 tally geometry")
        decode = counts[phases.index("teacher")]
        if not np.issubdtype(decode.dtype, np.integer) or np.any(decode < 0):
            raise ValueError("invalid real P16 decode counts")
        coldest_hot: tuple[int, int, int] | None = None
        warmest_cold: tuple[int, int, int] | None = None
        for layer, ranks in enumerate(manifest["layers"]):
            for expert in ranks[0]["hot"]:
                value = (int(decode[layer, expert]), layer, expert)
                if coldest_hot is None or value < coldest_hot:
                    coldest_hot = value
            for expert in ranks[0]["cold"]:
                value = (int(decode[layer, expert]), layer, expert)
                if warmest_cold is None or value > warmest_cold:
                    warmest_cold = value
        if coldest_hot is None or warmest_cold is None:
            raise ValueError("P1 requires both hot and cold experts")
        if warmest_cold >= coldest_hot:
            raise AssertionError(
                "partition does not select the coldest P16 decode experts"
            )
    checkpoint = Checkpoint(weights, verification_log)
    checked, full_layers = 0, (5, 12, 27)
    total_bytes, cold_bytes = 0, 0
    for layer, ranks in enumerate(manifest["layers"]):
        layer_result = accept_layer(checkpoint, packs, layer, ranks)
        checked += layer_result.values_checked
        total_bytes += layer_result.pack_bytes
        cold_bytes += layer_result.cold_bytes
        print(
            json.dumps(
                {
                    "layer": layer,
                    "experts_checked": layer_result.experts_checked,
                    "values_checked": checked,
                }
            ),
            flush=True,
        )
    if not 79 * 2**30 <= cold_bytes <= 81 * 2**30:
        raise AssertionError(
            f"cold allocation is not approximately 80 GiB: {cold_bytes}"
        )
    result = {
        "status": "PASS",
        "full_layers": list(full_layers),
        "sampled_experts_other_layers": [0, 191, 383],
        "values_checked": checked,
        "relative_l2": 0,
        "pack_bytes": total_bytes,
        "cold_bytes": cold_bytes,
        "manifest_sha256": hashlib.sha256(
            (packs / "manifest.json").read_bytes()
        ).hexdigest(),
        "seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    parser.add_argument(
        "--tally",
        type=Path,
        default=Path(os.environ["LANE"] + "/engine/private/p16-tally.npz"),
    )
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accept(args.weights, args.verification_log, args.tally, args.packs, args.output)
