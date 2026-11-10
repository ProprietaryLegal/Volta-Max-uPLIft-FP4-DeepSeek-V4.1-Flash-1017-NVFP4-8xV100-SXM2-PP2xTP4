"""Check one completed P1 layer early, without issuing a full P1 acceptance.

Expected expert ownership is derived independently from the private real tally.
Weight expectations use the existing raw NVFP4 formula oracle, never conversion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .accept_p1 import Checkpoint, accept_layer


def expected_ranks(tally: Path, layer: int) -> list[dict[str, list[int]]]:
    with np.load(tally, allow_pickle=False) as data:
        phases = list(data["phases"])
        counts = data["counts"]
        if counts.shape != (len(phases), 40, 384):
            raise ValueError("invalid real P16 tally geometry")
        decode = counts[phases.index("teacher")]
        if not np.issubdtype(decode.dtype, np.integer) or np.any(decode < 0):
            raise ValueError("invalid P16 decode counts")
        gate_bytes = 576 * 5120 * 17 // 32
        down_bytes = 5120 * 640 * 17 // 32
        cold_count = round(80 * 2**30 / (4 * (2 * gate_bytes + down_bytes)))
        ordered = sorted(
            (int(decode[layer_index, expert]), layer_index, expert)
            for layer_index in range(40)
            for expert in range(384)
        )
        cold = {
            expert
            for _, layer_index, expert in ordered[:cold_count]
            if layer_index == layer
        }
    return [
        {
            "hot": [expert for expert in range(384) if expert not in cold],
            "cold": sorted(cold),
        }
        for _ in range(4)
    ]


def probe(
    weights: Path,
    verification_log: Path,
    tally: Path,
    packs: Path,
    layer: int,
    output: Path,
) -> None:
    started = time.monotonic()
    checkpoint = Checkpoint(weights, verification_log)
    result = accept_layer(checkpoint, packs, layer, expected_ranks(tally, layer))
    receipt = {
        "status": "SINGLE_LAYER_PASS",
        "p1_accepted": False,
        "layer": layer,
        "experts_checked": result.experts_checked,
        "values_checked": result.values_checked,
        "relative_l2": 0,
        "pack_bytes": result.pack_bytes,
        "cold_bytes": result.cold_bytes,
        "seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    parser.add_argument("--tally", type=Path, required=True)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=range(40), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(
        args.weights,
        args.verification_log,
        args.tally,
        args.packs,
        args.layer,
        args.output,
    )
