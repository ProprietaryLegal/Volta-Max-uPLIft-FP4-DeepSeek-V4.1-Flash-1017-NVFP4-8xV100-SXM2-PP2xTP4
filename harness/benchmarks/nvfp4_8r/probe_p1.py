"""Measure the converter against actual checkpoint projections before bulk work."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .accept_p1 import Checkpoint, packed_projection
from .build_packs import Source, fold


def probe(weights: Path, verification_log: Path) -> None:
    source = Source(weights.resolve(), verification_log)
    oracle = Checkpoint(weights, verification_log)
    for layer in (5, 12, 27):
        for projection in ("w1", "w3", "w2"):
            started = time.monotonic()
            native = source.projection(layer, 0, projection)
            codes, scales = fold(native)
            actual = packed_projection(
                np.concatenate((codes.ravel(), scales.ravel())),
                codes.shape[0],
                codes.shape[1] * 2,
            )
            expected = oracle.projection(layer, 0, projection)
            if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
                raise AssertionError(
                    f"P1 real-weight probe failed: L{layer} {projection}"
                )
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "expert": 0,
                        "projection": projection,
                        "values": actual.size,
                        "relative_l2": 0,
                        "bit_equal": True,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    args = parser.parse_args()
    probe(args.weights, args.verification_log)
