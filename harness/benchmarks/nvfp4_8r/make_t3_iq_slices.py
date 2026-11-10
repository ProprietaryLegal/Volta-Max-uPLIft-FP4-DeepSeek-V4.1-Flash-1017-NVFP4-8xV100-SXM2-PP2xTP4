"""CPU only (the 8-card host): per-rank IQ slices of the GGUF copy's routed experts.

For A3 and the falsifier-1 run on the test host, which has no GGUF. For each layer
and TP4 rank, the bytes a card holds of every one of the 384 experts under
the row-split layout (VLLM_V41_DOWN_ROWS=1):
  w1 (gate) [384, 576, 20 x bb]   rows rank*576 .. +575 of the 2304
  w3 (up)   [384, 576, 20 x bb]   the same rows
  w2 (down) [384, 1280, 9 x bb]   rows rank*1280 .. +1279 of the 5120
Sliced with plain numpy indexing of the memory-mapped tensors (no engine
slicing code), so the engine's own reader can be checked against them.
Files: iq-L{layer}-r{rank}-{w1,w3,w2}.bin and iq-manifest.json (types, sizes,
sha256). Weights only: nothing here derives from the private tally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gguf
import numpy as np

ROLES = {"w1": ("gate", 2304, 576), "w3": ("up", 2304, 576), "w2": ("down", 5120, 1280)}
BLOCKS = {"w1": 20, "w3": 20, "w2": 9}  # 256-weight blocks per row (K 5120, 5120, 2304)
TYPES = (16, 17, 29)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True, help="first shard")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index

    started = time.monotonic()
    index = gguf_tensor_index(gguf_shard_paths(a.gguf))
    a.output.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"format": "nvfp4-8r-iq-slices-v1", "layers": {}}
    for layer in a.layers:
        entry: dict = {"files": {}}
        for role, (name, all_rows, local_rows) in ROLES.items():
            tensor = index[f"blk.{layer}.ffn_{name}_exps.weight"]
            kind = int(tensor.tensor_type)
            if kind not in TYPES:
                raise ValueError(f"L{layer} {name}: GGML type {kind} is not IQ1_M/IQ2_XXS/IQ2_XS")
            width = BLOCKS[role] * gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(kind)][1]
            if tensor.data.shape != (384, all_rows, width) or tensor.data.dtype != np.uint8:
                raise ValueError(f"L{layer} {name}: data {tensor.data.shape}, want (384, {all_rows}, {width})")
            entry[f"type_{role}"] = kind
            for rank in a.ranks:
                if not 0 <= rank < 4:
                    raise ValueError("ranks are 0..3 (TP4)")
                part = np.ascontiguousarray(
                    tensor.data[:, rank * local_rows : (rank + 1) * local_rows, :]
                )
                path = a.output / f"iq-L{layer:02d}-r{rank}-{role}.bin"
                part.tofile(path)
                entry["files"][path.name] = {
                    "shape": list(part.shape), "bytes": int(part.nbytes),
                    "sha256": hashlib.sha256(part).hexdigest(),
                }
                del part
        if entry["type_w1"] != entry["type_w3"]:
            raise ValueError(f"L{layer}: gate type {entry['type_w1']} != up type {entry['type_w3']}")
        manifest["layers"][str(layer)] = entry
        print(json.dumps({"layer": layer, "type13": entry["type_w1"], "type2": entry["type_w2"],
                          "seconds": round(time.monotonic() - started, 1)}), flush=True)
    (a.output / "iq-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
