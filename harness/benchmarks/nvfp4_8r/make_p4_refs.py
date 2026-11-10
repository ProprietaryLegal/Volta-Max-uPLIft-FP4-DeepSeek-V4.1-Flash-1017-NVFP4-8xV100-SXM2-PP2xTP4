"""P4 references (CPU, the 8-card host): sampled Engram rows and their fp32 values.

For both Engram tables (layers 1 and 14) and the same row ids:
- arm 1, production IQ4_NL sidecar (lane copy): raw 144-byte rows and fp32
  values from the IQ4_NL formula (per 32: fp16 d x kvalues[nibble], low
  nibbles are elements 0..15, high nibbles 16..31), cross-checked equal to
  gguf-py's independent dequantizer;
- arm 2, the NVFP4 checkpoint's MXFP4 table: raw codes [128] and E8M0 scales
  [8] per row and fp32 values from the MXFP4 formula (e2m1 x 2^(s - 127), low
  nibble = even element).
Both arms quantize one original table, so their rel L2 is logged as a sanity
check of layout (a wrong nibble order or row offset gives ~1.4, not ~0.1).
Rows: first, last, and a seeded uniform sample. Engine code is never called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np

from .accept_p1 import Checkpoint

LAYERS = (1, 14)
KVALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32
)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


def iq4nl_rows(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(-1, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)  # [B, 1]
    qs = blocks[:, 2:]
    q = np.concatenate((qs & 15, qs >> 4), axis=1)  # elements 0..15, 16..31
    return (d * KVALUES[q]).reshape(raw.shape[0], 256)


def mxfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if np.any(scales == 255):
        raise ValueError("E8M0 NaN scale in sampled Engram rows")
    nib = np.empty((codes.shape[0], 256), np.uint8)
    nib[:, 0::2], nib[:, 1::2] = codes & 15, codes >> 4
    values = np.where(nib & 8, -E2M1[nib & 7], E2M1[nib & 7])
    block = np.ldexp(np.float32(1), scales.astype(np.int32) - 127).astype(np.float32)
    return (values.reshape(-1, 8, 32) * block[:, :, None]).reshape(-1, 256)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ck = Checkpoint(a.checkpoint, a.checkpoint.parent.parent / "logs/copy_weights.log")
    reader = gguf.GGUFReader(a.sidecar, mode="r")
    tables = {t.name: t for t in reader.tensors}
    out: dict[str, np.ndarray] = {}
    meta: dict = {"layers": list(LAYERS), "arm_rel_l2": {}, "gguf_py_equal": {}}
    for layer in LAYERS:
        t = tables[f"blk.{layer}.engram_embd.weight"]
        if int(t.tensor_type) != 20:
            raise ValueError(f"layer {layer} sidecar is not IQ4_NL")
        n = int(t.shape[1])
        codes_all, ctype = ck.tensor(f"layers.{layer}.engram.embed.weight")
        scales_all, stype = ck.tensor(f"layers.{layer}.engram.embed.scale")
        if ctype != "U8" or stype != "F8_E8M0":
            raise ValueError(f"layer {layer} checkpoint Engram types {ctype}/{stype}")
        if codes_all.size != n * 128 or scales_all.size != n * 8:
            raise ValueError(f"layer {layer}: sidecar {n} rows, checkpoint differs")
        gen = np.random.default_rng(14_000 + layer)
        rows = np.unique(np.concatenate(([0, n - 1], gen.integers(0, n, a.rows))))
        iq4 = np.stack([np.asarray(t.data[r]) for r in rows]).astype(np.uint8)
        ref_iq4 = iq4nl_rows(iq4)
        check = gguf.quants.dequantize(iq4, gguf.GGMLQuantizationType.IQ4_NL)
        meta["gguf_py_equal"][layer] = bool(np.array_equal(check.astype(np.float32), ref_iq4))
        if not meta["gguf_py_equal"][layer]:
            raise SystemExit(f"FAIL: IQ4_NL formula != gguf-py on layer {layer}")
        codes = np.stack([codes_all[r * 128 : (r + 1) * 128] for r in rows])
        scales = np.stack([scales_all[r * 8 : (r + 1) * 8] for r in rows])
        ref_mx = mxfp4_rows(codes, scales)
        diff = np.linalg.norm(ref_mx - ref_iq4) / np.linalg.norm(ref_mx)
        meta["arm_rel_l2"][layer] = float(diff)
        out |= {f"rows_L{layer}": rows.astype(np.int64), f"iq4_L{layer}": iq4,
                f"ref_iq4_L{layer}": ref_iq4, f"mx_codes_L{layer}": codes,
                f"mx_scales_L{layer}": scales, f"ref_mx_L{layer}": ref_mx}
        print(json.dumps({"layer": layer, "rows": int(rows.size), "table_rows": n,
                          "iq4_vs_mx_rel_l2": float(diff)}), flush=True)
    out["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez(a.output, **out)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
