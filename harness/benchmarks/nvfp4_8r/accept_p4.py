"""P4 gate (the test host, one card): Engram rows equal to the fp32 reference.

References come from make_p4_refs.py (CPU formulas on the raw table bytes,
IQ4_NL cross-checked with gguf-py). The engine returns fp16 rows, so "equal"
means bit-equal to the fp32 reference rounded once to fp16 (every sampled
value is representable: |v| <= 6.6, scales 2^-3..2^1).
- arm 1, production IQ4_NL: the engine's own row dequantizer
  (dequantize_gguf_rows, type 20, width 256) on the raw 144-byte rows;
- arm 2, checkpoint MXFP4: rows repacked by the engine's pack_mxfp4_rows into
  the reader's 144-byte row (128 codes | 8 E8M0 | 8 zero) and dequantized by
  dequantize_mxfp4_rows on the GPU.
Logged, not gated: max |engine - fp32 ref| and rows/s at the decode batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .accept_p2 import Inconclusive, gpu_gate

LAYERS = (1, 14)


def rows_per_s(fn, n: int) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(50):
        fn()
    b.record()
    b.synchronize()
    return round(50 * n / (a.elapsed_time(b) / 1000.0))


def compare(out: torch.Tensor, ref: np.ndarray) -> dict:
    expected = torch.from_numpy(ref).half()
    got = out.cpu()
    return {
        "bit_equal_fp16": bool(torch.equal(got, expected)),
        "max_abs_vs_fp32": float((got.float() - torch.from_numpy(ref)).abs().max()),
        "rows": int(ref.shape[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--card", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        gate = gpu_gate(a.card)
    except Inconclusive as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        return 3
    from vllm.model_executor.layers.quantization.gguf import dequantize_gguf_rows

    dev = torch.device("cuda", 0)
    with np.load(a.refs, allow_pickle=False) as data:
        refs = {k: data[k] for k in data.files}
    res: dict = {"arm1_iq4nl": {}, "arm2_mxfp4": {}}
    fails: list[str] = []
    try:
        from vllm.model_executor.layers.deepseek_v41_engram_io import (
            dequantize_mxfp4_rows,
            pack_mxfp4_rows,
        )
    except ImportError as exc:
        dequantize_mxfp4_rows = pack_mxfp4_rows = None
        fails.append(f"arm 2 not available in the engine: {exc}")
    for layer in LAYERS:
        raw = torch.from_numpy(refs[f"iq4_L{layer}"]).to(dev)
        out = dequantize_gguf_rows(raw, 20, 256, torch.float16, True)
        r = compare(out, refs[f"ref_iq4_L{layer}"])
        r["rows_per_s_144"] = rows_per_s(
            lambda raw=raw: dequantize_gguf_rows(raw[:144], 20, 256, torch.float16, True), 144
        )
        res["arm1_iq4nl"][str(layer)] = r
        if not r["bit_equal_fp16"]:
            fails.append(f"arm 1 L{layer}: IQ4_NL rows != fp32 reference")
        if pack_mxfp4_rows is None:
            continue
        packed = pack_mxfp4_rows(
            torch.from_numpy(refs[f"mx_codes_L{layer}"]),
            torch.from_numpy(refs[f"mx_scales_L{layer}"]),
        )
        if packed.shape != (raw.shape[0], 144) or packed.dtype != torch.uint8:
            fails.append(f"arm 2 L{layer}: packed rows {tuple(packed.shape)} {packed.dtype}")
            continue
        packed = packed.to(dev)
        out = dequantize_mxfp4_rows(packed, torch.float16)
        r = compare(out, refs[f"ref_mx_L{layer}"])
        r["rows_per_s_144"] = rows_per_s(
            lambda packed=packed: dequantize_mxfp4_rows(packed[:144], torch.float16), 144
        )
        res["arm2_mxfp4"][str(layer)] = r
        if not r["bit_equal_fp16"]:
            fails.append(f"arm 2 L{layer}: MXFP4 rows != fp32 reference")
    receipt = {"gate": gate, "results": res, "failures": fails,
               "status": "P4_ROWS_PASS" if not fails else "P4_ROWS_FAIL"}
    a.output.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"status": receipt["status"], "failures": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
