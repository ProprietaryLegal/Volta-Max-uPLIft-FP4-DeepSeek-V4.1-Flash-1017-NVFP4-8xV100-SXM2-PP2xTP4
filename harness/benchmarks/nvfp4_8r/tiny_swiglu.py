"""Tiny first bit-check of the one-launch decode SwiGLU (CAMPAIGN 8.4) on one card.

Run before test_swiglu (bit-check on tiny inputs first). Rows S = 1, 6 and 36
(M = 1 and 6 at top-k 6). y13 holds seeded normal values times 4 plus the FP16 specials (+-0,
+-inf, NaN, +-10, +-10.0078, +-65504, the smallest subnormal and normal) in its first columns.
y13 and inter are row slices of larger buffers with one guard row above and below, all filled
with a sentinel: PASS needs inter[:, :576] bit-equal to the chain (NativeV41Moe.forward's three
lines; NaN equal only to NaN), inter's columns 576..639, both guard rows of inter and every byte
of y13's buffer unchanged. Also SwigluKernel(6, 6) is built and called once at S = 36.
Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig

SENTINEL = 0x7BFF
SPECIALS = [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x4900, 0xC900, 0x4901, 0xC901, 0x7BFF, 0xFBFF,
            0x0001, 0x8001, 0x0400, 0x8400]


def chain_into(y13: torch.Tensor, inter: torch.Tensor) -> None:
    g = y13[:, :nat.TP_INTER].float().clamp(max=nat.SWIGLU_LIMIT)
    u = y13[:, nat.TP_INTER:].float().clamp(-nat.SWIGLU_LIMIT, nat.SWIGLU_LIMIT)
    inter[:, :nat.TP_INTER].copy_(torch.nn.functional.silu(g) * u)


def one(s: int, dev: torch.device, use_layer_kernel: bool) -> dict:
    gen = torch.Generator().manual_seed(1000 + s)
    sent = torch.tensor([SENTINEL], dtype=torch.int16).view(torch.float16).item()
    vals = (torch.randn((s, swig.Y13_COLS), generator=gen) * 4).to(torch.float16)
    sp = torch.tensor(SPECIALS, dtype=torch.int32).to(torch.int16).view(torch.float16)
    for r in range(s):  # every special as gate against every special as up, spread over the rows
        vals[r, : len(SPECIALS)] = sp
        vals[r, swig.INTER: swig.INTER + len(SPECIALS)] = sp.roll(r)
    ybuf = torch.full((s + 2, swig.Y13_COLS), sent, dtype=torch.float16, device=dev)
    ybuf[1:s + 1] = vals.to(dev)
    ycopy = ybuf.clone()
    ic = torch.full((s + 2, swig.OUT_COLS), sent, dtype=torch.float16, device=dev)
    ik = ic.clone()
    chain_into(ybuf[1:s + 1], ic[1:s + 1])
    if use_layer_kernel:
        swig.SwigluKernel(6, nat.TOP_K, dev)(ybuf[1:s + 1], ik[1:s + 1])
    else:
        swig.swiglu_into(ybuf[1:s + 1], ik[1:s + 1])
    torch.cuda.synchronize(dev)
    a, b = ic[1:s + 1, :swig.INTER], ik[1:s + 1, :swig.INTER]
    both_nan = torch.isnan(a) & torch.isnan(b)
    differ = int(((a.view(torch.int16) != b.view(torch.int16)) & ~both_nan).sum())
    untouched = ik.clone()
    untouched[1:s + 1, :swig.INTER] = sent
    guard_ok = bool((untouched.view(torch.int16) == SENTINEL).all())
    y_ok = bool(torch.equal(ybuf.view(torch.int16), ycopy.view(torch.int16)))
    return {"S": s, "layer_kernel": use_layer_kernel, "values": s * swig.INTER, "differ": differ,
            "both_nan": int(both_nan.sum()), "guards_and_pad_untouched": guard_ok, "y13_untouched": y_ok,
            "pass": differ == 0 and guard_ok and y_ok}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gate-json", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    rows = [one(1, dev, False), one(6, dev, False), one(36, dev, False), one(36, dev, True)]
    res = {"check": "swiglu_tiny", "gate": json.loads(a.gate_json), "rows": rows,
           "device": torch.cuda.get_device_name(dev),
           "status": "PASS" if all(r["pass"] for r in rows) else "FAIL"}
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": res["status"], "rows": rows}))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
