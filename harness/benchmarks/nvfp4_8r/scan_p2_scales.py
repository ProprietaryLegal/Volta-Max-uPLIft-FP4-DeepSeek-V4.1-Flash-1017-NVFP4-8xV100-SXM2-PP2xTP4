"""CPU pre-check for P2: every packed expert's E8M0 rasters must pass the
engine's own rebase_e8m0_for_fp16 (dry run), exactly as NativeExpertSet
feeds them (w1|w3 concatenated; W2 padding scales set to the last real one).
Reads only the scale byte ranges of each record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    rebase_e8m0_for_fp16,
)

C13, S13 = nat._W13_CODES, nat._W13_SCALES
OFF_W1S, OFF_W3S = C13, 2 * C13 + S13
OFF_W2S = 2 * (C13 + S13) + nat._W2_CODES


def scan(packs: Path) -> int:
    manifest = nat.read_manifest(packs)
    bad: list[str] = []
    spans13: list[int] = []
    spans2: list[int] = []
    for layer in range(40):
        for rank in range(4):
            for temp in ("hot", "cold"):
                ids = manifest[layer][rank][temp]
                path = packs / f"L{layer:02d}-r{rank}-{temp}.bin"
                rec = np.memmap(path, np.uint8, "r").reshape(len(ids), nat.EXPERT_BYTES)
                s13 = np.concatenate(
                    (rec[:, OFF_W1S : OFF_W1S + S13], rec[:, OFF_W3S : OFF_W3S + S13]), 1
                )
                s2 = np.array(rec[:, OFF_W2S:]).reshape(len(ids), 5120, 20)
                s2[..., 18:] = s2[..., 17:18]
                for name, arr, spans in (("w13", s13, spans13), ("w2", s2, spans2)):
                    t = torch.from_numpy(np.ascontiguousarray(arr))
                    flat = t.view(t.size(0), -1)
                    spans.extend((flat.amax(1).int() - flat.amin(1).int()).tolist())
                    try:
                        rebase_e8m0_for_fp16(t, dry_run=True)
                    except ValueError as exc:
                        bad.append(f"L{layer} r{rank} {temp} {name}: {exc}")
        print(json.dumps({"layer": layer, "bad_so_far": len(bad)}), flush=True)
    print(json.dumps({
        "records": len(spans13), "max_span_w13": max(spans13),
        "max_span_w2": max(spans2), "bad": bad,
        "status": "PASS" if not bad else "FAIL",
    }), flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--packs", type=Path, required=True)
    sys.exit(scan(p.parse_args().packs))
