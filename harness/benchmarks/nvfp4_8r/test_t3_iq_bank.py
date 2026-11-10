"""Plan 01 step 4 (CPU): the warm IQ bank holds production's device bytes.

For layers 5, 12, 27, TP ranks 0 and 3 and eight experts each, the bank that
IqWarmBank.from_source reads from the real GGUF copy must equal, byte for
byte, what production's layout puts on that rank's card:
  SharedExpertLayout.for_type(5120, 2304, 4, rank, role, type).pack(original)
with VLLM_V41_DOWN_ROWS=1 (row-split down). The layout is production's
reference packer, not the code under test.
If per-rank slices exist (make_t3_iq_slices, plain numpy slicing), the bank
read from them through IqShardSource.from_slice_file must equal the same.

Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE (the missing input is named).
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ["LANE"])
GGUF = ROOT / "models/gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"
ROLES = {"w1": "gate", "w3": "up", "w2": "down"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", type=Path, required=True,
                        help="directory of make_t3_iq_slices (checked if its manifest exists)")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from vllm.model_executor.layers.quantization import gguf_shared_layout as layout

    def finish(status: str, detail: dict) -> int:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({"check": "T3-step4-iq-bank", "status": status, **detail},
                                       indent=1) + "\n")
        print(status, json.dumps(detail))
        return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[status]

    if not layout.DOWN_ROWS:
        return finish("INCONCLUSIVE", {"missing": "VLLM_V41_DOWN_ROWS=1 in the environment"})
    if not GGUF.is_file():
        return finish("INCONCLUSIVE", {"missing": str(GGUF)})
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native
    from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index

    index = gguf_tensor_index(gguf_shard_paths(GGUF))
    rng = np.random.default_rng(4)
    cpu = torch.device("cpu")
    fails: list[str] = []
    compared = {"gguf": 0, "slices": 0}
    have_slices = (a.slices / "iq-manifest.json").is_file()
    for layer in (5, 12, 27):
        tensors = {role: index[f"blk.{layer}.ffn_{name}_exps.weight"] for role, name in ROLES.items()}
        for rank in (0, 3):
            experts = sorted(rng.choice(384, 8, replace=False).tolist())
            expected = {
                role: np.stack([
                    layout.SharedExpertLayout.for_type(
                        5120, 2304, 4, rank, role, int(t.tensor_type)
                    ).pack(np.asarray(t.data[e]))
                    for e in experts
                ])
                for role, t in tensors.items()
            }
            sources = {"gguf": native.IqLayerSource(*(
                native.IqShardSource.from_gguf_tensor(tensors[role], role, rank) for role in ROLES
            ))}
            if have_slices:
                sources["slices"] = native.IqLayerSource(*(
                    native.IqShardSource.from_slice_file(
                        a.slices / f"iq-L{layer:02d}-r{rank}-{role}.bin", role,
                        int(tensors[role].tensor_type),
                    ) for role in ROLES
                ))
            for name, source in sources.items():
                bank = native.IqWarmBank.from_source(source, experts, cpu)
                if (bank.type13, bank.type2) != (int(tensors["w1"].tensor_type),
                                                 int(tensors["w2"].tensor_type)):
                    fails.append(f"{name} L{layer} r{rank}: bank types {bank.type13}/{bank.type2}")
                for role in ROLES:
                    got = getattr(bank, role).numpy()
                    if got.shape != expected[role].shape or not np.array_equal(got, expected[role]):
                        fails.append(f"{name} L{layer} r{rank} {role}: bytes differ from the layout pack")
                    compared[name] += int(expected[role].nbytes)
    detail = {"bytes_compared": compared, "slices_checked": have_slices, "failures": fails}
    return finish("FAIL" if fails else "PASS", detail)


if __name__ == "__main__":
    sys.exit(main())
