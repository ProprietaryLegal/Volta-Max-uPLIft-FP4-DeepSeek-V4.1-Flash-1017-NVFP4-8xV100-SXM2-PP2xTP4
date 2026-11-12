"""Build the NVFP4-8R prefill RB library (the RB 4 patch). CPU only: no card is touched.

    cp ENGINE/csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu ENGINE/csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh csrc/
    CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=7.0 python -I build_qpn_rb4.py

(ENGINE: the engine source tree, see engine/README.md; run from this folder with the engine's Python.)
gen/nvfp4_8r_qpn_rb.cu = csrc/skinny_moe_qpn_sm70.cu (must be sha256 31d6ea4f...) followed by qpn_rb4_append.cu;
the script checks that the generated file starts with the source's exact bytes. Built as a torch extension
(op torch.ops.nvfp4_8r_qpn_rb.moe_prefill) into build/, sm_70 only, -fvisibility=hidden, ptxas -v on (its register and spill lines
go to the log). Prints the .so path and its sha256 last. Exit 0 built, 1 refused. The engine loads the result only
under the name v41qpnrb4.so beside nvfp4_8r_qpn_rb.py, and only if its sha256 is the one that module pins.
"""

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "csrc" / "skinny_moe_qpn_sm70.cu"
APPEND = HERE / "qpn_rb4_append.cu"
GEN = HERE / "gen" / "nvfp4_8r_qpn_rb.cu"
SRC_SHA = "31d6ea4fcb310a57"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("REFUSED: set CUDA_VISIBLE_DEVICES= (empty); the build touches no card")
        return 1
    if not sha(SRC).startswith(SRC_SHA):
        print(f"REFUSED: {SRC} is sha256 {sha(SRC)[:16]}, want {SRC_SHA}")
        return 1
    src, app = SRC.read_bytes(), APPEND.read_bytes()
    GEN.parent.mkdir(exist_ok=True)
    (HERE / "build").mkdir(exist_ok=True)
    GEN.write_bytes(src + app)
    gen = GEN.read_bytes()
    if gen[: len(src)] != src:
        print("REFUSED: the generated file does not start with the source bytes")
        return 1
    print(f"gen {GEN} sha256 {sha(GEN)[:16]}: source {SRC_SHA} ({len(src)} B) + append {sha(APPEND)[:16]} ({len(app)} B)")
    from torch.utils import cpp_extension

    lib = cpp_extension.load(
        name="nvfp4_8r_qpn_rb",
        sources=[str(GEN)],
        extra_include_paths=[str(HERE / "csrc")],
        extra_cflags=["-O3", "-fvisibility=hidden"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "-Xptxas=-v",
                           "-Xcompiler=-fvisibility=hidden", "--expt-relaxed-constexpr"],
        build_directory=str(HERE / "build"),
        is_python_module=False,
        verbose=True,
    )
    del lib
    so = HERE / "build" / "nvfp4_8r_qpn_rb.so"
    print(f"BUILT {so} sha256 {sha(so)[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
