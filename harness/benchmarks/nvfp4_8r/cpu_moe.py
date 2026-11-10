"""ctypes wrapper of cpu_mxfp4_moe.c (P3 speed bench arm (b); bench only)."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def build(out_dir: Path) -> ctypes.CDLL:
    lib = out_dir / "libcpumoe.so"
    subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-fopenmp", "-shared", "-fPIC",
                    str(HERE / "cpu_mxfp4_moe.c"), "-o", str(lib), "-lm"], check=True)
    dll = ctypes.CDLL(str(lib))
    p = ctypes.c_void_p
    dll.expert_slice.argtypes = [p, p, p, p, ctypes.c_int, p, p, p, p]
    dll.expert_slice.restype = None
    return dll


class CpuMoe:
    def __init__(self, dll: ctypes.CDLL, max_m: int = 16):
        self.dll = dll
        self.xs = np.zeros((max_m, 5120), np.float32)
        self.gu = np.zeros((max_m, 1152), np.float32)
        self.act = np.zeros((max_m, 640), np.float32)

    def expert(self, rec: np.ndarray, x: np.ndarray, rows: np.ndarray, w: np.ndarray,
               out: np.ndarray) -> None:
        rows = np.ascontiguousarray(rows, np.int32)
        w = np.ascontiguousarray(w, np.float32)
        assert rec.flags.c_contiguous and x.flags.c_contiguous and out.flags.c_contiguous
        self.dll.expert_slice(rec.ctypes.data, x.ctypes.data, rows.ctypes.data, w.ctypes.data,
                              len(rows), out.ctypes.data, self.xs.ctypes.data,
                              self.gu.ctypes.data, self.act.ctypes.data)

    def layer(self, raw: np.ndarray, x: np.ndarray, ids: np.ndarray, tw: np.ndarray,
              experts=None) -> np.ndarray:
        """Sum over routed experts (all, or only ``experts``) for one TP slice."""
        out = np.zeros((x.shape[0], 5120), np.float32)
        wanted = set(np.unique(ids).tolist()) if experts is None else set(experts)
        for e in sorted(wanted & set(np.unique(ids).tolist())):
            rows, slots = np.nonzero(ids == e)
            self.expert(raw[e], x, rows, tw[rows, slots], out)
        return out


def threads() -> int:
    return int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
