"""P4 arm 2 (CPU, the 8-card host): repack the checkpoint's MXFP4 Engram into reader rows.

Output, in the lane only:
  OUT.bin   layer-1 table then layer-14 table, 144-byte rows
              (128 code bytes | 8 E8M0 scales | 8 zero bytes), row r of a
              table at table_offset + 144 r: the layout the node Engram reader
              already reads for IQ4_NL (row size 144), so it reads this unchanged;
  OUT.json  {"format", "file", "row_bytes", "tables": [{layer, offset, rows}]},
              written last, only after every row is written and verified.
Verification: every chunk is re-read from the written file and compared bytewise
with the checkpoint codes and scales (and zero padding) before the next chunk.
Gentle on the shared NVMe: ionice is the caller's; reads/writes are capped at
--mb-per-s and dropped from the page cache (POSIX_FADV_DONTNEED) after use.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .accept_p1 import Checkpoint

FORMAT = "nvfp4-8r-engram-mxfp4-144-v1"
ROW = 144
LAYERS = (1, 14)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="path without suffix")
    p.add_argument("--chunk-rows", type=int, default=1 << 20)
    p.add_argument("--mb-per-s", type=float, default=200.0)
    a = p.parse_args()
    ck = Checkpoint(a.checkpoint, a.checkpoint.parent.parent / "logs/copy_weights.log")
    binary = a.out.parent / (a.out.name + ".bin")
    manifest = a.out.parent / (a.out.name + ".json")
    if binary.exists() or manifest.exists():
        raise SystemExit(f"refusing to overwrite {binary} / {manifest}")
    tables, offset = [], 0
    sources = []
    for layer in LAYERS:
        codes, ctype = ck.tensor(f"layers.{layer}.engram.embed.weight")
        scales, stype = ck.tensor(f"layers.{layer}.engram.embed.scale")
        if ctype != "U8" or stype != "F8_E8M0" or codes.size != scales.size * 16:
            raise ValueError(f"layer {layer}: unexpected Engram tensors {ctype}/{stype}")
        rows = scales.size // 8
        sources.append((layer, codes, scales))
        tables.append({"layer": layer, "offset": offset, "rows": rows})
        offset += rows * ROW
    budget = a.mb_per_s * 1e6
    src_fds = {
        str(mm.filename): os.open(mm.filename, os.O_RDONLY)
        for _, c, s in sources for mm in (c, s)
    }
    fd = os.open(binary, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    start = time.monotonic()
    done_bytes = 0
    try:
        os.ftruncate(fd, offset)
        for (layer, codes_mm, scales_mm), table in zip(sources, tables):
            n = table["rows"]
            codes, scales = codes_mm.reshape(n, 128), scales_mm.reshape(n, 8)
            for r0 in range(0, n, a.chunk_rows):
                r1 = min(n, r0 + a.chunk_rows)
                block = np.zeros((r1 - r0, ROW), np.uint8)
                block[:, :128] = codes[r0:r1]
                block[:, 128:136] = scales[r0:r1]
                if np.any(block[:, 128:136] == 255):
                    raise ValueError(f"layer {layer}: E8M0 NaN scale near row {r0}")
                where = table["offset"] + r0 * ROW
                if os.pwrite(fd, block.tobytes(), where) != block.nbytes:
                    raise OSError(f"short write at {where}")
                os.fdatasync(fd)
                back = np.frombuffer(os.pread(fd, block.nbytes, where), np.uint8)
                if not np.array_equal(back.reshape(block.shape), block):
                    raise ValueError(f"layer {layer}: read-back mismatch at row {r0}")
                os.posix_fadvise(fd, where, block.nbytes, os.POSIX_FADV_DONTNEED)
                for mm, width in ((codes_mm, 128), (scales_mm, 8)):
                    os.posix_fadvise(src_fds[str(mm.filename)], mm.offset + r0 * width,
                                     (r1 - r0) * width, os.POSIX_FADV_DONTNEED)
                done_bytes += 2 * block.nbytes
                ahead = done_bytes / budget - (time.monotonic() - start)
                if ahead > 0:
                    time.sleep(ahead)
                if (r0 // a.chunk_rows) % 32 == 0:
                    print(json.dumps({"layer": layer, "row": r1, "rows": n,
                                      "elapsed_s": round(time.monotonic() - start)}),
                          flush=True)
    finally:
        os.close(fd)
        for f in src_fds.values():
            os.close(f)
    manifest.write_text(json.dumps({"format": FORMAT, "file": binary.name, "row_bytes": ROW,
                                    "tables": tables}, indent=1) + "\n")
    print(json.dumps({"status": "PACKED", "bytes": offset, "tables": tables,
                      "elapsed_s": round(time.monotonic() - start)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
