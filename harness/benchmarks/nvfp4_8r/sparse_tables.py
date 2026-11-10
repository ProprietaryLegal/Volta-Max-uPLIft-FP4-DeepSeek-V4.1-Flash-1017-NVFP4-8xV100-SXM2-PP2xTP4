"""P4 IO check inputs: sparse copies of the two Engram tables with only the
reference rows present (holes elsewhere), so the real node reader can run on
the test host without moving ~110 GB per arm.

  extract (the 8-card host):      python -m ...sparse_tables extract --refs R --sidecar G --mxfp4 M --out D
  materialize (the file host, stdlib only):  python3 sparse_tables.py materialize --src D --dst W

extract writes, per arm, ARM.blob (the bytes) and ARM.json
{"name", "size", "pieces": [[file_offset, blob_offset, length], ...], "extra"};
pieces are copied from the real table files (the GGUF header and tensor
infos for arm 1; the repacked .bin rows for arm 2), never synthesized.
materialize recreates each file at full logical size with holes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def extract(a) -> None:
    import numpy as np

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = np.load(a.refs, allow_pickle=False)
    import gguf

    reader = gguf.GGUFReader(a.sidecar, mode="r")
    tables = {t.name: t for t in reader.tensors}
    first = min(int(t.data_offset) for t in reader.tensors)
    arm1 = [(0, first)]
    for layer in (1, 14):
        t = tables[f"blk.{layer}.engram_embd.weight"]
        arm1 += [(int(t.data_offset) + int(r) * 144, 144) for r in refs[f"rows_L{layer}"]]
    manifest = json.loads(Path(a.mxfp4).read_text())
    arm2 = []
    for t in manifest["tables"]:
        arm2 += [(t["offset"] + int(r) * 144, 144) for r in refs[f"rows_L{t['layer']}"]]
    binary = Path(a.mxfp4).parent / manifest["file"]
    for arm, src, pieces, name, extra in (
        ("arm1", Path(a.sidecar), arm1, Path(a.sidecar).name, None),
        ("arm2", binary, arm2, binary.name, {"manifest_name": Path(a.mxfp4).name,
                                            "manifest": manifest}),
    ):
        fd = os.open(src, os.O_RDONLY)
        blob, index, pos = bytearray(), [], 0
        try:
            for off, length in sorted(set(pieces)):
                data = os.pread(fd, length, off)
                if len(data) != length:
                    raise OSError(f"short read {src}@{off}")
                blob += data
                index.append([off, pos, length])
                pos += length
        finally:
            os.close(fd)
        (out / f"{arm}.blob").write_bytes(bytes(blob))
        (out / f"{arm}.json").write_text(json.dumps(
            {"name": name, "size": src.stat().st_size, "pieces": index, "extra": extra}))
        print(arm, name, len(index), "pieces", len(blob), "bytes")


def materialize(a) -> None:
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for arm in ("arm1", "arm2"):
        meta = json.loads((src / f"{arm}.json").read_text())
        blob = (src / f"{arm}.blob").read_bytes()
        target = dst / meta["name"]
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        with open(target, "wb") as f:
            f.truncate(meta["size"])
            for off, pos, length in meta["pieces"]:
                f.seek(off)
                f.write(blob[pos : pos + length])
        if meta["extra"]:
            (dst / meta["extra"]["manifest_name"]).write_text(
                json.dumps(meta["extra"]["manifest"], indent=1) + "\n")
        print(arm, target, meta["size"], "logical bytes,", len(meta["pieces"]), "pieces")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    for k in ("--refs", "--sidecar", "--mxfp4", "--out"):
        e.add_argument(k, required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--src", required=True)
    m.add_argument("--dst", required=True)
    a = p.parse_args()
    extract(a) if a.cmd == "extract" else materialize(a)


if __name__ == "__main__":
    main()
