"""P4 arm 2 file check (CPU): the repacked MXFP4 Engram table holds, at the
offsets its manifest declares, exactly the checkpoint rows sampled by
make_p4_refs (codes, E8M0 scales, zero padding), and the table sizes equal the
checkpoint's. Expected bytes come from the raw checkpoint (the refs file),
not from the packer. Exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE (file absent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--table-rows", type=int, nargs=2, default=(384006168, 384016682))
    a = p.parse_args()
    if not a.manifest.is_file():
        print(f"INCONCLUSIVE: no manifest {a.manifest}")
        return 2
    m = json.loads(a.manifest.read_text())
    fails = []
    if m["format"] != "nvfp4-8r-engram-mxfp4-144-v1" or m["row_bytes"] != 144:
        fails.append(f"format {m['format']} row_bytes {m['row_bytes']}")
    binary = a.manifest.parent / m["file"]
    layers = [t["layer"] for t in m["tables"]]
    if layers != [1, 14] or [t["rows"] for t in m["tables"]] != list(a.table_rows):
        fails.append(f"tables {m['tables']}")
    if m["tables"][0]["offset"] != 0 or m["tables"][1]["offset"] != a.table_rows[0] * 144:
        fails.append("table offsets are not contiguous from 0")
    if binary.stat().st_size != sum(a.table_rows) * 144:
        fails.append(f"file size {binary.stat().st_size}")
    checked = 0
    with np.load(a.refs, allow_pickle=False) as refs:
        fd = os.open(binary, os.O_RDONLY)
        try:
            for t in m["tables"]:
                layer = t["layer"]
                for i, r in enumerate(refs[f"rows_L{layer}"]):
                    row = np.frombuffer(os.pread(fd, 144, t["offset"] + int(r) * 144), np.uint8)
                    if not (np.array_equal(row[:128], refs[f"mx_codes_L{layer}"][i])
                            and np.array_equal(row[128:136], refs[f"mx_scales_L{layer}"][i])
                            and not row[136:].any()):
                        fails.append(f"L{layer} row {int(r)} differs")
                    checked += 1
        finally:
            os.close(fd)
    print(json.dumps({"status": "PASS" if not fails else "FAIL", "rows_checked": checked,
                      "failures": fails[:20]}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
