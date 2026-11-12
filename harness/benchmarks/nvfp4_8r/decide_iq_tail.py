"""Plan 06: the one-card verdict from the three layers' bench_iq_tail
receipts, by plan 06 section 7 (iq_tail_core.decide_tail). CPU only, the
standard library only.

    python decide_iq_tail.py OUT.json L05.json L12.json L27.json

Exit 0 when a verdict is written (BUILD ACCEPTED ON ONE CARD or NOT ACCEPTED),
2 when the verdict is INCONCLUSIVE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iq_tail_core as core  # noqa: E402

K6 = (0, 1, 3, 6, 12, 16, 36)
K1 = (0, 1, 3, 6)


def main() -> int:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out, files = Path(sys.argv[1]), [Path(f) for f in sys.argv[2:]]
    layers: dict[str, dict] = {}
    swiglu = set()
    for f in files:
        r = json.loads(f.read_text())
        if r["bench"] != "bench_iq_tail" or r["group_plan"] != "kernel":
            raise ValueError(f"{f}: not a bench_iq_tail receipt on the kernel plan")
        name = f"L{r['layer']:02d}"
        if name in layers:
            raise ValueError(f"{f}: layer {name} given twice")
        swiglu.add(r["swiglu"])
        parts = {mk: {int(k): {n: v["median_us"] for n, v in row.items()} for k, row in rows.items()}
                 for mk, rows in r["parts_us"].items()}
        if tuple(sorted(parts.get("M6", {}))) != K6 or tuple(sorted(parts.get("M1", {}))) != K1:
            raise ValueError(f"{f}: k values {sorted(parts.get('M6', {}))} / {sorted(parts.get('M1', {}))}")
        # Recomputed from the timings, not read from the receipt.
        layers[name] = core.layer_tail_verdict(parts)
    if len(swiglu) != 1:
        raise ValueError(f"the receipts mix SwiGLU modes: {sorted(swiglu)}")
    res = {"receipts": [str(f) for f in files], "swiglu": swiglu.pop(), "layers": layers,
           **core.decide_tail(layers)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in res if k not in ("layers", "receipts")}))
    return 2 if res["verdict"] == "INCONCLUSIVE" else 0


if __name__ == "__main__":
    sys.exit(main())
