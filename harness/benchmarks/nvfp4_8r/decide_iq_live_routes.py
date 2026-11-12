"""Card 8.5: the decision from the three layers' bench_iq_live_routes
receipts, by DECISION-RULE.md (iq_live_core.decide). CPU only, numpy only.

    python decide_iq_live_routes.py OUT.json L05.json L12.json L27.json

Exit 0 when a verdict is written (BUILD WARRANTED or NO BUILD), 2 when the
verdict is INCONCLUSIVE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iq_live_core as core  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out, files = Path(sys.argv[1]), [Path(f) for f in sys.argv[2:]]
    layers: dict[str, dict] = {}
    for f in files:
        r = json.loads(f.read_text())
        if r["bench"] != "bench_iq_live_routes" or r["group_plan"] != "kernel":
            raise ValueError(f"{f}: not a bench_iq_live_routes receipt on the kernel plan")
        name = f"L{r['layer']:02d}"
        if name in layers:
            raise ValueError(f"{f}: layer {name} given twice")
        parts = {int(k): v for k, v in r["parts_us"]["M6"].items()}
        if tuple(sorted(parts)) != core.K_VALUES:
            raise ValueError(f"{f}: k values {sorted(parts)}, want {core.K_VALUES}")
        # Recomputed from the timings, not read from the receipt.
        layers[name] = core.layer_verdict(parts)
    res = {"receipts": [str(f) for f in files], "layers": layers, **core.decide(layers)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in res if k not in ("layers", "receipts")}))
    return 2 if res["verdict"] == "INCONCLUSIVE" else 0


if __name__ == "__main__":
    sys.exit(main())
