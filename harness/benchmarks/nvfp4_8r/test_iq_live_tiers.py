"""CPU test: the bench's routings against the product's tier module.

    python test_iq_live_tiers.py PATH/nvfp4_8r_tiers.py     (numpy only)

The product's slot_maps must accept the bench's tier array as a hot_iq layer
(96 WARM, 288 HOT, no HOST), and the count of routes the product's IQ slot
map calls live must be the k the routing was asked for.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iq_live_core as core  # noqa: E402


def main() -> int:
    spec = importlib.util.spec_from_file_location("nvfp4_8r_tiers", sys.argv[1])
    tiers = importlib.util.module_from_spec(spec)
    sys.modules["nvfp4_8r_tiers"] = tiers
    spec.loader.exec_module(tiers)
    fails = []
    if (int(tiers.Tier.HOT), int(tiers.Tier.WARM)) != (core.HOT, core.WARM):
        fails.append("the bench's HOT/WARM codes differ from tiers.Tier")
    tier = core.bench_tier()
    maps = tiers.slot_maps(tier)
    if int((maps.iq >= 0).sum()) != 96 or int((maps.hot >= 0).sum()) != 288:
        fails.append("slot_maps does not see 96 WARM and 288 HOT experts")
    if int((maps.host >= 0).sum()) != 0:
        fails.append("slot_maps sees a HOST expert")
    if sorted(maps.iq[maps.iq >= 0].tolist()) != list(range(96)):
        fails.append("the IQ slots are not 0..95")
    for m, ks in ((6, core.K_VALUES), (1, (0, 1, 3, 6))):
        for k in ks:
            ids = core.live_routes(k, m, tier, 7 * k + m)
            live = int((maps.iq[ids] >= 0).sum())
            hot = int((maps.hot[ids] >= 0).sum())
            if (live, hot) != (k, 6 * m - k):
                fails.append(f"k = {k}, M = {m}: the product maps see {live} live, {hot} HOT")
    for j in (1, 2, 6):
        ids = core.shared_routes(j, 6, tier, j)
        slots = maps.iq[ids]
        if int((slots >= 0).sum()) != 6 * j or len(set(slots[slots >= 0].tolist())) != j:
            fails.append(f"shared j = {j}: live routes or distinct experts are wrong")
    if fails:
        print("FAIL")
        for f in fails:
            print(" ", f)
        return 1
    print("PASS test_iq_live_tiers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
