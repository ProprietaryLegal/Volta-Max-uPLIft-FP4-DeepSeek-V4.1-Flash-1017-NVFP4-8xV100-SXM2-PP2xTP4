"""CPU test of iq_tail_core (plan 06 section 7's one-card speed rule). The
expected values below are worked by hand from the rule's text.

    python test_iq_tail_core.py        (exit 0 PASS, 1 FAIL)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iq_tail_core as core  # noqa: E402


def layer(t: float, f: float, **over) -> dict:
    """A layer whose every cell is torch t, fused f, except the overrides
    {"M6_6": {...}} given as full cells."""
    parts = {"M6": {k: {"t1": t, "f1": f, "f2": f, "t2": t} for k in (0, 1, 3, 6, 12, 16, 36)},
             "M1": {k: {"t1": t / 2, "f1": f / 2, "f2": f / 2, "t2": t / 2} for k in (0, 1, 3, 6)}}
    for key, times in over.items():
        m, k = key.split("_")
        parts[m][int(k)] = times
    return core.layer_tail_verdict(parts)


def main() -> int:
    fails = []

    def want(name: str, got, expected) -> None:
        if got != expected:
            fails.append(f"{name}: got {got!r}, want {expected!r}")

    c = core.cell({"t1": 390.0, "f1": 350.0, "f2": 352.0, "t2": 392.0})
    want("cell means", (c["torch_us"], c["fused_us"], c["saved_us"]), (391.0, 351.0, 40.0))
    want("cell agrees", c["runs_disagree"], [])
    # 3 % of 400 is 12: 388 against 412 differ by 24.
    want("cell disagrees", len(core.cell({"t1": 388.0, "f1": 350.0, "f2": 350.0, "t2": 412.0})["runs_disagree"]), 1)
    # The floor: 3 % of 20 is 0.6, the floor 2.05: 19 against 21 agree, 18.9 against 21 do not.
    want("floor agrees", core.cell({"t1": 19.0, "f1": 19.0, "f2": 21.0, "t2": 21.0})["runs_disagree"], [])
    want("floor disagrees", len(core.cell({"t1": 18.9, "f1": 20.0, "f2": 20.0, "t2": 21.0})["runs_disagree"]), 1)
    try:
        core.cell({"t1": 1.0, "f1": 1.0, "f2": 1.0})
        fails.append("a cell with three arms was taken")
    except ValueError:
        pass

    good, short = layer(390.0, 350.0), layer(390.0, 365.0)
    want("good layer", (good["settled"], good["s1_holds"], good["s2_slower"], good["saved_k6_us"]),
         (True, True, [], 40.0))
    want("exactly 30 holds", layer(390.0, 360.0)["s1_holds"], True)
    want("29.9 fails", layer(390.0, 360.1)["s1_holds"], False)
    slow = layer(390.0, 350.0, M1_0={"t1": 100.0, "f1": 102.5, "f2": 102.5, "t2": 100.0})
    want("slower by 2.5", len(slow["s2_slower"]), 1)
    want("slower by 2.0 passes", layer(390.0, 350.0, M1_0={"t1": 100.0, "f1": 102.0, "f2": 102.0,
                                                           "t2": 100.0})["s2_slower"], [])
    noisy = layer(390.0, 350.0, M6_12={"t1": 380.0, "f1": 350.0, "f2": 350.0, "t2": 420.0})
    want("noisy layer unsettled", noisy["settled"], False)

    v = core.decide_tail({"L05": good, "L12": good, "L27": good})
    want("three good", v["verdict"], "BUILD ACCEPTED ON ONE CARD")
    want("two good, one short", core.decide_tail({"L05": good, "L12": good, "L27": short})["verdict"],
         "BUILD ACCEPTED ON ONE CARD")
    want("two short", core.decide_tail({"L05": good, "L12": short, "L27": short})["verdict"], "NOT ACCEPTED")
    want("one slower", core.decide_tail({"L05": good, "L12": good, "L27": slow})["verdict"], "NOT ACCEPTED")
    want("two good, one noisy", core.decide_tail({"L05": good, "L12": good, "L27": noisy})["verdict"],
         "BUILD ACCEPTED ON ONE CARD")
    want("one good, two noisy", core.decide_tail({"L05": good, "L12": noisy, "L27": noisy})["verdict"],
         "INCONCLUSIVE")
    want("one good, one short, one noisy",
         core.decide_tail({"L05": good, "L12": short, "L27": noisy})["verdict"], "INCONCLUSIVE")
    noisy_slow = layer(390.0, 350.0, M6_12={"t1": 380.0, "f1": 350.0, "f2": 350.0, "t2": 420.0},
                       M1_0={"t1": 100.0, "f1": 110.0, "f2": 110.0, "t2": 100.0})
    want("a slower cell in an unsettled layer gives no input",
         core.decide_tail({"L05": good, "L12": good, "L27": noisy_slow})["verdict"],
         "BUILD ACCEPTED ON ONE CARD")
    print("PASS" if not fails else "FAIL: " + "; ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
