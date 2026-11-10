"""CPU check of the NVFP4-8R loader boundary (model integration, P3).

Expected behaviour comes from the P3 plan, not from the implementation:
- the GGUF loader admits model-loader-extra-config keys nvfp4_8r_packs,
  nvfp4_8r_tally and nvfp4_8r_reserve_bytes, and parses them once into a typed
  config (packs manifest format checked, reserve a positive int);
- a packs key without the tally or the reserve is refused at construction,
  and so is a reserve that is not a positive int (no default budget);
- plan 01 (three-tier): nvfp4_8r_residency is required with the packs keys
  and parses to TwoTier / HotIq / HotIqHost(share); nvfp4_8r_host_route_share
  is required with hot_iq_host, a float in [0, 0.05], and refused with the
  other two residencies or without the packs keys (no defaults);
- with packs configured, the stage's weight map drops every routed-expert
  tensor of the real GGUF copy (adapter's ".ffn.experts." split) and keeps
  every other tensor, shared experts included.
The GPU half (free HBM measured after dense load, packs attached, greedy
identity) is the truncated-model test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader

ROOT = Path(os.environ["LANE"])
GOOD = {
    "nvfp4_8r_packs": str(ROOT / "packs"),
    "nvfp4_8r_tally": str(ROOT / "engine/private/p16-tally.npz"),
    # Per pipeline stage, keyed by the stage's first layer (P9: the stages need
    # different head-room; one number OOMed stage 1 on a 100K-token prompt).
    "nvfp4_8r_reserve_bytes": {0: 5 * 1024**3, 20: 7 * 1024**3},
    "nvfp4_8r_residency": "two_tier",
    "nvfp4_8r_prefill_warm": "native_staged",
}
SHARE = "nvfp4_8r_host_route_share"


def loader(extra: dict) -> GGUFModelLoader:
    return GGUFModelLoader(LoadConfig(load_format="gguf", model_loader_extra_config=extra))


def refused(extra: dict) -> bool:
    try:
        loader(extra)
    except ValueError:
        return True
    return False


def main() -> int:
    fails = []
    ok = loader(GOOD)
    cfg = getattr(ok, "nvfp4_8r", None)
    if cfg is None or cfg.reserve_by_stage != GOOD["nvfp4_8r_reserve_bytes"]:
        fails.append(f"typed config not parsed: {cfg}")
    else:
        # Expected values are the configured numbers themselves (arithmetic).
        if cfg.reserve_for(range(0, 20)) != 5 * 1024**3 or cfg.reserve_for(range(20, 40)) != 7 * 1024**3:
            fails.append("reserve_for does not return the stage's own reserve")
        try:
            cfg.reserve_for(range(10, 30))
            fails.append("a stage with no configured reserve got one")
        except ValueError:
            pass
    # YAML/JSON carry the stage keys as strings: same typed result.
    as_text = loader({**GOOD, "nvfp4_8r_reserve_bytes": {"0": 5 * 1024**3, "20": 7 * 1024**3}}).nvfp4_8r
    if as_text is None or as_text.reserve_by_stage != GOOD["nvfp4_8r_reserve_bytes"]:
        fails.append("string stage keys not parsed to ints")
    for drop in ("nvfp4_8r_tally", "nvfp4_8r_reserve_bytes", "nvfp4_8r_residency",
                 "nvfp4_8r_prefill_warm"):
        if not refused({k: v for k, v in GOOD.items() if k != drop}):
            fails.append(f"packs without {drop} admitted")
    for bad in (0, -1, "6G", 1.5, 6 * 1024**3, {}, {0: 0}, {0: -5}, {"a": 1}, {0: "6G"}, {-1: 5}):
        if not refused({**GOOD, "nvfp4_8r_reserve_bytes": bad}):
            fails.append(f"reserve {bad!r} admitted")
    if not refused({**GOOD, "nvfp4_8r_packs": str(ROOT / "logs")}):
        fails.append("packs folder without manifest admitted")
    if loader({}).nvfp4_8r is not None:
        fails.append("config present without packs key")
    # Residency (plan 01 section 5.2). Expected types are the plan's table.
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

    if cfg is not None and cfg.residency != tiers.TwoTier():
        fails.append(f"two_tier parsed to {cfg.residency!r}")
    for extra, want in (
        ({"nvfp4_8r_residency": "hot_iq"}, tiers.HotIq()),
        ({"nvfp4_8r_residency": "hot_iq_host", SHARE: 0.0}, tiers.HotIqHost(0.0)),
        ({"nvfp4_8r_residency": "hot_iq_host", SHARE: 0.02}, tiers.HotIqHost(0.02)),
    ):
        got = loader({**GOOD, **extra}).nvfp4_8r
        if got is None or got.residency != want or type(got.residency) is not type(want):
            fails.append(f"{extra} parsed to {got and got.residency!r}, want {want!r}")
    for extra in (
        {"nvfp4_8r_residency": "hot_iq_host"},  # no share
        {"nvfp4_8r_residency": "hot_iq_host", SHARE: 0.06},
        {"nvfp4_8r_residency": "hot_iq_host", SHARE: -0.01},
        {"nvfp4_8r_residency": "hot_iq_host", SHARE: "0.0"},
        {"nvfp4_8r_residency": "two_tier", SHARE: 0.0},
        {"nvfp4_8r_residency": "hot_iq", SHARE: 0.0},
        {"nvfp4_8r_residency": "three_tier"},
        {"nvfp4_8r_residency": None},
    ):
        if not refused({**GOOD, **extra}):
            fails.append(f"{extra} admitted")
    if not refused({SHARE: 0.0}):
        fails.append("host route share admitted without the packs keys")
    # Routed-expert filter on a real map shape (names as the V4.1 adapter maps).
    names = {
        "blk.3.ffn_gate_exps.weight": "model.layers.3.ffn.experts.gate_proj.weight",
        "blk.3.ffn_up_exps.weight": "model.layers.3.ffn.experts.up_proj.weight",
        "blk.3.ffn_down_exps.weight": "model.layers.3.ffn.experts.down_proj.weight",
        "blk.3.ffn_gate_shexp.weight": "model.layers.3.ffn.shared_experts.gate_proj.weight",
        "blk.3.ffn_gate_inp.weight": "model.layers.3.ffn.gate.weight",
        "blk.3.attn_q_a.weight": "model.layers.3.attn.wq_a.weight",
    }
    kept = ok.drop_routed_experts(names)
    expected = {k for k in names if "_exps." not in k}
    if set(kept) != expected:
        fails.append(f"filtered map {sorted(kept)} != {sorted(expected)}")
    print("PASS" if not fails else "FAIL", *fails, sep="\n")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
