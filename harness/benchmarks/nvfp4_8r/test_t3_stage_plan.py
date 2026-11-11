"""Plan 01 step 7 (CPU): what the loader hands to the three-tier build.

Expected values come from outside the functions under test:
- routed-expert capture: over the real GGUF copy's tensor index, mapped with
  the V4.1 adapter's own name tables, every layer L must yield the GGUF names
  blk.L.ffn_{gate,up,down}_exps.weight as w1/w3/w2 (the GGUF naming);
- IQ record bytes per layer: 11520 x (2 x bb13 + bb2) from the types in the
  GGUF header (bb 56/66/74 for GGML 29/16/17), computed here;
- stage plan: budget = least free HBM - the stage's configured reserve
  (arithmetic on the profile's numbers); the per-tier share of decode routes
  is recomputed here from the tally and the tier arrays (aggregates only);
- two_tier: the tier array the loader now passes to build_layer must give the
  slot maps split_maps gave the P3 build, for every layer of both stages;
- digests: four equal digests pass, any other list is refused.
Prints and saves aggregates only (the tally is private).

Exit 0 PASS, 1 FAIL, 3 INCONCLUSIVE (the missing input is named).
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ["LANE"])
GGUF = ROOT / "models/gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"
TALLY = ROOT / "engine/private/p16-tally.npz"
BB = {29: 56, 16: 66, 17: 74}
# The serving profile's reserves and the measured free HBM before experts
# (run/serve-nvfp4-pp2.yaml; run/state/p6-boot2.log: 29.30 / 28.86 GiB).
RESERVE = {0: 4831838208, 20: 8053063680}
FREE = {0: int(29.30 * 2**30), 20: int(28.86 * 2**30)}
HOT, WARM, HOST = 0, 1, 2


def raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()

    def finish(status: str, detail: dict) -> int:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({"check": "T3-step7-stage-plan", "status": status,
                                        **detail}, indent=1) + "\n")
        print(status, json.dumps(detail))
        return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[status]

    for need in (GGUF, TALLY, ROOT / "packs/manifest.json"):
        if not need.is_file():
            return finish("INCONCLUSIVE", {"missing": str(need)})
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as native
    from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers
    from vllm.model_executor.model_loader.gguf_adapters import deepseek_v41 as adapter
    from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index

    fails: list[str] = []
    index = gguf_tensor_index(gguf_shard_paths(GGUF))
    name_map = {}
    for raw in index:
        if raw in adapter._GLOBAL:
            name_map[raw] = adapter._GLOBAL[raw]
        else:
            _, layer, suffix = raw.split(".", 2)
            name_map[raw] = f"model.layers.{layer}.{adapter._LAYER[suffix]}"

    def layer_of(prefix: str) -> int:  # the loader asks the module for layer_id
        return int(prefix.split(".")[2])

    # 1. Capture of the routed-expert tensors before the loader drops them.
    want = {
        layer: {"w1": f"blk.{layer}.ffn_gate_exps.weight", "w3": f"blk.{layer}.ffn_up_exps.weight",
                "w2": f"blk.{layer}.ffn_down_exps.weight"}
        for layer in range(40)
    }
    got = native.routed_expert_shards(name_map, layer_of)
    if got != want:
        fails.append("routed-expert capture differs from the GGUF names of layers 0..39")
    stage1 = {raw: name for raw, name in name_map.items()
              if not raw.startswith("blk.") or int(raw.split(".")[1]) >= 20}
    if native.routed_expert_shards(stage1, layer_of) != {k: v for k, v in want.items() if k >= 20}:
        fails.append("stage-1 capture is not layers 20..39")
    short = {raw: name for raw, name in name_map.items() if raw != "blk.7.ffn_up_exps.weight"}
    if not raises(lambda: native.routed_expert_shards(short, layer_of)):
        fails.append("a layer without its up shard was admitted")
    if native.routed_expert_shards({r: n for r, n in name_map.items() if "_exps." not in r},
                                   layer_of) != {}:
        fails.append("a map without routed experts gave shards")

    # 2. IQ sources and record bytes from the header types.
    sources = native.iq_layer_sources(index, got, 0)
    iq_bytes = {}
    for layer in range(40):
        t13 = int(index[f"blk.{layer}.ffn_gate_exps.weight"].tensor_type)
        t2 = int(index[f"blk.{layer}.ffn_down_exps.weight"].tensor_type)
        iq_bytes[layer] = 11520 * (2 * BB[t13] + BB[t2])
        s = sources.get(layer)
        if s is None or (s.type13, s.type2, s.record_bytes) != (t13, t2, iq_bytes[layer]):
            fails.append(f"L{layer}: IQ source types / record bytes differ from the header")
    if not raises(lambda: native.iq_layer_sources(index, got, 4)):
        fails.append("TP rank 4 admitted")

    # 3. Stage plans on the real tally.
    base = {"nvfp4_8r_packs": str(ROOT / "packs"), "nvfp4_8r_tally": str(TALLY),
            "nvfp4_8r_reserve_bytes": RESERVE, "nvfp4_8r_prefill_warm": "native_staged",
            "nvfp4_8r_group_plan": "chain"}
    cfg = native.NativePackConfig.from_extra(
        {**base, "nvfp4_8r_residency": "hot_iq_host", "nvfp4_8r_host_route_share": 0.0})
    two = native.NativePackConfig.from_extra({**base, "nvfp4_8r_residency": "two_tier"})
    with np.load(TALLY, allow_pickle=False) as data:
        phases = [str(p) for p in data["phases"]]
        counts = np.asarray(data["counts"][phases.index("teacher")])
    report = {}
    for first in (0, 20):
        layers = range(first, first + 20)
        stage = native.plan_tiers(cfg, list(layers), FREE[first],
                                  {layer: iq_bytes[layer] for layer in layers}, 0)
        budget = FREE[first] - RESERVE[first]
        plan = stage.plan
        if (stage.free_bytes, stage.reserve_bytes, plan.budget_bytes) != (
            FREE[first], RESERVE[first], budget
        ):
            fails.append(f"stage {first}: budget is not free - reserve")
        if plan.layers != layers or plan.residency != tiers.HotIqHost(0.0):
            fails.append(f"stage {first}: plan covers {plan.layers} / {plan.residency}")
        total = int(counts[first:first + 20].sum())
        share = [
            sum(int(counts[layer][plan.tiers[layer] == t].sum()) for layer in layers) / total
            for t in (HOT, WARM, HOST)
        ]
        if any(abs(g - w) > 1e-12 for g, w in zip(stage.route_share, share)):
            fails.append(f"stage {first}: route shares differ from the tally recount")
        if share[2] != 0.0 or abs(sum(share) - 1.0) > 1e-9:
            fails.append(f"stage {first}: HOST share {share[2]} at share limit 0.0")
        if plan.hbm_bytes > budget:
            fails.append(f"stage {first}: plan uses more than the budget")
        report[f"layers {first}-{first + 19}"] = {
            "budget_bytes": budget, "hbm_bytes": plan.hbm_bytes,
            "experts": {"hot": plan.count(tiers.Tier.HOT), "warm": plan.count(tiers.Tier.WARM),
                        "host": plan.count(tiers.Tier.HOST)},
            "route_share": [round(s, 6) for s in share],
        }
        # 4. two_tier: the tier array handed to build_layer gives split_maps' slots.
        old = native.plan_stage(two, list(layers), FREE[first])
        for layer in layers:
            hot, cold = old.residency[layer]
            maps = tiers.slot_maps(native.two_tier_array(hot, cold))
            hot_slot, cold_slot = native.split_maps(hot, cold, torch.device("cpu"))
            if not (np.array_equal(maps.hot, hot_slot.numpy())
                    and np.array_equal(maps.cold, cold_slot.numpy())
                    and not (maps.iq >= 0).any()):
                fails.append(f"L{layer}: two-tier tier array does not give split_maps' slots")
    if not raises(lambda: native.plan_tiers(two, list(range(20)), FREE[0],
                                            {layer: iq_bytes[layer] for layer in range(20)}, 0)):
        fails.append("plan_tiers admitted a two_tier config")
    if not raises(lambda: native.plan_tiers(cfg, [0, 1, 3], FREE[0],
                                            {layer: iq_bytes[layer] for layer in (0, 1, 3)}, 0)):
        fails.append("plan_tiers admitted non-contiguous layers")

    # 5. Rank symmetry guard.
    native.require_one_digest(["ab"] * 4)
    for bad in (["ab", "ab", "ab", "ac"], ["ab"] * 3, []):
        if not raises(lambda bad=bad: native.require_one_digest(bad)):
            fails.append(f"digests {bad} admitted")
    return finish("FAIL" if fails else "PASS", {"stages": report, "failures": fails})


if __name__ == "__main__":
    sys.exit(main())
