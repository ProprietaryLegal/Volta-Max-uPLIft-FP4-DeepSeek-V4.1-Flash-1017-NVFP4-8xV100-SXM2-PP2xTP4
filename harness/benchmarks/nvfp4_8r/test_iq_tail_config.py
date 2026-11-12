"""CPU checks of the IQ tail mode (plan 06): the key, the typed config, the
wiring, the untouched reference tail and prefill, profiles and the bench tree.

    python -m benchmarks.nvfp4_8r.test_iq_tail_config --base-native BEFORE.py \
        --packs PACKS --tally-file ANY_FILE --bench-dir engine/src/benchmarks/nvfp4_8r \
        (--run-dir run | --no-profiles) --output OUT.json

--base-native is nvfp4_8r_native.py before the plan 06 patch: forward_prefill,
_run_cold, _iq_routes, _iq_partial and the three statements of forward's torch
tail must be unchanged against it. --packs is a pack directory with a
manifest and --tally-file any existing file (from_extra checks only that both
exist; neither is read past the manifest). --run-dir checks every
serve-nvfp4-pp2*.yaml there; --no-profiles says the profiles are not checked
here and the receipt records that. Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import json
import re
import sys
from pathlib import Path

import yaml

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_iq_tail as iqt
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

KEY = "nvfp4_8r_iq_tail"
KEY_LINE = re.compile(r"^  nvfp4_8r_iq_tail: (torch|fused)$", re.M)
ANY_KEY_LINE = re.compile(r"^\s*nvfp4_8r_iq_tail\s*:", re.M)
CHAIN, KERNEL = gplan.GroupPlan.CHAIN, gplan.GroupPlan.KERNEL
WARM_RESIDENCIES = (tiers.HotIq(), tiers.HotIqHost(0.0))
# (value, group plan) -> admitted with a WARM tier. Written from plan 06 part D.
WANT = {
    ("torch", CHAIN): True, ("torch", KERNEL): True,
    ("fused", KERNEL): True, ("fused", CHAIN): False,
}
BAD_VALUES = ("FUSED", "Torch", "kernel", "chain", "", None, True, 1, ["fused"], {"fused": 1})


def refused(fn) -> bool:
    try:
        fn()
    except (ValueError, TypeError):
        return True
    return False


def method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise LookupError(f"{cls}.{name}")


def calls(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            yield name, n


def torch_tail(forward: ast.FunctionDef) -> list[str]:
    """The reference tail's three statements in forward."""
    out = []
    for node in ast.walk(forward):
        text = ast.unparse(node) if isinstance(node, (ast.Assign, ast.Expr, ast.Return)) else ""
        if isinstance(node, ast.Assign) and "_iq_routes" in text:
            out.append(ast.dump(node))
        if isinstance(node, ast.Expr) and "_mtp_weighted_reduce" in text and "w_native" in text:
            out.append(ast.dump(node))
        if isinstance(node, ast.Return) and "_iq_partial" in text:
            out.append(ast.dump(node))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-native", type=Path, required=True)
    p.add_argument("--packs", type=Path, required=True)
    p.add_argument("--tally-file", type=Path, required=True)
    p.add_argument("--bench-dir", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path)
    g.add_argument("--no-profiles", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fails: list[str] = []
    res: dict = {"check": "iq_tail_config"}

    # 1. the key: required with a WARM tier, refused with two_tier, two values, no default
    if iqt.IQ_TAIL_KEY != KEY:
        fails.append(f"the key is {iqt.IQ_TAIL_KEY!r}")
    if sorted(m.value for m in iqt.IqTail) != ["fused", "torch"]:
        fails.append(f"IqTail has {[m.value for m in iqt.IqTail]}")
    for residency in WARM_RESIDENCIES:
        for plan in (CHAIN, KERNEL):
            if not refused(lambda r=residency, pl=plan: iqt.parse_iq_tail({}, r, pl)):
                fails.append(f"{residency!r} {plan.value}: the key absent was admitted (a default)")
            for bad in BAD_VALUES:
                if not refused(lambda r=residency, pl=plan, b=bad: iqt.parse_iq_tail({KEY: b}, r, pl)):
                    fails.append(f"{residency!r} {plan.value}: {KEY}={bad!r} admitted")
        for (value, plan), admitted in WANT.items():
            try:
                got = iqt.parse_iq_tail({KEY: value}, residency, plan)
            except ValueError as exc:
                if admitted:
                    fails.append(f"{residency!r} {plan.value} {value}: refused: {exc}")
                continue
            if not admitted:
                fails.append(f"{residency!r} {plan.value} {value}: admitted")
            elif got is not iqt.IqTail(value):
                fails.append(f"{residency!r} {plan.value} {value}: parsed to {got!r}")
    for plan in (CHAIN, KERNEL):
        try:
            if iqt.parse_iq_tail({}, tiers.TwoTier(), plan) is not None:
                fails.append("two_tier without the key did not parse to None")
        except ValueError as exc:
            fails.append(f"two_tier without the key refused: {exc}")
        for value in ("torch", "fused"):
            if not refused(lambda v=value, pl=plan: iqt.parse_iq_tail({KEY: v}, tiers.TwoTier(), pl)):
                fails.append(f"two_tier with {KEY}={value} admitted")
    if not refused(lambda: iqt.check_iq_tail("fused", tiers.HotIq(), KERNEL)):
        fails.append("check_iq_tail took a bare string")
    if not refused(lambda: iqt.check_iq_tail(iqt.IqTail.TORCH, tiers.HotIq(), "kernel")) or \
            not refused(lambda: iqt.check_iq_tail(iqt.IqTail.FUSED, tiers.HotIq(), "kernel")):
        fails.append("check_iq_tail took a bare string as the group plan with fused")
    if KEY not in nat.CONDITIONAL_KEYS or KEY in nat.REQUIRED_KEYS or KEY in nat.EXTRA_KEYS:
        fails.append(f"{KEY} must be a conditional key only")
    module_src = Path(iqt.__file__).read_text()
    for banned in (".get(", "setdefault", "os.environ", "getenv"):
        if banned in module_src:
            fails.append(f"nvfp4_8r_iq_tail.py uses {banned}")

    # 2. the typed config
    base = (Path("/nonexistent/packs"), nat.TallySource(Path("/nonexistent/tally")), {0: 1})
    def cfg(residency, plan, *tail):
        return nat.NativePackConfig(*base, residency, tiers.PrefillWarm.NATIVE_STAGED, plan,
                                    swig.Swiglu.CHAIN, *tail, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)
    for residency in WARM_RESIDENCIES:
        for (value, plan), admitted in WANT.items():
            mode = iqt.IqTail(value)
            if admitted:
                try:
                    c = cfg(residency, plan, mode)
                    if c.iq_tail is not mode or c.warm_iq_tail() is not mode:
                        fails.append(f"NativePackConfig({residency!r}, {plan.value}, {value}) lost the tail")
                except (ValueError, TypeError) as exc:
                    fails.append(f"NativePackConfig({residency!r}, {plan.value}, {value}) refused: {exc}")
            elif not refused(lambda r=residency, pl=plan, m=mode: cfg(r, pl, m)):
                fails.append(f"NativePackConfig({residency!r}, {plan.value}, {value}) constructed")
        if not refused(lambda r=residency: cfg(r, KERNEL, None)):
            fails.append(f"NativePackConfig({residency!r}) constructed with no tail")
        if not refused(lambda r=residency: cfg(r, KERNEL, "fused")):
            fails.append(f"NativePackConfig({residency!r}) took a bare string as the tail")
        if not refused(lambda r=residency: cfg(r, KERNEL)):
            fails.append(f"NativePackConfig({residency!r}) constructed without the tail argument")
    try:
        two = cfg(tiers.TwoTier(), KERNEL, None)
        if two.iq_tail is not None:
            fails.append("two_tier config holds a tail")
        if not refused(two.warm_iq_tail):
            fails.append("two_tier config gave a tail to warm_iq_tail()")
    except (ValueError, TypeError) as exc:
        fails.append(f"NativePackConfig(TwoTier, None) refused: {exc}")
    for mode in iqt.IqTail:
        if not refused(lambda m=mode: cfg(tiers.TwoTier(), KERNEL, m)):
            fails.append(f"NativePackConfig(TwoTier, {mode.value}) constructed")

    # 3. from_extra, the loader's entry
    extra = {"nvfp4_8r_packs": str(a.packs), "nvfp4_8r_tally": str(a.tally_file),
             "nvfp4_8r_reserve_bytes": {0: 1, 20: 1}, "nvfp4_8r_prefill_warm": "native_staged",
             "nvfp4_8r_swiglu": "chain", "nvfp4_8r_prefill_swiglu": "chain", "nvfp4_8r_prefill_rb": "rb2"}
    res_keys = {"hot_iq": {"nvfp4_8r_residency": "hot_iq"},
                "hot_iq_host": {"nvfp4_8r_residency": "hot_iq_host", "nvfp4_8r_host_route_share": 0.0}}
    n_extra = 0
    for rname, rkeys in res_keys.items():
        for (value, plan), admitted in WANT.items():
            doc = {**extra, **rkeys, "nvfp4_8r_group_plan": plan.value, KEY: value}
            n_extra += 1
            try:
                c = nat.NativePackConfig.from_extra(doc)
            except ValueError as exc:
                if admitted:
                    fails.append(f"from_extra {rname} {plan.value} {value}: refused: {exc}")
                continue
            if not admitted:
                fails.append(f"from_extra {rname} {plan.value} {value}: admitted")
            elif c is None or c.iq_tail is not iqt.IqTail(value):
                fails.append(f"from_extra {rname} {plan.value} {value}: parsed to {c and c.iq_tail!r}")
        doc = {**extra, **rkeys, "nvfp4_8r_group_plan": "kernel"}
        if not refused(lambda d=doc: nat.NativePackConfig.from_extra(d)):
            fails.append(f"from_extra {rname} without {KEY} admitted (a default)")
    two_doc = {**extra, "nvfp4_8r_residency": "two_tier", "nvfp4_8r_group_plan": "kernel"}
    try:
        c = nat.NativePackConfig.from_extra(two_doc)
        if c is None or c.iq_tail is not None:
            fails.append("from_extra two_tier did not parse to no tail")
    except ValueError as exc:
        fails.append(f"from_extra two_tier refused: {exc}")
    if not refused(lambda: nat.NativePackConfig.from_extra({**two_doc, KEY: "torch"})):
        fails.append(f"from_extra two_tier with {KEY} admitted")
    if not refused(lambda: nat.NativePackConfig.from_extra({KEY: "torch"})):
        fails.append(f"{KEY} without the pack keys admitted")
    res["from_extra_documents"] = n_extra

    # 4. signatures: required, no default
    prm = inspect.signature(nat.NativeV41Moe.three_tier).parameters.get("iq_tail")
    if prm is None or prm.kind is not inspect.Parameter.KEYWORD_ONLY or prm.default is not inspect._empty:
        fails.append(f"three_tier: iq_tail is not a required keyword ({prm})")
    fields = {f.name: f for f in dataclasses.fields(nat.WarmTier)}
    f = fields.get("tail")
    if f is None or f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
        fails.append("WarmTier.tail is absent or has a default")
    f = {f.name: f for f in dataclasses.fields(nat.NativePackConfig)}.get("iq_tail")
    if f is None or f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
        fails.append("NativePackConfig.iq_tail is absent or has a default")
    for fn in (iqt.parse_iq_tail, iqt.check_iq_tail):
        if any(q.default is not inspect._empty for q in inspect.signature(fn).parameters.values()):
            fails.append(f"{fn.__name__} has a default argument")

    # 5. wiring and the untouched reference code, by source
    new, old = ast.parse(Path(nat.__file__).read_text()), ast.parse(a.base_native.read_text())
    for name in ("forward_prefill", "_run_cold", "_iq_routes", "_iq_partial", "_tier_groups",
                 "_split_groups", "_groups", "_compact"):
        if ast.dump(method(new, "NativeV41Moe", name)) != ast.dump(method(old, "NativeV41Moe", name)):
            fails.append(f"NativeV41Moe.{name} changed")
    tn, tb = (torch_tail(method(t, "NativeV41Moe", "forward")) for t in (new, old))
    if len(tb) != 3 or tn != tb:
        fails.append(f"forward's torch tail changed ({len(tn)} statements, the base has {len(tb)})")
    # The module never picks a mode: IqTail members appear in native only as the right side of "is".
    members = [n for n in ast.walk(new) if isinstance(n, ast.Attribute) and n.attr in ("TORCH", "FUSED")
               and "IqTail" in ast.unparse(n)]
    compared = {id(c) for n in ast.walk(new) if isinstance(n, ast.Compare)
                and all(isinstance(op, ast.Is) for op in n.ops) for c in n.comparators}
    stray = [ast.unparse(m) for m in members if id(m) not in compared]
    if stray or not members:
        fails.append(f"native names IqTail members outside an 'is' test: {stray} (of {len(members)})")
    build = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "build_layer")
    passed = [k for _, c in calls(build) for k in c.keywords
              if k.arg == "iq_tail" and ast.unparse(k.value) == "cfg.warm_iq_tail()"]
    if len(passed) != 1:
        fails.append(f"build_layer passes cfg.warm_iq_tail() {len(passed)} times, want 1")

    # 6. the kernel's source form: the upstream reduce's loop and one add
    ktree = ast.parse(module_src)
    kern = next(n for n in ktree.body if isinstance(n, ast.FunctionDef) and n.name == "_iq_tail_kernel")
    text = ast.unparse(kern)
    for need in ("tl.static_range(0, TOP_K_C)", "acc += values.to(tl.float32) * weight",
                 "tl.where(iq < 0, weight, 0.0)", "acc + part.to(tl.float32)"):
        if need not in text:
            fails.append(f"the tail kernel lacks: {need}")
    for banned in ("libdevice", "tl.fma", "fast", "tl.sum(", "tl.float64"):
        if banned in text:
            fails.append(f"the tail kernel uses {banned}")

    # 7. profiles
    if a.no_profiles:
        res["profiles"] = "NOT CHECKED (--no-profiles)"
    else:
        profs = sorted(a.run_dir.glob("serve-nvfp4-pp2*.yaml"))
        res["profiles"] = len(profs)
        if not profs:
            fails.append(f"no serve-nvfp4-pp2*.yaml in {a.run_dir}")
        n_warm = 0
        for f in profs:
            text = f.read_text()
            doc = (yaml.safe_load(text) or {}).get("model-loader-extra-config", {})
            try:
                residency = tiers.parse_residency(doc)
                plan = gplan.parse_group_plan(doc)
            except ValueError as exc:
                fails.append(f"{f.name}: {exc}")
                continue
            lines, any_lines = KEY_LINE.findall(text), ANY_KEY_LINE.findall(text)
            if isinstance(residency, tiers.TwoTier):
                if any_lines or KEY in doc:
                    fails.append(f"{f.name}: two_tier with an {KEY} line")
                continue
            n_warm += 1
            if len(lines) != 1 or len(any_lines) != 1:
                fails.append(f"{f.name}: {KEY} lines {lines} (of {len(any_lines)} naming the key)")
                continue
            try:
                iqt.parse_iq_tail(doc, residency, plan)
            except ValueError as exc:
                fails.append(f"{f.name}: {exc}")
        res["profiles_with_a_warm_tier"] = n_warm

    # 8. the bench tree: every three_tier call and typed config of the current module names the tail
    n_bench = 0
    for f in sorted(a.bench_dir.glob("*.py")):
        if f.name == Path(__file__).name:
            continue  # this file's own calls pass the tail as *tail, with and without it
        tree = ast.parse(f.read_text())
        for name, call in calls(tree):
            kws = {k.arg for k in call.keywords}
            seg = ast.unparse(call)
            starred = any(k.arg is None for k in call.keywords)
            if name == "three_tier" and "group_plan" in kws:
                n_bench += 1
                if "iq_tail" not in kws:
                    fails.append(f"{f.name}:{call.lineno} three_tier without iq_tail")
            elif name == "three_tier" and starred and "before" not in seg:
                n_bench += 1
                if "iq_tail" not in f.read_text():
                    fails.append(f"{f.name}:{call.lineno} three_tier(**...) in a file that never names iq_tail")
            if name == "NativePackConfig":
                # D3 and RB 4: the call as G1 read it, without the prefill_swiglu and prefill_rb keywords.
                seg = ast.unparse(ast.Call(call.func, call.args,
                                           [k for k in call.keywords if k.arg not in ("prefill_swiglu", "prefill_rb")]))
            if name == "NativePackConfig" and "Swiglu" in seg:
                n_bench += 1
                if "IqTail" not in seg and not seg.rstrip(")").rstrip().endswith("None"):
                    fails.append(f"{f.name}:{call.lineno} NativePackConfig without the tail")
    res["bench_calls_checked"] = n_bench
    res["failures"] = fails
    res["status"] = "FAIL" if fails else "PASS"
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": res["status"], "profiles": res["profiles"],
                      "bench_calls_checked": n_bench, "failures": fails[:10]}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
