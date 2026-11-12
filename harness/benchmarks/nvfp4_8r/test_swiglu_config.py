"""CPU checks of the swiglu mode (CAMPAIGN 8.4): config, wiring, prefill
untouched, kernel source form, profiles and the bench tree.

    python -m benchmarks.nvfp4_8r.test_swiglu_config --base-native BEFORE.py \
        --run-dir run --bench-dir engine/src/benchmarks/nvfp4_8r --output OUT.json

--base-native is nvfp4_8r_native.py before the swiglu patch: forward_prefill,
_run_cold and the chain lines of forward must be unchanged against it.
Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import sys
from pathlib import Path

import yaml

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

KEY_LINE = re.compile(r"^  nvfp4_8r_swiglu: (chain|kernel)$", re.M)


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


def chain_stmts(forward: ast.FunctionDef) -> list[str]:
    """The three chain statements of forward (g =, u =, inter copy_)."""
    out = []
    for node in ast.walk(forward):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("g", "u"):
            out.append(ast.dump(node))
        if isinstance(node, ast.Expr) and "copy_" in ast.dump(node) and "silu" in ast.dump(node):
            out.append(ast.dump(node))
    return out


def calls(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            yield name, n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-native", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--bench-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fails: list[str] = []
    res: dict = {"check": "swiglu_config"}

    # 1. the key: required, no default, two values, nothing else
    if not refused(lambda: swig.parse_swiglu({})):
        fails.append("parse_swiglu admitted a config without the key")
    for v in ("chain", "kernel"):
        if swig.parse_swiglu({swig.SWIGLU_KEY: v}) is not swig.Swiglu(v):
            fails.append(f"parse_swiglu({v!r})")
    for bad in ("", "Chain", "KERNEL", "triton", "fused", True, 1, None, ["chain"]):
        if not refused(lambda bad=bad: swig.parse_swiglu({swig.SWIGLU_KEY: bad})):
            fails.append(f"parse_swiglu admitted {bad!r}")
    if sorted(m.value for m in swig.Swiglu) != ["chain", "kernel"]:
        fails.append(f"Swiglu has {[m.value for m in swig.Swiglu]}")
    if not refused(lambda: swig.check_swiglu("chain")):
        fails.append("check_swiglu took a bare string")
    if swig.SWIGLU_KEY not in nat.REQUIRED_KEYS or swig.SWIGLU_KEY not in nat.EXTRA_KEYS:
        fails.append("nvfp4_8r_swiglu is not a required native key")
    # 2. the typed config and the constructors
    args = (Path("/nonexistent/packs"), nat.TallySource(Path("/nonexistent/tally")), {0: 1},
            tiers.HotIq(), tiers.PrefillWarm.NATIVE_STAGED, gplan.GroupPlan.CHAIN)
    try:
        cfg = nat.NativePackConfig(*args, swig.Swiglu.KERNEL, nat.iqt.IqTail.TORCH, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)
        if cfg.swiglu is not swig.Swiglu.KERNEL:
            fails.append("NativePackConfig lost swiglu")
    except (ValueError, TypeError) as exc:
        fails.append(f"NativePackConfig with swiglu refused: {exc}")
    if not refused(lambda: nat.NativePackConfig(*args, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)):
        fails.append("NativePackConfig constructed without swiglu")
    if not refused(lambda: nat.NativePackConfig(*args, "kernel", nat.iqt.IqTail.TORCH, prefill_swiglu=swig.Swiglu.CHAIN, prefill_rb=nat.qrb.PrefillRb.RB2)):
        fails.append("NativePackConfig took a bare string as swiglu")
    for fn in (nat.NativeV41Moe.__init__, nat.NativeV41Moe.three_tier):
        prm = inspect.signature(fn).parameters.get("swiglu")
        if prm is None or prm.kind is not inspect.Parameter.KEYWORD_ONLY or prm.default is not inspect._empty:
            fails.append(f"{fn.__qualname__}: swiglu is not a required keyword ({prm})")

    # 3. wiring and the untouched prefill, by source
    new_src = Path(nat.__file__).read_text()
    new, base = ast.parse(new_src), ast.parse(a.base_native.read_text())
    for name in ("forward_prefill", "_run_cold"):
        if ast.dump(method(new, "NativeV41Moe", name)) != ast.dump(method(base, "NativeV41Moe", name)):
            fails.append(f"NativeV41Moe.{name} changed")
    fc, fb = chain_stmts(method(new, "NativeV41Moe", "forward")), chain_stmts(method(base, "NativeV41Moe", "forward"))
    if len(fb) != 3 or fc != fb:
        fails.append(f"forward's chain statements changed ({len(fc)} vs {len(fb)})")
    n_ctor = 0
    for name, call in calls(new):
        if name in ("NativeV41Moe", "three_tier", "cls"):
            kws = {k.arg for k in call.keywords}
            if "group_plan" in kws:
                n_ctor += 1
                if "swiglu" not in kws:
                    fails.append(f"native line {call.lineno}: {name} without swiglu")
    res["native_constructor_calls"] = n_ctor
    build = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "build_layer")
    passed = [k for _, c in calls(build) for k in c.keywords
              if k.arg == "swiglu" and ast.unparse(k.value) == "cfg.swiglu"]
    if len(passed) != 3:
        fails.append(f"build_layer passes cfg.swiglu {len(passed)} times, want 3")

    # 4. the kernel's arithmetic form (NaN and rounding as the chain)
    ktree = ast.parse(Path(swig.__file__).read_text())
    kern = next(n for n in ktree.body if isinstance(n, ast.FunctionDef) and n.name == "_swiglu_kernel")
    text = ast.unparse(kern)
    for banned in ("tl.minimum", "tl.maximum", "tl.exp(", "tl.sigmoid", "tl.math.", "fast"):
        if banned in text:
            fails.append(f"kernel uses {banned}")
    if any(isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.Mult, ast.Add))
           and not ("row" in ast.unparse(n) or "offs" in ast.unparse(n) or "program_id" in ast.unparse(n)
                    or "INTER_C" in ast.unparse(n))
           for n in ast.walk(kern)):
        fails.append("kernel has a float +, * or / outside libdevice's _rn calls")
    for need in ("libdevice.exp(", "libdevice.div_rn(", "libdevice.add_rn(", "libdevice.mul_rn("):
        if need not in text:
            fails.append(f"kernel lacks {need}")

    # 5. profiles
    profs = sorted(a.run_dir.glob("serve-nvfp4-pp2*.yaml"))
    res["profiles"] = len(profs)
    if not profs:
        fails.append(f"no serve-nvfp4-pp2*.yaml in {a.run_dir}")
    for f in profs:
        text = f.read_text()
        lines = KEY_LINE.findall(text)
        extra = (yaml.safe_load(text) or {}).get("model-loader-extra-config", {})
        if len(lines) != 1 or extra.get(swig.SWIGLU_KEY) not in ("chain", "kernel"):
            fails.append(f"{f.name}: nvfp4_8r_swiglu lines {lines}, value {extra.get(swig.SWIGLU_KEY)!r}")
        else:
            try:
                swig.parse_swiglu(extra)
            except ValueError as exc:
                fails.append(f"{f.name}: {exc}")

    # 6. the bench tree: every constructor call that names group_plan names swiglu
    n_bench = 0
    for f in sorted(a.bench_dir.glob("*.py")):
        tree = ast.parse(f.read_text())
        for name, call in calls(tree):
            kws = {k.arg for k in call.keywords}
            seg = ast.unparse(call)
            if name in ("NativeV41Moe", "three_tier") and "group_plan" in kws:
                n_bench += 1
                if "swiglu" not in kws:
                    fails.append(f"{f.name}:{call.lineno} {name} without swiglu")
            if name == "NativePackConfig" and "GroupPlan" in seg:
                n_bench += 1
                if "Swiglu" not in seg:
                    fails.append(f"{f.name}:{call.lineno} NativePackConfig without swiglu")
    res["bench_calls_checked"] = n_bench
    res["failures"] = fails
    res["status"] = "FAIL" if fails else "PASS"
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": res["status"], "profiles": res["profiles"],
                      "bench_calls_checked": n_bench, "native_constructor_calls": n_ctor,
                      "failures": fails[:8]}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
