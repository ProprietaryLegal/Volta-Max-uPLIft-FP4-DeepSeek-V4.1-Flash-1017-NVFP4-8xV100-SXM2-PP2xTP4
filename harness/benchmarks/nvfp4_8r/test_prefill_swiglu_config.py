"""CPU checks of the prefill SwiGLU mode (D3, accept rule 1 of its method).

    python -m benchmarks.nvfp4_8r.test_prefill_swiglu_config --base-native BEFORE.py \
        --packs PACKS --tally-file ANY_FILE --bench-dir engine/src/benchmarks/nvfp4_8r \
        (--run-dir run | --no-profiles) --output OUT.json

Expected behaviour comes from the D3 method, not from the patch:
- nvfp4_8r_prefill_swiglu is a required key (no default), chain | kernel, independent of nvfp4_8r_swiglu;
- the typed config and both constructors require it (no default), and a bare string is refused;
- forward_prefill differs from BEFORE (the G1 tree's nvfp4_8r_native.py) only by one new branch in front of
  the unchanged chain: the If ``first and self.prefill_swiglu is swig.Swiglu.KERNEL`` whose body is the one
  statement ``swig.swiglu_into(y, inter)`` and whose else is BEFORE's ``if first:`` unchanged; forward,
  _run_cold, _iq_routes and _iq_partial are unchanged (decode is untouched); with the RB 4 patch on top, its one
  dispatch call in forward_prefill and in _run_cold, qrb.prefill_qpn(self.prefill_rb, *a), is first put back as
  the installed call torch.ops._C.skinny_moe_qpn_sm70(*a) (test_prefill_rb_config.py checks the patch itself);
- build_layer passes cfg.prefill_swiglu on its three constructor calls; every NativeV41Moe(...),
  .three_tier(...), cls(...) and NativePackConfig(...) call in the native module and the bench tree carries
  prefill_swiglu (AST scan, zero missing);
- swiglu_into refuses a row count whose element offsets would pass int32 (S x 1152 >= 2^31), checked on
  meta tensors (no launch);
- every serve-nvfp4-pp2*.yaml in --run-dir carries the key once as '  nvfp4_8r_prefill_swiglu: chain|kernel',
  except the port copies *-p18042.yaml (never edited; listed in the receipt, not checked; see section 5).
Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import re
import sys
from pathlib import Path

import torch
import yaml

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

KEY = "nvfp4_8r_prefill_swiglu"
KEY_LINE = re.compile(rf"^  {KEY}: (chain|kernel)$", re.M)
CTORS = ("NativeV41Moe", "three_tier")


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


def undo_branch(fn: ast.FunctionDef) -> tuple[ast.FunctionDef, list[str]]:
    """fn with the one D3 branch replaced by its else; the problems found."""
    fn = copy.deepcopy(fn)
    problems: list[str] = []
    want_test = "first and self.prefill_swiglu is swig.Swiglu.KERNEL"
    want_body = "swig.swiglu_into(y, inter)"
    hits = 0
    for node in ast.walk(fn):
        for field in ("body", "orelse"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list):
                continue
            for i, st in enumerate(stmts):
                if isinstance(st, ast.If) and ast.unparse(st.test) == want_test:
                    hits += 1
                    if [ast.unparse(b) for b in st.body] != [want_body]:
                        problems.append(f"D3 branch body is {[ast.unparse(b) for b in st.body]}")
                    if len(st.orelse) != 1 or not isinstance(st.orelse[0], ast.If) \
                            or ast.unparse(st.orelse[0].test) != "first":
                        problems.append("D3 branch's else is not one 'if first:'")
                    else:
                        stmts[i] = st.orelse[0]
    if hits != 1:
        problems.append(f"{hits} D3 branches in forward_prefill (want 1)")
    return fn, problems


def undo_rb(fn: ast.FunctionDef, want: int) -> tuple[ast.FunctionDef, list[str]]:
    """fn with the RB 4 patch's dispatch qrb.prefill_qpn(self.prefill_rb, *a) put back as the installed
    call torch.ops._C.skinny_moe_qpn_sm70(*a); the problems found (want: the number of such calls)."""
    fn = copy.deepcopy(fn)
    hits = 0
    installed = ast.parse("torch.ops._C.skinny_moe_qpn_sm70", mode="eval").body
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "qrb.prefill_qpn":
            if not node.args or ast.unparse(node.args[0]) != "self.prefill_rb" or node.keywords:
                return fn, [f"qrb.prefill_qpn call {ast.unparse(node)[:80]} is not (self.prefill_rb, *args)"]
            node.func = copy.deepcopy(installed)
            node.args = node.args[1:]
            hits += 1
    return fn, ([] if hits == want else [f"{hits} qrb.prefill_qpn calls in {fn.name} (want {want})"])


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
    res: dict = {"check": "prefill_swiglu_config"}
    n_ctor = n_bench = 0

    # 1. the key: required, no default, two values, independent of the decode key
    try:
        if swig.PREFILL_SWIGLU_KEY != KEY:
            fails.append(f"PREFILL_SWIGLU_KEY is {swig.PREFILL_SWIGLU_KEY!r}")
        if not refused(lambda: swig.parse_prefill_swiglu({})):
            fails.append("parse_prefill_swiglu admitted a config without the key")
        for dec in ("chain", "kernel"):
            if not refused(lambda dec=dec: swig.parse_prefill_swiglu({swig.SWIGLU_KEY: dec})):
                fails.append(f"the decode key {dec!r} alone set the prefill mode")
        for v in ("chain", "kernel"):
            if swig.parse_prefill_swiglu({KEY: v}) is not swig.Swiglu(v):
                fails.append(f"parse_prefill_swiglu({v!r})")
            for dec in ("chain", "kernel"):
                both = {KEY: v, swig.SWIGLU_KEY: dec}
                if swig.parse_prefill_swiglu(both) is not swig.Swiglu(v) or swig.parse_swiglu(both) is not swig.Swiglu(dec):
                    fails.append(f"prefill {v} with decode {dec} not independent")
        for bad in ("", "Chain", "KERNEL", "triton", "fused", True, 1, None, ["chain"]):
            if not refused(lambda bad=bad: swig.parse_prefill_swiglu({KEY: bad})):
                fails.append(f"parse_prefill_swiglu admitted {bad!r}")
        if KEY not in nat.REQUIRED_KEYS or KEY not in nat.EXTRA_KEYS:
            fails.append(f"{KEY} is not a required native key")

    except Exception as exc:  # recorded, so a tree without D3 FAILs every section
        fails.append(f"section 1 raised {type(exc).__name__}: {exc}")
    # 2. the typed config and the constructors
    try:
        args = (Path("/nonexistent/packs"), nat.TallySource(Path("/nonexistent/tally")), {0: 1},
                tiers.HotIq(), tiers.PrefillWarm.NATIVE_STAGED, gplan.GroupPlan.CHAIN, swig.Swiglu.CHAIN,
                nat.iqt.IqTail.TORCH)
        for v in swig.Swiglu:
            try:
                cfg = nat.NativePackConfig(*args, v, nat.qrb.PrefillRb.RB2)
                if cfg.prefill_swiglu is not v or cfg.swiglu is not swig.Swiglu.CHAIN:
                    fails.append(f"NativePackConfig({v.value}) lost a field")
            except (ValueError, TypeError) as exc:
                fails.append(f"NativePackConfig({v.value}) refused: {exc}")
        if not refused(lambda: nat.NativePackConfig(*args)):
            fails.append("NativePackConfig constructed without prefill_swiglu")
        if not refused(lambda: nat.NativePackConfig(*args, "kernel", nat.qrb.PrefillRb.RB2)):
            fails.append("NativePackConfig took a bare string as prefill_swiglu")
        for fn in (nat.NativeV41Moe.__init__, nat.NativeV41Moe.three_tier):
            prm = inspect.signature(fn).parameters.get("prefill_swiglu")
            if prm is None or prm.kind is not inspect.Parameter.KEYWORD_ONLY or prm.default is not inspect._empty:
                fails.append(f"{fn.__qualname__}: prefill_swiglu is not a required keyword ({prm})")
        # from_extra, the loader's entry (reads only that packs has a manifest and the tally exists)
        extra = {"nvfp4_8r_packs": str(a.packs), "nvfp4_8r_tally": str(a.tally_file),
                 "nvfp4_8r_reserve_bytes": {0: 1, 20: 1}, "nvfp4_8r_residency": "two_tier",
                 "nvfp4_8r_prefill_warm": "native_staged", "nvfp4_8r_group_plan": "chain",
                 "nvfp4_8r_swiglu": "chain", "nvfp4_8r_prefill_rb": "rb2"}
        if not refused(lambda: nat.NativePackConfig.from_extra(extra)):
            fails.append("from_extra admitted a config without the key")
        for v in ("chain", "kernel"):
            try:
                got = nat.NativePackConfig.from_extra({**extra, KEY: v}).prefill_swiglu
                if got is not swig.Swiglu(v):
                    fails.append(f"from_extra({v}) gave {got}")
            except (ValueError, TypeError) as exc:
                fails.append(f"from_extra({v}) refused: {exc}")
        if not refused(lambda: nat.NativePackConfig.from_extra({**extra, KEY: "fused"})):
            fails.append("from_extra admitted 'fused'")

    except Exception as exc:  # recorded, so a tree without D3 FAILs every section
        fails.append(f"section 2 raised {type(exc).__name__}: {exc}")
    # 3. the change, by source: one branch in forward_prefill, nothing else in the layer's paths
    try:
        new = ast.parse(Path(nat.__file__).read_text())
        base = ast.parse(a.base_native.read_text())
        # RB 4 (patched on top of D3): its one dispatch call per path is undone first.
        rb_undone, probs = undo_rb(method(new, "NativeV41Moe", "forward_prefill"), 1)
        fails += probs
        undone, probs = undo_branch(rb_undone)
        fails += probs
        if ast.dump(undone) != ast.dump(method(base, "NativeV41Moe", "forward_prefill")):
            fails.append("forward_prefill differs from BEFORE by more than the D3 branch")
        for name in ("forward", "_run_cold", "_iq_routes", "_iq_partial"):
            try:
                got = method(new, "NativeV41Moe", name)
                if name == "_run_cold":
                    got, probs = undo_rb(got, 1)
                    fails += probs
                same = ast.dump(got) == ast.dump(method(base, "NativeV41Moe", name))
            except LookupError:
                same = False
            if not same:
                fails.append(f"NativeV41Moe.{name} changed or missing")
        n_ctor = 0
        for name, call in calls(new):
            if name in CTORS + ("cls",):
                kws = {k.arg for k in call.keywords}
                if "group_plan" in kws:
                    n_ctor += 1
                    if "prefill_swiglu" not in kws:
                        fails.append(f"native line {call.lineno}: {name} without prefill_swiglu")
        res["native_constructor_calls"] = n_ctor
        build = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "build_layer")
        passed = [k for _, c in calls(build) for k in c.keywords
                  if k.arg == "prefill_swiglu" and ast.unparse(k.value) == "cfg.prefill_swiglu"]
        if len(passed) != 3:
            fails.append(f"build_layer passes cfg.prefill_swiglu {len(passed)} times, want 3")

    except Exception as exc:  # recorded, so a tree without D3 FAILs every section
        fails.append(f"section 3 raised {type(exc).__name__}: {exc}")
    # 4. the int32 guard (meta tensors: shapes only, nothing launches)
    try:
        big = -(-(2**31) // swig.Y13_COLS)
        y13 = torch.empty((big, swig.Y13_COLS), dtype=torch.float16, device="meta")
        inter = torch.empty((big, swig.OUT_COLS), dtype=torch.float16, device="meta")
        try:
            swig.swiglu_into(y13, inter)
            fails.append(f"swiglu_into took {big} rows")
        except ValueError as exc:
            if "int32" not in str(exc):
                fails.append(f"swiglu_into refused {big} rows for another reason: {exc}")

    except Exception as exc:  # recorded, so a tree without D3 FAILs every section
        fails.append(f"section 4 raised {type(exc).__name__}: {exc}")
    # 5. profiles
    try:
        if a.no_profiles:
            res["profiles"] = "not checked here"
        else:
            # Port copies (PROFILE-p18042.yaml, the default on :18042) are made once by make_p18042_profile.sh and
            # never edited, so a copy made before D3 never takes the key. They are listed, not checked: launch.sh
            # uplift boots only the copy of the profile hot_iq names and refuses one that is not that profile's
            # bytes with the port line changed, and that profile is checked here.
            allp = sorted(a.run_dir.glob("serve-nvfp4-pp2*.yaml"))
            profs = [f for f in allp if not f.name.endswith("-p18042.yaml")]
            res["port_copies_not_checked"] = [f.name for f in allp if f.name.endswith("-p18042.yaml")]
            res["profiles"] = len(profs)
            if not profs:
                fails.append(f"no serve-nvfp4-pp2*.yaml in {a.run_dir}")
            for f in profs:
                text = f.read_text()
                lines = KEY_LINE.findall(text)
                ext = (yaml.safe_load(text) or {}).get("model-loader-extra-config", {})
                if len(lines) != 1 or text.count(KEY) != 1 or ext.get(KEY) != lines[0]:
                    fails.append(f"{f.name}: {KEY} lines {lines}, value {ext.get(KEY)!r}")

    except Exception as exc:  # recorded, so a tree without D3 FAILs every section
        fails.append(f"section 5 raised {type(exc).__name__}: {exc}")
    # 6. the bench tree
    try:
        n_bench = 0
        for f in sorted(a.bench_dir.glob("*.py")):
            tree = ast.parse(f.read_text())
            for name, call in calls(tree):
                kws = {k.arg for k in call.keywords}
                if name in CTORS and "group_plan" in kws:
                    n_bench += 1
                    if "prefill_swiglu" not in kws:
                        fails.append(f"{f.name}:{call.lineno} {name} without prefill_swiglu")
                if name == "NativePackConfig" and "GroupPlan" in ast.unparse(call):
                    n_bench += 1
                    if "prefill_swiglu" not in kws:
                        fails.append(f"{f.name}:{call.lineno} NativePackConfig without prefill_swiglu")
    except Exception as exc:  # recorded, so a tree without D3 FAILs every section
        fails.append(f"section 6 raised {type(exc).__name__}: {exc}")
    res["bench_calls_checked"] = n_bench
    res["failures"] = fails
    res["status"] = "FAIL" if fails else "PASS"
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"status": res["status"], "profiles": res.get("profiles"), "bench_calls_checked": n_bench,
                      "native_constructor_calls": n_ctor, "n_failures": len(fails), "failures": fails[:8]}))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
