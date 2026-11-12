"""CPU checks of the prefill RB key (RB 4 patch, section 2 of its method).

    python -m benchmarks.nvfp4_8r.test_prefill_rb_config --base-native D3_NATIVE.py \
        --packs PACKS --tally-file ANY_FILE --bench-dir engine/src/benchmarks/nvfp4_8r \
        (--run-dir run | --no-profiles) --output OUT.json

Expected behaviour comes from the method, not from the patch:
- nvfp4_8r_prefill_rb is a required key (no default), rb2 | rb4, independent of nvfp4_8r_prefill_swiglu;
- the typed config and both constructors require it (no default), and a bare string is refused;
- forward_prefill and _run_cold differ from --base-native (D3's nvfp4_8r_native.py) only in that their one QPN
  call torch.ops._C.skinny_moe_qpn_sm70(*a) became qrb.prefill_qpn(self.prefill_rb, *a); forward, _iq_routes and
  _iq_partial are unchanged (decode is untouched); build_layer passes cfg.prefill_rb on its three constructor
  calls; every NativeV41Moe(...), .three_tier(...), cls(...) and NativePackConfig(...) call in the native module
  and the bench tree carries prefill_rb (AST scan, zero missing);
- the library: a missing file and a file of other bytes are refused before loading; the file beside the module
  has the pinned sha256, loads, and registers nvfp4_8r_qpn_rb.moe_prefill with the installed op's arguments
  plus `int rb`; no extension file beside the module has the stem of a module there (it would be imported in its
  place, as seen on a first install), and the library is beside the module;
- every serve-nvfp4-pp2*.yaml in --run-dir carries the key once as '  nvfp4_8r_prefill_rb: rb2|rb4', except the
  port copies *-p18042.yaml (never edited; listed in the receipt, not checked).
Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_group_plan as gplan
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_native as nat
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_qpn_rb as qrb
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_swiglu as swig
from vllm.model_executor.layers.fused_moe.experts import nvfp4_8r_tiers as tiers

KEY = "nvfp4_8r_prefill_rb"
KEY_LINE = re.compile(rf"^  {KEY}: (rb2|rb4)$", re.M)
CTORS = ("NativeV41Moe", "three_tier")
INSTALLED = "torch.ops._C.skinny_moe_qpn_sm70"


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


def undo_rb(fn: ast.FunctionDef) -> tuple[ast.FunctionDef, int, list[str]]:
    """fn with each qrb.prefill_qpn(self.prefill_rb, *a) put back as the installed call; (fn, hits, problems)."""
    fn = copy.deepcopy(fn)
    hits, problems = 0, []
    installed = ast.parse(INSTALLED, mode="eval").body
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "qrb.prefill_qpn":
            if not node.args or ast.unparse(node.args[0]) != "self.prefill_rb" or node.keywords:
                problems.append(f"{ast.unparse(node)[:80]} is not qrb.prefill_qpn(self.prefill_rb, *args)")
                continue
            node.func = copy.deepcopy(installed)
            node.args = node.args[1:]
            hits += 1
    return fn, hits, problems


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
    res: dict = {"check": "prefill_rb_config",
                 "modules_under_test": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes())
                                        .hexdigest()[:16] for m in (nat, qrb)}}
    n_ctor = n_bench = 0

    # 1. the key: required, no default, two values, independent of the prefill SwiGLU key
    try:
        if qrb.PREFILL_RB_KEY != KEY:
            fails.append(f"PREFILL_RB_KEY is {qrb.PREFILL_RB_KEY!r}")
        if [m.value for m in qrb.PrefillRb] != ["rb2", "rb4"]:
            fails.append(f"PrefillRb values {[m.value for m in qrb.PrefillRb]}")
        if not refused(lambda: qrb.parse_prefill_rb({})):
            fails.append("parse_prefill_rb admitted a config without the key")
        for sw in ("chain", "kernel"):
            if not refused(lambda sw=sw: qrb.parse_prefill_rb({swig.PREFILL_SWIGLU_KEY: sw})):
                fails.append(f"the prefill SwiGLU key {sw!r} alone set the RB")
        for v in ("rb2", "rb4"):
            if qrb.parse_prefill_rb({KEY: v}) is not qrb.PrefillRb(v):
                fails.append(f"parse_prefill_rb({v!r})")
            for sw in ("chain", "kernel"):
                both = {KEY: v, swig.PREFILL_SWIGLU_KEY: sw}
                if qrb.parse_prefill_rb(both) is not qrb.PrefillRb(v) \
                        or swig.parse_prefill_swiglu(both) is not swig.Swiglu(sw):
                    fails.append(f"rb {v} with prefill SwiGLU {sw} not independent")
        for bad in ("", "RB4", "rb8", "rb1", "4", 4, 2, True, None, ["rb4"]):
            if not refused(lambda bad=bad: qrb.parse_prefill_rb({KEY: bad})):
                fails.append(f"parse_prefill_rb admitted {bad!r}")
        if KEY not in nat.REQUIRED_KEYS or KEY not in nat.EXTRA_KEYS:
            fails.append(f"{KEY} is not a required native key")
        if not refused(lambda: qrb.check_prefill_rb("rb4")):
            fails.append("check_prefill_rb took a bare string")
    except Exception as exc:  # recorded, so a tree without the patch FAILs every section
        fails.append(f"section 1 raised {type(exc).__name__}: {exc}")

    # 2. the typed config and the constructors
    try:
        args = (Path("/nonexistent/packs"), nat.TallySource(Path("/nonexistent/tally")), {0: 1},
                tiers.HotIq(), tiers.PrefillWarm.NATIVE_STAGED, gplan.GroupPlan.CHAIN, swig.Swiglu.CHAIN,
                nat.iqt.IqTail.TORCH, swig.Swiglu.CHAIN)
        for v in qrb.PrefillRb:
            try:
                cfg = nat.NativePackConfig(*args, prefill_rb=v)
                if cfg.prefill_rb is not v or cfg.prefill_swiglu is not swig.Swiglu.CHAIN:
                    fails.append(f"NativePackConfig({v.value}) lost a field")
            except (ValueError, TypeError) as exc:
                fails.append(f"NativePackConfig({v.value}) refused: {exc}")
        if not refused(lambda: nat.NativePackConfig(*args)):
            fails.append("NativePackConfig constructed without prefill_rb")
        if not refused(lambda: nat.NativePackConfig(*args, prefill_rb="rb4")):
            fails.append("NativePackConfig took a bare string as prefill_rb")
        for fn in (nat.NativeV41Moe.__init__, nat.NativeV41Moe.three_tier):
            prm = inspect.signature(fn).parameters.get("prefill_rb")
            if prm is None or prm.kind is not inspect.Parameter.KEYWORD_ONLY or prm.default is not inspect._empty:
                fails.append(f"{fn.__qualname__}: prefill_rb is not a required keyword ({prm})")
        extra = {"nvfp4_8r_packs": str(a.packs), "nvfp4_8r_tally": str(a.tally_file),
                 "nvfp4_8r_reserve_bytes": {0: 1, 20: 1}, "nvfp4_8r_residency": "two_tier",
                 "nvfp4_8r_prefill_warm": "native_staged", "nvfp4_8r_group_plan": "chain",
                 "nvfp4_8r_swiglu": "chain", swig.PREFILL_SWIGLU_KEY: "chain"}
        if not refused(lambda: nat.NativePackConfig.from_extra(extra)):
            fails.append("from_extra admitted a config without the key")
        for v in ("rb2", "rb4"):
            try:
                got = nat.NativePackConfig.from_extra({**extra, KEY: v}).prefill_rb
                if got is not qrb.PrefillRb(v):
                    fails.append(f"from_extra({v}) gave {got}")
            except (ValueError, TypeError) as exc:
                fails.append(f"from_extra({v}) refused: {exc}")
        if not refused(lambda: nat.NativePackConfig.from_extra({**extra, KEY: "rb8"})):
            fails.append("from_extra admitted 'rb8'")
    except Exception as exc:
        fails.append(f"section 2 raised {type(exc).__name__}: {exc}")

    # 3. the change, by source: one dispatch call in forward_prefill and in _run_cold, nothing else
    try:
        new = ast.parse(Path(nat.__file__).read_text())
        base = ast.parse(a.base_native.read_text())
        for name in ("forward_prefill", "_run_cold"):
            fn = method(new, "NativeV41Moe", name)
            if any(ast.unparse(c.func) == INSTALLED for _, c in calls(fn)):
                fails.append(f"NativeV41Moe.{name} still calls {INSTALLED} directly")
            undone, hits, probs = undo_rb(fn)
            fails += probs
            if hits != 1:
                fails.append(f"{hits} qrb.prefill_qpn calls in {name} (want 1)")
            if ast.dump(undone) != ast.dump(method(base, "NativeV41Moe", name)):
                fails.append(f"NativeV41Moe.{name} differs from the base by more than the dispatch call")
        for name in ("forward", "_iq_routes", "_iq_partial"):
            try:
                same = ast.dump(method(new, "NativeV41Moe", name)) == ast.dump(method(base, "NativeV41Moe", name))
            except LookupError:
                same = False
            if not same:
                fails.append(f"NativeV41Moe.{name} changed or missing")
        for name, call in calls(new):
            if name in CTORS + ("cls",):
                kws = {k.arg for k in call.keywords}
                if "group_plan" in kws:
                    n_ctor += 1
                    if "prefill_rb" not in kws:
                        fails.append(f"native line {call.lineno}: {name} without prefill_rb")
        res["native_constructor_calls"] = n_ctor
        build = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "build_layer")
        passed = [k for _, c in calls(build) for k in c.keywords
                  if k.arg == "prefill_rb" and ast.unparse(k.value) == "cfg.prefill_rb"]
        if len(passed) != 3:
            fails.append(f"build_layer passes cfg.prefill_rb {len(passed)} times, want 3")
    except Exception as exc:
        fails.append(f"section 3 raised {type(exc).__name__}: {exc}")

    # 4. the library: refusals first (before anything is loaded), then the real file
    try:
        lib, ns = qrb.LIBRARY, qrb.OP_NAMESPACE
        if hasattr(getattr(torch.ops, ns), "moe_prefill"):
            fails.append("the library was loaded before section 4 (the refusals cannot be checked)")
        else:
            try:
                qrb.LIBRARY = Path("/nonexistent/v41qpnrb4.so")
                try:
                    qrb.load_library()
                    fails.append("load_library took a missing file")
                except RuntimeError:
                    pass
                with tempfile.TemporaryDirectory() as d:
                    other = Path(d) / "v41qpnrb4.so"
                    raw = bytearray(lib.read_bytes())
                    raw[-1] ^= 1  # one bit of the last byte
                    other.write_bytes(bytes(raw))
                    qrb.LIBRARY = other
                    try:
                        qrb.load_library()
                        fails.append("load_library took a file of other bytes")
                    except RuntimeError as exc:
                        if "pins" not in str(exc):
                            fails.append(f"load_library refused other bytes for another reason: {exc}")
            finally:
                qrb.LIBRARY = lib
            if hasattr(getattr(torch.ops, ns), "moe_prefill"):
                fails.append("a refused library registered the op")
        got = hashlib.sha256(lib.read_bytes()).hexdigest()
        res["library_sha256"] = got[:16]
        if got != qrb.LIBRARY_SHA256:
            fails.append(f"{lib.name} beside the module is {got[:16]}, the module pins {qrb.LIBRARY_SHA256[:16]}")
        qrb.load_library()
        schema = str(torch.ops.nvfp4_8r_qpn_rb.moe_prefill.default._schema)
        want = ("nvfp4_8r_qpn_rb::moe_prefill(Tensor x, Tensor qcodes, Tensor qscales, Tensor gscales, "
                "Tensor perm, Tensor gids, Tensor goff, int topk, Tensor(a!) y_slots, bool x_slot_major, "
                "int num_tokens, int splitk, int scale_mode, int fragment_order, int rb) -> ()")
        if schema != want:
            fails.append(f"moe_prefill schema {schema}")
        inst = str(torch.ops._C.skinny_moe_qpn_sm70.default._schema)
        res["installed_schema"] = inst
        args_of = lambda s: s.split("(", 1)[1].rsplit(") -> ", 1)[0]  # noqa: E731
        if args_of(schema) != args_of(inst) + ", int rb":
            fails.append("moe_prefill's arguments are not the installed op's plus rb")
    except Exception as exc:
        fails.append(f"section 4 raised {type(exc).__name__}: {exc}")

    # 4b. no file beside the module can be imported in place of a module there (as seen on a first install): Python's
    # finder tries the extension suffixes before .py, so STEMSUFFIX beside STEM.py shadows it. Checked on
    # the folder the module was loaded from, so it holds under a staging overlay too.
    try:
        from importlib.machinery import EXTENSION_SUFFIXES
        here = Path(qrb.__file__).resolve().parent
        stems = {p.stem for p in here.glob("*.py")}
        shadow = sorted(p.name for p in here.iterdir() for s in EXTENSION_SUFFIXES
                        if p.name.endswith(s) and p.name[: -len(s)] in stems)
        res["module_dir"] = str(here)
        res["library_file"] = qrb.LIBRARY.name
        if shadow:
            fails.append(f"extension files shadow modules in {here}: {shadow}")
        if qrb.LIBRARY.resolve().parent != here:
            fails.append(f"the library {qrb.LIBRARY} is not beside the module ({here})")
    except Exception as exc:
        fails.append(f"section 4b raised {type(exc).__name__}: {exc}")

    # 5. profiles (port copies are never edited: listed, not checked, as in test_prefill_swiglu_config)
    try:
        if a.no_profiles:
            res["profiles"] = "not checked here"
        else:
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
    except Exception as exc:
        fails.append(f"section 5 raised {type(exc).__name__}: {exc}")

    # 6. the bench tree
    try:
        for f in sorted(a.bench_dir.glob("*.py")):
            tree = ast.parse(f.read_text())
            for name, call in calls(tree):
                kws = {k.arg for k in call.keywords}
                if name in CTORS and "group_plan" in kws:
                    n_bench += 1
                    if "prefill_rb" not in kws:
                        fails.append(f"{f.name}:{call.lineno} {name} without prefill_rb")
                if name == "NativePackConfig" and "GroupPlan" in ast.unparse(call):
                    n_bench += 1
                    if "prefill_rb" not in kws:
                        fails.append(f"{f.name}:{call.lineno} NativePackConfig without prefill_rb")
    except Exception as exc:
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
