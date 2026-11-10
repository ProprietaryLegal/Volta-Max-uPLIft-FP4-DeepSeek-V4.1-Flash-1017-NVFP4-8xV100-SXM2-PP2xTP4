"""P5 acceptance (CPU, dry): every path the launcher uses stays in the lane.

Reads run/env.sh, run/run_serve.sh and run/serve-nvfp4-pp2.yaml. Every
absolute path found (YAML string values at any depth, env.sh assignments
including each PYTHONPATH entry, absolute paths in run_serve.sh) must exist
and resolve, symlinks followed, under $LANE/. The profile
must bind 127.0.0.1:18043. Paths the pre-check only probes (none) and the
production port number are not paths. Exit 0 PASS, 1 FAIL.

One exception, checked more strictly instead: a venv interpreter
(LANE/.../venv/bin/python*) is a symlink to the system binary by design.
For it the symlink itself must sit in the lane and the interpreter, when run,
must report sys.prefix and the imported vllm package inside the lane.
"""

from __future__ import annotations

import os
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ["LANE"])
RUN = ROOT / "run"
PROFILE = RUN / "serve-nvfp4-pp2.yaml"
ABS = re.compile(r"(?<![\w$])/[\w./@+-]+")


def strings(node) -> list[str]:
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in strings(k) + strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in strings(v)]
    return [node] if isinstance(node, str) else []


def candidates(text: str) -> list[str]:
    out = []
    for token in ABS.findall(text):
        out.extend(p for p in token.split(":") if p.startswith("/"))
    return out


def venv_python_ok(p: str) -> str | None:
    """None if the lane venv interpreter imports from the lane, else why not."""
    path = Path(p)
    if not path.parent.resolve().is_relative_to(ROOT):
        return f"{p}: venv bin dir outside {ROOT}"
    out = subprocess.run(
        [p, "-c", "import sys, vllm; print(sys.prefix); print(vllm.__file__)"],
        capture_output=True, text=True, check=True, timeout=600,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout.split("\n")
    for line in out[:2]:
        if not Path(line).resolve().is_relative_to(ROOT):
            return f"{p}: runs from {line}, outside {ROOT}"
    return None


def main() -> int:
    failures: list[str] = []
    checked: dict[str, str] = {}
    sources = {}
    for f in (RUN / "env.sh", RUN / "run_serve.sh", PROFILE):
        if not f.is_file():
            failures.append(f"missing launcher file {f}")
            continue
        sources[f] = f.read_text()
    if PROFILE in sources:
        profile = yaml.safe_load(sources[PROFILE])
        if profile.get("host") != "127.0.0.1" or profile.get("port") != 18043:
            failures.append(f"profile binds {profile.get('host')}:{profile.get('port')}")
        paths = [p for s in strings(profile) for p in candidates(s)]
    else:
        paths = []
    for f in (RUN / "env.sh", RUN / "run_serve.sh"):
        if f in sources:
            body = "\n".join(
                line for line in sources[f].splitlines() if not line.lstrip().startswith("#")
            )
            paths += candidates(body)
    allowed_system = {"/usr/bin/env", "/bin/bash", "/dev/null", "/proc/net/tcp"}
    for p in sorted(set(paths)):
        if p in allowed_system:
            continue
        real = Path(p).resolve()
        checked[p] = str(real)
        if re.fullmatch(r".*/venv/bin/python[\d.]*", p):
            why = venv_python_ok(p)
            if why is not None:
                failures.append(why)
            checked[p] += " (venv interpreter: prefix and vllm in lane)"
        elif not real.is_relative_to(ROOT):
            failures.append(f"{p} -> {real} is outside {ROOT}")
        elif not real.exists():
            failures.append(f"{p} does not exist")
    status = "P5_PASS" if not failures and checked else "P5_FAIL"
    if not checked and not failures:
        failures.append("no paths found: INCONCLUSIVE")
    print(json.dumps({"status": status, "paths_checked": len(checked),
                      "failures": failures, "resolved": checked}, indent=1))
    return 0 if status == "P5_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
