#!/bin/bash
# Run ON THE TEST HOST only. Usage: run_p4_testhost.sh CARD TAG. accept_p4's gate refuses
# (exit 3) on any compute app, a power limit > 150 W or MemAvailable < 60 GiB.
set -euo pipefail
card=${1:?card index}
tag=${2:?tag}
W=${W:?set W to the work directory on the test host}
cd "$W"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card" PYTHONPATH="$W/pypath"
export XDG_CACHE_HOME="$W/cache" TRITON_CACHE_DIR="$W/cache/triton" VLLM_NO_USAGE_STATS=1
"$W/venv/bin/python" -m benchmarks.nvfp4_8r.accept_p4 --refs "$W/p2/refs-p4-engram.npz" \
  --card "$card" --output "$W/logs/p4-rows-$tag.json" > "$W/logs/p4-rows-$tag.log" 2>&1
