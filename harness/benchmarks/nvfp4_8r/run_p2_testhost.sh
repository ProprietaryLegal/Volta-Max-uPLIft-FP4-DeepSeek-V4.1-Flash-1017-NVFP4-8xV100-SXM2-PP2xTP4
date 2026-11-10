#!/bin/bash
# Run ON THE TEST HOST only.
# Usage: run_p2_testhost.sh CARD. accept_p2's own gate refuses (exit 3) on a foreign
# compute app, a power limit > 150 W or MemAvailable < 60 GiB. Never changes caps.
set -euo pipefail
card=${1:?card index}
W=${W:?set W to the work directory on the test host}
cd "$W"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card" PYTHONPATH="$W/pypath"
export XDG_CACHE_HOME="$W/cache" TRITON_CACHE_DIR="$W/cache/triton" VLLM_NO_USAGE_STATS=1
stamp=$(date -u +%Y%m%dT%H%M%SZ)
"$W/venv/bin/python" -m benchmarks.nvfp4_8r.accept_p2 --packs "$W/packs" \
  --refs "$W/p2/refs-L05.npz" "$W/p2/refs-L12.npz" "$W/p2/refs-L27.npz" \
  --card "$card" --output "$W/logs/p2-acceptance-$stamp.json" > "$W/logs/p2-acceptance-$stamp.log" 2>&1
