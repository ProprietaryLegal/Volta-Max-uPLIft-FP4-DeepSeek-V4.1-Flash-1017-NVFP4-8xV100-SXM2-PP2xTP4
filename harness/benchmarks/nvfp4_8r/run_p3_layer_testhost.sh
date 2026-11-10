#!/bin/bash
# Run ON THE TEST HOST only, after P2 passes.
# Usage: run_p3_layer_testhost.sh CARD. accept_p3_layer (via accept_p2.gpu_gate)'s own gate refuses (exit 3) on a foreign
# compute app, a power limit > 150 W or MemAvailable < 60 GiB. Never changes caps.
set -euo pipefail
card=${1:?card index}
W=${W:?set W to the work directory on the test host}
cd "$W"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card" PYTHONPATH="$W/pypath"
export XDG_CACHE_HOME="$W/cache" TRITON_CACHE_DIR="$W/cache/triton" VLLM_NO_USAGE_STATS=1
stamp=$(date -u +%Y%m%dT%H%M%SZ)
"$W/venv/bin/python" -m benchmarks.nvfp4_8r.accept_p3_layer --packs "$W/packs" \
  --refs "$W/p2/refs-L05.npz" "$W/p2/refs-L12.npz" "$W/p2/refs-L27.npz" \
  --prefill-refs "$W/p2/refs-prefill-L05.npz" "$W/p2/refs-prefill-L12.npz" "$W/p2/refs-prefill-L27.npz" \
  --card "$card" --output "$W/logs/p3-layer-$stamp.json" > "$W/logs/p3-layer-$stamp.log" 2>&1
