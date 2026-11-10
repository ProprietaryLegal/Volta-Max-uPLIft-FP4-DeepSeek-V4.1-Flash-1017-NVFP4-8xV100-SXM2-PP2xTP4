#!/bin/bash
# Run ON THE TEST HOST only. Usage: run_p3_bench_testhost.sh CARD. The bench's gate refuses
# (exit 3) on any compute app, a power limit > 150 W or MemAvailable < 60 GiB.
set -euo pipefail
card=${1:?card index}
W=${W:?set W to the work directory on the test host}
mkdir -p "$W/bench"
cd "$W"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card" PYTHONPATH="$W/pypath"
export XDG_CACHE_HOME="$W/cache" TRITON_CACHE_DIR="$W/cache/triton" VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=10 OMP_PROC_BIND=close OMP_PLACES=cores
stamp=$(date -u +%Y%m%dT%H%M%SZ)
"$W/venv/bin/python" -m benchmarks.nvfp4_8r.bench_p3_miss_paths --packs "$W/packs" \
  --refs "$W/p2/refs-L05.npz" --card "$card" --build "$W/bench" --threads 10 \
  --output "$W/logs/p3-bench-$stamp.json" > "$W/logs/p3-bench-$stamp.log" 2>&1
