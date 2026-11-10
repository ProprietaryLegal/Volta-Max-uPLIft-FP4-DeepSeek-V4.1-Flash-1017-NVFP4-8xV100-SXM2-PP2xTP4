#!/bin/bash
# Run ON THE TEST HOST only. Usage: run_t3_layer_testhost.sh CARD
# A3 of plan 01 (accept_t3_layer.py): four rank processes on ONE card build the three-tier layer with the
# product functions on real packs of layers 5/12/27 and their IQ slices. The gate refuses (exit 3) on any
# compute app, a power limit > 150 W or MemAvailable < 120 GiB. Never changes caps. Kills only its own four PIDs.
set -uo pipefail
card=${1:?card index}
W=${W:?set W to the work directory on the test host}
T=$W/t3
cd "$W" || exit 3
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card" PYTHONPATH="$W/pypath"
export XDG_CACHE_HOME="$W/cache" TRITON_CACHE_DIR="$W/cache/triton" VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=2
# The server's lever set for the IQ expert path (run/run_serve.sh), with the staged copy of its library.
export VLLM_V41K_LIB="$T/v41k.so" VLLM_V41_DOWN_ROWS=1 VLLM_V41_EXPERT_FAST=1 VLLM_V41_PLAN_FUSE=1 VLLM_V41_EXPERT_DIET=3
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out=$W/logs/t3-layer-$stamp
refs=("$T/t3-refs-L05.npz" "$T/t3-refs-L12.npz" "$T/t3-refs-L27.npz")
for f in "$VLLM_V41K_LIB" "$T/iq-manifest.json" "$W/packs/manifest.json" "${refs[@]}"; do
  [ -s "$f" ] || { echo "INCONCLUSIVE: missing input $f" | tee "$out.status"; exit 3; }
done
py=$W/venv/bin/python
"$py" -m benchmarks.nvfp4_8r.accept_t3_layer --gate-only --card "$card" > "$out.gate.log" 2>&1 \
  || { echo "INCONCLUSIVE: gate refused, see $out.gate.log" | tee "$out.status"; exit 3; }
port=$((20000 + RANDOM % 20000))
pids=()
for r in 0 1 2 3; do
  "$py" -m benchmarks.nvfp4_8r.accept_t3_layer --card "$card" --rank "$r" --port "$port" \
    --packs "$W/packs" --slices "$T" --refs "${refs[@]}" --output "$out.json" > "$out.r$r.log" 2>&1 &
  pids+=($!)
done
# The ranks advance in lockstep: one that stops leaves the others waiting in a collective.
limit=${T3_LIMIT_S:-3000}
rc=0; early=0; reaped=(0 0 0 0)
while :; do
  alive=0
  for i in 0 1 2 3; do
    [ "${reaped[$i]}" = 1 ] && continue
    if kill -0 "${pids[$i]}" 2>/dev/null; then alive=$((alive + 1)); continue; fi
    wait "${pids[$i]}"; code=$?; reaped[$i]=1
    [ "$code" -gt "$rc" ] && rc=$code
    [ "$code" -ne 0 ] && early=1
  done
  [ "$alive" -eq 0 ] && break
  if [ "$early" = 1 ] || [ "$SECONDS" -gt "$limit" ]; then
    sleep 5
    for i in 0 1 2 3; do [ "${reaped[$i]}" = 1 ] || kill "${pids[$i]}" 2>/dev/null; done
    [ "$rc" -eq 0 ] && rc=3
    echo "INCONCLUSIVE: ranks stopped (exit code $rc, after ${SECONDS}s, limit ${limit}s); see $out.r*.log" | tee "$out.status"
    exit "$rc"
  fi
  sleep 2
done
echo "T3_LAYER_EXIT=$rc receipt $out.json" | tee "$out.status"
exit "$rc"
