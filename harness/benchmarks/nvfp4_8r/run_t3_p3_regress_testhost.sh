#!/bin/bash
# Run ON THE TEST HOST only. Usage: run_t3_p3_regress_testhost.sh CARD
# Plan 01 regression: the unchanged P3 layer acceptance (accept_p3_layer: hot/cold split bit-equal to all-resident,
# decode graphs, prefill) against the nvfp4_8r_native.py that now also holds the three-tier code. Same command as
# run_p3_layer_testhost.sh; only the gate is the T3 one (nvidia-smi calls 15 s apart, MemAvailable >= 120 GiB).
set -euo pipefail
card=${1:?card index}
W=${W:?set W to the work directory on the test host}
cd "$W"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card" PYTHONPATH="$W/pypath"
export XDG_CACHE_HOME="$W/cache" TRITON_CACHE_DIR="$W/cache/triton" VLLM_NO_USAGE_STATS=1
stamp=$(date -u +%Y%m%dT%H%M%SZ)
"$W/venv/bin/python" -c '
import sys
from benchmarks.nvfp4_8r import accept_p2, accept_p3_layer, falsify_t3_iq_drop
def gate(card):
    try:
        return falsify_t3_iq_drop.gate(card)
    except falsify_t3_iq_drop.Inconclusive as exc:
        raise accept_p2.Inconclusive(str(exc)) from exc
accept_p3_layer.gpu_gate = gate
sys.exit(accept_p3_layer.main())
' --packs "$W/packs" \
  --refs "$W/p2/refs-L05.npz" "$W/p2/refs-L12.npz" "$W/p2/refs-L27.npz" \
  --prefill-refs "$W/p2/refs-prefill-L05.npz" "$W/p2/refs-prefill-L12.npz" "$W/p2/refs-prefill-L27.npz" \
  --card "$card" --output "$W/logs/t3-p3-regress-$stamp.json" > "$W/logs/t3-p3-regress-$stamp.log" 2>&1
