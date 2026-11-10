#!/bin/bash
# CPU only: P3 prefill references for layers 5/12/27, rank 0, M=64/512 x 2 samples.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
cd $LANE
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=engine/pypath OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pids=()
for L in 5 12 27; do
  engine/venv/bin/python -m benchmarks.nvfp4_8r.make_p2_refs \
    --weights models/DeepSeek-V4.1-Flash-1017-NVFP4 --verification-log logs/copy_weights.log \
    --tally engine/private/p16-tally.npz --layer "$L" --ranks 0 --samples 2 --ms 64 512 \
    --output "p2/refs-prefill-L$(printf %02d "$L").npz" > "logs/p3-refs-prefill-L$L.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "P3_PREFILL_REFS_EXIT=$rc"
exit $rc
