#!/bin/bash
# CPU only (the 8-card host): plan 04 part B references and IQ slices, one layer after another, 4 threads.
# Layers 5, 12, 27: native + IQ references (mixed-tier acceptance). Layers 10, 39: IQ only
# (down codecs IQ2_XXS and IQ2_XS for the falsifier) and their per-rank IQ slices.
: "${LANE:?set LANE to the root directory of this build}"
set -uo pipefail
cd $LANE
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=engine/pypath OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
G=models/gguf/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf
mkdir -p p2/f logs/prefill
rc=0
engine/venv/bin/python -m benchmarks.nvfp4_8r.make_t3_iq_slices --gguf "$G" --layers 10 39 \
  --ranks 0 1 2 3 --output p2/f > logs/prefill/make-iq-slices.log 2>&1 || rc=1
for spec in "12:" "5:" "27:" "10:--iq-only" "39:--iq-only"; do
  L=${spec%%:*}; flag=${spec#*:}
  # shellcheck disable=SC2086
  engine/venv/bin/python -m benchmarks.nvfp4_8r.make_f_prefill_refs \
    --weights models/DeepSeek-V4.1-Flash-1017-NVFP4 --verification-log logs/copy_weights.log \
    --gguf "$G" --layer "$L" $flag --output "p2/f/f-refs-L$(printf %02d "$L").npz" \
    > "logs/prefill/make-refs-L$L.log" 2>&1 || rc=1
  echo "L$L done rc=$rc $(date -u +%H:%M:%SZ)"
done
echo "F_PREFILL_REFS_EXIT=$rc"
exit $rc
