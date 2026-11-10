#!/bin/bash
# The source readers require ALL-VERIFIED before accessing any checkpoint shard.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
cd $LANE
export CUDA_VISIBLE_DEVICES=''
export UV_CACHE_DIR=$LANE/engine/uv-cache
export PYTHONPATH=$LANE/engine/src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
trap 'result=$?; printf "P1_EXIT=%s\n" "$result" >> logs/p1-job.log' EXIT

uv run --no-project --python engine/venv/bin/python python \
    -m benchmarks.nvfp4_8r.probe_p1 \
    --weights models/DeepSeek-V4.1-Flash-1017-NVFP4 \
    --verification-log logs/copy_weights.log > logs/p1-probe.log 2>&1
uv run --no-project --python engine/venv/bin/python python \
    -m benchmarks.nvfp4_8r.build_packs \
    --weights models/DeepSeek-V4.1-Flash-1017-NVFP4 \
    --verification-log logs/copy_weights.log \
    --tally engine/private/p16-tally.npz --output packs --workers 8 > logs/p1-build.log 2>&1
uv run --no-project --python engine/venv/bin/python python \
    -m benchmarks.nvfp4_8r.accept_p1 \
    --weights models/DeepSeek-V4.1-Flash-1017-NVFP4 \
    --verification-log logs/copy_weights.log \
    --tally engine/private/p16-tally.npz \
    --packs packs --output logs/p1-acceptance.json > logs/p1-acceptance.log 2>&1
