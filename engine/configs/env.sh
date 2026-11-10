# NVFP4-8R environment (PP=2 x TP4 on the 8-card host's 8 V100s). Same structure as production's run-pp2/env.sh; every
# path is this lane's own copy under $LANE (checked by check_p5_paths.py). Not the default
# build: launch only after production is stopped.
: "${LANE:?set LANE to the root directory of this build}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the Board B indices then Board C (PP2), as production}"
export V41PKG_PP=2 V41PKG_DEFAULT_PROFILE=serve-nvfp4-pp2.yaml
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# site-packages first (the wheel's compiled ops), engine/src only for benchmarks.
export PYTHONPATH="$LANE/engine/venv/lib/python3.12/site-packages:$LANE/engine/src:$LANE/run/tools" XDG_CACHE_HOME="$LANE/run/cache" XDG_CONFIG_HOME="$LANE/run/cache/config"
export FLASHINFER_WORKSPACE_BASE="$LANE/run/cache" TRITON_CACHE_DIR="$LANE/run/cache/triton" VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1
export VLLM_V41K_LIB="$LANE/engine/src/benchmarks/deepseek_v41/v41k/build/v41k.so"
PY=$LANE/engine/venv/bin/python
cd "$LANE/run"
# fail fast: vllm must come from this lane's venv copy, with its compiled ops
"$PY" -c 'import sys, vllm, vllm._C, vllm._C_gguf; p = vllm.__file__; sys.exit(0 if p.startswith(sys.argv[1]) else "vllm imported from " + p)' "$LANE/engine/venv/lib/python3.12/site-packages" || { echo "env.sh: engine import check failed" >&2; return 1; }
