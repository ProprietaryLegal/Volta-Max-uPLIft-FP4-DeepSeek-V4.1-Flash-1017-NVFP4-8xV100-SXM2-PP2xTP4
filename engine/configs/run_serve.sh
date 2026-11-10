#!/usr/bin/env bash
# run_serve.sh [--precheck-only] NAME: NVFP4-8R API on 127.0.0.1:18043, own process group.
# Fail-fast pre-check FIRST (before env.sh, before any CUDA): refuses if production's :18042 (or :18043) has a
# listener, or if any GPU on the 8-card host has a compute app. Production stays the default; this build starts only after
# production was stopped with bench5/stop-pp2.sh and must be followed by restore-default. Stop: kill -TERM -- -$(cat state/NAME.pid)
: "${LANE:?set LANE to the root directory of this build}"
set -uo pipefail
RUN=$LANE/run
precheck() {
  local port
  for port in 18042 18043; do
    if ss -Hltn "sport = :$port" | grep -q .; then
      echo "precheck: REFUSED, port $port has a listener (stop production first)" >&2
      return 1
    fi
  done
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || { echo "precheck: INCONCLUSIVE, nvidia-smi failed" >&2; return 1; }
  if [ -n "$apps" ]; then
    echo "precheck: REFUSED, GPU compute apps present: $(echo "$apps" | tr '\n' ' ')" >&2
    return 1
  fi
  echo "precheck: clear (no listener on 18042/18043, no GPU compute app)"
}
if [ "${1:-}" = --precheck-only ]; then precheck; exit $?; fi
precheck || exit 1
source "$RUN/env.sh" || exit 1
name=${1:-nvfp4}
# Production's lever set (read from the live production process environment 2026-11-09), with this lane's
# byte-identical copies of the helper libraries. The native expert path bypasses the IQ-only levers.
E=$LANE/engine
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SM70_MOE_ADD_ALLREDUCE=1 VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES=1
export VLLM_V41_PREFILL_FUSED_SO="$E/venv/lib/python3.12/site-packages/vllm/models/deepseek_v41/ds41_sparse_prefill.so"
export VLLM_V41_INDEX_TILED_LIB="$E/src/benchmarks/deepseek_v41/v41idx/build/v41idx.so"
export VLLM_V41_INDEX_FUSE_LIB="$E/src/benchmarks/deepseek_v41/v41pidx/build/v41pidx.so"
for f in "$VLLM_V41_PREFILL_FUSED_SO" "$VLLM_V41_INDEX_TILED_LIB" "$VLLM_V41_INDEX_FUSE_LIB" "$VLLM_V41K_LIB"; do
  [ -f "$f" ] || { echo "run_serve: missing library $f" >&2; exit 1; }
done
export VLLM_V41_ATTN_CHAIN=1 VLLM_V41_COUPLED_DRAFTS=1 VLLM_V41_COUPLED_DRAFT_TEMP=1.0 VLLM_V41_COUPLED_LOCAL=1 \
  VLLM_V41_COUPLED_SHARD=1 VLLM_V41_DMV_X=1 VLLM_V41_DOWN_ROWS=1 VLLM_V41_DRAFT_MP_SHARD=1 \
  VLLM_V41_ENGRAM_GATHER_ROWS=1024 VLLM_V41_EXPERT_DIET=3 VLLM_V41_EXPERT_FAST=1 VLLM_V41_FUSE_ATTN_POST=1 \
  VLLM_V41_FUSE_ATTN_PREP=1 VLLM_V41_HANDOFF_RELEASE=exit VLLM_V41_INDEX_FUSE=ilp4 VLLM_V41_INDEX_REDUCE=1 \
  VLLM_V41_INDEX_TILED=1 VLLM_V41_INDEX_TILE_MB=32 VLLM_V41_KEYS_SHARE=1 VLLM_V41_KV_HOT_POSITIONS=524288 \
  VLLM_V41_MARKOV_SHARD=1 VLLM_V41_PACKED_KV=1 VLLM_V41_PLAN_FUSE=1 VLLM_V41_PP_KEEP_BATCH=1 \
  VLLM_V41_PREFILL_FUSED_ATTN=1 VLLM_V41_PREFILL_FUSED_MODE=3 VLLM_V41_PREFILL_IQ_GEMM=1 \
  VLLM_V41_PREFILL_IQ_GRID=160 VLLM_V41_PREFILL_IQ_TILES=4,4,4,4,4,4 VLLM_V41_PREFILL_SCRATCH_REUSE=1 \
  VLLM_V41_PREFILL_SKIP_ROWS=3072 VLLM_V41_PREFILL_WOA_BATCHED=1 VLLM_V41_ROUTE_FAST=1 VLLM_V41_ROUTE_WARPS=4 \
  VLLM_V41_SESSION_PREFIX=1 VLLM_V41_SUM2_PUSH_ANY=1
if [ "$V41PKG_PP" = 1 ] || [ "${V41_PP2_ASYNC:-0}" = 1 ]; then export VLLM_V41_ASYNC=1; else unset VLLM_V41_ASYNC; fi
# Prompt slots (P11; production's "a" modes with the caller sending one request at a time):
# NVFP4_SLOTS=2|3 gives every slot its own kept prompt, so chats that take turns keep theirs. Values are
# production's slot policy (BURST 2). Never VLLM_V41_MULTI_ROWS (the batched mode that lost).
# The profile's max-num-seqs must say the same number: a mismatch is refused here, not discovered at load.
SLOTS=${NVFP4_SLOTS:-1}
PROFILE_PATH="$RUN/${PROFILE:-$V41PKG_DEFAULT_PROFILE}"
seqs=$(awk '$1 == "max-num-seqs:" {print $2}' "$PROFILE_PATH")
[ "$seqs" = "$SLOTS" ] || { echo "run_serve: NVFP4_SLOTS=$SLOTS but $PROFILE_PATH has max-num-seqs: '$seqs'" >&2; exit 1; }
unset VLLM_V41_MULTI_ROWS VLLM_V41_MULTI_SEQ VLLM_V41_MULTI_SEQ_SINGLE_PREFILL_BUDGET VLLM_V41_MULTI_SEQ_DECODE_TURNS \
  VLLM_V41_MULTI_SEQ_PREFILL_BURST VLLM_V41_MULTI_SEQ_PREFILL_CHUNK
case "$SLOTS" in
  1) ;;
  2|3) export VLLM_V41_MULTI_SEQ=1 VLLM_V41_MULTI_SEQ_SINGLE_PREFILL_BUDGET=1 VLLM_V41_MULTI_SEQ_DECODE_TURNS=16 \
         VLLM_V41_MULTI_SEQ_PREFILL_BURST=2 VLLM_V41_MULTI_SEQ_PREFILL_CHUNK=0 ;;
  *) echo "run_serve: NVFP4_SLOTS must be 1, 2 or 3" >&2; exit 1 ;;
esac
setsid "$PY" -m vllm.entrypoints.openai.api_server --config "$RUN/${PROFILE:-$V41PKG_DEFAULT_PROFILE}" > "$RUN/state/$name.log" 2>&1 &
echo $! > "$RUN/state/$name.pid"; echo "serve pid/pgid $(cat "$RUN/state/$name.pid"), log $RUN/state/$name.log"
