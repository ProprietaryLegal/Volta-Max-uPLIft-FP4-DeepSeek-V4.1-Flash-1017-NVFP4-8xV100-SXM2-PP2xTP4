#!/usr/bin/env bash
# serve_hot_iq.sh PKG [PORT]: start the hot_iq build from the Hugging Face package unpacked at PKG (an absolute path).
# NVFP4_OPTION picks the serving option (default slots2, the published default):
#   slots1 slots2 slots3          1 to 3 prompt slots, fast prompt reading (compressed experts read from their IQ
#                                 records) with the two prompt-reading kernels (nvfp4_8r_prefill_swiglu: kernel,
#                                 nvfp4_8r_prefill_rb: rb4). Quality against native reading: not established.
#   native-slots1                 one slot, native prompt reading (every expert read from its NVFP4 record), the
#                                 reading of the closeness figures, with the kernels' fallback values (chain, rb2).
# Each option is the package's serve/hot_iq-OPTION.yaml.template with its tier map tier_maps/OPTION.json. The
# script writes the template with PKG and PORT (default 8000, loopback) filled in to $LANE/run/serve-hot_iq.yaml,
# then starts run_serve.sh with that profile and the template's slot count. The server's log and pid are
# $LANE/run/state/hot_iq.log and .pid. Stop: kill -TERM -- -$(cat "$LANE/run/state/hot_iq.pid")
# The package as first uploaded (2026-11-10: serve/hot_iq.yaml.template and tier_map.json only) does not boot on
# this engine: it lacks the keys nvfp4_8r_group_plan, nvfp4_8r_swiglu and nvfp4_8r_iq_tail, and templates downloaded
# before 2026-11-12 lack nvfp4_8r_prefill_swiglu and nvfp4_8r_prefill_rb. Download the package's serve/ and
# tier_maps/ folders again (small files, README.md, Reproduce).
: "${LANE:?set LANE to the root directory of this build}"
# CUDA_VISIBLE_DEVICES: the first board's four indices, then the second board's. Pair each card with the one
# four places later across two PCIe host bridges if your host has more than one NUMA node (our list is
# 0,4,1,5,2,7,3,8).
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the first board's four indices, then the second board's}"
set -euo pipefail
die() { echo "serve_hot_iq: $*" >&2; exit 1; }
[ $# -ge 1 ] && [ $# -le 2 ] || die "usage: serve_hot_iq.sh PKG [PORT]"
PKG=${1%/}
PORT=${2:-8000}
OPT=${NVFP4_OPTION:-slots2}
case "$PKG" in /*) ;; *) die "PKG must be an absolute path: $PKG" ;; esac
case "$PKG" in *'#'*|*$'\n'*) die "PKG must not contain '#' or a newline: $PKG" ;; esac
case "$PORT" in ''|*[!0-9]*) die "PORT must be a number: $PORT" ;; esac
case "$OPT" in
  slots1|slots2|slots3) READ=iq; SLOTS=${OPT#slots} ;;
  native-slots1) READ=native_staged; SLOTS=1 ;;
  *) die "NVFP4_OPTION must be slots1, slots2, slots3 or native-slots1: $OPT" ;;
esac
T="$PKG/serve/hot_iq-$OPT.yaml.template"
[ -f "$PKG/experts-mxfp4/manifest.json" ] || die "not a hot_iq package: $PKG/experts-mxfp4/manifest.json is missing"
[ -f "$T" ] && [ -f "$PKG/tier_maps/$OPT.json" ] || die "$T or $PKG/tier_maps/$OPT.json is missing: download the package's serve/ and tier_maps/ folders again"
grep -qx "  nvfp4_8r_prefill_warm: $READ" "$T" || die "$T has no line 'nvfp4_8r_prefill_warm: $READ'"
grep -qx "  nvfp4_8r_tier_map: \${PKG}/tier_maps/$OPT.json" "$T" || die "$T does not name tier_maps/$OPT.json"
grep -qx "max-num-seqs: $SLOTS" "$T" || die "$T is not the $SLOTS-slot profile"
for k in nvfp4_8r_group_plan nvfp4_8r_swiglu nvfp4_8r_prefill_swiglu nvfp4_8r_prefill_rb nvfp4_8r_iq_tail; do
  [ "$(grep -c "^  $k: " "$T")" = 1 ] || die "$T has no single $k line"
done
[ -f "$LANE/run/run_serve.sh" ] || die "$LANE/run/run_serve.sh is missing (README.md, Reproduce)"
mkdir -p "$LANE/run/state"
PROFILE_PATH="$LANE/run/serve-hot_iq.yaml"
sed -e "s#\${PKG}#$PKG#g" -e "s#^port: 8000\$#port: $PORT#" "$T" > "$PROFILE_PATH"
if grep -q '\${PKG}' "$PROFILE_PATH"; then die "$PROFILE_PATH still holds \${PKG}"; fi
grep -qx "port: $PORT" "$PROFILE_PATH" || die "$PROFILE_PATH has no line 'port: $PORT'"
echo "serve_hot_iq: option $OPT, $SLOTS slot(s), reading $READ"
# run_serve.sh sets the load-time allocator setting VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0 itself.
PROFILE=serve-hot_iq.yaml NVFP4_SLOTS=$SLOTS exec bash "$LANE/run/run_serve.sh" hot_iq
