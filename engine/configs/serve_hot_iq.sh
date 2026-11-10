#!/usr/bin/env bash
# serve_hot_iq.sh PKG [PORT]: start the hot_iq build from the Hugging Face package unpacked at PKG (an absolute path).
# It writes the package's profile template with PKG and PORT (default 8000, loopback) filled in to
# $LANE/run/serve-hot_iq.yaml, then starts run_serve.sh with that profile and one prompt slot. The server's log and
# pid are $LANE/run/state/hot_iq.log and .pid. Stop: kill -TERM -- -$(cat "$LANE/run/state/hot_iq.pid")
# It sets the load-time allocator setting VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0, as the hot_iq measurements of
# 2026-11-10 ran (docs/measurements.md). run_serve.sh itself does not set it.
: "${LANE:?set LANE to the root directory of this build}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the first board's four indices, then the second board's}"
set -euo pipefail
die() { echo "serve_hot_iq: $*" >&2; exit 1; }
[ $# -ge 1 ] && [ $# -le 2 ] || die "usage: serve_hot_iq.sh PKG [PORT]"
PKG=${1%/}
PORT=${2:-8000}
case "$PKG" in /*) ;; *) die "PKG must be an absolute path: $PKG" ;; esac
case "$PKG" in *'#'*|*$'\n'*) die "PKG must not contain '#' or a newline: $PKG" ;; esac
case "$PORT" in ''|*[!0-9]*) die "PORT must be a number: $PORT" ;; esac
for f in serve/hot_iq.yaml.template tier_map.json experts-mxfp4/manifest.json; do
  [ -f "$PKG/$f" ] || die "not a hot_iq package: $PKG/$f is missing"
done
[ -f "$LANE/run/run_serve.sh" ] || die "$LANE/run/run_serve.sh is missing (README.md, Reproduce)"
mkdir -p "$LANE/run/state"
PROFILE_PATH="$LANE/run/serve-hot_iq.yaml"
sed -e "s#\${PKG}#$PKG#g" -e "s#^port: 8000\$#port: $PORT#" "$PKG/serve/hot_iq.yaml.template" > "$PROFILE_PATH"
if grep -q '\${PKG}' "$PROFILE_PATH"; then die "$PROFILE_PATH still holds \${PKG}"; fi
grep -qx "port: $PORT" "$PROFILE_PATH" || die "$PROFILE_PATH has no line 'port: $PORT'"
grep -qx "max-num-seqs: 1" "$PROFILE_PATH" || die "$PROFILE_PATH is not the one-slot profile"
export VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0
PROFILE=serve-hot_iq.yaml NVFP4_SLOTS=1 exec bash "$LANE/run/run_serve.sh" hot_iq
