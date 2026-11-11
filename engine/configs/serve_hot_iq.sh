#!/usr/bin/env bash
# serve_hot_iq.sh PKG [PORT]: start the hot_iq build from the Hugging Face package unpacked at PKG (an absolute path).
# It writes the package's profile template with PKG and PORT (default 8000, loopback) filled in to
# $LANE/run/serve-hot_iq.yaml, then starts run_serve.sh with that profile and one prompt slot. The server's log and
# pid are $LANE/run/state/hot_iq.log and .pid. Stop: kill -TERM -- -$(cat "$LANE/run/state/hot_iq.pid")
# It sets the load-time allocator setting VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0, as the hot_iq measurements of
# 2026-11-10 ran (docs/measurements.md). run_serve.sh itself does not set it.
# The engine requires nvfp4_8r_group_plan (chain or kernel, no default). NVFP4_GROUP_PLAN=chain|kernel sets it in
# the written profile. Unset, the template's own value is used. A template without the key (the package as first
# uploaded, 2026-11-10) needs NVFP4_GROUP_PLAN: chain is the configuration that package was measured with.
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
GP=${NVFP4_GROUP_PLAN:-}
set_gp=()
case "$GP" in ''|chain|kernel) ;; *) die "NVFP4_GROUP_PLAN must be chain or kernel: $GP" ;; esac
T="$PKG/serve/hot_iq.yaml.template"
grep -qx '  nvfp4_8r_prefill_warm: native_staged' "$T" || die "$T has no line 'nvfp4_8r_prefill_warm: native_staged'"
if grep -q '^  nvfp4_8r_group_plan: ' "$T"; then
  [ -z "$GP" ] || set_gp=(-e "s#^  nvfp4_8r_group_plan: .*\$#  nvfp4_8r_group_plan: $GP#")
else
  [ -n "$GP" ] || die "$T has no nvfp4_8r_group_plan: set NVFP4_GROUP_PLAN=chain (as that package was measured) or kernel"
  set_gp=(-e "s#^  nvfp4_8r_prefill_warm: native_staged\$#&\n  nvfp4_8r_group_plan: $GP#")
fi
[ -f "$LANE/run/run_serve.sh" ] || die "$LANE/run/run_serve.sh is missing (README.md, Reproduce)"
mkdir -p "$LANE/run/state"
PROFILE_PATH="$LANE/run/serve-hot_iq.yaml"
sed -e "s#\${PKG}#$PKG#g" -e "s#^port: 8000\$#port: $PORT#" "${set_gp[@]}" "$T" > "$PROFILE_PATH"
if grep -q '\${PKG}' "$PROFILE_PATH"; then die "$PROFILE_PATH still holds \${PKG}"; fi
grep -qx "port: $PORT" "$PROFILE_PATH" || die "$PROFILE_PATH has no line 'port: $PORT'"
grep -qx "max-num-seqs: 1" "$PROFILE_PATH" || die "$PROFILE_PATH is not the one-slot profile"
[ "$(grep -cE '^  nvfp4_8r_group_plan: (chain|kernel)$' "$PROFILE_PATH")" = 1 ] || die "$PROFILE_PATH has no single nvfp4_8r_group_plan line"
echo "serve_hot_iq: $(grep '^  nvfp4_8r_group_plan: ' "$PROFILE_PATH" | sed 's/^ *//')"
export VLLM_WEIGHT_LOAD_MAX_SPLIT_SCOPE=0
PROFILE=serve-hot_iq.yaml NVFP4_SLOTS=1 exec bash "$LANE/run/run_serve.sh" hot_iq
