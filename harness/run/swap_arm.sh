#!/bin/bash
# swap_arm.sh PREV NEXT PROFILE [SLOTS=1]
# Stop the lane server PREV (by the process group its own launcher recorded in run/state/PREV.pid), then boot NEXT on
# port 18043 with PROFILE and wait until /health answers 200. PREV may be "-" when nothing is serving.
# Fails loudly: a server that dies during load, or a group that does not exit, ends the script non-zero.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
cd $LANE
prev=$1; next=$2; profile=$3; slots=${4:-1}
url=http://127.0.0.1:18043
[ -f "run/$profile" ] || { echo "swap_arm: no profile run/$profile" >&2; exit 1; }
[ ! -e "run/state/$next.pid" ] || { echo "swap_arm: run/state/$next.pid exists; pick a new name" >&2; exit 1; }
if [ "$prev" != "-" ]; then
  pg=$(cat "run/state/$prev.pid")
  tr '\0' ' ' < "/proc/$pg/cmdline" | grep -q "vllm.entrypoints.openai.api_server" \
    || { echo "swap_arm: pid $pg is not the lane server of $prev" >&2; exit 1; }
  running=$(curl -s "$url/metrics" | awk '/^vllm:num_requests_running/ {s += $NF} END {print s + 0}')
  [ "$running" = 0 ] || { echo "swap_arm: $prev has $running requests running" >&2; exit 1; }
  kill -TERM -- "-$pg"
  deadline=$((SECONDS + 180))
  while pgrep -g "$pg" > /dev/null; do
    [ $SECONDS -lt $deadline ] || { echo "swap_arm: group $pg still alive after 180 s" >&2; exit 1; }
    sleep 2
  done
  echo "STOPPED $prev (pgid $pg) $(date -u +%H:%M:%SZ)"
  sleep 20
fi
if curl -s -o /dev/null "$url/health"; then echo "swap_arm: port 18043 still answers" >&2; exit 1; fi
t0=$SECONDS
CUDA_VISIBLE_DEVICES=0,4,1,5,6,2,3,7 PROFILE=$profile NVFP4_SLOTS=$slots bash run/run_serve.sh "$next"
pid=$(cat "run/state/$next.pid")
echo "LAUNCHED $next profile $profile slots $slots pgid $pid $(date -u +%H:%M:%SZ)"
until [ "$(curl -s -o /dev/null -w '%{http_code}' "$url/health")" = 200 ]; do
  kill -0 "$pid" 2> /dev/null || { echo "swap_arm: $next died during load; see run/state/$next.log" >&2
                                   grep -E "Error|Traceback" "run/state/$next.log" | tail -5 | cut -c1-240 >&2; exit 1; }
  sleep 5
done
echo "READY $next $(date -u +%H:%M:%SZ) after $((SECONDS - t0)) s; tracebacks $(grep -c Traceback "run/state/$next.log" || true)"
grep -E "TwoTier|HotIq|free after" "run/state/$next.log" | cut -c1-330 | sort -u | tail -6
