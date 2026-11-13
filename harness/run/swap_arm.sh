#!/bin/bash
# swap_arm.sh PREV NEXT PROFILE [SLOTS=1]
# Stop the lane server PREV (by the process group its own launcher recorded in run/state/PREV.pid), then boot NEXT on
# port 18043 with PROFILE and wait until /health answers 200. PREV may be "-" when nothing is serving.
# ARM_PORT (environment, default 18043) is the port: 18043 for test arms and windows; 18042 only from
# `launch.sh uplift` (the default build), and then PREV is "-". The profile's own port line must name the same port.
# Fails loudly: a server that dies during load, or a group that does not exit, ends the script non-zero.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
cd $LANE
prev=$1; next=$2; profile=$3; slots=${4:-1}
port=${ARM_PORT:-18043}
case "$port" in 18042 | 18043) ;; *) echo "swap_arm: ARM_PORT must be 18042 or 18043, got '$port'" >&2; exit 1 ;; esac
url=http://127.0.0.1:$port
[ -f "run/$profile" ] || { echo "swap_arm: no profile run/$profile" >&2; exit 1; }
pport=$(awk '$1 == "port:" {print $2}' "run/$profile" | tr '\n' ' ' | sed 's/ $//')
[ "$pport" = "$port" ] || { echo "swap_arm: run/$profile has port line '$pport' but this boot waits on port $port" >&2; exit 1; }
[ "$port" = 18043 ] || [ "$prev" = "-" ] || { echo "swap_arm: on port $port PREV must be '-' (the default is booted from nothing serving)" >&2; exit 1; }
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
if curl -s -o /dev/null "$url/health"; then echo "swap_arm: port $port still answers" >&2; exit 1; fi
t0=$SECONDS
# Nine cards since 2026-11-11: the operator added an RTX 3090, nvidia-smi index 6 in PCI bus order. The eight V100s are the
# indices 0 1 2 3 4 5 7 8. Board B = 0 1 4 5, Board C = 2 3 7 8 (nvidia-smi topo -m, 2026-11-11 22:03Z). Before that
# day the list was 0,4,1,5,6,2,3,7. The list below is right for exactly this layout, so the layout is checked first
# (sysfs, no nvidia-smi): any other set of NVIDIA cards refuses the boot.
want="0000:03:00.0:0x1db5 0000:06:00.0:0x1db5 0000:25:00.0:0x1db5 0000:26:00.0:0x1db5 0000:29:00.0:0x1db5 0000:2c:00.0:0x1db5 0000:41:00.0:0x2204 0000:63:00.0:0x1db5 0000:64:00.0:0x1db5"
have=$(for d in /sys/bus/pci/devices/*; do
         [ "$(cat "$d/vendor")" = 0x10de ] || continue
         case "$(cat "$d/class")" in 0x0302* | 0x0300*) echo "$(basename "$d"):$(cat "$d/device")" ;; esac
       done | sort | tr '\n' ' ' | sed 's/ $//')
[ "$have" = "$want" ] || { echo "swap_arm: REFUSED, the NVIDIA cards on the PCI bus are not the nine-card layout of 2026-11-11 (have: $have); the card index list is wrong for any other layout" >&2; exit 1; }
# 2026-11-12 (after the RAM upgrade to 1 TB the host reports two NUMA nodes): the stage-1 order is 2,7,3,8, not
# 7,2,3,8. The PP hand-off pairs are (rank i, rank i+4). With 7,2,3,8 the pair (index 4, index 2) joined the two
# cards behind one PCIe host bridge, and since the upgrade a direct peer-to-peer transfer between two cards behind
# one host bridge hangs on this host (a pair-by-pair probe: the pairs behind one bridge hung, the pairs across two
# bridges passed). With 2,7,3,8 the pairs are (0,2) (4,7) (1,3) (5,8): each crosses two different host bridges.
# Stage 0 is still Board B and stage 1 still Board C.
CUDA_VISIBLE_DEVICES=0,4,1,5,2,7,3,8 PROFILE=$profile NVFP4_SLOTS=$slots bash run/run_serve.sh "$next"
pid=$(cat "run/state/$next.pid")
echo "LAUNCHED $next profile $profile slots $slots pgid $pid $(date -u +%H:%M:%SZ)"
until [ "$(curl -s -o /dev/null -w '%{http_code}' "$url/health")" = 200 ]; do
  kill -0 "$pid" 2> /dev/null || { echo "swap_arm: $next died during load; see run/state/$next.log" >&2
                                   grep -E "Error|Traceback" "run/state/$next.log" | tail -5 | cut -c1-240 >&2; exit 1; }
  sleep 5
done
echo "READY $next $(date -u +%H:%M:%SZ) after $((SECONDS - t0)) s; tracebacks $(grep -c Traceback "run/state/$next.log" || true)"
grep -E "TwoTier|HotIq|free after" "run/state/$next.log" | cut -c1-330 | sort -u | tail -6
