#!/bin/bash
# Run ON THE TEST HOST only, one process on one card (plan 03 steps 1, 4, 6).
# Usage: run_fuse_testhost.sh CARD TAG MODULE [module args...]
#   The module runs under run_staged with STAGE=$S/src laid over the installed engine.
#   Every "{OUT}" in the module args is replaced with the receipt path stem.
# Gate (one nvidia-smi call): no compute process on the card, power limit <= 150 W,
# MemAvailable >= 60 GiB. Kernel log: the count grep -ciE "xid|NVRM" and fault lines
# before and after; a new fault line ends the run INCONCLUSIVE+ (exit 4). Kills nothing.
set -uo pipefail
card=${1:?card}; tag=${2:?tag}; module=${3:?module}; shift 3
W=${W:?set W to the work directory on the test host}
S=$W/stage/dispatch
STAGE=${FUSE_STAGE:?set FUSE_STAGE to a run_staged tree, or engine/src}
cd "$W" || exit 3
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card" PYTHONPATH="$W/pypath"
export XDG_CACHE_HOME="$W/cache" TRITON_CACHE_DIR="$W/cache/triton" VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=8
export VLLM_V41K_LIB="$W/t3/v41k.so"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$W/logs/dispatch"
out=$W/logs/dispatch/$tag-$stamp
smi=$(nvidia-smi -q -i "$card" -d PIDS,POWER) || { echo "INCONCLUSIVE: nvidia-smi failed" | tee "$out.status"; exit 3; }
procs=$(printf '%s\n' "$smi" | awk -F: '/^    Processes/{gsub(/ /,"",$2); print $2}')
[ "$procs" = "None" ] || { echo "STOP: card $card has compute processes" | tee "$out.status"; exit 3; }
pw=$(printf '%s\n' "$smi" | awk -F: '/^        Current Power Limit/{gsub(/[ W]/,"",$2); print $2; exit}')
awk -v p="$pw" 'BEGIN{exit !(p+0 > 0 && p+0 <= 150.0)}' || { echo "STOP: card $card power limit '$pw' W" | tee "$out.status"; exit 3; }
avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
[ "$avail" -ge $((60*1024*1024)) ] || { echo "STOP: MemAvailable $avail kB < 60 GiB" | tee "$out.status"; exit 3; }
klog=$(sudo -n dmesg) || { echo "INCONCLUSIVE: the kernel log cannot be read" | tee "$out.status"; exit 3; }
lead_before=$(printf '%s\n' "$klog" | grep -ciE "xid|NVRM")
fault_before=$(printf '%s\n' "$klog" | grep -cE "NVRM: Xid|DMAR: \[DMA|DRHD: handling fault|BUG uvm")
gate=$(printf '{"card": %s, "power_limit_w": %s, "mem_available_kb": %s, "compute_processes": "none", "dmesg_xid_nvrm_before": %s, "fault_lines_before": %s}' \
  "$card" "$pw" "$avail" "$lead_before" "$fault_before")
echo "gate: $gate" | tee "$out.status"
args=()
for x in "$@"; do args+=("${x//\{OUT\}/$out}"); done
args=("${args[@]//\{GATE\}/$gate}")
run=("$W/venv/bin/python" -m benchmarks.nvfp4_8r.run_staged "$STAGE" "$module" "${args[@]}")
echo "cmd: ${run[*]}" >> "$out.status"
timeout --kill-after=30 "${F_LIMIT_S:-2400}" "${run[@]}" > "$out.log" 2>&1
rc=$?
tail -1 "$out.log" | cut -c1-1500 | tee -a "$out.status"
klog=$(sudo -n dmesg) || { echo "INCONCLUSIVE: the kernel log cannot be read after the run" | tee -a "$out.status"; exit 3; }
lead_after=$(printf '%s\n' "$klog" | grep -ciE "xid|NVRM")
fault_after=$(printf '%s\n' "$klog" | grep -cE "NVRM: Xid|DMAR: \[DMA|DRHD: handling fault|BUG uvm")
echo "dmesg xid|NVRM after $lead_after, fault lines after $fault_after" | tee -a "$out.status"
echo "FUSE_RUN_EXIT=$rc stem $out" | tee -a "$out.status"
[ "$fault_after" = "$fault_before" ] && [ "$lead_after" = "$lead_before" ] || { echo "INCONCLUSIVE+: the run added kernel log lines" | tee -a "$out.status"; exit 4; }
exit "$rc"
