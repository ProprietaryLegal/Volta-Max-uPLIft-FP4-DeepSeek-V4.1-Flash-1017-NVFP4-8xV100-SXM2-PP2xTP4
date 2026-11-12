#!/bin/bash
# memwatch.sh OUT SECONDS: card memory (V100 number 0..7, MiB used) every 15 s for SECONDS, one line per sample, to OUT.
# Nine cards since 2026-11-11 (an RTX 3090 is nvidia-smi index 6): only the eight V100s are listed, numbered 0..7 in
# nvidia-smi order, which are the indices these lines held before the 3090 (stage 0 = 0 1 4 5, stage 1 = 2 3 6 7).
# nvidia-smi is called at most once per 15 s (house rule). Prints index and memory only, never a UUID.
set -euo pipefail
OUT=$1 END=$(( $(date +%s) + $2 ))
while [ "$(date +%s)" -lt "$END" ]; do
  echo "$(date -u +%T) $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', *' '/V100/ {printf "%d,%d ", n, $2; n++}')" >> "$OUT"
  sleep 15
done
