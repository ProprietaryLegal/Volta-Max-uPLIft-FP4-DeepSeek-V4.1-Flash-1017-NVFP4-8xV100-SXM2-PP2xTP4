#!/bin/bash
# memwatch.sh OUT SECONDS: card memory (index, MiB used) every 15 s for SECONDS, one line per sample, to OUT.
# nvidia-smi is called at most once per 15 s (house rule). Prints index and memory only, never a UUID.
set -euo pipefail
OUT=$1 END=$(( $(date +%s) + $2 ))
while [ "$(date +%s)" -lt "$END" ]; do
  echo "$(date -u +%T) $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ' ')" >> "$OUT"
  sleep 15
done
