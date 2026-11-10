#!/bin/bash
# arm_measure.sh LABEL MODE [LONG]: measure one booted arm of the lane build at :18043, one request at a time.
#   MODE=short: measure_boot only (warm-up, general + legal-style public prompts, within-boot repeat).
#   MODE=full : the same, then the context curve (46 / 31K / 107K / 250K tokens) and the harness-shaped tool loop
#               (216K-token prompt, thinking on, 8 requests).
#   LONG (0|1|2) is passed to measure_boot.sh (2 = 100K + 500K memory check for builds that change card memory use).
# Receipts: logs/baseline/LABEL-*. No retries: the first failure stops the script.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
R=$LANE; D=$R/logs/baseline; U=http://127.0.0.1:18043
LABEL=$1 MODE=$2 LONG=${3:-0}
echo "ARM $LABEL $MODE start $(date -u +%H:%M:%SZ)"
bash "$R/run/measure_boot.sh" "$U" "$LABEL" "$LONG"
if [ "$MODE" = full ]; then
  for c in 0 120000 420000 1000000; do /usr/bin/python3 -I "$R/run/ctx_tput.py" "$U" "$c" "$D/$LABEL-ctx.jsonl" 512; done
  /usr/bin/python3 -I "$R/run/harness_loop.py" "$U" 830000 8 "$D/$LABEL-loop-think1.jsonl" 1 12000
elif [ "$MODE" != short ]; then echo "arm_measure: MODE must be short or full" >&2; exit 2
fi
curl -sf -m 10 -o /dev/null "$U/health"
echo "ARM $LABEL done $(date -u +%H:%M:%SZ)"
