#!/bin/bash
# measure_boot.sh URL LABEL [LONG]: the lane's standard measurement of one boot, one request at a time.
#   1. warm-up + general public prompts (bench5 set)      -> logs/baseline/LABEL-g1-*
#   2. legal-style public prompts                         -> logs/baseline/LABEL-l1-*
#   3. LONG=1: a fresh 100K-token prompt, then a 250K-token prompt, card memory sampled every 15 s
#      (the operator 2026-11-09: speed tests stop at 250K). LONG=2: 100K then 500K, only for a build that changes
#      card memory use (the server admits 524,288 tokens, so its peak must be checked there).
#   4. repeat of 1 and 2 without warm-up (within-boot noise; after the long prompts when LONG=1)
# No retries: the first failure stops the script with a nonzero exit.
: "${LANE:?set LANE to the root directory of this build}"
set -euo pipefail
R=$LANE; D=$R/logs/baseline
URL=$1 LABEL=$2 LONG=${3:-0}
legal() { # legal LABEL
  [ ! -e "$D/$1-tput.out" ] || { echo "measure_boot: $D/$1-tput.out exists" >&2; exit 2; }
  /usr/bin/python3 -I "$R/run/tput_prompts.py" "$URL" "$D/legal-public-prompts.txt" "$D/$1-tput.jsonl" > "$D/$1-tput.out"
  grep -q '^TPUT TOTAL' "$D/$1-tput.out"
}
long() { # long LABEL CHARS SKIP SECONDS
  bash "$R/run/memwatch.sh" "$D/$1-mem.txt" "$4" & local w=$!
  /usr/bin/python3 -I "$R/run/longprompt.py" "$URL" "$2" 48 "$3" > "$D/$1.out"
  wait "$w"
  curl -sf -m 10 -o /dev/null "$URL/health"
}
bash "$R/run/baseline.sh" "$URL" "$LABEL-g1" > /dev/null
legal "$LABEL-l1"
case "$LONG" in
  0) ;;
  1) long "$LABEL-long100k" 420000 2200000 100; long "$LABEL-long250k" 1000000 0 220 ;;
  2) long "$LABEL-long100k" 420000 2200000 100; long "$LABEL-long500k" 2000000 0 400 ;;
  *) echo "measure_boot: LONG must be 0, 1 or 2" >&2; exit 2 ;;
esac
WARM=0 bash "$R/run/baseline.sh" "$URL" "$LABEL-g2" > /dev/null
legal "$LABEL-l2"
for a in g1 l1 g2 l2; do echo "$LABEL-$a $(grep '^TPUT TOTAL' "$D/$LABEL-$a-tput.out")"; done
[ "$LONG" = 0 ] || cat "$D/$LABEL"-long*.out
echo "MEASURE DONE $LABEL $(date -u +%FT%TZ)"
