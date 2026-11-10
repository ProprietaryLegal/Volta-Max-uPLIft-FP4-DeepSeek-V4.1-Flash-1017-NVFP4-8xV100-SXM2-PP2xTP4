#!/bin/bash
# baseline.sh URL LABEL [N]: decode speed of a running server, measured the way production's arms are
# (bench5 warmup.py, then ab_t1_v2.py --tput: public prompts, T=0, thinking off, 1024 tokens, streamed,
# decode-only ms/round). Receipts: logs/baseline/LABEL-{warm,tput}.out and LABEL-tput.jsonl.
# Read-only use of production's harness scripts; one request at a time. No retries: any failure exits nonzero.
#   WARM=0 baseline.sh ...   skips the warm-up (second and later runs on the same boot)
: "${LANE:?set LANE to the root directory of this build}"
: "${PROD:?set PROD to the Volta sPLIt checkout}"
set -euo pipefail
O=$PROD/1cat-1277/bench5
D=$LANE/logs/baseline
URL=$1 LABEL=$2 N=${3:-3}
mkdir -p "$D"
[ ! -e "$D/$LABEL-tput.out" ] || { echo "baseline: $D/$LABEL-tput.out exists; pick a new label" >&2; exit 2; }
curl -sf -m 10 -o /dev/null "$URL/health" || { echo "baseline: $URL/health not answering" >&2; exit 1; }
if [ "${WARM:-1}" = 1 ]; then
  /usr/bin/python3 -I "$O/warmup.py" "$URL" > "$D/$LABEL-warm.out" 2>&1
  grep -q '^WARMUP OK' "$D/$LABEL-warm.out" || { echo "baseline: warm-up failed, see $D/$LABEL-warm.out" >&2; exit 1; }
fi
echo "$(date -u +%FT%TZ) baseline $LABEL $URL N=$N" > "$D/$LABEL-tput.out"
/usr/bin/python3 -I "$O/ab_t1_v2.py" --tput "$URL" "$N" "$D/$LABEL-tput.jsonl" | tee -a "$D/$LABEL-tput.out"
grep -q '^TPUT TOTAL' "$D/$LABEL-tput.out" || { echo "baseline: no TPUT TOTAL line" >&2; exit 1; }
curl -sf -m 10 -o /dev/null "$URL/health" || { echo "baseline: server not answering after the run" >&2; exit 1; }
