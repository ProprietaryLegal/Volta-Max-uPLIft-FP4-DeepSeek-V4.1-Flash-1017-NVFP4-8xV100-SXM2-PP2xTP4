#!/bin/bash
# benchvolta.sh URL [options]: Volta sPLIt README speed tables against one server, one request at a time.
# Options and method: /usr/bin/python3 -I bench_volta_run.py --help. Summary: bench_volta_summary.py RECEIPT_DIR.
# Fail fast: the first server error stops the run (exit 3); receipts hold numbers and hashes only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/python3 -I "$HERE/bench_volta_run.py" "$@"
