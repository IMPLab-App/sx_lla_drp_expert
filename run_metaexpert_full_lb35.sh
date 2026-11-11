#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kv/.conda/envs/cw/bin/python}"
if [[ $# -gt 0 ]]; then
    BATCH="$1"
else
    BATCH="$(<"$ROOT/logs/metaexpert_full_lb35.latest")"
fi
exec "$PYTHON_BIN" "$BATCH/code/run_metaexpert_full_lb35.py" "$BATCH"
