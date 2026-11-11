#!/usr/bin/env bash
# Sequential training and evaluation of the six 50/100 imbalance experiments.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_SET=long_tail
exec bash "$SCRIPT_DIR/run_requested_six_cases.sh"
