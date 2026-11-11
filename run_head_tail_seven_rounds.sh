#!/bin/bash
set -eu

PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
GPU_ID="${GPU_ID:-1}"
BASE_SEED="${BASE_SEED:-2}"
START_ROUND="${START_ROUND:-1}"
START_RUN="${START_RUN:-1}"
LOG_DIR="${LOG_DIR:-logs/head_tail_seven_rounds}"

ROUND_DIRS=(
  "saved_models_head_tail"
  "1 saved_models_head_tail"
  "2 saved_models_head_tail"
  "3 saved_models_head_tail"
  "4 saved_models_head_tail"
  "5 saved_models_head_tail"
  "6 saved_models_head_tail"
)

if ! [[ "$START_ROUND" =~ ^[1-7]$ ]]; then
  echo "START_ROUND must be an integer from 1 to 7, got: ${START_ROUND}" >&2
  exit 1
fi
if ! [[ "$START_RUN" =~ ^[1-6]$ ]]; then
  echo "START_RUN must be an integer from 1 to 6, got: ${START_RUN}" >&2
  exit 1
fi
if ! [[ "$BASE_SEED" =~ ^[0-9]+$ ]]; then
  echo "BASE_SEED must be a non-negative integer, got: ${BASE_SEED}" >&2
  exit 1
fi

for ROUND_INDEX in "${!ROUND_DIRS[@]}"; do
  ROUND_NUMBER=$((ROUND_INDEX + 1))
  if (( ROUND_NUMBER < START_ROUND )); then
    echo "[$(date '+%F %T')] Skipping completed round ${ROUND_NUMBER}"
    continue
  fi

  ROUND_SAVE_DIR="${ROUND_DIRS[$ROUND_INDEX]}"
  ROUND_SEED=$((BASE_SEED + ROUND_INDEX))
  RUNTIME_CONFIG_DIR="${LOG_DIR}/runtime_configs/round_${ROUND_NUMBER}"
  ROUND_START_RUN=1
  if (( ROUND_NUMBER == START_ROUND )); then
    ROUND_START_RUN="$START_RUN"
  fi

  echo "[$(date '+%F %T')] Starting round ${ROUND_NUMBER}/7 at run ${ROUND_START_RUN}/6: save_dir=${ROUND_SAVE_DIR}, seed=${ROUND_SEED}"
  PYTHON_CMD="$PYTHON_CMD" \
    GPU_ID="$GPU_ID" \
    SAVE_DIR="$ROUND_SAVE_DIR" \
    SEED="$ROUND_SEED" \
    START_RUN="$ROUND_START_RUN" \
    RUNTIME_CONFIG_DIR="$RUNTIME_CONFIG_DIR" \
    bash ./run_head_tail_six_cases.sh
  echo "[$(date '+%F %T')] Finished round ${ROUND_NUMBER}/7: save_dir=${ROUND_SAVE_DIR}, seed=${ROUND_SEED}"
done

echo "[$(date '+%F %T')] All seven Head/Tail rounds finished"
