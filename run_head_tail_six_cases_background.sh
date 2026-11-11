#!/bin/bash
set -u

LOG_DIR="${LOG_DIR:-logs/head_tail_six_cases}"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_head_tail_six_cases_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run_head_tail_six_cases.pid}"

(
  echo "[$(date '+%F %T')] started"
  echo "[$(date '+%F %T')] GPU_ID=${GPU_ID:-0} PYTHON_CMD=${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
  ./run_head_tail_six_cases.sh
  RC=$?
  echo "[$(date '+%F %T')] finished exit_code=${RC}"
  exit "$RC"
) >> "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "pid=$PID"
echo "log=$LOG_FILE"
