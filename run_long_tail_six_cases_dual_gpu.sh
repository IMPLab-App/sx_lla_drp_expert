#!/usr/bin/env bash
# GPU 0: experiments 1, 3, 5. GPU 1: experiments 2, 4, 6.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TZ=Asia/Shanghai
LOG_DIR="${LOG_DIR:-logs/long_tail_six_dual_gpu_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR/gpu0" "$LOG_DIR/gpu1"
exec 9>logs/long_tail_six_dual_gpu.lock
flock -n 9 || { echo 'The dual-GPU long-tail queue is already running.' >&2; exit 1; }
printf '%s\n' "$LOG_DIR" > logs/long_tail_six_dual_gpu.latest
echo "$$" > "$LOG_DIR/queue.pid"
printf 'running\n' > "$LOG_DIR/status.txt"

GPU_ID=0 WORKER_INDEX=0 WORKER_COUNT=2 LOG_DIR="$LOG_DIR/gpu0" \
  bash run_long_tail_six_cases.sh > "$LOG_DIR/gpu0/queue.log" 2>&1 &
PID0=$!
GPU_ID=1 WORKER_INDEX=1 WORKER_COUNT=2 LOG_DIR="$LOG_DIR/gpu1" \
  bash run_long_tail_six_cases.sh > "$LOG_DIR/gpu1/queue.log" 2>&1 &
PID1=$!
printf 'gpu\tpid\truns\n0\t%s\t1,3,5\n1\t%s\t2,4,6\n' "$PID0" "$PID1" > "$LOG_DIR/workers.tsv"
echo "[$(date -Is)] GPU 0 worker: $PID0; GPU 1 worker: $PID1"

CODE0=0
CODE1=0
wait "$PID0" || CODE0=$?
wait "$PID1" || CODE1=$?
printf 'gpu\texit_code\n0\t%s\n1\t%s\n' "$CODE0" "$CODE1" > "$LOG_DIR/exit_codes.tsv"
if (( CODE0 == 0 && CODE1 == 0 )); then
  printf 'completed\n' > "$LOG_DIR/status.txt"
  echo "[$(date -Is)] All six experiments completed."
else
  printf 'failed\n' > "$LOG_DIR/status.txt"
  echo "[$(date -Is)] Worker failures: GPU 0=$CODE0 GPU 1=$CODE1" >&2
  exit 1
fi
