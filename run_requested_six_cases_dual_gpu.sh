#!/usr/bin/env bash
# GPU 0: 1 -> 3 -> 5; GPU 1: 2 -> 4 -> 6.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TZ=Asia/Shanghai
PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
export PYTHON_CMD
LOG_DIR="${LOG_DIR:-logs/requested_six_dual_gpu_$(date +%Y%m%d_%H%M%S)}"
SOURCE_CONFIG_DIR="${CONFIG_DIR:-config/custom_sweep}"
mkdir -p logs
exec 9>logs/requested_six_dual_gpu.lock
flock -n 9 || { echo 'The requested dual-GPU queue is already running.' >&2; exit 1; }
# A fresh directory keeps checkpoints from earlier runs intact.
mkdir -- "$LOG_DIR"
mkdir -p "$LOG_DIR/configs" "$LOG_DIR/gpu0" "$LOG_DIR/gpu1"
"$PYTHON_CMD" - "$SOURCE_CONFIG_DIR" "$LOG_DIR" <<'PY'
import pathlib
import sys
import yaml

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
runs = [
    'run_1_pic_lb100_5_ulb250_5_exp',
    'run_2_pic_lb100_10_ulb250_10_exp',
    'run_3_pic_lb100_5_ulb315_1.0_exp',
    'run_4_pic_lb100_10_ulb315_1.0_exp',
    'run_5_pic_lb100_5_ulb250_-5_pxe',
    'run_6_pic_lb100_10_ulb250_-10_pxe',
]
for run in runs:
    settings = yaml.safe_load((source / (run + '.yaml')).read_text())
    assert settings['save_name'] == run, run
    settings['save_dir'] = str(output.resolve() / 'models')
    settings['gpu'] = 0  # Logical GPU after CUDA_VISIBLE_DEVICES masking.
    (output / 'configs' / (run + '.yaml')).write_text(
        yaml.safe_dump(settings, sort_keys=False))
PY
printf '%s\n' "$LOG_DIR" > logs/requested_six_dual_gpu.latest
echo "$$" > "$LOG_DIR/queue.pid"
printf 'running\n' > "$LOG_DIR/status.txt"
CONFIG_DIR="$LOG_DIR/configs" EXPERIMENT_SET=standard GPU_ID=0 WORKER_INDEX=0 WORKER_COUNT=2 LOG_DIR="$LOG_DIR/gpu0" \
  bash run_requested_six_cases.sh > "$LOG_DIR/gpu0/queue.log" 2>&1 &
PID0=$!
CONFIG_DIR="$LOG_DIR/configs" EXPERIMENT_SET=standard GPU_ID=1 WORKER_INDEX=1 WORKER_COUNT=2 LOG_DIR="$LOG_DIR/gpu1" \
  bash run_requested_six_cases.sh > "$LOG_DIR/gpu1/queue.log" 2>&1 &
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
