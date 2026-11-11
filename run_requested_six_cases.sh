#!/usr/bin/env bash
# Run exactly the six requested main experiments, in order.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
CONFIG_DIR="${CONFIG_DIR:-config/custom_sweep}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
if ! [[ "$WORKER_COUNT" =~ ^[1-6]$ && "$WORKER_INDEX" =~ ^[0-5]$ ]] || (( WORKER_INDEX >= WORKER_COUNT )); then
  echo 'Invalid WORKER_INDEX or WORKER_COUNT.' >&2
  exit 1
fi
EXPERIMENT_SET="${EXPERIMENT_SET:-standard}"
case "$EXPERIMENT_SET" in
  standard) LOG_PREFIX=requested_six ;;
  long_tail) LOG_PREFIX=long_tail_six ;;
  *) echo "Unknown EXPERIMENT_SET: $EXPERIMENT_SET" >&2; exit 1 ;;
esac
if (( WORKER_COUNT > 1 )); then
  LOG_PREFIX="${LOG_PREFIX}_gpu${GPU_ID:-1}"
fi
LOG_DIR="${LOG_DIR:-logs/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR/configs"

echo "$$" > "$LOG_DIR/queue.pid"
printf '%s\n' "$LOG_DIR" > "logs/${LOG_PREFIX}_cases.latest"
printf 'time\trun\tstage\tstatus\tcode\n' > "$LOG_DIR/status.tsv"
RUN_NAME=queue
STAGE=prepare
record() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$RUN_NAME" "$STAGE" "$1" "${2:-0}" >> "$LOG_DIR/status.tsv"
  echo "[$(date -Is)] $RUN_NAME $STAGE $1 ${2:-0}"
}
trap 'code=$?; record failed "$code"; exit "$code"' ERR
trap 'record interrupted 143; exit 143' TERM
trap 'record interrupted 130; exit 130' INT

if [[ "$EXPERIMENT_SET" == long_tail ]]; then
  RUNS=(
    run_1_pic_lb100_50_ulb250_50_exp
    run_2_pic_lb100_100_ulb250_100_exp
    run_3_pic_lb100_50_ulb315_1.0_exp
    run_4_pic_lb100_100_ulb315_1.0_exp
    run_5_pic_lb100_50_ulb250_-50_pxe
    run_6_pic_lb100_100_ulb250_-100_pxe
  )
else
  RUNS=(
    run_1_pic_lb100_5_ulb250_5_exp
    run_2_pic_lb100_10_ulb250_10_exp
    run_3_pic_lb100_5_ulb315_1.0_exp
    run_4_pic_lb100_10_ulb315_1.0_exp
    run_5_pic_lb100_5_ulb250_-5_pxe
    run_6_pic_lb100_10_ulb250_-10_pxe
  )
fi

# Each worker takes alternating experiments; a single worker takes all six.
ASSIGNED_RUNS=()
for INDEX in "${!RUNS[@]}"; do
  if (( INDEX % WORKER_COUNT == WORKER_INDEX )); then
    ASSIGNED_RUNS+=("${RUNS[$INDEX]}")
  fi
done
RUNS=("${ASSIGNED_RUNS[@]}")

# Lock each assigned experiment for the worker's entire lifetime. This allows
# disjoint queues to run together while preventing duplicate experiment runs.
for RUN_NAME in "${RUNS[@]}"; do
  exec {RUN_LOCK}>"logs/.${RUN_NAME}.lock"
  flock -n "$RUN_LOCK" || { echo "Experiment already running: $RUN_NAME" >&2; exit 1; }
done

for RUN_NAME in "${RUNS[@]}"; do
  test -f "$CONFIG_DIR/$RUN_NAME.yaml"
  cp -- "$CONFIG_DIR/$RUN_NAME.yaml" "$LOG_DIR/configs/$RUN_NAME.yaml"
  record pending
done

for RUN_NAME in "${RUNS[@]}"; do
  CONFIG="$LOG_DIR/configs/$RUN_NAME.yaml"
  STAGE=train
  record running
  if "$PYTHON_CMD" train.py --c "$CONFIG" > "$LOG_DIR/$RUN_NAME.train.log" 2>&1; then
    record completed
  else
    CODE=$?
    record retrying "$CODE"
    # Preserve the failed attempt before a clean retry. Avoid incomplete-epoch
    # resume behavior in the legacy trainer. A worker failure can be recovered
    # without changing batch size, optimization, or experiment settings.
    "$PYTHON_CMD" - "$CONFIG" "$LOG_DIR/$RUN_NAME.train.log" <<'PY'
import datetime
import pathlib
import shutil
import sys
import yaml

config = pathlib.Path(sys.argv[1])
settings = yaml.safe_load(config.read_text())
model_dir = pathlib.Path(settings['save_dir']) / settings['save_name']
if model_dir.exists():
    backup = model_dir.with_name(model_dir.name + '.failed_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.move(str(model_dir), str(backup))
text = pathlib.Path(sys.argv[2]).read_text(errors='replace')
if any(marker in text for marker in ('DataLoader worker', 'Bus error', 'Too many open files', 'No space left on device')):
    settings['num_workers'] = 0
    config.write_text(yaml.safe_dump(settings, sort_keys=False))
    print('Recovery: disabled multiprocessing data loaders for this run.')
PY
    sleep 10
    "$PYTHON_CMD" train.py --c "$CONFIG" > "$LOG_DIR/$RUN_NAME.train.retry.log" 2>&1
    record completed
  fi

  STAGE=eval
  record running
  CHECKPOINT_DIR=$("$PYTHON_CMD" - "$CONFIG" <<'PY'
import pathlib
import sys
import yaml
settings = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
print(pathlib.Path(settings['save_dir']) / settings['save_name'])
PY
  )
  "$PYTHON_CMD" eval_confusion_matrix_new.py --c "$CONFIG" \
    --load_path "$CHECKPOINT_DIR/model_best.pth" --gpu 0 \
    > "$LOG_DIR/$RUN_NAME.eval.log" 2>&1
  record completed
done
RUN_NAME=queue
STAGE=all
record completed
