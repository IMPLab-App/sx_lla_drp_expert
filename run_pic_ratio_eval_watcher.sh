#!/usr/bin/env bash
set -euo pipefail

# Watch/evaluate finished PIC ratio-sweep jobs on Linux.
# If checkpoints already exist, tasks 1-8 are evaluated immediately.
#
# Usage:
#   bash run_pic_ratio_eval_watcher.sh [gpu_id|-1|cpu] [num_workers] [python_cmd] [model_root] [start_at] [end_at] [poll_seconds]
#
# Examples:
#   bash run_pic_ratio_eval_watcher.sh 0 4 /home/kv/.conda/envs/cw/bin/python saved_models1
#   nohup bash run_pic_ratio_eval_watcher.sh 0 4 /home/kv/.conda/envs/cw/bin/python saved_models1 1 8 60 \
#     > logs/ratio_sweep/nohup_eval_watcher_1_8.log 2>&1 &

VISIBLE_GPU_ID="${1:-0}"
NUM_WORKERS="${2:-4}"
PYTHON_CMD="${3:-${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}}"
MODEL_ROOT="${4:-saved_models1}"
START_AT="${5:-1}"
END_AT="${6:-8}"
POLL_SECONDS="${7:-60}"

RUNTIME_GPU_ID="${RUNTIME_GPU_ID:-0}"
CUDA_PREFIX=()
if [[ "$VISIBLE_GPU_ID" == "-1" || "$VISIBLE_GPU_ID" == "cpu" || "$VISIBLE_GPU_ID" == "CPU" ]]; then
  RUNTIME_GPU_ID="-1"
else
  CUDA_PREFIX=(env "CUDA_VISIBLE_DEVICES=$VISIBLE_GPU_ID")
fi

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "[Warning] Python not found at: $PYTHON_CMD"
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    echo "[Error] python command not found"
    exit 1
  fi
fi

LOG_DIR="logs/ratio_sweep"
TMP_DIR="config/_linux_ratio_sweep"
RUN_LOG="$LOG_DIR/eval_watcher_linux.log"

mkdir -p "$LOG_DIR" "$TMP_DIR"

CONFIGS=(
  "config/ratio_sweep/001-fixmatch_metaexpert_pic_lb900_50_ulb1800_50_0.0_2.yaml"
  "config/ratio_sweep/002-fixmatch_metaexpert_pic_lb900_100_ulb1800_100_0.0_2.yaml"
  "config/ratio_sweep/003-fixmatch_metaexpert_pic_lb900_150_ulb1800_150_0.0_2.yaml"
  "config/ratio_sweep/004-fixmatch_metaexpert_pic_lb900_200_ulb1800_200_0.0_2.yaml"
  "config/ratio_sweep/005-fixmatch_metaexpert_pic_lb900_50_ulb1800_-50_0.0_2.yaml"
  "config/ratio_sweep/006-fixmatch_metaexpert_pic_lb900_100_ulb1800_-100_0.0_2.yaml"
  "config/ratio_sweep/007-fixmatch_metaexpert_pic_lb900_150_ulb1800_-150_0.0_2.yaml"
  "config/ratio_sweep/008-fixmatch_metaexpert_pic_lb900_200_ulb1800_-200_0.0_2.yaml"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval watcher started visible_gpu=$VISIBLE_GPU_ID runtime_gpu=$RUNTIME_GPU_ID num_workers=$NUM_WORKERS python=$PYTHON_CMD model_root=$MODEL_ROOT range=${START_AT}-${END_AT}" | tee "$RUN_LOG"

write_eval_config() {
  local src="$1"
  local dst="$2"
  "$PYTHON_CMD" - "$src" "$dst" "$RUNTIME_GPU_ID" "$NUM_WORKERS" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
gpu_id = sys.argv[3]
num_workers = sys.argv[4]

lines = src.read_text(encoding="utf-8").splitlines()
out = []
seen_gpu = False
seen_workers = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith("gpu:"):
        out.append(f"gpu: {gpu_id}")
        seen_gpu = True
    elif stripped.startswith("num_workers:"):
        out.append(f"num_workers: {num_workers}")
        seen_workers = True
    else:
        out.append(line)

if not seen_gpu:
    out.append(f"gpu: {gpu_id}")
if not seen_workers:
    out.append(f"num_workers: {num_workers}")

dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print(dst)
PY
}

read_save_name() {
  local cfg="$1"
  "$PYTHON_CMD" - "$cfg" <<'PY'
import pathlib
import sys

cfg = pathlib.Path(sys.argv[1])
for line in cfg.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped.startswith("save_name:"):
        print(stripped.split(":", 1)[1].strip().strip("'\""))
        break
else:
    raise SystemExit(f"save_name not found in {cfg}")
PY
}

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] missing config: $config" | tee -a "$RUN_LOG"
    exit 1
  fi

  base_name="$(basename "$config" .yaml)"
  task_id="${base_name%%-*}"
  task_id="$((10#$task_id))"
  if (( task_id < START_AT || task_id > END_AT )); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP $config outside range ${START_AT}-${END_AT}" | tee -a "$RUN_LOG"
    continue
  fi

  save_name="$(read_save_name "$config")"
  checkpoint="$MODEL_ROOT/${save_name}/model_best.pth"
  if [[ ! -f "$checkpoint" ]]; then
    checkpoint="$MODEL_ROOT/${save_name}/latest_model.pth"
  fi

  while [[ ! -f "$checkpoint" ]]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WAIT task=$task_id checkpoint under $MODEL_ROOT/$save_name" | tee -a "$RUN_LOG"
    sleep "$POLL_SECONDS"
    checkpoint="$MODEL_ROOT/${save_name}/model_best.pth"
    if [[ ! -f "$checkpoint" ]]; then
      checkpoint="$MODEL_ROOT/${save_name}/latest_model.pth"
    fi
  done

  tmp_cfg="$TMP_DIR/${base_name}_linux_eval.yaml"
  write_eval_config "$config" "$tmp_cfg" >/dev/null
  eval_log_path="$LOG_DIR/${base_name}_linux.eval.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EVAL task=$task_id config=$tmp_cfg checkpoint=$checkpoint" | tee -a "$RUN_LOG"
  if ! PYTHONUNBUFFERED=1 "${CUDA_PREFIX[@]}" "$PYTHON_CMD" eval_confusion_matrix.py \
      --c "$tmp_cfg" \
      --load_path "$checkpoint" \
      --gpu "$RUNTIME_GPU_ID" \
      --num_workers "$NUM_WORKERS" 2>&1 | tee "$eval_log_path"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED eval task=$task_id config=$tmp_cfg" | tee -a "$RUN_LOG"
    exit 1
  fi

  output_dir="$(dirname "$checkpoint")/confusion_matrix"
  checkpoint_stem="$(basename "$checkpoint" .pth)"
  missing_outputs=0
  for output_file in \
      "$output_dir/${checkpoint_stem}.png" \
      "$output_dir/${checkpoint_stem}.csv" \
      "$output_dir/avg_prob_mean_matrix.csv" \
      "$output_dir/avg_prob_std_matrix.csv" \
      "$output_dir/avg_prob_mean_std_matrix.csv" \
      "$output_dir/avg_prob_mean_std_matrix.png"; do
    if [[ ! -f "$output_file" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] missing output: $output_file" | tee -a "$RUN_LOG"
      missing_outputs=1
    fi
  done
  if (( missing_outputs != 0 )); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED output verification task=$task_id" | tee -a "$RUN_LOG"
    exit 1
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE eval task=$task_id output=$output_dir" | tee -a "$RUN_LOG"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] eval watcher finished" | tee -a "$RUN_LOG"
