#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/002-fixmatch_metaexpert_pic_lb900_150_ulb1800_150_0.0_2.yaml}"
NUM_RUNS="${2:-10}"
GPU_ID="${3:-0}"
START_SEED="${4:-7}"

PYTHON_CMD="/home/kv/.conda/envs/cw/bin/python"

BASE_NAME="$(basename "$CONFIG_PATH" .yaml)"
TMP_DIR="config/_tmp_runs"

echo "[Info] RESUMING From Seed $START_SEED. config=$CONFIG_PATH num_runs=$NUM_RUNS gpu=$GPU_ID"

for ((i=START_SEED; i<NUM_RUNS; i++)); do
  SEED="$i"
  RUN_NAME="${BASE_NAME}_seed${SEED}"
  TMP_CFG="$TMP_DIR/${RUN_NAME}.yaml"

  echo "[Info] ===== Run $((i+1))/$NUM_RUNS seed=$SEED ====="

  # Generate config
  $PYTHON_CMD - <<PY
import pathlib
src = pathlib.Path("$CONFIG_PATH")
text = src.read_text(encoding="utf-8")

lines = text.splitlines()
out = []
seen_save = False
seen_seed = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("save_name:"):
        out.append(f"save_name: $RUN_NAME")
        seen_save = True
    elif stripped.startswith("seed:"):
        out.append(f"seed: {int($SEED)}")
        seen_seed = True
    else:
        out.append(line)

if not seen_save:
    out.append(f"save_name: $RUN_NAME")
if not seen_seed:
    out.append(f"seed: {int($SEED)}")

pathlib.Path("$TMP_CFG").write_text("\n".join(out) + "\n", encoding="utf-8")
PY

  # If we have a saved model for this seed, try to resume, else run normally.
  RUN_DIR="saved_models/$RUN_NAME"
  if [[ -f "$RUN_DIR/latest_model.pth" ]]; then
      echo "[Info] Found existing model, RESUMING..."
      CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON_CMD train.py --c "$TMP_CFG" --resume --load_path "$RUN_DIR/latest_model.pth"
  else
      CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON_CMD train.py --c "$TMP_CFG"
  fi

  if [[ ! -d "$RUN_DIR" ]]; then
    echo "[Error] Run dir not found: $RUN_DIR"
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON_CMD eval_confusion_matrix.py \
    --c "$TMP_CFG" \
    --load_path "$RUN_DIR/model_best.pth" \
    --gpu 0

done

CSV_GLOB="saved_models/${BASE_NAME}_seed*/confusion_matrix/model_best.csv"
$PYTHON_CMD average_confusion_matrices.py \
  --input_glob "$CSV_GLOB" \
  --output_dir "saved_models/${BASE_NAME}_avg10" \
  --output_name "avg_confusion_matrix"

echo "[Done] Resumed and average confusion matrix generation finished."
