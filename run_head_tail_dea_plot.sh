#!/bin/bash
set -eu

PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
GPU_ID="${GPU_ID:-0}"
RUN_NAME="${RUN_NAME:-run_1_pic_lb100_5_ulb250_5_exp}"
CONFIG_DIR="${CONFIG_DIR:-config/custom_sweep_head_tail}"
SAVE_DIR="${SAVE_DIR:-saved_models_head_tail}"
SPLIT="${SPLIT:-test}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONFIG="${CONFIG_DIR}/${RUN_NAME}.yaml"
CHECKPOINT="${SAVE_DIR}/${RUN_NAME}/model_best.pth"
OUTPUT_DIR="${SAVE_DIR}/${RUN_NAME}/dea_head_tail_confidence"

echo "Training with Head=class 1/normal and Tail=classes 2-9: ${CONFIG}"
"$PYTHON_CMD" train.py --c "$CONFIG"

echo "Plotting raw DEA confidence from ${CHECKPOINT}"
"$PYTHON_CMD" plot_dea_head_tail_confidence.py \
  --c "$CONFIG" \
  --load_path "$CHECKPOINT" \
  --gpu 0 \
  --split "$SPLIT" \
  --weight_source dea \
  --output_dir "$OUTPUT_DIR" \
  --output_name "dea_head_tail_confidence_${SPLIT}"
