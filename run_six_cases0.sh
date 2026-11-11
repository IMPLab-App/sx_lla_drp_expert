#!/bin/bash
set -eu
PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
GPU_ID="${GPU_ID:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

RUNS=(
  "run_1_pic_lb100_5_ulb250_5_exp"
  "run_2_pic_lb100_10_ulb250_10_exp"
  "run_3_pic_lb100_5_ulb315_1.0_exp"
  "run_4_pic_lb100_10_ulb315_1.0_exp"
  "run_5_pic_lb100_5_ulb250_-5_pxe"
  "run_6_pic_lb100_10_ulb250_-10_pxe"
  "run_7_pic_lb100_50_ulb250_50_exp"
  "run_8_pic_lb100_100_ulb250_100_exp"
)

for RUN_NAME in "${RUNS[@]}"; do
  CONFIG="config/custom_sweep/${RUN_NAME}.yaml"
  CHECKPOINT="./saved_models/${RUN_NAME}/model_best.pth"

  echo "Running training for ${CONFIG}"
  "$PYTHON_CMD" train.py --c "$CONFIG"

  echo "Running eval for ${CONFIG}"
  "$PYTHON_CMD" eval_confusion_matrix_new.py --c "$CONFIG" --load_path "$CHECKPOINT" --gpu "$EVAL_GPU_ID"
done
