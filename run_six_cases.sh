#!/bin/bash
set -eu
PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
GPU_ID="${GPU_ID:-0}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONFIG_DIR="${CONFIG_DIR:-config/custom_sweep_ablation}"
SAVE_DIR="${SAVE_DIR:-saved_models_ablation}"

ABLATIONS=(
  "no_dea_aggregator"
  "no_mff"
)

RUNS=(
  "run_1_pic_lb100_5_ulb250_5_exp"
  "run_2_pic_lb100_10_ulb250_10_exp"
  "run_3_pic_lb100_5_ulb315_1.0_exp"
  "run_4_pic_lb100_10_ulb315_1.0_exp"
  "run_5_pic_lb100_5_ulb250_-5_pxe"
  "run_6_pic_lb100_10_ulb250_-10_pxe"
  # "run_7_pic_lb100_50_ulb250_50_exp"
  # "run_8_pic_lb100_100_ulb250_100_exp"
)

for RUN_NAME in "${RUNS[@]}"; do
  for ABLATION in "${ABLATIONS[@]}"; do
    CONFIG="${CONFIG_DIR}/${ABLATION}/${RUN_NAME}_${ABLATION}.yaml"
    CHECKPOINT="./${SAVE_DIR}/${ABLATION}/${RUN_NAME}_${ABLATION}/model_best.pth"

    if [ ! -f "$CONFIG" ]; then
      echo "Missing config: ${CONFIG}" >&2
      exit 1
    fi

    echo "Running ${ABLATION} training for ${CONFIG}"
    "$PYTHON_CMD" train.py --c "$CONFIG"

    echo "Running ${ABLATION} eval for ${CONFIG}"
    "$PYTHON_CMD" eval_confusion_matrix_new.py --c "$CONFIG" --load_path "$CHECKPOINT" --gpu "$EVAL_GPU_ID"
  done
done
