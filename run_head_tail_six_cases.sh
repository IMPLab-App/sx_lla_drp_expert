#!/bin/bash
set -eu

PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
GPU_ID="${GPU_ID:-0}"
CONFIG_DIR="${CONFIG_DIR:-config/custom_sweep_head_tail}"
SAVE_DIR="${SAVE_DIR:-saved_models_head_tail}"
SPLIT="${SPLIT:-test}"
START_RUN="${START_RUN:-1}"
SEED="${SEED:-2}"
RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-logs/head_tail_six_cases/runtime_configs/$(date +%Y%m%d_%H%M%S)_$$}"

RUNS=(
  "run_1_pic_lb100_5_ulb250_5_exp"
  "run_2_pic_lb100_10_ulb250_10_exp"
  "run_3_pic_lb100_5_ulb315_1.0_exp"
  "run_4_pic_lb100_10_ulb315_1.0_exp"
  "run_5_pic_lb100_5_ulb250_-5_pxe"
  "run_6_pic_lb100_10_ulb250_-10_pxe"
)

export CUDA_VISIBLE_DEVICES="$GPU_ID"

if ! [[ "$START_RUN" =~ ^[1-6]$ ]]; then
  echo "START_RUN must be an integer from 1 to 6, got: ${START_RUN}" >&2
  exit 1
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer, got: ${SEED}" >&2
  exit 1
fi

for RUN_INDEX in "${!RUNS[@]}"; do
  RUN_NUMBER=$((RUN_INDEX + 1))
  if (( RUN_NUMBER < START_RUN )); then
    echo "Skipping completed Head/Tail run ${RUN_NUMBER}"
    continue
  fi

  RUN_NAME="${RUNS[$RUN_INDEX]}"
  SOURCE_CONFIG="${CONFIG_DIR}/${RUN_NAME}.yaml"
  CONFIG="${RUNTIME_CONFIG_DIR}/${RUN_NAME}.yaml"
  CHECKPOINT="${SAVE_DIR}/${RUN_NAME}/model_best.pth"
  OUTPUT_DIR="${SAVE_DIR}/${RUN_NAME}/dea_head_tail_confidence"

  "$PYTHON_CMD" prepare_head_tail_runtime_config.py \
    "$SOURCE_CONFIG" \
    "$CONFIG" \
    --save-dir "$SAVE_DIR" \
    --seed "$SEED"

  echo "Training Head/Tail config: ${CONFIG} (save_dir=${SAVE_DIR}, seed=${SEED})"
  "$PYTHON_CMD" train.py --c "$CONFIG"

  echo "Plotting raw DEA confidence: ${CHECKPOINT}"
  "$PYTHON_CMD" plot_dea_head_tail_confidence.py \
    --c "$CONFIG" \
    --load_path "$CHECKPOINT" \
    --gpu 0 \
    --split "$SPLIT" \
    --weight_source dea \
    --output_dir "$OUTPUT_DIR" \
    --output_name "dea_head_tail_confidence_${SPLIT}"
done
