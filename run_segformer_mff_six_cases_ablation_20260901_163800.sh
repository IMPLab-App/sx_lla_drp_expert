#!/bin/bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
GPU_ID="${GPU_ID:-1}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
RUN_EVAL="${RUN_EVAL:-true}"
STOP_ON_ERROR="${STOP_ON_ERROR:-true}"
CONFIG_ROOT='config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800'
LOG_ROOT='logs/segformer_mff_six_cases_ablation_20260901_163800'
MASTER_LOG="$LOG_ROOT/queue.log"
STATUS_FILE="$LOG_ROOT/status.tsv"
mkdir -p "$LOG_ROOT"
exec >> "$MASTER_LOG" 2>&1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo -e "time\tphase\tlabel\trc\tconfig" > "$STATUS_FILE"
echo "[$(date '+%F %T')] queue started; physical GPU_ID=$GPU_ID; visible gpu=0; eval=$RUN_EVAL"
echo "[$(date '+%F %T')] config root: $CONFIG_ROOT"

run_task() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local task_log="$LOG_ROOT/${label}.log"
  echo "[$(date '+%F %T')] train start: $label -> $config"
  printf '%s\ttrain_start\t%s\t%s\t%s\n' "$(date '+%F %T')" "$label" "-" "$config" >> "$STATUS_FILE"
  "$PYTHON_CMD" train.py --c "$config" >> "$task_log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] train end: $label rc=$rc"
  printf '%s\ttrain_end\t%s\t%s\t%s\n' "$(date '+%F %T')" "$label" "$rc" "$config" >> "$STATUS_FILE"
  if [ "$rc" -ne 0 ]; then
    if [ "$STOP_ON_ERROR" = "true" ]; then
      echo "[$(date '+%F %T')] stopping queue after train failure: $label"
      exit "$rc"
    fi
    return "$rc"
  fi
  if [ "$RUN_EVAL" = "true" ]; then
    if [ ! -f "$checkpoint" ]; then
      echo "[$(date '+%F %T')] missing checkpoint for eval: $checkpoint"
      printf '%s\teval_missing_checkpoint\t%s\t%s\t%s\n' "$(date '+%F %T')" "$label" "98" "$config" >> "$STATUS_FILE"
      if [ "$STOP_ON_ERROR" = "true" ]; then exit 98; fi
      return 98
    fi
    echo "[$(date '+%F %T')] eval start: $label -> $checkpoint"
    printf '%s\teval_start\t%s\t%s\t%s\n' "$(date '+%F %T')" "$label" "-" "$config" >> "$STATUS_FILE"
    "$PYTHON_CMD" eval_confusion_matrix_new.py --c "$config" --load_path "$checkpoint" --gpu "$EVAL_GPU_ID" >> "$task_log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] eval end: $label rc=$rc"
    printf '%s\teval_end\t%s\t%s\t%s\n' "$(date '+%F %T')" "$label" "$rc" "$config" >> "$STATUS_FILE"
    if [ "$rc" -ne 0 ] && [ "$STOP_ON_ERROR" = "true" ]; then
      echo "[$(date '+%F %T')] stopping queue after eval failure: $label"
      exit "$rc"
    fi
  fi
}

run_task 'main__run_1_pic_lb100_5_ulb250_5_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/main/run_1_pic_lb100_5_ulb250_5_exp.yaml' 'saved_models_segformer_mff_six_cases/run_1_pic_lb100_5_ulb250_5_exp/model_best.pth'
run_task 'main__run_2_pic_lb100_10_ulb250_10_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/main/run_2_pic_lb100_10_ulb250_10_exp.yaml' 'saved_models_segformer_mff_six_cases/run_2_pic_lb100_10_ulb250_10_exp/model_best.pth'
run_task 'main__run_3_pic_lb100_5_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/main/run_3_pic_lb100_5_ulb315_1.0_exp.yaml' 'saved_models_segformer_mff_six_cases/run_3_pic_lb100_5_ulb315_1.0_exp/model_best.pth'
run_task 'main__run_4_pic_lb100_10_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/main/run_4_pic_lb100_10_ulb315_1.0_exp.yaml' 'saved_models_segformer_mff_six_cases/run_4_pic_lb100_10_ulb315_1.0_exp/model_best.pth'
run_task 'main__run_5_pic_lb100_5_ulb250_-5_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/main/run_5_pic_lb100_5_ulb250_-5_pxe.yaml' 'saved_models_segformer_mff_six_cases/run_5_pic_lb100_5_ulb250_-5_pxe/model_best.pth'
run_task 'main__run_6_pic_lb100_10_ulb250_-10_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/main/run_6_pic_lb100_10_ulb250_-10_pxe.yaml' 'saved_models_segformer_mff_six_cases/run_6_pic_lb100_10_ulb250_-10_pxe/model_best.pth'
run_task 'no_dea_aggregator__run_1_pic_lb100_5_ulb250_5_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_dea_aggregator/run_1_pic_lb100_5_ulb250_5_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_ablations/no_dea_aggregator/run_1_pic_lb100_5_ulb250_5_exp_no_dea_aggregator/model_best.pth'
run_task 'no_dea_aggregator__run_2_pic_lb100_10_ulb250_10_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_dea_aggregator/run_2_pic_lb100_10_ulb250_10_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_ablations/no_dea_aggregator/run_2_pic_lb100_10_ulb250_10_exp_no_dea_aggregator/model_best.pth'
run_task 'no_dea_aggregator__run_3_pic_lb100_5_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_dea_aggregator/run_3_pic_lb100_5_ulb315_1.0_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_ablations/no_dea_aggregator/run_3_pic_lb100_5_ulb315_1.0_exp_no_dea_aggregator/model_best.pth'
run_task 'no_dea_aggregator__run_4_pic_lb100_10_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_dea_aggregator/run_4_pic_lb100_10_ulb315_1.0_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_ablations/no_dea_aggregator/run_4_pic_lb100_10_ulb315_1.0_exp_no_dea_aggregator/model_best.pth'
run_task 'no_dea_aggregator__run_5_pic_lb100_5_ulb250_-5_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_dea_aggregator/run_5_pic_lb100_5_ulb250_-5_pxe_no_dea_aggregator.yaml' 'saved_models_segformer_mff_ablations/no_dea_aggregator/run_5_pic_lb100_5_ulb250_-5_pxe_no_dea_aggregator/model_best.pth'
run_task 'no_dea_aggregator__run_6_pic_lb100_10_ulb250_-10_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_dea_aggregator/run_6_pic_lb100_10_ulb250_-10_pxe_no_dea_aggregator.yaml' 'saved_models_segformer_mff_ablations/no_dea_aggregator/run_6_pic_lb100_10_ulb250_-10_pxe_no_dea_aggregator/model_best.pth'
run_task 'no_mff__run_1_pic_lb100_5_ulb250_5_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_mff/run_1_pic_lb100_5_ulb250_5_exp_no_mff.yaml' 'saved_models_segformer_mff_ablations/no_mff/run_1_pic_lb100_5_ulb250_5_exp_no_mff/model_best.pth'
run_task 'no_mff__run_2_pic_lb100_10_ulb250_10_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_mff/run_2_pic_lb100_10_ulb250_10_exp_no_mff.yaml' 'saved_models_segformer_mff_ablations/no_mff/run_2_pic_lb100_10_ulb250_10_exp_no_mff/model_best.pth'
run_task 'no_mff__run_3_pic_lb100_5_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_mff/run_3_pic_lb100_5_ulb315_1.0_exp_no_mff.yaml' 'saved_models_segformer_mff_ablations/no_mff/run_3_pic_lb100_5_ulb315_1.0_exp_no_mff/model_best.pth'
run_task 'no_mff__run_4_pic_lb100_10_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_mff/run_4_pic_lb100_10_ulb315_1.0_exp_no_mff.yaml' 'saved_models_segformer_mff_ablations/no_mff/run_4_pic_lb100_10_ulb315_1.0_exp_no_mff/model_best.pth'
run_task 'no_mff__run_5_pic_lb100_5_ulb250_-5_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_mff/run_5_pic_lb100_5_ulb250_-5_pxe_no_mff.yaml' 'saved_models_segformer_mff_ablations/no_mff/run_5_pic_lb100_5_ulb250_-5_pxe_no_mff/model_best.pth'
run_task 'no_mff__run_6_pic_lb100_10_ulb250_-10_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/no_mff/run_6_pic_lb100_10_ulb250_-10_pxe_no_mff.yaml' 'saved_models_segformer_mff_ablations/no_mff/run_6_pic_lb100_10_ulb250_-10_pxe_no_mff/model_best.pth'
run_task 'dynamic_no_fuse_loss__run_1_pic_lb100_5_ulb250_5_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/dynamic_no_fuse_loss/run_1_pic_lb100_5_ulb250_5_exp_dynamic_no_fuse_loss.yaml' 'saved_models_segformer_mff_ablations/dynamic_no_fuse_loss/run_1_pic_lb100_5_ulb250_5_exp_dynamic_no_fuse_loss/model_best.pth'
run_task 'dynamic_no_fuse_loss__run_2_pic_lb100_10_ulb250_10_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/dynamic_no_fuse_loss/run_2_pic_lb100_10_ulb250_10_exp_dynamic_no_fuse_loss.yaml' 'saved_models_segformer_mff_ablations/dynamic_no_fuse_loss/run_2_pic_lb100_10_ulb250_10_exp_dynamic_no_fuse_loss/model_best.pth'
run_task 'dynamic_no_fuse_loss__run_3_pic_lb100_5_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/dynamic_no_fuse_loss/run_3_pic_lb100_5_ulb315_1.0_exp_dynamic_no_fuse_loss.yaml' 'saved_models_segformer_mff_ablations/dynamic_no_fuse_loss/run_3_pic_lb100_5_ulb315_1.0_exp_dynamic_no_fuse_loss/model_best.pth'
run_task 'dynamic_no_fuse_loss__run_4_pic_lb100_10_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/dynamic_no_fuse_loss/run_4_pic_lb100_10_ulb315_1.0_exp_dynamic_no_fuse_loss.yaml' 'saved_models_segformer_mff_ablations/dynamic_no_fuse_loss/run_4_pic_lb100_10_ulb315_1.0_exp_dynamic_no_fuse_loss/model_best.pth'
run_task 'dynamic_no_fuse_loss__run_5_pic_lb100_5_ulb250_-5_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/dynamic_no_fuse_loss/run_5_pic_lb100_5_ulb250_-5_pxe_dynamic_no_fuse_loss.yaml' 'saved_models_segformer_mff_ablations/dynamic_no_fuse_loss/run_5_pic_lb100_5_ulb250_-5_pxe_dynamic_no_fuse_loss/model_best.pth'
run_task 'dynamic_no_fuse_loss__run_6_pic_lb100_10_ulb250_-10_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/dynamic_no_fuse_loss/run_6_pic_lb100_10_ulb250_-10_pxe_dynamic_no_fuse_loss.yaml' 'saved_models_segformer_mff_ablations/dynamic_no_fuse_loss/run_6_pic_lb100_10_ulb250_-10_pxe_dynamic_no_fuse_loss/model_best.pth'
run_task 'fix_fuse_mask__run_1_pic_lb100_5_ulb250_5_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/fix_fuse_mask/run_1_pic_lb100_5_ulb250_5_exp_fix_fuse_mask.yaml' 'saved_models_segformer_mff_ablations/fix_fuse_mask/run_1_pic_lb100_5_ulb250_5_exp_fix_fuse_mask/model_best.pth'
run_task 'fix_fuse_mask__run_2_pic_lb100_10_ulb250_10_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/fix_fuse_mask/run_2_pic_lb100_10_ulb250_10_exp_fix_fuse_mask.yaml' 'saved_models_segformer_mff_ablations/fix_fuse_mask/run_2_pic_lb100_10_ulb250_10_exp_fix_fuse_mask/model_best.pth'
run_task 'fix_fuse_mask__run_3_pic_lb100_5_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/fix_fuse_mask/run_3_pic_lb100_5_ulb315_1.0_exp_fix_fuse_mask.yaml' 'saved_models_segformer_mff_ablations/fix_fuse_mask/run_3_pic_lb100_5_ulb315_1.0_exp_fix_fuse_mask/model_best.pth'
run_task 'fix_fuse_mask__run_4_pic_lb100_10_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/fix_fuse_mask/run_4_pic_lb100_10_ulb315_1.0_exp_fix_fuse_mask.yaml' 'saved_models_segformer_mff_ablations/fix_fuse_mask/run_4_pic_lb100_10_ulb315_1.0_exp_fix_fuse_mask/model_best.pth'
run_task 'fix_fuse_mask__run_5_pic_lb100_5_ulb250_-5_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/fix_fuse_mask/run_5_pic_lb100_5_ulb250_-5_pxe_fix_fuse_mask.yaml' 'saved_models_segformer_mff_ablations/fix_fuse_mask/run_5_pic_lb100_5_ulb250_-5_pxe_fix_fuse_mask/model_best.pth'
run_task 'fix_fuse_mask__run_6_pic_lb100_10_ulb250_-10_pxe' 'config/_tmp_runs/segformer_mff_six_cases_ablation_20260901_163800/fix_fuse_mask/run_6_pic_lb100_10_ulb250_-10_pxe_fix_fuse_mask.yaml' 'saved_models_segformer_mff_ablations/fix_fuse_mask/run_6_pic_lb100_10_ulb250_-10_pxe_fix_fuse_mask/model_best.pth'

echo "[$(date '+%F %T')] queue finished"
