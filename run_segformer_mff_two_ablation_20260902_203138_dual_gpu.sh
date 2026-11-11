#!/bin/bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
RUN_EVAL="${RUN_EVAL:-true}"
STOP_ON_ERROR="${STOP_ON_ERROR:-true}"
LOG_ROOT='logs/segformer_mff_two_ablation_20260902_203138'
MASTER_LOG="$LOG_ROOT/dual_gpu_queue.log"
STATUS_FILE="$LOG_ROOT/dual_gpu_status.tsv"
mkdir -p "$LOG_ROOT"
exec >> "$MASTER_LOG" 2>&1
echo -e "time\tworker\tphysical_gpu\tphase\tlabel\trc\tconfig" > "$STATUS_FILE"
echo "[$(date '+%F %T')] two-ablation dual-gpu queue started; eval=$RUN_EVAL"

run_task() {
  local worker="$1"
  local physical_gpu="$2"
  local label="$3"
  local config="$4"
  local checkpoint="$5"
  local task_log="$LOG_ROOT/${worker}__${label}.log"
  export CUDA_VISIBLE_DEVICES="$physical_gpu"
  echo "[$(date '+%F %T')] [$worker gpu=$physical_gpu] train start: $label"
  printf '%s\t%s\t%s\ttrain_start\t%s\t%s\t%s\n' "$(date '+%F %T')" "$worker" "$physical_gpu" "$label" "-" "$config" >> "$STATUS_FILE"
  "$PYTHON_CMD" train.py --c "$config" >> "$task_log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] [$worker gpu=$physical_gpu] train end: $label rc=$rc"
  printf '%s\t%s\t%s\ttrain_end\t%s\t%s\t%s\n' "$(date '+%F %T')" "$worker" "$physical_gpu" "$label" "$rc" "$config" >> "$STATUS_FILE"
  if [ "$rc" -ne 0 ]; then
    if [ "$STOP_ON_ERROR" = "true" ]; then return "$rc"; fi
    return 0
  fi
  if [ "$RUN_EVAL" = "true" ]; then
    if [ ! -f "$checkpoint" ]; then
      echo "[$(date '+%F %T')] [$worker gpu=$physical_gpu] missing checkpoint: $checkpoint"
      printf '%s\t%s\t%s\teval_missing_checkpoint\t%s\t%s\t%s\n' "$(date '+%F %T')" "$worker" "$physical_gpu" "$label" "98" "$config" >> "$STATUS_FILE"
      if [ "$STOP_ON_ERROR" = "true" ]; then return 98; fi
      return 0
    fi
    echo "[$(date '+%F %T')] [$worker gpu=$physical_gpu] eval start: $label"
    printf '%s\t%s\t%s\teval_start\t%s\t%s\t%s\n' "$(date '+%F %T')" "$worker" "$physical_gpu" "$label" "-" "$config" >> "$STATUS_FILE"
    "$PYTHON_CMD" eval_confusion_matrix_new.py --c "$config" --load_path "$checkpoint" --gpu "$EVAL_GPU_ID" >> "$task_log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] [$worker gpu=$physical_gpu] eval end: $label rc=$rc"
    printf '%s\t%s\t%s\teval_end\t%s\t%s\t%s\n' "$(date '+%F %T')" "$worker" "$physical_gpu" "$label" "$rc" "$config" >> "$STATUS_FILE"
    if [ "$rc" -ne 0 ] && [ "$STOP_ON_ERROR" = "true" ]; then return "$rc"; fi
  fi
  return 0
}

worker_0() {
  local physical_gpu="0"
  run_task worker_0 "$physical_gpu" 'no_dea_aggregator__run_1_pic_lb100_5_ulb250_5_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_dea_aggregator/run_1_pic_lb100_5_ulb250_5_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_two_ablations/no_dea_aggregator/run_1_pic_lb100_5_ulb250_5_exp_no_dea_aggregator/model_best.pth' || return $?
  run_task worker_0 "$physical_gpu" 'no_dea_aggregator__run_3_pic_lb100_5_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_dea_aggregator/run_3_pic_lb100_5_ulb315_1.0_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_two_ablations/no_dea_aggregator/run_3_pic_lb100_5_ulb315_1.0_exp_no_dea_aggregator/model_best.pth' || return $?
  run_task worker_0 "$physical_gpu" 'no_dea_aggregator__run_5_pic_lb100_5_ulb250_-5_pxe' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_dea_aggregator/run_5_pic_lb100_5_ulb250_-5_pxe_no_dea_aggregator.yaml' 'saved_models_segformer_mff_two_ablations/no_dea_aggregator/run_5_pic_lb100_5_ulb250_-5_pxe_no_dea_aggregator/model_best.pth' || return $?
  run_task worker_0 "$physical_gpu" 'no_mff__run_1_pic_lb100_5_ulb250_5_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_mff/run_1_pic_lb100_5_ulb250_5_exp_no_mff.yaml' 'saved_models_segformer_mff_two_ablations/no_mff/run_1_pic_lb100_5_ulb250_5_exp_no_mff/model_best.pth' || return $?
  run_task worker_0 "$physical_gpu" 'no_mff__run_3_pic_lb100_5_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_mff/run_3_pic_lb100_5_ulb315_1.0_exp_no_mff.yaml' 'saved_models_segformer_mff_two_ablations/no_mff/run_3_pic_lb100_5_ulb315_1.0_exp_no_mff/model_best.pth' || return $?
  run_task worker_0 "$physical_gpu" 'no_mff__run_5_pic_lb100_5_ulb250_-5_pxe' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_mff/run_5_pic_lb100_5_ulb250_-5_pxe_no_mff.yaml' 'saved_models_segformer_mff_two_ablations/no_mff/run_5_pic_lb100_5_ulb250_-5_pxe_no_mff/model_best.pth' || return $?
}

worker_1() {
  local physical_gpu="1"
  run_task worker_1 "$physical_gpu" 'no_dea_aggregator__run_2_pic_lb100_10_ulb250_10_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_dea_aggregator/run_2_pic_lb100_10_ulb250_10_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_two_ablations/no_dea_aggregator/run_2_pic_lb100_10_ulb250_10_exp_no_dea_aggregator/model_best.pth' || return $?
  run_task worker_1 "$physical_gpu" 'no_dea_aggregator__run_4_pic_lb100_10_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_dea_aggregator/run_4_pic_lb100_10_ulb315_1.0_exp_no_dea_aggregator.yaml' 'saved_models_segformer_mff_two_ablations/no_dea_aggregator/run_4_pic_lb100_10_ulb315_1.0_exp_no_dea_aggregator/model_best.pth' || return $?
  run_task worker_1 "$physical_gpu" 'no_dea_aggregator__run_6_pic_lb100_10_ulb250_-10_pxe' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_dea_aggregator/run_6_pic_lb100_10_ulb250_-10_pxe_no_dea_aggregator.yaml' 'saved_models_segformer_mff_two_ablations/no_dea_aggregator/run_6_pic_lb100_10_ulb250_-10_pxe_no_dea_aggregator/model_best.pth' || return $?
  run_task worker_1 "$physical_gpu" 'no_mff__run_2_pic_lb100_10_ulb250_10_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_mff/run_2_pic_lb100_10_ulb250_10_exp_no_mff.yaml' 'saved_models_segformer_mff_two_ablations/no_mff/run_2_pic_lb100_10_ulb250_10_exp_no_mff/model_best.pth' || return $?
  run_task worker_1 "$physical_gpu" 'no_mff__run_4_pic_lb100_10_ulb315_1.0_exp' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_mff/run_4_pic_lb100_10_ulb315_1.0_exp_no_mff.yaml' 'saved_models_segformer_mff_two_ablations/no_mff/run_4_pic_lb100_10_ulb315_1.0_exp_no_mff/model_best.pth' || return $?
  run_task worker_1 "$physical_gpu" 'no_mff__run_6_pic_lb100_10_ulb250_-10_pxe' 'config/_tmp_runs/segformer_mff_two_ablation_20260902_203138/no_mff/run_6_pic_lb100_10_ulb250_-10_pxe_no_mff.yaml' 'saved_models_segformer_mff_two_ablations/no_mff/run_6_pic_lb100_10_ulb250_-10_pxe_no_mff/model_best.pth' || return $?
}

worker_0 &
PID0=$!
worker_1 &
PID1=$!
echo "[$(date '+%F %T')] worker_0 pid=$PID0 gpu=0 tasks=6"
echo "[$(date '+%F %T')] worker_1 pid=$PID1 gpu=1 tasks=6"
RC=0
wait $PID0 || RC=$?
wait $PID1 || RC=$?
echo "[$(date '+%F %T')] two-ablation dual-gpu queue finished rc=$RC"
exit "$RC"
