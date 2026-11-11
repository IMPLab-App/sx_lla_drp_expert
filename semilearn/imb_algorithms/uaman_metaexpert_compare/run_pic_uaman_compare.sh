#!/bin/bash
set -uo pipefail

python_bin=${PYTHON_BIN:-/home/kv/.conda/envs/cw/bin/python}
gpu=${GPU:-0}
epochs=${EPOCHS:-100}
batch_size=${BATCH_SIZE:-64}
adapt_steps=${ADAPT_STEPS:-200}
log_dir="logs/uaman_metaexpert_compare"
mkdir -p "${log_dir}"

script="semilearn/imb_algorithms/uaman_metaexpert_compare/train_uaman.py"

cases=(
  "run_1_pic_lb100_5_ulb250_5_exp exp 5 exp 5 250"
  "run_1_pic_lb100_50_ulb250_50_exp exp 50 exp 50 250"
  "run_2_pic_lb100_10_ulb250_10_exp exp 10 exp 10 250"
  "run_2_pic_lb100_100_ulb250_100_exp exp 100 exp 100 250"
  "run_3_pic_lb100_5_ulb315_1.0_exp exp 5 exp 1.0 315"
  "run_3_pic_lb100_50_ulb315_1.0_exp exp 50 exp 1.0 315"
  "run_4_pic_lb100_10_ulb315_1.0_exp exp 10 exp 1.0 315"
  "run_4_pic_lb100_100_ulb315_1.0_exp exp 100 exp 1.0 315"
  "run_5_pic_lb100_5_ulb250_-5_pxe exp 5 pxe -5 250"
  "run_5_pic_lb100_50_ulb250_-50_pxe exp 50 pxe -50 250"
  "run_6_pic_lb100_10_ulb250_-10_pxe exp 10 pxe -10 250"
  "run_6_pic_lb100_100_ulb250_-100_pxe exp 100 pxe -100 250"
)

failures=()
for row in "${cases[@]}"; do
  read -r case_name lb_type lb_ratio ulb_type ulb_ratio ulb_num_labels <<< "${row}"
  name="uaman_${case_name}"
  out_dir="saved_models/${name}"
  log_file="${log_dir}/${name}.log"
  final_png="${out_dir}/confusion_matrix/model_best_test.png"

  if [[ -f "${final_png}" ]]; then
    echo "[SKIP] ${name}: final test result exists"
    continue
  fi

  echo "[RUN] ${name} on GPU ${gpu}"
  if ! "${python_bin}" "${script}" \
    --dataset pic \
    --data_dir ./data \
    --num_classes 9 \
    --input_channels 3 \
    --img_size 32 \
    --crop_ratio 0.875 \
    --num_labels 100 \
    --ulb_num_labels "${ulb_num_labels}" \
    --lb_imb_type "${lb_type}" \
    --lb_imb_ratio "${lb_ratio}" \
    --ulb_imb_type "${ulb_type}" \
    --ulb_imb_ratio "${ulb_ratio}" \
    --num_steps 5 \
    --noise_ratio 0.0 \
    --seed 2 \
    --split_seed 2 \
    --epochs "${epochs}" \
    --batch_size "${batch_size}" \
    --adapt_steps "${adapt_steps}" \
    --adapt_split eval \
    --gpu "${gpu}" \
    --out_dir "${out_dir}" 2>&1 | tee "${log_file}"; then
      failures+=("${name}")
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "UA-MAN PIC compare finished with failures:"
  printf '  - %s\n' "${failures[@]}"
  exit 1
fi

echo "UA-MAN PIC compare finished successfully."
