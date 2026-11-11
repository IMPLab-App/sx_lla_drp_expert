#!/usr/bin/env bash
# Six sequential experiments per GPU, with frozen code and the new 0..8 labels.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TZ=Asia/Shanghai
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"
export PYTHON_CMD
mkdir -p logs
exec 9>logs/new_labels_twelve_dual_gpu.lock
flock -n 9 || { echo 'The new-label twelve-task queue is already running.' >&2; exit 1; }
LOG_DIR="${LOG_DIR:-logs/new_labels_twelve_dual_gpu_$(date +%Y%m%d_%H%M%S)}"
# Refuse to reuse an earlier batch directory or overwrite its checkpoints.
mkdir -- "$LOG_DIR"
LOG_DIR=$(realpath -- "$LOG_DIR")
export LOG_DIR

"$PYTHON_CMD" - "$LOG_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

root = Path.cwd()
output = Path(sys.argv[1])
expected_names = ['健康', '主轴疲劳', '缸体轴外圈疲劳', '缸体轴内圈疲劳',
                  '柱塞磨损', '斜盘磨损', '配油盘磨损', '弹簧失效', '滑靴塌边']
standard = [
    'run_1_pic_lb100_5_ulb250_5_exp',
    'run_2_pic_lb100_10_ulb250_10_exp',
    'run_3_pic_lb100_5_ulb315_1.0_exp',
    'run_4_pic_lb100_10_ulb315_1.0_exp',
    'run_5_pic_lb100_5_ulb250_-5_pxe',
    'run_6_pic_lb100_10_ulb250_-10_pxe',
]
long_tail = [
    'run_1_pic_lb100_50_ulb250_50_exp',
    'run_2_pic_lb100_100_ulb250_100_exp',
    'run_3_pic_lb100_50_ulb315_1.0_exp',
    'run_4_pic_lb100_100_ulb315_1.0_exp',
    'run_5_pic_lb100_50_ulb250_-50_pxe',
    'run_6_pic_lb100_100_ulb250_-100_pxe',
]
configs = []
for name in standard + long_tail:
    source = root / 'config/custom_sweep' / (name + '.yaml')
    settings = yaml.safe_load(source.read_text())
    assert settings['save_name'] == name, name
    assert settings['num_classes'] == 9, name
    configs.append((source, settings))

data_path = root / 'data/raw_signal_dataset_G_1024.npz'
with np.load(data_path, allow_pickle=False) as data:
    assert data['class_names'].tolist() == expected_names
    assert data['data'].shape == (4500, 1024)
    np.testing.assert_array_equal(np.unique(data['labels']), np.arange(9))

for directory in ('configs', 'data', 'models', 'code/logs', 'gpu0', 'gpu1'):
    (output / directory).mkdir(parents=True, exist_ok=True)
shutil.copy2(data_path, output / 'data' / data_path.name)
shutil.copytree(root / 'semilearn', output / 'code/semilearn',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ('train.py', 'eval_confusion_matrix.py', 'eval_confusion_matrix_new.py',
             'publication_plotting.py', 'run_requested_six_cases.sh'):
    shutil.copy2(root / name, output / 'code' / name)

tasks = []
for index, (source, settings) in enumerate(configs):
    settings.update(save_dir=str(output / 'models'), data_dir=str(output / 'data'),
                    resume=False, load_path=None, overwrite=False, gpu=0,
                    world_size=1, rank=0, multiprocessing_distributed=False)
    destination = output / 'configs' / source.name
    destination.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
    tasks.append(dict(name=settings['save_name'], gpu=index % 2,
                      sequence=index // 2 + 1, config=str(destination),
                      source_config=str(source), status='pending'))
code_hashes = {str(path.relative_to(output / 'code')):
               hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((output / 'code').rglob('*')) if path.is_file()}
manifest = dict(class_names=expected_names,
                label_mapping={str(i): name for i, name in enumerate(expected_names)},
                source_dataset=str(data_path), dataset=str(output / 'data' / data_path.name),
                dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                python=sys.executable, tasks=tasks, code_sha256=code_hashes)
(output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
with (output / 'tasks.tsv').open('w') as table:
    table.write('gpu\tsequence\trun\n')
    for task in sorted(tasks, key=lambda task: (task['gpu'], task['sequence'])):
        table.write(f"{task['gpu']}\t{task['sequence']}\t{task['name']}\n")
print(f'Prepared {len(tasks)} tasks; dataset labels: {expected_names}', flush=True)
PY

printf '%s\n' "$LOG_DIR" > logs/new_labels_twelve_dual_gpu.latest
echo "$$" > "$LOG_DIR/queue.pid"
printf 'running\n' > "$LOG_DIR/status.txt"
worker() {
  local GPU="$1"
  for SET in standard long_tail; do
    mkdir -p "$LOG_DIR/gpu$GPU/$SET"
    CONFIG_DIR="$LOG_DIR/configs" EXPERIMENT_SET="$SET" GPU_ID="$GPU" \
      WORKER_INDEX="$GPU" WORKER_COUNT=2 LOG_DIR="$LOG_DIR/gpu$GPU/$SET" \
      bash "$LOG_DIR/code/run_requested_six_cases.sh" \
      > "$LOG_DIR/gpu$GPU/$SET/queue.log" 2>&1 || return "$?"
  done
}
worker 0 > "$LOG_DIR/gpu0/queue.log" 2>&1 &
PID0=$!
worker 1 > "$LOG_DIR/gpu1/queue.log" 2>&1 &
PID1=$!
printf 'gpu\tpid\n0\t%s\n1\t%s\n' "$PID0" "$PID1" > "$LOG_DIR/workers.tsv"
echo "[$(date -Is)] Started: queue=$$ GPU0=$PID0 GPU1=$PID1; output=$LOG_DIR"
CODE0=0
CODE1=0
wait "$PID0" || CODE0=$?
wait "$PID1" || CODE1=$?
printf 'gpu\texit_code\n0\t%s\n1\t%s\n' "$CODE0" "$CODE1" > "$LOG_DIR/exit_codes.tsv"
if (( CODE0 == 0 && CODE1 == 0 )); then
  printf 'completed\n' > "$LOG_DIR/status.txt"
  echo "[$(date -Is)] All twelve training and evaluation tasks completed."
else
  printf 'failed\n' > "$LOG_DIR/status.txt"
  echo "[$(date -Is)] Worker failure: GPU0=$CODE0 GPU1=$CODE1" >&2
  exit 1
fi
