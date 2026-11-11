import re
import os

base_yaml = """algorithm: fixmatch
save_dir: ./saved_models
resume: False
load_path: None
overwrite: True
use_tensorboard: True
epoch: 100
num_train_iter: 20000
num_eval_iter: 500
num_log_iter: 50
noise_type: asym
num_steps: 5
noise_ratio: 0.0
noise_per_class: False
batch_size: 64
eval_batch_size: 256
hard_label: True
T: 0.5
p_cutoff: 0.95
ulb_loss_ratio: 1.0
uratio: 2
ema_m: 0.999
crop_ratio: 0.875
img_size: 32
optim: AdamW
lr: 0.0005
momentum: 0.9
weight_decay: 0.05
layer_decay: 1.0
amp: False
clip_grad: 0.0
use_cat: True
net: vit_tiny_metaexpert
net_from_name: False
data_dir: ./data
dataset: pic
train_sampler: RandomSampler
num_classes: 9
num_workers: 4
seed: 2
world_size: 1
rank: 0
multiprocessing_distributed: False
dist_url: tcp://127.0.0.1:10181
dist_backend: nccl
gpu: 0
include_lb_to_ulb: False
imb_algorithm: metaexpert
la_tau_lb1: 0.0
la_tau_lb2: 2.0
la_tau_lb3: 4.0
est_epoch: 18
ema_u: 0.9
cut1: 2
cut2: 4
beta1: 0.99
beta2: 0.99
"""

configs = [
    (100, 250, 5, 5, 'exp'),
    (100, 250, 10, 10, 'exp'),
    (100, 315, 5, 1.0, 'exp'),
    (100, 315, 10, 1.0, 'exp'),
    (100, 250, 5, -5, 'pxe'),
    (100, 250, 10, -10, 'pxe'),
    (100, 250, 50, 50, 'exp'),
    (100, 250, 100, 100, 'exp'),
]

os.makedirs('config/custom_sweep', exist_ok=True)
yamls = []
for i, (lbn, ulbn, lbr, ulbr, utype) in enumerate(configs):
    name = f"run_{i+1}_pic_lb{lbn}_{lbr}_ulb{ulbn}_{ulbr}_{utype}"
    content = base_yaml
    content += f"save_name: {name}\n"
    content += f"num_labels: {lbn}\n"
    content += f"ulb_num_labels: {ulbn}\n"
    content += f"lb_imb_type: exp\n"
    content += f"ulb_imb_type: {utype}\n"
    content += f"lb_imb_ratio: {lbr}\n"
    content += f"ulb_imb_ratio: {ulbr}\n"
    
    path = f"config/custom_sweep/{name}.yaml"
    with open(path, 'w') as f:
        f.write(content)
    yamls.append(path)

with open('run_six_cases.sh', 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("set -eu\n")
    f.write('PYTHON_CMD="${PYTHON_CMD:-/home/kv/.conda/envs/cw/bin/python}"\n')
    f.write('GPU_ID="${GPU_ID:-0}"\n')
    f.write('EVAL_GPU_ID="${EVAL_GPU_ID:-0}"\n')
    f.write('export CUDA_VISIBLE_DEVICES="$GPU_ID"\n\n')
    f.write("RUNS=(\n")
    for y in yamls:
        name = y.split('/')[-1].replace('.yaml', '')
        f.write(f'  "{name}"\n')
    f.write(")\n\n")
    f.write('for RUN_NAME in "${RUNS[@]}"; do\n')
    f.write('  CONFIG="config/custom_sweep/${RUN_NAME}.yaml"\n')
    f.write('  CHECKPOINT="./saved_models/${RUN_NAME}/model_best.pth"\n\n')
    f.write('  echo "Running training for ${CONFIG}"\n')
    f.write('  "$PYTHON_CMD" train.py --c "$CONFIG"\n\n')
    f.write('  echo "Running eval for ${CONFIG}"\n')
    f.write('  "$PYTHON_CMD" eval_confusion_matrix_new.py --c "$CONFIG" --load_path "$CHECKPOINT" --gpu "$EVAL_GPU_ID"\n')
    f.write('done\n')

print("Created custom_sweep YAMLs and run_six_cases.sh")
