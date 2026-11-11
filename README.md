# A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning

## Local experiment code / 本地实验代码

This repository contains the local raw-signal MetaExpert experiments, including
LLA/DRP extensions, baseline comparisons, ablations, evaluation, and plotting tools.
The original implementation and citation are described below.

- Core implementation: `semilearn/`.
- Training entry points: `train.py`, `train_lla_drp.py`, `train_lla_e3.py`.
- Experiment preparation and execution: root-level `prepare_*.py`, `run_*.py`,
  and shell/PowerShell scripts.
- Verification: `tests/` and root-level `verify_*.py`.
- Experiment source and YAML configurations available in this checkout are also
  retained under `logs/`; generated logs and checkpoints are ignored.

数据集、模型权重、生成的图表、运行日志及本地备份不随源码上传。
运行前请自行准备下文说明的数据文件。各实验配置也可通过相应的
`prepare_*.py` 脚本生成；请根据本机环境检查数据路径、Python 环境和 GPU 设置。

Yaxin Hou, Yuheng Jia, A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning, International Conference on Machine Learning, 13th-19th July, Vancouver, 2025.

This is an official [PyTorch](http://pytorch.org) implementation for **A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning**.

## Introduction

This code is based on the public and widely-used codebase [USB](https://github.com/microsoft/Semi-supervised-learning) and the previous method [CPE](https://github.com/machengcheng2016/CPE-LTSSL).

What I've done is just adding our Meta-Expert algorithm in `semilearn/imb_algorithms/metaexpert`.

Also, I've made corresponding modifications to `semilearn/nets/` and several `__init__.py`.

## How to run

For example, on CIFAR-10-LT with $\gamma_l=\gamma_u=150$

```
CUDA_VISIBLE_DEVICES=0 python train.py --c "/config/002-fixmatch_metaexpert_cifar10_lb1500_150_ulb3000_150_0.0_2.yaml"
```

(Note: I know that USB supports multi-GPUs, but I still recommend you to run on single GPU, as some weird problems may occur.)

The model will be automatically evaluated every 1024 iterations during training. After training, the last two lines in `saved_models/002-fixmatch_metaexpert_cifar10_lb1500_150_ulb3000_150_0.0_2/log.txt` will tell you the best accuracy.

### Raw vibration-signal dataset

The custom pump dataset is loaded from `data/raw_signal_dataset_G_1024.npz`. The
archive must contain `data` with shape `[num_samples, signal_length]`, `labels`,
and `class_names`. The active dataset has 4,500 signals of length 1,024 in
9 classes. Training uses 1-D weak/strong augmentation and the MetaExpert ViT
uses a Conv1d patch embedding; no files under `data/pic` are read.

当前训练标签（与 NPZ 中的 `labels` 和 `class_names` 一致）：

| 标签 | 类别 |
|---|---|
| 0 | 健康 |
| 1 | 主轴疲劳 |
| 2 | 缸体轴外圈疲劳 |
| 3 | 缸体轴内圈疲劳 |
| 4 | 柱塞磨损 |
| 5 | 斜盘磨损 |
| 6 | 配油盘磨损 |
| 7 | 弹簧失效 |
| 8 | 滑靴塌边 |

`python relabel_raw_signal.py` 可将三份原始 NPZ 数据迁移到上述标签，并自动备份；重复运行不会再次重映射。
迁移时同时更新类别名称与按类别存储的来源文件，保持信号及样本顺序不变。
采样缓存按类别名称顺序隔离，避免复用旧标签对应的索引。
旧模型、历史预测和实验结果仍使用其训练时的旧标签；新标签训练应使用新的输出目录并从头训练，避免恢复旧模型。

Run the main raw-signal configuration with:

```
python train.py --c config/002-fixmatch_metaexpert_pic_lb900_150_ulb1800_150_0.0_2.yaml
```

For example,

```
[2024-07-26 03:54:08,086 INFO] model saved: ./saved_models/002-fixmatch_metaexpert_cifar10_lb1500_150_ulb3000_150_0.0_2/latest_model.pth
[2024-07-26 03:54:08,089 INFO] Model result - eval/best_acc : 0.8248
[2024-07-26 03:54:08,090 INFO] Model result - eval/best_it : 179199
```

## Results

The reported accuracies in Table 3 and 4 in our paper are the average over three different runs (random seeds are 0/2/4).

## Citation

If you find our method useful, please consider citing our paper:

```
  @inproceedings{metaexperticml2025,
    title={A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning},
    author={Hou, Yaxin and Jia, Yuheng},
    booktitle={International Conference on Machine Learning},
    volume={},
    pages={},
    year={2025}
  }
```

linux服务器上运行：

nohup bash run_pic_ratio_sweep.sh 0 4 /home/kv/.conda/envs/cw/bin/python 1 > logs/ratio_sweep/nohup_1_8_train_eval.log 2>&1 &





eval 命令
/home/kv/.conda/envs/cw/bin/python eval_confusion_matrix.py
  --c config/002-fixmatch_metaexpert_pic_lb900_150_ulb1800_150_0.0_2.yaml
  --load_path saved_models/002-fixmatch_metaexpert_pic_lb900_150_ulb1800_150_0.0_2_seed0/model_best.pth
  --gpu 0
