# UA-MAN reproduction for comparison with Meta-Expert

This package implements a **PyTorch 1D signal version** of:

> Multi-expert Attention Network with Unsupervised Aggregation (UA-MAN)  
> for long-tailed fault diagnosis under speed variation.

It is designed so that you can compare it with **Meta-Expert** under the same fault-diagnosis split, input length, class order, random seed, and metrics.

## What is implemented

1. **Shared convolutional token embedding**
   - Input vibration segment: `[B, C, L]` or `[B, L]`.
   - A residual 1D CNN extracts local low-level features.
   - A patch embedding layer converts the feature map into token sequences.

2. **Three expert branches**
   - `E1`: long-tailed expert, trained with standard token-level cross entropy.
   - `E2`: uniform expert, trained with Balanced Softmax / log-prior adjustment.
   - `E3`: inverse long-tailed expert, trained with inverse log-prior adjustment.

3. **Unsupervised aggregation**
   - The model is frozen.
   - Only three expert weights are optimized on unlabeled target/test data.
   - Two augmented views are generated: `x1 = x + noise`, `x2 = -x + noise`.
   - Expert weights are optimized by maximizing cosine similarity between the two prediction distributions.

## Files

```text
uaman_metaexpert_compare/
├── train_uaman.py       # main training/evaluation script
├── uaman_model.py       # UA-MAN model: shared CNN + 1D Swin-like experts
├── losses.py            # LCE, LBalCE, LInvCE
├── data_utils.py        # npz dataset, long-tail sampler
├── metrics.py           # accuracy, macro-F1, G-mean, per-class recall
├── requirements.txt
└── README.md
```

## Dataset format

Prepare `.npz` files:

```python
import numpy as np

# x_train: [N, L] or [N, C, L]
# y_train: [N], labels must be 0,1,...,C-1
np.savez('train.npz', x=x_train, y=y_train)

# x_test: [N, L] or [N, C, L]
# y_test: [N]
np.savez('test.npz', x=x_test, y=y_test)
```

## Basic use

```bash
pip install -r requirements.txt

python train_uaman.py \
  --train_npz train.npz \
  --test_npz test.npz \
  --num_classes 9 \
  --input_channels 1 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --max_per_class 300 \
  --lt_rates "1,0.5,0.3,0.2,0.5,0.3,0.2,0.5,0.3" \
  --adapt_steps 200 \
  --out_dir runs/uaman_exp1
```

`--lt_rates` controls the long-tailed training-set size. If `--max_per_class 300`, then `lt_rates="1,0.5,0.3"` gives class counts `[300,150,90]`.


## PIC dataset use in this repository

This repo now supports using the shared Semilearn PIC loader directly:

```bash
python semilearn/imb_algorithms/uaman_metaexpert_compare/train_uaman.py \
  --dataset pic \
  --data_dir ./data \
  --num_classes 9 \
  --input_channels 3 \
  --num_labels 100 \
  --ulb_num_labels 250 \
  --lb_imb_type exp \
  --lb_imb_ratio 50 \
  --ulb_imb_type exp \
  --ulb_imb_ratio 50 \
  --seed 2 \
  --split_seed 2 \
  --epochs 100 \
  --batch_size 64 \
  --adapt_steps 200 \
  --adapt_split eval \
  --gpu 0 \
  --out_dir saved_models/uaman_run_1_pic_lb100_50_ulb250_50_exp
```

For `--dataset pic`, the script reuses `semilearn.datasets.cv_datasets.pic.get_pic()`, so it uses the same fixed class-wise split as the baselines:

```text
70% train_data, 10% eval, 20% test
PIC test: 9 classes x 100 images = 900 images
```

The UA-MAN model is supervised on the labeled train subset. Best checkpoint selection uses the fixed eval split. Final metrics are computed once on the fixed test split. PIC images are converted from `[C, H, W]` to UA-MAN's 1D input format `[C, H*W]`.

The main comparison result is saved as unsupervised aggregation on the test set:

```text
<out_dir>/results.json
<out_dir>/confusion_matrix/model_best_test.csv
<out_dir>/confusion_matrix/model_best_test.png
```

The equal expert average test result is also saved:

```text
<out_dir>/confusion_matrix/avg_expert_test.csv
<out_dir>/confusion_matrix/avg_expert_test.png
```

To run the 12 PIC case settings matching `config/custom_sweep/baselines`, use:

```bash
GPU=0 EPOCHS=100 BATCH_SIZE=64 ADAPT_STEPS=200 \
  bash semilearn/imb_algorithms/uaman_metaexpert_compare/run_pic_uaman_compare.sh
```

Outputs are written to `saved_models/uaman_<case_name>`.

## For fair comparison with Meta-Expert

Meta-Expert is a **long-tailed semi-supervised learning** method. UA-MAN is mainly a **supervised long-tailed multi-expert fault-diagnosis** method with an unlabeled target/test aggregation stage.

Recommended setting:

```text
Meta-Expert:
  train_lb + train_ulb -> test

UA-MAN:
  train_lb only for supervised multi-expert training
  target/test x without labels for unsupervised aggregation
  target/test y only for final evaluation
```

Use the same dataset, class order, signal length, train/test split, imbalance ratio, random seeds, and metrics: Accuracy, Macro-F1, G-mean, Tail-class Recall.
