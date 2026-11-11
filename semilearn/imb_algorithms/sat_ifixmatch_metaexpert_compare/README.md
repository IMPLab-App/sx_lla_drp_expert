# SAT-iFixMatch for comparison with Meta-Expert

This package is a PyTorch reproduction-style implementation of **SAT-iFixMatch** from the paper *A holistic semi-supervised method for imbalanced fault diagnosis of rotational machinery with out-of-distribution samples*.

It is designed as a comparison method for **Meta-Expert** under the same fault-diagnosis split.

## Implemented components

1. FixMatch-style semi-supervised learning: supervised loss on labeled data plus pseudo-label loss on unlabeled data.
2. Weak augmentation: amplitude scaling.
3. Strong augmentation: random masking noise.
4. Self-adaptive threshold: global threshold + class-wise local threshold.
5. OOD detection by prediction consistency between weak and strong views.
6. OOD uniformity loss and bidirectional KL consistency regularization.

## Files

```text
sat_ifixmatch_metaexpert_compare/
├── train_sat_ifixmatch.py
├── make_splits.py
├── model.py
├── augment.py
├── sat_ifixmatch.py
├── data_utils.py
├── metrics.py
├── requirements.txt
└── README.md
```

## Data format

```python
import numpy as np
np.savez('train_lb.npz', x=x_lb, y=y_lb)
np.savez('train_ulb.npz', x=x_ulb)      # y is optional and ignored
np.savez('test.npz', x=x_test, y=y_test)
```

Accepted signal shapes: `[N, L]`, `[N, C, L]`, or `[N, L, C]` with `--channel_last`.
Known labels must be `0,...,K-1`. OOD labels can be `-1` or labels outside `[0,K-1]`.

## Train

```bash
pip install -r requirements.txt
python train_sat_ifixmatch.py \
  --train_lb_npz train_lb.npz \
  --train_ulb_npz train_ulb.npz \
  --test_npz test.npz \
  --num_classes 9 \
  --epochs 100 \
  --batch_size_lb 32 \
  --batch_size_ulb 128 \
  --out_dir runs/sat_ifixmatch_exp1
```

## Create comparable splits

```bash
python make_splits.py \
  --full_npz all_data.npz \
  --out_dir splits/exp1 \
  --num_classes 9 \
  --max_per_class 300 \
  --lt_rates "1,0.5,0.3,0.2,0.5,0.3,0.2,0.5,0.3" \
  --label_ratio 0.1 \
  --test_ratio 0.2
```

For strict comparison with Meta-Expert, use the same `train_lb`, `train_ulb`, and `test` files for both methods. If OOD samples are not part of your study, keep `--ood_classes ""` and compare Accuracy, Macro-F1, G-mean, and tail-class recall.

## PIC comparison runs

This folder can also run directly on the repository PIC dataset with the same fixed split used by the baselines:

- 70% `train_data`
- 10% `eval` for best-checkpoint selection during training
- 20% `test` for final reporting only

```bash
bash semilearn/imb_algorithms/sat_ifixmatch_metaexpert_compare/run_pic_sat_ifixmatch_compare.sh
```

The batch script runs the same PIC case names used by the baseline sweep and writes outputs to:

```text
saved_models/sat_ifixmatch_<case_name>/
logs/sat_ifixmatch_metaexpert_compare/
```

Each finished case writes the main final test confusion matrix to:

```text
confusion_matrix/model_best_test.csv
confusion_matrix/model_best_test.png
```

The CSV is row-normalized by the fixed PIC test class counts. Rejected/OOD predictions are counted as wrong for accuracy and are recorded in `model_best_test_counts.json`.

