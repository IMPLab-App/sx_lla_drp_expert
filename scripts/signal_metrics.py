"""Common closed-set metrics, counting every rejected prediction as an error."""
import csv
import json
from pathlib import Path
import numpy as np


def write_signal_metrics(out_dir, y_true, y_pred, num_classes, class_names=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    assert true.shape == pred.shape and len(true)
    counts = np.zeros((num_classes, num_classes + 1), dtype=np.int64)
    valid = (pred >= 0) & (pred < num_classes)
    np.add.at(counts, (true, np.where(valid, pred, num_classes)), 1)
    support = counts.sum(1)
    tp = np.diag(counts[:, :num_classes])
    predicted = counts[:, :num_classes].sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(num_classes), where=support + predicted > 0)
    metrics = dict(samples=int(len(true)), accuracy=float((true == pred).mean()),
                   macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                   macro_f1=float(f1.mean()), weighted_f1=float(np.average(f1, weights=support)),
                   rejected=int((~valid).sum()), class_support=support.tolist(),
                   class_recall=recall.tolist(), confusion_counts=counts.tolist())
    (out_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    np.savez_compressed(out_dir / 'test_predictions.npz', y_true=true, y_pred=pred)
    names = class_names or [str(i) for i in range(num_classes)]
    with (out_dir / 'test_confusion_counts.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true/pred'] + names + ['rejected'])
        writer.writerows([[name] + row.tolist() for name, row in zip(names, counts)])
    return metrics
