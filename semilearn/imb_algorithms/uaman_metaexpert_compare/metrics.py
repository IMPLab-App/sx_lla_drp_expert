import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm

def macro_f1(y_true, y_pred, num_classes, eps=1e-12):
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1s.append(2 * precision * recall / (precision + recall + eps))
    return float(np.mean(f1s))

def per_class_recall(y_true, y_pred, num_classes, eps=1e-12):
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    return [float(cm[c, c] / (cm[c, :].sum() + eps)) for c in range(num_classes)]

def gmean(y_true, y_pred, num_classes, eps=1e-12):
    recalls = np.asarray(per_class_recall(y_true, y_pred, num_classes, eps=eps))
    return float(np.exp(np.mean(np.log(recalls + eps))))

def summarize_metrics(y_true, y_pred, num_classes):
    return {
        'accuracy': accuracy(y_true, y_pred),
        'macro_f1': macro_f1(y_true, y_pred, num_classes),
        'gmean': gmean(y_true, y_pred, num_classes),
        'per_class_recall': per_class_recall(y_true, y_pred, num_classes),
        'confusion_matrix': confusion_matrix_np(y_true, y_pred, num_classes).tolist(),
    }
