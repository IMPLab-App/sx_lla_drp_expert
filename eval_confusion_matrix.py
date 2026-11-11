# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Standalone evaluation script for a trained raw-signal model.

It loads an existing checkpoint, uses the held-out raw-signal test split with the same
configuration/seed, and saves the confusion matrix as both CSV and PNG.

This script does not train or modify the existing training pipeline.
"""

import argparse
import csv
import glob
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from semilearn.core.utils import get_net_builder
from semilearn.core.utils.misc import over_write_args_from_file
from semilearn.datasets import get_raw_signal


def _get_cli_override_dests(parser):
    """
    collect argparse dest names explicitly provided from CLI
    """
    option_to_dest = {}
    for action in parser._actions: 
        for opt in action.option_strings:
            option_to_dest[opt] = action.dest

    override_dests = set()
    for token in sys.argv[1:]:
        if not token.startswith('-'):
            continue
        if '=' in token:
            opt = token.split('=', 1)[0]
        else:
            opt = token
        dest = option_to_dest.get(opt)
        if dest:
            override_dests.add(dest)
    return override_dests


def build_args():
    parser = argparse.ArgumentParser(description="Standalone PIC evaluation with confusion matrix export")

    parser.add_argument("--load_path", type=str, required=True, help="Path to a checkpoint file or checkpoint directory")
    parser.add_argument("--c", type=str, default="", help="Path to the training YAML config")

    # model / data config
    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--dataset", type=str, default="raw_signal")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--crop_ratio", type=float, default=0.875)

    # split config used by get_raw_signal
    parser.add_argument("--num_labels", type=int, default=1500)
    parser.add_argument("--ulb_num_labels", type=int, default=3000)
    parser.add_argument("--lb_imb_ratio", type=int, default=150)
    parser.add_argument("--ulb_imb_ratio", type=int, default=150)
    parser.add_argument("--lb_imb_type", type=str, default="exp")
    parser.add_argument("--ulb_imb_type", type=str, default="exp")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--noise_type", type=str, default="asym")
    parser.add_argument("--noise_ratio", type=float, default=0.0)
    parser.add_argument("--noise_per_class", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--include_lb_to_ulb", type=lambda x: str(x).lower() == "true", default=False)

    # runtime config
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use. Set to -1 for CPU evaluation.")

    # output config
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save confusion matrix files")

    args = parser.parse_args()
    cli_values = vars(args).copy()
    cli_override_dests = _get_cli_override_dests(parser)

    if args.c:
        config_path = args.c
        if not os.path.exists(config_path):
            candidate = os.path.join("config", args.c)
            if os.path.exists(candidate):
                config_path = candidate
            else:
                raise FileNotFoundError(
                    f"Config file not found: {args.c}. "
                    f"Please pass an existing path, e.g. config/{args.c}"
                )
        args.c = config_path
        cli_values['c'] = config_path
    over_write_args_from_file(args, args.c)

    # Keep CLI arguments as the highest-priority source.
    for dest in cli_override_dests:
        if dest in cli_values:
            setattr(args, dest, cli_values[dest])

    return args


def resolve_checkpoint_path(load_path):
    if os.path.isdir(load_path):
        best_path = os.path.join(load_path, "model_best.pth")
        latest_path = os.path.join(load_path, "latest_model.pth")
        if os.path.exists(best_path):
            return best_path
        if os.path.exists(latest_path):
            return latest_path
        raise FileNotFoundError(f"No checkpoint found in directory: {load_path}")
    if not os.path.exists(load_path):
        # Convenience: allow bare checkpoint filename and search in saved_models.
        if os.path.sep not in load_path and '/' not in load_path:
            matches = glob.glob(os.path.join('saved_models', '*', load_path))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Multiple checkpoints matched '{load_path}': {matches}. "
                    f"Please pass a full path."
                )
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    return load_path


def get_pic_class_names(data_dir):
    from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal_class_names
    return get_raw_signal_class_names(data_dir)


def strip_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def adapt_state_dict_for_backbone(state_dict):
    """
    adapt checkpoint keys to backbone-only model keys
    """
    if not state_dict:
        return state_dict

    has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict.keys())
    if not has_backbone_prefix:
        return state_dict

    adapted = {}
    prefix = "backbone."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            adapted[key[len(prefix):]] = value
    return adapted


def configure_cjk_font(plt):
    """
    Configure matplotlib to display CJK labels.
    """
    # Use common CJK fonts in priority order; matplotlib will pick the first available.
    plt.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC',
        'Noto Sans CJK TC',
        'Noto Sans CJK JP',
        'WenQuanYi Zen Hei',
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
        'DejaVu Sans',
    ]
    # Avoid minus sign rendering issue when using CJK fonts.
    plt.rcParams['axes.unicode_minus'] = False


def build_avg_probability_matrix(probs, labels, num_classes):
    """
    Build mean/std softmax-probability matrices grouped by true class.

    mean_matrix[k, j] is the average probability assigned to class j among
    samples whose true class is k. This is not a hard-label confusion matrix.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    mean_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    std_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for class_idx in range(num_classes):
        mask = labels == class_idx
        class_counts[class_idx] = int(mask.sum())
        if class_counts[class_idx] == 0:
            continue

        class_probs = probs[mask]
        mean_matrix[class_idx] = class_probs.mean(axis=0)
        std_matrix[class_idx] = class_probs.std(axis=0)

    return mean_matrix, std_matrix, class_counts


def save_avg_probability_matrix(mean_matrix, std_matrix, class_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    mean_path = os.path.join(save_dir, "avg_prob_mean_matrix.csv")
    std_path = os.path.join(save_dir, "avg_prob_std_matrix.csv")
    mean_std_path = os.path.join(save_dir, "avg_prob_mean_std_matrix.csv")

    with open(mean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, mean_matrix):
            writer.writerow([name] + [f"{value * 100.0:.6f}" for value in row])

    with open(std_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, std_matrix):
            writer.writerow([name] + [f"{value * 100.0:.6f}" for value in row])

    with open(mean_std_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, mean_row, std_row in zip(class_names, mean_matrix, std_matrix):
            writer.writerow(
                [name] + [
                    f"{mean_value * 100.0:.2f}±{std_value * 100.0:.2f}"
                    for mean_value, std_value in zip(mean_row, std_row)
                ]
            )

    return mean_path, std_path, mean_std_path


def plot_avg_probability_matrix(mean_matrix, std_matrix, class_names, save_path):
    from publication_plotting import plot_membership_matrix
    return plot_membership_matrix(mean_matrix, std_matrix, class_names, save_path)


import math
from semilearn.datasets.utils import make_imbalance_data
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

def get_eval_logits(outputs):
    if not isinstance(outputs, dict):
        return outputs

    if all(key in outputs for key in ("w1", "w3", "logits", "aux_logits2")):
        return outputs["w1"].reshape(-1, 1) * outputs["logits"] + outputs["w3"].reshape(-1, 1) * outputs["aux_logits2"]

    logits = outputs.get("logits", None)
    if logits is not None:
        return logits
    return next(iter(outputs.values()))


def evaluate_and_plot(net, eval_loader, save_root, args, class_names, device, use_cuda, suffix_title):
    y_true = []
    y_pred = []
    all_probs = []

    with torch.no_grad():
        for data in eval_loader:
            x = data["x_lb"]
            y = data["y_lb"]

            if isinstance(x, dict):
                x = {k: v.to(device, non_blocking=use_cuda) for k, v in x.items()}
            else:
                x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            logits = get_eval_logits(net(x))

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    all_probs = np.asarray(all_probs, dtype=np.float64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    cf_mat = __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(y_true, y_pred, normalize="true")
    cf_mat = np.nan_to_num(cf_mat, nan=0.0) 

    os.makedirs(save_root, exist_ok=True)

    csv_path = os.path.join(save_root, f"model_best_{suffix_title}.csv")
    png_path = os.path.join(save_root, f"model_best_{suffix_title}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, cf_mat):
            writer.writerow([name] + [f"{value:.6f}" for value in row])

    mean_matrix, std_matrix, class_counts = build_avg_probability_matrix(
        all_probs, y_true, args.num_classes
    )
    mean_path, std_path, mean_std_path = save_avg_probability_matrix(
        mean_matrix, std_matrix, class_names, save_root
    )
    # rename for suffix
    mean_path_new = os.path.join(save_root, f"avg_prob_mean_matrix_{suffix_title}.csv")
    std_path_new = os.path.join(save_root, f"avg_prob_std_matrix_{suffix_title}.csv")
    mean_std_path_new = os.path.join(save_root, f"avg_prob_mean_std_matrix_{suffix_title}.csv")
    os.replace(mean_path, mean_path_new)
    os.replace(std_path, std_path_new)
    os.replace(mean_std_path, mean_std_path_new)
    
    avg_prob_png_path = os.path.join(save_root, f"avg_prob_mean_std_matrix_{suffix_title}.png")
    plot_avg_probability_matrix(mean_matrix, std_matrix, class_names, avg_prob_png_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_cjk_font(plt)

    fig_size = max(6, args.num_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cf_mat, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(args.num_classes),
        yticks=np.arange(args.num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({suffix_title}) - Eval Accuracy: {acc:.2%}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cf_mat.max() / 2.0 if cf_mat.size > 0 else 0.0
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            ax.text(
                j,
                i,
                f"{cf_mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cf_mat[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[{suffix_title}] Test Accuracy: {acc:.4f}")
    print(f"[{suffix_title}] Confusion matrix image saved to: {png_path}")
    print(f"[{suffix_title}] Average probability mean/std image saved to: {avg_prob_png_path}")

def main():
    args = build_args()
    
    # 填充默认参数
    from train import get_config
    default_args = get_config()
    for k, v in vars(default_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)
            
    if not hasattr(args, 'distributed'):
        args.distributed = False
    if not hasattr(args, 'resume'):
        args.resume = False

    use_cuda = args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    if not use_cuda:
        args.gpu = None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = resolve_checkpoint_path(args.load_path)

    # 1. 完整初始化算法，这样 MetaExpert 会读取真实的所有设置
    net_builder = get_net_builder(args.net, args.net_from_name)
    if hasattr(args, 'imb_algorithm') and args.imb_algorithm is not None:
        from semilearn.imb_algorithms import get_imb_algorithm
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        algorithm = get_imb_algorithm(args, net_builder, None, logging.getLogger("eval"))
    else:
        from semilearn.algorithms import get_algorithm
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        algorithm = get_algorithm(args, net_builder, None, logging.getLogger("eval"))
        
    # 我们不仅构建了骨架，接下来加载 checkpoint
    try:
        algorithm.load_model(checkpoint_path)
    except Exception as e:
        print(f"[Warning] using safe load_state_dict. Error: {e}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("ema_model", checkpoint.get("model"))
        algorithm.ema_model.load_state_dict(state_dict)

    # 评测使用 ema_model
    net = algorithm.ema_model.to(device)
    
    # Check if MetaExpert attributes need to be moved to device
    if hasattr(net, 'model'):
        m = net.model
    else:
        m = net
    for attr in ['hat', 'tau1', 'tau2', 'tau3']:
        if hasattr(m, attr):
            t = getattr(m, attr)
            if isinstance(t, torch.Tensor):
                setattr(m, attr, t.to(device))

    net.eval()

    # Use the held-out test split created during algorithm initialization.
    test_dset = None
    if hasattr(algorithm, "dataset_dict") and algorithm.dataset_dict is not None:
        test_dset = algorithm.dataset_dict.get("test")
    if test_dset is None:
        _, _, _, test_dset, _ = get_raw_signal(
            args,
            name=args.dataset,
            data_dir=args.data_dir,
            include_lb_to_ulb=args.include_lb_to_ulb,
            return_test=True,
        )

    data = test_dset.data
    targets = np.array(test_dset.targets)
    
    # 构建平衡数据集
    # 找出最小分类的数量
    class_indices = {c: np.where(targets == c)[0] for c in range(args.num_classes)}
    # Avoid classes with 0 samples
    min_count = min(len(indices) for c, indices in class_indices.items() if len(indices) > 0)
    
    print(f"Min elements found in any class in the test set to build balanced set: {min_count}")
    
    balanced_indices = []
    for c in range(args.num_classes):
        if len(class_indices[c]) > 0:
            c_idx = np.random.choice(class_indices[c], min_count, replace=False)
            balanced_indices.extend(c_idx)
        
    balanced_indices = np.array(balanced_indices)
    balanced_data = [data[i] for i in balanced_indices] if isinstance(data, list) else data[balanced_indices]
    balanced_targets = targets[balanced_indices].tolist()
    
    balanced_dset = BasicDataset(balanced_data, balanced_targets, None, args.num_classes, False, weak_transform=test_dset.weak_transform, strong_transform=None, onehot=False)

    balanced_loader = DataLoader(
        balanced_dset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # 构建长尾数据集 (使用 lb_imb_ratio 缩放)
    num_classes = args.num_classes
    target_counts = make_imbalance_data(100000, num_classes, args.lb_imb_ratio, "exp", num_steps=0) # 这里的 100000 只是为了算出比例
    proportions = np.array(target_counts) / target_counts[0]
    
    # 为了最大化利用测试集，我们要找到符合这个比例的最大 scale
    max_scale = float('inf')
    for c in range(num_classes):
        if len(class_indices[c]) == 0 and proportions[c] > 0:
            print(f"[Warning] Class {c} has 0 samples in test set but requires {proportions[c]} proportion. Ignoring proportion limits for this class.")
            continue
        if proportions[c] > 0:
            scale_c = len(class_indices[c]) / proportions[c]
            if scale_c < max_scale:
                max_scale = scale_c
                
    lt_indices = []
    for c in range(num_classes):
        if proportions[c] > 0 and max_scale != float('inf'):
            c_count = int(max_scale * proportions[c])
            if c_count > 0 and len(class_indices[c]) >= c_count:
                c_idx = np.random.choice(class_indices[c], c_count, replace=False)
                lt_indices.extend(c_idx)

    lt_indices = np.array(lt_indices)
    lt_data = [data[i] for i in lt_indices] if isinstance(data, list) else data[lt_indices]
    lt_targets = targets[lt_indices].tolist()
    print(f"Total imbalanced tests target length: {len(lt_targets)}")
    
    lt_dset = BasicDataset(lt_data, lt_targets, None, args.num_classes, False, weak_transform=test_dset.weak_transform, strong_transform=None, onehot=False)
    
    lt_loader = DataLoader(
        lt_dset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    save_root = args.output_dir
    if not save_root:
        if os.path.isdir(checkpoint_path):
            save_root = checkpoint_path
        else:
            save_root = os.path.dirname(checkpoint_path)
    save_root = os.path.join(save_root, "confusion_matrix")
    
    class_names = get_pic_class_names(args.data_dir)
    if len(class_names) != args.num_classes:
        class_names = [str(i) for i in range(args.num_classes)]
    else:
        class_names = [str(x) for x in class_names]

    # Evaluate Balanced
    print("Evaluating balanced testing set:")
    evaluate_and_plot(net, balanced_loader, save_root, args, class_names, device, use_cuda, "balanced")
    
    # Evaluate Long Tail
    print("Evaluating imbalanced testing set (matching lb_imb_ratio):")
    evaluate_and_plot(net, lt_loader, save_root, args, class_names, device, use_cuda, "imbalanced")
    
if __name__ == '__main__':
    main()
