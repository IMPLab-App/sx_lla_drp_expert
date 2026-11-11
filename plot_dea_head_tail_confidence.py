import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_confusion_matrix_new import (
    configure_cjk_font,
    get_pic_class_names,
    resolve_checkpoint_path,
)
from semilearn.core.utils import get_net_builder
from semilearn.core.utils.misc import over_write_args_from_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DEA Head/Tail expert assignment confidence by true class."
    )
    parser.add_argument("--c", required=True, help="Training YAML config path.")
    parser.add_argument("--load_path", required=True, help="Checkpoint path or checkpoint directory.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id. This MetaExpert code path requires CUDA.")
    parser.add_argument("--split", choices=["test", "eval"], default="test")
    parser.add_argument("--output_dir", default="", help="Directory for PNG/CSV outputs.")
    parser.add_argument("--output_name", default="dea_head_tail_confidence")
    parser.add_argument(
        "--weight_source",
        choices=["dea", "final"],
        default="dea",
        help="dea uses softmax(fuse_w_logit); final uses returned w1/w3 after post-processing.",
    )
    parser.add_argument(
        "--head_classes",
        default="0",
        help="Comma-separated zero-based Head class ids. Default: 0, i.e. normal/healthy.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def get_default_train_args(config_path):
    from train import get_config

    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--c", config_path]
    try:
        return get_config()
    finally:
        sys.argv = old_argv


def build_runtime_args(cli_args):
    args = get_default_train_args(cli_args.c)
    over_write_args_from_file(args, cli_args.c)

    args.c = cli_args.c
    args.load_path = cli_args.load_path
    args.gpu = cli_args.gpu
    args.distributed = False
    args.resume = False

    if cli_args.eval_batch_size is not None:
        args.eval_batch_size = cli_args.eval_batch_size
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers

    head_classes = tuple(
        int(item.strip()) for item in cli_args.head_classes.split(",") if item.strip() != ""
    )
    if head_classes == (0,):
        args.cut1 = 1
        args.cut2 = 1
    else:
        raise ValueError("This two-expert MetaExpert implementation supports contiguous Head classes from class 0.")

    return args, head_classes


def build_algorithm(args, checkpoint_path):
    if not torch.cuda.is_available():
        raise RuntimeError("Current MetaExpert initialization calls .cuda(...); please run on a CUDA machine.")

    torch.cuda.set_device(args.gpu)
    net_builder = get_net_builder(args.net, args.net_from_name)
    logging.getLogger().setLevel(logging.ERROR)

    if getattr(args, "imb_algorithm", None) is not None:
        from semilearn.imb_algorithms import get_imb_algorithm

        algorithm = get_imb_algorithm(args, net_builder, None, logging.getLogger("plot_dea"))
    else:
        from semilearn.algorithms import get_algorithm

        algorithm = get_algorithm(args, net_builder, None, logging.getLogger("plot_dea"))

    try:
        algorithm.load_model(checkpoint_path)
    except Exception as exc:
        print(f"[Warning] algorithm.load_model failed, loading EMA weights only: {exc}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("ema_model", checkpoint.get("model"))
        algorithm.ema_model.load_state_dict(state_dict)

    net = algorithm.ema_model.cuda(args.gpu)
    for attr in ["hat", "tau1", "tau2", "tau3"]:
        if hasattr(net, attr):
            value = getattr(net, attr)
            if isinstance(value, torch.Tensor):
                setattr(net, attr, value.cuda(args.gpu))
    net.eval()
    return algorithm, net


def get_split_dataset(algorithm, split):
    dataset_dict = getattr(algorithm, "dataset_dict", None) or {}
    dataset = dataset_dict.get(split)
    if dataset is None:
        raise RuntimeError(f"Dataset split '{split}' is not available in algorithm.dataset_dict.")
    return dataset


def collect_dea_confidence(net, loader, num_classes, gpu, weight_source):
    sums = np.zeros((num_classes, 2), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            x = batch["x_lb"].cuda(gpu, non_blocking=True)
            y = batch["y_lb"]
            outputs = net(x)

            if weight_source == "dea":
                if "fuse_w_logit" not in outputs:
                    raise KeyError("Model output does not contain 'fuse_w_logit'.")
                weights = torch.softmax(outputs["fuse_w_logit"], dim=1)
            else:
                if "w1" not in outputs or "w3" not in outputs:
                    raise KeyError("Model output does not contain 'w1' and 'w3'.")
                weights = torch.cat(
                    [outputs["w1"].reshape(-1, 1), outputs["w3"].reshape(-1, 1)], dim=1
                )

            weights_np = weights.detach().cpu().numpy()
            labels_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            for label, pair in zip(labels_np, weights_np):
                label = int(label)
                counts[label] += 1
                sums[label] += pair[:2]

    means = np.divide(
        sums,
        np.maximum(counts, 1)[:, None],
        out=np.full_like(sums, np.nan),
        where=counts[:, None] > 0,
    )
    return means, counts


def write_csv(path, means, counts, class_names, head_classes):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "group", "num_samples", "w1_head", "w3_tail"])
        for idx, (name, mean, count) in enumerate(zip(class_names, means, counts)):
            group = "Head" if idx in head_classes else "Tail"
            writer.writerow([idx + 1, name, group, int(count), mean[0], mean[1]])


def plot_confidence(path, means, class_names, head_classes, dpi):
    import matplotlib.pyplot as plt

    configure_cjk_font(plt)
    x = np.arange(1, len(class_names) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, means[:, 0], color="#d84a3a", linewidth=2.0, label=r"$w_1$ Head")
    ax.plot(x, means[:, 1], color="#4c91b8", linewidth=2.0, label=r"$w_3$ Tail")

    last_head = max(head_classes) + 1
    ax.axvline(last_head + 0.5, color="0.72", linestyle="--", linewidth=1.2)

    finite = means[np.isfinite(means)]
    y_min = max(0.0, np.floor((float(finite.min()) - 0.05) * 10.0) / 10.0)
    y_max = min(1.0, np.ceil((float(finite.max()) + 0.05) * 10.0) / 10.0)
    if y_max - y_min < 0.2:
        center = (y_min + y_max) / 2.0
        y_min = max(0.0, center - 0.1)
        y_max = min(1.0, center + 0.1)
    ax.set_ylim(y_min, y_max)

    label_y = y_max - 0.04 * (y_max - y_min)
    ax.text((min(head_classes) + 1 + last_head) / 2.0, label_y, "Head", ha="center", va="top")
    ax.text((last_head + 1.5 + len(class_names)) / 2.0, label_y, "Tail", ha="center", va="top")

    ax.set_xlim(0.6, len(class_names) + 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("Class index")
    ax.set_ylabel("Confidence")
    ax.legend(loc="best", frameon=True)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    cli_args = parse_args()
    args, head_classes = build_runtime_args(cli_args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = resolve_checkpoint_path(cli_args.load_path)
    algorithm, net = build_algorithm(args, checkpoint_path)
    dataset = get_split_dataset(algorithm, cli_args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_names = get_pic_class_names(args.data_dir)
    if len(class_names) != args.num_classes:
        class_names = [f"class_{idx}" for idx in range(args.num_classes)]

    means, counts = collect_dea_confidence(
        net, loader, args.num_classes, args.gpu, cli_args.weight_source
    )

    output_dir = cli_args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(checkpoint_path), "dea_head_tail_confidence")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{cli_args.output_name}.csv")
    png_path = os.path.join(output_dir, f"{cli_args.output_name}.png")
    write_csv(csv_path, means, counts, class_names, set(head_classes))
    plot_confidence(png_path, means, class_names, set(head_classes), cli_args.dpi)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {png_path}")
    print("Class groups: Head = class 1/正常; Tail = classes 2-9/其他故障状态")


if __name__ == "__main__":
    main()
