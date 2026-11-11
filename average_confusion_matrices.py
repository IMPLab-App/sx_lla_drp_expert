#!/usr/bin/env python3
import argparse
import csv
import glob
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Average confusion matrix CSV files and export CSV/PNG.")
    p.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob for input CSV files, e.g. 'saved_models/xxx_seed*/confusion_matrix/model_best.csv'",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save averaged outputs",
    )
    p.add_argument(
        "--output_name",
        type=str,
        default="avg_confusion_matrix",
        help="Base file name for output CSV/PNG",
    )
    return p.parse_args()


def read_confusion_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Invalid confusion matrix csv: {path}")

    class_names = rows[0][1:]
    matrix_rows = []
    row_names = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        row_names.append(row[0])
        matrix_rows.append([float(x) for x in row[1:]])

    matrix = np.array(matrix_rows, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square in {path}: shape={matrix.shape}")

    if len(class_names) != matrix.shape[0]:
        raise ValueError(
            f"Header class count mismatch in {path}: "
            f"header={len(class_names)} matrix={matrix.shape}"
        )

    if row_names != class_names:
        raise ValueError(f"Row class names differ from header in {path}")

    return class_names, matrix


def configure_cjk_font(plt):
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_csv(path, class_names, matrix):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + [f"{x:.6f}" for x in row])


def save_png(path, class_names, matrix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_cjk_font(plt)

    n = len(class_names)
    mean_diag_acc = float(np.mean(np.diag(matrix))) if matrix.size > 0 else 0.0
    fig_size = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Average Confusion Matrix - Mean Diag Acc: {mean_diag_acc:.2%}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    class_names_ref = None
    mats = []

    for p in paths:
        class_names, mat = read_confusion_csv(p)
        if class_names_ref is None:
            class_names_ref = class_names
        elif class_names != class_names_ref:
            raise ValueError(f"Class names mismatch: {p}")
        mats.append(mat)

    avg_mat = np.mean(np.stack(mats, axis=0), axis=0)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.output_name}.csv")
    png_path = os.path.join(args.output_dir, f"{args.output_name}.png")

    save_csv(csv_path, class_names_ref, avg_mat)
    save_png(png_path, class_names_ref, avg_mat)

    print(f"Input files: {len(paths)}")
    print(f"Average CSV saved to: {csv_path}")
    print(f"Average PNG saved to: {png_path}")


if __name__ == "__main__":
    main()
