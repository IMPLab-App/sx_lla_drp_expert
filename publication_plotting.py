"""Shared publication styles and the fault order specified in the reference table."""
from pathlib import Path

import numpy as np


MEMBERSHIP_CMAP_NAME = "reference_purple"
MEMBERSHIP_BASE_CMAP = "custom_light_to_deep_purple"
MEMBERSHIP_WHITE_MIX = 0.0


def membership_colormap():
    """Reference palette: near-white lavender at zero, deep purple at one."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        MEMBERSHIP_CMAP_NAME,
        ["#f4f1fa", "#d9d0ec", "#ada0d2", "#7e6ab4", "#514391"],
        N=256,
    )


# H1/C1...C8 describe faults, not the original NPZ integer order.
FAULTS = (
    ("H1", "健康", ("健康", "正常")),
    ("C1", "主轴疲劳", ("主轴疲劳",)),
    ("C2", "缸体轴外圈疲劳", ("缸体轴外圈疲劳",)),
    ("C3", "缸体轴内圈疲劳", ("缸体轴内圈疲劳",)),
    ("C4", "柱塞磨损", ("柱塞磨损",)),
    ("C5", "斜盘磨损", ("斜盘磨损",)),
    ("C6", "配油盘磨损", ("配油盘磨损",)),
    ("C7", "弹簧失效", ("弹簧失效",)),
    ("C8", "滑靴塌边", ("滑靴塌边",)),
)


def paper_style():
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral")
                 if name in available), "DejaVu Serif")
    return {"font.family": font, "mathtext.fontset": "stix",
            "axes.unicode_minus": False, "lines.antialiased": True,
            "pdf.fonttype": 42, "ps.fonttype": 42}


def fault_order(class_names):
    """Return both-axis permutation and labels without relabeling unrelated data."""
    names = [str(name).strip() for name in class_names]
    if len(names) == 9 and all(any(alias in names for alias in aliases)
                               for _, _, aliases in FAULTS):
        order = [next(names.index(alias) for alias in aliases if alias in names)
                 for _, _, aliases in FAULTS]
        labels = [r"$H_1$"] + [rf"$C_{i}$" for i in range(1, 9)]
        return np.asarray(order), labels
    symbolic = ["H1"] + [f"C{i}" for i in range(1, 9)]
    if names == symbolic:
        return np.arange(9), [r"$H_1$"] + [rf"$C_{i}$" for i in range(1, 9)]
    # Generic evaluation datasets keep their own semantic labels.
    return np.arange(len(names)), names


def save_figure(fig, output_base, formats=("png", "tif", "pdf"), dpi=600):
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = base.with_suffix("." + ext)
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext in ("tif", "tiff") else {}
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", **options)
        paths.append(path)
    return paths


def membership_figure(mean_matrix, std_matrix, class_names):
    """Square probability cells annotated with means only.

    Inputs are probabilities in [0, 1]. The caller supplies the SD and records
    its scope: within-class samples for per-task plots, or across task means
    for the six-task aggregate.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    means = np.asarray(mean_matrix, dtype=float)
    stds = np.asarray(std_matrix, dtype=float)
    n = len(class_names)
    if means.shape != (n, n) or stds.shape != means.shape:
        raise ValueError("Probability matrices and class names must have matching dimensions")
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Probability matrices contain non-finite values")
    if means.min() < -1e-8 or means.max() > 1 + 1e-8 or stds.min() < 0:
        raise ValueError("Expected mean probabilities in [0, 1] and nonnegative SD")
    order, labels = fault_order(class_names)
    means = means[np.ix_(order, order)] * 100
    stds = stds[np.ix_(order, order)] * 100
    style = paper_style()
    style["mathtext.default"] = "regular"
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10.1, 10.1))
        # A fixed 0..100 scale keeps all six configurations comparable.
        cmap = membership_colormap()
        image = ax.imshow(means, cmap=cmap, vmin=0, vmax=100, interpolation="nearest", aspect="equal")
        # Attach an equally tall colorbar without changing the square matrix.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        colorbar_ax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.18)
        colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=np.arange(0, 101, 20))
        colorbar.ax.tick_params(labelsize=16)
        colorbar.outline.set_visible(False)
        ax.set_box_aspect(1)
        ax.set_xticks(np.arange(n), labels=labels, fontsize=18)
        ax.set_yticks(np.arange(n), labels=labels, fontsize=18)
        ax.set_xlabel(r"Fault status membership ($\times 10^2$)", fontsize=22, labelpad=9)
        ax.set_ylabel("Fault scenario", fontsize=22, labelpad=10)
        ax.tick_params(which="major", length=0, pad=6)
        for i in range(n):
            for j in range(n):
                text = f"{means[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=18, fontstyle="normal",
                        fontweight="bold" if i == j else "normal",
                        color="white" if means[i, j] >= 65 else "#242424")
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.05)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.7)
    return fig, ax


def plot_membership_matrix(mean_matrix, std_matrix, class_names, save_path, dpi=300):
    from matplotlib import pyplot as plt
    fig, _ = membership_figure(mean_matrix, std_matrix, class_names)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
