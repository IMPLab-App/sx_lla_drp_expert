"""Head/Tail half-violin plots from this project's actual Full-model results.

Each star is one configuration's class-mean weight (six configurations by
 default). The right half-violins describe these class means across cases.
No example weights are embedded in the source.
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde

from publication_plotting import FAULTS, fault_order, paper_style, save_figure

ROOT = Path(__file__).resolve().parent
W1_COLOR = '#0C72B8'
W2_COLOR = '#8A2BE2'


def smooth_curve(x, y, samples=400):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dense = np.linspace(x.min(), x.max(), samples)
    return dense, np.clip(PchipInterpolator(x, y)(dense), 0, 1)


def kde_values(values, grid, bw_method=0.35):
    values = np.asarray(values, dtype=float)
    if len(values) > 1 and np.ptp(values) > 1e-8:
        return gaussian_kde(values, bw_method=bw_method)(grid)
    # A singleton/constant class still has a defined display profile.
    bandwidth = 0.008
    return np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2).sum(1)


def violin_profile(values, half_width):
    values = np.asarray(values, dtype=float)
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    spread = max(np.ptp(values), std * 2.5, 0.018)
    grid = np.linspace(max(0, values.min() - spread * .55),
                       min(1, values.max() + spread * .55), 260)
    density = kde_values(values, grid)
    return grid, density / max(density.max(), 1e-12) * half_width


def half_violin(ax, data, positions, color, edgecolor, width=.58, alpha=.70):
    for i, center in enumerate(positions):
        grid, density = violin_profile(data[:, i], width / 2)
        ax.fill_betweenx(grid, center, center + density, color=color, alpha=alpha,
                         edgecolor='none', zorder=3)
        ax.plot(np.full_like(grid, center), grid, color=edgecolor, lw=1, zorder=4)
        ax.plot(center + density, grid, color=edgecolor, lw=1, zorder=4)


def load_case_weights(paths, weight_source='dea'):
    rows, rows_tail, records = [], [], []
    expected_names = None
    key = 'dea_weights' if weight_source == 'dea' else 'final_weights'
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            names = data['class_names'].tolist()
            if expected_names is None:
                expected_names = names
            if names != expected_names:
                raise ValueError(f'Class name/order mismatch: {path}')
            order, _ = fault_order(names)
            if len(order) != 9:
                raise ValueError('The requested figure expects the nine fault classes')
            weights, labels = data[key], data['labels']
            if weights.shape != (len(labels), 2) or not np.isfinite(weights).all():
                raise ValueError(f'Invalid two-expert weights in {path}')
            if weights.min() < 0 or weights.max() > 1 or not np.allclose(weights.sum(1), 1, atol=1e-6):
                raise ValueError(f'Expert weights must be probabilities summing to 1: {path}')
            means = []
            for old_class in order:
                selected = weights[labels == old_class]
                if not len(selected):
                    raise ValueError(f'Missing true class {old_class} in {path}')
                means.append(selected.mean(0, dtype=np.float64))
            means = np.asarray(means)
            rows.append(means[:, 0]); rows_tail.append(means[:, 1])
            records.append(dict(source=str(Path(path).resolve()), run=int(data['run'])))
    if not rows:
        raise ValueError('No prediction files supplied')
    return np.asarray(rows), np.asarray(rows_tail), expected_names, records


def confidence_figure(w1, w2):
    if w1.shape != w2.shape or w1.ndim != 2 or w1.shape[1] != 9:
        raise ValueError('Expected matching [configurations, 9 classes] weights')
    positions = np.arange(1, 10)
    with plt.rc_context(paper_style()):
        fig, ax = plt.subplots(figsize=(10.5, 7.2))
        half_violin(ax, w1, positions, '#9AD0F5', '#4E9FD8')
        half_violin(ax, w2, positions, '#D7B5F5', '#9B6AD6')
        rng = np.random.default_rng(2)
        for values, color in ((w1, W1_COLOR), (w2, '#B94BD3')):
            for i, x in enumerate(positions):
                ax.scatter(x - rng.uniform(.045, .105, len(values)), values[:, i],
                           marker='*', s=85, color=color, linewidths=.5, zorder=5)
        effects = [pe.Stroke(linewidth=4.3, foreground='white', alpha=.86), pe.Normal()]
        for values, color in ((w1, W1_COLOR), (w2, W2_COLOR)):
            means = values.mean(0)
            dense_x, dense_y = smooth_curve(positions, means)
            ax.plot(dense_x, dense_y, color=color, lw=2.6,
                    path_effects=effects, solid_capstyle='round', zorder=6)
            ax.scatter(positions, means, s=45, color=color, edgecolors='white', lw=.9, zorder=7)
        # The Head/Tail boundary is semantic (healthy versus faults), not a
        # crossing inferred from whichever values this experiment produced.
        ax.axvline(1.5, color='#666666', ls=(0, (4, 4)), lw=1.2, zorder=2)
        ax.set(xlim=(.5, 9.72), ylim=(0, 1.065))
        ax.text(1.0, 1.035, 'Head', ha='center', va='center', fontsize=17)
        ax.text(3.55, 1.035, 'Tail', ha='center', va='center', fontsize=17)
        ax.set_xticks(positions, labels=[str(i) for i in positions], fontsize=19)
        ticks = np.arange(0, 1.01, .1)
        ax.set_yticks(ticks, labels=['0' if value == 0 else f'{value:.1f}' for value in ticks], fontsize=18)
        ax.set_xlabel('class label', fontsize=23, labelpad=10)
        ax.set_ylabel('Confidence', fontsize=23, labelpad=9)
        ax.grid(axis='y', color='#B8B8B8', ls=(0, (3, 3)), lw=.8)
        ax.set_axisbelow(True)
        # Reference annotations request a closed frame and inward ticks.
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.2)
        ax.tick_params(axis='both', direction='in', length=6, width=1.2, pad=6)
        handles = [Line2D([0], [0], color=color, lw=3.2, label=label)
                   for color, label in ((W1_COLOR, r'$\alpha_1$'), (W2_COLOR, r'$\alpha_2$'))]
        legend = ax.legend(handles=handles, loc='center right', bbox_to_anchor=(.91, .64),
                           frameon=True, fancybox=False, framealpha=1, facecolor='white',
                           edgecolor='black', fontsize=19, handlelength=1.4)
        legend.get_frame().set_linewidth(1)
        fig.tight_layout(pad=.8)
    return fig, ax


def render(paths, output_base, weight_source='dea', dpi=600):
    w1, w2, names, sources = load_case_weights(paths, weight_source)
    fig, _ = confidence_figure(w1, w2)
    outputs = save_figure(fig, output_base, dpi=dpi)
    plt.close(fig)
    base = Path(output_base)
    with base.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['run', 'class_label', 'fault_code', 'fault_name', 'alpha_1_head', 'alpha_2_tail'])
        for row, source in enumerate(sources):
            for col, (code, name, _) in enumerate(FAULTS):
                writer.writerow([source['run'], col + 1, code, name, w1[row, col], w2[row, col]])
    base.with_suffix('.json').write_text(json.dumps(dict(
        weight_source=weight_source, source_field='softmax(fuse_w_logit)' if weight_source == 'dea' else 'w1/w3',
        star_meaning='One class-mean weight per configuration; six cases are not independent repeats',
        number_of_configurations=len(paths), sources=sources,
        original_class_names=names, class_order=[code for code, _, _ in FAULTS],
        font=paper_style()['font.family']), ensure_ascii=False, indent=2) + '\n')
    return outputs


def render_given_w1(input_path, output_base, dpi=600):
    """Render supplied numeric rows in exactly their given column order."""
    input_path = Path(input_path)
    if input_path.suffix == '.json':
        supplied = json.loads(input_path.read_text())
        w1 = np.asarray(supplied['w1'], dtype=np.float64)
    else:
        w1 = np.asarray(np.load(input_path, allow_pickle=False), dtype=np.float64)
    if w1.ndim != 2 or w1.shape[1] != 9 or len(w1) == 0:
        raise ValueError('Supplied w1 must have shape [rows, 9]')
    if not np.isfinite(w1).all() or w1.min() < 0 or w1.max() > 1:
        raise ValueError('Supplied w1 values must be finite probabilities in [0, 1]')
    w2 = 1.0 - w1
    fig, _ = confidence_figure(w1, w2)
    outputs = save_figure(fig, output_base, dpi=dpi)
    plt.close(fig)
    base = Path(output_base)
    with base.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['supplied_row', 'class_label', 'w1', 'w2'])
        for row in range(len(w1)):
            for col in range(9):
                writer.writerow([row + 1, col + 1, w1[row, col], w2[row, col]])
    base.with_suffix('.json').write_text(json.dumps(dict(
        data_source='User-provided w1 values', input_file=str(input_path.resolve()),
        w2_definition='1 - w1', input_shape=list(w1.shape),
        column_order='Original supplied columns 1-9, no permutation or inferred fault-name mapping',
        stars='Each star is one supplied row value for that column',
        curves='Column means across the supplied rows',
        font=paper_style()['font.family']), ensure_ascii=False, indent=2) + '\n')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_glob', default=str(ROOT / 'results/publication_figures/full/data/full_run_*_plot_data.npz'))
    parser.add_argument('--output_base', type=Path, default=ROOT / 'results/publication_figures/full/half_violin_w1_w2')
    parser.add_argument('--weight_source', choices=('dea', 'final'), default='dea')
    parser.add_argument('--w1_file', type=Path, help='JSON containing w1, or a numeric .npy array; w2=1-w1')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    if args.w1_file is not None:
        output_base = args.output_base
        if output_base == parser.get_default('output_base'):
            output_base = output_base.with_name('half_violin_given_w1')
        for path in render_given_w1(args.w1_file, output_base, args.dpi):
            print(path)
        return
    paths = sorted(glob.glob(args.input_glob))
    if not paths and args.input_glob == parser.get_default('input_glob'):
        from export_publication_plot_data import export_full_prediction_data
        paths = export_full_prediction_data(ROOT / 'results/publication_figures/full/data')
    if not paths:
        parser.error(f'No prediction files matched {args.input_glob}')
    for path in render(paths, args.output_base, args.weight_source, args.dpi):
        print(path)


if __name__ == '__main__':
    main()
