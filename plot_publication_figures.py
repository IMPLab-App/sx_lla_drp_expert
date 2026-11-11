"""Generate confidence plots and one equal-weight six-task membership matrix."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from export_publication_plot_data import ROOT, export_full_prediction_data
from plot_half_violin_w1_w2_tif import render, render_given_w1
from publication_plotting import (FAULTS, MEMBERSHIP_CMAP_NAME, MEMBERSHIP_BASE_CMAP,
                                  MEMBERSHIP_WHITE_MIX,
                                  fault_order, membership_figure, save_figure)


def write_figure_guide(output):
    """Keep the explanation alongside the figures and preserve it on regeneration."""
    (output / 'README.md').write_text('''# publication_figures 图片说明

主目录下保留 4 张不同内容的图，每张各保存 PNG、TIFF、PDF 三种格式，共 12 个图像文件：两张模型专家权重图、一张用户给定数值的图，以及一张六任务平均隶属度总图。
模型数据来自本项目 Full 模型的六组配置，统一 seed=2；每组使用 900 个真实留出测试样本，各类别 100 个。`half_violin_given_w1` 使用用户提供的数值。
原六张任务分图及其 CSV 已归档到 `per_task/`，主目录的隶属度结果以 `fault_membership_mean` 为准。
这批图不是八种对比方法的结果，也不是后续 seed=3/4/5 的重复训练汇总。

## 1. half_violin_w1_w2：原始 DEA 专家路由权重

- 横坐标 1–9 对应下表 H1、C1…C8；纵坐标是专家权重，范围 0–1。
- 蓝色 α1 是 Head 专家路由权重；紫色 α2 是 Tail 专家路由权重，Tail 在模型内部编号为 E3。
- 权重来自 `softmax(fuse_w_logit)`。它表示“更倾向使用哪位专家”，不是某种故障的分类概率，也不是分类准确率。
- 每个星号表示一组配置中，该真实类别所有测试样本的平均权重；每类共六个星号。
- 右侧半小提琴表示六组配置的类平均权重分布。轮廓更宽的位置表示这些均值更集中；它不表示样本数量更多。
- 圆点和曲线表示六组配置的平均值；平滑线只是连接离散类别的视觉辅助，没有连续类别的物理含义。
- Head/Tail 虚线分隔健康类别与八种故障类别。六组配置是不同实验条件，不是六次独立重复。

[PNG 预览](half_violin_w1_w2.png) · [600 dpi TIFF](half_violin_w1_w2.tif) · [矢量 PDF](half_violin_w1_w2.pdf)

## 2. half_violin_final_blend_weights：最终参与融合的专家权重

读图方式与上一张相同，但统计的是模型经过分类置信度后处理后输出的 `w1/w3`。
这些权重实际用于融合 Head/Tail 的分类 logits。原始路由权重与最终融合权重属于不同计算阶段，数值和分布不必相同。
当前模型的后处理会压缩权重范围，因此原始路由图中接近 0/1 的值，在最终融合图中可能更接近中间范围；图中数值均来自模型实际输出。

[PNG 预览](half_violin_final_blend_weights.png) · [600 dpi TIFF](half_violin_final_blend_weights.tif) · [矢量 PDF](half_violin_final_blend_weights.pdf)

## 3. fault_membership_mean：六任务等权平均隶属度总图

先分别计算各任务按真实类别分组的预测概率均值矩阵 M1…M6，再按元素等权平均：M = (M1+…+M6)/6。每个任务权重均为 1/6。
总图的“±”是六个任务均值之间的样本标准差（ddof=1），表示跨任务波动；它不是原分图中的类内样本标准差，也不是六个标准差直接平均。
总图使用 0–100% 色标。配色保留参考图的蓝→浅蓝→浅黄→橙→红序列，在原始 RdYlBu_r 颜色中混入 60% 白色，得到浅色版，文字统一为深灰。
单元格统一为正方形；对角线只标均值，其他格标“均值±标准差”。配色调整不改变任何模型或结果数值。

[总图 PNG](fault_membership_mean.png) · [600 dpi TIFF](fault_membership_mean.tif) · [矢量 PDF](fault_membership_mean.pdf)

参与平均的六个任务如下；表中的历史分图保存在 `per_task/`：

| 图片 | 有标签不平衡比 | 无标签数量参数 | 无标签分布 |
|---|---:|---:|---|
| [fault_membership_run_1](per_task/fault_membership_run_1.png) | 50 | 250 | 同向长尾，比例 50 |
| [fault_membership_run_2](per_task/fault_membership_run_2.png) | 100 | 250 | 同向长尾，比例 100 |
| [fault_membership_run_3](per_task/fault_membership_run_3.png) | 50 | 315 | 均匀，每类 35 个 |
| [fault_membership_run_4](per_task/fault_membership_run_4.png) | 100 | 315 | 均匀，每类 35 个 |
| [fault_membership_run_5](per_task/fault_membership_run_5.png) | 50 | 250 | 反向长尾，比例 -50 |
| [fault_membership_run_6](per_task/fault_membership_run_6.png) | 100 | 250 | 反向长尾，比例 -100 |

有标签头类数量均为 100。不平衡时，无标签参数 250 是头类数量；均匀时，315 是无标签总数。

读图方法：

- 行表示样本的真实故障类别；列表示模型分配预测概率的类别。
- 每格先在各任务内对该真实类别的 softmax 概率求均值，再对六个任务的均值等权平均，并乘 100。
- 对角线是分配给正确类别的平均预测概率，数值高通常表示更有把握；它不能直接当作该类准确率。
- 非对角线表示分配给其他类别的概率，均值较大说明存在相应混淆倾向。
- 总图的标准差统计六组任务均值之间的波动，单位是百分点。六组是不同配置，不能把它们称作六次独立重复训练。
- `per_task/` 历史分图仍使用各任务内、同一真实类别样本的概率标准差（ddof=0），与总图的“±”口径不同。
- 每行的概率均值之和约为 100%；显示值保留两位小数，可能有舍入误差。`100.00` 也是四舍五入显示，不能据此声称每个样本概率精确等于 1。

## 4. half_violin_given_w1：用户给定数值的独立置信度图

此图精确使用用户给定的 6×9 数组 w1，并按 `w2 = 1 - w1` 计算另一条曲线。
蓝色为 w1/α1，紫色为 w2/α2；每个星号代表所给数组的一行在该列的数值，圆点和曲线为六行的列平均值，半小提琴展示六行数值的分布。
本图保留输入列 1–9 的原顺序，不推断这些列的故障名称，也不把这组输入当作本项目重新导出的模型结果。
它沿用第一张图的闭合边框、向内刻度、字体及蓝紫色曲线。

[PNG 预览](half_violin_given_w1.png) · [600 dpi TIFF](half_violin_given_w1.tif) · [矢量 PDF](half_violin_given_w1.pdf)

输入文件：`data/user_provided_w1.json`。可单独重新生成：

`/home/kv/.conda/envs/cw/bin/python plot_half_violin_w1_w2_tif.py --w1_file results/publication_figures/full/data/user_provided_w1.json`

## 5. 模型结果图片的类别映射

| 横坐标/矩阵序号 | 标签 | 故障状态 |
|---:|---|---|
| 1 | H1 | 健康（数据中的“正常”） |
| 2 | C1 | 主轴疲劳 |
| 3 | C2 | 缸体轴外圈疲劳 |
| 4 | C3 | 缸体轴内圈疲劳 |
| 5 | C4 | 柱塞磨损 |
| 6 | C5 | 斜盘磨损 |
| 7 | C6 | 配油盘磨损 |
| 8 | C7 | 弹簧失效 |
| 9 | C8 | 滑靴塌边 |

为与附件故障表一致，矩阵的行列均按原始类别索引 `0,7,5,6,1,3,2,8,4` 重排；两张模型权重图也使用这一顺序。用户提供数值的独立图保留其输入列序。

## 6. 格式和辅助文件

- `.png`：直接查看和插入普通文档。
- `.tif`：600 dpi、LZW 压缩，适合需要位图的论文排版；与同名 PNG 内容相同。
- `.pdf`：矢量图，适合缩放和排版；与同名 PNG 内容相同。
- `half_violin_*.csv`：每组配置、每类的两种专家平均权重；同名 `.json` 记录统计口径和来源。
- `half_violin_given_w1.csv`：用户提供的每行、每列 w1 以及计算得到的 w2。
- `fault_membership_mean.csv`：六任务等权平均矩阵，单位为百分数。
- `fault_membership_mean_std.csv`：六任务之间的样本标准差（ddof=1），单位为百分点。
- `fault_membership_mean.json` / `_statistics.npz`：聚合定义、来源、六个任务矩阵及聚合结果，便于重算。
- `per_task/fault_membership_run_*_mean.csv` / `_std.csv`：原分任务概率均值/类内标准差，保留六位小数。
- `data/full_run_*_plot_data.npz`：逐样本标签、两类专家权重和最终分类概率。
- `data/provenance.json`：实际检查点、模型代码快照、数据及历史预测的来源和哈希。
- `figure_manifest.json`：图片清单、故障映射、色标与标注约定。
- `verification.txt`：生成检查记录；`palette_verification.json`：浅色调整后数据未变的检查记录。
- `aggregate_verification.json`：等权平均、跨任务标准差与正方形单元格核验结果。

字体优先 Times New Roman；系统缺少时使用 Nimbus Roman 等 Times 风格衬线字体。

重新生成全部图片：`/home/kv/.conda/envs/cw/bin/python plot_publication_figures.py`

只绘原始路由权重图：`/home/kv/.conda/envs/cw/bin/python plot_half_violin_w1_w2_tif.py`

只绘图不会训练模型；默认使用已导出的逐样本数据。显式指定 `--refresh_predictions` 才会重新从检查点推理，并核对历史预测。
''')


def probability_moments(probabilities, labels, num_classes):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.shape != (len(labels), num_classes):
        raise ValueError('Probability matrix shape does not match the labels')
    means, stds = [], []
    for class_id in range(num_classes):
        selected = probabilities[labels == class_id]
        if not len(selected):
            raise ValueError(f'No test samples for class {class_id}')
        means.append(selected.mean(0))
        stds.append(selected.std(0, ddof=0))
    return np.asarray(means), np.asarray(stds)


def aggregate_membership(paths, output, dpi=600, per_task=False):
    """Equal-weight mean of six task matrices; SD is across task means."""
    import matplotlib.pyplot as plt
    import shutil

    output = Path(output)
    archive = output / 'per_task'
    archive.mkdir(parents=True, exist_ok=True)
    # Preserve historical per-task outputs while presenting a single total plot.
    for previous in output.glob('fault_membership_run_*'):
        target = archive / previous.name
        if target.exists():
            if target.read_bytes() != previous.read_bytes():
                raise FileExistsError(f'Conflicting historical output: {target}')
            continue
        shutil.move(str(previous), str(target))
    tasks, matrices, names = [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as values:
            run = int(values['run'])
            current_names = values['class_names'].tolist()
            if names is None:
                names = current_names
            if names != current_names:
                raise ValueError(f'Class order differs between tasks: {path}')
            means, within_std = probability_moments(values['probabilities'], values['labels'], len(names))
        matrices.append(means)
        order, _ = fault_order(names)
        codes = [code for code, _, _ in FAULTS]
        base = archive / f'fault_membership_run_{run}'
        for suffix, matrix in (('mean', means), ('std', within_std)):
            destination = base.with_name(base.name + '_' + suffix).with_suffix('.csv')
            if not destination.exists():
                save_matrix_csv(destination, matrix[np.ix_(order, order)], codes)
        if per_task:
            fig, _ = membership_figure(means, within_std, names)
            save_figure(fig, base, dpi=dpi)
            plt.close(fig)
        tasks.append(dict(run=run, source=str(Path(path).resolve()), weight=1/6))
    if len(tasks) != 6 or sorted(task['run'] for task in tasks) != list(range(1, 7)):
        raise ValueError('The aggregate requires exactly one prediction file for each task 1–6')
    stacked = np.stack(matrices)
    mean = stacked.mean(axis=0)
    between_std = stacked.std(axis=0, ddof=1)
    order, _ = fault_order(names)
    ordered_mean = mean[np.ix_(order, order)]
    ordered_std = between_std[np.ix_(order, order)]
    base = output / 'fault_membership_mean'
    fig, ax = membership_figure(mean, between_std, names)
    fig.canvas.draw()
    assert abs(ax.get_window_extent().width - ax.get_window_extent().height) < 1e-5
    images = save_figure(fig, base, dpi=dpi)
    plt.close(fig)
    save_matrix_csv(base.with_suffix('.csv'), ordered_mean, codes)
    save_matrix_csv(output / 'fault_membership_mean_std.csv', ordered_std, codes)
    np.savez_compressed(output / 'fault_membership_mean_statistics.npz',
                        task_mean_matrices=stacked, mean=mean, between_task_std=between_std,
                        original_class_names=np.asarray(names), display_order=order)
    record = dict(tasks=tasks, images=[str(p) for p in images],
                  aggregation='Equal-weight arithmetic mean of six per-task class-mean probability matrices',
                  standard_deviation='Sample SD across six task mean matrices, ddof=1',
                  original_label_permutation=order.tolist(), cells_are_square=True,
                  diagonal_annotation='six-task mean only',
                  off_diagonal_annotation='six-task mean only',
                  color_scale_percent=[0, 100], cmap=MEMBERSHIP_CMAP_NAME,
                  reference_cmap=MEMBERSHIP_BASE_CMAP, white_mix=MEMBERSHIP_WHITE_MIX)
    base.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('Six-task mean membership figure:', images, flush=True)
    return record


def save_matrix_csv(path, matrix, codes):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true/pred'] + codes)
        writer.writerows([[code] + [f'{value * 100:.6f}' for value in row]
                          for code, row in zip(codes, matrix)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'results/publication_figures/full')
    parser.add_argument('--refresh_predictions', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--per-task', action='store_true', help='Also render archived per-task plots')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    data = output / 'data'
    paths = sorted(data.glob('full_run_*_plot_data.npz'))
    if args.refresh_predictions or len(paths) != 6:
        paths = export_full_prediction_data(data, args.device)
    confidence_outputs = render(paths, output / 'half_violin_w1_w2', dpi=args.dpi)
    print('Confidence figure:', confidence_outputs, flush=True)
    # Also expose actual final blending weights explicitly. They are not the
    # raw routing probabilities and must not be silently interchanged.
    render(paths, output / 'half_violin_final_blend_weights', weight_source='final', dpi=args.dpi)
    given_file = data / 'user_provided_w1.json'
    given_outputs = None
    if given_file.exists():
        given_outputs = render_given_w1(given_file, output / 'half_violin_given_w1', dpi=args.dpi)
        print('User-provided w1 figure:', given_outputs, flush=True)
    aggregate = aggregate_membership(paths, output, args.dpi, per_task=args.per_task)
    mapping = [dict(code=code, fault_name=name) for code, name, _ in FAULTS]
    (output / 'figure_manifest.json').write_text(json.dumps(dict(
        model='Full, original six configurations, seed 2',
        confidence_weight_source='raw DEA routing: softmax(fuse_w_logit)',
        separate_final_blend_figure='half_violin_final_blend_weights',
        user_provided_figure=[str(p) for p in given_outputs] if given_outputs else None,
        class_mapping=mapping, membership_figures=[aggregate]), ensure_ascii=False, indent=2) + '\n')
    write_figure_guide(output)
    print('All figures saved:', output, flush=True)


if __name__ == '__main__':
    main()
