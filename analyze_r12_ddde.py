"""Export recorded r12 DDDE states; distinguish exact records from snapshots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from semilearn.imb_algorithms.metaexpert.cola_ddde import symmetric_kl, normalize_distribution
from semilearn.imb_algorithms.metaexpert.five_variants import l1_distance


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(source, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    tasks = {t['version']: t for t in manifest['tasks'] if t['scenario'] == 'r12'}
    with np.load(tasks['V5']['reference_indices'], allow_pickle=False) as indices:
        with np.load(manifest['data_file'], allow_pickle=False) as dataset:
            unlabeled = np.bincount(dataset['labels'][indices['unlabeled']], minlength=9)
    assert np.array_equal(unlabeled, tasks['V5']['unlabeled_counts'])
    truth = unlabeled / unlabeled.sum()
    prior = normalize_distribution(torch.tensor(tasks['V5']['labeled_counts']))
    uni, rev = torch.ones(9) / 9, prior.flip(0)
    write_csv(output / 'true_unlabeled_distribution_analysis_only.csv', [dict(
        class_id=c, class_name=manifest['class_names'][c], count=int(n), probability=float(p))
        for c, (n, p) in enumerate(zip(unlabeled, truth))])
    states, classes, epochs, counterfactuals = [], [], [], []
    for version in ('V2', 'V5'):
        records = [json.loads(line) for line in (Path(tasks[version]['output']) / 'variant_history.jsonl').open()]
        for r in records:
            pi = torch.tensor(r['pi_estimated'])
            distances = dict(KL_uni=float(symmetric_kl(pi, uni)), KL_rev=float(symmetric_kl(pi, rev)),
                             L1_uni=float(l1_distance(pi, uni)), L1_rev=float(l1_distance(pi, rev)))
            row = dict(version=version, epoch=r['epoch'], event=r['event'], iteration=r['iteration'],
                       tau_used=r['tau_used'], **distances,
                       KL_uni_used_window=r['KL_uni_window'], KL_rev_used_window=r['KL_rev_window'],
                       L1_uni_used_window=r['L1_uni_window'], L1_rev_used_window=r['L1_rev_window'],
                       valid_batches_since_epoch_start=r['valid_batches'],
                       cross_distances_scope='recomputed on the recorded EMA state, not full-epoch mean')
            states.append(row)
            for temperature in (.25, .5, 1., 2., 4.):
                counterfactuals.append(dict(version=version, epoch=r['epoch'], event=r['event'], iteration=r['iteration'],
                    map_temperature=temperature,
                    tau_KL_on_same_recorded_state=float(4 * torch.softmax(-torch.tensor([distances['KL_uni'],distances['KL_rev']]) / temperature, 0)[1]),
                    tau_L1_T1_on_same_recorded_state=float(4 * torch.softmax(-torch.tensor([distances['L1_uni'],distances['L1_rev']]), 0)[1]),
                    scope='offline counterfactual on recorded pi; no retraining; not the actual epoch-window tau'))
            if r['event'] != 'epoch':
                continue
            epochs.append(row)
            for c in range(9):
                classes.append(dict(version=version, epoch=r['epoch'], state_timing='epoch_start',
                    class_id=c, class_name=manifest['class_names'][c],
                    pi_ema=r['pi_estimated'][c], pi_raw_last_observed_batch=r['pi_raw'][c],
                    true_unlabeled_probability=float(truth[c]), true_unlabeled_count=int(unlabeled[c]),
                    estimate_to_true_ratio=r['pi_estimated'][c] / truth[c],
                    high_confidence_pseudo_count=None, effective_rank=None,
                    unavailable_reason='not recorded in historical training; cannot reconstruct from normalized ranks'))
    write_csv(output / 'historical_recorded_states.csv', states)
    write_csv(output / 'same_state_temperature_counterfactual.csv', counterfactuals)
    write_csv(output / 'historical_epoch_start_distances_tau.csv', epochs)
    write_csv(output / 'historical_epoch_start_class_estimates.csv', classes)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for v in ('V2', 'V5'):
        selected = [r for r in epochs if r['version'] == v]
        ax.plot([r['epoch'] for r in selected], [r['tau_used'] for r in selected], label=v)
    ax.set(xlabel='Epoch (zero-based)', ylabel='Tail logit-adjustment coefficient τ', ylim=(0, 4.1))
    ax.legend(); fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / ('historical_tau.' + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, version in zip(axes, ('V2','V5')):
        selected=[r for r in epochs if r['version']==version]
        for metric in ('KL','L1'):
            ax.plot([r['epoch'] for r in selected], [r[metric+'_rev']-r[metric+'_uni'] for r in selected], label=metric+' gap')
        ax.set(title=version+' same-state distance gaps', xlabel='Epoch (zero-based)', ylabel='Distance to reverse minus distance to uniform')
        ax.legend()
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        fig.savefig(output/('historical_distance_gaps.'+ext),dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    stats = {}
    for ax, v in zip(axes, ('V2', 'V5')):
        selected = [r for r in epochs if r['version'] == v and r['epoch'] >= 19]
        matrix = np.asarray([[r['pi_ema'] for r in classes if r['version'] == v and r['epoch'] == e]
                             for e in range(19, 100)])
        ax.plot(range(9), truth * 100, 'o-', label='True U (analysis only)')
        ax.plot(range(9), matrix.mean(0) * 100, 's-', label='Mean recorded EMA, epochs 19–99')
        ax.set(title=v, xlabel='Class', xticks=range(9), xticklabels=[f'C{c}' for c in range(9)])
        ax.legend(fontsize=8)
        stats[v] = dict(mean_tau=float(np.mean([r['tau_used'] for r in selected])),
            min_tau=float(min(r['tau_used'] for r in selected)), max_tau=float(max(r['tau_used'] for r in selected)),
            mean_pi_ema=matrix.mean(0).tolist(), mean_ratio_to_truth=(matrix.mean(0) / truth).tolist(),
            underestimated_classes=[f'C{c}' for c in range(9) if matrix[:, c].mean() < truth[c]],
            KL_gap_mean=float(np.mean([r['KL_rev'] - r['KL_uni'] for r in selected])),
            L1_gap_mean=float(np.mean([r['L1_rev'] - r['L1_uni'] for r in selected])))
    axes[0].set_ylabel('Class probability (%)'); fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / ('historical_distribution.' + ext), dpi=180)
    plt.close(fig)
    (output / 'historical_summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    lines=['# r12历史DDDE与τ诊断','','来源：已完成的192项批次中r12 V2/V5原始日志。',
           '真实U分布由原始NPZ标签和已冻结的unlabeled索引计算，仅离线分析使用。',
           '逐轮表为epoch开始时的精确记录，代表上一轮结束状态；预热期的pi_raw为0，不作为有效估计。',
           'KL/L1跨距离值是在记录的同一pi上重算；不是未记录的完整epoch距离均值。产生实际tau的距离均值单独放在*_used_window列。','',
           '| 类别 | 真实U占比 (%) | V2平均估计 (%) | V5平均估计 (%) | V5/真实 |',
           '|---|---:|---:|---:|---:|']
    for c in range(9):
        lines.append(f'| C{c} | {truth[c]*100:.3f} | {stats["V2"]["mean_pi_ema"][c]*100:.3f} | {stats["V5"]["mean_pi_ema"][c]*100:.3f} | {stats["V5"]["mean_ratio_to_truth"][c]:.3f} |')
    lines+=['', '以上平均取epoch 19–99开始时状态；不是对所有训练批次的均值。',
        f'V2实际tau均值：{stats["V2"]["mean_tau"]:.4f}；V5：{stats["V5"]["mean_tau"]:.4f}。',
        'C5被低估，而C6–C8整体被高估；不能概括为所有尾部类别都被低估。C0也明显低估。',
        '在同一记录分布上，KL的反向/均匀参考距离差大于L1，当前映射更趋向较小tau。两次训练还存在分布估计反馈差异，不能仅凭曲线证明因果。',
        '历史记录没有逐类高置信计数与未归一化有效秩，CSV保留为空，不能从pi_raw倒推。另行带诊断重训V2/V5 T=1，严格标明是复现而非原始历史补录。',
        'same_state_temperature_counterfactual.csv仅展示同一估计分布下温度的机械作用，不是温度重训结果。',
        '', '温度实验预设T=0.25/0.5/1/2/4，保留DDDE+EMA+对称KL。仅按450样本验证集Macro-F1选T，同分比较验证Accuracy，再优先接近T=1。测试前冻结selection.json，仅测试选定温度。',
        '新实验目录：[logs/r12_temperature_20261109](../../logs/r12_temperature_20261109/RESULTS.md)。','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('logs/v1v5_192_20261108'))
    parser.add_argument('--output', type=Path, default=Path('analysis/r12_ddde_temperature_20261109'))
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve())
