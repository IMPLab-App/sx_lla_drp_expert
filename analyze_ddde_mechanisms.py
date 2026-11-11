"""Offline distribution diagnostics only: no training, inference or selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


VERSIONS=('V2','V3','V5')


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def normalize(p):
    p=np.asarray(p,dtype=np.float64)
    assert np.isfinite(p).all() and (p>=0).all() and p.sum()>0
    return p/p.sum()


def errors(p,q):
    p,q=normalize(p),normalize(q)
    midpoint=(p+q)/2
    def kl(a,b):
        mask=a>0
        return float(np.sum(a[mask]*np.log(a[mask]/b[mask])))
    return float(np.abs(p-q).sum()), .5*(kl(p,midpoint)+kl(q,midpoint))


def save_plot(fig, base):
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):
        fig.savefig(base.with_suffix('.'+suffix),dpi=200)
    plt.close(fig)


def main(source,replay,output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    with np.load(manifest['data_file'],allow_pickle=False) as data:
        all_labels=data['labels'].copy()
    summaries=[];state_errors=[];class_rows=[];cache={};provenance={}
    distinct=[s for s in manifest['protocol']['scenarios'] if s not in ('lu1','lu2')]
    for scenario in manifest['protocol']['scenarios']:
        for version in VERSIONS:
            task=next(t for t in manifest['tasks'] if t['scenario']==scenario and t['version']==version)
            with np.load(task['reference_indices'],allow_pickle=False) as ids:
                truth_counts=np.bincount(all_labels[ids['unlabeled']],minlength=9)
            assert np.array_equal(truth_counts,task['unlabeled_counts'])
            truth=normalize(truth_counts)
            path=Path(task['output'])/'variant_history.jsonl'
            provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            logs=[json.loads(line) for line in path.open()]
            epochs=[r for r in logs if r['event']=='epoch' and r['epoch']>=19]
            assert len(epochs)==81
            matrices={key:np.stack([normalize(r[key]) for r in epochs]) for key in ('pi_estimated','pi_raw')}
            scores=json.loads((Path(task['output'])/'test_metrics.json').read_text())
            for start in (19,20,40):
                mask=np.array([r['epoch']>=start for r in epochs])
                for key,matrix in matrices.items():
                    values=matrix[mask]
                    e=np.array([errors(p,truth) for p in values])
                    l1_mean,js_mean=errors(values.mean(0),truth)
                    summaries.append(dict(scenario=scenario,version=version,seed=task['seed'],
                        duplicate_setting=scenario in ('lu1','lu2'),epoch_start=start,epoch_end=99,
                        state='EMA' if key=='pi_estimated' else 'raw_last_batch',
                        mean_epoch_L1=float(e[:,0].mean()),mean_epoch_JSD_nats=float(e[:,1].mean()),
                        L1_of_mean_distribution=l1_mean,JSD_of_mean_distribution_nats=js_mean,
                        temporal_L1_sd=float(e[:,0].std()),test_accuracy_historical=scores['accuracy'],
                        test_macro_f1_historical=scores['macro_f1']))
            if scenario!='r12':
                continue
            cache[version]=dict(epochs=epochs,matrices=matrices,truth=truth)
            for r in logs:
                if r['epoch']<19:
                    continue
                for key in ('pi_estimated','pi_raw'):
                    l1,js=errors(r[key],truth)
                    state_errors.append(dict(version=version,epoch=r['epoch'],iteration=r['iteration'],
                        event=r['event'],state='EMA' if key=='pi_estimated' else 'raw_last_batch',
                        L1=l1,JSD_nats=js,tau_used=r['tau_used']))
            for r in epochs:
                for c in range(9):
                    class_rows.append(dict(version=version,epoch=r['epoch'],class_id=c,
                        true_probability=float(truth[c]),pi_ema=r['pi_estimated'][c],
                        pi_raw_last_batch=r['pi_raw'][c],ema_bias_pp=(r['pi_estimated'][c]-truth[c])*100))
    save_csv(output/'all_settings_distribution_error_summary.csv',summaries)
    save_csv(output/'r12_recorded_state_errors.csv',state_errors)
    save_csv(output/'r12_epoch_start_class_estimates.csv',class_rows)
    er_rows=[];er_summary=[]
    for version in ('V2','V5'):
        path=replay/'models'/f'r12_{version}_mapT1'/'epoch_diagnostics.jsonl'
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=[json.loads(s) for s in path.open()]
        rows=[r for r in rows if r['epoch']>=20]
        for r in rows:
            for c in range(9):
                n=r['high_confidence_pseudo_counts'][c]/r['rank_batches']
                rank=r['rank_mean'][c]
                assert rank<=n+1e-4
                er_rows.append(dict(version=version,epoch=r['epoch'],class_id=c,
                    pseudo_count_occurrences=r['high_confidence_pseudo_counts'][c],
                    pseudo_count_mean_per_rank_batch=n,effective_rank_mean_per_rank_batch=rank,
                    ER_to_count_ratio=rank/n if n else None,
                    absence_fraction=r['absent_pseudo_class_batches'][c]/r['rank_batches'],
                    pi_raw_mean=r['pi_raw_mean'][c],pi_ema_mean=r['pi_ema_mean'][c]))
        count=np.array([r['high_confidence_pseudo_counts'] for r in rows]).sum(0)
        for c in range(9):
            selected=[r for r in er_rows if r['version']==version and r['class_id']==c]
            n=np.array([r['pseudo_count_mean_per_rank_batch'] for r in selected])
            rank=np.array([r['effective_rank_mean_per_rank_batch'] for r in selected])
            er_summary.append(dict(version=version,class_id=c,epoch_start=20,epoch_end=99,
                true_probability=float(cache[version]['truth'][c]),
                high_conf_pseudo_share=float(count[c]/count.sum()),
                mean_count_per_rank_batch=float(n.mean()),mean_ER_per_rank_batch=float(rank.mean()),
                ratio_of_mean_ER_to_mean_count=float(rank.mean()/n.mean()),
                count_temporal_cv=float(n.std()/n.mean()),rank_temporal_cv=float(rank.std()/rank.mean()),
                mean_absence_fraction=float(np.mean([r['absence_fraction'] for r in selected])),
                mean_raw_normalized_ER=float(np.mean([r['pi_raw_mean'] for r in selected])),
                mean_pi_ema=float(np.mean([r['pi_ema_mean'] for r in selected]))))
    save_csv(output/'ER_count_epoch_aggregates.csv',er_rows)
    save_csv(output/'ER_count_class_summary.csv',er_summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for version in VERSIONS:
        d=cache[version]
        values=np.array([errors(p,d['truth']) for p in d['matrices']['pi_estimated']])
        for i,ax in enumerate(axes):
            ax.plot([r['epoch'] for r in d['epochs']],values[:,i],label=version)
    for ax,label in zip(axes,('L1 error','Jensen–Shannon divergence (nats)')):
        ax.set(xlabel='Epoch start (zero-based)',ylabel=label);ax.legend()
    save_plot(fig,output/'r12_estimation_error_curves')
    fig,ax=plt.subplots(figsize=(10,4.8));positions=np.arange(9);width=.2
    truth=cache['V5']['truth']
    ax.bar(positions-1.5*width,truth*100,width,label='True U (offline only)')
    for offset,version in enumerate(VERSIONS):
        ax.bar(positions+(offset-.5)*width,cache[version]['matrices']['pi_estimated'].mean(0)*100,width,label=version)
    ax.set(xticks=positions,xticklabels=[f'C{i}' for i in range(9)],ylabel='Probability (%)')
    ax.legend();save_plot(fig,output/'r12_true_and_estimated_prior')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,version in zip(axes,('V2','V5')):
        for c in range(9):
            r=[r for r in er_rows if r['version']==version and r['class_id']==c]
            ax.scatter([x['pseudo_count_mean_per_rank_batch'] for x in r],
                       [x['effective_rank_mean_per_rank_batch'] for x in r],s=9,alpha=.5,label=f'C{c}')
        ax.set(title=version,xlabel='Mean high-confidence pseudo count / rank batch',ylabel='Mean effective rank / rank batch')
        ax.legend(fontsize=7,ncol=3)
    save_plot(fig,output/'r12_ER_vs_count_epoch_means')
    # Aggregate class/epoch means cannot establish a within-batch saturation curve.
    report=['# DDDE机制分析：只使用已有日志','','没有新增训练、推理、温度搜索或估计器修改。真实U标签仅离线计算误差。',
        '主口径：历史epoch19–99开始时的81个状态；另导出20–99和40–99敏感性统计。JSD使用自然对数，不开平方，范围[0,ln2]。',
        '跨epoch数值是时间描述，不是独立随机重复；不据此计算多种子显著性。','',
        '| r12版本 | 平均逐轮L1 | 平均逐轮JSD（nats） | 平均分布的L1 |','|---|---:|---:|---:|']
    for version in VERSIONS:
        r=next(r for r in summaries if r['scenario']=='r12' and r['version']==version and r['epoch_start']==19 and r['state']=='EMA')
        report.append(f'| {version} | {r["mean_epoch_L1"]:.6f} | {r["mean_epoch_JSD_nats"]:.6f} | {r["L1_of_mean_distribution"]:.6f} |')
    report+=['','V3在r12的整体频率估计更接近真实分布，但不能据此证明准确估计是性能提升的唯一原因。估计、预测和tau存在训练反馈。','',
        '| 类别 | 真实U (%) | V2平均EMA (%) | V3平均EMA (%) | V5平均EMA (%) |','|---|---:|---:|---:|---:|']
    for c in (0,5,8):
        estimates=[cache[v]['matrices']['pi_estimated'][:,c].mean()*100 for v in VERSIONS]
        report.append(f'| C{c} | {truth[c]*100:.3f} | '+ ' | '.join(f'{x:.3f}' for x in estimates)+' |')
    report+=['','V3仍低估C5并高估C8：Histogram没有消除高置信伪标签选择偏差。','',
        '## 有效秩与计数','','逐类报告完整epoch20–99的高置信出现次数、ER均值、平均归一化ER、EMA、缺失比例及跨epoch变异系数。',
        '高置信次数包含重复抽样；ER是原始逐批有效秩的平均，不是把整轮特征拼接后重算的有效秩。mean(normalized ER)与normalize(mean ER)不相同，未混用。','',
        '| V5类别 | 平均高置信数/批 | 平均ER/批 | ER/计数 | 缺失批次 (%) | 计数跨轮CV | ER跨轮CV |','|---|---:|---:|---:|---:|---:|---:|']
    for c in range(9):
        r=next(r for r in er_summary if r['version']=='V5' and r['class_id']==c)
        report.append(f'| C{c} | {r["mean_count_per_rank_batch"]:.3f} | {r["mean_ER_per_rank_batch"]:.3f} | {r["ratio_of_mean_ER_to_mean_count"]:.3f} | {r["mean_absence_fraction"]*100:.2f} | {r["count_temporal_cv"]:.3f} | {r["rank_temporal_cv"]:.3f} |')
    report+=['','ER相对于计数的比例随类别变化，支持“有效秩占比不能直接当作频率”的判断。',
        '但现有记录仅保留每轮聚合量，没有每批(N_c,ER_c)配对、奇异值谱或固定特征下的受控子采样曲线，不能证明大类发生有效秩饱和，也不能确定单样本返回ER=1的具体贡献。',
        '特征维度为192，单批U为128，不能把观察到的ER偏小直接归因于192维的硬上限。','',
        '## DDDE怎样影响Logit Adjustment','','当前代码并不直接使用log(pi_DDDE)作为逐类校正项。真实链条是：',
        'pi_DDDE/EMA → 与Uniform和Reverse的距离 → 全局标量tau → z_c + tau * log(pi_l,c)。',
        'pi_l仍来自有标签样本计数（metaexpert.py第22、97、334行）。DDDE误差通过tau间接影响专家损失和融合路由。',
        '因此应区分“类别频率估计误差”和“估计信息经过当前映射后是否有利于诊断”，不能把两者当成同一个目标。','',
        '## 已有场景与种子证据','','排除lu1=r1、lu2=r2的重复设置后，剩22个不同设置；它们仍共享源数据、测试样本和seed=2，不视为统计独立重复。','',
        '| 比较 | L1误差更低 | JSD更低 | 分类Accuracy胜/平/负 | 设置总数 |','|---|---:|---:|---:|---:|']
    overview=[]
    for a,b in [('V3','V5'),('V2','V5')]:
        paired=[]
        for scenario in distinct:
            first=next(r for r in summaries if r['scenario']==scenario and r['version']==a and r['epoch_start']==19 and r['state']=='EMA')
            second=next(r for r in summaries if r['scenario']==scenario and r['version']==b and r['epoch_start']==19 and r['state']=='EMA')
            paired.append((first,second))
        l1wins=sum(a['mean_epoch_L1']<b['mean_epoch_L1'] for a,b in paired)
        jswins=sum(a['mean_epoch_JSD_nats']<b['mean_epoch_JSD_nats'] for a,b in paired)
        differences=np.asarray([a['test_accuracy_historical']-b['test_accuracy_historical'] for a,b in paired])
        wins,ties,losses=int((differences>1e-12).sum()),int((np.abs(differences)<=1e-12).sum()),int((differences < -1e-12).sum())
        report.append(f'| {a} vs {b} | {l1wins} | {jswins} | {wins}/{ties}/{losses} | {len(paired)} |')
        overview.append(dict(comparison=f'{a} vs {b}',distinct_settings=len(paired),L1_wins=l1wins,JSD_wins=jswins,
            accuracy_wins=wins,accuracy_ties=ties,accuracy_losses=losses,
            historical_accuracy_mean_difference_pp=float(np.mean([a['test_accuracy_historical']-b['test_accuracy_historical'] for a,b in paired])*100)))
    report+=['','V3的频率误差在22/22个设置中更低，但Accuracy仅在11个设置中更高；更准确的频率估计并未自动产生更好的分类结果。',
        'V2/V3/V5当前可匹配训练记录均只有seed=2。温度复现不增加种子数，不能完成多种子优势验证。',
        '未用上述离线真实先验误差挑选新模型、融合系数或超参数。Histogram+ER融合保持为未测试假设，不添加到当前方法。','',
        '论文目前可表述为：在r12、seed=2中，归一化有效秩与真实类别频率存在偏离，Histogram变体具有更低的整体估计误差；跨种子稳定性和因果贡献尚待验证。','']
    (output/'REPORT.md').write_text('\n'.join(report))
    (output/'setting_comparisons.json').write_text(json.dumps(overview,indent=2)+'\n')
    for name,value in provenance.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==value
    (output/'provenance.json').write_text(json.dumps(dict(
        source=str(source),replay=str(replay),source_sha256=provenance,
        versions=list(VERSIONS),seeds=[2],training_runs_started=0,
        JSD_definition='0.5 KL(p||m)+0.5 KL(q||m), m=(p+q)/2, natural log; no sqrt',
        primary_scope='epoch-start EMA snapshots, epochs19-99; 20-99 and40-99 also reported',
        limitations='Epoch aggregates do not recover within-minibatch ER/count relationships. No multi-seed inference.'),indent=2)+'\n')
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(output.parent))
    print('\n'.join(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('logs/v1v5_192_20261108'))
    parser.add_argument('--replay',type=Path,default=Path('logs/r12_temperature_20261109'))
    parser.add_argument('--output',type=Path,default=Path('analysis/ddde_mechanisms_20261109'))
    args=parser.parse_args()
    main(args.source.resolve(),args.replay.resolve(),args.output.resolve())
