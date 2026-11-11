"""Match recorded adjustment states to existing validation/test outcomes."""
import argparse
import csv
import json
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def save_plot(fig,path):
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(path.with_suffix('.'+ext),dpi=180)
    plt.close(fig)


def main(source,output):
    torch.set_num_threads(4);output.mkdir(parents=True,exist_ok=True)
    m=json.loads((source/'manifest.json').read_text())
    versions=('V1','V2','V3','V5')
    scenarios=[s for s in m['protocol']['scenarios'] if s not in ('lu1','lu2')]
    summaries=[];classes=[];validation=[]
    for scenario in scenarios:
        for version in versions:
            t=next(t for t in m['tasks'] if t['scenario']==scenario and t['version']==version)
            folder=Path(t['output'])
            history=[json.loads(s) for s in (folder/'variant_history.jsonl').open()]
            epochs={r['epoch']:r for r in history if r['event']=='epoch'}
            steps_per_epoch=200
            for line in (folder/'log.txt').read_text().splitlines():
                if 'eval/F1:' not in line:continue
                iteration=int(re.search(r'(\d+) iteration',line).group(1))
                epoch=(iteration-1)//steps_per_epoch
                record=epochs[epoch]
                prefix='L1' if version=='V2' else 'KL'
                a,b=record[prefix+'_uni_window'],record[prefix+'_rev_window']
                validation.append(dict(scenario=scenario,version=version,iteration_one_based=iteration,
                    epoch_zero_based=epoch,tau_used=record['tau_used'],
                    mapping_distance=prefix if version!='V1' else 'disabled',
                    mapping_window_uni=a,mapping_window_rev=b,mapping_gap=None if a is None else b-a,
                    validation_macro_f1_logged=float(re.search(r'eval/F1: ([\d.eE+-]+)',line).group(1)),
                    validation_accuracy_logged=float(re.search(r'eval/top-1-acc: ([\d.eE+-]+)',line).group(1)),
                    precision='log values rounded to four decimals; no temporal tail recall available'))
            checkpoint=torch.load(folder/'model_best.pth',map_location='cpu',weights_only=False)
            controller=checkpoint.get('cola_two_kl')
            if controller is None:
                tau=float(checkpoint['five_variant_settings']['fixed_tau']);uni=rev=gap=None
                actual_epoch=checkpoint['best_it']//steps_per_epoch
            else:
                tau=float(controller['tau_dyn']);uni,rev=map(float,controller['last_kl'])
                gap=rev-uni;actual_epoch=int(controller['stats_epoch'])
                expected=float(4*torch.softmax(-controller['last_kl'],0)[1])
                assert abs(tau-expected)<1e-6
            assert abs(epochs[actual_epoch]['tau_used']-tau)<1e-7
            with np.load(folder/'evaluation/confusion_matrix/predictions_heldout.npz',allow_pickle=False) as p:
                precision,recall,f1,support=precision_recall_fscore_support(p['y_true'],p['y_pred'],labels=range(9),zero_division=0)
                acc=float(np.mean(p['y_true']==p['y_pred']))
            saved=json.loads((folder/'test_metrics.json').read_text())
            assert abs(f1.mean()-saved['macro_f1'])<1e-12
            row=dict(scenario=scenario,version=version,seed=t['seed'],
                best_iteration_one_based=checkpoint['best_it']+1,controller_epoch_zero_based=actual_epoch,
                best_validation_accuracy=float(checkpoint['best_eval_acc']),
                tau_best_checkpoint=tau,tau_mean_epoch20_99=float(np.mean([epochs[e]['tau_used'] for e in range(20,100)])),
                mapping_distance='disabled' if version=='V1' else 'L1' if version=='V2' else 'symmetric_KL',
                D_uni_best=uni,D_rev_best=rev,delta_D_best=gap,
                test_accuracy=acc,test_macro_f1=float(f1.mean()),
                C5_recall=float(recall[5]),C8_recall=float(recall[8]),
                C5_C8_macro_recall=float(recall[5:9].mean()),C5_C8_macro_f1=float(f1[5:9].mean()),
                C6_C8_macro_recall=float(recall[6:9].mean()),C6_C8_macro_f1=float(f1[6:9].mean()))
            summaries.append(row)
            for c in range(9):
                classes.append(dict(scenario=scenario,version=version,seed=t['seed'],class_id=c,
                    labeled_count=t['labeled_counts'][c],unlabeled_count=t['unlabeled_counts'][c],
                    tau_best_checkpoint=tau,delta_D_best=gap,support=int(support[c]),
                    precision=float(precision[c]),recall=float(recall[c]),f1=float(f1[c])))
            del checkpoint
    save_csv(output/'best_checkpoint_adjustment_and_metrics.csv',summaries)
    save_csv(output/'per_class_recall_f1.csv',classes)
    save_csv(output/'validation_time_series.csv',validation)
    paired=[]
    for scenario in scenarios:
        candidate=next(r for r in summaries if r['scenario']==scenario and r['version']=='V2')
        for comparator in ('V1','V3','V5'):
            baseline=next(r for r in summaries if r['scenario']==scenario and r['version']==comparator)
            row=dict(scenario=scenario,comparison='V2-'+comparator,seed=2,
                     delta_tau_best=candidate['tau_best_checkpoint']-baseline['tau_best_checkpoint'])
            for key in ('test_accuracy','test_macro_f1','C5_recall','C5_C8_macro_recall','C5_C8_macro_f1','C6_C8_macro_recall'):
                row['delta_'+key+'_pp']=(candidate[key]-baseline[key])*100
            paired.append(row)
    save_csv(output/'within_setting_paired_differences.csv',paired)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for version in versions:
        rows=[r for r in validation if r['scenario']=='r12' and r['version']==version]
        axes[0].plot([r['iteration_one_based'] for r in rows],[r['tau_used'] for r in rows],label=version)
        axes[1].plot([r['iteration_one_based'] for r in rows],[r['validation_macro_f1_logged']*100 for r in rows],label=version)
    axes[0].set_ylabel('τ at validation');axes[1].set(ylabel='Validation Macro-F1 (%)',xlabel='Training iteration (one-based)')
    for ax in axes:ax.legend(ncol=4)
    save_plot(fig,output/'r12_tau_and_validation_macro_f1')
    fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(5,9);width=.19
    for index,version in enumerate(versions):
        values=[r['recall']*100 for r in classes if r['scenario']=='r12' and r['version']==version and r['class_id']>=5]
        ax.bar(x+(index-1.5)*width,values,width,label=version)
    ax.set(xticks=x,xticklabels=[f'C{i}' for i in x],ylabel='Test recall (%)',ylim=(0,105));ax.legend(ncol=4)
    save_plot(fig,output/'r12_tail_class_recall')
    report=['# 调整机制与诊断收益：已有结果分析','','没有新增训练或推理，没有修改模型、映射或超参数。',
        '纳入22个不同设置的V1/V2/V3/V5，均为seed=2；lu1/r1、lu2/r2仅保留一份。',
        '当前τ来自最佳检查点的控制器状态；不使用全程平均τ替代最佳模型τ。checkpoint epoch可能是已完成轮数，另保存controller的零起始epoch避免差一轮。',
        'C5–C8与C6–C8均报告，逐类指标全部导出。它们是按原有类别次序定义的组，不把C0健康/故障类别含义与类别频率混为一谈。分布反向的LU任务仍按有标签类别次序解释该组。',
        '组Macro-F1是完整九类混淆矩阵得到的逐类F1平均，未删去组外样本的误报。','',
        '| r12版本 | 最佳τ | C5 Recall (%) | C8 Recall (%) | C5–C8平均Recall (%) | 全类Macro-F1 (%) |','|---|---:|---:|---:|---:|---:|']
    for version in versions:
        r=next(r for r in summaries if r['scenario']=='r12' and r['version']==version)
        report.append(f'| {version} | {r["tau_best_checkpoint"]:.4f} | {r["C5_recall"]*100:.2f} | {r["C8_recall"]*100:.2f} | {r["C5_C8_macro_recall"]*100:.2f} | {r["test_macro_f1"]*100:.2f} |')
    report+=['','C0/C1/C2/C4在r12各版本Recall均为100%，主要差异集中于C5及部分C8；仅看全类Accuracy或C6–C8平均会掩盖C5。',
        'ΔD→τ的单调关系由公式强制规定，不是“有效性”的独立实验证据。ΔD的L1与KL数值也不能不加区分地跨版本合并。',
        '验证日志仅有每500步的全类F1/Accuracy，且已四舍五入；没有逐时点类别Recall或逐时点预测。现有数据无法建立每轮τ与尾类Recall的时间对应，不能编造尾类训练曲线。',
        '已导出同一设置内V2相对其他版本的差值。版本与训练路径同时改变，差值是描述性配对，不是隔离τ的因果效应。','',
        '| 比较 | 全类Macro-F1胜/平/负 | C5–C8平均Recall胜/平/负 |','|---|---:|---:|']
    for v in ('V1','V3','V5'):
        rows=[r for r in paired if r['comparison']=='V2-'+v]
        def counts(key):
            a=np.array([r[key] for r in rows]);return f'{int((a>1e-9).sum())}/{int((np.abs(a)<=1e-9).sum())}/{int((a < -1e-9).sum())}'
        report.append(f'| V2 vs {v} | {counts("delta_test_macro_f1_pp")} | {counts("delta_C5_C8_macro_recall_pp")} |')
    report+=['','上述22设置共享样本、测试集及seed，不能作为22次独立随机重复。没有报告伪多种子均值/标准差或显著性。','',
        '## 多种子验证预案（未执行）','','候选固定为V2，对照固定V1/V3/V5，不加入Histogram+ER融合，不扩展温度网格。',
        '按输入设置覆盖类型，建议预先选择r2（中等长尾）、r12（极端长尾）、CD3（标签分布不匹配）、LU6（无标签反向分布）。r12保留为探索性场景，不能称为全新确认数据。',
        '建议训练种子2/3/4/5/6，按同一场景和种子配对四版本；checkpoint选择规则保持原验证Accuracy。主指标Macro-F1，补充逐类及预先指定尾类组Recall/F1。',
        '每个设置分别报告跨种子均值、样本标准差(ddof=1)，以及同种子V2−各对照的配对差值。不要用22个设置或训练epoch充当种子重复。',
        '现有args.seed同时控制划分、L/U抽样及训练随机性。直接改seed会改变数据身份，不能宣称固定划分。若目标是隔离训练随机性，后续执行前须将split_seed与train_seed分离，并检查各版本数据索引；本轮没有实现该改动，也未生成或启动新训练。',
        '若选择多划分重复，应明确报告它混合了划分与优化随机性；不同种子仍共享原始采集记录，不等于跨记录或外部验证。','']
    (output/'REPORT.md').write_text('\n'.join(report))
    (output/'metadata.json').write_text(json.dumps(dict(
        versions=list(versions),seed=2,distinct_settings=len(scenarios),models=len(summaries),
        new_training_runs=0,new_inference_runs=0,checkpoint_tau_exact=True,
        temporal_tail_recall_available=False,temporal_validation_metrics_rounded=True,
        candidate_selection_or_new_hyperparameters=False),indent=2)+'\n')
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(output.parent))
    print('\n'.join(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('logs/v1v5_192_20261108'))
    parser.add_argument('--output',type=Path,default=Path('analysis/adjustment_outcomes_20261109'))
    args=parser.parse_args();main(args.source.resolve(),args.output.resolve())
