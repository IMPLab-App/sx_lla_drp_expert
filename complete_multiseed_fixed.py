"""Finalize five-training-seed statistics without treating scenes as repeats."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np


def csv_file(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),sd=float(values.std(ddof=1)))


def finish(batch,allow_smoke=False):
    m=json.loads((batch/'manifest.json').read_text())
    assert not m['smoke'] or allow_smoke
    sys.path.insert(0,m['code_snapshot'])
    from multiseed_fixed import report
    assert report(batch)==80
    folder=batch/'comparison';rows=list(csv.DictReader((folder/'per_seed_results.csv').open(encoding='utf-8-sig')))
    classes=list(csv.DictReader((folder/'per_class_metrics.csv').open(encoding='utf-8-sig')))
    class_stats=[];error_stats=[];distribution=[]
    for scenario in m['protocol']['scenarios']:
        for version in m['protocol']['versions']:
            tasks=[t for t in m['tasks'] if t['scenario']==scenario and t['version']==version]
            assert len(tasks)==5 and {t['seed'] for t in tasks}=={2,3,4,5,6}
            matrices=[]
            for t in tasks:
                value=json.loads((Path(t['output'])/'test_metrics.json').read_text())
                matrices.append(np.asarray(value['confusion_counts'])[:,:9])
            for c in range(9):
                selected=[r for r in classes if r['scenario']==scenario and r['version']==version and int(r['class_id'])==c]
                assert len(selected)==5
                row=dict(scenario=scenario,version=version,class_id=c,n=5)
                for key in ('precision','recall','f1'):
                    s=stats([float(r[key]) for r in selected]);row.update({key+'_mean':s['mean'],key+'_sd':s['sd']})
                class_stats.append(row)
            for direction in ('C5_false_negative_destination','C5_false_positive_source'):
                for c in range(9):
                    if c==5:continue
                    values=[matrix[5,c] if 'negative' in direction else matrix[c,5] for matrix in matrices]
                    s=stats(values)
                    error_stats.append(dict(scenario=scenario,version=version,direction=direction,other_class=c,n=5,
                                            count_sum=int(sum(values)),count_mean=s['mean'],count_sd=s['sd']))
            if version!='V1':
                scores=[json.loads((Path(t['output'])/'distribution_errors_analysis_only.json').read_text()) for t in tasks]
                row=dict(scenario=scenario,version=version,n=5)
                for key in ('best_checkpoint_L1','best_checkpoint_JSD_nats','mean_epoch_start_L1','mean_epoch_start_JSD_nats'):
                    values=[s[key] for s in scores if s[key] is not None]
                    s=stats(values) if len(values)>1 else dict(mean=None,sd=None)
                    row.update({key+'_mean':s['mean'],key+'_sd':s['sd']})
                distribution.append(row)
    csv_file(folder/'per_class_five_seed_summary.csv',class_stats)
    csv_file(folder/'c5_confusion_destinations_five_seed_summary.csv',error_stats)
    csv_file(folder/'distribution_errors_five_seed_summary.csv',distribution)
    text=(folder/'RESULTS.md').read_text()
    if m['smoke']:
        text='# 短训练流程验证：禁止作为正式实验结果\n\n'+text
    text+='\n## C5 Precision / Recall / F1\n\n均值与样本标准差均跨5个训练种子计算，单位%。\n\n'
    text+='| 场景 | 版本 | Precision均值±SD | Recall均值±SD | F1均值±SD |\n|---|---|---:|---:|---:|\n'
    for r in class_stats:
        if r['class_id']!=5:continue
        text+=f'| {r["scenario"]} | {r["version"]} | '+ ' | '.join(f'{r[k+"_mean"]*100:.2f}±{r[k+"_sd"]*100:.2f}' for k in ('precision','recall','f1'))+' |\n'
    text+='\n漏诊去向与误报来源分别导出；每个种子的零计数也计入均值和标准差。各原始混淆矩阵保存在对应模型目录。\n'
    text+='\n分布误差只作为离线补充指标，不参与检查点或方法选择。V1没有分布估计器，不将其误差填为0。\n'
    text+='\n这里的标准差反映固定数据身份下训练随机性的变化，不包括重新划分数据或更换采集记录的变化。\n'
    text+='\n80项为全新训练；采样器显式使用train_seed+epoch，因此新seed=2不要求复刻历史epoch-only采样轨迹。\n'
    (folder/'FINAL_RESULTS.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from publication_plotting import paper_style
    with plt.rc_context(paper_style()):
        fig,axes=plt.subplots(2,2,figsize=(11,7.5))
        for ax,scenario in zip(axes.flat,m['protocol']['scenarios']):
            for position,version in enumerate(m['protocol']['versions']):
                group=[r for r in rows if r['scenario']==scenario and r['version']==version]
                group.sort(key=lambda r:int(r['train_seed']))
                y=np.array([float(r['macro_f1'])*100 for r in group])
                ax.errorbar(position,y.mean(),yerr=y.std(ddof=1),fmt='o',color='black',capsize=5,zorder=3)
                ax.scatter(position+np.linspace(-.12,.12,5),y,s=25,zorder=2)
            ax.set(title=scenario,xticks=range(4),xticklabels=m['protocol']['versions'],ylabel='Macro-F1 (%)')
        fig.suptitle('Fixed split; five training seeds; bars = sample SD')
        fig.tight_layout()
        for ext in ('png','svg','pdf'):fig.savefig(folder/('macro_f1_five_training_seeds.'+ext),dpi=250)
        plt.close(fig)
    archive=batch/'five_seed_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in folder.rglob('*'):
            if path.is_file():z.write(path,'comparison/'+str(path.relative_to(folder)))
        for name in ('manifest.json','PROTOCOL.md','verification.json','gpu_audit.json','smoke_audit.json'):
            if (batch/name).exists():z.write(batch/name,name)
            elif not m['smoke']:raise FileNotFoundError(batch/name)
        for path in (batch/'splits').glob('*.npz'):z.write(path,'splits/'+path.name)
    (batch/'postprocess_status.json').write_text(json.dumps(dict(state='completed',runs=80,
        report=str(folder/'FINAL_RESULTS.md'),archive=str(archive)),indent=2)+'\n')
    print('Final five-seed report:',folder/'FINAL_RESULTS.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path);parser.add_argument('--watch',action='store_true');parser.add_argument('--allow-smoke',action='store_true')
    a=parser.parse_args();batch=a.batch.resolve()
    while True:
        status=(batch/'status.txt').read_text().strip() if (batch/'status.txt').exists() else 'prepared'
        if status=='completed':break
        if status=='completed_with_failures':raise RuntimeError('Queue has failed tasks; inspect issue logs')
        if not a.watch:raise RuntimeError('Queue not complete')
        time.sleep(15)
    finish(batch,a.allow_smoke)
