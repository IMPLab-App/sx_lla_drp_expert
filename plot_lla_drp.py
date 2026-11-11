"""Generate exportable figures only from completed models' raw validation logs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    records=json.loads((batch/'comparison/raw_validation_results.json').read_text())
    lookup={(r['scenario'],r['version']):r for r in records if r['status']=='complete'}
    if not all((f'LU{i}',v) in lookup for i in range(1,7) for v in ('M0','L0','M1','V5')):return
    destination=batch/'comparison/figures';destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    inputs=set();outputs=[]
    def save(fig,name):
        for extension in ('png','pdf'):
            path=destination/(name+'.'+extension);fig.savefig(path,dpi=240,bbox_inches='tight');outputs.append(str(path))
        plt.close(fig)
    for row in lookup.values():inputs.add(row['metric_file'])
    names={'M0':'Full LLA+DRP','L0':'LLA-only','M1':'Fixed tau4','V5':'DDDE+KL','M2':'Fixed DEA weights'}
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(6);width=.19
    for index,v in enumerate(('M0','L0','M1','V5')):
        ax.bar(x+(index-1.5)*width,[lookup[(f'LU{i}',v)]['val_macro_f1']*100 for i in range(1,7)],width,label=names[v])
    ax.set_xticks(x,[f'LU{i}' for i in range(1,7)]);ax.set_ylabel('Validation Macro-F1 (%)');ax.legend(fontsize=8,ncol=2)
    save(fig,'six_task_replacement')
    fig,ax=plt.subplots(figsize=(8,4))
    for v in ('M0','L0','M1','V5'):
        recall=np.mean([lookup[(f'LU{i}',v)]['branches']['fusion_main']['recall'] for i in range(1,7)],axis=0)
        ax.plot(range(9),recall*100,marker='o',label=names[v])
    ax.set_xticks(range(9),[f'C{i}' for i in range(9)]);ax.set_ylabel('Mean validation recall across six tasks (%)');ax.legend(fontsize=8)
    save(fig,'class_recall')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,metric in zip(axes,('recall','f1')):
        for index,v in enumerate(('M0','L0','M1','V5')):
            values=[np.mean([lookup[(f'LU{i}',v)]['branches']['fusion_main'][f'C{c}'][metric] for i in range(1,7)])*100 for c in (5,8)]
            ax.bar(np.arange(2)+(index-1.5)*width,values,width,label=names[v])
        ax.set_xticks([0,1],['C5','C8']);ax.set_ylabel('Mean validation '+metric+' (%)')
    axes[0].legend(fontsize=7);save(fig,'C5_C8_diagnostics')
    complement=np.asarray([[lookup[(f'LU{i}','M0')]['complementarity_lla'][k] for k in ('e1_only','e3_only','both_correct','both_wrong')] for i in range(1,7)])
    fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(complement,cmap='Blues',aspect='auto')
    ax.set_xticks(range(4),['E1 only correct','E3 only correct','Both correct','Both wrong'],rotation=15)
    ax.set_yticks(range(6),[f'LU{i}' for i in range(1,7)])
    for i in range(6):
        for j in range(4):ax.text(j,i,str(complement[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Validation samples');save(fig,'expert_complementarity_counts')
    for i in range(1,7):
        task=next(t for t in manifest['tasks'] if t['scenario']==f'LU{i}' and t['version']=='M0');folder=Path(task['output'])
        lambda_path=folder/'training_history.jsonl';drp_path=folder/'drp_history.jsonl';inputs.update((str(lambda_path),str(drp_path)))
        steps=[];values=[]
        with lambda_path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['iteration']%50==0:steps.append(row['iteration']);values.append(row['lambda_softmax'])
        fig,ax=plt.subplots(figsize=(7,4));value=np.asarray(values)
        for c in range(9):ax.plot(steps,value[:,c],label=f'C{c}')
        ax.set_xlabel('Training step');ax.set_ylabel('softmax(lambda)');ax.legend(ncol=3,fontsize=7)
        save(fig,f'LU{i}_lambda_trajectory')
        totals=np.zeros(9);counts=np.zeros(9);xs=[];means=[];ess=[];mff=[]
        with drp_path.open() as stream:
            for line in stream:
                row=json.loads(line);stats=row['drp']
                totals+=stats['class_effective_weight_sums'];counts+=stats['predicted_class_counts']
                if row['iteration']%50==0:
                    xs.append(row['iteration']);means.append(stats['weight_mean']);ess.append(stats['effective_sample_size']);mff.append(row['mff_gradient_norm'])
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        axes[0].bar(range(9),counts);axes[0].set_ylabel('Pseudo-label counts over training draws')
        axes[1].bar(range(9),totals);axes[1].set_ylabel('Effective DRP weight sums over training draws')
        for ax in axes:ax.set_xticks(range(9),[f'C{c}' for c in range(9)])
        save(fig,f'LU{i}_pseudo_label_participation')
        fig,axes=plt.subplots(1,3,figsize=(12,3));axes[0].plot(xs,means);axes[0].set_ylabel('Mean DRP weight')
        axes[1].plot(xs,ess);axes[1].set_ylabel('Effective unlabeled sample size')
        axes[2].plot(xs,mff);axes[2].set_ylabel('MFF gradient norm')
        for ax in axes:ax.set_xlabel('Training step')
        save(fig,f'LU{i}_drp_gradient_diagnostics')
    if all((f'LU{i}','M2') in lookup for i in range(1,7)):
        fig,ax=plt.subplots(figsize=(8,4))
        for v in ('M0','M1','M2'):ax.plot(range(1,7),[lookup[(f'LU{i}',v)]['val_macro_f1']*100 for i in range(1,7)],marker='o',label=names[v])
        ax.set_xticks(range(1,7),[f'LU{i}' for i in range(1,7)]);ax.set_ylabel('Validation Macro-F1 (%)');ax.legend()
        save(fig,'six_task_dynamic_weighting_ablation')
    (destination/'provenance.json').write_text(json.dumps(dict(inputs={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(inputs)},outputs=outputs,
        data='Raw completed-model validation metrics and training logs only',single_seed=2,
        R_CD_figures='not generated because those stages have not run'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);a=p.parse_args();plot(a.batch.resolve())
