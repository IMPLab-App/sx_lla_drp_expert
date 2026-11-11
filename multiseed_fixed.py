"""Fixed split-seed=2, training-seeds=2..6, four scenarios/four variants."""
import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import yaml


SCENARIOS=('r2','r12','cd3','lu6')
VERSIONS=('V1','V2','V3','V5')
SEEDS=(2,3,4,5,6)
KEYS=('labeled','unlabeled','validation','test')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def csv_file(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def prepare(batch,smoke=False):
    from semilearn.datasets.cv_datasets.raw_signal import _stratified_split
    from semilearn.imb_algorithms.metaexpert.five_variants import SPECS
    repo=Path(__file__).resolve().parent;reference=repo/'logs/v1v5_192_20261108'
    original=json.loads((reference/'manifest.json').read_text())
    batch.mkdir(parents=True,exist_ok=False);code=batch/'code'
    shutil.copytree(reference/'code',code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    overlays=('train.py','semilearn/datasets/cv_datasets/raw_signal.py',
              'semilearn/core/utils/build.py','semilearn/datasets/samplers/sampler.py',
              'multiseed_fixed.py','verify_multiseed_fixed.py','publication_plotting.py',
              'analyze_ddde_mechanisms.py')
    for name in overlays:shutil.copy2(repo/name,code/name)
    (batch/'data').mkdir();data=batch/'data'/Path(original['data_file']).name
    shutil.copy2(original['data_file'],data);assert digest(data)==original['data_sha256']
    with np.load(data,allow_pickle=False) as values:labels=values['labels'];names=values['class_names']
    split=_stratified_split(labels,9,2)
    (batch/'splits').mkdir();(batch/'configs').mkdir();tasks=[];split_records={}
    for case,scenario in enumerate(SCENARIOS,1):
        source=next(t for t in original['tasks'] if t['scenario']==scenario and t['version']=='V5')
        with np.load(source['reference_indices'],allow_pickle=False) as saved:
            ids={key:saved[key].copy() for key in KEYS}
        np.testing.assert_array_equal(ids['validation'],split['eval'])
        np.testing.assert_array_equal(ids['test'],split['test'])
        frozen=batch/'splits'/(scenario+'.npz')
        np.savez_compressed(frozen,train_pool=split['train'],class_names=names,**ids)
        split_records[scenario]=dict(file=str(frozen),sha256=digest(frozen),reference=source['reference_indices'],
            source_sha256=digest(source['reference_indices']),full_counts=source['full_counts'],
            labeled_counts=source['labeled_counts'],unlabeled_counts=source['unlabeled_counts'])
        base=yaml.safe_load(Path(source['config']).read_text())
        for seed_index,seed in enumerate(SEEDS):
            for version_index,version in enumerate(VERSIONS):
                spec=SPECS[version];name=source['case_name']+f'__{version}__train{seed}_split2'
                cfg=dict(base)
                cfg.update(seed=seed,train_seed=seed,split_seed=2,
                    fixed_split_file=str(frozen),fixed_split_sha256=digest(frozen),
                    data_dir=str(batch/'data'),save_dir=str(batch/'models'),save_name=name,
                    five_variant=version,five_fixed_tau=spec['fixed_tau'] if spec['fixed_tau'] is not None else 4.,
                    ablation_no_ddde_kl=not spec['dynamic'],ablation_no_dea_aggregator=False,
                    overwrite=False,resume=False,load_path=None)
                if smoke:cfg.update(epoch=3,num_train_iter=6,num_eval_iter=2,num_log_iter=1,
                    batch_size=8,num_workers=0,est_epoch=0,use_tensorboard=False)
                path=batch/'configs'/(name+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                tasks.append(dict(name=name,case=case,scenario=scenario,version=version,method=version,
                    repeat=seed_index+1,seed=seed,train_seed=seed,split_seed=2,
                    gpu=(case-1+seed_index+version_index)%2,config=str(path),output=str(batch/'models'/name),
                    commands=[[str(code/'train.py'),'--c',str(path)],
                              [str(code/'multiseed_fixed.py'),'evaluate',str(batch),'--task',name],
                              [str(code/'verify_multiseed_fixed.py'),str(batch),'--check-task',name]]))
    assert Counter(t['gpu'] for t in tasks)=={0:40,1:40}
    # Numerical model files are exactly the old four variants, including the KL map.
    models=('semilearn/imb_algorithms/metaexpert/metaexpert.py',
            'semilearn/imb_algorithms/metaexpert/five_variants.py',
            'semilearn/imb_algorithms/metaexpert/cola_ddde.py',
            'semilearn/imb_algorithms/metaexpert/bacn_ablation.py',
            'semilearn/nets/vit/vit_metaexpert.py')
    for name in models:assert digest(code/name)==digest(reference/'code'/name)
    m=dict(title='固定划分、多训练种子：4场景×4版本×5种子',created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        smoke=smoke,code_snapshot=str(code),reference_batch=str(reference),
        data_file=str(data),data_sha256=digest(data),class_names=names.tolist(),
        methods=list(VERSIONS),tasks=tasks,splits=split_records,overlays=list(overlays),
        model_sha256={name:digest(code/name) for name in models},
        protocol=dict(scenarios=list(SCENARIOS),versions=list(VERSIONS),train_seeds=list(SEEDS),split_seed=2,
            total=80,repeats=5,fresh_training_all_80=True,gpu_tasks={'0':40,'1':40},
            epochs=3 if smoke else 100,iterations=6 if smoke else 20000,
            checkpoint_selection='maximum validation Accuracy; unchanged original checkpoint rule; first maximum wins',
            temperature_selection='none; all V5 use original T_map=1; FixMatch T=0.5 unchanged',
            primary_metric='held-out test Macro-F1',sd='sample SD across 5 training seeds, ddof=1',
            paired='same scenario, same train_seed: V2 minus V1/V3/V5; report raw differences and mean/sample SD',
            secondary_metrics=['Accuracy','C5-C8 mean Recall','C5-C8 mean class F1','per-class P/R/F1','distribution errors'],
            test_scope='internal validation only: same previously studied test split and recordings; not external confirmation',
            true_unlabeled_labels='offline distribution-error analysis only; never passed to DDDE or selection',
            model_changes='none: original MFF, DEA, estimators and mappings unchanged; separate split/train RNG only'))
    m['protocol']['sampler_seed']='explicit train_seed + epoch; old configs without train_seed retain legacy epoch-only sampler'
    m['protocol']['historical_seed2_note']='Frozen data and initialization match old seed2, but explicit sampler_seed=2 changes the old epoch-only ordering; all 80 runs are fresh.'
    atomic_json(batch/'manifest.json',m)
    atomic_json(batch/'source_sha256.json',{str(p.relative_to(code)):digest(p) for p in code.rglob('*.py')})
    atomic_json(batch/'config_sha256.json',{t['name']:digest(t['config']) for t in tasks})
    print('Prepared',batch,'80 tasks, 40 per GPU; frozen split seed=2',flush=True)


def evaluate(batch,name):
    import torch
    from sklearn.metrics import precision_recall_fscore_support,accuracy_score
    from verify_bacn_ablation import make_algorithm
    from eval_confusion_matrix_new import get_eval_logits
    from scripts.signal_metrics import write_signal_metrics
    from analyze_ddde_mechanisms import errors
    torch.set_num_threads(4)
    m=json.loads((batch/'manifest.json').read_text());t=next(t for t in m['tasks'] if t['name']==name)
    cfg=yaml.safe_load(Path(t['config']).read_text());out=Path(t['output'])
    algorithm=make_algorithm(t['config'],'metaexpert_v1v5',False,False,batch/'evaluation_constructor')
    with np.load(m['splits'][t['scenario']]['file'],allow_pickle=False) as saved:
        for key,dataset in zip(KEYS,('train_lb','train_ulb','eval','test')):
            np.testing.assert_array_equal(algorithm.dataset_dict[dataset].sample_indices,saved[key])
    checkpoint=torch.load(out/'model_best.pth',map_location='cpu',weights_only=False)
    algorithm.ema_model.load_state_dict(checkpoint['ema_model']);algorithm.restore_ddde_state(checkpoint)
    algorithm.ema_model.eval();summary={}
    for split,dataset_key,per_class in [('validation','eval',50),('test','test',100)]:
        labels=[];probabilities=[]
        with torch.no_grad():
            for data in torch.utils.data.DataLoader(algorithm.dataset_dict[dataset_key],batch_size=256,shuffle=False,num_workers=0):
                logits=get_eval_logits(algorithm.ema_model(data['x_lb'].cuda(0)))
                labels.extend(data['y_lb'].tolist());probabilities.extend(logits.softmax(1).cpu().numpy())
        labels=np.asarray(labels);probabilities=np.asarray(probabilities);pred=probabilities.argmax(1)
        assert np.array_equal(np.bincount(labels,minlength=9),[per_class]*9)
        precision,recall,f1,_=precision_recall_fscore_support(labels,pred,labels=range(9),zero_division=0)
        np.savez_compressed(out/(split+'_probabilities.npz'),y_true=labels,y_pred=pred,probabilities=probabilities,class_names=np.asarray(m['class_names']))
        if split=='validation':
            validation=dict(samples=len(labels),accuracy=float(accuracy_score(labels,pred)),macro_f1=float(f1.mean()))
            assert abs(validation['accuracy']-checkpoint['best_eval_acc'])<1e-12
            atomic_json(out/'validation_metrics.json',validation)
        else:
            summary=write_signal_metrics(out,labels,pred,9,m['class_names'])
            np.savez_compressed(out/'test_predictions.npz',y_true=labels,y_pred=pred)
            cm=np.asarray(summary['confusion_counts'])[:,:9]
            summary.update(class_precision=precision.tolist(),class_f1=f1.tolist(),
                C5_C8_macro_recall=float(recall[5:9].mean()),C5_C8_macro_f1=float(f1[5:9].mean()),
                C5_precision=float(precision[5]),C5_recall=float(recall[5]),C5_f1=float(f1[5]))
            atomic_json(out/'c5_error_destinations.json',dict(
                false_negative_destinations={f'C{c}':int(cm[5,c]) for c in range(9) if c!=5 and cm[5,c]},
                false_positive_sources={f'C{c}':int(cm[c,5]) for c in range(9) if c!=5 and cm[c,5]},
                C5_support=int(cm[5].sum()),predicted_C5=int(cm[:,5].sum())))
    if checkpoint.get('cola_two_kl') is not None:
        truth=np.asarray(m['splits'][t['scenario']]['unlabeled_counts'],dtype=float);truth/=truth.sum()
        pi=checkpoint['cola_two_kl']['pi_u'].cpu().numpy()
        l1,js=errors(pi,truth)
        history=[json.loads(line) for line in (out/'variant_history.jsonl').open()]
        records=[r for r in history if r['event']=='epoch' and r['epoch']>=19]
        active=np.array([errors(r['pi_estimated'],truth) for r in records])
        atomic_json(out/'distribution_errors_analysis_only.json',dict(
            best_checkpoint_L1=l1,best_checkpoint_JSD_nats=js,
            mean_epoch_start_L1=float(active[:,0].mean()) if len(active) else None,
            mean_epoch_start_JSD_nats=float(active[:,1].mean()) if len(active) else None,
            true_unlabeled_distribution=truth.tolist(),scope='offline only; no role in training or checkpoint selection'))
    atomic_json(out/'test_metrics.json',summary)
    atomic_json(out/'selection_audit.json',dict(
        criterion='validation Accuracy',best_iteration_one_based=checkpoint['best_it']+1,
        best_validation_accuracy=checkpoint['best_eval_acc'],
        validated_reloaded_accuracy=validation['accuracy'],train_seed=t['seed'],split_seed=2,
        fixed_split_sha256=m['splits'][t['scenario']]['sha256'],tau_restored=float(algorithm.tau_lb3)))
    print(name,'test Accuracy/Macro-F1',summary['accuracy'],summary['macro_f1'],flush=True)


def report(batch):
    m=json.loads((batch/'manifest.json').read_text());rows=[];class_rows=[];c5=[]
    for t in m['tasks']:
        state=batch/'state'/(t['name']+'.json')
        if not state.exists() or json.loads(state.read_text())['state']!='completed':continue
        out=Path(t['output']);score=json.loads((out/'test_metrics.json').read_text())
        row=dict(scenario=t['scenario'],version=t['version'],train_seed=t['seed'],split_seed=2,
                 **{k:score[k] for k in ('accuracy','macro_f1','C5_C8_macro_recall','C5_C8_macro_f1','C5_precision','C5_recall','C5_f1')})
        rows.append(row)
        for c in range(9):class_rows.append(dict(scenario=t['scenario'],version=t['version'],train_seed=t['seed'],class_id=c,
            precision=score['class_precision'][c],recall=score['class_recall'][c],f1=score['class_f1'][c]))
        errors=json.loads((out/'c5_error_destinations.json').read_text())
        for direction in ('false_negative_destinations','false_positive_sources'):
            for label,n in errors[direction].items():c5.append(dict(scenario=t['scenario'],version=t['version'],train_seed=t['seed'],direction=direction,class_label=label,count=n))
    folder=batch/'comparison';folder.mkdir(exist_ok=True)
    csv_file(folder/'per_seed_results.csv',rows);csv_file(folder/'per_class_metrics.csv',class_rows);csv_file(folder/'c5_errors.csv',c5)
    aggregate=[];paired=[];pair_summaries=[]
    keys=('accuracy','macro_f1','C5_C8_macro_recall','C5_C8_macro_f1','C5_precision','C5_recall','C5_f1')
    lines=['# 固定划分，多训练种子结果','',f'完成{len(rows)}/80。主指标Macro-F1；数值单位为%。标准差为跨训练种子的样本标准差(ddof=1)。',
        '全部80项重新训练；split_seed=2及L/U/验证/测试身份固定；train_seed=2/3/4/5/6。检查点统一按验证Accuracy选择。',
        '这是已研究数据上的内部复核，不作为跨记录或外部独立确认实验。','',
        '| 场景 | 版本 | n/5 | Macro-F1 均值±SD | Accuracy 均值±SD | C5–C8 Recall 均值±SD |','|---|---|---:|---:|---:|---:|']
    for scenario in SCENARIOS:
        for version in VERSIONS:
            group=[r for r in rows if r['scenario']==scenario and r['version']==version]
            result=dict(scenario=scenario,version=version,n=len(group))
            for k in keys:
                vals=[r[k] for r in group];result[k+'_mean']=float(np.mean(vals)) if vals else None
                result[k+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            aggregate.append(result)
            def fmt(k):
                mean,sd=result[k+'_mean'],result[k+'_sd']
                if mean is None:return '待完成'
                return f'{mean*100:.2f}±{sd*100:.2f}' if sd is not None else f'{mean*100:.2f}（仅1次）'
            lines.append(f'| {scenario} | {version} | {len(group)}/5 | {fmt("macro_f1")} | {fmt("accuracy")} | {fmt("C5_C8_macro_recall")} |')
        for baseline in ('V1','V3','V5'):
            for seed in SEEDS:
                a=next((r for r in rows if r['scenario']==scenario and r['version']=='V2' and r['train_seed']==seed),None)
                b=next((r for r in rows if r['scenario']==scenario and r['version']==baseline and r['train_seed']==seed),None)
                if a is not None and b is not None:
                    paired.append(dict(scenario=scenario,comparison='V2-'+baseline,train_seed=seed,
                                       **{k+'_difference_pp':(a[k]-b[k])*100 for k in keys}))
            group=[r for r in paired if r['scenario']==scenario and r['comparison']=='V2-'+baseline]
            if group:
                vals=np.array([r['macro_f1_difference_pp'] for r in group])
                pair_summaries.append(dict(scenario=scenario,comparison='V2-'+baseline,n=len(group),
                    macro_f1_difference_mean_pp=float(vals.mean()),macro_f1_difference_sd_pp=float(vals.std(ddof=1)) if len(vals)>1 else None,
                    wins=int((vals>1e-10).sum()),ties=int((np.abs(vals)<=1e-10).sum()),losses=int((vals < -1e-10).sum())))
    csv_file(folder/'aggregate.csv',aggregate);csv_file(folder/'paired_seed_differences.csv',paired);csv_file(folder/'paired_summary.csv',pair_summaries)
    lines+=['','## 同种子配对差异','','每行比较同一场景、同一训练种子。禁止将场景或epoch当成独立种子重复。','',
        '| 场景 | 比较 | 配对数 | ΔMacro-F1均值±SD (pp) | 胜/平/负 |','|---|---|---:|---:|---:|']
    for r in pair_summaries:
        sd=r['macro_f1_difference_sd_pp'];value=f'{r["macro_f1_difference_mean_pp"]:.2f}'+(f'±{sd:.2f}' if sd is not None else '（仅1对）')
        lines.append(f'| {r["scenario"]} | {r["comparison"]} | {r["n"]}/5 | {value} | {r["wins"]}/{r["ties"]}/{r["losses"]} |')
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return len(rows)


def run(batch,worker=None):
    if worker is not None:
        import run_signal_baselines as queue
        return queue.worker(batch,worker)
    from verify_multiseed_fixed import verify_inputs
    m=verify_inputs(batch)
    if not m['smoke']:
        assert json.loads((batch/'verification.json').read_text())['status']=='passed'
        assert json.loads((batch/'gpu_audit.json').read_text())['status']=='passed'
        assert json.loads((batch/'smoke_audit.json').read_text())['status']=='passed'
    import run_signal_baselines as queue
    old_report=queue.report
    def combined(path):
        n=old_report(path);report(path);return n
    queue.report=combined
    # Worker processes must start this entry so their imports use the frozen code.
    original=queue.__file__;queue.__file__=str(Path(__file__))
    sys.argv=[str(Path(__file__)),str(batch)]
    try:return queue.main()
    finally:queue.__file__=original


if __name__=='__main__':
    # run_signal_baselines worker dispatch supplies BATCH --worker N.
    if len(sys.argv)>1 and sys.argv[1] not in ('prepare','evaluate','report','run'):
        sys.argv.insert(1,'run')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','evaluate','report','run'])
    parser.add_argument('batch',type=Path);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--task');parser.add_argument('--worker',type=int,choices=[0,1])
    a=parser.parse_args();batch=a.batch.resolve()
    if a.action=='prepare':prepare(batch,a.smoke)
    elif a.action=='evaluate':evaluate(batch,a.task)
    elif a.action=='report':print('Completed',report(batch))
    else:sys.exit(run(batch,a.worker))
