"""Frozen r12 replays and temperature search; select using validation only."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import yaml


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare(batch, smoke=False):
    repo = Path(__file__).resolve().parent
    reference = repo/'logs/v1v5_192_20261108'
    manifest = json.loads((reference/'manifest.json').read_text())
    old = next(t for t in manifest['tasks'] if t['scenario']=='r12' and t['version']=='V5')
    batch.mkdir(parents=True, exist_ok=False)
    code = batch/'code'
    shutil.copytree(reference/'code', code, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    overlays = ['semilearn/imb_algorithms/metaexpert/__init__.py',
                'semilearn/imb_algorithms/metaexpert/temperature_audit.py',
                'r12_temperature_experiment.py', 'analyze_r12_ddde.py', 'verify_r12_temperature.py']
    for name in overlays:
        shutil.copy2(repo/name, code/name)
    (batch/'configs').mkdir()
    # Input data and all previous checkpoints remain read-only.
    tasks=[]
    specifications=[('V5',1.,0),('V2',1.,1),('V5',.5,0),('V5',.25,1),('V5',2.,0),('V5',4.,1)]
    for version, temperature, gpu in specifications:
        cfg=yaml.safe_load(Path(old['config']).read_text())
        name=f'r12_{version}_mapT{temperature:g}'.replace('.','p')
        cfg.update(imb_algorithm='metaexpert_temperature_audit', five_variant=version,
            map_temperature=temperature, save_dir=str(batch/'models'),save_name=name,
            resume=False,load_path=None,overwrite=False)
        if smoke:
            cfg.update(epoch=3,num_train_iter=6,num_eval_iter=2,num_log_iter=1,
                batch_size=8,num_workers=0,est_epoch=0,use_tensorboard=False)
        path=batch/'configs'/(name+'.yaml')
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        tasks.append(dict(name=name,version=version,temperature=temperature,gpu=gpu,
            config=str(path),output=str(batch/'models'/name),config_sha256=digest(path)))
    protocol=dict(scenario='r12', seed=2, epochs=3 if smoke else 100,
        iterations=6 if smoke else 20000, mapping_temperatures=[.25,.5,1.,2.,4.],
        selection='maximum validation Macro-F1, then validation Accuracy, then closest log2(T) to 0, then smaller T',
        checkpoint_selection='unchanged original best validation Accuracy checkpoint',
        test_policy='No candidate test evaluation; test only the validation-selected V5 after selection.json is frozen',
        U_truth='analysis only; never passed to controller or selection',
        diagnostic_scope='Original training minibatches, repeated sample occurrences, per-minibatch effective ranks',
        historical_missing='Old normalized distributions cannot recover original pseudo-label counts/ranks; V2/T1 and V5/T1 are explicitly new instrumented replays',
        split='Same original seed=2 split: 350 train pool / 50 validation / 100 test per class; no record-level independence',
        temperature_scope='Only double-reference KL-to-tau Softmax; FixMatch T=0.5 unchanged')
    m=dict(created=now(),smoke=smoke,reference_batch=str(reference),
        reference_indices=old['reference_indices'], data_file=manifest['data_file'],
        data_sha256=manifest['data_sha256'], code_snapshot=str(code), tasks=tasks,
        source_sha256={str(p.relative_to(code)):digest(p) for p in code.rglob('*.py')},protocol=protocol)
    write_json(batch/'manifest.json',m)
    write_json(batch/'status.json',dict(state='prepared',time=now(),completed=0,total=len(tasks)))
    print('Prepared',batch,flush=True)


def evaluate(batch, task_name, split):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from verify_bacn_ablation import make_algorithm
    from eval_confusion_matrix_new import get_eval_logits
    torch.set_num_threads(4)
    m=json.loads((batch/'manifest.json').read_text())
    task=next(t for t in m['tasks'] if t['name']==task_name)
    assert digest(task['config'])==task['config_sha256']
    if split=='test':
        selection=json.loads((batch/'selection.json').read_text())
        assert selection['selected_task']==task_name
        assert selection['checkpoint_sha256']==digest(Path(task['output'])/'model_best.pth')
    a=make_algorithm(task['config'],'metaexpert_temperature_audit',False,False,batch/'validation_constructor')
    assert 'test' not in a.loader_dict
    with np.load(m['reference_indices'],allow_pickle=False) as indices:
        # Historical audit keys are original NPZ identities, not local offsets.
        for dataset_key, candidates in [('train_lb',('labeled','train_lb','lb')),('train_ulb',('unlabeled','train_ulb','ulb')),('eval',('eval','validation','val')),('test',('test',))]:
            key=next((k for k in candidates if k in indices.files),None)
            if key is None:
                raise ValueError(f'Missing split identity key for {dataset_key}: {indices.files}')
            assert np.array_equal(a.dataset_dict[dataset_key].sample_indices,indices[key]),dataset_key
    out=Path(task['output'])
    checkpoint=torch.load(out/'model_best.pth',map_location='cpu',weights_only=False)
    a.ema_model.load_state_dict(checkpoint['ema_model'])
    a.restore_ddde_state(checkpoint)
    a.ema_model.eval()
    dataset=a.dataset_dict['eval' if split=='validation' else 'test']
    labels, probabilities=[],[]
    with torch.no_grad():
        for data in torch.utils.data.DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0):
            logits=get_eval_logits(a.ema_model(data['x_lb'].cuda(0)))
            labels.extend(data['y_lb'].tolist())
            probabilities.extend(logits.softmax(1).cpu().numpy())
    labels=np.asarray(labels);probabilities=np.asarray(probabilities)
    assert len(labels)==(450 if split=='validation' else 900)
    assert np.array_equal(np.bincount(labels,minlength=9),[50 if split=='validation' else 100]*9)
    predictions=probabilities.argmax(1)
    result=dict(split=split,task=task_name,temperature=task['temperature'],version=task['version'],
        samples=len(labels),accuracy=float(accuracy_score(labels,predictions)),
        macro_f1=float(f1_score(labels,predictions,average='macro')),
        checkpoint_sha256=digest(out/'model_best.pth'),best_iteration=int(checkpoint['best_it']),
        checkpoint_epoch=int(checkpoint['epoch']),tau_restored=float(a.tau_lb3),time=now())
    np.savez_compressed(out/(split+'_probabilities.npz'),y_true=labels,y_pred=predictions,probabilities=probabilities)
    write_json(out/(split+'_metrics.json'),result)
    print(json.dumps(result),flush=True)


def export_epoch_diagnostics(batch):
    from analyze_r12_ddde import write_csv
    m=json.loads((batch/'manifest.json').read_text())
    cfg=yaml.safe_load(Path(m['tasks'][0]['config']).read_text())
    truth_counts=np.asarray(cfg['full_train_counts'])-np.asarray(cfg['labeled_counts'])
    truth=truth_counts/truth_counts.sum()
    distances,classes=[] ,[]
    for t in m['tasks']:
        path=Path(t['output'])/'epoch_diagnostics.jsonl'
        if not path.exists():
            continue
        for row in [json.loads(line) for line in path.open()]:
            distances.append(dict(task=t['name'],version=t['version'],temperature=t['temperature'],
                epoch=row['epoch'],tau_used=row['tau_used'],valid_batches=row['valid_batches'],
                KL_uni=row['KL_uni_mean'],KL_rev=row['KL_rev_mean'],
                L1_uni=row['L1_uni_mean'],L1_rev=row['L1_rev_mean'],
                mapping_window_uni=row['map_distance_window'][0],mapping_window_rev=row['map_distance_window'][1]))
            for c in range(9):
                classes.append(dict(task=t['name'],version=t['version'],temperature=t['temperature'],epoch=row['epoch'],
                    class_id=c,true_unlabeled_count=int(truth_counts[c]),true_unlabeled_probability=float(truth[c]),
                    collecting=row['collecting'],high_confidence_pseudo_occurrences=row['high_confidence_pseudo_counts'][c] if row['collecting'] else None,
                    effective_rank_mean=row['rank_mean'][c] if row['rank_mean'] is not None else None,
                    effective_rank_sum=row['rank_sum'][c] if row['collecting'] else None,
                    absent_pseudo_class_batches=row['absent_pseudo_class_batches'][c] if row['collecting'] else None,
                    pi_raw_mean=row['pi_raw_mean'][c] if row['pi_raw_mean'] is not None else None,
                    pi_ema_mean=row['pi_ema_mean'][c] if row['pi_ema_mean'] is not None else None,
                    pi_ema_end=row['pi_ema_end'][c],ratio_to_true_at_epoch_end=row['pi_ema_end'][c]/truth[c]))
    if distances:
        write_csv(batch/'analysis/epoch_distances_tau.csv',distances)
        write_csv(batch/'analysis/epoch_class_diagnostics.csv',classes)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4.5))
        for task in m['tasks']:
            selected=[r for r in distances if r['task']==task['name']]
            if selected:
                ax.plot([r['epoch'] for r in selected],[r['tau_used'] for r in selected],label=task['name'])
        ax.set(xlabel='Epoch (zero-based)',ylabel='τ used',ylim=(0,4.1));ax.legend(fontsize=8,ncol=2)
        fig.tight_layout()
        for ext in ('png','svg','pdf'):
            fig.savefig(batch/'analysis'/('temperature_tau_curves.'+ext),dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        for ax,version in zip(axes,('V2','V5')):
            selected=[r for r in distances if r['version']==version and r['temperature']==1]
            for key in ('KL_uni','KL_rev','L1_uni','L1_rev'):
                ax.plot([r['epoch'] for r in selected],[np.nan if r[key] is None else r[key] for r in selected],label=key)
            ax.set(title=version+' T=1 replay',xlabel='Epoch (zero-based)',ylabel='Full-epoch mean distance');ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ('png','svg','pdf'):
            fig.savefig(batch/'analysis'/('replay_four_distances.'+ext),dpi=180)
        plt.close(fig)
    return distances


def report(batch):
    m=json.loads((batch/'manifest.json').read_text())
    lines=['# r12 DDDE诊断与KL映射温度实验','','仅使用验证集选温度；真实U类别标签仅离线分析。',
           '所有候选保持原始seed=2、数据划分、训练轮数及超参数；只改变KL映射温度。V2是原L1对照。','',
           '| 任务 | T | 验证Accuracy (%) | 验证Macro-F1 (%) |','|---|---:|---:|---:|']
    for t in m['tasks']:
        p=Path(t['output'])/'validation_metrics.json'
        if p.exists():
            v=json.loads(p.read_text());cells=[f'{v[k]*100:.2f}' for k in ('accuracy','macro_f1')]
        else:
            cells=['待完成','待完成']
        lines.append(f'| {t["name"]} | {t["temperature"]} | '+ ' | '.join(cells)+' |')
    selection=batch/'selection.json'
    if selection.exists():
        s=json.loads(selection.read_text())
        lines+=['',f'验证集选定：{s["selected_task"]}，T={s["temperature"]}。选择文件在测试前写入。']
        test=Path(s['output'])/'test_metrics.json'
        if test.exists():
            v=json.loads(test.read_text())
            lines += [f'选定模型测试Accuracy / Macro-F1：{v["accuracy"]*100:.2f} / {v["macro_f1"]*100:.2f}%。']
    lines += ['', 'epoch_class_diagnostics.csv中的高置信计数是训练采样出现次数，含重复抽样；Effective Rank是原逐批有效秩的均值/总和。预热期未采集，保留为空。',
              'epoch_distances_tau.csv的当轮距离均值用于下一轮映射；mapping_window是产生当前tau的上一轮距离均值。','']
    (batch/'RESULTS.md').write_text('\n'.join(lines))
    export_epoch_diagnostics(batch)


def run(batch):
    m=json.loads((batch/'manifest.json').read_text())
    code=Path(m['code_snapshot'])
    for name, expected in m['source_sha256'].items():
        assert digest(code/name)==expected,name
    assert digest(m['data_file'])==m['data_sha256']
    if not m['smoke']:
        assert json.loads((batch/'gpu_equivalence_audit.json').read_text())['status']=='passed'
        assert json.loads((batch/'smoke_audit.json').read_text())['status']=='passed'
    with (batch/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(batch/'status.json',dict(state='running',time=now(),total=len(m['tasks'])))
        def worker(gpu):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            for task in [t for t in m['tasks'] if t['gpu']==gpu]:
                state=batch/'state'/(task['name']+'.json')
                if state.exists() and json.loads(state.read_text())['state']=='completed':
                    continue
                for attempt in (1,2):
                    write_json(state,dict(state='running',gpu=gpu,attempt=attempt,time=now()))
                    try:
                        log=batch/'gpu_logs'/f'{task["name"]}_attempt{attempt}.log'
                        log.parent.mkdir(exist_ok=True)
                        with log.open('w') as stream:
                            subprocess.run([sys.executable,str(code/'train.py'),'--c',task['config']],cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                            subprocess.run([sys.executable,str(code/'r12_temperature_experiment.py'),'evaluate',str(batch),'--task',task['name'],'--split','validation'],cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                        rows=[json.loads(s) for s in (Path(task['output'])/'epoch_diagnostics.jsonl').open()]
                        assert len(rows)==m['protocol']['epochs']
                        assert not (Path(task['output'])/'test_metrics.json').exists()
                        write_json(state,dict(state='completed',gpu=gpu,attempt=attempt,time=now()))
                        print('Completed',task['name'],flush=True)
                        break
                    except Exception as error:
                        with (batch/f'issues_gpu{gpu}.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(task=task['name'],attempt=attempt,error=repr(error),time=now()))+'\n')
                        out=Path(task['output'])
                        if out.exists():
                            archived=batch/'failed_attempts'/f'{task["name"]}_{attempt}'
                            archived.parent.mkdir(exist_ok=True)
                            shutil.move(str(out),archived)
                        if attempt==2:
                            write_json(state,dict(state='failed',time=now(),error=repr(error)))
                            raise
            return gpu
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(worker,(0,1)))
            report(batch)
            candidates=[]
            for t in m['tasks']:
                if t['version']=='V5':
                    metric=json.loads((Path(t['output'])/'validation_metrics.json').read_text())
                    candidates.append(dict(t,validation=metric))
            chosen=max(candidates,key=lambda t:(t['validation']['macro_f1'],t['validation']['accuracy'],
                                               -abs(math.log2(t['temperature'])),-t['temperature']))
            selection=dict(selected_task=chosen['name'],temperature=chosen['temperature'],output=chosen['output'],
                checkpoint_sha256=chosen['validation']['checkpoint_sha256'],validation=chosen['validation'],
                rule=m['protocol']['selection'],frozen_before_test=now())
            write_json(batch/'selection.json',selection)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (batch/'selected_test.log').open('w') as stream:
                subprocess.run([sys.executable,str(code/'r12_temperature_experiment.py'),'evaluate',str(batch),
                    '--task',chosen['name'],'--split','test'],cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            report(batch)
            write_json(batch/'status.json',dict(state='completed',completed=len(m['tasks']),total=len(m['tasks']),time=now(),selected=chosen['name']))
        except Exception as error:
            report(batch)
            write_json(batch/'status.json',dict(state='failed',time=now(),error=repr(error)))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','evaluate','report'])
    parser.add_argument('batch',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--task');parser.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args();batch=args.batch.resolve()
    if args.action=='prepare':prepare(batch,args.smoke)
    elif args.action=='evaluate':evaluate(batch,args.task,args.split)
    elif args.action=='report':report(batch)
    else:run(batch)
