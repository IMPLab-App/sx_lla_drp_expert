"""Reproducible, artifact-only metrics, shot groups and DDDE/tau diagnostics."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SNAPSHOT = OUT / 'runtime_snapshot_verified'
VARIANTS = ['full', 'no_mff', 'no_dea_aggregator', 'no_ddde_kl']
GROUPS = {'head_3': [0,1,2], 'middle_3': [3,4,5], 'tail_3': [6,7,8], 'route_head': [0], 'route_tail': list(range(1,9))}


def write_csv(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def data_splits(labels, seed=2):
    rng=np.random.RandomState(seed)
    result={name:[] for name in ('train','eval','test')}
    for k in range(9):
        indices=np.flatnonzero(labels==k).copy();rng.shuffle(indices)
        a,b=int(len(indices)*.7),int(len(indices)*.7)+int(len(indices)*.1)
        for name,subset in zip(result,(indices[:a],indices[a:b],indices[b:])):
            result[name].extend(subset)
    for name in result:
        result[name]=np.asarray(result[name]);rng.shuffle(result[name])
    return result


def numerical_module():
    path=SNAPSHOT/'semilearn/imb_algorithms/metaexpert/cola_ddde.py'
    spec=importlib.util.spec_from_file_location('runtime_cola_ddde',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def metrics_from_matrix(counts):
    y=np.repeat(np.arange(len(counts)),counts.sum(1))
    pred=np.concatenate([np.repeat(np.arange(len(counts)),row) for row in counts])
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=np.arange(9),zero_division=0)
    result=dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_precision=float(p.mean()),macro_recall=float(r.mean()),macro_f1=float(f.mean()),
        weighted_f1=float(np.average(f,weights=s)),micro_f1=float(accuracy_score(y,pred)),
        gmean_recall=float(np.prod(r)**(1/9)),mcc=float(matthews_corrcoef(y,pred)),
        kappa=float(cohen_kappa_score(y,pred)),worst_class_recall=float(r.min()),errors=int((y!=pred).sum()))
    for name,indices in GROUPS.items():
        result[name+'_accuracy']=float(counts[indices,indices].sum()/counts[indices].sum())
        result[name+'_macro_f1']=float(f[indices].mean())
    return result,p,r,f,s


def main():
    rows=list(csv.DictReader((ROOT/'results/cola_two_kl_24/results.csv').open()))
    manifest=json.loads((ROOT/'config/cola_two_kl_24/manifest.json').read_text())
    with np.load(ROOT/'data/raw_signal_dataset_G_1024.npz',allow_pickle=False) as data:
        labels=data['labels'];names=data['class_names'].tolist()
    split=data_splits(labels);train_labels=labels[split['train']]
    hold_support=np.bincount(labels[split['test']],minlength=9)
    assert np.array_equal(hold_support,np.full(9,100))
    ddde=numerical_module()
    overall=[];classes=[];tau_rows=[];dist_rows=[];epoch_rows=[];metadata={}
    for task in manifest['tasks']:
        index=int(task['run_name'].split('_')[1]);variant=task['variant']
        source=next(r for r in rows if r['variant']==variant and r['run_name']==task['run_name'])
        folder=ROOT/task['model_dir'];evaluation=json.loads((folder/'evaluation/evaluation.json').read_text())
        # Obtain true training priors from the dataset indices actually cached.
        ratio=50 if index in (1,3,5) else 100
        ulb_num=315 if index in (3,4) else 250
        ulb_ratio='1.0' if index in (3,4) else str(ratio if index in (1,2) else -ratio)
        suffix=f'100_{ratio}_{ulb_num}_{ulb_ratio}_noise_0.0_seed_2_idx.npy'
        lb=np.load(ROOT/'data/raw_signal/labeled_idx'/('lb_labels_'+suffix))
        ulb=np.load(ROOT/'data/raw_signal/labeled_idx'/('ulb_labels_'+suffix))
        lb_counts=np.bincount(train_labels[lb],minlength=9)
        ulb_counts=np.bincount(train_labels[ulb],minlength=9)
        if index not in metadata:
            metadata[index]=dict(run_name=task['run_name'],labeled_counts=lb_counts.tolist(),
                unlabeled_counts=ulb_counts.tolist(),unlabeled_indices=str(ROOT/'data/raw_signal/labeled_idx'/('ulb_labels_'+suffix)),
                scenario='matched' if index<3 else 'uniform' if index<5 else 'reverse')
        for split_name,support in [('heldout',hold_support),('balanced',hold_support),('imbalanced',lb_counts)]:
            cm=np.array(evaluation[split_name]['confusion_matrix'])
            counts=np.rint(cm*support[:,None]).astype(int)
            assert np.array_equal(counts.sum(1),support)
            metrics,p,r,f,s=metrics_from_matrix(counts)
            for key in ('macro_precision','macro_recall','macro_f1','balanced_accuracy'):
                assert abs(metrics[key]-evaluation[split_name][key])<1e-10,(variant,index,split_name,key)
            assert abs(metrics['accuracy']-evaluation[split_name]['top1_accuracy'])<1e-10
            row=dict(variant=variant,run=index,scenario=metadata[index]['scenario'],split=split_name,**metrics,
                best_epoch=evaluation['best_epoch'],best_iteration=evaluation['best_iteration'])
            overall.append(row)
            if split_name=='heldout':
                for k in range(9):
                    tp=int(counts[k,k]);fn=int(s[k]-tp);fp=int(counts[:,k].sum()-tp)
                    classes.append(dict(variant=variant,run=index,class_id=k,class_name=names[k],
                        labeled_count=int(lb_counts[k]),unlabeled_count=int(ulb_counts[k]),support=int(s[k]),
                        precision=float(p[k]),recall=float(r[k]),f1=float(f[k]),tp=tp,fn=fn,fp=fp,
                        specificity=float((counts.sum()-tp-fn-fp)/(counts.sum()-s[k]))))
        if variant=='full':
            q=ulb_counts/ulb_counts.sum()
            best=torch.load(folder/'model_best.pth',map_location='cpu',weights_only=False)
            latest=torch.load(folder/'latest_model.pth',map_location='cpu',weights_only=False)
            pi_l=best['cola_two_kl']['pi_l'];uni=torch.ones(9)/9;rev=pi_l.flip(0)
            oracle_tau=float(ddde.tau_from_kl(ddde.symmetric_kl(torch.tensor(q),uni),ddde.symmetric_kl(torch.tensor(q),rev),4))
            records=[json.loads(line) for line in (folder/'cola_history.jsonl').read_text().splitlines() if line]
            batch=[r for r in records if r['event']=='batch'];epochs=[r for r in records if r['event']=='epoch']
            best_u=np.array(best['cola_two_kl']['pi_u']);final_u=np.array(latest['cola_two_kl']['pi_u'])
            tau_rows.append(dict(run=index,scenario=metadata[index]['scenario'],best_epoch=evaluation['best_epoch'],
                best_tau=float(best['cola_two_kl']['tau_dyn']),final_tau=float(latest['cola_two_kl']['tau_dyn']),
                oracle_tau=oracle_tau,valid_batch_count=len(batch),expected_batch_count=82*200,
                tv_best=float(np.abs(best_u-q).sum()/2),tv_final=float(np.abs(final_u-q).sum()/2),
                kl_uni_final=float(latest['cola_two_kl']['last_kl'][0]),kl_rev_final=float(latest['cola_two_kl']['last_kl'][1]),
                tau_mean_after_warmup=float(np.mean([r['tau_dyn'] for r in epochs if r['epoch']>=19])),
                tau_std_after_warmup=float(np.std([r['tau_dyn'] for r in epochs if r['epoch']>=19]))))
            zero_rate=np.mean(np.asarray([r['pi_DDDE'] for r in batch])<=1e-12,axis=0)
            for k in range(9):
                dist_rows.append(dict(run=index,class_id=k,class_name=names[k],true_prior=float(q[k]),
                    best_pi_u=float(best_u[k]),final_pi_u=float(final_u[k]),ddde_zero_batch_rate=float(zero_rate[k])))
            for item in epochs:
                epoch_rows.append(dict(run=index,epoch=item['epoch'],tau=item['tau_dyn'],
                    kl_uni=item['KL_uni'],kl_rev=item['KL_rev'],tv=float(np.abs(np.array(item['pi_u'])-q).sum()/2),
                    **{f'pi_u_{k}':item['pi_u'][k] for k in range(9)}))
            del best,latest,records,batch
    means=[]
    for variant in VARIANTS:
        for split_name in ('heldout','imbalanced'):
            group=[r for r in overall if r['variant']==variant and r['split']==split_name]
            keys=[k for k in group[0] if k not in ('variant','run','scenario','split','best_epoch','best_iteration')]
            means.append(dict(variant=variant,split=split_name,**{k:float(np.mean([r[k] for r in group])) for k in keys}))
    write_csv('metrics_by_run.csv',overall);write_csv('metrics_means.csv',means)
    write_csv('per_class_metrics.csv',classes);write_csv('tau_diagnostics.csv',tau_rows)
    write_csv('distribution_diagnostics.csv',dist_rows);write_csv('epoch_history.csv',epoch_rows)
    (OUT/'data_metadata.json').write_text(json.dumps(dict(class_names=names,runs=metadata,groups=GROUPS,
        test_support=hold_support.tolist(),snapshot=str(SNAPSHOT)),ensure_ascii=False,indent=2))
    print('MEAN_HELDOUT')
    for row in means:
        if row['split']=='heldout':
            print(json.dumps({k:row[k] for k in ['variant','accuracy','macro_precision','macro_recall','macro_f1','gmean_recall','head_3_accuracy','middle_3_accuracy','tail_3_accuracy','tail_3_macro_f1','route_head_accuracy','route_tail_accuracy']},ensure_ascii=False))
    print('TAU_DIAGNOSTICS')
    for row in tau_rows:print(json.dumps(row))
    print('PER_CLASS_MEANS_FULL_FIXED')
    for k in range(9):
        a=[r for r in classes if r['variant']=='full' and r['class_id']==k]
        b=[r for r in classes if r['variant']=='no_ddde_kl' and r['class_id']==k]
        print(k,names[k],*[round(np.mean([r[key] for r in group])*100,4) for key in ['precision','recall','f1'] for group in [a,b]])


if __name__=='__main__':
    main()
