"""Audit immutable identities, actual seed separation and original model logic."""
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from multiseed_fixed import SCENARIOS,VERSIONS,SEEDS,KEYS,atomic_json,digest
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal


def verify_inputs(batch):
    m=json.loads((batch/'manifest.json').read_text());code=Path(m['code_snapshot'])
    for name,h in json.loads((batch/'source_sha256.json').read_text()).items():assert digest(code/name)==h,name
    for name,h in m['model_sha256'].items():assert digest(code/name)==h,name
    for name,h in json.loads((batch/'config_sha256.json').read_text()).items():
        assert digest(batch/'configs'/(name+'.yaml'))==h,name
    assert digest(m['data_file'])==m['data_sha256']
    for group in m['splits'].values():assert digest(group['file'])==group['sha256']
    return m


def data_audit(batch):
    m=verify_inputs(batch)
    assert len(m['tasks'])==80 and [sum(t['gpu']==g for t in m['tasks']) for g in (0,1)]==[40,40]
    original=json.loads((Path(m['reference_batch'])/'manifest.json').read_text())
    records=[];snapshots=batch/'split_indices';snapshots.mkdir(exist_ok=True)
    for task in m['tasks']:
        cfg=yaml.safe_load(Path(task['config']).read_text())
        assert cfg['seed']==cfg['train_seed']==task['seed'] in SEEDS
        assert cfg['split_seed']==2 and cfg['fixed_split_file']==m['splits'][task['scenario']]['file']
        assert cfg['epoch']==(3 if m['smoke'] else 100) and cfg['num_train_iter']==(6 if m['smoke'] else 20000)
        reference=next(t for t in original['tasks'] if t['scenario']==task['scenario'] and t['version']==task['version'])
        baseline=yaml.safe_load(Path(reference['config']).read_text())
        allowed={'seed','train_seed','split_seed','fixed_split_file','fixed_split_sha256','data_dir','save_dir','save_name','resume','load_path','overwrite'}
        if m['smoke']:allowed.update(('epoch','num_train_iter','num_eval_iter','num_log_iter','batch_size','num_workers','est_epoch','use_tensorboard'))
        for key in set(cfg)|set(baseline):
            if key not in allowed:assert cfg.get(key)==baseline.get(key),(task['name'],key)
        random.seed(100+task['seed']);np.random.seed(200+task['seed']);torch.manual_seed(300+task['seed'])
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        with contextlib.redirect_stdout(io.StringIO()):
            *datasets,counts=get_raw_signal(SimpleNamespace(**cfg),data_dir=cfg['data_dir'],return_test=True)
        assert random.getstate()==rng[0]
        state=np.random.get_state();assert state[0]==rng[1][0] and np.array_equal(state[1],rng[1][1]) and state[2:]==rng[1][2:]
        assert torch.equal(torch.get_rng_state(),rng[2])
        with np.load(m['splits'][task['scenario']]['file'],allow_pickle=False) as frozen:
            for key,ds in zip(KEYS,datasets):np.testing.assert_array_equal(ds.sample_indices,frozen[key])
        group=m['splits'][task['scenario']]
        assert counts['lb_count']==group['labeled_counts'] and counts['ulb_count']==group['unlabeled_counts']
        ids={key:ds.sample_indices for key,ds in zip(KEYS,datasets)}
        np.savez_compressed(snapshots/(task['name']+'.npz'),**ids)
        # Explicit backward compatibility: old seed=2 and newly frozen seed=2.
        if task['seed']==2:
            legacy=dict(cfg)
            for key in ('train_seed','split_seed','fixed_split_file','fixed_split_sha256'):legacy.pop(key,None)
            with contextlib.redirect_stdout(io.StringIO()):
                *old,_=get_raw_signal(SimpleNamespace(**legacy),data_dir=cfg['data_dir'],return_test=True)
            for a,b in zip(old,datasets):
                np.testing.assert_array_equal(a.sample_indices,b.sample_indices)
                np.testing.assert_array_equal(a.data,b.data)
                assert (a.weak_transform.mean,a.weak_transform.std)==(b.weak_transform.mean,b.weak_transform.std)
        records.append(dict(name=task['name'],train_seed=task['seed'],split_seed=2,split_sha256=cfg['fixed_split_sha256']))
    result=dict(status='passed',tasks=80,all_indices_frozen_and_equal=True,
        legacy_seed2_data_and_normalization_exact=True,loading_does_not_consume_training_rng=True,
        non_seed_model_hyperparameters_unchanged=True,records=records)
    atomic_json(batch/'verification.json',result)
    print('PASS all 80 identities, counts, normalization, RNG independence and legacy seed=2.',flush=True)
    return result


def check_task(batch,name):
    from verify_v1v5 import check_task as base_check
    result=base_check(batch,name)
    m=json.loads((batch/'manifest.json').read_text());t=next(t for t in m['tasks'] if t['name']==name)
    out=Path(t['output']);metrics=json.loads((out/'test_metrics.json').read_text())
    audit=json.loads((out/'selection_audit.json').read_text())
    assert audit['train_seed']==t['seed'] and audit['split_seed']==2
    assert audit['fixed_split_sha256']==m['splits'][t['scenario']]['sha256']
    assert audit['criterion']=='validation Accuracy'
    assert audit['best_validation_accuracy']==audit['validated_reloaded_accuracy']
    text=(out/'log.txt').read_text()
    assert f'train_seed={t["seed"]}' in text and 'split_seed=2' in text
    assert all(0<=metrics[k]<=1 for k in ('C5_precision','C5_recall','C5_f1','C5_C8_macro_recall'))
    result.update(train_seed=t['seed'],split_seed=2,frozen_split_sha256=audit['fixed_split_sha256'])
    atomic_json(out/'artifact_audit.json',result)
    return result


def gpu_audit(batch):
    from verify_bacn_ablation import make_algorithm,assert_state_equal,optimizer_signature
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic=True
    m=verify_inputs(batch);old_manifest=json.loads((Path(m['reference_batch'])/'manifest.json').read_text())
    records=[]
    with tempfile.TemporaryDirectory(prefix='fixed_multiseed_gpu_') as tmp:
        for scenario in ('r12','lu6'):
            for version in VERSIONS:
                task=next(t for t in m['tasks'] if t['scenario']==scenario and t['version']==version and t['seed']==2)
                ref=next(t for t in old_manifest['tasks'] if t['scenario']==scenario and t['version']==version)
                old=make_algorithm(ref['config'],'metaexpert_v1v5',False,False,Path(tmp)/'old')
                new=make_algorithm(task['config'],'metaexpert_v1v5',False,False,Path(tmp)/'new')
                assert_state_equal(old.model.state_dict(),new.model.state_dict());assert optimizer_signature(old)==optimizer_signature(new)
                lb=next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_lb'],batch_size=8)))
                ulb=next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_ulb'],batch_size=8)))
                inputs=old.process_batch(**lb,**ulb)
                if m['smoke'] and old.cola is not None:
                    old.cola.est_epoch=new.cola.est_epoch
                for epoch in (0,20):
                    if epoch==20 and new.cola is not None:
                        features=torch.randn(27,192,device='cuda');logits=torch.eye(9,device='cuda').repeat_interleave(3,0)*20
                        for a in (old,new):a.cola.begin_epoch(18);a.cola.observe(logits,features)
                    old.epoch=new.epoch=epoch;old.it=new.it=0
                    cpu,cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
                    old.optimizer.zero_grad(set_to_none=True);first,_=old.train_step(**inputs);first['loss'].backward()
                    torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
                    new.optimizer.zero_grad(set_to_none=True);second,_=new.train_step(**inputs);second['loss'].backward()
                    torch.testing.assert_close(first['loss'],second['loss'],rtol=0,atol=0)
                    for (n,p),(other,q) in zip(old.model.named_parameters(),new.model.named_parameters()):
                        assert n==other and (p.grad is None)==(q.grad is None)
                        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=n)
                    old.optimizer.step();new.optimizer.step();assert_state_equal(old.model.state_dict(),new.model.state_dict())
                    if old.cola is not None:assert_state_equal(old.cola.state_dict(),new.cola.state_dict())
                records.append(dict(scenario=scenario,version=version,seed2_loss_gradients_updates_exact=True))
                print('PASS seed2 numerical equivalence',scenario,version,flush=True)
                del old,new,inputs,lb,ulb,first,second,p,q
                gc.collect();torch.cuda.empty_cache()
        # Show that training randomness changes while actual dataset identities do not.
        weights=[];orders=[]
        for seed in SEEDS:
            t=next(t for t in m['tasks'] if t['scenario']=='r2' and t['version']=='V2' and t['seed']==seed)
            a=make_algorithm(t['config'],'metaexpert_v1v5',False,False,Path(tmp)/str(seed))
            weights.append(a.model.lin1[0].weight.detach().cpu().clone())
            sampler=iter(a.loader_dict['train_lb'].sampler);orders.append([int(next(sampler)) for _ in range(16)])
            del a;gc.collect();torch.cuda.empty_cache()
        assert all(not torch.equal(weights[i],weights[j]) and orders[i]!=orders[j] for i in range(5) for j in range(i+1,5))
    result=dict(status='passed',records=records,distinct_initial_weights_and_sampler_orders_all_5_seeds=True,
        seed2_input_loss_gradients_equivalent=True,historical_seed2_sampler_order_intentionally_changed=True)
    atomic_json(batch/'gpu_audit.json',result)
    print('PASS all 5 train seeds change initialization/sampling; frozen identities preserved.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path);parser.add_argument('--gpu-audit',action='store_true');parser.add_argument('--check-task')
    a=parser.parse_args();batch=a.batch.resolve()
    if a.gpu_audit:gpu_audit(batch)
    elif a.check_task:print(json.dumps(check_task(batch,a.check_task)))
    else:data_audit(batch)
