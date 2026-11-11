"""Check 39 configurations, exact sample identities and real short-run outputs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from prepare_selected13 import SCENARIOS,VARIANTS,digest
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal


def verify_inputs(batch):
    m=json.loads((batch/'manifest.json').read_text());code=Path(m['code_snapshot'])
    for filename in ('source_sha256.json',):
        for name,expected in json.loads((batch/filename).read_text()).items():assert digest(code/name)==expected,name
    for name,expected in m['parent_source_sha256'].items():assert digest(code/name)==expected,name
    config_hashes=json.loads((batch/'config_sha256.json').read_text())
    for t in m['tasks']:assert digest(t['config'])==config_hashes[t['name']],t['name']
    assert digest(m['data_file'])==m['data_sha256']
    return m


def check_task(batch,name):
    m=json.loads((batch/'manifest.json').read_text());t=next(t for t in m['tasks'] if t['name']==name)
    out=Path(t['output']);cfg=yaml.safe_load(Path(t['config']).read_text());v=t['variant']
    checkpoint=torch.load(out/'model_best.pth',map_location='cpu',weights_only=False)
    if v=='no_ddde_kl':assert 'cola_two_kl' not in checkpoint
    else:
        assert checkpoint['bacn_module_settings']==dict(tau_max=float(cfg['la_tau_lb3']),est_epoch=cfg['est_epoch'],ema_u=cfg['ema_u'],no_dea=v=='no_dea')
        assert 0<=float(checkpoint['cola_two_kl']['tau_dyn'])<=4
    for key in ('model','ema_model')+(() if v=='no_ddde_kl' else ('cola_two_kl',)):
        assert checkpoint[key] and all(not torch.is_tensor(a) or not a.is_floating_point() or torch.isfinite(a).all() for a in checkpoint[key].values()),key
    del checkpoint
    metrics=json.loads((out/'test_metrics.json').read_text());assert metrics['samples']==900
    from sklearn.metrics import f1_score
    with np.load(out/'test_predictions.npz') as p:
        assert len(p['y_true'])==900 and np.bincount(p['y_true'],minlength=9).tolist()==[100]*9
        assert metrics['accuracy']==float(np.mean(p['y_true']==p['y_pred']))
        assert abs(metrics['macro_f1']-f1_score(p['y_true'],p['y_pred'],labels=range(9),average='macro',zero_division=0))<1e-12
    with np.load(out/'test_probabilities.npz') as p:
        a=p['probabilities'];assert a.shape==(900,9) and np.isfinite(a).all()
        assert np.allclose(a.sum(1),1,atol=1e-6)
    text=(out/'log.txt').read_text();assert not re.search(r':\s*[+-]?(?:nan|inf)\b',text,re.I)
    if v=='no_dea':
        losses=re.findall(r'train/total_fuse_loss: ([\d.eE+-]+)',text)
        assert losses and all(float(a)==0 for a in losses)
    history=out/'cola_history.jsonl'
    if v=='no_ddde_kl':assert not history.exists()
    else:
        records=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        assert records and all(np.isfinite(a['tau_dyn']) and 0<=a['tau_dyn']<=4 for a in records)
    result=dict(status='passed',name=name,scenario=t['scenario'],variant=v,samples=900,
                accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def verify(batch,smoke_batch=None):
    m=verify_inputs(batch);assert len(m['tasks'])==39 and m['methods']==list(VARIANTS)
    assert [sum(t['gpu']==g for t in m['tasks']) for g in (0,1)]==[20,19]
    assert [(t['scenario'],t['variant']) for t in m['tasks']]==[(s,v) for s in SCENARIOS for v in VARIANTS]
    for i,t in enumerate(m['tasks']):assert t['gpu']==i%2
    baseline=Path(m['model_reference_batch']).parent/'bacn_ablation_22_20261105/verification.json'
    gpu=json.loads(baseline.read_text())['baseline_reuse']
    code=Path(m['code_snapshot']);assert gpu['passed']
    assert gpu['parent_sha256']==digest(code/'semilearn/imb_algorithms/metaexpert/metaexpert.py')
    assert gpu['extension_sha256']==digest(code/'semilearn/imb_algorithms/metaexpert/bacn_ablation.py')
    assert all(a['bitwise_equal'] for a in gpu['cases'])
    for a in gpu['dynamic_variants']:assert a['gradients'] and a['checkpoint_restore'] and a['evaluation_tau_restore']
    snapshots=batch/'split_indices';snapshots.mkdir(exist_ok=True)
    prior={};records=[];fixed_test=None
    allowed={'save_name','save_dir','data_dir','imb_algorithm','ablation_no_ddde_kl','ablation_no_dea_aggregator'}
    if m['smoke']:allowed.update({'epoch','num_train_iter','num_eval_iter','num_log_iter','batch_size','eval_batch_size','num_workers','est_epoch','use_tensorboard'})
    for t in m['tasks']:
        cfg=yaml.safe_load(Path(t['config']).read_text());v=t['variant']
        assert digest(t['source_config'])==t['source_config_sha256']
        source=yaml.safe_load(Path(t['source_config']).read_text())
        assert all(cfg[k]==value for k,value in source.items() if k not in allowed)
        assert cfg['epoch']==(3 if m['smoke'] else 100)
        assert cfg['num_train_iter']==(6 if m['smoke'] else 20000) and cfg['seed']==2
        assert cfg['full_train_counts']==t['full_counts'] and cfg['labeled_counts']==t['labeled_counts']
        assert cfg['imb_algorithm']==('metaexpert' if v=='no_ddde_kl' else 'metaexpert_bacn')
        if v=='no_ddde_kl':assert 'ablation_no_ddde_kl' not in cfg and 'ablation_no_dea_aggregator' not in cfg
        else:assert not cfg['ablation_no_ddde_kl'] and cfg['ablation_no_dea_aggregator']==(v=='no_dea')
        if t['scenario'] in ('lu3','lu4'):assert t['unlabeled_counts']==[130]*9 and cfg['ulb_num_labels']==1170
        with contextlib.redirect_stdout(io.StringIO()):
            lb,ulb,val,test,counts=get_raw_signal(SimpleNamespace(**cfg),data_dir=cfg['data_dir'],return_test=True)
        assert counts['lb_count']==t['labeled_counts'] and counts['ulb_count']==t['unlabeled_counts']
        assert np.bincount(val.targets,minlength=9).tolist()==[50]*9 and np.bincount(test.targets,minlength=9).tolist()==[100]*9
        groups=[set(ds.sample_indices.tolist()) for ds in (lb,ulb,val,test)]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i+1,4))
        a={key:ds.sample_indices for key,ds in zip(('labeled','unlabeled','validation','test'),(lb,ulb,val,test))}
        with np.load(t['reference_indices']) as expected:
            for key in a:np.testing.assert_array_equal(a[key],expected[key])
        if t['scenario'] in prior:
            for key in a:np.testing.assert_array_equal(a[key],prior[t['scenario']][key])
        else:prior[t['scenario']]=a
        if fixed_test is None:fixed_test=test.sample_indices.copy()
        np.testing.assert_array_equal(test.sample_indices,fixed_test)
        np.savez_compressed(snapshots/(t['name']+'.npz'),**a)
        records.append(dict(scenario=t['scenario'],variant=v,L=len(lb),U=len(ulb),test=900))
    result=dict(status='passed',tasks=39,records=records,exact_source_sample_identities=True,
        variants_identical_samples=True,uniform_U_1170_confirmed=True,
        same_code_dynamic_gpu_audit=str(baseline),all_39_smoke_train_reload_tests_passed=False)
    if smoke_batch:
        sm=json.loads((smoke_batch/'manifest.json').read_text());assert sm['smoke'] and len(sm['tasks'])==39
        assert (smoke_batch/'status.txt').read_text().strip()=='completed'
        assert json.loads((smoke_batch/'source_sha256.json').read_text())==json.loads((batch/'source_sha256.json').read_text())
        assert sm['data_sha256']==m['data_sha256']
        for t in sm['tasks']:
            state=json.loads((smoke_batch/'state'/(t['name']+'.json')).read_text())
            assert state['state']=='completed' and state['attempt']==1
            check_task(smoke_batch,t['name'])
            with np.load(smoke_batch/'split_indices'/(t['name']+'.npz')) as a,np.load(snapshots/(t['name']+'.npz')) as b:
                for key in ('labeled','unlabeled','validation','test'):np.testing.assert_array_equal(a[key],b[key])
        result.update(all_39_smoke_train_reload_tests_passed=True,smoke_batch=str(smoke_batch))
    (batch/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'PASS 39-task exact-count/source-identity audit; smoke={bool(smoke_batch)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    parser.add_argument('--smoke-batch',type=Path);parser.add_argument('--check-task')
    args=parser.parse_args();batch=args.batch.resolve()
    if args.check_task:print(json.dumps(check_task(batch,args.check_task)),flush=True)
    else:verify(batch,args.smoke_batch.resolve() if args.smoke_batch else None)
