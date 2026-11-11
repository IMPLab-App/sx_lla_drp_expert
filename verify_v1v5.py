"""Exact data/protocol checks and per-task diagnostics for all five ablations."""
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

from prepare_v1v5 import SCENARIOS,VERSIONS,digest
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal
from semilearn.imb_algorithms.metaexpert.five_variants import SPECS


def verify_inputs(batch):
    m=json.loads((batch/'manifest.json').read_text());code=Path(m['code_snapshot'])
    for name,h in json.loads((batch/'source_sha256.json').read_text()).items():assert digest(code/name)==h,name
    for name,h in m['base_model_sha256'].items():
        if name!='semilearn/imb_algorithms/metaexpert/__init__.py':assert digest(code/name)==h,name
    hashes=json.loads((batch/'config_sha256.json').read_text())
    for t in m['tasks']:assert digest(t['config'])==hashes[t['name']],t['name']
    assert digest(m['data_file'])==m['data_sha256'];return m


def check_task(batch,name):
    m=json.loads((batch/'manifest.json').read_text());t=next(t for t in m['tasks'] if t['name']==name)
    cfg=yaml.safe_load(Path(t['config']).read_text());out=Path(t['output']);spec=SPECS[t['version']]
    expected=dict(version=t['version'],**spec,tau_max=4.,est_epoch=cfg['est_epoch'],ema_u=.9,dea=True)
    c=torch.load(out/'model_best.pth',map_location='cpu',weights_only=False)
    assert c['five_variant_settings']==expected
    assert any(k.startswith('lin1.') for k in c['model']) and any(k.startswith('predict.') for k in c['model'])
    if spec['dynamic']:assert 'cola_two_kl' in c and 0<=float(c['cola_two_kl']['tau_dyn'])<=4
    else:assert 'cola_two_kl' not in c
    for key in ('model','ema_model')+(('cola_two_kl',) if spec['dynamic'] else ()):
        assert all(not torch.is_tensor(a) or not a.is_floating_point() or torch.isfinite(a).all() for a in c[key].values())
    del c
    metrics=json.loads((out/'test_metrics.json').read_text());assert metrics['samples']==900
    from sklearn.metrics import f1_score
    with np.load(out/'test_predictions.npz') as p:
        assert len(p['y_true'])==900 and np.bincount(p['y_true'],minlength=9).tolist()==[100]*9
        assert metrics['accuracy']==float(np.mean(p['y_true']==p['y_pred']))
        assert abs(metrics['macro_f1']-f1_score(p['y_true'],p['y_pred'],labels=range(9),average='macro',zero_division=0))<1e-12
    with np.load(out/'test_probabilities.npz') as p:
        a=p['probabilities'];assert a.shape==(900,9) and np.isfinite(a).all() and np.allclose(a.sum(1),1,atol=1e-6)
    history=[json.loads(x) for x in (out/'variant_history.jsonl').read_text().splitlines() if x.strip()]
    assert history and all(0<=x['tau_used']<=4 and x['version']==t['version'] for x in history)
    if not spec['dynamic']:
        assert all(x['tau_used']==spec['fixed_tau'] and x['pi_estimated'] is None and x['KL_uni'] is None and x['L1_uni'] is None for x in history)
        assert not (out/'cola_history.jsonl').exists()
    elif t['version']=='V2':assert all(x['KL_uni'] is None and x['distance']=='l1' for x in history)
    else:assert all(x['L1_uni'] is None and x['distance']=='symmetric_kl' for x in history)
    text=(out/'log.txt').read_text();assert not re.search(r':\s*[+-]?(?:nan|inf)\b',text,re.I)
    result=dict(status='passed',name=name,scenario=t['scenario'],version=t['version'],samples=900,
        settings=expected,accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def verify(batch,smoke_batch=None):
    m=verify_inputs(batch);assert len(m['tasks'])==192 and m['methods']==list(VERSIONS)
    assert [sum(t['gpu']==g for t in m['tasks']) for g in (0,1)]==[96,96]
    assert [(t['scenario'],t['version']) for t in m['tasks']]==[(s,v) for s in SCENARIOS for v in VERSIONS]
    snapshots=batch/'split_indices';snapshots.mkdir(exist_ok=True);samples={};records=[]
    allowed={'save_name','save_dir','data_dir','imb_algorithm','five_variant','five_fixed_tau','ablation_no_ddde_kl','ablation_no_dea_aggregator'}
    if m['smoke']:allowed.update({'epoch','num_train_iter','num_eval_iter','num_log_iter','batch_size','eval_batch_size','num_workers','est_epoch','use_tensorboard'})
    for t in m['tasks']:
        cfg=yaml.safe_load(Path(t['config']).read_text());spec=SPECS[t['version']]
        assert digest(t['source_config'])==t['source_config_sha256']
        original=yaml.safe_load(Path(t['source_config']).read_text())
        assert all(cfg[k]==v for k,v in original.items() if k not in allowed)
        assert cfg['imb_algorithm']=='metaexpert_v1v5' and cfg['five_variant']==t['version']
        assert cfg['ablation_no_ddde_kl']==(not spec['dynamic']) and not cfg['ablation_no_dea_aggregator']
        assert cfg['epoch']==(3 if m['smoke'] else 100) and cfg['num_train_iter']==(6 if m['smoke'] else 20000)
        assert cfg['seed']==2
        if t['scenario'] not in samples:
            with contextlib.redirect_stdout(io.StringIO()):
                lb,ulb,val,test,counts=get_raw_signal(SimpleNamespace(**cfg),data_dir=cfg['data_dir'],return_test=True)
            assert counts['lb_count']==t['labeled_counts'] and counts['ulb_count']==t['unlabeled_counts']
            groups=[set(ds.sample_indices.tolist()) for ds in (lb,ulb,val,test)]
            assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i+1,4))
            assert np.bincount(val.targets,minlength=9).tolist()==[50]*9 and np.bincount(test.targets,minlength=9).tolist()==[100]*9
            a={k:ds.sample_indices for k,ds in zip(('labeled','unlabeled','validation','test'),(lb,ulb,val,test))}
            with np.load(t['reference_indices']) as ref:
                for k in a:np.testing.assert_array_equal(a[k],ref[k])
            samples[t['scenario']]=a
        # All samplers depend only on these fixed counts/seed, not version/model.
        assert cfg['full_train_counts']==t['full_counts'] and cfg['labeled_counts']==t['labeled_counts']
        assert t['full_counts']==original['full_train_counts'] and t['labeled_counts']==original['labeled_counts']
        np.savez_compressed(snapshots/(t['name']+'.npz'),**samples[t['scenario']])
        records.append(dict(scenario=t['scenario'],version=t['version'],settings=spec,L=sum(t['labeled_counts']),U=sum(t['unlabeled_counts']),test=900))
    result=dict(status='passed',tasks=192,records=records,all_variants_identical_source_indices=True,
        V4_fixed_tau_2_no_inactive_modules=True,sensitivity_aliases=m['sensitivity_aliases'],
        all_192_smoke_train_reload_tests_passed=False,gpu_audit_passed=False)
    if (batch/'gpu_audit.json').exists():
        g=json.loads((batch/'gpu_audit.json').read_text());assert g['status']=='passed' and g['model_sha256']==m['reference_model_sha256']
        assert len(g['modes'])==8 and len(g['V5_equivalence'])==4;result['gpu_audit_passed']=True
    if smoke_batch:
        sm=json.loads((smoke_batch/'manifest.json').read_text());assert sm['smoke'] and len(sm['tasks'])==192
        assert (smoke_batch/'status.txt').read_text().strip()=='completed'
        assert json.loads((smoke_batch/'source_sha256.json').read_text())==json.loads((batch/'source_sha256.json').read_text())
        for t in sm['tasks']:
            state=json.loads((smoke_batch/'state'/(t['name']+'.json')).read_text());assert state['state']=='completed'
            check_task(smoke_batch,t['name'])
        result.update(all_192_smoke_train_reload_tests_passed=True,smoke_batch=str(smoke_batch))
    (batch/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'PASS 192 configurations/source identities; GPU={result["gpu_audit_passed"]}; smoke={bool(smoke_batch)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--smoke-batch',type=Path);p.add_argument('--check-task')
    a=p.parse_args();batch=a.batch.resolve()
    if a.check_task:print(json.dumps(check_task(batch,a.check_task)),flush=True)
    else:verify(batch,a.smoke_batch.resolve() if a.smoke_batch else None)
