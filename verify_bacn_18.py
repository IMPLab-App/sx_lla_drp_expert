"""Audit the 54 fresh tasks using exact samples and matching prior smoke runs."""
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

from prepare_bacn_18 import SCENARIOS,VARIANTS,digest
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal


def verify_inputs(batch):
    m=json.loads((batch/'manifest.json').read_text());code=Path(m['code_snapshot'])
    for name,expected in json.loads((batch/'source_sha256.json').read_text()).items():
        assert digest(code/name)==expected,name
    for name,expected in m['parent_source_sha256'].items():assert digest(code/name)==expected,name
    configs=json.loads((batch/'config_sha256.json').read_text())
    for t in m['tasks']:assert digest(t['config'])==configs[t['name']],t['name']
    assert digest(m['data_file'])==m['data_sha256']
    assert digest(Path(m['dependency_batch'])/'manifest.json')==m['dependency_manifest_sha256']
    return m


def check_task(batch,name):
    m=json.loads((batch/'manifest.json').read_text());t=next(t for t in m['tasks'] if t['name']==name)
    cfg=yaml.safe_load(Path(t['config']).read_text());out=Path(t['output']);v=t['variant']
    ckpt=torch.load(out/'model_best.pth',map_location='cpu',weights_only=False)
    if v=='no_ddde_kl':assert 'cola_two_kl' not in ckpt
    else:
        assert ckpt['bacn_module_settings']==dict(tau_max=4.0,est_epoch=18,ema_u=0.9,no_dea=v=='no_dea')
        assert 0<=float(ckpt['cola_two_kl']['tau_dyn'])<=4
    for key in ('model','ema_model')+(() if v=='no_ddde_kl' else ('cola_two_kl',)):
        assert ckpt[key] and all(not torch.is_tensor(a) or not a.is_floating_point() or torch.isfinite(a).all() for a in ckpt[key].values())
    del ckpt
    metrics=json.loads((out/'test_metrics.json').read_text());assert metrics['samples']==900
    from sklearn.metrics import f1_score
    with np.load(out/'test_predictions.npz') as p:
        assert np.bincount(p['y_true'],minlength=9).tolist()==[100]*9
        assert metrics['accuracy']==float(np.mean(p['y_true']==p['y_pred']))
        assert abs(metrics['macro_f1']-f1_score(p['y_true'],p['y_pred'],labels=range(9),average='macro',zero_division=0))<1e-12
    with np.load(out/'test_probabilities.npz') as p:
        a=p['probabilities'];assert a.shape==(900,9) and np.isfinite(a).all()
        assert np.allclose(a.sum(1),1,atol=1e-6)
    text=(out/'log.txt').read_text();assert not re.search(r':\s*[+-]?(?:nan|inf)\b',text,re.I)
    if v=='no_dea':
        losses=re.findall(r'train/total_fuse_loss: ([\d.eE+-]+)',text)
        assert losses and all(float(a)==0 for a in losses)
    h=out/'cola_history.jsonl'
    if v=='no_ddde_kl':assert not h.exists()
    else:
        history=[json.loads(line) for line in h.read_text().splitlines() if line.strip()]
        assert history and all(np.isfinite(a['tau_dyn']) and 0<=a['tau_dyn']<=4 for a in history)
    result=dict(status='passed',name=name,variant=v,samples=900,
                accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def verify(batch):
    m=verify_inputs(batch);assert len(m['tasks'])==54 and m['methods']==list(VARIANTS)
    assert [sum(t['gpu']==g for t in m['tasks']) for g in (0,1)]==[27,27]
    for v in VARIANTS:assert [t['scenario'] for t in m['tasks'] if t['variant']==v]==SCENARIOS
    historical_gpu=Path(m['dependency_batch']).parent/'bacn_ablation_22_20261105/verification.json'
    gpu=json.loads(historical_gpu.read_text())['baseline_reuse'];code=Path(m['code_snapshot'])
    assert gpu['passed'] and len(gpu['cases'])==11 and all(a['bitwise_equal'] for a in gpu['cases'])
    assert gpu['parent_sha256']==digest(code/'semilearn/imb_algorithms/metaexpert/metaexpert.py')
    assert gpu['extension_sha256']==digest(code/'semilearn/imb_algorithms/metaexpert/bacn_ablation.py')
    for a in gpu['dynamic_variants']:
        assert a['gradients'] and a['checkpoint_restore'] and a['evaluation_tau_restore'] and 0<=a['tau']<4
    indices=batch/'split_indices';indices.mkdir(exist_ok=True)
    arrays={};pools={};labels={};records=[];smokes={}
    for t in m['tasks']:
        cfg=yaml.safe_load(Path(t['config']).read_text());v=t['variant'];scenario=t['scenario']
        assert cfg['epoch']==100 and cfg['num_train_iter']==20000 and cfg['seed']==2
        assert cfg['batch_size']==64 and cfg['uratio']==2 and cfg['est_epoch']==18
        assert not cfg['resume'] and not cfg['overwrite'] and cfg['load_path'] is None
        assert cfg['full_train_counts']==t['full_counts'] and cfg['labeled_counts']==t['labeled_counts']
        assert min(t['labeled_counts'])>=1 and max(t['full_counts'])<=350
        assert cfg['imb_algorithm']==('metaexpert' if v=='no_ddde_kl' else 'metaexpert_bacn')
        if v!='no_ddde_kl':assert not cfg['ablation_no_ddde_kl'] and cfg['ablation_no_dea_aggregator']==(v=='no_dea')
        else:assert 'ablation_no_ddde_kl' not in cfg and 'ablation_no_dea_aggregator' not in cfg
        assert digest(t['source_config'])==t['source_config_sha256']
        src=yaml.safe_load(Path(t['source_config']).read_text())
        allowed={'save_name','save_dir','data_dir','imb_algorithm','ablation_no_ddde_kl','ablation_no_dea_aggregator'}
        assert all(cfg[k]==value for k,value in src.items() if k not in allowed)
        with contextlib.redirect_stdout(io.StringIO()):
            lb,ulb,val,test,counts=get_raw_signal(SimpleNamespace(**cfg),data_dir=cfg['data_dir'],return_test=True)
        assert counts['lb_count']==t['labeled_counts'] and counts['ulb_count']==t['unlabeled_counts']
        groups=[set(ds.sample_indices.tolist()) for ds in (lb,ulb,val,test)]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i+1,4))
        assert np.bincount(val.targets,minlength=9).tolist()==[50]*9
        assert np.bincount(test.targets,minlength=9).tolist()==[100]*9
        a={key:ds.sample_indices for key,ds in zip(('labeled','unlabeled','validation','test'),(lb,ulb,val,test))}
        with np.load(t['reference_indices']) as ref:
            for key in a:np.testing.assert_array_equal(a[key],ref[key])
        if scenario in arrays:
            for key in a:np.testing.assert_array_equal(a[key],arrays[scenario][key])
        else:
            arrays[scenario]=a;pools[scenario]=groups[0]|groups[1];labels[scenario]=groups[0]
        np.savez_compressed(indices/(t['name']+'.npz'),**a)
        sb=Path(t['historical_smoke_batch'])
        if str(sb) not in smokes:
            sm=json.loads((sb/'manifest.json').read_text());assert sm['smoke']
            assert (sb/'status.txt').read_text().strip()=='completed' and sm['data_sha256']==m['data_sha256']
            for name,expected in m['reference_model_sha256'].items():
                # Fixed-tau historical runs predate opt-in registration/eval support.
                if sb.name=='bacn_11_smoke_20261105' and name in (
                    'semilearn/imb_algorithms/metaexpert/bacn_ablation.py',
                    'semilearn/imb_algorithms/metaexpert/__init__.py',
                    'eval_metaexpert_lb35.py','eval_confusion_matrix_new.py'):continue
                assert digest(Path(sm['code_snapshot'])/name)==expected,name
            smokes[str(sb)]=sm
        st=next(x for x in smokes[str(sb)]['tasks'] if x['name']==t['historical_smoke_name'])
        sc=yaml.safe_load(Path(st['config']).read_text())
        smoke_allowed=allowed|{'epoch','num_train_iter','num_eval_iter','num_log_iter','batch_size','eval_batch_size','num_workers','est_epoch','use_tensorboard'}
        assert all(sc[k]==value for k,value in cfg.items() if k not in smoke_allowed)
        assert sc['imb_algorithm']==cfg['imb_algorithm']
        if v!='no_ddde_kl':assert sc['ablation_no_dea_aggregator']==cfg['ablation_no_dea_aggregator'] and not sc['ablation_no_ddde_kl']
        state=json.loads((sb/'state'/(st['name']+'.json')).read_text());assert state['state']=='completed'
        out=Path(st['output']);assert (out/'model_best.pth').is_file()
        metrics=json.loads((out/'test_metrics.json').read_text());assert metrics['samples']==900
        with np.load(out/'test_predictions.npz') as pred:
            assert metrics['accuracy']==float(np.mean(pred['y_true']==pred['y_pred']))
        with np.load(out/'test_probabilities.npz') as prob:
            values=prob['probabilities'];assert values.shape==(900,9) and np.isfinite(values).all()
            assert np.allclose(values.sum(1),1,atol=1e-6)
        records.append(dict(scenario=scenario,variant=v,labeled=len(lb),unlabeled=len(ulb),test=900,
                            historical_smoke=str(sb),historical_smoke_name=st['name']))
    assert pools['cd1']==pools['cd2']==pools['cd3']
    assert pools['r5']==pools['cd4']==pools['cd5']==pools['cd6']
    assert labels['cd6']<labels['cd5']<labels['cd4']
    assert all(pools[f'r{i+1}']<=pools[f'r{i}'] for i in range(1,12))
    result=dict(status='passed',tasks=54,records=records,
        all_54_matching_historical_smoke_checks_passed=True,identical_variant_samples=True,
        exact_previous_sample_identities=True,cd_full_pools_and_nested_labels=True,
        same_code_gpu_audit=str(historical_gpu),fresh_formal_training=True,gpu_tasks={'0':27,'1':27})
    (batch/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS: 54 exact configurations/identities and matching historical short train/reload/test checks; no GPU launched.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    parser.add_argument('--check-task');args=parser.parse_args()
    if args.check_task:print(json.dumps(check_task(args.batch.resolve(),args.check_task)),flush=True)
    else:verify(args.batch.resolve())
