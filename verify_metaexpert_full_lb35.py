"""Audit the Full six-case protocol, dynamic execution and held-out artifacts."""
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from prepare_metaexpert_full_lb35 import CASES, digest, expected_counts
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal,load_raw_signal_data,_stratified_split


def check_task(batch,name):
    m=json.loads((batch/'manifest.json').read_text())
    task=next(t for t in m['tasks'] if t['name']==name)
    cfg=yaml.safe_load(Path(task['config']).read_text());out=Path(task['output'])
    ckpt=torch.load(out/'model_best.pth',map_location='cpu',weights_only=False)
    settings=dict(tau_max=4.0,est_epoch=cfg['est_epoch'],ema_u=0.9,no_dea=False)
    assert ckpt['bacn_module_settings']==settings
    assert 'cola_two_kl' in ckpt and 0<=float(ckpt['cola_two_kl']['tau_dyn'])<=4
    for key in ('model','ema_model','cola_two_kl'):
        assert ckpt[key]
        assert all(not torch.is_tensor(x) or not x.is_floating_point() or torch.isfinite(x).all()
                   for x in ckpt[key].values()),key
    del ckpt
    metrics=json.loads((out/'test_metrics.json').read_text());assert metrics['samples']==900
    assert all(np.isfinite(metrics[k]) for k in ('accuracy','macro_f1','macro_recall'))
    from sklearn.metrics import f1_score
    with np.load(out/'test_predictions.npz') as p:
        assert len(p['y_true'])==900 and np.bincount(p['y_true'],minlength=9).tolist()==[100]*9
        assert metrics['accuracy']==float(np.mean(p['y_true']==p['y_pred']))
        assert abs(metrics['macro_f1']-f1_score(p['y_true'],p['y_pred'],labels=range(9),average='macro',zero_division=0))<1e-12
    with np.load(out/'test_probabilities.npz') as p:
        a=p['probabilities'];assert a.shape==(900,9) and np.isfinite(a).all()
        assert np.allclose(a.sum(1),1,atol=1e-6)
    text=(out/'log.txt').read_text()
    assert not re.search(r':\s*[+-]?(?:nan|inf)\b',text,re.I)
    hist=[json.loads(x) for x in (out/'cola_history.jsonl').read_text().splitlines() if x.strip()]
    assert hist and all(np.isfinite(x['tau_dyn']) and 0<=x['tau_dyn']<=4 for x in hist)
    result=dict(name=name,status='passed',samples=900,ddde_checkpoint=True,dea_enabled=True,
                accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def gpu_audit(batch):
    from verify_bacn_ablation import make_algorithm, assert_state_equal
    from eval_confusion_matrix_new import get_eval_logits
    torch.set_num_threads(4);torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic=True
    m=json.loads((batch/'manifest.json').read_text());records=[]
    # One real case per U-distribution, using the formal warmup and batch setup.
    with tempfile.TemporaryDirectory(prefix='full_six_audit_') as directory:
        for task in m['tasks']:
            if task['case'] not in (1,3,5):continue
            algorithm=make_algorithm(task['config'],'metaexpert_bacn',False,False,Path(directory))
            module=algorithm.model
            assert algorithm.cola is not None and hasattr(module,'predict') and hasattr(module,'lin1')
            lb=next(iter(torch.utils.data.DataLoader(algorithm.dataset_dict['train_lb'],batch_size=8)))
            ulb=next(iter(torch.utils.data.DataLoader(algorithm.dataset_dict['train_ulb'],batch_size=8)))
            inputs=algorithm.process_batch(**lb,**ulb)
            algorithm.cola.begin_epoch(algorithm.est_epoch)
            algorithm.cola.observe(torch.eye(9,device='cuda').repeat_interleave(3,0)*20,
                                   torch.randn(27,192,device='cuda'))
            algorithm.epoch,algorithm.it=algorithm.est_epoch+2,0
            before=module.predict[0].weight.detach().clone();captured=[]
            hook=module.register_forward_hook(lambda mod,args,outputs:captured.append(outputs))
            with patch.object(algorithm.cola,'observe',wraps=algorithm.cola.observe) as spy:
                result,logs=algorithm.train_step(**inputs)
                assert spy.call_count==1
                logits,features=spy.call_args.args
                torch.testing.assert_close(logits,captured[0]['aux_logits2'][8:16])
                torch.testing.assert_close(features,captured[0]['feat'][8:16])
            hook.remove()
            assert 0<=algorithm.tau_lb3<4 and float(module.tau3)==algorithm.tau_lb3
            assert torch.isfinite(result['loss'])
            algorithm.optimizer.zero_grad(set_to_none=True);result['loss'].backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
            for p in (module.predict[0].weight,module.lin1[0].weight,
                      module.backbone.classifier.weight,module.backbone.aux_classifier2.weight):
                assert p.grad is not None and p.grad.abs().sum()>0
            algorithm.optimizer.step();assert not torch.equal(before,module.predict[0].weight)
            dest=Path(directory)/str(task['case']);algorithm.save_model('check.pth',str(dest))
            saved=torch.load(dest/'check.pth',map_location='cpu',weights_only=False)
            algorithm.cola.pi_u.zero_();algorithm.tau_lb3=0
            algorithm.load_model(str(dest/'check.pth'))
            assert_state_equal({k:v.cpu() for k,v in algorithm.cola.state_dict().items()},saved['cola_two_kl'])
            algorithm.ema_model.load_state_dict(saved['ema_model']);algorithm.ema_model.tau3.zero_()
            algorithm.restore_ddde_state(saved)
            assert float(algorithm.ema_model.tau3)==algorithm.tau_lb3
            assert torch.isfinite(get_eval_logits(captured[0])).all()
            records.append(dict(case=task['case'],tau=algorithm.tau_lb3,ddde=True,
                                weak_features_aligned=True,dea_gradients_and_update=True,
                                checkpoint_and_independent_eval_restore=True))
            print(f'PASS actual GPU Full dynamic execution: case {task["case"]}',flush=True)
            del algorithm,module,inputs,lb,ulb,captured,result,saved,before,p,logits,features
            gc.collect();torch.cuda.empty_cache()
    (batch/'gpu_audit.json').write_text(json.dumps(dict(status='passed',records=records,
        model_sha256=m['reference_model_sha256']),indent=2)+'\n')
    return records


def verify(batch,gpu_check=False,smoke_batch=None):
    m=json.loads((batch/'manifest.json').read_text())
    assert m['methods']==['full'] and [t['name'] for t in m['tasks']]==CASES
    assert [sum(t['gpu']==g for t in m['tasks']) for g in (0,1)]==[3,3]
    for name,expected in json.loads((batch/'source_sha256.json').read_text()).items():
        assert digest(Path(m['code_snapshot'])/name)==expected,name
    for name,expected in m['reference_model_sha256'].items():
        assert digest(Path(m['code_snapshot'])/name)==expected,name
    assert digest(m['data_file'])==m['data_sha256']
    for name,expected in json.loads((batch/'cache_sha256.json').read_text()).items():
        assert digest(batch/'data'/name)==expected
    assert digest(Path(m['previous_batch'])/'manifest.json')==m['previous_manifest_sha256']
    configs=json.loads((batch/'config_sha256.json').read_text());records=[]
    _,labels,names=load_raw_signal_data(str(batch/'data'))
    reference_split=_stratified_split(labels,len(names),2)
    labeled_identities={}
    indices=batch/'split_indices';indices.mkdir(exist_ok=True)
    allowed={'save_dir','save_name','data_dir','imb_algorithm','ulb_num_labels'}
    if m['smoke']:allowed.update({'epoch','num_train_iter','num_eval_iter','num_log_iter',
                                 'batch_size','eval_batch_size','num_workers','est_epoch','use_tensorboard'})
    for task in m['tasks']:
        assert digest(task['config'])==configs[task['name']]
        assert digest(task['previous_config'])==task['previous_config_sha256']
        cfg=yaml.safe_load(Path(task['config']).read_text())
        old=yaml.safe_load(Path(task['previous_config']).read_text())
        assert set(cfg)-set(old)=={'ablation_no_ddde_kl','ablation_no_dea_aggregator','full_train_counts','labeled_counts'}
        assert all(cfg[k]==v for k,v in old.items() if k not in allowed)
        assert cfg['imb_algorithm']=='metaexpert_bacn'
        assert not cfg['ablation_no_ddde_kl'] and not cfg['ablation_no_dea_aggregator']
        expected_l,expected_u=expected_counts(task['case'])
        assert cfg['labeled_counts']==expected_l
        assert cfg['full_train_counts']==[a+b for a,b in zip(expected_l,expected_u)]
        assert cfg['ulb_num_labels']==(1170 if task['case'] in (3,4) else 315)
        with contextlib.redirect_stdout(io.StringIO()):
            datasets=get_raw_signal(SimpleNamespace(**cfg),data_dir=cfg['data_dir'],return_test=True)
        lb,ulb,val,test,counts=datasets;expected_l,expected_u=expected_counts(task['case'])
        assert counts['lb_count']==expected_l and counts['ulb_count']==expected_u
        assert len(lb)==sum(expected_l) and len(ulb)==sum(expected_u)
        groups=[set(ds.sample_indices.tolist()) for ds in datasets[:4]]
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i+1,4))
        np.testing.assert_array_equal(val.sample_indices,reference_split['eval'])
        np.testing.assert_array_equal(test.sample_indices,reference_split['test'])
        ratio=cfg['lb_imb_ratio']
        if ratio in labeled_identities:np.testing.assert_array_equal(lb.sample_indices,labeled_identities[ratio])
        else:labeled_identities[ratio]=lb.sample_indices.copy()
        assert np.bincount(val.targets,minlength=9).tolist()==[50]*9
        assert np.bincount(test.targets,minlength=9).tolist()==[100]*9
        np.savez_compressed(indices/(task['name']+'.npz'),labeled=lb.sample_indices,
                            unlabeled=ulb.sample_indices,validation=val.sample_indices,test=test.sample_indices)
        records.append(dict(case=task['case'],labeled=len(lb),unlabeled=len(ulb),test=900,
                            previous_validation_test_identities_equal=True,labeled_counts=expected_l,
                            unlabeled_counts=expected_u,disjoint=True))
    result=dict(status='passed',tasks=6,cases=records,full_model_enabled=True,
                reference_model_sha256=m['reference_model_sha256'],
                all_six_smoke_train_reload_tests_passed=False)
    if gpu_check:result['actual_gpu_dynamic_execution']=gpu_audit(batch)
    elif (batch/'gpu_audit.json').exists():
        audit=json.loads((batch/'gpu_audit.json').read_text())
        assert audit['status']=='passed' and audit['model_sha256']==m['reference_model_sha256']
        result['actual_gpu_dynamic_execution']=audit['records']
    if smoke_batch:
        sm=json.loads((smoke_batch/'manifest.json').read_text());assert sm['smoke']
        assert (smoke_batch/'status.txt').read_text().strip()=='completed'
        assert sm['reference_model_sha256']==m['reference_model_sha256']
        assert json.loads((smoke_batch/'source_sha256.json').read_text())==json.loads((batch/'source_sha256.json').read_text())
        for task in sm['tasks']:
            state=json.loads((smoke_batch/'state'/(task['name']+'.json')).read_text())
            assert state['state']=='completed' and state['attempt']==1
            check_task(smoke_batch,task['name'])
            with np.load(smoke_batch/'split_indices'/(task['name']+'.npz')) as a, np.load(indices/(task['name']+'.npz')) as b:
                for key in ('labeled','unlabeled','validation','test'):np.testing.assert_array_equal(a[key],b[key])
        result.update(all_six_smoke_train_reload_tests_passed=True,smoke_batch=str(smoke_batch))
    (batch/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'PASS Full six-case audit; gpu={bool(result.get("actual_gpu_dynamic_execution"))}; smoke={bool(smoke_batch)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--gpu-check',action='store_true')
    parser.add_argument('--smoke-batch',type=Path)
    parser.add_argument('--check-task')
    args=parser.parse_args();batch=args.batch.resolve()
    if args.check_task:print(json.dumps(check_task(batch,args.check_task)),flush=True)
    else:verify(batch,args.gpu_check,args.smoke_batch.resolve() if args.smoke_batch else None)
