"""Verify sample identities, actual baseline equivalence, and DDDE/DEA execution."""
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from prepare_bacn_ablation import digest
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal


def make_algorithm(config_path, algorithm_name, disable_ddde, no_dea, output):
    from train import get_config
    from semilearn.core.utils import get_net_builder, send_model_cuda
    from semilearn.imb_algorithms import get_imb_algorithm
    previous = sys.argv
    try:
        sys.argv = ['verify', '--c', str(config_path)]
        args = get_config()
    finally:
        sys.argv = previous
    args.imb_algorithm = algorithm_name
    args.ablation_no_ddde_kl = disable_ddde
    args.ablation_no_dea_aggregator = no_dea
    args.num_workers, args.distributed, args.gpu = 0, False, 0
    args.save_dir, args.save_name = str(output), 'audit'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        algorithm = get_imb_algorithm(args, get_net_builder(args.net, args.net_from_name), None, None)
        algorithm.model = send_model_cuda(args, algorithm.model)
        algorithm.ema_model = send_model_cuda(args, algorithm.ema_model, clip_batch=False)
        algorithm.finalize_model_setup()
    return algorithm


def assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0, msg=name)


def optimizer_signature(algorithm):
    names = {id(p): n for n, p in algorithm.model.named_parameters()}
    return [dict(names=[names[id(p)] for p in g['params']],
                 settings={k: v for k, v in g.items() if k != 'params'})
            for g in algorithm.optimizer.param_groups]


def gpu_audit(batch):
    from semilearn.imb_algorithms.metaexpert.cola_ddde import TwoKLDDDE
    from eval_confusion_matrix_new import get_eval_logits
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    manifest = json.loads((batch / 'manifest.json').read_text())
    baseline = Path(manifest['baseline_batch'])
    original = json.loads((baseline / 'manifest.json').read_text())
    parent = Path(manifest['code_snapshot']) / 'semilearn/imb_algorithms/metaexpert/metaexpert.py'
    assert digest(parent) == digest(Path(original['code_snapshot']) / 'semilearn/imb_algorithms/metaexpert/metaexpert.py')
    records, dynamic = [], []
    with tempfile.TemporaryDirectory(prefix='bacn_ablation_audit_') as directory:
        output = Path(directory)
        for task in original['tasks']:
            old = make_algorithm(task['config'], 'metaexpert', True, False, output)
            # Disabled really means no DDDE construction, not a frozen observer.
            with patch('semilearn.imb_algorithms.metaexpert.bacn_ablation.TwoKLDDDE', side_effect=AssertionError('Disabled DDDE constructed')):
                new = make_algorithm(task['config'], 'metaexpert_bacn', True, False, output)
            assert new.cola is None
            assert_state_equal(old.model.state_dict(), new.model.state_dict())
            assert_state_equal(old.ema_model.state_dict(), new.ema_model.state_dict())
            assert optimizer_signature(old) == optimizer_signature(new)
            assert [(n,p.requires_grad) for n,p in old.model.named_parameters()] == [(n,p.requires_grad) for n,p in new.model.named_parameters()]
            for key in ('train_lb','train_ulb','eval','test'):
                np.testing.assert_array_equal(old.dataset_dict[key].sample_indices, new.dataset_dict[key].sample_indices)
            lb = next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_lb'], batch_size=8)))
            ulb = next(iter(torch.utils.data.DataLoader(old.dataset_dict['train_ulb'], batch_size=8)))
            inputs = old.process_batch(**lb, **ulb)
            old.model.eval(); new.model.eval()
            with torch.no_grad():
                a, b = old.model(inputs['x_lb']), new.model(inputs['x_lb'])
                assert a.keys() == b.keys()
                for key in ('logits','aux_logits2','feat','w1','w3','fuse_logit_l','fuse_w_logit',
                            'fuse_logit_HMT_c_1','fuse_logit_HMT_c_3'):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
            old.model.train(); new.model.train()
            for epoch in (0,20):
                old.epoch = new.epoch = epoch
                old.it = new.it = 0
                cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                old.optimizer.zero_grad(set_to_none=True)
                a, _ = old.train_step(**inputs)
                a['loss'].backward()
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng)
                new.optimizer.zero_grad(set_to_none=True)
                with patch.object(TwoKLDDDE, 'observe', side_effect=AssertionError('Disabled DDDE executed')):
                    b, _ = new.train_step(**inputs)
                b['loss'].backward()
                torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
                for (name,p), (other,q) in zip(old.model.named_parameters(), new.model.named_parameters()):
                    assert name == other and (p.grad is None) == (q.grad is None)
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all()
                        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
                old.optimizer.step(); new.optimizer.step()
                assert_state_equal(old.model.state_dict(), new.model.state_dict())
            records.append(dict(case=task['case'], initialization=True, forward=True,
                                loss=True, gradients=True, optimizer_update=True,
                                sample_identities=True, bitwise_equal=True,
                                checked_epochs=[0,20], disabled_ddde_never_called=True))
            print(f'PASS exact baseline reuse: case {task["case"]}', flush=True)
            del old, new, inputs, a, b, lb, ulb
            gc.collect(); torch.cuda.empty_cache()
        for variant in ('full','no_dea'):
            task = next(t for t in manifest['tasks'] if t['variant'] == variant)
            algorithm = make_algorithm(task['config'], 'metaexpert_bacn', False, variant == 'no_dea', output)
            module = algorithm.model
            lb = next(iter(torch.utils.data.DataLoader(algorithm.dataset_dict['train_lb'], batch_size=8)))
            ulb = next(iter(torch.utils.data.DataLoader(algorithm.dataset_dict['train_ulb'], batch_size=8)))
            inputs = algorithm.process_batch(**lb, **ulb)
            # Seed a valid previous window to test the next-epoch dynamic update.
            algorithm.cola.begin_epoch(algorithm.est_epoch)
            algorithm.cola.observe(torch.eye(9, device='cuda').repeat_interleave(3,0)*20,
                                   torch.randn(27,192,device='cuda'))
            algorithm.epoch, algorithm.it = algorithm.est_epoch+2, 0
            captured = []
            hook = module.register_forward_hook(lambda m,a,o: captured.append(o))
            with patch.object(algorithm.cola, 'observe', wraps=algorithm.cola.observe) as spy:
                result, logs = algorithm.train_step(**inputs)
                assert spy.call_count == 1
                observed_logits, observed_features = spy.call_args.args
                torch.testing.assert_close(observed_logits, captured[0]['aux_logits2'][8:16])
                torch.testing.assert_close(observed_features, captured[0]['feat'][8:16])
            hook.remove()
            assert 0 <= algorithm.tau_lb3 < 4
            assert float(module.tau3) == algorithm.tau_lb3
            assert float(module.tau1) == algorithm.args.la_tau_lb1
            assert torch.isfinite(result['loss'])
            algorithm.optimizer.zero_grad(set_to_none=True)
            result['loss'].backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
            for layer in (module.backbone.classifier, module.backbone.aux_classifier2):
                assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
            if variant == 'full':
                assert module.predict[0].weight.grad.abs().sum() > 0
                assert module.lin1[0].weight.grad.abs().sum() > 0
            else:
                assert not hasattr(module, 'predict') and not hasattr(module, 'lin1')
                assert logs['train/total_fuse_loss'] == 0.0
                assert torch.equal(captured[0]['w1'], torch.full_like(captured[0]['w1'], 0.5))
                torch.testing.assert_close(get_eval_logits(captured[0]),
                                           (captured[0]['logits']+captured[0]['aux_logits2'])*0.5)
            algorithm.optimizer.step()
            algorithm.save_model('checkpoint.pth', str(output / variant))
            saved = torch.load(output / variant / 'checkpoint.pth', map_location='cpu', weights_only=False)
            algorithm.cola.pi_u.zero_()
            algorithm.tau_lb3 = 0
            algorithm.load_model(str(output / variant / 'checkpoint.pth'))
            assert_state_equal({k:v.cpu() for k,v in algorithm.cola.state_dict().items()}, saved['cola_two_kl'])
            assert algorithm.tau_lb3 == float(saved['cola_two_kl']['tau_dyn'])
            # Exercise the same strict EMA-load + DDDE restore used by evaluation.
            algorithm.ema_model.load_state_dict(saved['ema_model'])
            algorithm.ema_model.tau3.zero_()
            algorithm.restore_ddde_state(saved)
            assert float(algorithm.ema_model.tau3) == algorithm.tau_lb3
            dynamic.append(dict(variant=variant, tau=algorithm.tau_lb3,
                                weak_logits_features_aligned=True, gradients=True,
                                checkpoint_restore=True, evaluation_tau_restore=True,
                                uniform_weights=variant=='no_dea', fusion_losses_off=variant=='no_dea'))
            print(f'PASS DDDE/DEA execution and checkpoint: {variant}', flush=True)
            del algorithm, module, inputs, captured, result, saved, lb, ulb
            gc.collect(); torch.cuda.empty_cache()
    return dict(passed=True, cases=records, dynamic_variants=dynamic,
                baseline_manifest_sha256=manifest['baseline_manifest_sha256'],
                parent_sha256=digest(parent),
                extension_sha256=digest(Path(manifest['code_snapshot']) / 'semilearn/imb_algorithms/metaexpert/bacn_ablation.py'))


def verify(batch, gpu_check=False, reuse_audit=None, smoke_batch=None):
    manifest = json.loads((batch / 'manifest.json').read_text())
    assert len(manifest['tasks']) == 22
    assert [sum(t['gpu']==g for t in manifest['tasks']) for g in (0,1)] == [11,11]
    assert digest(Path(manifest['data_file'])) == manifest['data_sha256']
    for name, expected in json.loads((batch / 'source_sha256.json').read_text()).items():
        assert digest(Path(manifest['code_snapshot']) / name) == expected, name
    baseline = Path(manifest['baseline_batch'])
    assert digest(baseline / 'manifest.json') == manifest['baseline_manifest_sha256']
    allowed = {'save_name','save_dir','data_dir','imb_algorithm','ablation_no_ddde_kl','ablation_no_dea_aggregator'}
    if manifest['smoke']:
        allowed.update({'epoch','num_train_iter','num_eval_iter','num_log_iter','batch_size',
                        'eval_batch_size','num_workers','est_epoch','use_tensorboard'})
    counts = []
    for task in manifest['tasks']:
        cfg = yaml.safe_load(Path(task['config']).read_text())
        original = yaml.safe_load(Path(task['baseline_task']['config']).read_text())
        assert all(cfg[k] == v for k,v in original.items() if k not in allowed)
        assert set(cfg).difference(original) == {'ablation_no_ddde_kl','ablation_no_dea_aggregator'}
        assert cfg['imb_algorithm'] == 'metaexpert_bacn' and not cfg['ablation_no_ddde_kl']
        assert cfg['ablation_no_dea_aggregator'] == (task['variant']=='no_dea')
        with contextlib.redirect_stdout(io.StringIO()):
            lb,ulb,val,test,number = get_raw_signal(SimpleNamespace(**cfg),data_dir=cfg['data_dir'],return_test=True)
        with np.load(baseline / 'split_indices' / (task['baseline_task']['name']+'.npz')) as expected:
            for ds,key in zip((lb,ulb,val,test),('labeled','unlabeled','validation','test')):
                np.testing.assert_array_equal(ds.sample_indices, expected[key])
        assert len(test)==900 and number['lb_count']==task['labeled_counts'] and number['ulb_count']==task['unlabeled_counts']
        counts.append(dict(case=task['case'],variant=task['variant'],labeled=len(lb),unlabeled=len(ulb),test=len(test),identical_to_baseline_indices=True))
    result = dict(status='passed', tasks=22, cases=counts,
                  data_sha256=manifest['data_sha256'], gpu_tasks={'0':11,'1':11})
    if gpu_check:
        result['baseline_reuse'] = gpu_audit(batch)
    elif reuse_audit:
        audit = json.loads(reuse_audit.read_text())['baseline_reuse']
        assert audit['passed'] and audit['baseline_manifest_sha256']==manifest['baseline_manifest_sha256']
        assert audit['parent_sha256']==digest(Path(manifest['code_snapshot'])/'semilearn/imb_algorithms/metaexpert/metaexpert.py')
        assert audit['extension_sha256']==digest(Path(manifest['code_snapshot'])/'semilearn/imb_algorithms/metaexpert/bacn_ablation.py')
        result['baseline_reuse'] = audit
    if smoke_batch:
        smoke = json.loads((smoke_batch / 'manifest.json').read_text())
        assert smoke['smoke'] and len(smoke['tasks'])==22
        assert (smoke_batch/'status.txt').read_text().strip()=='completed'
        assert not list(smoke_batch.glob('issues_gpu*.jsonl'))
        for task in smoke['tasks']:
            state=json.loads((smoke_batch/'state'/(task['name']+'.json')).read_text())
            assert state['state']=='completed'
            out=Path(task['output'])
            metrics=json.loads((out/'test_metrics.json').read_text())
            assert metrics['samples']==900 and (out/'model_best.pth').is_file()
            with np.load(out/'test_predictions.npz') as d:
                assert metrics['accuracy']==float(np.mean(d['y_true']==d['y_pred']))
            with np.load(out/'test_probabilities.npz') as d:
                assert d['probabilities'].shape==(900,9) and np.isfinite(d['probabilities']).all()
                assert np.allclose(d['probabilities'].sum(1),1,atol=1e-6)
            saved=torch.load(out/'latest_model.pth',map_location='cpu',weights_only=False)
            assert 'cola_two_kl' in saved and saved['bacn_module_settings']['no_dea']==(task['variant']=='no_dea')
        result['all_22_smoke_train_reload_tests_passed']=True
    (batch/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('PASS: 22 configurations, unchanged training parameters, identical original sample indices.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--gpu-check',action='store_true')
    parser.add_argument('--reuse-audit',type=Path)
    parser.add_argument('--smoke-batch',type=Path)
    args=parser.parse_args()
    verify(args.batch.resolve(),args.gpu_check,args.reuse_audit,args.smoke_batch)
