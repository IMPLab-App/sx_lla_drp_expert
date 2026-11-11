"""Verify current-method six-case sampling and actual routing-layer updates."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from prepare_metaexpert_lb35 import CASES
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal, load_raw_signal_data, _stratified_split


def gpu_routing_check(config):
    from train import get_config
    from semilearn.core.utils import get_net_builder
    from semilearn.imb_algorithms import get_imb_algorithm
    previous = sys.argv
    try:
        sys.argv = ['train.py', '--c', str(config)]
        args = get_config()
    finally:
        sys.argv = previous
    args.distributed = False
    args.num_workers = 0
    args.gpu = 0
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    algorithm = get_imb_algorithm(args, get_net_builder(args.net, args.net_from_name), None, None)
    algorithm.model.cuda(0); algorithm.ema_model.cuda(0)
    algorithm.finalize_model_setup()
    # The user's loss intentionally disables fusion through est_epoch+1.
    # Verify trainability in its enabled phase, without changing that schedule.
    algorithm.epoch = algorithm.est_epoch + 2
    algorithm.model.train()
    watched = {'routing': algorithm.model.predict[0].weight,
               'fusion': algorithm.model.lin1[0].weight}
    before = {name: p.detach().clone() for name, p in watched.items()}
    labeled = next(iter(algorithm.loader_dict['train_lb']))
    unlabeled = next(iter(algorithm.loader_dict['train_ulb']))
    algorithm.optimizer.zero_grad(set_to_none=True)
    output, _ = algorithm.train_step(**algorithm.process_batch(**labeled, **unlabeled))
    loss = output['loss']
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in watched.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        gradients[name] = float(parameter.grad.norm())
        assert gradients[name] > 0, name
    algorithm.optimizer.step()
    changes = {name: float((p.detach() - before[name]).abs().max()) for name, p in watched.items()}
    assert all(change > 0 for change in changes.values())
    record = dict(loss=float(loss.detach()), checked_epoch=algorithm.epoch,
                  original_fusion_warmup_preserved=True, gradient_norms=gradients,
                  parameter_max_updates=changes, optimizer_covers_all_current_parameters=True)
    print('PASS: actual GPU routing/fusion gradients and parameter updates:', record, flush=True)
    return record


def verify(batch, gpu_check=False, smoke_batch=None):
    manifest = json.loads((batch / 'manifest.json').read_text())
    tasks = manifest['tasks']; smoke = manifest['smoke']
    assert manifest['methods'] == ['metaexpert'] and len(tasks) == 6
    assert [task['name'] for task in tasks] == CASES
    for gpu in (0, 1):
        assigned = [t for t in tasks if t['gpu'] == gpu]
        assert [t['case'] for t in assigned] == [gpu + 1, gpu + 3, gpu + 5]
    source = Path(manifest['data_file'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['data_sha256']
    _, labels, names = load_raw_signal_data(str(batch / 'data'))
    split = _stratified_split(labels, len(names), 2)
    for a, b in (('train', 'eval'), ('train', 'test'), ('eval', 'test')):
        assert not np.intersect1d(split[a], split[b]).size
    training_counts = np.bincount(labels[split['train']], minlength=9)
    audit = []
    for task in tasks:
        cfg = yaml.safe_load(Path(task['config']).read_text())
        case = task['case']; ratio = 5 if case % 2 else 10
        assert cfg['imb_algorithm'] == 'metaexpert' and cfg['net'] == 'vit_tiny_metaexpert'
        assert cfg['num_labels'] == 35 and cfg['lb_imb_ratio'] == ratio
        assert cfg['ulb_num_labels'] == (1162 if case in (3, 4) else 315)
        assert cfg['ulb_imb_ratio'] == (1.0 if case in (3, 4) else (-ratio if case >= 5 else ratio))
        assert cfg['ulb_imb_type'] == ('pxe' if case >= 5 else 'exp')
        assert cfg['seed'] == 2 and cfg['gpu'] == 0 and cfg['include_lb_to_ulb'] is False
        assert cfg['overwrite'] is False and cfg['resume'] is False
        assert cfg['num_train_iter'] == (6 if smoke else 20000)
        assert cfg['epoch'] == (3 if smoke else 100)
        args = SimpleNamespace(**cfg)
        np.random.seed(2); random.seed(2); torch.manual_seed(2)
        with contextlib.redirect_stdout(io.StringIO()):
            lb, ulb, val, test, counts = get_raw_signal(args, data_dir=args.data_dir, return_test=True)
        expected_lb = [int(35 * ratio ** (-c / 8)) for c in range(9)]
        expected_ulb = ([130] + [129] * 8) if case in (3, 4) else [int(315 * ratio ** (-c / 8)) for c in range(9)]
        if case >= 5:
            expected_ulb = expected_ulb[::-1]
        assert counts['lb_count'] == expected_lb
        assert counts['ulb_count'] == expected_ulb
        assert np.all(np.array(expected_lb) + expected_ulb <= training_counts)
        assert tuple(lb[0]['x_lb'].shape) == tuple(ulb[0]['x_ulb_w'].shape) == (1, 1024)
        suffix = f'{args.num_labels}_{args.lb_imb_ratio}_{args.ulb_num_labels}_{args.ulb_imb_ratio}_noise_{args.noise_ratio}_seed_{args.seed}_idx.npy'
        # The current loader namespaces caches by a class-name fingerprint.
        cached = list((batch / 'data').glob('raw_signal*/labeled_idx/lb_labels_' + suffix))
        assert len(cached) == 1, cached
        cache = cached[0].parent
        lb_idx = np.load(cached[0])
        ulb_idx = np.load(cache / ('ulb_labels_' + suffix))
        assert not np.intersect1d(lb_idx, ulb_idx).size
        audit.append(dict(case=case, lb_count=expected_lb, ulb_count=expected_ulb,
                          labeled_samples=len(lb), unlabeled_samples=len(ulb),
                          test_samples=len(test), lb_ulb_disjoint=True,
                          normalization_mean=lb.weak_transform.mean, normalization_std=lb.weak_transform.std))
    record = dict(status='passed', tasks=6, gpu_cases={'0': [1, 3, 5], '1': [2, 4, 6]},
                  only_current_metaexpert=True, input_shape=[1, 1024], seed=2,
                  dataset_sha256=manifest['data_sha256'], cases=audit)
    if gpu_check:
        record['actual_gpu_optimizer_check'] = gpu_routing_check(tasks[0]['config'])
    if smoke_batch is not None:
        assert (smoke_batch / 'status.txt').read_text().strip() == 'completed'
        smoke_manifest = json.loads((smoke_batch / 'manifest.json').read_text())
        assert len(smoke_manifest['tasks']) == 6
        for task in smoke_manifest['tasks']:
            state = json.loads((smoke_batch / 'state' / (task['name'] + '.json')).read_text())
            assert state['state'] == 'completed'
            out = Path(task['output'])
            metrics = json.loads((out / 'test_metrics.json').read_text())
            with np.load(out / 'test_predictions.npz') as data:
                assert metrics['accuracy'] == float(np.mean(data['y_true'] == data['y_pred']))
            with np.load(out / 'test_probabilities.npz') as data:
                assert data['probabilities'].shape == (900, 9)
                assert np.allclose(data['probabilities'].sum(1), 1, atol=1e-6)
        record['six_smoke_training_reload_evaluation_tasks_passed'] = True
    (batch / 'verification.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print('PASS: six current-method configurations, three per GPU, lb35/ulb315/1162 counts, disjoint splits and genuine 1-D inputs.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--gpu-check', action='store_true')
    parser.add_argument('--smoke-batch', type=Path)
    args = parser.parse_args()
    verify(args.batch.resolve(), args.gpu_check, args.smoke_batch.resolve() if args.smoke_batch else None)
