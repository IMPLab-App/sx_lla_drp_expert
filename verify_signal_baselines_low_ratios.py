"""Check requested sampling, identical per-method splits and pinned 1-D code."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from prepare_signal_baselines_low_ratios import METHODS, CASES
from semilearn.datasets.cv_datasets.raw_signal import (get_raw_signal, load_raw_signal_data,
                                                      _stratified_split)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(batch, smoke_batch=None):
    manifest = json.loads((batch / 'manifest.json').read_text())
    tasks = manifest['tasks']
    smoke = manifest['smoke']
    active_cases = manifest.get('active_cases', [1, 2] if smoke else list(range(1, 7)))
    case_names = manifest.get('case_names', CASES)
    sampling = manifest.get('sampling_spec', dict(label_head=100, unlabeled_head=250, balanced_unlabeled=315))
    total = len(active_cases) * len(METHODS)
    assert len(tasks) == len({t['name'] for t in tasks}) == total
    assert manifest['protocol']['repeats'] == 1 and manifest['protocol']['seeds'] == [2]
    assert digest(Path(manifest['data_file'])) == manifest['data_sha256']
    for rel, expected in manifest['proven_training_source_sha256'].items():
        assert digest(batch / 'code' / rel) == expected, rel
    for gpu in (0, 1):
        assigned = [task for task in tasks if task['gpu'] == gpu]
        cases = [case for case in active_cases if (case - 1) % 2 == gpu]
        assert len(assigned) == total // 2
        assert [t['case'] for t in assigned] == [case for case in cases for _ in METHODS]
    _, all_targets, names = load_raw_signal_data(str(batch / 'data'))
    split = _stratified_split(all_targets, len(names), 2)
    for a, b in (('train', 'eval'), ('train', 'test'), ('eval', 'test')):
        assert not np.intersect1d(split[a], split[b]).size
    audit, test_hash = {}, None
    for task in tasks:
        config = yaml.safe_load(Path(task['config']).read_text())
        case = task['case']
        assert task['case_name'] == case_names[case - 1]
        assert config['dataset'] == 'raw_signal' and config['seed'] == 2
        assert config['gpu'] == 0 and config['overwrite'] is False and config['resume'] is False
        assert config['lb_imb_ratio'] == (5 if case % 2 else 10)
        assert config['num_labels'] == sampling['label_head']
        assert config['ulb_num_labels'] == (sampling['balanced_unlabeled'] if case in (3, 4) else sampling['unlabeled_head'])
        assert config['ulb_imb_ratio'] == (1 if case in (3, 4) else (-1 if case >= 5 else 1) * config['lb_imb_ratio'])
        assert config['ulb_imb_type'] == ('pxe' if case >= 5 else 'exp')
        assert config['num_train_iter'] == (6 if smoke else 20000)
        assert config['epoch'] == (3 if smoke else 100)
        assert config['batch_size'] == manifest['protocol']['lb_batch']
        assert config['uratio'] == 2
        for command in task['commands']:
            assert Path(command[0]).is_file()
            if '--c' in command:
                assert command[command.index('--c') + 1] == task['config']
            else:
                assert command[command.index('--input_channels') + 1] == '1'
                assert command[command.index('--data_dir') + 1] == config['data_dir']
                assert command[command.index('--out_dir') + 1] == task['output']
                assert command[command.index('--steps_per_epoch') + 1] == str(config['num_train_iter'] // config['epoch'])
                for key in ('num_labels', 'lb_imb_ratio', 'ulb_imb_ratio', 'ulb_num_labels'):
                    assert float(command[command.index('--' + key) + 1]) == config[key]
        # Independently load every method's split, including standalone float
        # ratio parsing, so common class counts alone cannot hide different data.
        args = SimpleNamespace(**config)
        if task['method'] in ('uaman', 'sat_ifixmatch'):
            args.lb_imb_ratio = float(args.lb_imb_ratio)
            args.ulb_imb_ratio = float(args.ulb_imb_ratio)
        np.random.seed(2); random.seed(2); torch.manual_seed(2)
        with contextlib.redirect_stdout(io.StringIO()):
            lb, ulb, val, test, counts = get_raw_signal(args, data_dir=args.data_dir, return_test=True)
        expected_lb = [int(args.num_labels * args.lb_imb_ratio ** (-c / 8)) for c in range(9)]
        if case in (3, 4):
            base, remainder = divmod(args.ulb_num_labels, 9)
            expected_ulb = [base + (c < remainder) for c in range(9)]
        else:
            expected_ulb = [int(args.ulb_num_labels * abs(args.ulb_imb_ratio) ** (-c / 8)) for c in range(9)]
        if case >= 5:
            expected_ulb = expected_ulb[::-1]
        assert counts['lb_count'] == expected_lb
        assert counts['ulb_count'] == expected_ulb
        training_counts = np.bincount(all_targets[split['train']], minlength=9)
        assert np.all(np.asarray(expected_lb) + expected_ulb <= training_counts)
        assert tuple(lb[0]['x_lb'].shape) == tuple(ulb[0]['x_ulb_s'].shape) == (1, 1024)
        suffix = f'{args.num_labels}_{args.lb_imb_ratio}_{args.ulb_num_labels}_{args.ulb_imb_ratio}_noise_{args.noise_ratio}_seed_{args.seed}_idx.npy'
        cached = list((batch / 'data').glob('raw_signal*/labeled_idx/lb_labels_' + suffix))
        assert len(cached) == 1, cached
        assert not np.intersect1d(np.load(cached[0]), np.load(cached[0].with_name('ulb_labels_' + suffix))).size
        fingerprints = {key: hashlib.sha256(ds.data.tobytes() + np.asarray(ds.targets).tobytes()).hexdigest()
                        for key, ds in (('labeled', lb), ('unlabeled', ulb), ('validation', val), ('test', test))}
        if test_hash is None:
            test_hash = fingerprints['test']
        assert fingerprints['test'] == test_hash
        if case in audit:
            assert audit[case]['fingerprints'] == fingerprints, task['name']
            audit[case]['verified_methods'].append(task['method'])
        else:
            audit[case] = dict(case=case, seed=2, counts=counts, fingerprints=fingerprints,
                               normalization_mean=lb.weak_transform.mean,
                               normalization_std=lb.weak_transform.std,
                               verified_methods=[task['method']])
    assert all(sorted(item['verified_methods']) == sorted(METHODS) for item in audit.values())
    if smoke_batch is not None:
        assert (smoke_batch / 'status.txt').read_text().strip() == 'completed'
        smoke_manifest = json.loads((smoke_batch / 'manifest.json').read_text())
        assert len(smoke_manifest['tasks']) == len(smoke_manifest.get('active_cases', [1, 2])) * len(METHODS)
        for task in smoke_manifest['tasks']:
            state = json.loads((smoke_batch / 'state' / (task['name'] + '.json')).read_text())
            assert state['state'] == 'completed'
            metrics = json.loads((Path(task['output']) / 'test_metrics.json').read_text())
            with np.load(Path(task['output']) / 'test_predictions.npz') as pred:
                assert metrics['accuracy'] == float(np.mean(pred['y_true'] == pred['y_pred']))
        assert '待第二次' not in (smoke_batch / 'summary/comparison.md').read_text()
    result = dict(status='passed',tasks=total,gpu_tasks=total//2,single_repeat=True,seed=2,
                  input_shape=[1,1024],dataset_sha256=manifest['data_sha256'],
                  training_source_matches_completed_batch=True,all_methods_share_splits=True,
                  train_validation_test_disjoint=True,cases=list(audit.values()),
                  smoke_tasks_passed=len(smoke_manifest['tasks']) if smoke_batch is not None else 0,
                  sampling_spec=sampling)
    (batch / 'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f'PASS: {total} configurations; {total//2} per GPU; seed 2; correct ratios 5/10; all eight methods share identical case-specific data; native [1,1024]; proven training code unchanged.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke-batch', type=Path)
    args = parser.parse_args()
    verify(args.batch.resolve(),args.smoke_batch.resolve() if args.smoke_batch else None)
