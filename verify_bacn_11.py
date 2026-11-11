"""Audit exact class counts and original sample identities for all eleven tasks."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from prepare_bacn_11 import SCENARIOS, digest
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal

EXPECTED_LU = [(157,1443), (126,1162), (118,1103), (104,969), (94,885),
               (147,1356), (147,1356), (147,1356), (131,848), (95,884), (60,919)]


def verify(batch, smoke_batch=None):
    manifest = json.loads((batch / 'manifest.json').read_text())
    assert manifest['methods'] == ['metaexpert'] and len(manifest['tasks']) == 11
    assert digest(Path(manifest['data_file'])) == manifest['data_sha256']
    for name, expected in json.loads((batch / 'source_sha256.json').read_text()).items():
        assert digest(Path(manifest['code_snapshot']) / name) == expected, name
    indices = batch / 'split_indices'
    indices.mkdir(exist_ok=True)
    rows, pools, labeled_sets = [], [], []
    fixed_eval = fixed_test = None
    for task, (name, ratio, full, labeled), totals in zip(manifest['tasks'], SCENARIOS, EXPECTED_LU):
        cfg = yaml.safe_load(Path(task['config']).read_text())
        assert task['name'] == cfg['save_name'] == name
        assert cfg['full_train_counts'] == full and cfg['labeled_counts'] == labeled
        assert cfg['seed'] == 2 and cfg['gpu'] == 0 and not cfg['include_lb_to_ulb']
        assert not cfg['overwrite'] and not cfg['resume'] and cfg['noise_ratio'] == 0
        assert cfg['net'] == 'vit_tiny_metaexpert' and cfg['imb_algorithm'] == 'metaexpert'
        assert cfg['epoch'] == (3 if manifest['smoke'] else 100)
        assert cfg['num_train_iter'] == (6 if manifest['smoke'] else 20000)
        with contextlib.redirect_stdout(io.StringIO()):
            lb, ulb, val, test, counts = get_raw_signal(
                SimpleNamespace(**cfg), data_dir=cfg['data_dir'], return_test=True)
        unlabeled = [n - l for n, l in zip(full, labeled)]
        assert counts['lb_count'] == labeled and counts['ulb_count'] == unlabeled
        assert (len(lb), len(ulb)) == totals
        assert np.array_equal(np.bincount(val.targets, minlength=9), [50] * 9)
        assert np.array_equal(np.bincount(test.targets, minlength=9), [100] * 9)
        groups = [set(d.sample_indices.tolist()) for d in (lb, ulb, val, test)]
        assert all(len(g) == len(d) for g, d in zip(groups, (lb, ulb, val, test)))
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i + 1, 4))
        if fixed_test is None:
            fixed_eval, fixed_test = val.sample_indices.copy(), test.sample_indices.copy()
        assert np.array_equal(val.sample_indices, fixed_eval)
        assert np.array_equal(test.sample_indices, fixed_test)
        # Changing external RNG state must not affect explicit-count sampling.
        np.random.seed(9341)
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = get_raw_signal(SimpleNamespace(**cfg), data_dir=cfg['data_dir'], return_test=True)
        assert np.array_equal(lb.sample_indices, repeated[0].sample_indices)
        assert np.array_equal(ulb.sample_indices, repeated[1].sample_indices)
        assert tuple(lb[0]['x_lb'].shape) == tuple(ulb[0]['x_ulb_w'].shape) == (1,1024)
        pools.append(groups[0] | groups[1])
        labeled_sets.append(groups[0])
        np.savez_compressed(indices / (name + '.npz'),
                            labeled=lb.sample_indices, unlabeled=ulb.sample_indices,
                            validation=val.sample_indices, test=test.sample_indices)
        rows.append(dict(case=task['case'], name=name, full_counts=full,
                         labeled_counts=labeled, unlabeled_counts=unlabeled,
                         labeled_samples=len(lb), unlabeled_samples=len(ulb),
                         validation_samples=len(val), test_samples=len(test),
                         target_ratio=ratio, actual_full_ratio=max(full)/min(full),
                         disjoint=True, deterministic=True))
    assert pools[5] == pools[6] == pools[7]
    assert pools[4] == pools[8] == pools[9] == pools[10]
    assert labeled_sets[10] < labeled_sets[9] < labeled_sets[8]
    result = dict(status='passed', tasks=11, input_shape=[1,1024], cases=rows,
                  identical_validation_and_test_indices=True,
                  cd1_cd3_identical_full_sample_identities=True,
                  r5_cd4_cd6_identical_full_sample_identities=True,
                  cd4_cd6_nested_label_sets=True, independent_of_global_rng=True,
                  split_limitation=manifest['protocol']['split_limitation'])
    if smoke_batch:
        smoke = json.loads((smoke_batch / 'manifest.json').read_text())
        assert smoke['smoke'] and len(smoke['tasks']) == 11
        assert (smoke_batch / 'status.txt').read_text().strip() == 'completed'
        for task in smoke['tasks']:
            state = json.loads((smoke_batch / 'state' / (task['name'] + '.json')).read_text())
            assert state['state'] == 'completed'
            output = Path(task['output'])
            metrics = json.loads((output / 'test_metrics.json').read_text())
            assert metrics['samples'] == 900 and (output / 'model_best.pth').is_file()
            with np.load(output / 'test_predictions.npz') as prediction:
                assert metrics['accuracy'] == float(np.mean(prediction['y_true'] == prediction['y_pred']))
            with np.load(output / 'test_probabilities.npz') as probability:
                assert probability['probabilities'].shape == (900,9)
                assert np.isfinite(probability['probabilities']).all()
                assert np.allclose(probability['probabilities'].sum(1), 1, atol=1e-6)
        result['eleven_smoke_train_reload_and_test_tasks_passed'] = True
    (batch / 'verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('PASS: 11 exact L/U counts, disjoint splits, identical CD full pools and fixed 900-sample test.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke-batch', type=Path)
    args = parser.parse_args()
    verify(args.batch.resolve(), args.smoke_batch.resolve() if args.smoke_batch else None)
