"""Check high-ratio counts, frozen protocol, identities and actual smoke outputs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import yaml

from prepare_bacn_r6_r12 import RATIOS, VARIANTS, class_counts, digest
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal


def verify(batch, smoke_batch=None):
    m = json.loads((batch / 'manifest.json').read_text())
    assert len(m['tasks']) == 21 and m['methods'] == list(VARIANTS)
    assert [sum(t['gpu']==g for t in m['tasks']) for g in (0,1)] == [11,10]
    assert digest(m['data_file']) == m['data_sha256']
    code = Path(m['code_snapshot'])
    for name, expected in json.loads((batch / 'source_sha256.json').read_text()).items():
        assert digest(code / name) == expected, name
    for name, expected in m['reference_source_sha256'].items():
        assert digest(code / name) == expected, name
    assert digest(Path(m['reference_batch'])/'manifest.json') == m['reference_manifest_sha256']
    prior_audit = json.loads((Path(m['reference_batch'])/'verification.json').read_text())['baseline_reuse']
    assert prior_audit['passed']
    assert prior_audit['parent_sha256'] == digest(code/'semilearn/imb_algorithms/metaexpert/metaexpert.py')
    assert prior_audit['extension_sha256'] == digest(code/'semilearn/imb_algorithms/metaexpert/bacn_ablation.py')
    assert {x['variant'] for x in prior_audit['dynamic_variants']} == {'full','no_dea'}
    for record in prior_audit['dynamic_variants']:
        assert 0<=record['tau']<4 and record['gradients'] and record['weak_logits_features_aligned']
        assert record['checkpoint_restore'] and record['evaluation_tau_restore']
    # Confirm the generation rule exactly reproduces all five earlier r cases.
    original = json.loads((Path(m['fixed_reference_batch'])/'manifest.json').read_text())
    for task in original['tasks'][:5]:
        cfg = yaml.safe_load(Path(task['config']).read_text())
        full, labeled, unlabeled, _ = class_counts(cfg['lb_imb_ratio'])
        assert full == task['full_counts'] and labeled == task['labeled_counts']
        assert unlabeled == task['unlabeled_counts']
    ref = np.load(Path(m['fixed_reference_batch'])/'split_indices'/(original['tasks'][0]['name']+'.npz'))
    allowed = {'save_name','save_dir','data_dir','full_train_counts','labeled_counts',
               'num_labels','ulb_num_labels','lb_imb_ratio','ulb_imb_ratio'}
    if m['smoke']:
        allowed.update({'epoch','num_train_iter','num_eval_iter','num_log_iter','batch_size',
                        'eval_batch_size','num_workers','est_epoch','use_tensorboard'})
    snapshots = batch/'split_indices'
    snapshots.mkdir(exist_ok=True)
    config_hashes = json.loads((batch/'config_sha256.json').read_text())
    records, identities, pools = [], {}, {}
    for task in m['tasks']:
        assert digest(task['config']) == config_hashes[task['name']]
        assert digest(task['template_config']) == task['template_config_sha256']
        cfg = yaml.safe_load(Path(task['config']).read_text())
        template = yaml.safe_load(Path(task['template_config']).read_text())
        assert set(cfg) == set(template)
        assert all(cfg[k] == value for k,value in template.items() if k not in allowed)
        full, labeled, unlabeled, raw = class_counts(task['target_ratio'])
        assert cfg['full_train_counts'] == full == task['full_counts']
        assert cfg['labeled_counts'] == labeled == task['labeled_counts']
        assert unlabeled == task['unlabeled_counts'] and raw == task['unclamped_labeled_counts']
        assert task['target_ratio'] == RATIOS[task['case']-6]
        assert cfg['epoch'] == (3 if m['smoke'] else 100)
        assert cfg['num_train_iter'] == (6 if m['smoke'] else 20000)
        assert min(labeled)>=1 and full[0]==350 and max(full)<=350
        v = task['variant']
        if v == 'no_ddde_kl':
            assert cfg['imb_algorithm']=='metaexpert' and cfg['la_tau_lb3']==4
            assert 'ablation_no_ddde_kl' not in cfg
        else:
            assert cfg['imb_algorithm']=='metaexpert_bacn' and not cfg['ablation_no_ddde_kl']
            assert cfg['ablation_no_dea_aggregator'] == (v=='no_dea')
        with contextlib.redirect_stdout(io.StringIO()):
            lb,ulb,val,test,counts = get_raw_signal(SimpleNamespace(**cfg),data_dir=cfg['data_dir'],return_test=True)
        assert counts['lb_count']==labeled and counts['ulb_count']==unlabeled
        assert len(lb)==sum(labeled) and len(ulb)==sum(unlabeled)
        assert np.bincount(val.targets,minlength=9).tolist()==[50]*9
        assert np.bincount(test.targets,minlength=9).tolist()==[100]*9
        groups=[set(ds.sample_indices.tolist()) for ds in (lb,ulb,val,test)]
        assert all(len(g)==len(ds) for g,ds in zip(groups,(lb,ulb,val,test)))
        assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i+1,4))
        np.testing.assert_array_equal(val.sample_indices,ref['validation'])
        np.testing.assert_array_equal(test.sample_indices,ref['test'])
        current=[ds.sample_indices.copy() for ds in (lb,ulb,val,test)]
        if task['case'] in identities:
            for a,b in zip(current,identities[task['case']]):np.testing.assert_array_equal(a,b)
        else:
            identities[task['case']] = current
            pools[task['case']] = groups[0]|groups[1]
        np.savez_compressed(snapshots/(task['name']+'.npz'),
                            labeled=current[0],unlabeled=current[1],validation=current[2],test=current[3])
        records.append(dict(name=task['name'],case=task['case'],variant=v,
                            labeled=len(lb),unlabeled=len(ulb),test=len(test),
                            target_ratio=task['target_ratio'],actual_full_ratio=full[0]/full[-1]))
    assert all(pools[c+1] <= pools[c] for c in range(6,12))
    result=dict(status='passed',tasks=21,records=records,
                frozen_reference_source_identical=True,previous_r1_r5_counts_reproduced=True,
                three_variants_identical_samples=True,identical_previous_validation_test=True,
                nested_full_pools=True,disjoint=True,minimum_labeled_per_class=1,
                identical_code_prior_dynamic_gpu_audit=prior_audit['dynamic_variants'],
                all_21_smoke_train_reload_tests_passed=False)
    if smoke_batch:
        smoke=json.loads((smoke_batch/'manifest.json').read_text())
        assert smoke['smoke'] and len(smoke['tasks'])==21
        assert (smoke_batch/'status.txt').read_text().strip()=='completed'
        assert smoke['data_sha256']==m['data_sha256']
        assert json.loads((smoke_batch/'source_sha256.json').read_text()) == json.loads((batch/'source_sha256.json').read_text())
        for t in smoke['tasks']:
            state=json.loads((smoke_batch/'state'/(t['name']+'.json')).read_text())
            assert state['state']=='completed' and state['attempt']==1
            dest=Path(t['output']);metrics=json.loads((dest/'test_metrics.json').read_text())
            assert (dest/'model_best.pth').is_file() and metrics['samples']==900
            with np.load(dest/'test_predictions.npz') as pred:
                assert metrics['accuracy']==float(np.mean(pred['y_true']==pred['y_pred']))
            with np.load(dest/'test_probabilities.npz') as prob:
                values=prob['probabilities']
                assert values.shape==(900,9) and np.isfinite(values).all()
                assert np.allclose(values.sum(1),1,atol=1e-6)
            text=(dest/'log.txt').read_text()
            assert 'train/' in text and not re.search(r':\s*[+-]?(?:nan|inf)\b',text,re.I)
            history=dest/'cola_history.jsonl'
            if t['variant']=='no_ddde_kl':
                assert not history.exists()
            else:
                h=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
                # Six steps need not yield confidence >= .95, so tau may stay
                # at its fallback 4. Dynamic updates/restores were GPU-audited
                # above against these exact, unchanged algorithm sources.
                assert any(x['epoch']>=1 for x in h)
                assert all(np.isfinite(x['tau_dyn']) and 0<=x['tau_dyn']<=4 for x in h)
            if t['variant']=='no_dea':
                losses=re.findall(r'train/total_fuse_loss: ([\d.eE+-]+)',(dest/'log.txt').read_text())
                assert losses and all(float(x)==0 for x in losses)
        assert not any(list(smoke_batch.glob('issues_gpu*.jsonl')))
        result.update(all_21_smoke_train_reload_tests_passed=True,smoke_batch=str(smoke_batch))
    (batch/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f'PASS: 21 configurations, exact counts, disjoint identities, unchanged reference code; smoke={bool(smoke_batch)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--smoke-batch',type=Path)
    args=parser.parse_args()
    verify(args.batch.resolve(),args.smoke_batch.resolve() if args.smoke_batch else None)
