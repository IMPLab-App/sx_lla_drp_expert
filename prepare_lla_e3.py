"""Freeze only stage 1 (six tasks x V1/V5/M0). Never schedules later stages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'logs/lla_e3_replacement_20261111'
HISTORY = Path('/media/docs_1/mete-expert/logs/v1v5_192_20261108')
TASK_NAMES = (
    'run_1_pic_lb35_5_ulb315_5_exp',
    'run_2_pic_lb35_10_ulb315_10_exp',
    'run_3_pic_lb35_5_ulb1170_1.0_exp',
    'run_4_pic_lb35_10_ulb1170_1.0_exp',
    'run_5_pic_lb35_5_ulb315_-5_pxe',
    'run_6_pic_lb35_10_ulb315_-10_pxe',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(batch=DEFAULT):
    batch = batch.resolve()
    if (batch / 'manifest.json').exists():
        raise FileExistsError('Existing experiment manifest is immutable; use a new root')
    historical = json.loads((HISTORY / 'manifest.json').read_text())
    data = Path(historical['data_file'])
    assert digest(data) == historical['data_sha256']
    (batch / 'data').mkdir(exist_ok=True)
    shutil.copy2(data, batch / 'data' / data.name)
    with np.load(data, allow_pickle=False) as d:
        classes = d['class_names'].tolist()
        all_targets = d['labels'].copy()
    from semilearn.datasets.cv_datasets.raw_signal import _stratified_split
    train_pool = _stratified_split(all_targets, len(classes), 2)['train']
    frozen = batch / 'split_indices'
    frozen.mkdir(exist_ok=True)
    tasks, counts, baseline_audit = [], [], []
    for i, case in enumerate(TASK_NAMES, 1):
        old = next(t for t in historical['tasks'] if t['scenario'] == f'lu{i}' and t['version'] == 'V1')
        assert old['case_name'] == case
        source = Path(old['reference_indices'])
        destination = frozen / (case + '.npz')
        shutil.copy2(source, frozen / (case + '.source.npz'))
        with np.load(source, allow_pickle=False) as z:
            indices = {k:z[k].copy() for k in z.files}
        np.savez_compressed(destination, **indices, train_pool=train_pool, class_names=np.asarray(classes))
        cfg0 = yaml.safe_load(Path(old['config']).read_text())
        if i in (3, 4):
            assert sum(np.asarray(cfg0['full_train_counts']) - cfg0['labeled_counts']) == 1170
        for version, mode in (('V1', 'fixed'), ('V5', 'ddde_kl'), ('M0', 'lla_hard')):
            cfg = dict(cfg0)
            cfg.pop('five_variant', None)
            cfg.pop('five_fixed_tau', None)
            name = case + '__' + version
            cfg.update(imb_algorithm='metaexpert_lla_e3', e3_adjust_mode=mode,
                ablation_no_ddde_kl=mode != 'ddde_kl', ablation_no_dea_aggregator=False,
                save_dir=str(batch / 'lu_replacement'), save_name=name,
                data_dir=str(batch / 'data'), data_sha256=historical['data_sha256'],
                fixed_split_file=str(destination), fixed_split_sha256=digest(destination),
                split_seed=2, train_seed=None, seed=2, num_warmup_iter=cfg0.get('num_warmup_iter', 0),
                resume=False, load_path=None, overwrite=False, gpu=0,
                lla_init=0., lla_epsilon=1e-8, lla_q_stop_gradient=True,
                lla_optimizer='Adam', lla_lr=cfg0['lr'], lla_weight_decay=0.,
                lla_betas=[.9, .999], lla_optimizer_epsilon=1e-8,
                lla_scheduler='constant', lla_clip_grad=0., lla_tail_class_ids=[6, 7, 8],
                selection_metric='validation_macro_f1', inference_protocol='prespecified_by_mode',
                automatic_test_evaluation=False, use_tensorboard=False,
                network_scheduler='none_legacy_ImbAlgorithmBase_pic_branch',
                gradient_accumulation_steps=1, bn_structure='original_independent_BNH_BNT',
                ddde_kl_enabled=mode=='ddde_kl', fixed_tau_la_enabled=mode=='fixed',
                e3_supervised_logits='raw' if mode=='lla_hard' else 'legacy_tau_adjusted',
                e3_pseudo_label_logits='lla_adjusted' if mode=='lla_hard' else 'raw',
                main_inference='lla_adjusted_fusion' if mode=='lla_hard' else 'legacy_raw_fusion',
                fusion_training_loss_enabled=True, lambda_ema=mode=='lla_hard')
            path = batch / 'configs' / (name + '.yaml')
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            tasks.append(dict(scenario=f'LU{i}', case_name=case, version=version, mode=mode,
                config=str(path), config_sha256=digest(path), output=str(batch / 'lu_replacement' / name),
                fixed_split_sha256=digest(destination), source_config=old['config'],
                source_config_sha256=digest(old['config']), status='pending', seed=2))
            if version in ('V1', 'V5'):
                source_task = next(t for t in historical['tasks'] if t['case_name'] == case and t['version'] == version)
                baseline_audit.append(dict(case_name=case, version=version, reuse=False,
                    reasons=['Historical best checkpoint selected by validation Accuracy, new protocol uses Macro-F1',
                             'Current raw_signal.py / metaexpert.py hashes differ; identity audit required'],
                    historical_output=source_task['output'], action='fresh_strictly_matched_baseline'))
        counts.append(dict(case_name=case, L=cfg0['labeled_counts'],
                           U=(np.asarray(cfg0['full_train_counts'])-cfg0['labeled_counts']).tolist()))
    manifest = dict(stage=1, tasks=tasks, task_names=list(TASK_NAMES), classes=classes,
        data_file=str(batch/'data'/data.name), data_sha256=digest(data),
        historical_manifest=str(HISTORY/'manifest.json'), historical_manifest_sha256=digest(HISTORY/'manifest.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_tree=True, selection_metric='validation_macro_f1', test_used_for_selection=False,
        code_snapshot=str(batch/'code'),
        phase2_gate=dict(mean_m0_gt_v1=True, mean_m0_gt_v5=True, m0_gt_v1_tasks_at_least=4,
            tail_class_ids=[6,7,8], severe_degradation_absolute_f1=.05,
            severe_degradation_metrics=['C5_F1','C8_F1','tail_macro_f1'],
            severe_degradation_rule='For every LU, M0 must not drop more than .05 against either V1 or V5 for any listed validation metric'),
        stages2_and3_enqueued=False, sample_counts=counts,
        split_limitation=historical['protocol']['split_limitation'],
        seed_protocol=dict(train=2,split=2,sampler=0,train_seed_alias=None))
    (batch/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (batch/'code_audit/baseline_reuse.json').write_text(json.dumps(baseline_audit,ensure_ascii=False,indent=2)+'\n')
    originals = [p for p in (ROOT/'semilearn').rglob('*.py') if p.name not in ('lla_e3.py','lla_algorithm.py','lla_runtime.py')]
    originals += [ROOT/'train.py']
    (batch/'code_audit/original_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in originals},indent=2)+'\n')
    snapshot(batch)
    print(f'Prepared {len(tasks)} stage-1 runs. No training launched. {batch}')


def snapshot(batch):
    code = batch/'code'
    # Refresh allowed only before formal training and for this new snapshot.
    if any((batch/'lu_replacement').glob('*/training_history.jsonl')):
        raise RuntimeError('Cannot update source snapshot after training has started')
    code.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'semilearn',code/'semilearn',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('train.py','train_lla_e3.py','prepare_lla_e3.py','run_lla_e3_stage1.py',
                 'verify_lla_e3.py','summarize_lla_e3.py','lla_experiment_utils.py','verify_lla_e3_gpu.py','verify_lla_e3_ddp.py'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,code/name)
    hashes={str(p.relative_to(code)):digest(p) for p in code.rglob('*') if p.is_file()}
    (batch/'code_audit/source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,default=DEFAULT)
    p.add_argument('--snapshot-only',action='store_true')
    a=p.parse_args()
    if a.snapshot_only:snapshot(a.batch.resolve())
    else:prepare(a.batch)
