"""Freeze 11 full + 11 no-DEA tasks against the existing fixed-tau batch."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(batch, baseline, smoke=False):
    batch, baseline = batch.resolve(), baseline.resolve()
    original = json.loads((baseline / 'manifest.json').read_text())
    assert len(original['tasks']) == 11 and original['methods'] == ['metaexpert']
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    shutil.copytree(Path(original['code_snapshot']), code,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    overlays = (
        'semilearn/imb_algorithms/metaexpert/__init__.py',
        'semilearn/imb_algorithms/metaexpert/bacn_ablation.py',
        'eval_confusion_matrix_new.py', 'eval_metaexpert_lb35.py',
        'prepare_bacn_ablation.py', 'verify_bacn_ablation.py',
        'wait_bacn_ablation.py', 'summarize_bacn_ablation.py',
    )
    for name in overlays:
        shutil.copy2(ROOT / name, code / name)
    # Preserve the exact initialized architecture, training loop and data loader.
    baseline_hashes = json.loads((baseline / 'source_sha256.json').read_text())
    for name, expected in baseline_hashes.items():
        if name not in overlays:
            assert digest(code / name) == expected, f'Unexpected base code change: {name}'
    data = batch / 'data'
    data.mkdir()
    source = Path(original['data_file'])
    shutil.copy2(source, data)
    assert digest(data / source.name) == original['data_sha256']
    configs = batch / 'configs'
    configs.mkdir()
    tasks = []
    for variant in ('full', 'no_dea'):
        for base_task in original['tasks']:
            cfg = yaml.safe_load(Path(base_task['config']).read_text())
            name = base_task['name'] + '__' + variant
            cfg.update(save_name=name, save_dir=str(batch / 'models'), data_dir=str(data),
                       imb_algorithm='metaexpert_bacn',
                       ablation_no_ddde_kl=False,
                       ablation_no_dea_aggregator=variant == 'no_dea',
                       overwrite=False, resume=False, load_path=None)
            if smoke:
                cfg.update(epoch=3, num_train_iter=6, num_eval_iter=2, num_log_iter=1,
                           batch_size=8, eval_batch_size=128, num_workers=0,
                           est_epoch=0, use_tensorboard=False)
            path = configs / (name + '.yaml')
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            task = dict(base_task)
            task.update(method=variant, variant=variant, name=name, config=str(path),
                        output=str(batch / 'models' / name), baseline_task=base_task,
                        gpu=(base_task['case'] - 1 + (variant == 'no_dea')) % 2,
                        commands=[[str(code / 'train.py'), '--c', str(path)],
                                  [str(code / 'eval_metaexpert_lb35.py'), '--c', str(path)]])
            tasks.append(task)
    manifest = dict(title='BACN 11 cases: Full and No-DEA; reuse fixed-tau No-DDDE',
                    created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), data_file=str(data / source.name),
                    data_sha256=original['data_sha256'], class_names=original['class_names'],
                    methods=['full', 'no_dea'], tasks=tasks, smoke=smoke,
                    baseline_batch=str(baseline),
                    baseline_manifest_sha256=digest(baseline / 'manifest.json'),
                    reused_variant='no_ddde_kl', source_overlays=list(overlays),
                    protocol=dict(repeats=1, seeds=[2], epochs=3 if smoke else 100,
                                  train_steps=6 if smoke else 20000,
                                  no_dea_definition='Entire dynamic aggregation removed; raw logits averaged 0.5/0.5; no fusion losses',
                                  ddde='Tail weak raw logits + shared features, confidence >=0.95, effective rank, EMA 0.9',
                                  tau='Two symmetric KLs against uniform/reversed labeled prior; tau in [0,4]; previous-epoch statistics',
                                  full_baseline='Current BACN architecture plus DDDE/tau; not historical SegFormer MFF',
                                  split_limitation=original['protocol']['split_limitation']))
    (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    hashes = {str(p.relative_to(code)): digest(p) for p in code.rglob('*') if p.is_file()}
    (batch / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    shutil.copy2(baseline / 'sample_counts.csv', batch / 'sample_counts.csv')
    print(f'Prepared {len(tasks)} new tasks, 11 per GPU: {batch}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'logs/bacn_11_20261105')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.batch, args.baseline, args.smoke)
