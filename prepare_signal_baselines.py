"""Prepare an isolated, reproducible 8 x 6 x 2 baseline comparison batch."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parent
METHODS = ['acr', 'bacon', 'cossl', 'cpe', 'simpro', 'fixmatch', 'uaman', 'sat_ifixmatch']
CASES = ['run_1_pic_lb100_50_ulb250_50_exp', 'run_2_pic_lb100_100_ulb250_100_exp',
         'run_3_pic_lb100_50_ulb315_1.0_exp', 'run_4_pic_lb100_100_ulb315_1.0_exp',
         'run_5_pic_lb100_50_ulb250_-50_pxe', 'run_6_pic_lb100_100_ulb250_-100_pxe']


def prepare(batch, smoke=False):
    batch = batch.resolve()
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    shutil.copytree(ROOT / 'semilearn', code / 'semilearn', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'scripts', code / 'scripts', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('train.py', 'eval_signal_baseline.py', 'run_signal_baselines.py', 'prepare_signal_baselines.py'):
        shutil.copy2(ROOT / name, code / name)
    data = batch / 'data'
    data.mkdir()
    shutil.copy2(ROOT / 'data/raw_signal_dataset_G_1024.npz', data)
    digest = hashlib.sha256((data / 'raw_signal_dataset_G_1024.npz').read_bytes()).hexdigest()
    tasks = []
    for repeat, seed in ((1, 2), (2, 3)):
        for case_id, case in enumerate(CASES, 1):
            if smoke and case_id != 1:
                continue
            for method in METHODS:
                if smoke and repeat != 1:
                    continue
                name = f'{method}_{case}__repeat{repeat}_seed{seed}'
                config = yaml.safe_load((ROOT / 'config/custom_sweep' / (case + '.yaml')).read_text())
                config.update(save_dir=str(batch / 'models'), save_name=name, dataset='raw_signal',
                              data_dir=str(data), seed=seed, gpu=0, num_workers=2, overwrite=False,
                              use_tensorboard=True, resume=False, load_path=None,
                              net={'bacon':'wrn_28_2_signal_bacon', 'cpe':'wrn_28_2_signal_cpe'}.get(method, 'wrn_28_2_signal'),
                              imb_algorithm=method, cut1=2, cut2=4,
                              cossl_warm_epoch=0, save_eval_confusion=False)
                # Match the reference project: 100 epochs, 20,000 steps, 64/128 samples per step.
                # Smoke spans epoch boundaries, ACR adaptation and BaCon contrastive phase.
                if smoke:
                    config.update(epoch=3, num_train_iter=6, num_eval_iter=2, num_log_iter=1,
                                  batch_size=8, eval_batch_size=128, num_workers=0,
                                  est_epoch=0, cossl_tfe_warm_bs=32, use_tensorboard=False)
                cfg_path = batch / 'configs' / f'{name}.yaml'
                cfg_path.parent.mkdir(exist_ok=True)
                cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
                out = batch / 'models' / name
                if method in METHODS[:6]:
                    commands = [[str(code / 'train.py'), '--c', str(cfg_path)],
                                [str(code / 'eval_signal_baseline.py'), '--c', str(cfg_path)]]
                else:
                    folder = code / 'semilearn/imb_algorithms' / f'{method}_metaexpert_compare'
                    script = 'train_uaman.py' if method == 'uaman' else 'train_sat_ifixmatch.py'
                    common = ['--dataset', 'raw_signal', '--input_channels', '1', '--data_dir', str(data),
                              '--num_classes', str(config['num_classes']), '--epochs', str(config['epoch']),
                              '--steps_per_epoch', str(config['num_train_iter'] // config['epoch']),
                              '--seed', str(seed), '--gpu', '0', '--out_dir', str(out),
                              '--num_workers', str(config['num_workers']), '--lr', str(config['lr']),
                              '--weight_decay', str(config['weight_decay'])]
                    for key in ('num_labels', 'ulb_num_labels', 'lb_imb_type', 'lb_imb_ratio', 'ulb_imb_type',
                                'ulb_imb_ratio', 'num_steps', 'noise_type', 'noise_ratio'):
                        common += ['--'+key, str(config[key])]
                    if method == 'uaman':
                        common += ['--batch_size', str(config['batch_size']), '--adapt_split', 'train_ulb',
                                   '--adapt_steps', '3' if smoke else '200']
                    else:
                        common += ['--batch_size_lb', str(config['batch_size']), '--batch_size_ulb', str(config['batch_size']*config['uratio'])]
                    commands = [[str(folder / script)] + common]
                tasks.append(dict(method=method, case=case_id, case_name=case, repeat=repeat, seed=seed,
                                  gpu=repeat-1 if not smoke else METHODS.index(method)%2,
                                  name=name, config=str(cfg_path), output=str(out), commands=commands))
    manifest = dict(created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), data_file=str(data / 'raw_signal_dataset_G_1024.npz'),
                    data_sha256=digest, methods=METHODS, smoke=smoke, tasks=tasks,
                    protocol=dict(repeats=2, seeds=[2,3], train_eval_test='70/10/20 stratified by class',
                                  epochs=100, train_steps=20000, lb_batch=64, ulb_batch=128,
                                  normalization='training split global mean/std',
                                  uaman='Native supervised multi-expert loss, plus 200 unsupervised aggregation steps on train_ulb',
                                  sat='Native ResNet1D, adaptive thresholds and consistency/OOD losses; rejected predictions counted as incorrect',
                                  checkpoint='Highest validation accuracy; test used only for reporting'))
    (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    # Pin source checksums for reproducibility and runtime diagnostics.
    hashes = {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in code.rglob('*') if p.is_file()}
    (batch / 'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(f'{batch}: prepared {len(tasks)} tasks; data SHA256={digest}')
    return batch


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args=parser.parse_args()
    prepare(args.batch, args.smoke)
