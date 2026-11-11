"""Generate isolated ablation configs, preserving every baseline training setting."""
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
VARIANTS = ('full', 'no_mff', 'no_dea_aggregator', 'no_ddde_kl')
RUNS = (
    'run_1_pic_lb100_50_ulb250_50_exp',
    'run_2_pic_lb100_100_ulb250_100_exp',
    'run_3_pic_lb100_50_ulb315_1.0_exp',
    'run_4_pic_lb100_100_ulb315_1.0_exp',
    'run_5_pic_lb100_50_ulb250_-50_pxe',
    'run_6_pic_lb100_100_ulb250_-100_pxe',
)
PROTECTED = ('num_labels', 'ulb_num_labels', 'lb_imb_type', 'ulb_imb_type',
             'lb_imb_ratio', 'ulb_imb_ratio', 'seed', 'batch_size', 'epoch',
             'num_train_iter', 'lr', 'num_classes', 'p_cutoff')


def prepare():
    tasks = []
    for variant in VARIANTS:
        for run in RUNS:
            source = ROOT / 'config/custom_sweep' / (run + '.yaml')
            original = yaml.safe_load(source.read_text())
            cfg = dict(original)
            cfg.update(save_name=f'{run}__{variant}', save_dir=f'./saved_models_cola_two_kl/{variant}',
                       overwrite=False, resume=False, load_path=None, gpu=0,
                       ablation_no_mff=variant == 'no_mff',
                       ablation_no_dea_aggregator=variant == 'no_dea_aggregator',
                       ablation_no_ddde_kl=variant == 'no_ddde_kl')
            assert all(cfg[k] == original[k] for k in PROTECTED)
            dest = ROOT / 'config/cola_two_kl_24' / variant / (run + '.yaml')
            content = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            if dest.exists() and dest.read_text() != content:
                raise FileExistsError(f'Refusing to replace an existing different config: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            tasks.append(dict(variant=variant, run_name=run, seed=cfg['seed'],
                              config=str(dest.relative_to(ROOT)), source=str(source.relative_to(ROOT)),
                              config_sha256=hashlib.sha256(content.encode()).hexdigest(),
                              model_dir=str(Path(cfg['save_dir']) / cfg['save_name']),
                              num_train_iter=cfg['num_train_iter']))
    path = ROOT / 'config/cola_two_kl_24/manifest.json'
    path.write_text(json.dumps(dict(variants=VARIANTS, runs=RUNS, tasks=tasks), indent=2) + '\n')
    print(f'Prepared {len(tasks)} configs: {path}')


if __name__ == '__main__':
    prepare()
