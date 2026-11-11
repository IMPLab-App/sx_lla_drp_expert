"""Freeze seven high-imbalance cases and all three previously tested variants."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'logs/bacn_ablation_22_20261105'
FIXED_REFERENCE = ROOT / 'logs/bacn_11_20261105'
VARIANTS = ('full', 'no_ddde_kl', 'no_dea')
RATIOS = (40, 50, 60, 70, 80, 90, 100)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_counts(ratio):
    # Identical np.power + integer truncation to make_imbalance_data.
    labeled_raw = [int(35 * np.power(ratio, -c / 8)) for c in range(9)]
    labeled = [max(1, n) for n in labeled_raw]
    unlabeled = [int(315 * np.power(ratio, -c / 8)) for c in range(9)]
    return [a+b for a, b in zip(labeled, unlabeled)], labeled, unlabeled, labeled_raw


def prepare(batch, smoke=False):
    batch = batch.resolve()
    reference = json.loads((REFERENCE / 'manifest.json').read_text())
    fixed_reference = json.loads((FIXED_REFERENCE / 'manifest.json').read_text())
    assert not reference['smoke'] and (REFERENCE / 'status.txt').read_text().strip() == 'completed'
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    shutil.copytree(Path(reference['code_snapshot']), code,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    reference_hashes = json.loads((REFERENCE / 'source_sha256.json').read_text())
    for name, expected in reference_hashes.items():
        assert digest(code / name) == expected, f'Reference source changed: {name}'
    for name in ('prepare_bacn_r6_r12.py', 'verify_bacn_r6_r12.py', 'run_bacn_r6_r12.py'):
        shutil.copy2(ROOT / name, code / name)
    data_dir = batch / 'data'
    data_dir.mkdir()
    source = Path(reference['data_file'])
    shutil.copy2(source, data_dir)
    assert digest(data_dir / source.name) == reference['data_sha256']
    configs = batch / 'configs'
    configs.mkdir()
    templates = {
        variant: next(t for t in reference['tasks'] if t['case'] == 1 and t['variant'] == variant)
        for variant in ('full', 'no_dea')
    }
    templates['no_ddde_kl'] = fixed_reference['tasks'][0]
    tasks, count_rows = [], []
    for variant in VARIANTS:
        template_task = templates[variant]
        template = yaml.safe_load(Path(template_task['config']).read_text())
        for case, ratio in enumerate(RATIOS, 6):
            full, labeled, unlabeled, raw = class_counts(ratio)
            case_name = f'run_{case+6}_r{case}_pic_lb35_{ratio}_ulb315_{ratio}_exp'
            name = f'{case_name}__{variant}'
            cfg = dict(template)
            cfg.update(save_name=name, save_dir=str(batch / 'models'), data_dir=str(data_dir),
                       full_train_counts=full, labeled_counts=labeled,
                       num_labels=35, ulb_num_labels=315,
                       lb_imb_ratio=ratio, ulb_imb_ratio=ratio,
                       overwrite=False, resume=False, load_path=None)
            if smoke:
                cfg.update(epoch=3, num_train_iter=6, num_eval_iter=2, num_log_iter=1,
                           batch_size=8, eval_batch_size=128, num_workers=0,
                           est_epoch=0, use_tensorboard=False)
            config = configs / (name + '.yaml')
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            tasks.append(dict(method=variant, variant=variant, case=case, scenario=f'r{case}',
                              case_name=case_name, name=name, repeat=1, seed=2,
                              gpu=len(tasks) % 2, config=str(config),
                              output=str(batch / 'models' / name),
                              target_ratio=ratio, actual_full_ratio=full[0]/full[-1],
                              full_counts=full, labeled_counts=labeled,
                              unlabeled_counts=unlabeled, unclamped_labeled_counts=raw,
                              template_config=template_task['config'],
                              template_config_sha256=digest(template_task['config']),
                              commands=[[str(code / 'train.py'), '--c', str(config)],
                                        [str(code / 'eval_metaexpert_lb35.py'), '--c', str(config)]]))
            if variant == 'full':
                for c in range(9):
                    count_rows.append(dict(scenario=f'r{case}', target_ratio=ratio,
                                           actual_full_ratio=full[0]/full[-1], class_id=c,
                                           class_name=reference['class_names'][c],
                                           full=full[c], labeled=labeled[c], unlabeled=unlabeled[c],
                                           unclamped_labeled=raw[c], validation=50, test=100))
    manifest = dict(title='BACN r6-r12: Full, No-DDDE/adaptive-tau, No-DEA (21 tasks)',
                    created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), data_file=str(data_dir / source.name),
                    data_sha256=reference['data_sha256'], class_names=reference['class_names'],
                    methods=list(VARIANTS), tasks=tasks, smoke=smoke,
                    reference_batch=str(REFERENCE), fixed_reference_batch=str(FIXED_REFERENCE),
                    reference_manifest_sha256=digest(REFERENCE / 'manifest.json'),
                    reference_source_sha256=reference_hashes,
                    protocol=dict(repeats=1, seeds=[2], epochs=3 if smoke else 100,
                                  train_steps=6 if smoke else 20000,
                                  sampling='L=max(1,int(35*ratio**(-c/8))); U=int(315*ratio**(-c/8)); full=L+U',
                                  minimum_labeled_per_class=1,
                                  no_ddde_definition='Original MetaExpert; fixed tail tau=4; DEA retained',
                                  no_dea_definition=reference['protocol']['no_dea_definition'],
                                  ddde=reference['protocol']['ddde'], tau=reference['protocol']['tau'],
                                  split='Existing seed=2 random 350/50/100 per class, identical to prior 33 tasks',
                                  split_limitation=reference['protocol']['split_limitation'],
                                  fixed_test_samples=900, labeled_unlabeled_disjoint=True,
                                  gpu_tasks={str(g):[t['name'] for t in tasks if t['gpu']==g] for g in (0,1)}))
    (batch / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    with (batch / 'sample_counts.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(count_rows[0]))
        writer.writeheader()
        writer.writerows(count_rows)
    (batch / 'source_sha256.json').write_text(json.dumps(
        {str(p.relative_to(code)):digest(p) for p in code.rglob('*') if p.is_file()}, indent=2)+'\n')
    (batch / 'config_sha256.json').write_text(json.dumps(
        {t['name']:digest(t['config']) for t in tasks}, indent=2)+'\n')
    (batch / 'README.md').write_text(
        '# r6–r12：完整模型及两组消融，共21项\n\n'
        '完全复制此前33项正式实验的代码和数据，保持seed=2、100 epochs、20,000步、batch=64、uratio=2。\n'
        '目标长尾比40/50/60/70/80/90/100；L和U分别由头类35和315作指数取整，L至少1个，完整集为L+U。\n'
        '目标比值受整数截断及最少1标签影响，实际值见sample_counts.csv；没有静默改成零标签故障识别。\n'
        'No-DDDE使用原MetaExpert固定τ=4；Full与No-DEA启用DDDE/双KL自适应τ。\n'
        'No-DEA移除整条动态聚合路径及专用融合损失，双专家原始logits固定0.5/0.5平均。\n'
        'GPU0为11项，GPU1为10项；每卡内部顺序执行，训练后重载最佳模型并独立测试900样本。\n'
        '继续使用用户确认的350训练/50验证/100测试记录内随机划分，测试索引与此前33项完全相同。\n'
        '配置在configs，逐类样本表在sample_counts.csv，样本索引在split_indices，验证记录在verification.json。\n'
        '日志在gpu0/gpu1，模型在models，实时汇总在summary/comparison.md。\n'
        '队列有互斥锁，失败最多自动重试一次，失败日志和输出另存。smoke批次仅用于流程验证。\n')
    print(f'Prepared {len(tasks)} tasks: {batch}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.batch, args.smoke)
