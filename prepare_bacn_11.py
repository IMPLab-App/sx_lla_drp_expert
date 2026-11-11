"""Prepare the requested eleven exact-count experiments with current MetaExpert."""
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
CD_MATCH_TOTAL = [350, 278, 223, 177, 142, 113, 91, 72, 57]
CD_SCARCE_TOTAL = [350, 227, 148, 96, 63, 41, 26, 17, 11]
# Names retain the requested ratio parameters; the arrays govern actual sampling.
SCENARIOS = [
    ('run_1_r1_pic_lb35_5_ulb315_5_exp', 5,
     [350,285,233,191,155,127,104,85,70], [35,28,23,19,15,12,10,8,7]),
    ('run_2_r2_pic_lb35_10_ulb315_10_exp', 10,
     [350,262,196,146,110,82,62,46,34], [35,26,19,14,11,8,6,4,3]),
    ('run_3_r3_pic_lb35_12_ulb315_12_exp', 12,
     [350,255,187,137,100,73,53,38,28], [35,25,18,13,10,7,5,3,2]),
    ('run_4_r4_pic_lb35_20_ulb315_20_exp', 20,
     [350,240,164,113,77,53,36,24,16], [35,24,16,11,7,5,3,2,1]),
    ('run_5_r5_pic_lb35_30_ulb315_30_exp', 30,
     CD_SCARCE_TOTAL, [35,22,14,9,6,4,2,1,1]),
    ('run_6_cd1_pic_lb35_6_ulb315_6_match', 6,
     CD_MATCH_TOTAL, [35,27,22,17,14,11,9,7,5]),
    ('run_7_cd2_pic_lb35_6_ulb315_6_step', 6,
     CD_MATCH_TOTAL, [35,22,22,22,12,12,12,5,5]),
    ('run_8_cd3_pic_lb35_6_ulb315_6_partialrev', 6,
     CD_MATCH_TOTAL, [35,18,22,26,9,13,17,3,4]),
    ('run_9_cd4_pic_lb35_30_ulb315_30_fixed', 30,
     CD_SCARCE_TOTAL, [35,28,21,15,11,8,6,4,3]),
    ('run_10_cd5_pic_lb30_30_ulb320_30_fixed', 30,
     CD_SCARCE_TOTAL, [30,22,15,10,7,5,3,2,1]),
    ('run_11_cd6_pic_lb20_30_ulb330_30_fixed', 30,
     CD_SCARCE_TOTAL, [20,14,9,6,4,3,2,1,1]),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(batch, smoke=False):
    batch = batch.resolve()
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    for folder in ('semilearn', 'scripts'):
        shutil.copytree(ROOT / folder, code / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('train.py', 'eval_confusion_matrix_new.py', 'publication_plotting.py',
                 'eval_metaexpert_lb35.py', 'run_signal_baselines.py', 'status_signal_baselines.py',
                 'prepare_bacn_11.py', 'verify_bacn_11.py'):
        shutil.copy2(ROOT / name, code / name)
    data = batch / 'data'
    data.mkdir()
    source = ROOT / 'data/raw_signal_dataset_G_1024.npz'
    shutil.copy2(source, data)
    assert digest(data / source.name) == digest(source)
    with np.load(data / source.name, allow_pickle=False) as dataset:
        class_names = dataset['class_names'].tolist()
        assert dataset['data'].shape == (4500, 1024)
        assert np.array_equal(np.bincount(dataset['labels']), [500] * 9)
        assert np.isfinite(dataset['data']).all()

    configs = batch / 'configs'
    configs.mkdir()
    template = yaml.safe_load((ROOT / 'config/custom_sweep/run_1_pic_lb100_5_ulb250_5_exp.yaml').read_text())
    assert template['net'] == 'vit_tiny_metaexpert' and template['imb_algorithm'] == 'metaexpert'
    tasks, count_rows = [], []
    for case, (name, ratio, full, labeled) in enumerate(SCENARIOS, 1):
        unlabeled = [n - l for n, l in zip(full, labeled)]
        cfg = dict(template)
        cfg.update(save_name=name, save_dir=str(batch / 'models'), data_dir=str(data),
                   full_train_counts=full, labeled_counts=labeled,
                   num_labels=labeled[0], ulb_num_labels=unlabeled[0],
                   lb_imb_ratio=ratio, ulb_imb_ratio=ratio,
                   lb_imb_type='exp', ulb_imb_type='exp',
                   include_lb_to_ulb=False, noise_ratio=0.0,
                   seed=2, gpu=0, num_workers=2, overwrite=False, resume=False,
                   load_path=None, save_eval_confusion=False)
        if smoke:
            cfg.update(epoch=3, num_train_iter=6, num_eval_iter=2, num_log_iter=1,
                       batch_size=8, eval_batch_size=128, num_workers=0,
                       est_epoch=0, use_tensorboard=False)
        config = configs / (name + '.yaml')
        config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        tasks.append(dict(method='metaexpert', case=case, case_name=name, repeat=1,
                          seed=2, gpu=(case - 1) % 2, name=name, config=str(config),
                          output=str(batch / 'models' / name),
                          full_counts=full, labeled_counts=labeled, unlabeled_counts=unlabeled,
                          commands=[[str(code / 'train.py'), '--c', str(config)],
                                    [str(code / 'eval_metaexpert_lb35.py'), '--c', str(config)]]))
        for c in range(9):
            count_rows.append(dict(case=case, task=name, class_id=c, class_name=class_names[c],
                                   full=full[c], labeled=labeled[c], unlabeled=unlabeled[c],
                                   validation=50, test=100, target_ratio=ratio,
                                   actual_full_ratio=max(full) / min(full)))
    manifest = dict(title='MetaExpert BACN-inspired 11 exact-count tasks',
                    created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), data_file=str(data / source.name),
                    data_sha256=digest(data / source.name), class_names=class_names,
                    methods=['metaexpert'], smoke=smoke, tasks=tasks,
                    protocol=dict(repeats=1, seeds=[2], epochs=3 if smoke else 100,
                                  train_steps=6 if smoke else 20000,
                                  split='Existing seeded stratified random 350/50/100 per class',
                                  split_limitation='One source record per class; within-record evaluation only',
                                  split_approved_by_user=True, labeled_unlabeled_disjoint=True,
                                  fixed_test_samples=900,
                                  gpu_cases={'0': [1,3,5,7,9,11], '1': [2,4,6,8,10]}))
    (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    with (batch / 'sample_counts.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(count_rows[0]))
        writer.writeheader()
        writer.writerows(count_rows)
    hashes = {str(p.relative_to(code)): digest(p) for p in code.rglob('*') if p.is_file()}
    (batch / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (batch / 'README.md').write_text(
        '# MetaExpert 11组实验\n\n'
        '只训练当前MetaExpert，骨干vit_tiny_metaexpert，真实一维输入[1,1024]。\n'
        '正式训练沿用100 epochs、20,000步、batch=64、uratio=2、seed=2，融合预热est_epoch=18。\n'
        '所有任务使用full_train_counts和labeled_counts精确采样，U为完整训练集扣除L。\n'
        '同seed下CD1–CD3完整样本身份相同；r5与CD4–CD6完整样本身份相同；CD4–CD6标注集合嵌套。\n'
        '固定随机350训练/50验证/100测试，每类仅一条来源记录，用户已确认沿用记录内划分，不能据此宣称记录间泛化。\n'
        '两张GPU各有一条顺序队列：GPU0为1/3/5/7/9/11，GPU1为2/4/6/8/10；保留现有任务。\n'
        '代码和数据固定复制到本批次，配置在configs/，逐类数量在sample_counts.csv，原始行索引在split_indices/。\n'
        '日志在gpu0/和gpu1/，模型和900样本测试结果在models/，进度在state/，汇总在summary/。\n'
        '队列自动完成训练后最佳模型重载及评估，失败日志保留，最多自动重试一次。\n', encoding='utf-8')
    print(f'Prepared {len(tasks)} tasks: {batch}', flush=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.batch, args.smoke)
