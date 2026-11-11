"""Prepare the requested 48 low-imbalance baseline tasks in two 24-task queues.

The proven training implementation is copied from the completed original
baseline snapshot. Only dataset sampling and independent output paths change.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
METHODS = ['acr', 'bacon', 'cossl', 'cpe', 'simpro', 'fixmatch', 'uaman', 'sat_ifixmatch']
CASES = ['run_1_pic_lb100_5_ulb250_5_exp', 'run_2_pic_lb100_10_ulb250_10_exp',
         'run_3_pic_lb100_5_ulb315_1.0_exp', 'run_4_pic_lb100_10_ulb315_1.0_exp',
         'run_5_pic_lb100_5_ulb250_-5_pxe', 'run_6_pic_lb100_10_ulb250_-10_pxe']
SAMPLING = ('num_labels', 'ulb_num_labels', 'lb_imb_type', 'lb_imb_ratio',
            'ulb_imb_type', 'ulb_imb_ratio', 'noise_type', 'noise_ratio',
            'noise_per_class', 'num_steps', 'include_lb_to_ulb')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(batch, smoke=False):
    batch = batch.resolve()
    source = ROOT / 'logs/signal_baselines_96_20261027'
    old = json.loads((source / 'manifest.json').read_text())
    expected = json.loads((source / 'source_sha256.json').read_text())
    assert (source / 'status.txt').read_text().strip() == 'completed'
    for rel, value in expected.items():
        assert digest(source / 'code' / rel) == value, rel
    current_data = ROOT / 'data/raw_signal_dataset_G_1024.npz'
    current_digest = digest(current_data)
    with np.load(current_data, allow_pickle=False) as current, np.load(old['data_file'], allow_pickle=False) as previous:
        assert current['data'].shape == (4500, 1024)
        assert np.isfinite(current['data']).all()
        assert np.array_equal(np.unique(current['labels']), np.arange(9))
        previous_names = previous['class_names'].tolist()
        class_mapping = []
        for i, name in enumerate(current['class_names'].tolist()):
            old_name = '正常' if name == '健康' else name
            j = previous_names.index(old_name)
            class_mapping.append(j)
            assert np.array_equal(current['data'][current['labels'] == i],
                                  previous['data'][previous['labels'] == j]), name
        current_names = current['class_names'].tolist()
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    shutil.copytree(source / 'code', code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Summarization now supports a single repeat as well as the previous batches.
    for name in ('run_signal_baselines.py', 'prepare_signal_baselines_low_ratios.py',
                 'verify_signal_baselines_low_ratios.py', 'status_signal_baselines.py'):
        shutil.copy2(ROOT / name, code / name)
    data = batch / 'data'
    data.mkdir()
    shutil.copy2(current_data, data)
    assert digest(data / current_data.name) == current_digest
    configs = batch / 'configs'
    configs.mkdir()
    tasks = []
    for case_id, case in enumerate(CASES, 1):
        if smoke and case_id not in (1, 2):
            continue
        sampling = yaml.safe_load((ROOT / 'config/custom_sweep' / (case + '.yaml')).read_text())
        for method in METHODS:
            template = next(t for t in old['tasks'] if t['repeat'] == 1 and t['case'] == case_id and t['method'] == method)
            cfg = yaml.safe_load(Path(template['config']).read_text())
            cfg.update({key: sampling[key] for key in SAMPLING})
            name = f'{method}_{case}__seed2'
            output = batch / 'models' / name
            config = configs / (name + '.yaml')
            cfg.update(seed=2, gpu=0, save_name=name, save_dir=str(batch / 'models'),
                       data_dir=str(data), overwrite=False, resume=False, load_path=None)
            if smoke:
                cfg.update(epoch=3, num_train_iter=6, num_eval_iter=2, num_log_iter=1,
                           batch_size=8, eval_batch_size=128, num_workers=0,
                           est_epoch=0, cossl_tfe_warm_bs=32, use_tensorboard=False)
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            commands = []
            for original in template['commands']:
                command = [token.replace(str(source / 'code'), str(code)) for token in original]
                if '--c' in command:
                    command[command.index('--c') + 1] = str(config)
                else:
                    replacements = dict(data_dir=str(data), out_dir=str(output), seed='2', gpu='0',
                                        epochs=str(cfg['epoch']), steps_per_epoch=str(cfg['num_train_iter'] // cfg['epoch']),
                                        num_workers=str(cfg['num_workers']),
                                        **{key: str(cfg[key]) for key in SAMPLING if '--' + key in command})
                    if method == 'uaman':
                        replacements.update(batch_size=str(cfg['batch_size']), adapt_steps='3' if smoke else '200')
                    else:
                        replacements.update(batch_size_lb=str(cfg['batch_size']),
                                            batch_size_ulb=str(cfg['batch_size'] * cfg['uratio']))
                    for key, value in replacements.items():
                        option = '--' + key
                        if option in command:
                            command[command.index(option) + 1] = value
                assert str(source) not in ' '.join(command), command
                commands.append(command)
            tasks.append(dict(method=method, case=case_id, case_name=case, repeat=1, seed=2,
                              gpu=(case_id - 1) % 2, name=name, config=str(config), output=str(output), commands=commands))
    unchanged = {rel: value for rel, value in expected.items()
                 if rel == 'train.py' or rel == 'eval_signal_baseline.py' or rel.startswith('semilearn/')}
    assert all(digest(code / rel) == value for rel, value in unchanged.items())
    manifest = dict(created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), data_file=str(data / 'raw_signal_dataset_G_1024.npz'),
                    data_sha256=current_digest, methods=METHODS, smoke=smoke, tasks=tasks,
                    source_batch=str(source), proven_training_source_sha256=unchanged,
                    dataset_provenance=dict(input_file=str(current_data), class_names=current_names,
                                            original_batch_data_sha256=old['data_sha256'],
                                            new_to_original_class_indices=class_mapping,
                                            signals_identical_within_each_named_fault_class=True),
                    protocol=dict(old['protocol'], repeats=1, seeds=[2],
                                  epochs=3 if smoke else 100, train_steps=6 if smoke else 20000,
                                  lb_batch=8 if smoke else 64, ulb_batch=16 if smoke else 128,
                                  gpu_cases={'0': [1, 3, 5], '1': [2, 4, 6]}))
    (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    hashes = {str(path.relative_to(code)): digest(path) for path in code.rglob('*') if path.is_file()}
    (batch / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (batch / 'setup_issues.md').write_text(
        '# 检查发现的问题与处理\n\n'
        '1. 原准备脚本固定为比例 50/100、两个种子共 96 项；本次新增独立准备与校验脚本，生成比例 5/10、seed=2 的 48 项，两个 GPU 各 24 项。\n'
        '2. 原汇总脚本将单次结果标为“待第二次”；现按 manifest 中计划重复数汇总，本批次直接报告单次指标。\n'
        '3. 当前数据指纹与旧批次不同：逐故障类核对确认信号数组完全一致，仅类别顺序和健康名称变化。本次固定复制当前数据文件，保留当前 H1/C1…C8 标签顺序；新旧映射与 SHA256 记录在 manifest.json。\n'
        '4. 训练模型、损失、一维输入与原批次一致，不修改已有训练结果。运行期异常如有发生将写入 issues_gpu*.jsonl。\n')
    (batch / 'README.md').write_text(
        '# 一维信号八方法：比例 5/10 六任务对比\n\n'
        f'任务数：{len(tasks)}；每组合训练一次，统一 seed=2。\n'
        'GPU 0：run_1、run_3、run_5；GPU 1：run_2、run_4、run_6；每卡顺序执行各配置的八种方法。\n'
        '正式预算：100 epochs、20,000 steps，有标签 batch=64、无标签 batch=128。\n'
        '各方法读取同一份 raw_signal_dataset_G_1024.npz，输入 [B,1,1024]，同 seed 共享数据划分和归一化。\n'
        'ACR/BaCon/CoSSL/CPE/SimPro/FixMatch 使用已验证的一维 WRN 及各自输出头；UA-MAN 和 SAT-iFixMatch 保留各自一维网络与损失。\n'
        'UA-MAN 的无标签集参与原生 200 步专家聚合；SAT 的拒识在统一指标中计为错误。\n'
        '训练实现来自此前完整完成的批次代码快照，源文件校验见 manifest.json。\n'
        '固定数据及代码快照保存在 data/、code/，原有批次和结果完整保留。\n\n'
        '结果：summary/comparison.md、summary/runs.csv、models/；单次训练直接报告指标，不报告重复间标准差。\n'
        '异常：issues_gpu0.jsonl、issues_gpu1.jsonl；失败自动从头重试一次并归档失败产物，仍失败则记录并继续队列。\n'
        '首次运行前的数据和配置校验见 verification.json；短训练校验见独立 smoke 批次。\n\n'
        f'查看状态：`{ROOT / "status_signal_baselines.py"} {batch}`（使用当前 Python 执行）。\n')
    print(f'Prepared {len(tasks)} tasks: {batch}', flush=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.batch, args.smoke)
