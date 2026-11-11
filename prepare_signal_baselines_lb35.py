"""Prepare 48 lb35 baseline tasks using the completed eight-method snapshot."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CASES = ['run_1_pic_lb35_5_ulb315_5_exp', 'run_2_pic_lb35_10_ulb315_10_exp',
         'run_3_pic_lb35_5_ulb1162_1.0_exp', 'run_4_pic_lb35_10_ulb1162_1.0_exp',
         'run_5_pic_lb35_5_ulb315_-5_pxe', 'run_6_pic_lb35_10_ulb315_-10_pxe']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(batch, smoke=False):
    batch = batch.resolve()
    source = ROOT / 'logs/signal_baselines_5_10_48_20261101'
    original = json.loads((source / 'manifest.json').read_text())
    assert (source / 'status.txt').read_text().strip() == 'completed'
    expected = json.loads((source / 'source_sha256.json').read_text())
    for relative, value in expected.items():
        assert digest(source / 'code' / relative) == value, relative
    methods = original['methods']
    assert methods == ['acr', 'bacon', 'cossl', 'cpe', 'simpro', 'fixmatch', 'uaman', 'sat_ifixmatch']
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    shutil.copytree(source / 'code', code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('prepare_signal_baselines_lb35.py', 'verify_signal_baselines_low_ratios.py',
                 'run_signal_baselines.py', 'status_signal_baselines.py'):
        shutil.copy2(ROOT / name, code / name)
    data = batch / 'data'
    data.mkdir()
    input_file = ROOT / 'data/raw_signal_dataset_G_1024.npz'
    shutil.copy2(input_file, data)
    assert digest(data / input_file.name) == digest(input_file)
    with np.load(data / input_file.name, allow_pickle=False) as dataset:
        assert dataset['data'].shape == (4500, 1024)
        assert np.isfinite(dataset['data']).all()
        names = dataset['class_names'].tolist()
        assert len(names) == 9 and np.array_equal(np.unique(dataset['labels']), np.arange(9))
    (batch / 'configs').mkdir()
    tasks = []
    for previous in original['tasks']:
        case = previous['case']; method = previous['method']; case_name = CASES[case - 1]
        assert previous['repeat'] == 1 and previous['seed'] == 2
        cfg = yaml.safe_load(Path(previous['config']).read_text())
        old_cfg = dict(cfg)
        name = f'{method}_{case_name}__seed2'
        output = batch / 'models' / name
        config = batch / 'configs' / (name + '.yaml')
        cfg.update(num_labels=35, ulb_num_labels=1162 if case in (3, 4) else 315,
                   save_name=name, save_dir=str(batch / 'models'), data_dir=str(data))
        identity_changes = {'num_labels', 'ulb_num_labels', 'save_name', 'save_dir', 'data_dir'}
        assert {k: v for k, v in cfg.items() if k not in identity_changes} == {
            k: v for k, v in old_cfg.items() if k not in identity_changes}
        if smoke:
            # Test the actual formal batch sizes, including CoSSL's small
            # labeled sampler and SAT's drop_last behavior with lb35.
            cfg.update(epoch=3, num_train_iter=6, num_eval_iter=2, num_log_iter=1,
                       num_workers=0, est_epoch=0, use_tensorboard=False)
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        commands = []
        for command in previous['commands']:
            updated = [token.replace(str(source), str(batch)).replace(previous['name'], name)
                       for token in command]
            if '--c' in updated:
                updated[updated.index('--c') + 1] = str(config)
            else:
                replacements = dict(num_labels='35', ulb_num_labels=str(cfg['ulb_num_labels']),
                                    data_dir=str(data), out_dir=str(output),
                                    epochs=str(cfg['epoch']),
                                    steps_per_epoch=str(cfg['num_train_iter'] // cfg['epoch']),
                                    num_workers=str(cfg['num_workers']))
                if method == 'uaman':
                    replacements['adapt_steps'] = '3' if smoke else '200'
                for key, value in replacements.items():
                    updated[updated.index('--' + key) + 1] = value
            assert all(str(source) not in token for token in updated)
            commands.append(updated)
        tasks.append(dict(previous, name=name, case_name=case_name, config=str(config),
                          output=str(output), commands=commands))
    assert len(tasks) == 48
    training_sources = {rel: value for rel, value in expected.items()
                        if rel.startswith('semilearn/') or rel in ('train.py', 'eval_signal_baseline.py', 'scripts/signal_metrics.py')}
    assert all(digest(code / rel) == value for rel, value in training_sources.items())
    manifest = dict(original)
    manifest.update(title='一维信号八方法：lb35 六配置',
                    created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), source_batch=str(source),
                    data_file=str(data / input_file.name), data_sha256=digest(data / input_file.name),
                    class_names=names, smoke=smoke, tasks=tasks, case_names=CASES,
                    active_cases=list(range(1, 7)),
                    sampling_spec=dict(label_head=35, unlabeled_head=315, balanced_unlabeled=1162),
                    proven_training_source_sha256=training_sources,
                    dataset_provenance=dict(input_file=str(input_file), current_sha256=digest(input_file),
                                            previous_batch_data_sha256=original['data_sha256'],
                                            dataset_identical_to_previous_batch=digest(input_file) == original['data_sha256']))
    manifest['protocol'] = dict(original['protocol'], repeats=1, seeds=[2],
                                epochs=3 if smoke else 100, train_steps=6 if smoke else 20000,
                                lb_batch=64, ulb_batch=128,
                                gpu_cases={'0': [1, 3, 5], '1': [2, 4, 6]})
    (batch / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    hashes = {str(path.relative_to(code)): digest(path) for path in code.rglob('*') if path.is_file()}
    (batch / 'source_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (batch / 'setup_issues.md').write_text(
        '# 检查与问题处理\n\n'
        '- 沿用已完成八方法一维实验的真实训练源码快照，训练代码逐文件校验，避免意外改用近期自有 MetaExpert 实现。\n'
        '- 原准备和验证脚本将100/250/315写死；本批次按manifest记录35/315/1162并逐类核验。\n'
        '- 均匀无标签1162按原采样器分配130+8×129，完整保留1162；长尾无标签315为头类数量。\n'
        '- 每方法统一seed=2、同一数据快照、相同划分及归一化。48项短训练采用正式batch=64/128覆盖全部六任务。\n'
        '- 每卡24项顺序执行，保留GPU上现有其他计算进程。运行期异常写入issues_gpu*.jsonl，失败产物归档后重试一次。\n')
    (batch / 'README.md').write_text(
        '# lb35：八方法六配置实验\n\n'
        '共48项，8方法×6配置，每组合单次seed=2。\n'
        'GPU0：run_1→run_3→run_5；GPU1：run_2→run_4→run_6，各24项。\n'
        '方法顺序：ACR、BaCon、CoSSL、CPE、SimPro、FixMatch、UA-MAN、SAT-iFixMatch。\n'
        '模型与损失复用上一批已完成的一维基线实现：前六方法为各自输出头的一维WRN，UA-MAN和SAT-iFixMatch为各自一维网络。\n'
        '正式预算100 epochs/20,000步，batch=64/128，AdamW lr=0.0005、weight_decay=0.05。\n'
        'UA-MAN保留原生监督训练及200步train_ulb无监督专家聚合；SAT拒识在统一准确率中计错。\n'
        '全部使用固定raw_signal_dataset_G_1024.npz，输入[B,1,1024]，同seed共享70/10/20分层划分、训练集归一化。\n'
        'lb35为标签头类数量；长尾ulb315为无标签头类数量；均匀ulb1162为总量，130+8×129。\n'
        '准备校验见verification.json，异常记录见setup_issues.md及运行期issues_gpu*.jsonl。\n'
        '实时结果见summary/comparison.md、summary/runs.csv，模型及原始测试预测见models/。\n'
        '旧批次结果与当前自有方法六任务结果均保留。\n\n'
        f'查看进度：`/home/kv/.conda/envs/cw/bin/python {ROOT / "status_signal_baselines.py"} {batch}`\n')
    print(f'Prepared {len(tasks)} lb35 baseline tasks: {batch}', flush=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.batch, args.smoke)
