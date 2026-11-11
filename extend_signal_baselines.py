"""Append two independent repeats using a completed batch's exact code and data.

Usage: python extend_signal_baselines.py NEW_BATCH [--launch]
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, batch):
    original = json.loads((source / 'manifest.json').read_text())
    assert (source / 'status.txt').read_text().strip() == 'completed'
    assert len(original['tasks']) == 96
    assert [t['seed'] for t in original['tasks'][:1]] == [2]
    for relative, expected in json.loads((source / 'source_sha256.json').read_text()).items():
        assert digest(source / 'code' / relative) == expected, relative
    assert digest(Path(original['data_file'])) == original['data_sha256']
    # Repeats must be new across every previously prepared baseline batch.
    for candidate in (ROOT / 'logs').glob('signal_baselines*/manifest.json'):
        previous = json.loads(candidate.read_text())
        if not previous.get('smoke'):
            used = {task['seed'] for task in previous['tasks']}
            assert used.isdisjoint({4, 5}), f'Seeds 4/5 already prepared: {candidate}'
    batch.mkdir(parents=True, exist_ok=False)
    code = batch / 'code'
    shutil.copytree(source / 'code', code,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    data = batch / 'data'
    data.mkdir()
    target_data = data / Path(original['data_file']).name
    shutil.copy2(original['data_file'], target_data)
    assert digest(target_data) == original['data_sha256']
    (batch / 'configs').mkdir()
    tasks = []
    changes = {'seed', 'save_dir', 'save_name', 'data_dir'}
    for task in original['tasks']:
        old = yaml.safe_load(Path(task['config']).read_text())
        repeat, seed = task['repeat'] + 2, task['seed'] + 2
        assert (task['gpu'], repeat, seed) in ((0, 3, 4), (1, 4, 5))
        name = task['name'].replace(
            f"__repeat{task['repeat']}_seed{task['seed']}",
            f'__repeat{repeat}_seed{seed}')
        output = batch / 'models' / name
        config_path = batch / 'configs' / (name + '.yaml')
        config = dict(old)
        config.update(seed=seed, save_dir=str(batch / 'models'), save_name=name,
                      data_dir=str(data))
        assert config['overwrite'] is False and config['resume'] is False
        assert config['epoch'] == 100 and config['num_train_iter'] == 20000
        assert {k: v for k, v in config.items() if k not in changes} == {
            k: v for k, v in old.items() if k not in changes}
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        commands = []
        for command in task['commands']:
            rewritten = []
            for token in command:
                token = token.replace(str(source), str(batch))
                token = token.replace(task['name'], name)
                rewritten.append(token)
            if '--seed' in rewritten:
                rewritten[rewritten.index('--seed') + 1] = str(seed)
            commands.append(rewritten)
        updated = dict(task, name=name, seed=seed, repeat=repeat,
                       config=str(config_path), output=str(output), commands=commands)
        assert all(str(source) not in token for command in commands for token in command)
        for command in commands:
            assert Path(command[0]).is_file()
            if '--c' in command:
                assert command[command.index('--c') + 1] == str(config_path)
            else:
                assert command[command.index('--seed') + 1] == str(seed)
                assert command[command.index('--out_dir') + 1] == str(output)
        tasks.append(updated)
    assert len(tasks) == len({task['name'] for task in tasks}) == 96
    for gpu in (0, 1):
        group = [task for task in tasks if task['gpu'] == gpu]
        assert len(group) == 48
        assert [t['case'] for t in group] == [c for c in range(1, 7) for _ in original['methods']]
        assert {t['seed'] for t in group} == {gpu + 4}
    manifest = dict(original)
    from datetime import datetime, timezone, timedelta
    manifest.update(created=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    code_snapshot=str(code), data_file=str(target_data), tasks=tasks,
                    source_batch=str(source), additional_repeats=[3, 4])
    manifest['protocol'] = dict(original['protocol'], seeds=[4, 5], repeat_ids=[3, 4])
    (batch / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    hashes = {str(p.relative_to(code)): digest(p) for p in code.rglob('*') if p.is_file()}
    # The training code must exactly match the previously completed experiment.
    expected_hashes = json.loads((source / 'source_sha256.json').read_text())
    assert all(hashes.get(k) == v for k, v in expected_hashes.items())
    (batch / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    with (batch / 'split_verification.log').open('w') as handle:
        result = subprocess.run([sys.executable, str(code / 'scripts/audit_signal_splits.py'),
                                 str(batch)], cwd=code, stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (batch / 'split_verification.log').read_text()
    (batch / 'repeat_verification.json').write_text(json.dumps({
        'source_batch': str(source), 'tasks': 96, 'seeds': [4, 5], 'repeat_ids': [3, 4],
        'training_source_matches_previous_batch': True,
        'data_matches_previous_batch': True,
        'hyperparameters_match_previous_batch': True,
        'split_audit_passed': True,
        'gpu_task_counts': {'0': 48, '1': 48},
    }, indent=2) + '\n')
    (batch / 'README.md').write_text(
        '# 追加第 3、4 次重复\n\n'
        '共 96 次新增训练：8 方法 × 6 配置 × 2 次重复。\n'
        'GPU 0：repeat 3 / seed 4；GPU 1：repeat 4 / seed 5。\n'
        '沿用上一批实际代码、数据、方法顺序、训练参数及评估口径；每张卡按 run_1 到 run_6 执行。\n'
        '旧 seed 2、3 的全部结果保留在原批次，不会覆盖。\n\n'
        f'原批次：{source}\n\n'
        f'查看进度：`{sys.executable} {ROOT / "status_signal_baselines.py"} {batch}`\n\n'
        '汇总见 `summary/comparison.md`、`summary/runs.csv`。\n'
        '异常记录在 `issues_gpu0.jsonl`、`issues_gpu1.jsonl`，失败会重试一次。\n')
    print(f'Prepared and verified: {batch}; 96 new tasks, seeds 4/5', flush=True)


def launch(batch):
    code = batch / 'code'
    with (batch / 'queue.log').open('ab') as handle:
        process = subprocess.Popen([sys.executable, '-u', str(code / 'run_signal_baselines.py'),
                                    str(batch)], cwd=code, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    (batch / 'launcher.pid').write_text(str(process.pid) + '\n')
    print(f'Background supervisor PID: {process.pid}', flush=True)
    for _ in range(20):
        if process.poll() is not None:
            raise RuntimeError((batch / 'queue.log').read_text())
        states = [json.loads(p.read_text()) for p in (batch / 'state').glob('*.json')]
        running = [s for s in states if s['state'] == 'running']
        if len(running) == 2 and all('pid' in s for s in running):
            for state in running:
                os.kill(state['pid'], 0)
                print(f"GPU {state['gpu']}: {state['name']}, PID {state['pid']}", flush=True)
            print('Queue states:', dict(Counter(s['state'] for s in states)), flush=True)
            return
        time.sleep(1)
    raise RuntimeError('Supervisor started; inspect queue logs before attempting another launch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'logs/signal_baselines_96_20261027')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    batch = args.batch.resolve()
    prepare(args.source.resolve(), batch)
    if args.launch:
        launch(batch)
