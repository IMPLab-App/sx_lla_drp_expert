"""Wait for all eleven baseline runs, then run the verified 22-task batch."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_signal_baselines import atomic_json, now
from summarize_bacn_ablation import summarize


def dependency_ready(batch):
    manifest = json.loads((batch / 'manifest.json').read_text())
    baseline = Path(manifest['baseline_batch'])
    content = (baseline / 'manifest.json').read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest['baseline_manifest_sha256']:
        raise RuntimeError('Baseline manifest changed after preparation')
    original = json.loads(content)
    status = (baseline / 'status.txt').read_text().strip()
    if status in ('failed', 'completed_with_failures'):
        raise RuntimeError(f'Baseline failed: {status}; refusing to start successor')
    if status != 'completed':
        pid = int((baseline / 'queue.pid').read_text())
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        except FileNotFoundError as error:
            raise RuntimeError('Baseline queue exited without successful completion') from error
        if original['code_snapshot'] not in command or 'run_signal_baselines.py' not in command:
            raise RuntimeError('Baseline PID no longer belongs to its scheduler')
        return False
    if len(original['tasks']) != 11:
        raise RuntimeError('Expected exactly eleven baseline tasks')
    for task in original['tasks']:
        state = json.loads((baseline / 'state' / (task['name'] + '.json')).read_text())
        output = Path(task['output'])
        if state['state'] != 'completed' or not (output / 'model_best.pth').is_file():
            raise RuntimeError(f'Incomplete baseline artifacts: {task["name"]}')
        metrics = json.loads((output / 'test_metrics.json').read_text())
        if metrics['samples'] != 900:
            raise RuntimeError(f'Wrong held-out sample count: {task["name"]}')
    return True


def main(batch):
    with (batch / 'successor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((batch / 'manifest.json').read_text())
        audit = json.loads((batch / 'verification.json').read_text())
        if audit['status'] != 'passed' or not audit['baseline_reuse']['passed']:
            raise RuntimeError('A passing baseline-reuse audit is required')
        if not audit.get('all_22_smoke_train_reload_tests_passed'):
            raise RuntimeError('All 22 short train/reload/test checks are required')
        for name, expected in json.loads((batch / 'source_sha256.json').read_text()).items():
            if hashlib.sha256((Path(manifest['code_snapshot']) / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen source modified: {name}')
        (batch / 'state').mkdir(exist_ok=True)
        for task in manifest['tasks']:
            path = batch / 'state' / (task['name'] + '.json')
            if not path.exists():
                atomic_json(path, dict(name=task['name'], gpu=task['gpu'], state='pending', updated=now()))
        (batch / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
        (batch / 'status.txt').write_text('waiting_for_baseline\n')
        print(f'[{now()}] Waiting for all eleven baseline tasks', flush=True)
        while not dependency_ready(batch):
            atomic_json(batch / 'successor.json', dict(pid=os.getpid(), state='waiting_for_baseline',
                                                      baseline=manifest['baseline_batch'], updated=now()))
            summarize(batch)
            time.sleep(15)
        atomic_json(batch / 'successor.json', dict(pid=os.getpid(), state='starting', updated=now()))
        print(f'[{now()}] Baseline fully completed; starting 22 new tasks', flush=True)
        process = subprocess.Popen([sys.executable, str(batch / 'code/run_signal_baselines.py'), str(batch)],
                                   cwd=manifest['code_snapshot'], stdin=subprocess.DEVNULL)
        atomic_json(batch / 'successor.json', dict(pid=os.getpid(), queue_pid=process.pid,
                                                  state='running', updated=now()))
        while process.poll() is None:
            summarize(batch)
            time.sleep(15)
        summarize(batch)
        atomic_json(batch / 'successor.json', dict(pid=os.getpid(), queue_pid=process.pid,
                                                  state='completed' if process.returncode == 0 else 'failed',
                                                  exit_code=process.returncode, updated=now()))
        return process.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    try:
        sys.exit(main(batch))
    except BlockingIOError:
        print('This batch already has a successor supervisor; leaving its state unchanged.', file=sys.stderr)
        sys.exit(2)
    except Exception as error:
        (batch / 'status.txt').write_text('successor_failed\n')
        atomic_json(batch / 'successor.json', dict(pid=os.getpid(), state='failed',
                                                  error=str(error), updated=now()))
        raise
