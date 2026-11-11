"""Strictly serial training/evaluation driver, guarded by the Bash flock wrapper."""
import argparse
import contextlib
import fcntl
import tempfile
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'config/cola_two_kl_24/manifest.json'
LOGS = ROOT / 'logs/cola_two_kl_24'


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temp = Path(stream.name)
    temp.replace(path)


def state_dir():
    worker = os.environ.get('COLA_QUEUE_WORKER')
    return LOGS / f'gpu_{worker}' if worker is not None else LOGS


def select_tasks(tasks, worker=None):
    if len(tasks) != 24:
        raise RuntimeError('Expected exactly 24 tasks')
    if worker is None:
        return tasks
    if worker not in (0, 1):
        raise ValueError('Worker must be 0 or 1')
    return tasks[worker * 12:(worker + 1) * 12]


@contextlib.contextmanager
def task_lock(task):
    folder = LOGS / 'task_locks'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (task['variant'] + '__' + task['run_name'] + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Task is already owned by another worker: {task['run_name']}")
        yield


def status(task, stage, state, pid='', code=''):
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / 'status.tsv'
    record = dict(time=now(), variant=task['variant'], run_name=task['run_name'],
                  seed=task['seed'], stage=stage, status=state, pid=pid, exit_code=code)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write('\t'.join(record) + '\n')
        stream.write('\t'.join(str(x) for x in record.values()) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(state_dir() / 'current.json', dict(record, gpu=os.environ.get('GPU_ID', '0')))
    print(json.dumps(record), flush=True)


def validate_task(task):
    config = ROOT / task['config']
    if sha(config) != task['config_sha256']:
        raise RuntimeError(f'Config changed since preparation: {config}')
    import yaml
    from prepare_cola_two_kl_24 import PROTECTED
    cfg = yaml.safe_load(config.read_text())
    original = yaml.safe_load((ROOT / task['source']).read_text())
    if any(cfg[k] != original[k] for k in PROTECTED):
        raise RuntimeError('Protected baseline training/data settings changed')
    if cfg['gpu'] != 0 or cfg['world_size'] != 1 or cfg['multiprocessing_distributed'] or cfg['overwrite']:
        raise RuntimeError('Queue requires logical gpu=0, one process, overwrite=False')
    expected = ROOT / 'saved_models_cola_two_kl' / task['variant'] / cfg['save_name']
    if (ROOT / task['model_dir']).resolve() != expected.resolve():
        raise RuntimeError('Unexpected output path')


def validate_outputs(task):
    import torch
    model_dir = ROOT / task['model_dir']
    latest_path, best_path = model_dir / 'latest_model.pth', model_dir / 'model_best.pth'
    latest = torch.load(latest_path, map_location='cpu', weights_only=False)
    for key in ('model', 'ema_model', 'optimizer', 'best_it', 'epoch', 'it'):
        if key not in latest:
            raise RuntimeError(f'Incomplete latest checkpoint: missing {key}')
    if latest['it'] != task['num_train_iter']:
        raise RuntimeError(f'Training incomplete: {latest["it"]}/{task["num_train_iter"]}')
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    if not best.get('ema_model') or best.get('best_it', -1) != latest['best_it']:
        raise RuntimeError('Best/latest checkpoints disagree')
    if task['variant'] != 'no_ddde_kl' and 'cola_two_kl' not in latest:
        raise RuntimeError('DDDE checkpoint state is missing')
    evaluation_path = model_dir / 'evaluation/evaluation.json'
    metrics = json.loads(evaluation_path.read_text())
    if not metrics.get('complete') or Path(metrics['checkpoint']).resolve() != best_path.resolve():
        raise RuntimeError('Evaluation missing or refers to a different checkpoint')
    for split in ('heldout', 'balanced', 'imbalanced'):
        for key in ('top1_accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
            value = metrics[split][key]
            if not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise RuntimeError(f'Invalid evaluation metric: {split}/{key}')
        for key in ('confusion_matrix_csv', 'confusion_matrix_png'):
            if not Path(metrics[split][key]).is_file():
                raise RuntimeError('Evaluation artifact missing')
    return dict(config_sha256=task['config_sha256'], latest_sha256=sha(latest_path),
                best_sha256=sha(best_path), evaluation_sha256=sha(evaluation_path))


def completed(task):
    marker = ROOT / task['model_dir'] / 'completed.json'
    if not marker.exists():
        return False
    expected = json.loads(marker.read_text())
    actual = validate_outputs(task)
    if any(expected.get(k) != v for k, v in actual.items()):
        raise RuntimeError('Completed artifact fingerprints changed; refusing to skip or overwrite')
    return True


def run_stage(task, stage, command, log_path):
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"{now()} task={task['run_name']} variant={task['variant']} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} stage={stage} log={log_path}", flush=True)
    child = None
    try:
        with open(log_path, 'x') as stream:
            stream.write(f'START {now()} command={json.dumps(command)}\n'); stream.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status(task, stage, 'running', child.pid)
            code = child.wait()
            stream.write(f'END {now()} pid={child.pid} exit_code={code}\n')
        status(task, stage, 'success' if code == 0 else 'failed', child.pid, code)
        if code:
            raise RuntimeError(f'{stage} failed with exit {code}; inspect {log_path}')
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            status(task, stage, 'failed', child.pid, child.returncode)
        raise


def run_queue(dry_run=False, worker=None):
    tasks = json.loads(MANIFEST.read_text())['tasks']
    tasks = select_tasks(tasks, worker)
    for task in tasks:
        validate_task(task)
    gpu = os.environ.get('GPU_ID', '0')
    if dry_run:
        for i, task in enumerate(tasks, 1):
            print(f"DRY RUN {i:02d}/{len(tasks)} task={task['run_name']} variant={task['variant']} GPU={gpu} time={now()} train_log={LOGS / (task['variant'] + '__' + task['run_name'] + '.train.log')}")
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU')
    atomic_json(state_dir() / 'queue.json', dict(pid=os.getpid(), python=sys.executable,
                                         gpu=gpu, start=now(), status='running'))
    active = tasks[0]
    try:
        for active in tasks:
            if (LOGS / 'stop_requested.json').exists():
                raise RuntimeError('Another worker failed; no new tasks will be started')
            if completed(active):
                status(active, 'task', 'skipped_complete', code=0)
                continue
            with task_lock(active):
                model_dir = ROOT / active['model_dir']
                if model_dir.exists():
                    raise RuntimeError(f'Existing unfinished output preserved: {model_dir}. Resolve manually before relaunch.')
                stem = active['variant'] + '__' + active['run_name']
                train_log, eval_log = LOGS / (stem + '.train.log'), LOGS / (stem + '.eval.log')
                if train_log.exists() or eval_log.exists():
                    raise RuntimeError(f'Existing task logs preserved: {stem}')
                run_stage(active, 'train', [sys.executable, '-u', 'train.py', '--c', active['config']], train_log)
                # Evaluation uses the same config/seed and the existing evaluation tool.
                run_stage(active, 'eval', [sys.executable, '-u', 'eval_confusion_matrix_new.py',
                           '--c', active['config'], '--load_path', str(model_dir / 'model_best.pth'),
                           '--output_dir', str(model_dir / 'evaluation'), '--gpu', '0'], eval_log)
                fingerprints = validate_outputs(active)
                atomic_json(model_dir / 'completed.json', dict(**fingerprints, completed=now()))
                status(active, 'task', 'complete', code=0)
                subprocess.run([sys.executable, 'summarize_cola_two_kl_24.py'], cwd=ROOT, check=True)
        atomic_json(state_dir() / 'queue.json', dict(pid=os.getpid(), gpu=gpu, end=now(), status='complete'))
    except BaseException as error:
        atomic_json(LOGS / 'stop_requested.json', dict(time=now(), worker=worker, task=active['run_name'], error=str(error)))
        status(active, 'task', 'failed', code=1)
        atomic_json(state_dir() / 'queue.json', dict(pid=os.getpid(), gpu=gpu, end=now(), status='failed', error=str(error)))
        subprocess.run([sys.executable, 'summarize_cola_two_kl_24.py'], cwd=ROOT, check=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is not None:
        os.environ['COLA_QUEUE_WORKER'] = str(args.worker)
    def interrupted(signum, frame):
        raise InterruptedError(f'Queue interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run_queue(args.dry_run, args.worker)
