"""Transfer the legacy scheduler only after its current task is fully committed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs/cola_two_kl_24'
REQUEST = LOGS / 'handoff_request.json'
READY = LOGS / 'handoff_ready.json'


def is_legacy_scheduler(pid):
    try:
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return 'cola_two_kl_queue.py' in command and '--worker' not in command
    except FileNotFoundError:
        return False


def after_summary():
    """Called by the old scheduler's own summary child at its task boundary."""
    if not REQUEST.exists():
        return
    request = json.loads(REQUEST.read_text())
    parent = os.getppid()
    if parent != request['legacy_pid'] or not is_legacy_scheduler(parent):
        return
    current = json.loads((LOGS / 'current.json').read_text())
    if current['stage'] != 'task' or current['status'] != 'complete':
        raise RuntimeError('Scheduler handoff requires a committed, evaluated task')
    from cola_two_kl_queue import atomic_json, now
    # Stop the scheduler itself here. Training and evaluation already exited.
    os.kill(parent, signal.SIGSTOP)
    atomic_json(READY, dict(legacy_pid=parent, time=now(), completed_task=current))
    print('Legacy scheduler held at completed-task boundary for dual-GPU handoff', flush=True)


def launch_worker(worker):
    from cola_two_kl_queue import atomic_json, now
    folder = LOGS / f'gpu_{worker}';folder.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env.update(GPU_ID=str(worker), PYTHON_BIN=sys.executable, TZ='Asia/Shanghai')
    with (folder / 'nohup.log').open('ab',buffering=0) as stream:
        process = subprocess.Popen(['nohup', str(ROOT/'run_cola_two_kl_24.sh'), '--worker', str(worker)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    atomic_json(folder/'launch.json', dict(pid=process.pid,gpu=worker,time=now(),python=sys.executable))
    return process.pid


def supervise():
    from cola_two_kl_queue import atomic_json, now
    import fcntl
    with (LOGS/'handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        request=json.loads(REQUEST.read_text()); old=request['legacy_pid']
        atomic_json(LOGS/'handoff_supervisor.json',dict(pid=os.getpid(),status='waiting_for_task_boundary',legacy_pid=old,time=now()))
        print(f'{now()} Waiting for scheduler {old} to finish its current task',flush=True)
        while not READY.exists():
            if not is_legacy_scheduler(old):
                raise RuntimeError('Legacy scheduler exited before a safe handoff; inspect its logs')
            # Keep the GPU-0 view current while the legacy scheduler finishes.
            for name in ('queue.json', 'current.json'):
                source = LOGS / name
                if source.exists():
                    atomic_json(LOGS/'gpu_0'/name, dict(json.loads(source.read_text()), handoff_pending=True))
            time.sleep(1)
        ready=json.loads(READY.read_text())
        if ready['legacy_pid'] != old or not is_legacy_scheduler(old):
            raise RuntimeError('Handoff PID mismatch')
        stat=(Path('/proc')/str(old)/'stat').read_text().rsplit(')',1)[1].split()[0]
        if stat not in ('T','t'):
            raise RuntimeError('Legacy scheduler was not held at the task boundary')
        # Its summary child has established the safe point. Replacing this idle
        # scheduler cannot stop a training or evaluation process.
        os.kill(old,signal.SIGKILL)
        # Its tee process inherited the legacy lock. Wait for the stdout pipe to
        # close before starting the successor, so no transient lock race occurs.
        deadline=time.monotonic()+30
        with (LOGS/'queue.lock').open('a') as legacy_lock:
            while True:
                try:
                    fcntl.flock(legacy_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(legacy_lock,fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    if time.monotonic()>deadline:
                        raise RuntimeError('Legacy queue lock did not release')
                    time.sleep(.2)
        pid=launch_worker(0)
        deadline=time.monotonic()+30
        while True:
            state=LOGS/'gpu_0/queue.json'
            if state.exists() and json.loads(state.read_text()).get('pid')==pid:
                if json.loads(state.read_text()).get('status')=='failed':
                    raise RuntimeError('GPU 0 successor failed to start')
                break
            if time.monotonic()>deadline:
                raise RuntimeError('GPU 0 successor did not report its running state')
            time.sleep(.5)
        atomic_json(LOGS/'handoff_supervisor.json',dict(pid=os.getpid(),status='complete',legacy_pid=old,worker0_pid=pid,time=now()))
        atomic_json(LOGS/'queue.json',dict(status='running',scheduler='dual_gpu',gpu0_pid=pid,time=now(),worker_states=['gpu_0/queue.json','gpu_1/queue.json']))
        print(f'{now()} Replaced idle legacy scheduler; GPU 0 worker PID={pid}',flush=True)


if __name__=='__main__':
    try:
        supervise()
    except BaseException as error:
        from cola_two_kl_queue import atomic_json, now
        atomic_json(LOGS/'handoff_supervisor.json',dict(pid=os.getpid(),status='failed',error=str(error),time=now()))
        atomic_json(LOGS/'stop_requested.json',dict(error=str(error),source='handoff',time=now()))
        raise
