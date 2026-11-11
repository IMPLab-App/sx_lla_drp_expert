"""Retry an authorized DRP queue after the completed predecessor has exited.

Training code/configurations remain frozen. The current original queue is
reversibly suspended, with a separate recovery process restoring it on failure.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from chain_lla_drp import (atomic, cleanup_own_training, completed_stage, match,
    now, start_pair, state)
from lla_experiment_utils import sha256
from plot_lla_drp import plot
from prioritize_lla_e3_stage1 import identity, resume_old
from run_lla_drp import (verify_predecessor, verify_reference_matching,
    verify_sources, wait_idle)
from summarize_lla_drp import summarize

ROOT = Path(__file__).resolve().parent


def suspend_recorded_tree(manifest, root_item):
    root = match(root_item)
    if root is None:
        raise RuntimeError('Recorded original supervisor no longer exists')
    pause = dict(authorization='user_requested_priority_pause',
        user_instruction='直接开始训练，原有训练等这个训练完成后再继续进行训练',
        state='pausing', created_at=now(), root=root_item,
        processes=[identity(p) for p in [root] + root.children(recursive=True)])
    atomic(manifest, pause)
    # Suspend supervisors before descendants, rescanning for startup races.
    for _ in range(3):
        known = {p['pid'] for p in pause['processes']}
        for process in root.children(recursive=True):
            if process.pid not in known:
                pause['processes'].append(identity(process))
                known.add(process.pid)
        atomic(manifest, pause)
        for item in pause['processes']:
            process = match(item)
            if process is not None and process.status() != psutil.STATUS_STOPPED:
                process.suspend()
        time.sleep(.25)
    running = [item['pid'] for item in pause['processes']
        if (p := match(item)) is not None and p.status() != psutil.STATUS_STOPPED]
    if running:
        raise RuntimeError('Original queue not fully paused: ' + str(running))
    pause.update(state='paused', paused_at=now())
    atomic(manifest, pause)
    return pause


def guard(batch, pause, owner):
    while match(owner) is not None:
        time.sleep(3)
    if pause.exists() and json.loads(pause.read_text()).get('state') != 'resumed':
        cleanup_own_training(batch)
        restored = resume_old(pause)
        state(batch, 'retry_guard_restored_original', resumed_original_pids=restored)


def run(args):
    batch = args.batch.resolve()
    pause = args.pause_manifest.resolve()
    if args.guard:
        guard(batch, pause, dict(pid=args.owner_pid, created=args.owner_created))
        return 0
    lock = (batch/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources(batch)
    verify_predecessor(batch)
    verify_reference_matching(batch)
    request = json.loads(args.request.read_text())
    root = match(request['original_supervisor'])
    if root is None or root.cmdline() != request['expected_command']:
        raise RuntimeError('Original queue identity changed; no processes paused')
    if pause.exists():
        raise RuntimeError('Use a new pause manifest for every attempt')
    owner = psutil.Process(os.getpid())
    with (batch/'retry_recovery_guard.log').open('a') as stream:
        guardian = subprocess.Popen([sys.executable, '-u', str(Path(__file__)),
            '--batch', str(batch), '--pause-manifest', str(pause), '--guard',
            '--owner-pid', str(owner.pid), '--owner-created', str(owner.create_time())],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Retry controller interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    success = False
    failure = None
    try:
        state(batch, 'retry_pausing_original', recovery_guard_pid=guardian.pid,
              pause_manifest=str(pause), reason='parent GPU check lacked pause-manifest environment')
        suspend_recorded_tree(pause, request['original_supervisor'])
        # Crucial: both parent checks AND subprocesses require the authorization.
        os.environ['LLA_PAUSED_PROCESS_MANIFEST'] = str(pause)
        os.environ['LLA_DRP_BATCH'] = str(batch)
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        env.pop('CUDA_VISIBLE_DEVICES', None)
        for gpu in (0, 1):
            wait_idle(gpu)
        code = Path(json.loads((batch/'manifest.json').read_text())['code_snapshot'])
        state(batch, 'gpu_preflight', predecessor_runs_verified=18,
              pause_manifest=str(pause), recovery_guard_pid=guardian.pid)
        start_pair(batch, code/'verify_lla_drp_gpu.py', [], env, 'gpu_preflight_retry')
        for gpu in (0, 1):
            proof = json.loads((batch/'smoke_tests'/f'gpu_result_{gpu}.json').read_text())
            if proof['status'] != 'passed':
                raise RuntimeError('GPU preflight failed')
            for name, expected in proof['source_sha256'].items():
                if sha256(code/name) != expected:
                    raise RuntimeError('GPU proof source mismatch')
        state(batch, 'training_full_six', predecessor_runs_verified=18,
              pause_manifest=str(pause), recovery_guard_pid=guardian.pid)
        start_pair(batch, ROOT/'run_lla_drp.py', ['--stage', 'full'], env, 'full')
        completed_stage(batch, 'M0')
        gate = summarize(batch)
        plot(batch)
        if gate['eligible_for_m2']:
            state(batch, 'training_fixed_dea_six', validation_gate=gate)
            start_pair(batch, ROOT/'run_lla_drp.py', ['--stage', 'm2'], env, 'm2')
            completed_stage(batch, 'M2')
            summarize(batch)
            plot(batch)
        else:
            state(batch, 'full_complete_gate_failed', validation_gate=gate, m2_started=False)
        success = True
    except BaseException as exception:
        failure = repr(exception)
        cleanup_own_training(batch)
    finally:
        restored = resume_old(pause)
        state(batch, 'complete' if success else 'failed', error=failure,
              resumed_original_pids=restored, r_cd_54_started=False)
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--pause-manifest', type=Path, required=True)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--owner-pid', type=int)
    parser.add_argument('--owner-created', type=float)
    sys.exit(run(parser.parse_args()))
