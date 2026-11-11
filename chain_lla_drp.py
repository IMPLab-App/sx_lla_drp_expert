"""Keep active LLA trainers untouched, then immediately run the DRP successor.

Only the predecessor's supervisory process is held; its GPU worker subprocesses
continue uninterrupted. On release it performs its normal verification and
restores the original paused experiments. A separate guard handles parent loss.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil
from lla_experiment_utils import sha256
from run_lla_drp import verify_sources,verify_predecessor,verify_reference_matching,wait_idle
from summarize_lla_drp import summarize
from plot_lla_drp import plot

ROOT=Path(__file__).resolve().parent


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def atomic(path,value):
    temp=path.with_name(path.name+'.partial')
    with temp.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def match(item):
    try:
        p=psutil.Process(item['pid'])
        if abs(p.create_time()-item['created'])<.01 and p.status()!=psutil.STATUS_ZOMBIE:return p
    except psutil.NoSuchProcess:pass
    return None


def state(batch,phase,**extra):
    record=dict(phase=phase,updated_at=now(),pid=os.getpid(),**extra)
    atomic(batch/'pipeline_status.json',record)
    with (batch/'pipeline_events.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    (batch/'reports/TRAINING_STATUS.md').write_text('# LLA+DRP pipeline status\n\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n')


def cleanup_own_training(batch):
    # Match exact new-study executables/configuration paths. Never terminate
    # the original queue or predecessor LLA trainers.
    scripts={str(batch/'code/train_lla_drp.py'),str(batch/'code/verify_lla_drp_gpu.py')}
    worker_script=str(ROOT/'run_lla_drp.py')
    targets=[]
    for process in psutil.process_iter(['pid','cmdline']):
        cmd=process.info.get('cmdline') or []
        if len(cmd)<2:continue
        if any(script in cmd for script in scripts) or (worker_script in cmd and str(batch) in cmd):targets.append(process)
    for process in targets:
        try:
            for child in process.children(recursive=True):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            process.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(targets,timeout=15)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(alive,timeout=10)


def release_predecessor(batch):
    path=batch/'handoff_ownership.json'
    if not path.exists():return
    record=json.loads(path.read_text());process=match(record['predecessor_supervisor'])
    if process is not None and record.get('was_running_before_handoff',False):process.resume()
    record.update(released_at=now(),released=True);atomic(path,record)


def guard(batch,owner):
    while match(owner) is not None:time.sleep(5)
    ownership=batch/'handoff_ownership.json'
    if ownership.exists() and not json.loads(ownership.read_text()).get('released',False):
        cleanup_own_training(batch);release_predecessor(batch)
        state(batch,'recovery_guard_released_predecessor',owner=owner)


def wait_processes(processes):
    while any(p.poll() is None for p in processes):
        failures=[p.returncode for p in processes if p.poll() not in (None,0)]
        if failures:raise RuntimeError('New-study worker failed: '+str(failures))
        time.sleep(2)
    if any(p.returncode for p in processes):raise RuntimeError('New-study worker failed after poll')


def start_pair(batch,script,arguments,env,tag):
    children=[]
    for gpu in (0,1):
        stream=(batch/f'{tag}_gpu{gpu}.log').open('a')
        child=subprocess.Popen([sys.executable,'-u',str(script),'--batch',str(batch),'--gpu',str(gpu)]+arguments,
            cwd=script.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        stream.close();children.append(child)
    wait_processes(children)


def completed_stage(batch,version):
    m=json.loads((batch/'manifest.json').read_text())
    for task in (t for t in m['tasks'] if t['version']==version):
        out=Path(task['output']);proof=json.loads((out/'checkpoint_manifest.json').read_text())
        if proof['config_sha256']!=task['config_sha256']:raise RuntimeError('Completed task config mismatch')
        for value in proof['checkpoints'].values():
            if sha256(value['path'])!=value['sha256']:raise RuntimeError('Completed checkpoint hash mismatch')
        assert (out/'validation_selection.json').exists()
        if version=='M2' and proof['mff_gradient_monitor']['nonzero']<=0:raise RuntimeError('M2 has no effective MFF training')


def main(args):
    batch=args.batch.resolve()
    if args.guard:
        guard(batch,dict(pid=args.owner_pid,created=args.owner_created));return 0
    lock=(batch/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources(batch)
    request=json.loads(args.request.read_text());prior=match(request['predecessor_supervisor'])
    if prior is None:raise RuntimeError('Predecessor supervisor exited before handoff; do not preempt other experiments')
    if str(ROOT/'prioritize_lla_e3_stage1.py') not in prior.cmdline():raise RuntimeError('Unexpected predecessor supervisor')
    ownership=dict(predecessor_supervisor=request['predecessor_supervisor'],was_running_before_handoff=prior.status()!=psutil.STATUS_STOPPED,
        acquired_at=now(),released=False,scope='supervisor only; active GPU trainers must continue')
    atomic(batch/'handoff_ownership.json',ownership)
    owner=psutil.Process(os.getpid())
    with (batch/'recovery_guard.log').open('a') as stream:
        guardian=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--batch',str(batch),
            '--guard','--owner-pid',str(owner.pid),'--owner-created',str(owner.create_time())],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    env=dict(os.environ,LLA_DRP_BATCH=str(batch),LLA_PAUSED_PROCESS_MANIFEST=request['original_pause_manifest'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    success=False;failure=None
    def terminate(signum,frame):raise KeyboardInterrupt('Chain supervisor interrupted')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        # Hold only the process which would restore the original queue at the
        # old boundary. Its independent GPU workers keep their exact execution.
        prior.suspend()
        old_workers=[]
        for process in prior.children():
            cmd=process.cmdline()
            if str(ROOT/'run_lla_e3_stage1.py') in cmd:old_workers.append(dict(pid=process.pid,created=process.create_time(),gpu=int(cmd[cmd.index('--gpu')+1])))
        state(batch,'waiting_for_predecessor_18',predecessor_supervisor=prior.pid,recovery_guard_pid=guardian.pid,
              old_gpu_workers=old_workers,active_lla_trainers_interrupted=False)
        predecessor=Path(request['predecessor_batch'])
        old=json.loads((predecessor/'manifest.json').read_text())
        last_count=-1
        while True:
            if match(request['predecessor_supervisor']) is None:
                raise RuntimeError('Predecessor supervisor disappeared during handoff')
            complete=sum((Path(t['output'])/'checkpoint_manifest.json').exists() and (Path(t['output'])/'validation_selection.json').exists() for t in old['tasks'])
            if complete!=last_count:
                state(batch,'waiting_for_predecessor_18',completed_predecessor_runs=complete,total=18,active_lla_trainers_interrupted=False)
                last_count=complete
            if complete==18 and all(match(item) is None for item in old_workers):break
            for item in old_workers:
                if match(item) is None and complete<18:
                    # One GPU shard may finish earlier; fail only if it left
                    # any of its own nine expected tasks incomplete.
                    expected=[t for t in old['tasks'] if (int(t['scenario'][2:])-1)%2 == item['gpu']]
                    if expected and any(not (Path(t['output'])/'checkpoint_manifest.json').exists() for t in expected):
                        raise RuntimeError('Predecessor GPU worker exited before its expected tasks completed')
            time.sleep(2)
        verify_predecessor(batch);verify_reference_matching(batch)
        state(batch,'gpu_preflight',predecessor_runs_verified=18)
        m=json.loads((batch/'manifest.json').read_text());code=Path(m['code_snapshot'])
        for gpu in (0,1):wait_idle(gpu)
        start_pair(batch,code/'verify_lla_drp_gpu.py',[],env,'gpu_preflight')
        for gpu in (0,1):
            proof=json.loads((batch/'smoke_tests'/f'gpu_result_{gpu}.json').read_text())
            if proof['status']!='passed':raise RuntimeError('GPU preflight failed')
            for name,h in proof['source_sha256'].items():
                if sha256(code/name)!=h:raise RuntimeError('GPU source proof mismatch')
        state(batch,'training_full_six',predecessor_runs_verified=18)
        start_pair(batch,ROOT/'run_lla_drp.py',['--stage','full'],env,'full')
        completed_stage(batch,'M0');gate=summarize(batch);plot(batch)
        if gate['eligible_for_m2']:
            state(batch,'training_fixed_dea_six',validation_gate=gate)
            start_pair(batch,ROOT/'run_lla_drp.py',['--stage','m2'],env,'m2')
            completed_stage(batch,'M2');summarize(batch);plot(batch)
        else:state(batch,'full_complete_gate_failed',validation_gate=gate,m2_started=False)
        success=True
    except BaseException as exception:
        failure=repr(exception);cleanup_own_training(batch)
    finally:
        release_predecessor(batch)
        state(batch,'complete' if success else 'failed',error=failure,original_queue_restore_delegated_to_predecessor=True,
              r_cd_54_started=False)
    return 0 if success else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--request',type=Path);p.add_argument('--guard',action='store_true')
    p.add_argument('--owner-pid',type=int);p.add_argument('--owner-created',type=float)
    sys.exit(main(p.parse_args()))
