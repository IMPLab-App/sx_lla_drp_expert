"""Durable, technically checked continuation; no validation performance gate.

Hold only the current supervisor. Its active Full GPU workers continue. Finish
Full, run six fixed-weight LU ablations, then the unchanged R/CD 54 experiments.
Preserve failed attempts and resume valid checkpoints; restore the original queue
through its existing supervisor on completion or controller loss.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import psutil
import yaml
from chain_lla_drp import atomic,match,cleanup_own_training
from lla_experiment_utils import sha256
from run_lla_drp import verify_sources,wait_idle
from summarize_lla_drp_expansion import summarize

ROOT=Path(__file__).resolve().parent


def timestamp():return datetime.now().astimezone().isoformat()


def record(expansion,phase,**extra):
    row=dict(phase=phase,pid=os.getpid(),updated_at=timestamp(),**extra)
    atomic(expansion/'status.json',row)
    with (expansion/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def problem(expansion,**extra):
    with (expansion/'problems.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=timestamp(),**extra),ensure_ascii=False)+'\n')


def verify_plan(expansion):
    proof=json.loads((expansion/'tests/config_and_integration.json').read_text())
    assert proof['status']=='passed' and proof['manifest_sha256']==sha256(expansion/'manifest.json')
    for name,digest in proof['source_sha256'].items():
        assert sha256(ROOT/name)==digest,'Continuation source differs from tested code: '+name
    return proof


def validate_task(task,hashes=True):
    out=Path(task['output']);proof=out/'checkpoint_manifest.json';sel=out/'validation_selection.json'
    if not proof.exists() or not sel.exists():return False
    p=json.loads(proof.read_text())
    if p['config_sha256']!=task['config_sha256']:raise ValueError('Completed configuration mismatch')
    assert p['selection']=='validation_macro_f1' and not p['test_evaluated']
    if hashes:
        for value in p['checkpoints'].values():
            assert sha256(value['path'])==value['sha256'],'Completed checkpoint hash mismatch'
    if task['version']=='M2':assert p['mff_gradient_monitor']['nonzero']>0
    return True


def execution_config(expansion,task,attempt):
    """Preserve every failed attempt before modifying its working continuation."""
    out=Path(task['output']);canonical=Path(task['config'])
    if not out.exists():return canonical,None
    stamp=str(time.time_ns());archive=expansion/'failed_attempts'/(task['scenario']+'_'+task['version']+'_'+stamp)
    checkpoint=out/'latest_model.pth'
    saved=None
    if checkpoint.exists():
        try:
            import torch
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert isinstance(saved['it'],int) and saved['it']>=0
            assert all(torch.isfinite(x).all() for x in saved['model'].values() if x.is_floating_point())
        except Exception as error:
            problem(expansion,task=task['scenario'],version=task['version'],action='archive_and_restart',reason=repr(error))
            saved=None
    if saved is None:
        shutil.move(str(out),str(archive))
        return canonical,dict(action='restart_same_seed',preserved_attempt=str(archive))
    shutil.copytree(out,archive)
    cursor=saved['it'];del saved
    # Preserve complete original logs in archive; remove only replayed suffix
    # from the working copy so plots do not double-count optimizer updates.
    for name in ('training_history.jsonl','drp_history.jsonl'):
        p=out/name
        if p.exists():
            keep=[]
            for line in p.read_text().splitlines():
                try:
                    if json.loads(line).get('iteration',-1)<cursor:keep.append(line)
                except ValueError:pass
            p.write_text('\n'.join(keep)+ ('\n' if keep else ''))
    for p in (out/'metrics').glob('*iter*.*'):
        try:iteration=int(p.stem.rsplit('iter',1)[1])
        except ValueError:continue
        if iteration>=cursor:p.unlink()
    cfg=yaml.safe_load(canonical.read_text());cfg.update(resume=True,load_path=str(checkpoint))
    p=expansion/'configs'/(task['scenario']+'_'+task['version']+f'_resume_{stamp}.yaml')
    p.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    return p,dict(action='resume_exact_checkpoint_state',cursor=cursor,checkpoint_sha256=sha256(checkpoint),
                  preserved_attempt=str(archive),execution_config=str(p),execution_config_sha256=sha256(p))


def task_list(expansion,stage):
    m=json.loads((expansion/'manifest.json').read_text())
    if stage=='full_recovery':
        parent=json.loads((Path(m['batch'])/'manifest.json').read_text())
        return [dict(t,stage=stage) for t in parent['tasks'] if t['version']=='M0']
    return [t for t in m['tasks'] if t['stage']==stage]


def worker(expansion,stage,gpu):
    lock=(expansion/f'{stage}_gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((expansion/'manifest.json').read_text());batch=Path(m['batch']);verify_sources(batch)
    verify_plan(expansion)
    hardware=json.loads((batch/'smoke_tests'/f'gpu_result_{gpu}.json').read_text());assert hardware['status']=='passed'
    code=Path(m['code_snapshot'])
    for name,digest in hardware['source_sha256'].items():assert sha256(code/name)==digest
    env=dict(os.environ,LLA_GPU_INDEX=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    failures=[]
    for task in task_list(expansion,stage):
        if task['gpu']!=gpu:continue
        if validate_task(task):continue
        assert sha256(task['config'])==task['config_sha256']
        success=False
        for attempt in range(1,4):
            log=expansion/'logs'/(stage+'_'+task['scenario']+'_'+task['version']+f'_attempt{time.time_ns()}.log')
            try:
                config,recovery=execution_config(expansion,task,attempt)
                wait_idle(gpu)
                if recovery:problem(expansion,task=task['scenario'],version=task['version'],recovery=recovery)
                with log.open('x') as stream:
                    subprocess.run([sys.executable,'-u',str(code/'train_lla_drp.py'),'--c',str(config)],
                        cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                output=Path(task['output']);p=output/'checkpoint_manifest.json';completed=json.loads(p.read_text())
                assert completed['config_sha256']==sha256(config)
                # Canonical protocol/config is unchanged by resume=True/load_path.
                if config!=Path(task['config']):
                    completed.update(config_sha256=task['config_sha256'],execution_config_sha256=sha256(config),
                        canonical_config=task['config'],resume_provenance=recovery)
                    atomic(p,completed)
                assert validate_task(task)
                success=True;break
            except Exception as error:
                tail=log.read_text()[-6000:] if log.exists() else ''
                problem(expansion,task=task['scenario'],version=task['version'],attempt=attempt,error=repr(error),
                        log=str(log),log_tail=tail,action='preserve_and_retry' if attempt<3 else 'record_failure_continue_other_tasks')
                time.sleep(3)
        if not success:failures.append(task['scenario']+'_'+task['version'])
    (expansion/f'{stage}_gpu{gpu}_result.json').write_text(json.dumps(dict(status='failed' if failures else 'passed',failures=failures))+'\n')
    return int(bool(failures))


def cleanup(expansion):
    m=json.loads((expansion/'manifest.json').read_text());cleanup_own_training(Path(m['batch']))
    own=[]
    for p in psutil.process_iter(['pid','cmdline']):
        cmd=p.info.get('cmdline') or []
        if str(ROOT/'run_lla_drp_expansion.py') in cmd and '--worker' in cmd and str(expansion) in cmd:own.append(p)
    for p in own:
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(own,timeout=10)
    for p in alive:
        try:p.kill()
        except psutil.NoSuchProcess:pass


def release(expansion):
    p=expansion/'ownership.json'
    if not p.exists():return
    data=json.loads(p.read_text());prior=match(data['supervisor'])
    if prior is not None and data['was_running']:prior.resume()
    data.update(released=True,released_at=timestamp());atomic(p,data)


def guard(expansion,owner):
    while match(owner) is not None:time.sleep(3)
    p=expansion/'ownership.json'
    if p.exists() and not json.loads(p.read_text()).get('released',False):
        cleanup(expansion);release(expansion)
        record(expansion,'recovery_guard_released_original_supervisor')


def start_workers(expansion,stage,env,gpus=(0,1)):
    children=[]
    for gpu in gpus:
        log=expansion/'logs'/f'{stage}_worker_gpu{gpu}_{time.time_ns()}.log'
        with log.open('x') as stream:
            children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                '--expansion',str(expansion),'--worker','--stage',stage,'--gpu',str(gpu)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
    while any(p.poll() is None for p in children):time.sleep(3)
    if any(p.returncode for p in children):
        raise RuntimeError('Stage completed with recorded failures: '+stage)


def run(args):
    expansion=args.expansion.resolve()
    if args.guard:
        guard(expansion,dict(pid=args.owner_pid,created=args.owner_created));return 0
    if args.worker:return worker(expansion,args.stage,args.gpu)
    lock=(expansion/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((expansion/'manifest.json').read_text());batch=Path(m['batch']);verify_sources(batch)
    verify_plan(expansion)
    request=json.loads(args.request.read_text());prior=match(request['supervisor'])
    if prior is None or prior.cmdline()!=request['expected_command']:
        raise RuntimeError('Current supervisor identity changed; no active jobs touched')
    # Verify all prepared task identities before taking ownership.
    for task in m['tasks']:
        assert sha256(task['config'])==task['config_sha256']
        cfg=yaml.safe_load(Path(task['config']).read_text())
        assert sha256(cfg['fixed_split_file'])==task['fixed_split_sha256']
    owner=psutil.Process(os.getpid())
    atomic(expansion/'ownership.json',dict(supervisor=request['supervisor'],was_running=prior.status()!=psutil.STATUS_STOPPED,
        acquired_at=timestamp(),released=False,scope='only retry supervisor; active GPU trainers continue'))
    with (expansion/'logs/recovery_guard.log').open('a') as stream:
        guardian=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--expansion',str(expansion),
            '--guard','--owner-pid',str(owner.pid),'--owner-created',str(owner.create_time())],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    def interrupted(signum,frame):raise KeyboardInterrupt('Expansion supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    success=False;failure=None
    env=dict(os.environ,LLA_PAUSED_PROCESS_MANIFEST=request['original_pause_manifest'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.environ['LLA_PAUSED_PROCESS_MANIFEST']=request['original_pause_manifest']
    env.pop('CUDA_VISIBLE_DEVICES',None)
    try:
        prior.suspend()
        old_workers=[]
        for process in prior.children():
            cmd=process.cmdline()
            if str(ROOT/'run_lla_drp.py') in cmd:
                old_workers.append(dict(pid=process.pid,created=process.create_time(),gpu=int(cmd[cmd.index('--gpu')+1])))
        assert old_workers,'No active Full workers; inspect boundary before handoff'
        atomic(expansion/'active_full_workers.json',old_workers)
        record(expansion,'waiting_for_full_six',performance_gate_controls_scheduling=False,
            active_trainers_interrupted=False,recovery_guard_pid=guardian.pid,queued_lu_m2=6,queued_rcd=54,
            held_supervisor=prior.pid)
        parent=json.loads((batch/'manifest.json').read_text());full=[t for t in parent['tasks'] if t['version']=='M0']
        last=-1
        while True:
            count=sum(validate_task(t,hashes=False) for t in full)
            if count!=last:
                record(expansion,'waiting_for_full_six',full_completed=count,total=6,queued_lu_m2=6,queued_rcd=54,
                    performance_gate_controls_scheduling=False,active_trainers_interrupted=False)
                last=count
            if all(match(p) is None for p in old_workers):
                if count<6:
                    record(expansion,'recovering_incomplete_full')
                    start_workers(expansion,'full_recovery',env)
                break
            time.sleep(3)
        for task in full:assert validate_task(task)
        verify_plan(expansion)
        for stage in ('lu_m2','rcd'):
            record(expansion,'training_'+stage,performance_gate_controls_scheduling=False,
                   stage_tasks=6 if stage=='lu_m2' else 54,recovery_guard_pid=guardian.pid)
            start_workers(expansion,stage,env)
            for task in task_list(expansion,stage):assert validate_task(task)
            try:summarize(expansion)
            except Exception as error:problem(expansion,action='reporting_failed_training_artifacts_preserved',error=repr(error))
        success=True
    except BaseException as error:
        failure=repr(error);problem(expansion,action='controller_failure',error=failure);cleanup(expansion)
    finally:
        release(expansion)
        record(expansion,'complete' if success else 'failed',error=failure,
            original_queue_restore_delegated_to_prior=True,
            completed_additional_tasks=sum(validate_task(t,hashes=False) for t in m['tasks']))
    return int(not success)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expansion',type=Path,required=True)
    p.add_argument('--request',type=Path);p.add_argument('--worker',action='store_true');p.add_argument('--stage',choices=['full_recovery','lu_m2','rcd'])
    p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--guard',action='store_true')
    p.add_argument('--owner-pid',type=int);p.add_argument('--owner-created',type=float)
    sys.exit(run(p.parse_args()))
