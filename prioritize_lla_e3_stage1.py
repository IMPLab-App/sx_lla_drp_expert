"""User-authorized priority run with reversible queue suspension and recovery.

SIGSTOP retains the old queue's exact in-memory state and CUDA allocations.
No original trainer is terminated. SIGCONT restores it after stage 1 or failure.
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
from lla_experiment_utils import gpu_status, sha256

ROOT=Path(__file__).resolve().parent


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def atomic(path,data):
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def identity(process):
    return dict(pid=process.pid,created=process.create_time(),command=process.cmdline(),
                original_status=process.status())


def match(item):
    try:
        process=psutil.Process(item['pid'])
        if abs(process.create_time()-item['created'])<.01 and process.status()!=psutil.STATUS_ZOMBIE:
            return process
    except psutil.NoSuchProcess:pass
    return None


def status(batch,state,**fields):
    record=dict(state=state,updated_at=now(),pid=os.getpid(),stage=1,**fields)
    atomic(batch/'priority_status.json',record)
    atomic(batch/'launch_status.json',record)
    with (batch/'priority_events.jsonl').open('a') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    words={
        'pausing_original_queue':'按用户最新指令，正在可恢复地暂停原实验队列。',
        'cpu_verification':'原队列已暂停；正在重新核验资源控制修改后的CPU测试。',
        'gpu_preflight_and_stage1':'两张GPU优先执行本轮GPU核验及第一阶段训练；各LU三模型在同一GPU。',
        'complete':'本轮第一阶段18项训练及验证核验完成，原队列已恢复。',
        'failed':'本轮启动/核验失败并已停止本轮队列；原队列已恢复，未跳过门禁。',
        'recovered_original_queue':'恢复保护进程检测到本轮监督进程退出，已恢复原队列。',
    }[state]
    text=f'# 训练状态\n\n{words}\n\n时间：{record["updated_at"]}\n\n本轮监督PID：{os.getpid()}。详情见priority_status.json、priority_run.log及priority_gpu*.log。\n\n仅执行第一阶段六任务V1/V5/M0共18项；第二阶段、54项全面消融及测试集选优均不执行。\n'
    for target in (batch/'TRAINING_STATUS.md',batch/'reports/TRAINING_STATUS.md'):target.write_text(text)


def resume_old(manifest):
    if not manifest.exists():return []
    pause=json.loads(manifest.read_text());restored=[]
    # Resume deepest children first and their supervisors last. This avoids
    # supervisors advancing the queue while a child is still stopped.
    for item in reversed(pause['processes']):
        process=match(item)
        if process is not None and item['original_status']!=psutil.STATUS_STOPPED:
            process.resume();restored.append(process.pid)
    pause.update(resumed_at=now(),resumed_pids=restored,state='resumed')
    atomic(manifest,pause)
    return restored


def guard(batch,manifest,parent_pid,parent_created):
    while match(dict(pid=parent_pid,created=parent_created)) is not None:
        time.sleep(5)
    pause=json.loads(manifest.read_text()) if manifest.exists() else {}
    if pause.get('state')!='resumed':
        pids=resume_old(manifest)
        status(batch,'recovered_original_queue',resumed_pids=pids)


def suspend_tree(batch,manifest,root_item,guard_log):
    root=match(root_item)
    if root is None:raise RuntimeError('Original queue supervisor already exited; inspect before priority pause')
    if '/media/docs_1/mete-expert/logs/e3_specialization_20261111/' not in ' '.join(root.cmdline()):
        raise RuntimeError('Refusing to pause an unexpected process tree')
    planned=[identity(p) for p in [root]+root.children(recursive=True)]
    pause=dict(authorization='user_requested_priority_pause',user_instruction='直接开始训练，原有训练等这个训练完成后再继续进行训练',
        state='pausing',created_at=now(),root=root_item,processes=planned)
    atomic(manifest,pause)
    supervisor=psutil.Process(os.getpid())
    guardian=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--guard',
        '--batch',str(batch),'--pause-manifest',str(manifest),'--parent-pid',str(os.getpid()),
        '--parent-created',str(supervisor.create_time())],stdin=subprocess.DEVNULL,
        stdout=guard_log,stderr=subprocess.STDOUT,start_new_session=True)
    # Stop the supervisor first, then descendants; rescan after parents stop to
    # catch children created during initial enumeration.
    for _ in range(3):
        known={p['pid'] for p in pause['processes']}
        for process in root.children(recursive=True):
            if process.pid not in known:
                pause['processes'].append(identity(process));known.add(process.pid)
        atomic(manifest,pause)
        for item in pause['processes']:
            process=match(item)
            if process is not None and process.status()!=psutil.STATUS_STOPPED:process.suspend()
        time.sleep(.25)
    running=[item['pid'] for item in pause['processes'] if (p:=match(item)) is not None and p.status()!=psutil.STATUS_STOPPED]
    if running:raise RuntimeError('Queue did not stop completely: '+str(running))
    pause.update(state='paused',paused_at=now(),recovery_guard_pid=guardian.pid)
    atomic(manifest,pause)
    return pause


def stop_own_workers(workers):
    # Only terminate this new queue's workers on failure. Original processes
    # are never sent TERM/KILL; they are resumed from SIGSTOP in finally.
    for worker in workers:
        if worker.poll() is not None:continue
        try:root=psutil.Process(worker.pid);children=root.children(recursive=True)
        except psutil.NoSuchProcess:continue
        for process in reversed(children):
            try:process.terminate()
            except psutil.NoSuchProcess:pass
        worker.terminate()
    for worker in workers:
        try:worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                process=psutil.Process(worker.pid)
                for child in process.children(recursive=True):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                process.kill()
            except psutil.NoSuchProcess:pass
            worker.wait(timeout=10)


def verify_all(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    for task in manifest['tasks']:
        output=Path(task['output'])
        result=json.loads((output/'checkpoint_manifest.json').read_text())
        assert result['config_sha256']==task['config_sha256']
        assert result['selection']=='validation_macro_f1'
        for value in result['checkpoints'].values():assert sha256(value['path'])==value['sha256']
        assert (output/'validation_selection.json').exists()
    return len(manifest['tasks'])


def main(args):
    batch=args.batch.resolve();manifest=args.pause_manifest or batch/'priority_paused_processes.json'
    if args.guard:
        guard(batch,manifest,args.parent_pid,args.parent_created);return 0
    if args.resume_only:
        print(resume_old(manifest));return 0
    lock=(batch/'priority.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Hold a shared queue lock during both worker shards. A standalone serial
    # runner requires an exclusive lock and therefore cannot double-launch.
    queue_lock=(batch/'stage1.lock').open('a');fcntl.flock(queue_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise KeyboardInterrupt('priority supervisor received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    request=json.loads(args.request.read_text())
    workers=[];success=False;error=None
    guard_log=(batch/'priority_recovery_guard.log').open('a')
    try:
        status(batch,'pausing_original_queue',request=str(args.request))
        pause=suspend_tree(batch,manifest,request['original_supervisor'],guard_log)
        # Supersede only our previous passive waiting supervisor.
        own_waiter=request.get('previous_waiter')
        if own_waiter:
            waiter=match(own_waiter)
            if waiter is not None and str(ROOT/'wait_and_run_lla_e3_stage1.py') in waiter.cmdline():waiter.terminate()
        env=dict(os.environ,LLA_PAUSED_PROCESS_MANIFEST=str(manifest))
        env.pop('CUDA_VISIBLE_DEVICES',None)
        idle_observations=0
        for _ in range(60):
            states=[gpu_status(gpu) for gpu in (0,1)]
            # gpu_status here must see the same explicit pause authorization.
            os.environ['LLA_PAUSED_PROCESS_MANIFEST']=str(manifest)
            states=[gpu_status(gpu) for gpu in (0,1)]
            good=all(not s['jobs'] and s['utilization']<=10 and s['total_memory_mib']-s['memory_mib']>=8192 for s in states)
            idle_observations=idle_observations+1 if good else 0
            if idle_observations>=3:break
            time.sleep(2)
        else:raise RuntimeError('Paused GPUs did not become safe for priority work')
        status(batch,'cpu_verification',paused_pids=[p['pid'] for p in pause['processes']])
        cpu_env=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for flag,label,marker in (('--units','priority_verified_units','unit_tests/result.json'),
                                  ('--smoke','priority_verified_smoke','smoke_tests/result.json')):
            proof=json.loads((batch/marker).read_text())
            current=(proof['status']=='passed' and proof.get('source_sha256') and
                     all(sha256(ROOT/name)==digest for name,digest in proof['source_sha256'].items()))
            if current:
                print('Reusing fresh passed CPU verification: '+marker,flush=True)
                continue
            log_path=batch/('unit_tests' if flag=='--units' else 'smoke_tests')/(label+'.log')
            with log_path.open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'verify_lla_e3.py'),flag],cwd=ROOT,
                    env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(ROOT/'prepare_lla_e3.py'),'--batch',str(batch),'--snapshot-only'],cwd=ROOT,env=env,check=True)
        status(batch,'gpu_preflight_and_stage1',paused_process_manifest=str(manifest),gpus=[0,1])
        # LU1/3/5 go to GPU0 and LU2/4/6 to GPU1; each comparison stays on one GPU.
        for gpu in (0,1):
            stream=(batch/f'priority_gpu{gpu}.log').open('a')
            workers.append(subprocess.Popen([sys.executable,'-u',str(ROOT/'run_lla_e3_stage1.py'),
                '--batch',str(batch),'--gpu',str(gpu),'--shard-index',str(gpu),'--shards','2'],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        while any(worker.poll() is None for worker in workers):
            failed=[worker.returncode for worker in workers if worker.poll() not in (None,0)]
            if failed:raise RuntimeError('Priority worker failed: '+str(failed))
            time.sleep(5)
        if any(worker.returncode for worker in workers):raise RuntimeError('Priority worker failed after final poll')
        verified=verify_all(batch)
        subprocess.run([sys.executable,str(ROOT/'summarize_lla_e3.py'),'--batch',str(batch)],cwd=ROOT,check=True,env=env)
        assert verified==18
        success=True
    except BaseException as exception:
        error=repr(exception)
        stop_own_workers(workers)
    finally:
        restored=resume_old(manifest)
        status(batch,'complete' if success else 'failed',verified_formal_runs=18 if success else None,
            error=error,resumed_original_pids=restored,later_stages_started=False)
        guard_log.close()
    return 0 if success else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--request',type=Path)
    parser.add_argument('--pause-manifest',type=Path);parser.add_argument('--guard',action='store_true')
    parser.add_argument('--resume-only',action='store_true');parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--parent-created',type=float)
    sys.exit(main(parser.parse_args()))
