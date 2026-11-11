"""Wait for existing research queues to exit, then run verified stage 1 only.

This supervisor never sends signals to another process and never initializes
CUDA. The existing runner retains all CPU/GPU verification and protocol gates.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from lla_experiment_utils import gpu_status, sha256


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def same_process(item):
    try:
        process=psutil.Process(item['pid'])
        return abs(process.create_time()-item['created']) < .01 and process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def write_status(batch, state, **details):
    record=dict(state=state,updated_at=now(),pid=os.getpid(),stage=1,**details)
    path=batch/'launch_status.json'
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)
    with (batch/'launch_events.jsonl').open('a') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    message={
        'waiting_for_existing_queue':'等待既有实验队列完整退出，尚未使用GPU。',
        'waiting_for_idle_gpu':'既有队列已退出；等待GPU持续空闲，尚未开始训练。',
        'running_stage1':'正在执行GPU核验及第一阶段队列；正式进度见日志，尚未宣称完成。',
        'complete':'18项第一阶段训练及验证结果核验完成；后续阶段未启动。',
        'failed':'第一阶段启动/核验失败；未自动跳过检查或启动后续阶段。',
    }[state]
    text=f'# 第一阶段运行状态\n\n{message}\n\n更新时间：{record["updated_at"]}\n\n监督进程PID：{os.getpid()}；结构化状态见launch_status.json。\n\n六任务V1/V5/M0共18项。测试集未用于选择；第二阶段及54项训练不在本队列。\n'
    for target in (batch/'reports/TRAINING_STATUS.md',batch/'TRAINING_STATUS.md'):
        target.write_text(text)


def verify_complete(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    assert len(manifest['tasks'])==18
    for task in manifest['tasks']:
        output=Path(task['output'])
        selection=json.loads((output/'validation_selection.json').read_text())
        assert selection['criterion']=='validation_macro_f1'
        record=json.loads((output/'checkpoint_manifest.json').read_text())
        assert record['config_sha256']==task['config_sha256']
        for info in record['checkpoints'].values():
            assert sha256(info['path'])==info['sha256']
    gate=json.loads((batch/'comparison/stage1_gate.json').read_text())
    assert gate['status'] in ('passed','failed')
    return gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args();batch=args.batch.resolve()
    lock=(batch/'launch_request.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=json.loads(args.request.read_text())
    runner=Path(__file__).resolve().parent/'run_lla_e3_stage1.py'
    previous=None;idle_counts={gpu:0 for gpu in request['gpu_candidates']}
    while True:
        blockers=[item for item in request['wait_for_processes'] if same_process(item)]
        if blockers:
            state='waiting_for_existing_queue'
            if state!=previous:
                write_status(batch,state,blocking_processes=blockers,request=str(args.request))
                previous=state
            time.sleep(request['poll_seconds']);continue
        states=[gpu_status(gpu) for gpu in request['gpu_candidates']]
        for state in states:
            gpu=state['index']
            idle_counts[gpu]=(idle_counts[gpu]+1 if not state['jobs'] and state['utilization']<=10 else 0)
        candidates=[state['index'] for state in states if idle_counts[state['index']]>=request['idle_observations']]
        if not candidates:
            state='waiting_for_idle_gpu'
            if state!=previous:
                write_status(batch,state,gpus=states,request=str(args.request));previous=state
            time.sleep(request['poll_seconds']);continue
        gpu=candidates[0]
        write_status(batch,'running_stage1',gpu=gpu,log=str(batch/'stage1_supervised.log'))
        with (batch/'stage1_supervised.log').open('a') as stream:
            result=subprocess.run([sys.executable,str(runner),'--batch',str(batch),'--gpu',str(gpu)],
                cwd=runner.parent,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode==75:
            idle_counts={gpu:0 for gpu in request['gpu_candidates']}
            previous=None;time.sleep(request['poll_seconds']);continue
        if result.returncode:
            write_status(batch,'failed',gpu=gpu,exit_code=result.returncode,
                         log=str(batch/'stage1_supervised.log'))
            return result.returncode
        gate=verify_complete(batch)
        write_status(batch,'complete',gpu=gpu,verified_formal_runs=18,
                     validation_gate=gate['status'],later_stages_started=False)
        return 0


if __name__=='__main__':
    sys.exit(main())
