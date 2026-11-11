"""Durable two-GPU sequential queues with logs, bounded retries and live summaries.

python run_signal_baselines.py BATCH [--report-only]
Tasks run in manifest order on their assigned physical GPU.
"""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def report(batch):
    import numpy as np
    manifest = json.loads((batch / 'manifest.json').read_text())
    planned_repeats = manifest.get('protocol', {}).get('repeats', 2)
    rows = []
    for task in manifest['tasks']:
        path = Path(task['output']) / 'test_metrics.json'
        done = batch / 'state' / (task['name'] + '.json')
        if not path.exists() or not done.exists() or json.loads(done.read_text())['state'] != 'completed':
            continue
        metrics = json.loads(path.read_text())
        row = {k: task[k] for k in ('method','case','repeat','seed','gpu')}
        row.update({k: metrics[k] for k in ('samples','accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','rejected')})
        rows.append(row)
    dest=batch/'summary';dest.mkdir(exist_ok=True)
    if rows:
        with (dest/'runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ' + manifest.get('title', '一维信号八方法对比'), '', f'更新：{now()}；完成 {len(rows)}/{len(manifest["tasks"])}。指标范围 0–1。', '',
           ('同 seed 共享数据划分；本批次每组合训练一次，直接报告测试指标；拒识计为错误。'
            if planned_repeats == 1 else
            f'同 seed 共享数据划分；每组合计划 {planned_repeats} 次重复，多次完成后报告均值与样本标准差；拒识计为错误。'), '',
           '| 方法 | 配置 | 完成重复数 | Accuracy (mean ± sd) | Macro F1 (mean ± sd) |',
           '|---|---:|---:|---:|---:|']
    aggregate=[]
    for method in manifest['methods']:
        for case in sorted({task['case'] for task in manifest['tasks'] if task['method'] == method}):
            group=[r for r in rows if r['method']==method and r['case']==case]
            if not group:continue
            item={'method':method,'case':case,'repeats':len(group)}
            for key in ('accuracy','macro_f1','macro_recall'):
                vals=[r[key] for r in group]
                item[key+'_mean']=float(np.mean(vals))
                item[key+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            aggregate.append(item)
            def fmt(k):
                if item[k+'_std'] is not None:
                    return f'{item[k+"_mean"]:.4f} ± {item[k+"_std"]:.4f}'
                suffix = '' if planned_repeats == 1 else f' (完成 {len(group)}/{planned_repeats})'
                return f'{item[k+"_mean"]:.4f}{suffix}'
            lines.append(f'| {method} | {case} | {len(group)} | {fmt("accuracy")} | {fmt("macro_f1")} |')
    atomic_json(dest/'aggregate.json',aggregate)
    (dest/'comparison.md').write_text('\n'.join(lines)+'\n')
    return len(rows)


def worker(batch,gpu):
    manifest=json.loads((batch/'manifest.json').read_text())
    folder=batch/f'gpu{gpu}';folder.mkdir(exist_ok=True)
    (folder/'worker.pid').write_text(str(os.getpid())+'\n')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
               MPLBACKEND='Agg',PYTHONUNBUFFERED='1',TZ='Asia/Shanghai')
    tasks=[t for t in manifest['tasks'] if t['gpu']==gpu]
    failure=False
    for task in tasks:
        state_path=batch/'state'/(task['name']+'.json')
        state=json.loads(state_path.read_text())
        if state['state']=='completed':continue
        success=False
        for attempt in (1,2):
            output=Path(task['output'])
            if output.exists():
                archive=batch/'failed_attempts'/f'{task["name"]}_{int(time.time())}'
                archive.parent.mkdir(exist_ok=True)
                output.rename(archive)
            for stage,command in enumerate(task['commands'],1):
                logfile=folder/f'{task["name"]}.attempt{attempt}.stage{stage}.log'
                state.update(state='running',stage=stage,attempt=attempt,updated=now(),log=str(logfile))
                atomic_json(state_path,state)
                print(f'[{now()}] {task["name"]} attempt={attempt} stage={stage}',flush=True)
                with logfile.open('wb') as log:
                    process=subprocess.Popen([sys.executable]+command,cwd=manifest['code_snapshot'],env=env,
                                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    state.update(pid=process.pid);atomic_json(state_path,state)
                    code=process.wait()
                if code:
                    tail='\n'.join(logfile.read_text(errors='replace').splitlines()[-35:])
                    issue=dict(time=now(),name=task['name'],gpu=gpu,attempt=attempt,stage=stage,
                               exit_code=code,log=str(logfile),tail=tail)
                    with (batch/f'issues_gpu{gpu}.jsonl').open('a') as f:f.write(json.dumps(issue,ensure_ascii=False)+'\n')
                    state.update(state='retry_pending' if attempt==1 else 'failed',exit_code=code,updated=now())
                    atomic_json(state_path,state)
                    break
            else:
                metrics=output/'test_metrics.json'
                checkpoint=output/('best_model.pt' if task['method'] in ('uaman','sat_ifixmatch') else 'model_best.pth')
                if metrics.exists() and checkpoint.exists():
                    success=True
                    state.update(state='completed',updated=now(),exit_code=0)
                    atomic_json(state_path,state)
                    print(f'[{now()}] {task["name"]} completed',flush=True)
                    break
                state.update(state='failed',updated=now(),error='Missing checkpoint or standardized test metrics')
                atomic_json(state_path,state)
        if not success:failure=True
    return int(failure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--worker',type=int,choices=[0,1])
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();batch=args.batch.resolve()
    if args.report_only:
        print(f'Completed {report(batch)} tasks');return 0
    if args.worker is not None:return worker(batch,args.worker)
    with (batch/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((batch/'manifest.json').read_text())
        state_dir=batch/'state';state_dir.mkdir(exist_ok=True)
        for task in manifest['tasks']:
            p=state_dir/(task['name']+'.json')
            if not p.exists():atomic_json(p,dict(name=task['name'],gpu=task['gpu'],state='pending',updated=now()))
        (batch/'queue.pid').write_text(str(os.getpid())+'\n')
        (batch/'status.txt').write_text('running\n')
        workers=[]
        for gpu in (0,1):
            folder=batch/f'gpu{gpu}';folder.mkdir(exist_ok=True)
            with (folder/'queue.log').open('ab') as log:
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(batch),'--worker',str(gpu)],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            workers.append(p)
            print(f'[{now()}] GPU {gpu} worker PID {p.pid}',flush=True)
        while any(p.poll() is None for p in workers):
            report(batch)
            time.sleep(15)
        completed=report(batch)
        codes=[p.returncode for p in workers]
        status='completed' if completed==len(manifest['tasks']) and not any(codes) else 'completed_with_failures'
        (batch/'status.txt').write_text(status+'\n')
        atomic_json(batch/'exit_codes.json',dict(codes=codes,completed=completed,total=len(manifest['tasks']),time=now()))
        print(f'[{now()}] {status}: {completed}/{len(manifest["tasks"])}',flush=True)
        return int(status!='completed')


if __name__=='__main__':
    sys.exit(main())
