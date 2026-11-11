"""Wait for six successful Full runs, then execute the fresh 54-task sweep."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_bacn_18 import VARIANTS,SCENARIOS,digest
from run_signal_baselines import atomic_json,now


def dependency_ready(batch):
    m=json.loads((batch/'manifest.json').read_text());dep=Path(m['dependency_batch'])
    if digest(dep/'manifest.json')!=m['dependency_manifest_sha256']:
        raise RuntimeError('Predecessor manifest changed')
    original=json.loads((dep/'manifest.json').read_text())
    if len(original['tasks'])!=6:raise RuntimeError('Expected six predecessor tasks')
    status=(dep/'status.txt').read_text().strip()
    if status in ('failed','completed_with_failures','supervisor_failed'):
        raise RuntimeError(f'Predecessor failed: {status}')
    if status!='completed':
        pid=int((dep/'queue.pid').read_text())
        try:command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError as error:raise RuntimeError('Predecessor queue exited without completion') from error
        if original['code_snapshot'] not in command or not any(s in command for s in ('run_metaexpert_full_lb35.py','run_signal_baselines.py')):
            raise RuntimeError('Predecessor PID does not belong to its queue')
        return False
    exits=json.loads((dep/'exit_codes.json').read_text())
    if exits['codes']!=[0,0] or exits['completed']!=6:raise RuntimeError('Predecessor exit codes are not successful')
    for t in original['tasks']:
        state=json.loads((dep/'state'/(t['name']+'.json')).read_text());out=Path(t['output'])
        if state['state']!='completed' or not (out/'model_best.pth').is_file():
            raise RuntimeError(f'Incomplete predecessor task: {t["name"]}')
        metrics=json.loads((out/'test_metrics.json').read_text())
        audit=json.loads((out/'artifact_audit.json').read_text())
        if metrics['samples']!=900 or audit['status']!='passed' or audit['samples']!=900:
            raise RuntimeError(f'Predecessor test/artifact audit failed: {t["name"]}')
    return True


def summarize(batch):
    m=json.loads((batch/'manifest.json').read_text());rows=[]
    for t in m['tasks']:
        file=batch/'state'/(t['name']+'.json');state=json.loads(file.read_text())['state'] if file.exists() else 'pending'
        row=dict(case=t['case'],scenario=t['scenario'],variant=t['variant'],gpu=t['gpu'],state=state,accuracy=None,macro_f1=None)
        if state=='completed':
            metrics=json.loads((Path(t['output'])/'test_metrics.json').read_text())
            assert metrics['samples']==900;row.update(accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])
        rows.append(row)
    folder=batch/'comparison';folder.mkdir(exist_ok=True)
    with (folder/'three_variants.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    completed=sum(r['state']=='completed' for r in rows)
    lines=['# r1–r12、CD1–CD6：完整模型与两组消融','',f'更新：{now()}；完成{completed}/54项。全部重新训练，指标为Accuracy / Macro-F1（%）。','',
           '| 场景 | 完整模型 | 无DDDE/自适应τ | 无DEA |','|---|---:|---:|---:|']
    for scenario in SCENARIOS:
        cells=[]
        for v in VARIANTS:
            r=next(r for r in rows if r['scenario']==scenario and r['variant']==v)
            cells.append(f'{r["accuracy"]*100:.2f} / {r["macro_f1"]*100:.2f}' if r['accuracy'] is not None else r['state'])
        lines.append('| '+scenario+' | '+' | '.join(cells)+' |')
    (folder/'comparison.md').write_text('\n'.join(lines)+'\n');return completed


def main(batch):
    with (batch/'successor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from verify_bacn_18 import verify_inputs
        m=verify_inputs(batch);proof=json.loads((batch/'verification.json').read_text())
        if proof['status']!='passed' or not proof['all_54_matching_historical_smoke_checks_passed']:
            raise RuntimeError('Passing 54-task verification required')
        if (batch/'status.txt').exists() and (batch/'status.txt').read_text().strip()=='completed':
            summarize(batch);return 0
        state_dir=batch/'state';state_dir.mkdir(exist_ok=True)
        for t in m['tasks']:
            path=state_dir/(t['name']+'.json')
            if not path.exists():atomic_json(path,dict(name=t['name'],gpu=t['gpu'],state='pending',updated=now()))
        (batch/'supervisor.pid').write_text(str(os.getpid())+'\n')
        (batch/'status.txt').write_text('waiting_for_dependency\n')
        while not dependency_ready(batch):
            atomic_json(batch/'successor.json',dict(pid=os.getpid(),state='waiting_for_dependency',dependency=m['dependency_batch'],updated=now()))
            summarize(batch);time.sleep(15)
        verify_inputs(batch)
        print(f'[{now()}] Six Full runs completed; starting 54 fresh tasks, 27 per GPU',flush=True)
        process=subprocess.Popen([sys.executable,str(batch/'code/run_signal_baselines.py'),str(batch)],
                                 cwd=m['code_snapshot'],stdin=subprocess.DEVNULL)
        while process.poll() is None:
            atomic_json(batch/'successor.json',dict(pid=os.getpid(),queue_pid=process.pid,state='running',updated=now()))
            summarize(batch);time.sleep(15)
        summarize(batch)
        atomic_json(batch/'successor.json',dict(pid=os.getpid(),queue_pid=process.pid,
                    state='completed' if process.returncode==0 else 'failed',exit_code=process.returncode,updated=now()))
        return process.returncode


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    args=parser.parse_args();batch=args.batch.resolve()
    try:sys.exit(main(batch))
    except BlockingIOError:
        print('Supervisor already running; existing state preserved.',file=sys.stderr);sys.exit(2)
    except Exception as error:
        issue=dict(time=now(),state='supervisor_failed',error=str(error),pid=os.getpid())
        with (batch/'supervisor_issues.jsonl').open('a') as stream:stream.write(json.dumps(issue,ensure_ascii=False)+'\n')
        (batch/'status.txt').write_text('supervisor_failed\n');atomic_json(batch/'successor.json',issue)
        raise
