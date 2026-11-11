"""Durable 96/96 GPU queues plus main-ablation and sensitivity live tables."""
import argparse
import csv
import json
from pathlib import Path
import sys


def summarize(batch):
    m=json.loads((batch/'manifest.json').read_text());rows=[]
    for t in m['tasks']:
        p=batch/'state'/(t['name']+'.json');state=json.loads(p.read_text())['state'] if p.exists() else 'pending'
        row=dict(scenario=t['scenario'],version=t['version'],gpu=t['gpu'],state=state,accuracy=None,macro_f1=None)
        if state=='completed':
            v=json.loads((Path(t['output'])/'test_metrics.json').read_text());row.update(accuracy=v['accuracy'],macro_f1=v['macro_f1'])
        rows.append(row)
    folder=batch/'comparison';folder.mkdir(exist_ok=True)
    with (folder/'all_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    data={(r['scenario'],r['version']):r for r in rows}
    for file,title,columns in [('main_results.md','V1–V5主消融',[(v,v) for v in ('V1','V2','V3','V4','V5')]),
                              ('sensitivity.md','固定τ敏感性（τ2复用V4，τ4复用V1）',[(f'τ={k}',v) for k,v in m['sensitivity_aliases'].items()])]:
        lines=['# '+title,'','指标为Accuracy / Macro-F1（%）；未完成项显示状态。','',
               '| 场景 | '+' | '.join(k for k,v in columns)+' |','|---|'+'---:|'*len(columns)]
        for s in m['protocol']['scenarios']:
            cells=[]
            for label,v in columns:
                r=data[s,v];cells.append(f'{r["accuracy"]*100:.2f} / {r["macro_f1"]*100:.2f}' if r['accuracy'] is not None else r['state'])
            lines.append('| '+s+' | '+' | '.join(cells)+' |')
        (folder/file).write_text('\n'.join(lines)+'\n')


def main(batch):
    from verify_v1v5 import verify_inputs
    m=verify_inputs(batch);v=json.loads((batch/'verification.json').read_text());assert v['status']=='passed'
    if not m['smoke']:assert v['gpu_audit_passed'] and v['all_192_smoke_train_reload_tests_passed']
    import run_signal_baselines as queue
    original=queue.report
    def report_with_versions(path):
        n=original(path);summarize(path);return n
    queue.report=report_with_versions;sys.argv=[str(batch/'code/run_signal_baselines.py'),str(batch)]
    return queue.main()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);a=p.parse_args();sys.exit(main(a.batch.resolve()))
