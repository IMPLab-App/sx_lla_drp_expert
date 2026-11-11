"""Validation-only DRP comparisons and preregistered gate. Never schedules R/CD."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def collect(task):
    out=Path(task['output']);selection=out/'validation_selection.json';manifest=out/'checkpoint_manifest.json'
    row=dict(scenario=task['scenario'],case_name=task['case_name'],version=task['version'],status='pending')
    if not selection.exists() or not manifest.exists():return row
    chosen=json.loads(selection.read_text());proof=json.loads(manifest.read_text())
    if proof['config_sha256']!=task['config_sha256']:raise ValueError('Completed configuration hash differs: '+str(out))
    p=out/'metrics'/f"eval_iter{chosen['eval/best_it']:06d}.json"
    metrics=json.loads(p.read_text());main=metrics['branches']['fusion_main']
    assert abs(main['macro_f1']-chosen['eval/best_F1'])<1e-12
    row.update(status='complete',val_macro_f1=main['macro_f1'],val_accuracy=main['accuracy'],
        best_epoch=chosen['eval/best_epoch'],best_iteration=chosen['eval/best_it']+1,
        C5_recall=main['C5']['recall'],C5_f1=main['C5']['f1'],C8_recall=main['C8']['recall'],C8_f1=main['C8']['f1'],
        tail_macro_f1=main['tail_macro_f1'],branches=metrics['branches'],
        complementarity_lla=metrics['complementarity_lla'],complementarity_raw=metrics['complementarity_raw'],
        checkpoint=proof['checkpoints']['model_best.pth'],std=None,seed=2,
        metric_file=str(p),mff_gradient_monitor=proof.get('mff_gradient_monitor'))
    return row


def summarize(batch):
    manifest=json.loads((batch/'manifest.json').read_text())
    rows=[collect(t) for t in manifest['tasks']+manifest['references']]
    lookup={(r['scenario'],r['version']):r for r in rows if r['status']=='complete'}
    full=all((f'LU{i}',v) in lookup for i in range(1,7) for v in ('M0','L0','M1','V5'))
    gate=dict(status='pending',eligible_for_m2=False,test_metrics_read=False,rcd_started=False,
        conditions=manifest['drp_gate'])
    if full:
        scores={v:np.asarray([lookup[(f'LU{i}',v)]['val_macro_f1'] for i in range(1,7)]) for v in ('M0','L0','M1','V5')}
        means={v:float(a.mean()) for v,a in scores.items()}
        wins=int((scores['M0']>scores['L0']).sum())
        passed=means['M0']>means['L0'] and wins>=4 and means['M0']-means['M1']>=-manifest['drp_gate']['fixed_tau4_mean_max_drop']
        gate.update(status='passed' if passed else 'failed',eligible_for_m2=passed,
            mean_validation_macro_f1=means,full_wins_over_lla_only=wins,
            mean_drp_increment=means['M0']-means['L0'],mean_full_minus_fixed=means['M0']-means['M1'])
    (batch/'comparison/drp_validation_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n')
    (batch/'comparison/raw_validation_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for row in rows:
        item={k:v for k,v in row.items() if k not in ('branches','complementarity_lla','complementarity_raw','checkpoint','mff_gradient_monitor')}
        item['checkpoint_path']=row.get('checkpoint',{}).get('path')
        item['checkpoint_sha256']=row.get('checkpoint',{}).get('sha256')
        flat.append(item)
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with (batch/'comparison/validation_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    report=['# LU validation results','',
        'Single training seed=2; Macro-F1 selected checkpoints. No test-based selection. Missing results are pending; standard deviation is NA.','',
        '## Table LU-1: DRP increment','',
        '| Task | L0 LLA-only | Full LLA+DRP | Full−L0 (pp) | fixed tau4 | V5 DDDE+KL |',
        '|---|---:|---:|---:|---:|---:|']
    def score(i,v):return lookup.get((f'LU{i}',v),{}).get('val_macro_f1')
    def fmt(value):return 'pending' if value is None else f'{value*100:.4f}'
    for i,case in enumerate(manifest['task_names'],1):
        a,b=score(i,'M0'),score(i,'L0')
        delta=a-b if a is not None and b is not None else None
        report.append(f'| {case} | {fmt(b)} | {fmt(a)} | {fmt(delta)} | {fmt(score(i,"M1"))} | {fmt(score(i,"V5"))} |')
    report += ['', '## Table LU-2: primary ablation','',
        '| Task | M0 Full | M1 w/o LLA+DRP | M2 fixed expert weights | Full−M1 (pp) | Full−M2 (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for i,case in enumerate(manifest['task_names'],1):
        a,b,c=score(i,'M0'),score(i,'M1'),score(i,'M2')
        report.append(f'| {case} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {fmt(a-b if a is not None and b is not None else None)} | {fmt(a-c if a is not None and c is not None else None)} |')
    report += ['', 'M2 retains lin1–lin8 MFF and the original predict auxiliary losses. It removes dynamic weighting from final fusion decisions.',
        '',f'DRP validation gate: {gate["status"]}. R/CD 54-model expansion is never started by this report script.']
    for path in (batch/'reports/LU_RESULTS.md',batch/'LU_RESULTS.md'):path.write_text('\n'.join(report)+'\n')
    for group in ('R','CD'):
        path=batch/'reports'/f'{group}_RESULTS.md'
        if not path.exists():path.write_text(f'# {group} results\n\nNot started. Existing scenario definitions are archived; no results are fabricated.\n')
    return gate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);a=p.parse_args()
    print(json.dumps(summarize(a.batch.resolve()),ensure_ascii=False,indent=2))
