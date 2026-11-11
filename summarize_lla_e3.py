"""Validation-only stage-1 comparison; missing runs remain explicitly missing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

DEFAULT=Path(__file__).resolve().parent/'logs/lla_e3_replacement_20261111'


def summarize(batch):
    m=json.loads((batch/'manifest.json').read_text());rows=[];details={};missing=[]
    for task in m['tasks']:
        out=Path(task['output']);selection=out/'validation_selection.json'
        if not selection.exists():
            missing.append((task['scenario'],task['version']))
            rows.append(dict(scenario=task['scenario'],task=task['case_name'],model=task['version'],status='pending'))
            continue
        selected=json.loads(selection.read_text())
        metric_file=out/'metrics'/f"eval_iter{selected['eval/best_it']:06d}.json"
        if not metric_file.exists():raise FileNotFoundError(metric_file)
        record=json.loads(metric_file.read_text());main=record['branches']['fusion_main']
        assert abs(main['macro_f1']-selected['eval/best_F1'])<1e-12
        details[(task['scenario'],task['version'])]=main
        rows.append(dict(scenario=task['scenario'],task=task['case_name'],model=task['version'],status='complete',
            val_accuracy=main['accuracy'],val_macro_f1=main['macro_f1'],C5_F1=main['C5']['f1'],
            C8_F1=main['C8']['f1'],tail_macro_f1=main['tail_macro_f1'],best_epoch=selected['eval/best_epoch'],
            best_iteration=selected['eval/best_it'],checkpoint=str(out/'model_best.pth'),
            raw_fusion_f1=record['branches']['fusion_raw']['macro_f1'],seed=task['seed'],std=None))
    target=batch/'comparison';target.mkdir(exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (target/'replacement_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    report=['# Stage 1 validation comparison','',
        'Single seed=2. All values are validation-only; no statistical significance claim. Missing results are not imputed.','',
        '| Task | V1 fixed tau4 | V5 DDDE+KL | M0 LLA-E3 Hard | M0−V5 | M0−V1 |',
        '|---|---:|---:|---:|---:|---:|']
    for i,case in enumerate(m['task_names'],1):
        values=[details.get((f'LU{i}',v),{}).get('macro_f1') for v in ('V1','V5','M0')]
        show=lambda x:'pending' if x is None else f'{x:.6f}'
        a,b,c=values
        report.append('| '+case+' | '+' | '.join(map(show,values))+f' | {show(None if b is None or c is None else c-b)} | {show(None if a is None or c is None else c-a)} |')
    gate=dict(status='pending',eligible_for_stage2=False,missing=missing,test_metrics_read=False,
              stage2_started=False,stage3_started=False)
    if not missing:
        f1={v:np.asarray([details[(f'LU{i}',v)]['macro_f1'] for i in range(1,7)]) for v in ('V1','V5','M0')}
        means={v:float(x.mean()) for v,x in f1.items()}
        wins=int((f1['M0']>f1['V1']).sum());degradations=[]
        for i in range(1,7):
            for v in ('V1','V5'):
                a,b=details[(f'LU{i}','M0')],details[(f'LU{i}',v)]
                for key in ('C5','C8','tail_macro_f1'):
                    av,bv=(a[key]['f1'],b[key]['f1']) if key!='tail_macro_f1' else (a[key],b[key])
                    if av-bv < -m['phase2_gate']['severe_degradation_absolute_f1']:
                        degradations.append(dict(scenario=f'LU{i}',reference=v,metric=key,delta=av-bv))
        eligible=means['M0']>means['V1'] and means['M0']>means['V5'] and wins>=4 and not degradations
        gate.update(status='passed' if eligible else 'failed',eligible_for_stage2=eligible,
            mean_validation_f1=means,m0_wins_over_v1=wins,
            m0_wins_over_v5=int((f1['M0']>f1['V5']).sum()),severe_degradations=degradations)
        report.extend(['',f'Gate: {gate["status"]}. No automatic stage-2/3 launch from this script.',
                       '',f'Mean validation Macro-F1: {means}; M0 > V1 in {wins}/6 tasks.'])
    else:report.extend(['','Gate pending: incomplete first-stage training. DEA and R/CD experiments remain unstarted.'])
    (target/'stage1_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n')
    (batch/'reports/COMPARISON_RESULTS.md').write_text('\n'.join(report)+'\n')
    # No LU DEA / R / CD performance tables are fabricated before those stages.
    return gate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,default=DEFAULT)
    a=p.parse_args();print(json.dumps(summarize(a.batch.resolve()),ensure_ascii=False,indent=2))
