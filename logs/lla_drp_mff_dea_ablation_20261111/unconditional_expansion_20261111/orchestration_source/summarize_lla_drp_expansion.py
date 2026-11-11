"""Summarize only measured validation results for the unconditional queue."""
import argparse
import csv
import json
from pathlib import Path
from summarize_lla_drp import collect


def summarize(expansion):
    m=json.loads((expansion/'manifest.json').read_text());batch=Path(m['batch'])
    old=json.loads((batch/'manifest.json').read_text())
    tasks=old['tasks']+old['references']+[t for t in m['tasks'] if t['stage']=='rcd']
    rows=[collect(t) for t in tasks]
    lookup={(r['scenario'],r['version']):r for r in rows}
    (expansion/'raw_validation_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for row in rows:
        item={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        item.update(checkpoint_path=row.get('checkpoint',{}).get('path'),
                    checkpoint_sha256=row.get('checkpoint',{}).get('sha256'))
        flat.append(item)
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with (expansion/'validation_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    def score(s,v):return lookup.get((s,v),{}).get('val_macro_f1')
    def fmt(x):return 'pending' if x is None else f'{100*x:.4f}'
    for group,scenes in [('LU',[f'LU{i}' for i in range(1,7)]),
                         ('R',[f'R{i}' for i in range(1,13)]),
                         ('CD',[f'CD{i}' for i in range(1,7)])]:
        text=[f'# {group}: validation Macro-F1 (%)','',
            'Single seed=2; standard deviation NA. Checkpoints selected by validation Macro-F1. '
            'Training continues regardless of performance under the user authorization amendment. '
            'No test metrics used for selection. Pending cells are not imputed.','',
            '| Scenario | Full LLA+DRP | fixed tau4 | fixed expert weights | Full−M1 (pp) | Full−M2 (pp) |',
            '|---|---:|---:|---:|---:|---:|']
        for s in scenes:
            a,b,c=(score(s,v) for v in ('M0','M1','M2'))
            text.append(f'| {s} | {fmt(a)} | {fmt(b)} | {fmt(c)} | '
                        f'{fmt(a-b if a is not None and b is not None else None)} | '
                        f'{fmt(a-c if a is not None and c is not None else None)} |')
        text+=['','M2 retains lin1–lin8 MFF and fusion auxiliary training; only expert weights are fixed to 0.5/0.5.']
        if all(score(s,v) is not None for s in scenes for v in ('M0','M1','M2')):
            means={v:sum(score(s,v) for s in scenes)/len(scenes) for v in ('M0','M1','M2')}
            text+=['',f'Mean F1: {means}. Full wins over M1: '
                f'{sum(score(s,"M0")>score(s,"M1") for s in scenes)}/{len(scenes)}; '
                f'over M2: {sum(score(s,"M0")>score(s,"M2") for s in scenes)}/{len(scenes)}.']
        (expansion/'reports'/f'{group}_RESULTS.md').write_text('\n'.join(text)+'\n')
    return dict(complete=sum(r['status']=='complete' for r in rows),total=len(rows),test_used=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('expansion',type=Path)
    print(json.dumps(summarize(p.parse_args().expansion.resolve())))
