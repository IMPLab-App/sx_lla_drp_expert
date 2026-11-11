"""Continuously compare the 11 reused No-DDDE runs with Full and No-DEA."""
import argparse
import csv
import json
from pathlib import Path


def summarize(batch):
    manifest = json.loads((batch / 'manifest.json').read_text())
    baseline = Path(manifest['baseline_batch'])
    original = json.loads((baseline / 'manifest.json').read_text())
    tasks = [(baseline, 'no_ddde_kl', t) for t in original['tasks']]
    tasks += [(batch, t['variant'], t) for t in manifest['tasks']]
    rows = []
    for folder, variant, task in tasks:
        path = folder / 'state' / (task['name'] + '.json')
        state = json.loads(path.read_text())['state'] if path.exists() else 'pending'
        row = dict(case=task['case'], variant=variant, state=state,
                   accuracy=None, macro_f1=None, samples=None, output=task['output'])
        if state == 'completed':
            metrics = json.loads((Path(task['output']) / 'test_metrics.json').read_text())
            assert metrics['samples'] == 900
            row.update({k: metrics[k] for k in ('accuracy', 'macro_f1', 'samples')})
        rows.append(row)
    rows.sort(key=lambda r: (r['case'], ['full', 'no_ddde_kl', 'no_dea'].index(r['variant'])))
    folder = batch / 'comparison'
    folder.mkdir(exist_ok=True)
    with (folder / 'three_variants.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# BACN 11组：完整模型与两组消融', '',
             'No-DDDE复用原11组固定τ=4结果；Full和No-DEA均启用DDDE/双KL自适应τ。',
             'No-DEA移除整条动态聚合及专用融合损失，双专家原始logits固定平均。',
             '同seed、相同样本索引和训练参数；指标范围0–1，短训练批次仅供流程验证。', '',
             '| Case | Variant | Status | Accuracy | Macro-F1 |', '|---|---|---|---:|---:|']
    for r in rows:
        acc = f'{r["accuracy"]:.4f}' if r['accuracy'] is not None else '—'
        f1 = f'{r["macro_f1"]:.4f}' if r['macro_f1'] is not None else '—'
        lines.append(f'| {r["case"]} | {r["variant"]} | {r["state"]} | {acc} | {f1} |')
    (folder / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    summarize(args.batch.resolve())
