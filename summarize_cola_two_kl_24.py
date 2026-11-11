"""Report actual completed experiments; never infer metrics for pending/failed work."""
import csv
import fcntl
import json
from pathlib import Path
import statistics
import argparse

ROOT = Path(__file__).resolve().parent
METRICS = ('top1_accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1')


def _summarize(output=None):
    tasks = json.loads((ROOT / 'config/cola_two_kl_24/manifest.json').read_text())['tasks']
    status_path = ROOT / 'logs/cola_two_kl_24/status.tsv'
    states = {}
    if status_path.exists():
        with status_path.open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                states[(row['variant'], row['run_name'])] = row['status']
    rows = []
    for task in tasks:
        variant, name = task['variant'], task['run_name']
        folder = ROOT / task['model_dir']
        previous = states.get((variant, name), 'pending')
        row = dict(variant=variant, run_name=name, seed=task['seed'],
                   status='failed' if previous == 'failed' else 'pending')
        for key in (*METRICS, 'best_epoch', 'best_iteration', 'final_tau_dyn',
                    'KL_uni_start', 'KL_uni_final', 'KL_rev_start', 'KL_rev_final',
                    'DDDE_distribution_start', 'DDDE_distribution_final',
                    'confusion_matrix', 'evaluation', 'history'):
            row[key] = ''
        if (folder / 'completed.json').is_file():
            from cola_two_kl_queue import completed
            if not completed(task):
                raise RuntimeError(f'Invalid completion marker for {variant}/{name}')
            evaluation = folder / 'evaluation/evaluation.json'
            data = json.loads(evaluation.read_text())
            row.update(status='complete', **{m: data['heldout'][m] for m in METRICS},
                       best_epoch=data['best_epoch'], best_iteration=data['best_iteration'],
                       confusion_matrix=data['heldout']['confusion_matrix_csv'], evaluation=str(evaluation))
            import torch
            latest = torch.load(folder / 'latest_model.pth', map_location='cpu', weights_only=False)
            cola = latest.get('cola_two_kl')
            row['final_tau_dyn'] = float(cola['tau_dyn']) if cola is not None else 4.0
            history = folder / 'cola_history.jsonl'
            if history.exists():
                records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
                valid = [r for r in records if r['event'] == 'batch']
                if valid:
                    for key in ('KL_uni', 'KL_rev'):
                        row[key + '_start'] = valid[0][key]
                        row[key + '_final'] = valid[-1][key]
                    row['DDDE_distribution_start'] = json.dumps(valid[0]['pi_DDDE'])
                    row['DDDE_distribution_final'] = json.dumps(valid[-1]['pi_DDDE'])
                row['history'] = str(history)
        rows.append(row)
    out = Path(output) if output else ROOT / 'results/cola_two_kl_24'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summaries = []
    for variant in dict.fromkeys(t['variant'] for t in tasks):
        group = [r for r in rows if r['variant'] == variant]
        done = [r for r in group if r['status'] == 'complete']
        row = dict(variant=variant, completed=len(done), pending=sum(r['status'] == 'pending' for r in group),
                   failed=sum(r['status'] == 'failed' for r in group))
        for metric in METRICS:
            values = [r[metric] for r in done]
            row[metric + '_mean'] = statistics.mean(values) if values else ''
            row[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else ''
        # These six rows use different data settings. This is descriptive only.
        row['note'] = 'Descriptive across six different data settings; not repeated-seed uncertainty'
        summaries.append(row)
    with (out / 'ablation_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    print(f'Summary: {sum(r["status"] == "complete" for r in rows)}/24 complete; {out}')


def summarize(output=None):
    folder = Path(output) if output else ROOT / 'results/cola_two_kl_24'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    summarize(parser.parse_args().output)
    from cola_two_kl_handoff import after_summary
    after_summary()
