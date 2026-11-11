"""Print progress and recent training output for the background comparison."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',nargs='?',type=Path,default=Path('logs/signal_baselines_96_20261027'))
    args=parser.parse_args();batch=args.batch.resolve()
    states=[json.loads(p.read_text()) for p in (batch/'state').glob('*.json')]
    print('Batch:',batch)
    print('Status:',(batch/'status.txt').read_text().strip())
    print('Tasks:',dict(Counter(s['state'] for s in states)))
    for gpu in (0,1):
        jobs=[s for s in states if s['gpu']==gpu and s['state']=='running']
        for job in jobs:
            try:os.kill(job['pid'],0);alive=True
            except (ProcessLookupError,PermissionError):alive=False
            print(f'GPU {gpu}: {job["name"]}; PID {job["pid"]}; alive={alive}')
            p=Path(job['log'])
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-8000));tail=f.read().decode(errors='replace').replace('\r','\n').splitlines()
            useful=[line for line in tail if ' iteration' in line or 'Epoch ' in line or 'epoch ' in line or 'Traceback' in line]
            for line in useful[-2:]:print(line)
    for p in batch.glob('issues_gpu*.jsonl'):
        print('Recorded issues:',p,len(p.read_text().splitlines()))
    print('Live summary:',batch/'summary/comparison.md')


if __name__=='__main__':main()
