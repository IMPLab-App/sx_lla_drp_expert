"""Independent held-out evaluation with strict five-version state restoration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from scripts.signal_metrics import write_signal_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--c',type=Path,required=True)
    args=parser.parse_args();cfg=yaml.safe_load(args.c.read_text())
    assert cfg['imb_algorithm']=='metaexpert_v1v5'
    out=Path(cfg['save_dir'])/cfg['save_name'];evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('eval_confusion_matrix_new.py')),
        '--c',str(args.c),'--load_path',str(out/'model_best.pth'),'--output_dir',str(evaluation),'--gpu','0'],check=True)
    with np.load(evaluation/'confusion_matrix/predictions_heldout.npz',allow_pickle=False) as p:
        metrics=write_signal_metrics(out,p['y_true'],p['y_pred'],cfg['num_classes'],p['class_names'].tolist())
        np.savez_compressed(out/'test_probabilities.npz',y_true=p['y_true'],probabilities=p['probabilities'],class_names=p['class_names'])
    original=json.loads((evaluation/'evaluation.json').read_text())
    assert abs(metrics['accuracy']-original['heldout']['top1_accuracy'])<1e-12
    print('Five-version held-out metrics:',json.dumps(metrics),flush=True)


if __name__=='__main__':main()
