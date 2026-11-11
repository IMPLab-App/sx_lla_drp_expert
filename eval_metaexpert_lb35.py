"""Run the current project's evaluator and publish auditable held-out metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from scripts.signal_metrics import write_signal_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.c.read_text())
    if config['imb_algorithm'] not in ('metaexpert', 'metaexpert_bacn'):
        raise ValueError('This queue only evaluates the current MetaExpert method')
    output = Path(config['save_dir']) / config['save_name']
    evaluation = output / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('eval_confusion_matrix_new.py')),
                    '--c', str(args.c), '--load_path', str(output / 'model_best.pth'),
                    '--output_dir', str(evaluation), '--gpu', '0'], check=True)
    with np.load(evaluation / 'confusion_matrix/predictions_heldout.npz', allow_pickle=False) as data:
        metrics = write_signal_metrics(output, data['y_true'], data['y_pred'],
                                       config['num_classes'], data['class_names'].tolist())
        np.savez_compressed(output / 'test_probabilities.npz',
                            y_true=data['y_true'], probabilities=data['probabilities'],
                            class_names=data['class_names'])
    original = json.loads((evaluation / 'evaluation.json').read_text())
    assert abs(metrics['accuracy'] - original['heldout']['top1_accuracy']) < 1e-12
    assert abs(metrics['macro_f1'] - original['heldout']['macro_f1']) < 1e-12
    print('Current-method held-out metrics verified:', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
