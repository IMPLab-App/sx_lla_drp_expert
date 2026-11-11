"""Evaluate each baseline's validation-selected EMA checkpoint with its own head."""
from pathlib import Path
import random
import numpy as np
import torch
from train import get_config
from semilearn.core.utils import get_net_builder
from semilearn.imb_algorithms import get_imb_algorithm
from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal_class_names
from scripts.signal_metrics import write_signal_metrics


def main():
    args = get_config()
    args.distributed = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    algorithm = get_imb_algorithm(args, get_net_builder(args.net, False), None, None)
    output = Path(args.save_dir) / args.save_name
    checkpoint = torch.load(output / 'model_best.pth', map_location='cpu', weights_only=False)
    algorithm.model.load_state_dict(checkpoint['ema_model'], strict=True)
    algorithm.model.cuda(args.gpu).eval()
    key = {'acr': 'logits_aux', 'bacon': 'aux_logits', 'cpe': 'aux_logits1',
           'cossl': 'logits_tfe' if checkpoint['epoch'] >= args.cossl_warm_epoch else 'logits'}.get(args.imb_algorithm, 'logits')
    true, pred = [], []
    with torch.no_grad():
        for batch in algorithm.loader_dict['test']:
            logits = algorithm.model(batch['x_lb'].cuda(args.gpu))[key]
            true.extend(batch['y_lb'].tolist())
            pred.extend(logits.argmax(-1).cpu().tolist())
    names = get_raw_signal_class_names(args.data_dir)
    metrics = write_signal_metrics(output, true, pred, args.num_classes, names)
    metrics.update(checkpoint='model_best.pth', output_head=key, best_it=int(checkpoint['best_it']))
    import json
    (output / 'test_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = np.asarray(metrics['confusion_counts'])[:, :args.num_classes]
    normalized = counts / np.maximum(1, counts.sum(1, keepdims=True))
    folder = output / 'confusion_matrix'
    folder.mkdir(exist_ok=True)
    np.savetxt(folder / 'model_best_test.csv', normalized, delimiter=',')
    fig, ax = plt.subplots(figsize=(7, 6))
    plot = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(plot, ax=ax)
    ax.set(xlabel='Predicted class', ylabel='True class', title=args.save_name,
           xticks=range(args.num_classes), yticks=range(args.num_classes))
    fig.tight_layout()
    fig.savefig(folder / 'model_best_test.png', dpi=160)
    plt.close(fig)
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
