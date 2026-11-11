from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import (
    ImageDictSignalDataset,
    class_counts_from_dataset,
    infinite_loader,
    load_npz,
    make_loader,
)
from model import ResNet1D
from augment import weak_augment, strong_augment
from sat_ifixmatch import SelfAdaptiveThreshold, sat_ifixmatch_step, predict_with_sat_rejection
from metrics import summarize


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, sat, args, desc='eval'):
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        pred, _, _ = predict_with_sat_rejection(
            model,
            x,
            lambda z: weak_augment(z, args.scale_min, args.scale_max),
            lambda z: strong_augment(z, args.mask_prob, args.noise_std),
            sat,
            args.tau_ood,
            args.reject_unselected,
        )
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


@torch.no_grad()
def evaluate(model, loader, device, sat, args, desc='eval'):
    y_true, y_pred = predict(model, loader, device, sat, args, desc=desc)
    return summarize(y_true, y_pred, args.num_classes)


def save_confusion_outputs(y_true, y_pred, num_classes: int, out_dir: Path, base_name: str):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    true_counts = np.zeros((num_classes,), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes:
            true_counts[int(t)] += 1
            if 0 <= int(p) < num_classes:
                cm_counts[int(t), int(p)] += 1

    denom = true_counts.reshape(-1, 1).clip(min=1)
    cm = cm_counts.astype(np.float64) / denom
    save_root = out_dir / 'confusion_matrix'
    save_root.mkdir(parents=True, exist_ok=True)
    labels = [str(i) for i in range(num_classes)]

    csv_path = save_root / f'{base_name}.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + [f'{float(v):.6f}' for v in row])

    with (save_root / f'{base_name}_counts.json').open('w', encoding='utf-8') as f:
        json.dump(
            {
                'confusion_matrix_counts': cm_counts.tolist(),
                'true_counts': true_counts.tolist(),
                'rejected_or_ood_predictions': int(((y_pred < 0) | (y_pred >= num_classes)).sum()),
            },
            f,
            indent=2,
        )

    try:
        import matplotlib.pyplot as plt
        fig_size = max(6, num_classes * 0.8)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=labels,
            yticklabels=labels,
            ylabel='True label',
            xlabel='Predicted label',
            title=f'Confusion Matrix ({base_name})',
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    f'{cm[i, j]:.2f}',
                    ha='center',
                    va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=8,
                )
        fig.tight_layout()
        fig.savefig(save_root / f'{base_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        print(f'[Warning] failed to save confusion matrix image for {base_name}: {exc}')

    return cm_counts.tolist()


def build_datasets(args):
    if args.dataset in ('pic', 'raw_signal'):
        from semilearn.datasets.cv_datasets.raw_signal import get_raw_signal

        lb_dset, ulb_dset, eval_dset, test_dset, split_info = get_raw_signal(
            args,
            'raw_signal',
            data_dir=args.data_dir,
            include_lb_to_ulb=args.include_lb_to_ulb,
            return_test=True,
        )
        lb = ImageDictSignalDataset(lb_dset, x_key='x_lb', y_key='y_lb')
        ulb = ImageDictSignalDataset(ulb_dset, x_key='x_ulb_w', y_key='y_ulb')
        eval_set = ImageDictSignalDataset(eval_dset, x_key='x_lb', y_key='y_lb')
        test_set = ImageDictSignalDataset(test_dset, x_key='x_lb', y_key='y_lb')
        class_counts = np.asarray(split_info['lb_count'], dtype=np.int64)
        print('PIC split/count info:', split_info)
        return lb, ulb, eval_set, test_set, class_counts

    if not args.train_lb_npz or not args.train_ulb_npz or not args.test_npz:
        raise ValueError('--train_lb_npz, --train_ulb_npz, and --test_npz are required when --dataset npz')
    lb = load_npz(args.train_lb_npz, args.channel_last, not args.no_normalize)
    ulb = load_npz(args.train_ulb_npz, args.channel_last, not args.no_normalize)
    eval_set = load_npz(args.eval_npz or args.test_npz, args.channel_last, not args.no_normalize)
    test_set = load_npz(args.test_npz, args.channel_last, not args.no_normalize)
    class_counts = class_counts_from_dataset(lb, args.num_classes)
    return lb, ulb, eval_set, test_set, class_counts


def parse_args():
    p = argparse.ArgumentParser(description='Train SAT-iFixMatch for Meta-Expert comparison.')
    p.add_argument('--dataset', type=str, default='npz', choices=['npz', 'pic', 'raw_signal'])
    p.add_argument('--train_lb_npz', type=str, default='')
    p.add_argument('--train_ulb_npz', type=str, default='')
    p.add_argument('--eval_npz', type=str, default='')
    p.add_argument('--test_npz', type=str, default='')
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--num_classes', type=int, default=9)
    p.add_argument('--input_channels', type=int, default=None)
    p.add_argument('--channel_last', action='store_true')
    p.add_argument('--no_normalize', action='store_true')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps_per_epoch', type=int, default=None)
    p.add_argument('--batch_size_lb', type=int, default=32)
    p.add_argument('--batch_size_ulb', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lambda_p', type=float, default=1.0)
    p.add_argument('--lambda_o', type=float, default=0.5)
    p.add_argument('--lambda_r', type=float, default=0.1)
    p.add_argument('--lambda_t', type=float, default=0.5)
    p.add_argument('--sat_momentum', type=float, default=0.9)
    p.add_argument('--tau_ood', type=float, default=0.5)
    p.add_argument('--ood_temperature', type=float, default=100.0)
    p.add_argument('--reject_unselected', action='store_true')
    p.add_argument('--scale_min', type=float, default=0.8)
    p.add_argument('--scale_max', type=float, default=1.2)
    p.add_argument('--mask_prob', type=float, default=0.1)
    p.add_argument('--noise_std', type=float, default=0.0)
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--split_seed', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--out_dir', type=str, default='runs/sat_ifixmatch')

    # PIC / Semilearn split arguments. These mirror the baseline YAML fields.
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--crop_ratio', type=float, default=0.875)
    p.add_argument('--num_labels', type=int, default=100)
    p.add_argument('--ulb_num_labels', type=int, default=250)
    p.add_argument('--lb_imb_type', type=str, default='exp')
    p.add_argument('--lb_imb_ratio', type=float, default=50)
    p.add_argument('--ulb_imb_type', type=str, default='exp')
    p.add_argument('--ulb_imb_ratio', type=float, default=50)
    p.add_argument('--num_steps', type=int, default=5)
    p.add_argument('--noise_type', type=str, default='asym')
    p.add_argument('--noise_ratio', type=float, default=0.0)
    p.add_argument('--noise_per_class', action='store_true')
    p.add_argument('--include_lb_to_ulb', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.input_channels is None:
        args.input_channels = 1
    if args.gpu is not None:
        args.device = f'cuda:{args.gpu}'

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lb, ulb, eval_set, test_set, class_counts = build_datasets(args)
    print('Labeled class counts:', class_counts.tolist())
    print('Eval class counts:', class_counts_from_dataset(eval_set, args.num_classes).tolist())
    print('Test class counts:', class_counts_from_dataset(test_set, args.num_classes).tolist())

    lb_loader = make_loader(lb, args.batch_size_lb, True, args.num_workers, drop_last=True)
    ulb_loader = make_loader(ulb, args.batch_size_ulb, True, args.num_workers, drop_last=True)
    eval_loader = make_loader(eval_set, args.batch_size_ulb, False, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size_ulb, False, args.num_workers)

    if len(lb_loader) == 0 or len(ulb_loader) == 0:
        raise ValueError('Batch size is larger than the labeled or unlabeled dataset with drop_last=True.')

    device = torch.device(args.device)
    model = ResNet1D(args.num_classes, args.input_channels, args.base_channels, dropout=args.dropout).to(device)
    sat = SelfAdaptiveThreshold(args.num_classes, args.lambda_t, args.sat_momentum, device=device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lb_iter = infinite_loader(lb_loader)
    ulb_iter = infinite_loader(ulb_loader)
    steps = args.steps_per_epoch or max(len(lb_loader), len(ulb_loader))
    best_eval_acc = -1.0
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        run = []
        bar = tqdm(range(steps), desc=f'epoch {ep}/{args.epochs}', leave=False)
        for _ in bar:
            xlb, ylb = next(lb_iter)
            xulb, _ = next(ulb_iter)
            xlb = xlb.to(device, non_blocking=True)
            ylb = ylb.to(device, non_blocking=True)
            xulb = xulb.to(device, non_blocking=True)
            loss, st = sat_ifixmatch_step(
                model,
                xlb,
                ylb,
                xulb,
                lambda z: weak_augment(z, args.scale_min, args.scale_max),
                lambda z: strong_augment(z, args.mask_prob, args.noise_std),
                sat,
                args.lambda_p,
                args.lambda_o,
                args.lambda_r,
                args.tau_ood,
                args.ood_temperature,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run.append(st)
            bar.set_postfix({'loss': f"{st['loss']:.3f}", 'sel': f"{st['selected_ratio']:.2f}", 'ood': f"{st['ood_like_ratio']:.2f}"})

        met = evaluate(model, eval_loader, device, sat, args, desc='eval')
        item = {
            'epoch': ep,
            'train': {k: float(np.mean([s[k] for s in run])) for k in run[0] if k != 'thresholds'},
            'thresholds': sat.thresholds().detach().cpu().tolist(),
            'eval': met,
        }
        hist.append(item)
        eval_acc = met['known_accuracy']
        print(
            f"Epoch {ep:03d} | Eval Known-ACC={eval_acc*100:.2f}% | "
            f"Overall-ACC={met['overall_accuracy_with_ood']*100:.2f}% | "
            f"MacroF1={met['macro_f1_known']*100:.2f}% | FDR={met['fdr']}"
        )
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(
                {
                    'model': model.state_dict(),
                    'sat_thresholds': sat.thresholds().detach().cpu(),
                    'sat_local': sat.local.detach().cpu(),
                    'sat_global_value': sat.global_value.detach().cpu(),
                    'sat_initialized': sat.initialized,
                    'args': vars(args),
                    'history': hist,
                    'best_eval_acc': best_eval_acc,
                },
                out_dir / 'best_model.pt',
            )
        (out_dir / 'history.json').write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding='utf-8')

    if not (out_dir / 'best_model.pt').exists():
        raise RuntimeError('No best_model.pt was saved. Use --epochs >= 1.')

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    saved = ckpt['sat_thresholds'].to(device)
    sat.local = ckpt.get("sat_local", saved).to(device).clone()
    sat.global_value = ckpt.get("sat_global_value", saved.max()).to(device).clone()
    sat.initialized = ckpt.get("sat_initialized", True)

    eval_final = evaluate(model, eval_loader, device, sat, args, desc='eval-best')
    y_test, pred_test = predict(model, test_loader, device, sat, args, desc='test-best')
    test_final = summarize(y_test, pred_test, args.num_classes)
    save_confusion_outputs(y_test, pred_test, args.num_classes, out_dir, 'model_best_test')
    from scripts.signal_metrics import write_signal_metrics
    write_signal_metrics(out_dir, y_test, pred_test, args.num_classes)

    results = {
        'args': vars(args),
        'class_counts': class_counts.tolist(),
        'best_eval_accuracy_during_training': best_eval_acc,
        'best_eval_metrics': eval_final,
        'final_test_metrics': test_final,
        'history': hist,
    }
    (out_dir / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\nFinal metrics:')
    print(json.dumps(test_final, ensure_ascii=False, indent=2))
    print(f'Saved to: {out_dir}')


if __name__ == '__main__':
    main()
