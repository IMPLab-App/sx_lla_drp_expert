from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import (
    ImageDictSignalDataset,
    load_npz_dataset,
    build_loader,
    make_long_tail_indices,
    make_exp_long_tail_rates,
    class_counts_from_dataset,
)
from uaman_model import UAMAN, ExpertAggregator
from losses import uaman_total_loss
from metrics import summarize_metrics, confusion_matrix_np


def parse_rates(s: str, num_classes: int):
    if s is None or s.strip() == '':
        return None
    vals = [float(v.strip()) for v in s.split(',')]
    if len(vals) != num_classes:
        raise ValueError(f'--lt_rates has {len(vals)} values, expected {num_classes}')
    return vals


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_view(x: torch.Tensor, noise_std: float = 0.05, invert: bool = False) -> torch.Tensor:
    z = torch.randn_like(x) * noise_std
    return -x + z if invert else x + z


def train_one_epoch(model, loader, optimizer, device, class_counts, beta: float):
    model.train(); total_loss = 0.0; n = 0
    for x, y in tqdm(loader, desc='train', leave=False):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        losses = uaman_total_loss(outputs, y, class_counts=class_counts, beta=beta)
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = x.size(0); total_loss += float(losses['loss'].detach().cpu()) * bs; n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def predict(model, loader, device, aggregator=None):
    model.eval(); y_true, y_pred = [], []
    for x, y in tqdm(loader, desc='eval', leave=False):
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits = torch.stack(out['logits'], dim=1).mean(dim=1) if aggregator is None else aggregator(out['logits'])
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(pred.tolist()); y_true.extend(y.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def adapt_aggregator(model, loader, device, steps: int = 200, lr: float = 5e-2, noise_std: float = 0.05):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    aggregator = ExpertAggregator(num_experts=3).to(device)
    opt = optim.Adam(aggregator.parameters(), lr=lr)
    if steps <= 0:
        return aggregator
    iterator = iter(loader)
    pbar = tqdm(range(steps), desc='unsup-aggregate', leave=False)
    for _ in pbar:
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader); x, _ = next(iterator)
        x = x.to(device, non_blocking=True)
        x1 = augment_view(x, noise_std=noise_std, invert=False)
        x2 = augment_view(x, noise_std=noise_std, invert=True)
        with torch.no_grad():
            out1 = model(x1); out2 = model(x2)
        pred1 = F.softmax(aggregator(out1['logits']), dim=1)
        pred2 = F.softmax(aggregator(out2['logits']), dim=1)
        loss = 1.0 - F.cosine_similarity(pred1, pred2, dim=1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pbar.set_postfix({'loss': f'{float(loss.detach().cpu()):.4f}', 'alpha': aggregator.alpha().detach().cpu().numpy().round(3).tolist()})
    return aggregator


def save_confusion_outputs(y_true, y_pred, num_classes: int, out_dir: Path, base_name: str):
    cm_counts = confusion_matrix_np(y_true, y_pred, num_classes)
    denom = cm_counts.sum(axis=1, keepdims=True).clip(min=1)
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
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black', fontsize=8)
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
        train_set = ImageDictSignalDataset(lb_dset, x_key='x_lb', y_key='y_lb')
        eval_set = ImageDictSignalDataset(eval_dset, x_key='x_lb', y_key='y_lb')
        test_set = ImageDictSignalDataset(test_dset, x_key='x_lb', y_key='y_lb')
        class_counts = np.asarray(split_info['lb_count'], dtype=np.int64)
        print('PIC split/count info:', split_info)
        return train_set, eval_set, test_set, class_counts, ImageDictSignalDataset(ulb_dset, x_key="x_ulb_w", y_key="y_ulb")

    if not args.train_npz or not args.test_npz:
        raise ValueError('--train_npz and --test_npz are required when --dataset npz')
    train_set = load_npz_dataset(args.train_npz, channel_last=args.channel_last, normalize=not args.no_normalize)
    eval_set = load_npz_dataset(args.eval_npz or args.test_npz, channel_last=args.channel_last, normalize=not args.no_normalize)
    test_set = load_npz_dataset(args.test_npz, channel_last=args.channel_last, normalize=not args.no_normalize)
    if args.max_per_class is not None:
        if train_set.y is None:
            raise ValueError('Long-tail subsampling requires labels in train npz.')
        rates = parse_rates(args.lt_rates, args.num_classes) if args.lt_rates else make_exp_long_tail_rates(args.num_classes, args.exp_imbalance_ratio)
        idx, selected_counts = make_long_tail_indices(train_set.y.numpy(), args.num_classes, args.max_per_class, rates, seed=args.seed)
        train_set = Subset(train_set, idx.tolist()); class_counts = selected_counts
    else:
        class_counts = class_counts_from_dataset(train_set, args.num_classes)
    return train_set, eval_set, test_set, class_counts, None


def main():
    parser = argparse.ArgumentParser(description='Train UA-MAN for long-tailed fault diagnosis and Meta-Expert comparison.')
    parser.add_argument('--dataset', type=str, default='npz', choices=['npz', 'pic', 'raw_signal'])
    parser.add_argument('--train_npz', type=str, default='')
    parser.add_argument('--eval_npz', type=str, default='')
    parser.add_argument('--test_npz', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--input_channels', type=int, default=None)
    parser.add_argument('--channel_last', action='store_true')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--conv_channels', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--expert_depth', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--max_per_class', type=int, default=None)
    parser.add_argument('--lt_rates', type=str, default=None)
    parser.add_argument('--exp_imbalance_ratio', type=float, default=None)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--adapt_steps', type=int, default=200)
    parser.add_argument('--adapt_lr', type=float, default=5e-2)
    parser.add_argument('--adapt_split', type=str, default='eval', choices=['train_ulb', 'eval', 'test'])
    parser.add_argument('--noise_std', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--split_seed', type=int, default=2)
    parser.add_argument('--out_dir', type=str, default='runs/uaman')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu', type=int, default=None)

    # PIC / Semilearn split arguments. These mirror the baseline YAML fields.
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)
    parser.add_argument('--num_labels', type=int, default=100)
    parser.add_argument('--ulb_num_labels', type=int, default=250)
    parser.add_argument('--lb_imb_type', type=str, default='exp')
    parser.add_argument('--lb_imb_ratio', type=float, default=50)
    parser.add_argument('--ulb_imb_type', type=str, default='exp')
    parser.add_argument('--ulb_imb_ratio', type=float, default=50)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--noise_type', type=str, default='asym')
    parser.add_argument('--noise_ratio', type=float, default=0.0)
    parser.add_argument('--noise_per_class', action='store_true')
    parser.add_argument('--include_lb_to_ulb', action='store_true')

    args = parser.parse_args()
    if args.input_channels is None:
        args.input_channels = 1
    if args.gpu is not None:
        args.device = f'cuda:{args.gpu}'

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_set, eval_set, test_set, class_counts, ulb_set = build_datasets(args)
    print('Class counts used for training:', class_counts.tolist())
    print('Eval class counts:', class_counts_from_dataset(eval_set, args.num_classes).tolist())
    print('Test class counts:', class_counts_from_dataset(test_set, args.num_classes).tolist())

    if args.steps_per_epoch:
        from torch.utils.data import DataLoader, RandomSampler
        sampler = RandomSampler(train_set, replacement=True, num_samples=args.steps_per_epoch * args.batch_size)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = build_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = build_loader(eval_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = UAMAN(args.num_classes, args.input_channels, args.conv_channels, args.embed_dim, args.patch_size,
                  args.expert_depth, args.num_heads, args.window_size, args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_eval_acc = -1.0; history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_counts, args.beta)
        y_true, y_pred = predict(model, eval_loader, device, aggregator=None)
        metrics = summarize_metrics(y_true, y_pred, args.num_classes)
        eval_acc = metrics['accuracy']
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={loss:.4f} | eval avg-expert acc={eval_acc*100:.2f}% | macroF1={metrics['macro_f1']*100:.2f}%")
        history.append({'epoch': epoch, 'loss': loss, **metrics})
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save({'model': model.state_dict(), 'args': vars(args), 'class_counts': class_counts.tolist(), 'best_eval_acc': best_eval_acc}, out_dir / 'best_model.pt')

    if not (out_dir / 'best_model.pt').exists():
        raise RuntimeError('No best_model.pt was saved. Use --epochs >= 1.')

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model'])

    y_eval, pred_eval_avg = predict(model, eval_loader, device, aggregator=None)
    eval_avg_metrics = summarize_metrics(y_eval, pred_eval_avg, args.num_classes)
    y_test, pred_test_avg = predict(model, test_loader, device, aggregator=None)
    test_avg_metrics = summarize_metrics(y_test, pred_test_avg, args.num_classes)
    save_confusion_outputs(y_test, pred_test_avg, args.num_classes, out_dir, 'avg_expert_test')

    if args.adapt_split == 'train_ulb':
        if ulb_set is None:
            raise ValueError('train_ulb adaptation requires raw_signal data')
        adapt_loader = build_loader(ulb_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        adapt_loader = eval_loader if args.adapt_split == 'eval' else test_loader
    aggregator = adapt_aggregator(model, adapt_loader, device, args.adapt_steps, args.adapt_lr, args.noise_std)
    y_test, pred_test_ua = predict(model, test_loader, device, aggregator=aggregator)
    ua_test_metrics = summarize_metrics(y_test, pred_test_ua, args.num_classes)
    save_confusion_outputs(y_test, pred_test_ua, args.num_classes, out_dir, 'model_best_test')
    from scripts.signal_metrics import write_signal_metrics
    write_signal_metrics(out_dir, y_test, pred_test_ua, args.num_classes)

    results = {
        'args': vars(args),
        'class_counts': class_counts.tolist(),
        'best_eval_accuracy_during_training': best_eval_acc,
        'equal_average_eval_metrics': eval_avg_metrics,
        'equal_average_test_metrics': test_avg_metrics,
        'adapt_split': args.adapt_split,
        'unsupervised_aggregation_alpha': aggregator.alpha().detach().cpu().numpy().tolist(),
        'unsupervised_aggregation_test_metrics': ua_test_metrics,
        'history': history,
    }
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    torch.save({'aggregator': aggregator.state_dict()}, out_dir / 'aggregator.pt')

    print('\nFinal results')
    print('Best eval accuracy during training:', f'{best_eval_acc*100:.2f}%')
    print('Equal expert average test:', json.dumps(test_avg_metrics, indent=2))
    print('Unsupervised aggregation alpha:', np.round(results['unsupervised_aggregation_alpha'], 4).tolist())
    print('Unsupervised aggregation test:', json.dumps(ua_test_metrics, indent=2))
    print(f'\nSaved to: {out_dir}')


if __name__ == '__main__':
    main()
