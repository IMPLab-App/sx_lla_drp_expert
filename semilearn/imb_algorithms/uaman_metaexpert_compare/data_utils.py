from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Tuple, List

class SignalArrayDataset(Dataset):
    """
    Dataset for vibration signals stored in arrays.
    x shape: [N,L] -> [N,1,L], [N,C,L] kept, [N,L,C] with channel_last=True.
    y shape: [N], integer labels in [0, C-1].
    """
    def __init__(self, x, y=None, channel_last: bool = False, normalize: bool = True):
        x = np.asarray(x)
        if x.ndim == 2:
            x = x[:, None, :]
        elif x.ndim == 3 and channel_last:
            x = np.transpose(x, (0, 2, 1))
        elif x.ndim != 3:
            raise ValueError(f'x must have shape [N,L], [N,C,L], or [N,L,C], got {x.shape}')
        x = x.astype(np.float32)
        if normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-8
            x = (x - mean) / std
        self.x = torch.from_numpy(x)
        self.y = None if y is None else torch.from_numpy(np.asarray(y).astype(np.int64))
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx], -1
        return self.x[idx], self.y[idx]

def load_npz_dataset(path: str, channel_last: bool = False, normalize: bool = True) -> SignalArrayDataset:
    data = np.load(path)
    if 'x' not in data:
        raise KeyError(f"{path} must contain array key 'x'")
    y = data['y'] if 'y' in data else None
    return SignalArrayDataset(data['x'], y, channel_last=channel_last, normalize=normalize)

def class_counts_from_dataset(dataset: Dataset, num_classes: int) -> np.ndarray:
    y = []
    for _, label in dataset:
        y.append(int(label))
    y = np.asarray(y)
    return np.array([(y == c).sum() for c in range(num_classes)], dtype=np.int64)

def make_long_tail_indices(y: Sequence[int], num_classes: int, max_per_class: int, rates: Sequence[float], seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if len(rates) != num_classes:
        raise ValueError(f'len(rates)={len(rates)} must equal num_classes={num_classes}')
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(np.int64)
    selected, actual_counts = [], []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = int(round(max_per_class * float(rates[c])))
        n = max(1, min(n, len(idx)))
        selected.append(idx[:n])
        actual_counts.append(n)
    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return selected.astype(np.int64), np.asarray(actual_counts, dtype=np.int64)

def make_exp_long_tail_rates(num_classes: int, imbalance_ratio: float) -> List[float]:
    if num_classes <= 1:
        return [1.0]
    return [(1.0 / imbalance_ratio) ** (c / (num_classes - 1)) for c in range(num_classes)]

def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, drop_last: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last)


class ImageDictSignalDataset(Dataset):
    """
    Wrap a Semilearn image BasicDataset sample into UA-MAN's [C, L] signal format.
    PIC images arrive as torchvision tensors [C, H, W]; UA-MAN consumes [C, L].
    """
    def __init__(self, dataset: Dataset, x_key: str = 'x_lb', y_key: str = 'y_lb'):
        self.dataset = dataset
        self.x_key = x_key
        self.y_key = y_key

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _as_signal(x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        x = x.float()
        if x.ndim == 3:       # [C, H, W] -> [C, H*W]
            return x.reshape(x.shape[0], -1)
        if x.ndim == 2:       # [H, W] -> [1, H*W]
            return x.reshape(1, -1)
        if x.ndim == 1:       # [L] -> [1, L]
            return x.unsqueeze(0)
        raise ValueError(f'Expected image/signal tensor with 1-3 dims, got {tuple(x.shape)}')

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            x = sample[self.x_key] if self.x_key in sample else sample.get('x_ulb_w')
            y = sample[self.y_key] if self.y_key in sample else sample.get('y_ulb', -1)
        else:
            x, y = sample
        return self._as_signal(x), torch.as_tensor(int(y), dtype=torch.long)
