from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SignalDataset(Dataset):
    def __init__(self, x, y=None, channel_last=False, normalize=True):
        x=np.asarray(x)
        if x.ndim==2:
            x=x[:,None,:]
        elif x.ndim==3 and channel_last:
            x=np.transpose(x,(0,2,1))
        elif x.ndim!=3:
            raise ValueError(f'x must be [N,L], [N,C,L], or [N,L,C], got {x.shape}')
        x=x.astype(np.float32)
        if normalize:
            x=(x-x.mean(axis=-1,keepdims=True))/(x.std(axis=-1,keepdims=True)+1e-8)
        self.x=torch.from_numpy(x)
        self.y=torch.full((x.shape[0],),-999,dtype=torch.long) if y is None else torch.from_numpy(np.asarray(y).astype(np.int64))
    def __len__(self): return self.x.shape[0]
    def __getitem__(self,i): return self.x[i], self.y[i]

def load_npz(path, channel_last=False, normalize=True):
    a=np.load(path)
    if 'x' not in a.files: raise KeyError(f'{path} must contain key x')
    y=a['y'] if 'y' in a.files else None
    return SignalDataset(a['x'],y,channel_last,normalize)

def make_loader(ds,batch_size,shuffle,num_workers=0,drop_last=False):
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=torch.cuda.is_available(),drop_last=drop_last)

def infinite_loader(loader):
    while True:
        for b in loader:
            yield b

def class_counts_from_dataset(dataset: Dataset, num_classes: int):
    y = []
    for _, label in dataset:
        y.append(int(label))
    y = np.asarray(y)
    return np.array([(y == c).sum() for c in range(num_classes)], dtype=np.int64)


class ImageDictSignalDataset(Dataset):
    """
    Wrap a Semilearn image BasicDataset sample into SAT-iFixMatch's [C, L] signal format.
    PIC images arrive as torchvision tensors [C, H, W]; the 1D ResNet consumes [C, L].
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
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        if x.ndim == 2:
            return x.reshape(1, -1)
        if x.ndim == 1:
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
