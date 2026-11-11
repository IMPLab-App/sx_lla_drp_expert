from __future__ import annotations
import torch

def weak_augment(x, scale_min=0.8, scale_max=1.2):
    b=x.shape[0]
    scale=torch.empty((b,1,1),device=x.device).uniform_(scale_min,scale_max)
    return x*scale

def strong_augment(x, mask_prob=0.1, noise_std=0.0):
    mask=(torch.rand_like(x)>mask_prob).float()
    out=x*mask
    if noise_std>0: out=out+torch.randn_like(out)*noise_std
    return out
