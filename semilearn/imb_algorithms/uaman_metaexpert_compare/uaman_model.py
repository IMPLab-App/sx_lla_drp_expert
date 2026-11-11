from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(channels, channels, kernel_size=kernel_size), ConvBNAct(channels, channels, kernel_size=kernel_size, act=False))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.net(x))

class SharedConvTokenEmbedding(nn.Module):
    def __init__(self, input_channels: int = 1, conv_channels: int = 64, embed_dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(input_channels, conv_channels, kernel_size=15, stride=2),
            ResidualBlock1D(conv_channels, kernel_size=9),
            ConvBNAct(conv_channels, conv_channels, kernel_size=7, stride=2),
            ResidualBlock1D(conv_channels, kernel_size=5),
        )
        self.patch_embed = nn.Conv1d(conv_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        feat = self.stem(x)
        tok = self.patch_embed(feat).transpose(1, 2)
        return self.norm(tok)

def _pad_to_window_size(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int]:
    b, n, d = x.shape
    pad_len = (window_size - n % window_size) % window_size
    if pad_len:
        x = F.pad(x, (0, 0, 0, pad_len))
    return x, pad_len

def window_partition_1d(x: torch.Tensor, window_size: int):
    b, n, d = x.shape
    return x.view(b, n // window_size, window_size, d).reshape(-1, window_size, d)

def window_reverse_1d(windows: torch.Tensor, window_size: int, batch_size: int, seq_len: int):
    d = windows.shape[-1]
    return windows.view(batch_size, seq_len // window_size, window_size, d).reshape(batch_size, seq_len, d)

class SwinBlock1D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if shift_size >= window_size:
            raise ValueError('shift_size must be smaller than window_size')
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        hidden = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed_dim), nn.Dropout(dropout))
    def forward(self, x):
        b, n, d = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        if self.shift_size > 0:
            x_norm = torch.roll(x_norm, shifts=-self.shift_size, dims=1)
        x_pad, pad_len = _pad_to_window_size(x_norm, self.window_size)
        _, n_pad, _ = x_pad.shape
        windows = window_partition_1d(x_pad, self.window_size)
        attn_windows, _ = self.attn(windows, windows, windows, need_weights=False)
        x_attn = window_reverse_1d(attn_windows, self.window_size, b, n_pad)
        if pad_len:
            x_attn = x_attn[:, :-pad_len, :]
        if self.shift_size > 0:
            x_attn = torch.roll(x_attn, shifts=self.shift_size, dims=1)
        x = shortcut + self.drop1(x_attn)
        return x + self.mlp(self.norm2(x))

class ExpertBranch(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, depth: int = 2, num_heads: int = 4, window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(SwinBlock1D(embed_dim, num_heads, window_size, shift_size=0, dropout=dropout))
            blocks.append(SwinBlock1D(embed_dim, num_heads, window_size, shift_size=window_size // 2, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, tokens):
        z = self.norm(self.blocks(tokens))
        token_logits = self.classifier(z)
        pooled_logits = token_logits.mean(dim=1)
        return token_logits, pooled_logits

class UAMAN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 1, conv_channels: int = 64, embed_dim: int = 128, patch_size: int = 16, expert_depth: int = 1, num_heads: int = 4, window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.shared = SharedConvTokenEmbedding(input_channels=input_channels, conv_channels=conv_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.experts = nn.ModuleList([ExpertBranch(embed_dim, num_classes, depth=expert_depth, num_heads=num_heads, window_size=window_size, dropout=dropout) for _ in range(3)])
    def forward(self, x):
        tokens = self.shared(x)
        token_logits, pooled_logits = [], []
        for expert in self.experts:
            tl, pl = expert(tokens)
            token_logits.append(tl)
            pooled_logits.append(pl)
        return {'token_logits': token_logits, 'logits': pooled_logits}
    @torch.no_grad()
    def predict_logits(self, x, alpha: Optional[torch.Tensor] = None):
        out = self.forward(x)
        logits = torch.stack(out['logits'], dim=1)
        if alpha is None:
            alpha = torch.ones(3, device=x.device) / 3.0
        return (logits * alpha.view(1, 3, 1)).sum(dim=1)

class ExpertAggregator(nn.Module):
    def __init__(self, num_experts: int = 3):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(num_experts))
    def alpha(self):
        return torch.softmax(self.raw, dim=0)
    def forward(self, expert_logits: List[torch.Tensor]):
        logits = torch.stack(expert_logits, dim=1)
        alpha = self.alpha().to(logits.device)
        return (logits * alpha.view(1, -1, 1)).sum(dim=1)
