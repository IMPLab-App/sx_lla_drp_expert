"""CoLA effective-rank DDDE with the project's custom TWO-reference tau map.

This deliberately differs from CoLA's original three-reference mapping and has
no learnable atau phase. All statistics are buffers for exact state_dict resume.
"""
import torch
from torch import nn


def normalize_distribution(p):
    p = torch.as_tensor(p)
    p = p.to(dtype=torch.float32)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    # Scale first so even very large finite inputs cannot overflow the sum.
    p = p / p.max().clamp_min(torch.finfo(p.dtype).tiny)
    if p.sum() <= 0:
        p = torch.ones_like(p)
    p = p.clamp_min(1e-12)
    return p / p.sum()


@torch.no_grad()
def effective_rank(features):
    """exp(entropy(normalized singular values)) of centered class features."""
    features = features.detach().float()
    features = features[torch.isfinite(features).all(dim=1)]
    if features.shape[0] == 0:
        return features.new_tensor(0.)
    if features.shape[0] == 1:
        return features.new_tensor(1.)
    # Scaling preserves effective rank and avoids overflow when centering.
    scale = features.abs().max()
    if scale <= 0:
        return features.new_tensor(0.)
    features = features / scale
    features = features - features.mean(dim=0, keepdim=True)
    try:
        singular = torch.linalg.svdvals(features)
    except RuntimeError:
        return features.new_tensor(0.)
    singular = singular[torch.isfinite(singular) & (singular > 1e-12)]
    if not singular.numel():
        return features.new_tensor(0.)
    p = singular / singular.sum()
    return torch.exp(-(p * p.log()).sum())


def symmetric_kl(p, q):
    p = normalize_distribution(p).double()
    q = normalize_distribution(q).to(device=p.device, dtype=p.dtype)
    return (0.5 * ((p - q) * (p.log() - q.log())).sum()).clamp_min(0).float()


def tau_from_kl(d_uni, d_rev, tau_max):
    scores = -torch.stack((d_uni, d_rev))
    return (float(tau_max) * torch.softmax(scores, dim=0)[1]).clamp(0, float(tau_max))


class TwoKLDDDE(nn.Module):
    def __init__(self, pi_l, tau_max=4.0, est_epoch=18, ema_u=0.9):
        super().__init__()
        if not 0 <= ema_u <= 1 or tau_max < 0:
            raise ValueError('Invalid DDDE EMA/tau configuration')
        self.tau_max, self.est_epoch, self.ema_u = float(tau_max), int(est_epoch), float(ema_u)
        prior = normalize_distribution(pi_l)
        self.register_buffer('pi_l', prior)
        self.register_buffer('pi_uni', torch.ones_like(prior) / prior.numel())
        self.register_buffer('pi_rev', prior.flip(0))
        self.register_buffer('pi_u', torch.ones_like(prior) / prior.numel())
        self.register_buffer('pi_ddde', torch.zeros_like(prior))
        self.register_buffer('tau_dyn', prior.new_tensor(tau_max))
        self.register_buffer('kl_sum', prior.new_zeros(2))
        self.register_buffer('last_kl', prior.new_zeros(2))
        self.register_buffer('valid_batches', torch.zeros((), dtype=torch.long, device=prior.device))
        self.register_buffer('total_valid_batches', torch.zeros((), dtype=torch.long, device=prior.device))
        self.register_buffer('stats_epoch', torch.full((), -1, dtype=torch.long, device=prior.device))

    @torch.no_grad()
    def begin_epoch(self, epoch):
        if epoch == self.stats_epoch.item():
            return
        if epoch < self.est_epoch:
            self.tau_dyn.fill_(self.tau_max)
        elif self.valid_batches.item():
            self.last_kl.copy_(self.kl_sum / self.valid_batches)
            self.tau_dyn.copy_(tau_from_kl(*self.last_kl, self.tau_max))
        self.kl_sum.zero_()
        self.valid_batches.zero_()
        self.stats_epoch.fill_(epoch)

    @torch.no_grad()
    def observe(self, tail_raw_logits, shared_features):
        if self.stats_epoch.item() < self.est_epoch:
            return None
        if tail_raw_logits.ndim != 2 or shared_features.ndim != 2 or tail_raw_logits.shape[0] != shared_features.shape[0]:
            raise ValueError('DDDE requires aligned [B,C] logits and [B,D] shared features')
        # Do not turn a NaN/Inf row into an artificially confident prediction.
        valid = torch.isfinite(tail_raw_logits).all(dim=1) & torch.isfinite(shared_features).all(dim=1)
        logits = tail_raw_logits.detach().float()[valid]
        feats = shared_features.detach()[valid]
        if not len(logits):
            return None
        probs, labels = logits.softmax(dim=-1).max(dim=-1)
        select = probs.ge(0.95)
        if not select.any():
            return None
        feats, labels = feats[select], labels[select]
        ranks = torch.stack([effective_rank(feats[labels == k]) for k in range(self.pi_l.numel())])
        if not torch.isfinite(ranks).all() or ranks.sum() <= 0:
            return None
        self.pi_ddde.copy_(ranks / ranks.sum())
        self.pi_u.copy_(normalize_distribution(self.ema_u * self.pi_u + (1 - self.ema_u) * self.pi_ddde))
        kl = torch.stack((symmetric_kl(self.pi_u, self.pi_uni), symmetric_kl(self.pi_u, self.pi_rev)))
        if not torch.isfinite(kl).all():
            return None
        self.kl_sum.add_(kl)
        self.valid_batches.add_(1)
        self.total_valid_batches.add_(1)
        return kl

    def snapshot(self):
        return dict(tau_dyn=self.tau_dyn.item(), KL_uni=self.last_kl[0].item(),
                    KL_rev=self.last_kl[1].item(), pi_u=self.pi_u.cpu().tolist(),
                    pi_DDDE=self.pi_ddde.cpu().tolist(), valid_batches=self.valid_batches.item())
