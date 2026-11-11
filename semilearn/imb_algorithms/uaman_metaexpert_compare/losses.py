from __future__ import annotations
import torch
import torch.nn.functional as F

def _expand_labels_for_tokens(y: torch.Tensor, num_tokens: int) -> torch.Tensor:
    return y[:, None].expand(-1, num_tokens).reshape(-1)

def token_ce_loss(logits_tokens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, npatch, c = logits_tokens.shape
    return F.cross_entropy(logits_tokens.reshape(b * npatch, c), _expand_labels_for_tokens(y, npatch))

def token_adjusted_ce_loss(logits_tokens: torch.Tensor, y: torch.Tensor, logit_adjustment: torch.Tensor) -> torch.Tensor:
    b, npatch, c = logits_tokens.shape
    adjusted = logits_tokens + logit_adjustment.view(1, 1, c)
    return F.cross_entropy(adjusted.reshape(b * npatch, c), _expand_labels_for_tokens(y, npatch))

def class_prior_from_counts(class_counts, device=None, eps: float = 1e-12) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device).clamp_min(1.0)
    return counts / counts.sum().clamp_min(eps)

def balanced_softmax_loss(logits_tokens: torch.Tensor, y: torch.Tensor, class_counts) -> torch.Tensor:
    prior = class_prior_from_counts(class_counts, device=logits_tokens.device)
    return token_adjusted_ce_loss(logits_tokens, y, torch.log(prior.clamp_min(1e-12)))

def inverse_longtail_loss(logits_tokens: torch.Tensor, y: torch.Tensor, class_counts, beta: float = 2.0) -> torch.Tensor:
    prior = class_prior_from_counts(class_counts, device=logits_tokens.device)
    inv_prior = torch.flip(prior, dims=[0])
    adjustment = -float(beta) * torch.log(inv_prior.clamp_min(1e-12))
    return token_adjusted_ce_loss(logits_tokens, y, adjustment)

def uaman_total_loss(outputs: dict, y: torch.Tensor, class_counts, beta: float = 2.0) -> dict:
    e1, e2, e3 = outputs['token_logits']
    loss_e1 = token_ce_loss(e1, y)
    loss_e2 = balanced_softmax_loss(e2, y, class_counts)
    loss_e3 = inverse_longtail_loss(e3, y, class_counts, beta=beta)
    total = loss_e1 + loss_e2 + loss_e3
    return {'loss': total, 'loss_e1': loss_e1.detach(), 'loss_e2': loss_e2.detach(), 'loss_e3': loss_e3.detach()}
