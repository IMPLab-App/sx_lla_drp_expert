"""E3-only DRP adaptation of SAGE's MIT-licensed WeightingHook.

Reference: Yaxin-ML/SAGE, commit d3f3ca71abc9828daa8273cc2290187baeb9f4f7,
semilearn/imb_algorithms/sage/utils.py. No GRI, anchors, or extra classifier.
"""
import math
import torch
from torch import nn


class DRPE3Controller(nn.Module):
    def __init__(self, num_classes, momentum=.999, n_sigma=2., per_class=False, epsilon=1e-8):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('DRP requires at least two actual classes')
        if not 0 <= momentum < 1 or not math.isfinite(n_sigma) or n_sigma <= 0:
            raise ValueError('Invalid DRP momentum / n_sigma')
        if not 0 < epsilon < 1 or per_class:
            raise ValueError('Main protocol requires global statistics and epsilon in (0,1)')
        self.num_classes, self.momentum, self.n_sigma = num_classes, momentum, n_sigma
        self.epsilon = epsilon
        for name, initial in (('mu_max', 1 / num_classes), ('var_max', 1.),
                              ('mu_gap', 1 / num_classes), ('var_gap', 1.)):
            self.register_buffer(name, torch.tensor(initial, dtype=torch.float32))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('observations', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update(self, confidence, gap):
        # Sufficient statistics allow unequal/empty local DDP batches. Use
        # float64 reductions to prevent cancellation of near-zero variance.
        q = torch.stack((confidence, gap), dim=-1).double()
        stats = torch.cat((q.new_tensor([len(q)]), q.sum(0), q.square().sum(0)))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        count = int(stats[0].item())
        if count == 0:
            return
        means = stats[1:3] / count
        # Official unbiased=True is undefined for n=1. A single observation
        # has zero within-batch variance in this explicit numerical adaptation.
        variance = ((stats[3:5] - count * means.square()) / (count - 1)).clamp_min(0) if count > 1 else means.new_zeros(2)
        if not torch.isfinite(means).all() or not torch.isfinite(variance).all():
            raise FloatingPointError('Non-finite DRP batch moments')
        for name, value in zip(('mu_max', 'mu_gap'), means):
            getattr(self, name).mul_(self.momentum).add_(value.float(), alpha=1-self.momentum)
        for name, value in zip(('var_max', 'var_gap'), variance):
            getattr(self, name).mul_(self.momentum).add_(value.float(), alpha=1-self.momentum)
        self.updates.add_(1)
        self.observations.add_(count)

    @torch.no_grad()
    def forward(self, probabilities, update=True):
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError('DRP probabilities must match actual class count')
        p = probabilities.detach().float()
        if (not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not torch.allclose(p.sum(-1), torch.ones(len(p), device=p.device), atol=1e-5, rtol=1e-5)):
            raise ValueError('DRP requires finite normalized probabilities')
        if p.device != self.mu_max.device:
            raise ValueError('Move DRP state to the model device before use')
        top = p.topk(2, dim=-1).values
        confidence, gap = top[:, 0], top[:, 0] - top[:, 1]
        if update:
            self._update(confidence, gap)
        if not all(torch.isfinite(getattr(self, k)) for k in ('mu_max','mu_gap','var_max','var_gap')):
            raise FloatingPointError('Non-finite DRP EMA state')
        # Official order: update EMA first, then use the updated moments.
        max_term = (confidence-self.mu_max).clamp_max(0).square() / (2*(self.var_max/self.n_sigma**2+self.epsilon))
        gap_term = (gap-self.mu_gap).clamp_max(0).square() / (2*(self.var_gap/self.n_sigma**2+self.epsilon))
        # Evaluate the product in log-space. Only numerical underflow is
        # floored; no extra learning threshold or class weighting is added.
        log_weight = -(max_term+gap_term)
        weight = log_weight.clamp_min(math.log(2*torch.finfo(torch.float32).tiny)).exp()
        if not torch.isfinite(weight).all() or (weight <= 0).any() or (weight > 1).any():
            raise FloatingPointError('DRP weight must be in (0,1]')
        return weight.detach(), confidence, gap

    def settings(self):
        return dict(num_classes=self.num_classes, momentum=self.momentum, n_sigma=self.n_sigma,
                    per_class=False, epsilon=self.epsilon, variance='unbiased_n_gt_1_else_zero',
                    update_order='update_before_weight', positive_floor='2_times_float32_tiny_log_space')

    @torch.no_grad()
    def diagnostics(self, probabilities, weights, effective_weights, cutoff=.95):
        p=probabilities.detach().float()
        pseudo=p.argmax(-1)
        top=p.topk(2, dim=-1).values
        def values(x):
            return x.detach().cpu().tolist()
        counts=torch.bincount(pseudo,minlength=self.num_classes)
        totals=weights.new_zeros(self.num_classes).scatter_add_(0,pseudo,weights)
        effective=weights.new_zeros(self.num_classes).scatter_add_(0,pseudo,effective_weights)
        total=effective_weights.sum()
        normalized=effective_weights/effective_weights.max() if len(weights) and effective_weights.max()>0 else effective_weights
        ess=normalized.sum().square()/normalized.square().sum() if normalized.square().sum()>0 else weights.new_zeros(())
        hard=(top[:,0]>=cutoff)
        return dict(mu_max=float(self.mu_max),var_max=float(self.var_max),mu_gap=float(self.mu_gap),var_gap=float(self.var_gap),
            updates=int(self.updates),observations=int(self.observations),sample_count=len(p),
            max_confidence_mean=float(top[:,0].mean()) if len(p) else None,
            top2_gap_mean=float((top[:,0]-top[:,1]).mean()) if len(p) else None,
            weight_mean=float(weights.mean()) if len(p) else None,
            weight_min=float(weights.min()) if len(p) else None,weight_max=float(weights.max()) if len(p) else None,
            weight_quantiles=values(torch.quantile(weights,weights.new_tensor([0.,.25,.5,.75,1.]))) if len(p) else [],
            weight_histogram=values(torch.histc(weights,bins=10,min=0,max=1)),
            predicted_class_counts=values(counts),class_weight_sums=values(totals),
            class_effective_weight_sums=values(effective),
            class_weighted_participation=values(effective/total) if total>0 else [0.]*self.num_classes,
            class_mean_effective_weights=values(effective/counts.clamp_min(1)),effective_sample_size=float(ess),
            effective_nonzero_samples=int((effective_weights>0).sum()),
            original_hard_accepted=int(hard.sum()),
            original_hard_class_counts=values(torch.bincount(pseudo[hard],minlength=self.num_classes)))
