"""Opt-in, FlexDA-inspired residual on top of the fixed-tau=4 baseline.

This is an experimental extension, not a reproduction of FlexDA/ADELLO.
Only the E3 supervised adjustment and its existing fusion/routing path change.
Pseudo-label generation and unsupervised expert losses retain their raw logits.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn

from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.utils import IMB_ALGORITHMS
from .cola_ddde import effective_rank, normalize_distribution
from .metaexpert import MetaExpert


@torch.no_grad()
def prior_difference_residual(pi_l, pi_u, strength=.5, bound=2., eps=1e-4):
    if not all(math.isfinite(x) for x in (strength, bound, eps)) or strength < 0 or bound <= 0 or eps <= 0:
        raise ValueError('Invalid prior-residual strength/bound/epsilon')
    pi_l = torch.as_tensor(pi_l).detach().float()
    if strength == 0:
        return torch.zeros_like(pi_l)
    pi_u = torch.as_tensor(pi_u, device=pi_l.device).detach().float()
    if pi_l.ndim != 1 or pi_u.shape != pi_l.shape:
        raise ValueError('Aligned one-dimensional class priors required')
    if not torch.isfinite(pi_l).all() or not torch.isfinite(pi_u).all() or (pi_l < 0).any() or (pi_u < 0).any():
        raise ValueError('Class priors must be finite and nonnegative')
    p, q = pi_l.clamp_min(eps), pi_u.clamp_min(eps)
    p, q = p / p.sum(), q / q.sum()
    return strength * (p.log() - q.log()).clamp(-bound, bound)


class PriorResidualController(nn.Module):
    """Estimate from raw weak-view predictions only; no distance-to-tau map."""
    def __init__(self, pi_l, estimator='histogram', ema_u=.9, est_epoch=18,
                 strength=.5, bound=2., eps=1e-4, confidence=.95):
        super().__init__()
        if estimator not in ('histogram', 'effective_rank'):
            raise ValueError('Unknown class-prior estimator')
        if not 0 <= ema_u < 1 or est_epoch < 0 or not 0 < confidence <= 1:
            raise ValueError('Invalid estimation settings')
        prior_difference_residual(pi_l, pi_l, strength, bound, eps)
        self.estimator, self.ema_u, self.est_epoch = estimator, float(ema_u), int(est_epoch)
        self.strength, self.bound, self.eps, self.confidence = float(strength), float(bound), float(eps), float(confidence)
        pi_l = torch.as_tensor(pi_l).detach().float().clone()
        self.register_buffer('pi_l', pi_l)
        self.register_buffer('pi_u', torch.ones_like(pi_l) / pi_l.numel())
        self.register_buffer('pi_raw', torch.zeros_like(pi_l))
        self.register_buffer('pi_used', torch.zeros_like(pi_l))
        self.register_buffer('applied_residual', torch.zeros_like(pi_l))
        self.register_buffer('last_counts', torch.zeros_like(pi_l))
        self.register_buffer('last_rank', torch.zeros_like(pi_l))
        self.register_buffer('total_valid_batches', torch.zeros((), device=pi_l.device, dtype=torch.long))
        self.register_buffer('stats_epoch', torch.full((), -1, device=pi_l.device, dtype=torch.long))

    @torch.no_grad()
    def begin_step(self, epoch):
        self.stats_epoch.fill_(epoch)
        # No correction based solely on the uniform initialization. The current
        # batch uses only statistics from previously completed training batches.
        if epoch >= self.est_epoch and self.total_valid_batches.item():
            self.pi_used.copy_(self.pi_u)
            self.applied_residual.copy_(prior_difference_residual(
                self.pi_l, self.pi_used, self.strength, self.bound, self.eps))
        else:
            self.pi_used.zero_()
            self.applied_residual.zero_()

    @torch.no_grad()
    def observe(self, tail_raw_logits, shared_features):
        self.last_counts.zero_()
        self.last_rank.zero_()
        if self.stats_epoch.item() < self.est_epoch:
            return False
        if tail_raw_logits.ndim != 2 or shared_features.ndim != 2 or len(tail_raw_logits) != len(shared_features):
            raise ValueError('Aligned weak logits/features required')
        valid = torch.isfinite(tail_raw_logits).all(1) & torch.isfinite(shared_features).all(1)
        logits, features = tail_raw_logits.detach().float()[valid], shared_features.detach()[valid]
        if not len(logits):
            return False
        confidence, labels = logits.softmax(-1).max(-1)
        selected = confidence.ge(self.confidence)
        if not selected.any():
            return False
        features, labels = features[selected], labels[selected]
        counts = torch.bincount(labels, minlength=self.pi_l.numel()).to(self.pi_u)
        self.last_counts.copy_(counts)
        if self.estimator == 'histogram':
            estimate = counts
        else:
            estimate = torch.stack([effective_rank(features[labels == c]) for c in range(self.pi_l.numel())])
            self.last_rank.copy_(estimate)
        if not torch.isfinite(estimate).all() or estimate.sum() <= 0:
            return False
        self.pi_raw.copy_(estimate / estimate.sum())
        self.pi_u.copy_(normalize_distribution(self.ema_u * self.pi_u + (1 - self.ema_u) * self.pi_raw))
        self.total_valid_batches.add_(1)
        return True


@IMB_ALGORITHMS.register('metaexpert_prior_residual')
class PriorResidualMetaExpert(MetaExpert):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        if float(args.la_tau_lb3) != 4.:
            raise ValueError('The prior-residual experiment retains the fixed tau=4 baseline')
        strength, bound, eps = args.prior_residual_lambda, args.prior_residual_bound, args.prior_residual_eps
        if not all(math.isfinite(x) for x in (strength, bound, eps)) or strength < 0 or bound <= 0 or eps <= 0:
            raise ValueError('Invalid prior-residual strength/bound/epsilon')
        if args.prior_residual_estimator not in ('histogram', 'effective_rank'):
            raise ValueError('Unknown class-prior estimator')
        super().__init__(args, net_builder, tb_log, logger)
        self.prior_residual_active = bool(args.prior_residual_enabled and strength > 0)
        self.prior_controller = None
        if self.prior_residual_active:
            self.prior_controller = PriorResidualController(self.p_hat_lb,
                args.prior_residual_estimator, args.ema_u, args.est_epoch,
                strength, bound, eps, args.p_cutoff)
            for model in (self.model, self.ema_model):
                module = model.module if hasattr(model, 'module') else model
                module.register_buffer('tail_prior_residual', torch.zeros_like(self.p_hat_lb))
        self._prior_record_epoch = None

    def prior_settings(self):
        a = self.args
        return dict(enabled=bool(a.prior_residual_enabled), active=self.prior_residual_active,
            estimator=a.prior_residual_estimator, strength=float(a.prior_residual_lambda),
            bound=float(a.prior_residual_bound), eps=float(a.prior_residual_eps),
            tau_fixed=4., ema_u=float(a.ema_u), est_epoch=int(a.est_epoch), confidence=float(a.p_cutoff),
            update='previous completed batch; no gradients; same applied vector in E3 supervision and DEA',
            unsupervised_adjustment=False)

    def tail_logit_adjustment(self):
        base = super().tail_logit_adjustment()
        return base if self.prior_controller is None else base + self.prior_controller.applied_residual

    @torch.no_grad()
    def _sync_prior_residual(self):
        if self.prior_controller is not None:
            for model in (self.model, self.ema_model):
                module = model.module if hasattr(model, 'module') else model
                module.tail_prior_residual.copy_(self.prior_controller.applied_residual)

    def get_save_dict(self):
        saved = super().get_save_dict()
        saved['prior_residual_settings'] = self.prior_settings()
        if self.prior_controller is not None:
            saved['prior_residual_controller'] = self.prior_controller.state_dict()
        return saved

    def restore_prior_state(self, checkpoint):
        if checkpoint.get('prior_residual_settings') != self.prior_settings():
            raise ValueError('Prior-residual checkpoint settings mismatch')
        if self.prior_controller is not None:
            self.prior_controller.load_state_dict(checkpoint['prior_residual_controller'])
            # Restore the vector used by the saved model, not a new vector
            # computed from the subsequently updated pi_u.
            self._sync_prior_residual()
        elif 'prior_residual_controller' in checkpoint:
            raise ValueError('Disabled estimator cannot load an active estimator')

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.restore_prior_state(checkpoint)
        return checkpoint

    def _record_prior(self, event, updated=None):
        if self.distributed and self.rank != 0:
            return
        record = dict(event=event, epoch=self.epoch, iteration=self.it,
            **self.prior_settings(), tau_used=4., estimator_updated=updated,
            pi_l=self.p_hat_lb.detach().cpu().tolist(), adjustment=self.tail_logit_adjustment().detach().cpu().tolist())
        if self.prior_controller is not None:
            c = self.prior_controller
            record.update(pi_used=c.pi_used.cpu().tolist(), pi_next=c.pi_u.cpu().tolist(),
                pi_raw=c.pi_raw.cpu().tolist(), residual=c.applied_residual.cpu().tolist(),
                high_confidence_counts=c.last_counts.cpu().tolist(), effective_rank=c.last_rank.cpu().tolist(),
                valid_batches=int(c.total_valid_batches))
        folder = Path(self.save_dir) / self.save_name
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'prior_residual_history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def train_step(self, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb):
        if self.prior_controller is None:
            # Includes lambda=0: no observer/hooks, extra tensor ops or RNG use.
            return super().train_step(x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb)
        if not self.use_cat:
            raise ValueError('Prior-residual experiments require use_cat=True')
        self.prior_controller.begin_step(self.epoch)
        self._sync_prior_residual()
        if self._prior_record_epoch != self.epoch:
            self._prior_record_epoch = self.epoch
            self._record_prior('epoch')
        captured = []
        n, u = len(x_lb), len(x_ulb_w)
        def capture(module, inputs, outputs):
            captured.append((outputs['aux_logits2'][n:n+u].detach(), outputs['feat'][n:n+u].detach()))
        hook = self.model.register_forward_hook(capture)
        try:
            result, logs = super().train_step(x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError('Exactly one aligned three-view forward required')
        updated = self.prior_controller.observe(*captured[0])
        logs['train/tau_T'] = 4.
        logs['train/prior_residual_max_abs'] = float(self.prior_controller.applied_residual.abs().max())
        logs['train/prior_estimator_updated'] = float(updated)
        if self.it % self.num_log_iter == 0:
            self._record_prior('batch', updated)
        return result, logs

    @staticmethod
    def get_argument():
        return MetaExpert.get_argument() + [
            SSL_Argument('--prior_residual_enabled', str2bool, False),
            SSL_Argument('--prior_residual_estimator', str, 'histogram'),
            SSL_Argument('--prior_residual_lambda', float, .5),
            SSL_Argument('--prior_residual_bound', float, 2.),
            SSL_Argument('--prior_residual_eps', float, 1e-4),
        ]
