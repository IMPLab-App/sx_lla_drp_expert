"""Independent estimator / distance / static-tau controls; original fusion kept."""
import json
from pathlib import Path

import torch

from semilearn.algorithms.utils import SSL_Argument
from semilearn.core.utils import IMB_ALGORITHMS
from .bacn_ablation import BacnMetaExpert
from .cola_ddde import TwoKLDDDE, effective_rank, normalize_distribution, symmetric_kl


SPECS = {
    'V1': dict(estimator='none', distance='none', dynamic=False, fixed_tau=4.),
    'V2': dict(estimator='effective_rank', distance='l1', dynamic=True, fixed_tau=None),
    'V3': dict(estimator='histogram', distance='symmetric_kl', dynamic=True, fixed_tau=None),
    'V4': dict(estimator='none', distance='none', dynamic=False, fixed_tau=2.),
    'V5': dict(estimator='effective_rank', distance='symmetric_kl', dynamic=True, fixed_tau=None),
    'S0': dict(estimator='none', distance='none', dynamic=False, fixed_tau=0.),
    'S1': dict(estimator='none', distance='none', dynamic=False, fixed_tau=1.),
    'S3': dict(estimator='none', distance='none', dynamic=False, fixed_tau=3.),
}


def l1_distance(p, q):
    return (normalize_distribution(p) - normalize_distribution(q)).abs().sum()


class DistributionController(TwoKLDDDE):
    """V2 uses effective-rank/L1; V3 uses pseudo-label histogram/symmetric KL."""

    def __init__(self, pi_l, version, tau_max=4., est_epoch=18, ema_u=.9):
        if version not in ('V2', 'V3'):
            raise ValueError('Only V2/V3 need the replacement controller')
        super().__init__(pi_l, tau_max, est_epoch, ema_u)
        self.version = version

    @torch.no_grad()
    def begin_epoch(self, epoch):
        if epoch == self.stats_epoch.item():
            return
        if epoch < self.est_epoch:
            self.tau_dyn.fill_(self.tau_max)
        elif self.valid_batches.item():
            self.last_kl.copy_(self.kl_sum / self.valid_batches)
            # The same two-reference softmax map; V2 contains no KL evaluation.
            self.tau_dyn.copy_((self.tau_max * torch.softmax(-self.last_kl, dim=0)[1]).clamp(0, self.tau_max))
        self.kl_sum.zero_()
        self.valid_batches.zero_()
        self.stats_epoch.fill_(epoch)

    @torch.no_grad()
    def observe(self, tail_raw_logits, shared_features):
        if self.stats_epoch.item() < self.est_epoch:
            return None
        if tail_raw_logits.ndim != 2 or shared_features.ndim != 2 or len(tail_raw_logits) != len(shared_features):
            raise ValueError('Aligned weak logits/features required')
        valid = torch.isfinite(tail_raw_logits).all(1) & torch.isfinite(shared_features).all(1)
        logits = tail_raw_logits.detach().float()[valid]
        features = shared_features.detach()[valid]
        if not len(logits):
            return None
        confidence, labels = logits.softmax(-1).max(-1)
        select = confidence.ge(.95)
        if not select.any():
            return None
        features, labels = features[select], labels[select]
        if self.version == 'V2':
            estimate = torch.stack([effective_rank(features[labels == k]) for k in range(self.pi_l.numel())])
            distance = l1_distance
        else:
            estimate = torch.bincount(labels, minlength=self.pi_l.numel()).to(self.pi_u)
            distance = symmetric_kl
        if not torch.isfinite(estimate).all() or estimate.sum() <= 0:
            return None
        self.pi_ddde.copy_(estimate / estimate.sum())
        self.pi_u.copy_(normalize_distribution(self.ema_u*self.pi_u + (1-self.ema_u)*self.pi_ddde))
        distances = torch.stack((distance(self.pi_u, self.pi_uni), distance(self.pi_u, self.pi_rev)))
        if not torch.isfinite(distances).all():
            return None
        self.kl_sum.add_(distances)
        self.valid_batches.add_(1)
        self.total_valid_batches.add_(1)
        return distances

    def snapshot(self):
        result = super().snapshot()
        if self.version == 'V2':
            result['L1_uni'] = result.pop('KL_uni')
            result['L1_rev'] = result.pop('KL_rev')
        else:
            result['pi_histogram'] = result.pop('pi_DDDE')
        result['estimator'] = SPECS[self.version]['estimator']
        result['distance'] = SPECS[self.version]['distance']
        return result


@IMB_ALGORITHMS.register('metaexpert_v1v5')
class FiveVariantMetaExpert(BacnMetaExpert):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.five_variant = args.five_variant.upper()
        if self.five_variant not in SPECS:
            raise ValueError('Unknown five-variant version')
        self.spec = dict(SPECS[self.five_variant])
        if self.spec['fixed_tau'] is not None and args.five_fixed_tau != self.spec['fixed_tau']:
            raise ValueError('Static tau must match the declared version')
        args.ablation_no_ddde_kl = not self.spec['dynamic']
        args.ablation_no_dea_aggregator = False
        super().__init__(args, net_builder, tb_log, logger)
        if self.five_variant in ('V2', 'V3'):
            self.cola = DistributionController(self.p_hat_lb, self.five_variant,
                self.tau_max, args.est_epoch, args.ema_u)
        # V5 retains the exact original TwoKLDDDE object and all its operations.
        self._fixed_epoch = None
        self._sync_tail_tau()

    def _sync_tail_tau(self):
        self.tau_lb3 = (float(self.cola.tau_dyn) if self.spec['dynamic']
                        else self.spec['fixed_tau'])
        for model in (self.model, self.ema_model):
            module = model.module if hasattr(model, 'module') else model
            module.tau3.fill_(self.tau_lb3)

    def variant_settings(self):
        return dict(version=self.five_variant, **self.spec, tau_max=self.tau_max,
                    est_epoch=self.est_epoch, ema_u=self.ema_u, dea=True)

    def get_save_dict(self):
        saved = super().get_save_dict()
        saved['five_variant_settings'] = self.variant_settings()
        return saved

    def restore_ddde_state(self, checkpoint):
        if checkpoint.get('five_variant_settings') != self.variant_settings():
            raise ValueError('Checkpoint version/estimator/distance/tau settings mismatch')
        super().restore_ddde_state(checkpoint)
        self._sync_tail_tau()

    def _record_variant(self, event, logs=None):
        if self.distributed and self.rank != 0:
            return
        record = dict(event=event, epoch=self.epoch, iteration=self.it,
                      **self.variant_settings(), tau_used=self.tau_lb3,
                      KL_uni=None, KL_rev=None, L1_uni=None, L1_rev=None,
                      KL_uni_window=None, KL_rev_window=None,
                      L1_uni_window=None, L1_rev_window=None,
                      pi_estimated=None, pi_raw=None, valid_batches=0)
        if self.cola is not None:
            name = 'L1' if self.five_variant == 'V2' else 'KL'
            record[name+'_uni_window'] = float(self.cola.last_kl[0])
            record[name+'_rev_window'] = float(self.cola.last_kl[1])
            if logs is not None:
                record[name+'_uni'] = logs.get('train/'+name+'_uni')
                record[name+'_rev'] = logs.get('train/'+name+'_rev')
            record.update(pi_estimated=self.cola.pi_u.cpu().tolist(),
                          pi_raw=self.cola.pi_ddde.cpu().tolist(),
                          valid_batches=int(self.cola.valid_batches))
        folder = Path(self.save_dir)/self.save_name
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'variant_history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')

    def _record_ddde(self, event):
        super()._record_ddde(event)
        if event == 'epoch':
            self._record_variant(event)

    def train_step(self, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb):
        if self.cola is None and self._fixed_epoch != self.epoch:
            self._fixed_epoch = self.epoch
            self._record_variant('epoch')
        result, logs = super().train_step(x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb)
        if self.five_variant == 'V2':
            for ref in ('uni', 'rev'):
                key = 'train/KL_'+ref
                if key in logs:
                    logs['train/L1_'+ref] = logs.pop(key)
        logs['train/tau_T'] = self.tau_lb3
        if self.cola is None:
            logs['train/tau_fixed'] = self.tau_lb3
        if self.it % self.num_log_iter == 0:
            self._record_variant('batch', logs)
        return result, logs

    @staticmethod
    def get_argument():
        return BacnMetaExpert.get_argument() + [
            SSL_Argument('--five_variant', str, 'V5'),
            SSL_Argument('--five_fixed_tau', float, 4.),
        ]
