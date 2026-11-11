"""Opt-in r12 replay diagnostics and validation-selected KL map temperature."""
import json
from pathlib import Path
from unittest.mock import patch

import torch

from semilearn.algorithms.utils import SSL_Argument
from semilearn.core.utils import IMB_ALGORITHMS
from . import cola_ddde, five_variants
from .cola_ddde import TwoKLDDDE, normalize_distribution, symmetric_kl
from .five_variants import DistributionController, FiveVariantMetaExpert, l1_distance


class AuditedController(TwoKLDDDE):
    def __init__(self, pi_l, version, temperature, destination, **kwargs):
        super().__init__(pi_l, **kwargs)
        if version not in ('V2', 'V5') or temperature <= 0:
            raise ValueError('Expected V2/V5 and a positive map temperature')
        if version == 'V2' and temperature != 1:
            raise ValueError('V2 is the unchanged L1 control')
        self.version, self.temperature = version, float(temperature)
        self.destination = Path(destination)
        self._epoch = None
        self._summary = None

    def flush_epoch(self):
        if self._summary is None:
            return
        row = dict(self._summary)
        count = row['valid_batches']
        row.update(pi_ema_end=self.pi_u.cpu().tolist(), pi_raw_last=self.pi_ddde.cpu().tolist())
        for key in ('rank_sum', 'pi_raw_sum', 'pi_ema_sum'):
            denominator = row['rank_batches'] if key == 'rank_sum' else count
            row[key.replace('_sum', '_mean')] = ([x / denominator for x in row[key]] if denominator else None)
        for key in ('KL_uni', 'KL_rev', 'L1_uni', 'L1_rev'):
            row[key + '_mean'] = row.pop(key + '_sum') / count if count else None
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        self._summary = None

    def snapshot(self):
        result = super().snapshot()
        if self.version == 'V2':
            result['L1_uni'] = result.pop('KL_uni')
            result['L1_rev'] = result.pop('KL_rev')
        return result

    @torch.no_grad()
    def begin_epoch(self, epoch):
        if self.stats_epoch.item() == epoch:
            return
        self.flush_epoch()
        if self.version == 'V2':
            DistributionController.begin_epoch(self, epoch)
        else:
            super().begin_epoch(epoch)
            if epoch >= self.est_epoch and self._epoch is not None and self._previous_valid:
                # T=1 executes the original mapping exactly, including dtype.
                if self.temperature != 1:
                    self.tau_dyn.copy_(self.tau_max * torch.softmax(-self.last_kl / self.temperature, dim=0)[1])
        self._epoch = epoch
        zeros = lambda: [0.] * self.pi_l.numel()
        self._summary = dict(epoch=epoch, version=self.version, map_temperature=self.temperature,
            tau_used=float(self.tau_dyn), collecting=epoch >= self.est_epoch,
            valid_batches=0, observed_batches=0, rank_batches=0, high_confidence_pseudo_counts=[0] * self.pi_l.numel(),
            absent_pseudo_class_batches=[0] * self.pi_l.numel(), rank_sum=zeros(),
            pi_raw_sum=zeros(), pi_ema_sum=zeros(),
            KL_uni_sum=0., KL_rev_sum=0., L1_uni_sum=0., L1_rev_sum=0.,
            map_distance_window=self.last_kl.cpu().tolist(),
            counts_scope='training sample occurrences, including sampler replacement and repeated epochs',
            rank_scope='mean/sum of the original per-minibatch class effective ranks; not rank of concatenated features')
        self._previous_valid = False

    @torch.no_grad()
    def observe(self, tail_raw_logits, shared_features):
        captured_ranks, captured_counts = [], []
        module = five_variants if self.version == 'V2' else cola_ddde
        original = module.effective_rank
        def record_rank(features):
            rank = original(features)
            captured_counts.append(len(features))
            captured_ranks.append(float(rank))
            return rank
        with patch.object(module, 'effective_rank', record_rank):
            distances = (DistributionController.observe(self, tail_raw_logits, shared_features)
                         if self.version == 'V2' else super().observe(tail_raw_logits, shared_features))
        if self._summary is None:
            return distances
        row = self._summary
        row['observed_batches'] += 1
        if captured_ranks:
            assert len(captured_ranks) == self.pi_l.numel()
            row['rank_batches'] += 1
            for key, values in [('high_confidence_pseudo_counts', captured_counts),
                            ('absent_pseudo_class_batches', [int(n == 0) for n in captured_counts]),
                            ('rank_sum', captured_ranks)]:
                row[key] = [a + b for a, b in zip(row[key], values)]
        if distances is None:
            return distances
        self._previous_valid = True
        row['valid_batches'] += 1
        for key, values in [('pi_raw_sum', self.pi_ddde.cpu().tolist()),
                            ('pi_ema_sum', self.pi_u.cpu().tolist())]:
            row[key] = [a + b for a, b in zip(row[key], values)]
        for prefix, distance in [('KL', symmetric_kl), ('L1', l1_distance)]:
            for name, reference in [('uni', self.pi_uni), ('rev', self.pi_rev)]:
                row[f'{prefix}_{name}_sum'] += float(distance(self.pi_u, reference))
        return distances


@IMB_ALGORITHMS.register('metaexpert_temperature_audit')
class TemperatureAuditMetaExpert(FiveVariantMetaExpert):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.map_temperature = float(args.map_temperature)
        original = self.cola.state_dict()
        self.cola = AuditedController(self.p_hat_lb, self.five_variant, self.map_temperature,
            Path(self.save_dir) / self.save_name / 'epoch_diagnostics.jsonl',
            tau_max=self.tau_max, est_epoch=self.est_epoch, ema_u=self.ema_u)
        self.cola.load_state_dict(original)
        self._sync_tail_tau()
        # Candidate runs must never trigger EvaluationHook.after_run's test.
        self.loader_dict.pop('test', None)

    def variant_settings(self):
        return dict(super().variant_settings(), map_temperature=self.map_temperature,
                    diagnostics_schema=1, candidate_test_disabled=True)

    def train(self):
        result = super().train()
        self.cola.flush_epoch()
        return result

    @staticmethod
    def get_argument():
        return FiveVariantMetaExpert.get_argument() + [SSL_Argument('--map_temperature', float, 1.)]
