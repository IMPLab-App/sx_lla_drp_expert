"""Opt-in DDDE/tau extension of the exact fixed-tau BACN baseline.

The disabled-DDDE branch delegates directly to the original MetaExpert. The
original backbone, fusion network, initialization and losses are not replaced
with those of the older SegFormer experiment.
"""
import json
import os

import torch
from torch import nn

from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.utils import IMB_ALGORITHMS
from .cola_ddde import TwoKLDDDE
from .metaexpert import MetaExpert


class StaticExpertAverage(nn.Module):
    """Remove the entire aggregation branch, reusing the initialized backbone."""

    def __init__(self, original):
        super().__init__()
        self.backbone = original.backbone
        self.channels = original.channels
        self.hat, self.tau1, self.tau3 = original.hat, original.tau1, original.tau3

    def forward(self, x, **kwargs):
        outputs = self.backbone(x, **kwargs)
        head, tail = outputs['logits'], outputs['aux_logits2']
        outputs['w1'] = head.new_full((len(head), 1), 0.5)
        outputs['w3'] = head.new_full((len(head), 1), 0.5)
        outputs['fuse_logit_l'] = (head + tail) * 0.5
        return outputs

    def group_matcher(self, coarse=False):
        return self.backbone.group_matcher(coarse, prefix='backbone.')


@IMB_ALGORITHMS.register('metaexpert_bacn')
class BacnMetaExpert(MetaExpert):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.ablation_no_ddde_kl = args.ablation_no_ddde_kl
        self.ablation_no_dea_aggregator = args.ablation_no_dea_aggregator
        self.tau_max = float(args.la_tau_lb3)
        self.cola = None if self.ablation_no_ddde_kl else TwoKLDDDE(
            self.p_hat_lb, self.tau_max, args.est_epoch, args.ema_u)
        if self.ablation_no_dea_aggregator:
            self.model = StaticExpertAverage(self.model)
            self.ema_model = StaticExpertAverage(self.ema_model)

    def _sync_tail_tau(self):
        self.tau_lb3 = self.tau_max if self.cola is None else float(self.cola.tau_dyn)
        for model in (self.model, self.ema_model):
            module = model.module if hasattr(model, 'module') else model
            module.tau3.fill_(self.tau_lb3)

    def restore_ddde_state(self, checkpoint):
        if self.cola is None:
            if 'cola_two_kl' in checkpoint:
                raise ValueError('Fixed-tau ablation cannot load a DDDE checkpoint')
            return
        expected = dict(tau_max=self.tau_max, est_epoch=self.est_epoch,
                        ema_u=self.ema_u, no_dea=self.ablation_no_dea_aggregator)
        if checkpoint.get('bacn_module_settings') != expected:
            raise ValueError('Checkpoint DDDE/DEA settings do not match this configuration')
        self.cola.load_state_dict(checkpoint['cola_two_kl'])
        self._sync_tail_tau()

    def get_save_dict(self):
        checkpoint = super().get_save_dict()
        if self.cola is not None:
            checkpoint['cola_two_kl'] = self.cola.state_dict()
            checkpoint['bacn_module_settings'] = dict(
                tau_max=self.tau_max, est_epoch=self.est_epoch,
                ema_u=self.ema_u, no_dea=self.ablation_no_dea_aggregator)
        return checkpoint

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.restore_ddde_state(checkpoint)
        return checkpoint

    def _record_ddde(self, event):
        if self.cola is None or (self.distributed and self.rank != 0):
            return
        folder = os.path.join(self.save_dir, self.save_name)
        os.makedirs(folder, exist_ok=True)
        record = dict(event=event, epoch=self.epoch, iteration=self.it, **self.cola.snapshot())
        with open(os.path.join(folder, 'cola_history.jsonl'), 'a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def _expert_loss(self, outputs, prefix, num_lb, labels, tau):
        """The original expert's supervised and strong-view pseudo-label losses."""
        logits = outputs['logits' if prefix == '' else 'aux_logits2']
        logits_h = outputs['logitsH' if prefix == '' else 'aux_logitsH2']
        logits_t = outputs['logitsT' if prefix == '' else 'aux_logitsT2']
        weak, _ = logits[num_lb:].chunk(2)
        _, strong_h = logits_h[num_lb:].chunk(2)
        _, strong_t = logits_t[num_lb:].chunk(2)
        supervised = self.ce_loss(logits[:num_lb] + tau * self.p_hat_lb.log(), labels, reduction='mean')
        probabilities = self.compute_prob(weak.detach())
        mask = probabilities.amax(dim=-1).ge(self.p_cutoff)
        pseudo = probabilities.argmax(dim=-1)
        # Original H path covers every class; T path covers classes >= cut1.
        head_mask = mask.float()
        tail_mask = (pseudo >= self.cut1).float() * mask.float()
        unsupervised = (self.ce_loss(strong_h, pseudo, reduction='none') * head_mask).sum()
        unsupervised += (self.ce_loss(strong_t, pseudo, reduction='none') * tail_mask).sum()
        unsupervised /= head_mask.sum() + tail_mask.sum() + 1e-12
        return supervised, unsupervised, mask

    def train_step(self, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb):
        if self.cola is None and not self.ablation_no_dea_aggregator:
            # No added forward hooks, RNG use, logging or tensor operations.
            return super().train_step(x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb)
        if not self.use_cat:
            raise ValueError('BACN DDDE experiments require use_cat=True')
        if self.cola is not None and self.cola.stats_epoch.item() != self.epoch:
            self.cola.begin_epoch(self.epoch)
            self._sync_tail_tau()
            self._record_ddde('epoch')
        num_lb, num_ulb = len(x_lb), len(x_ulb_w)
        if self.ablation_no_dea_aggregator:
            labels = y_lb_noised if self.args.noise_ratio > 0 else y_lb
            with self.amp_cm():
                outputs = self.model(torch.cat((x_lb, x_ulb_w, x_ulb_s)))
                sup1, unsup1, mask = self._expert_loss(outputs, '', num_lb, labels, self.tau_lb1)
                sup3, unsup3, _ = self._expert_loss(outputs, 'tail', num_lb, labels, self.tau_lb3)
                loss = (sup1 + sup3) + (self.lambda_u * unsup1 + self.lambda_u * unsup3)
            result = self.process_out_dict(loss=loss, feat={})
            logs = self.process_log_dict(sup_loss=sup1.item(), unsup_loss=unsup1.item(),
                                         total_loss=(sup1+self.lambda_u*unsup1).item(),
                                         total_fuse_loss=0.0, util_ratio=mask.float().mean().item())
            tail_logits = outputs['aux_logits2'][num_lb:num_lb+num_ulb].detach()
            weak_features = outputs['feat'][num_lb:num_lb+num_ulb].detach()
        else:
            captured = []
            def capture(module, inputs, outputs):
                captured.append((outputs['aux_logits2'][num_lb:num_lb+num_ulb].detach(),
                                 outputs['feat'][num_lb:num_lb+num_ulb].detach()))
            hook = self.model.register_forward_hook(capture)
            try:
                result, logs = super().train_step(x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise RuntimeError('DDDE requires one aligned three-view forward pass')
            tail_logits, weak_features = captured[0]
        if self.cola is not None:
            kl = self.cola.observe(tail_logits, weak_features)
            logs['train/tau_dyn'] = self.tau_lb3
            if kl is not None:
                logs['train/KL_uni'], logs['train/KL_rev'] = map(float, kl)
            if self.it % self.num_log_iter == 0:
                self._record_ddde('batch')
        logs['train/tau_T'] = self.tau_lb3
        return result, logs

    @staticmethod
    def get_argument():
        return MetaExpert.get_argument() + [
            SSL_Argument('--ablation_no_ddde_kl', str2bool, False),
            SSL_Argument('--ablation_no_dea_aggregator', str2bool, False),
        ]
