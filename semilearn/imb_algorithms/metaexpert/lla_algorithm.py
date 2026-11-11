"""Opt-in LLA-E3 protocol. Legacy source files and registry entries are unchanged."""
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.hooks import Hook
from semilearn.core.utils import IMB_ALGORITHMS
from .metaexpert import MetaExpertNet
from .bacn_ablation import BacnMetaExpert, StaticExpertAverage
from .lla_e3 import LLAE3Controller, assert_parameter_partition
from .lla_runtime import (IsolatedUpdateHook, MacroF1SelectionHook, branch_metrics,
                          capture_rng, restore_rng, ProtocolCheckpointHook)


class LLAExpertNet(MetaExpertNet):
    @classmethod
    def from_original(cls, original, controller):
        # Reuse every initialized module without consuming any RNG or changing
        # parameter names. Old network is released after conversion.
        result = cls.__new__(cls)
        result.__dict__ = original.__dict__.copy()
        result._modules = original._modules.copy()
        result.lla = controller
        return result

    def tail_logit_adjustment(self):
        return self.lla.adjustment(detach=True)


class LLAStaticExpertAverage(StaticExpertAverage):
    def __init__(self, original, controller):
        super().__init__(original)
        self.lla = controller

    def forward(self, x, **kwargs):
        out = self.backbone(x, **kwargs)
        head, tail = out['logits'], out['aux_logits2']
        out['w1'] = head.new_full((len(head), 1), .5)
        out['w3'] = head.new_full((len(head), 1), .5)
        out['fuse_logit_l'] = .5 * head + .5 * self.lla.adjust(tail)
        return out


def prediction_paths(outputs, controller=None):
    e1, e3 = outputs['logits'], outputs['aux_logits2']
    adjusted = e3 if controller is None else controller.adjust(e3)
    raw = outputs['w1'] * e1 + outputs['w3'] * e3
    main = raw if controller is None else outputs['w1'] * e1 + outputs['w3'] * adjusted
    return dict(e1=e1, e3_raw=e3, e3_lla=adjusted, fusion_raw=raw, fusion_main=main)


@IMB_ALGORITHMS.register('metaexpert_lla_e3')
class LLAE3MetaExpert(BacnMetaExpert):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.e3_adjust_mode = args.e3_adjust_mode
        if self.e3_adjust_mode not in ('fixed', 'ddde_kl', 'lla_hard'):
            raise ValueError('Mutually exclusive e3_adjust_mode required: fixed/ddde_kl/lla_hard')
        if args.ablation_no_ddde_kl != (self.e3_adjust_mode != 'ddde_kl'):
            raise ValueError('Conflicting DDDE flag and e3_adjust_mode')
        if args.la_tau_lb3 != 4.0:
            raise ValueError('This preregistered replacement protocol requires legacy tau3=4')
        if args.p_cutoff != .95 or not args.hard_label or not args.use_cat:
            raise ValueError('LLA-E3 Hard protocol requires hard_label, cutoff=.95, use_cat=True')
        if getattr(args, 'gradient_accumulation_steps', 1) != 1:
            raise ValueError('Legacy MetaExpert has one optimizer step per batch; accumulation must be 1')
        for flag in ('cb_loss', 'use_cb', 'rp_enabled', 'efcc', 'shared_bn'):
            if getattr(args, flag, False):
                raise ValueError('Incompatible additional module: ' + flag)
        self.lambda_optimizer = self.lambda_scheduler = None
        self.best_eval_f1, self.best_epoch = -1.0, -1
        self.optimizer_steps = 0
        self._resume_rng = self._epoch_rng = None
        self._checkpoint_during_step = False
        self._pending_pseudo = None
        super().__init__(args, net_builder, tb_log, logger)
        if self.e3_adjust_mode == 'lla_hard':
            if self.cola is not None:
                raise RuntimeError('LLA must never construct DDDE')
            for name in ('model', 'ema_model'):
                old = getattr(self, name)
                controller = LLAE3Controller(self.num_classes, args.lla_init, args.lla_epsilon,
                                             args.lla_q_stop_gradient).to(self.p_hat_lb.device)
                new = (LLAStaticExpertAverage(old, controller) if self.ablation_no_dea_aggregator
                       else LLAExpertNet.from_original(old, controller))
                setattr(self, name, new)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.model._ddp_params_and_buffers_to_ignore = {'lla.lambda_logits'}
        self._sync_tail_tau()

    @property
    def module(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def set_hooks(self):
        super().set_hooks()
        replacements = {'ParamUpdateHook': IsolatedUpdateHook, 'EvaluationHook': MacroF1SelectionHook,
                        'CheckpointHook': ProtocolCheckpointHook}
        selected = []
        for h in self._hooks:
            if h.name == 'TimerHook':
                continue  # CUDA-only timing hook is not needed for the protocol.
            if h.name in replacements:
                new = replacements[h.name]()
                new.name, new.priority = h.name, h.priority
                h = new
            selected.append(h)
        self._hooks = selected
        self.hooks_dict = OrderedDict((h.name, h) for h in selected)

    def set_optimizer(self):
        optimizer, scheduler = super().set_optimizer()
        # Early AlgorithmBase construction precedes attaching the controller.
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if not any(
                p is q for n, q in self.model.named_parameters() if n.endswith('lla.lambda_logits'))]
        return optimizer, scheduler

    def finalize_model_setup(self):
        self.optimizer, self.scheduler = self.set_optimizer()
        if self.e3_adjust_mode == 'lla_hard':
            if self.args.lla_optimizer != 'Adam' or self.args.lla_scheduler != 'constant':
                raise ValueError('Only preregistered Adam / constant lambda schedule is implemented')
            self.lambda_optimizer = torch.optim.Adam([self.module.lla.lambda_logits],
                lr=self.args.lla_lr, weight_decay=self.args.lla_weight_decay,
                betas=tuple(self.args.lla_betas), eps=self.args.lla_optimizer_epsilon)
            # DDP must ignore lambda in its forward reducer: lambda has an
            # independent detached-logit objective and explicit gradient allreduce.
        self.assert_optimizer_coverage()

    def assert_optimizer_coverage(self):
        return assert_parameter_partition(self.model, self.optimizer, self.lambda_optimizer)

    def _sync_tail_tau(self):
        if self.e3_adjust_mode == 'lla_hard':
            self.tau_lb3 = 0.0
            for model in (self.model, self.ema_model):
                module = model.module if hasattr(model, 'module') else model
                module.tau3.fill_(0.0)
        else:
            super()._sync_tail_tau()

    def _lla_expert_loss(self, outputs, num_lb, labels):
        logits = outputs['aux_logits2']
        weak, _ = logits[num_lb:].chunk(2)
        _, strong_h = outputs['aux_logitsH2'][num_lb:].chunk(2)
        _, strong_t = outputs['aux_logitsT2'][num_lb:].chunk(2)
        supervised = self.ce_loss(logits[:num_lb], labels, reduction='mean')
        with torch.no_grad():
            probs = self.module.lla.adjust(weak.detach()).softmax(-1)
            confidence, pseudo = probs.max(-1)
            mask = confidence.ge(self.p_cutoff)
        hm = mask.float()
        tm = (pseudo >= self.cut1).float() * hm
        unsupervised = (self.ce_loss(strong_h, pseudo, reduction='none') * hm).sum()
        unsupervised += (self.ce_loss(strong_t, pseudo, reduction='none') * tm).sum()
        unsupervised /= hm.sum() + tm.sum() + 1e-12
        counts = torch.bincount(pseudo[mask], minlength=self.num_classes)
        confidence_sum = confidence.new_zeros(self.num_classes).scatter_add_(0, pseudo[mask], confidence[mask])
        self._pending_pseudo = (counts.detach(), confidence_sum.detach())
        return supervised, unsupervised, mask

    def _aggregation_loss(self, outputs, n, labels):
        # Same original fusion losses, including original unmasked CE and
        # weight_kl=[.5,.5]. No new loss or change to DEA's second softmax.
        if self.ablation_no_dea_aggregator or self.epoch <= self.est_epoch + 1:
            return outputs['logits'].new_zeros(()), outputs['logits'].new_zeros(())
        sup1 = F.cross_entropy(outputs['fuse_logit_l'][:n], labels, self.weight1)
        weak1, strong1 = outputs['fuse_logit_HMT_c_1'][n:].chunk(2)
        weak3, strong3 = outputs['fuse_logit_HMT_c_3'][n:].chunk(2)
        pseudo1 = self.compute_prob(weak1.detach()).argmax(-1)
        pseudo3 = self.compute_prob(weak3.detach()).argmax(-1)
        unsup1 = (self.weight_kl[0] * F.cross_entropy(strong1, pseudo1) +
                  self.weight_kl[1] * F.cross_entropy(strong3, pseudo3)) if len(strong1) else sup1.new_zeros(())
        route = outputs['fuse_w_logit']
        route_labels = torch.where(labels < self.cut1, torch.zeros_like(labels), torch.ones_like(labels))
        sup2 = F.cross_entropy(route[:n], route_labels, self.weight2)
        weak_route, strong_route = route[n:].chunk(2)
        unsup2 = (F.cross_entropy(strong_route, self.compute_prob(weak_route.detach()).argmax(-1))
                  if len(strong_route) else sup2.new_zeros(()))
        return sup1 + sup2, unsup1 + unsup2

    def _pseudo_statistics(self, logits):
        with torch.no_grad():
            confidence, pseudo = self.compute_prob(logits.detach().float()).max(-1)
            selected = confidence.ge(self.p_cutoff)
            count = torch.bincount(pseudo[selected], minlength=self.num_classes)
            total = confidence.new_zeros(self.num_classes).scatter_add_(0, pseudo[selected], confidence[selected])
        return count, total

    def train_step(self, x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb):
        # y_ulb is accepted only for framework compatibility and never read.
        if self.e3_adjust_mode != 'lla_hard':
            n, u = len(x_lb), len(x_ulb_w)
            def capture(module, inputs, outputs):
                self._pending_pseudo = self._pseudo_statistics(outputs['aux_logits2'][n:n+u])
                self._pending_pseudo_e1 = self._pseudo_statistics(outputs['logits'][n:n+u])
            hook = self.model.register_forward_hook(capture)
            try:
                return super().train_step(x_lb, y_lb, y_lb_noised, x_ulb_w, x_ulb_s, y_ulb)
            finally:
                hook.remove()
        labels = y_lb_noised if self.args.noise_ratio > 0 else y_lb
        n, u = len(x_lb), len(x_ulb_w)
        self._lambda_used = self.module.lla.lambda_logits.detach().cpu().tolist()
        if not n or len(x_ulb_s) != u:
            raise ValueError('Nonempty labeled and aligned weak/strong batches required')
        with self.amp_cm():
            outputs = self.model(torch.cat((x_lb, x_ulb_w, x_ulb_s)))
            self._pending_pseudo_e1 = self._pseudo_statistics(outputs['logits'][n:n+u])
            sup1, unsup1, mask1 = self._expert_loss(outputs, '', n, labels, self.tau_lb1)
            sup3, unsup3, mask3 = self._lla_expert_loss(outputs, n, labels)
            sf, uf = self._aggregation_loss(outputs, n, labels)
            base = (sup1 + sup3 + sf) + (self.lambda_u * unsup1 + self.lambda_u * unsup3 + self.lambda_u * uf)
        # Explicitly leave autocast for the small controller objective.
        with torch.autocast(device_type=outputs['logits'].device.type, enabled=False):
            ll, lu, lc = self.module.lla.class_average_loss(outputs['aux_logits2'][:n], labels,
                outputs['aux_logits2'][n:n+u], outputs['aux_logits2'][n+u:])
        logs = self.process_log_dict(sup_loss=sup1.item(), unsup_loss=unsup1.item(),
            sup_loss_e3=sup3.item(), unsup_loss_e3=unsup3.item(), total_loss=base.item(),
            total_fuse_loss=(sf + self.lambda_u * uf).item(),
            util_ratio=mask1.float().mean().item() if u else 0.,
            util_ratio_e3=mask3.float().mean().item() if u else 0.,
            L_l=ll.item(), L_u=lu.item(), L_lambda=lc.item())
        return self.process_out_dict(loss=base, lambda_loss=lc, feat={}), logs

    def write_json(self, name, data):
        if self.distributed and self.rank != 0:
            return
        path = Path(self.save_dir) / self.save_name / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def record_trajectory(self):
        if self.distributed and self.rank != 0:
            return
        record = dict(epoch=self.epoch, iteration=self.it, optimizer_steps=self.optimizer_steps,
                      mode=self.e3_adjust_mode, dea=not self.ablation_no_dea_aggregator,
                      losses={k: v for k, v in self.log_dict.items() if k.startswith('train/') and k not in ('train/w1', 'train/w3')})
        # Legacy weight diagnostics use NaN for classes absent from the batch
        # window. Represent these missing diagnostics as null, never as a loss.
        for k in ('train/w1', 'train/w3'):
            if k in self.log_dict:
                record[k] = [float(v) if np.isfinite(v) else None for v in self.log_dict[k]]
        if self.e3_adjust_mode == 'lla_hard':
            value = self.module.lla.lambda_logits.detach().float().cpu()
            record.update(lambda_used_for_batch=self._lambda_used,
                          lambda_logits=value.tolist(), lambda_softmax=value.softmax(0).tolist(),
                          lambda_lr=self.lambda_optimizer.param_groups[0]['lr'])
        counts, confidence = self._pending_pseudo
        record.update(pseudo_accepted=counts.cpu().tolist(), pseudo_confidence_sum=confidence.cpu().tolist(),
            pseudo_confidence_mean=[float(s / c) if c else None for s, c in zip(confidence.cpu(), counts.cpu())])
        counts1, confidence1 = self._pending_pseudo_e1
        record['e1_pseudo'] = dict(accepted=counts1.cpu().tolist(), confidence_sum=confidence1.cpu().tolist())
        path = Path(self.save_dir) / self.save_name / 'training_history.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        if eval_dest == 'test' and not getattr(self, '_final_report_authorized', False):
            raise RuntimeError('Test metrics require the separate locked-model final report command')
        training = self.model.training
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
        collected, truth, weights = {}, [], []
        try:
            with torch.no_grad():
                device = next(self.model.parameters()).device
                controller = self.module.lla if self.e3_adjust_mode == 'lla_hard' else None
                for data in self.loader_dict[eval_dest]:
                    outputs = self.model(data['x_lb'].to(device))
                    for name, value in prediction_paths(outputs, controller).items():
                        collected.setdefault(name, []).append(value.float().cpu())
                    truth.append(data['y_lb'].cpu())
                    weights.append(torch.cat((outputs['w1'], outputs['w3']), dim=1).cpu())
            y = torch.cat(truth).numpy()
            logits = {k: torch.cat(v) for k, v in collected.items()}
            predictions = {k: v.argmax(-1).numpy() for k, v in logits.items()}
            # Tail definition is locked by labeled counts, independent of U/test.
            tail = list(self.args.lla_tail_class_ids)
            metrics = {k: branch_metrics(y, v, self.num_classes, tail) for k, v in predictions.items()}
            e1ok, e3ok = predictions['e1'] == y, predictions['e3_lla'] == y
            e3rawok = predictions['e3_raw'] == y
            finalok = predictions['fusion_main'] == y
            def complement(tailok):
                only = ~e1ok & tailok
                return dict(e1_only=int((e1ok & ~tailok).sum()), e3_only=int(only.sum()),
                    both_correct=int((e1ok & tailok).sum()), both_wrong=int((~e1ok & ~tailok).sum()),
                    fusion_retained_e3_only=int((only & finalok).sum()),
                    fusion_lost_e3_only=int((only & ~finalok).sum()))
            details = dict(epoch=getattr(self, 'epoch', 0), iteration=self.it, branches=metrics,
                mode=self.e3_adjust_mode, main_prediction='fusion_main',
                complementarity_lla=complement(e3ok), complementarity_raw=complement(e3rawok),
                tail_class_ids=tail, class_names=self._get_confusion_matrix_labels(),
                C5_C8_indexing='zero-based class IDs 5 and 8',
                lambda_logits=controller.lambda_logits.detach().cpu().tolist() if controller is not None else None)
            self.write_json(f'metrics/{eval_dest}_iter{self.it:06d}.json', details)
            if not self.distributed or self.rank == 0:
                path = Path(self.save_dir) / self.save_name / 'metrics' / f'{eval_dest}_iter{self.it:06d}.npz'
                np.savez_compressed(path, y_true=y, weights=torch.cat(weights).numpy(),
                    **{k: v.numpy() for k, v in logits.items()})
            main = metrics['fusion_main']
            loss = F.cross_entropy(logits['fusion_main'], torch.from_numpy(y)).item()
            result = {eval_dest+'/top-1-acc': main['accuracy'], eval_dest+'/F1': main['macro_f1'],
                      eval_dest+'/loss': loss, eval_dest+'/tail_F1': main['tail_macro_f1']}
            if return_logits:
                result[eval_dest+'/logits'] = logits['fusion_main'].numpy()
            return result
        finally:
            if self.ema is not None:
                self.ema.restore()
            self.model.train(training)

    def protocol_settings(self):
        return dict(mode=self.e3_adjust_mode, no_dea=self.ablation_no_dea_aggregator,
            classes=self.num_classes, lambda_lr=self.args.lla_lr,
            lambda_optimizer=self.args.lla_optimizer, lambda_weight_decay=self.args.lla_weight_decay,
            lambda_betas=self.args.lla_betas, lambda_optimizer_epsilon=self.args.lla_optimizer_epsilon,
            q_stop_gradient=self.args.lla_q_stop_gradient, lambda_init=self.args.lla_init,
            lambda_epsilon=self.args.lla_epsilon, lambda_scheduler=self.args.lla_scheduler,
            lambda_clip_grad=self.args.lla_clip_grad, selection='validation_macro_f1',
            fixed_split_sha256=getattr(self.args, 'fixed_split_sha256', None),
            data_sha256=getattr(self.args, 'data_sha256', None), tail_ids=self.args.lla_tail_class_ids,
            network=dict(net=self.args.net, optim=self.args.optim, lr=self.args.lr,
                weight_decay=self.args.weight_decay, layer_decay=self.args.layer_decay,
                batch_size=self.args.batch_size, uratio=self.args.uratio, seed=self.args.seed,
                workers=self.args.num_workers, world_size=self.world_size, epochs=self.epochs, iterations=self.num_train_iter,
                warmup=self.args.num_warmup_iter, est_epoch=self.est_epoch, amp=self.use_amp))

    def get_save_dict(self):
        saved = super().get_save_dict()
        completed = self.it + int(self._checkpoint_during_step)
        saved.update(it=completed, epoch=completed // self.num_iter_per_epoch,
            lla_protocol=self.protocol_settings(), best_eval_f1=self.best_eval_f1,
            best_epoch=self.best_epoch, optimizer_steps=self.optimizer_steps,
            rng=capture_rng(), epoch_start_rng=self._epoch_rng,
            rank_resume_states=getattr(self, '_rank_resume_states', None),
            sampler_state={k: dict(seed=v.sampler.seed, epoch=v.sampler.epoch)
                           for k, v in self.loader_dict.items() if k in ('train_lb', 'train_ulb')},
            legacy_metaexpert={k: copy.deepcopy(getattr(self, k)) for k in
                ('record_w1', 'record_w3', 'current_epoch', 'current_est_step', 'current_mask',
                 'est_class_dist', 'count_kl', 'weight_kl') if hasattr(self, k)},
            lambda_optimizer=None if self.lambda_optimizer is None else self.lambda_optimizer.state_dict(),
            lambda_scheduler=None if self.lambda_scheduler is None else self.lambda_scheduler.state_dict())
        return saved

    def load_model(self, load_path):
        saved = torch.load(load_path, map_location='cpu', weights_only=False)
        if saved.get('lla_protocol') != self.protocol_settings():
            raise ValueError('LLA protocol mismatch: refusing an incompatible checkpoint')
        # Loading is explicit because legacy load_model assumes epoch-only resume.
        self.resume = True  # EMAHook.before_run must load saved EMA, not re-register online weights.
        self.model.load_state_dict(saved['model'])
        self.ema_model.load_state_dict(saved['ema_model'])
        local_rank_state = None
        if self.distributed:
            states = saved.get('rank_resume_states')
            if states is None or len(states) != self.world_size:
                raise ValueError('Checkpoint missing per-rank state for this world size')
            local_rank_state = states[self.rank]
            for model, key in ((self.model, 'model_buffers'), (self.ema_model, 'ema_buffers')):
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(local_rank_state[key][name].to(value.device))
        if self.ema is not None:
            self.ema.load(self.ema_model)
        self.optimizer.load_state_dict(saved['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved['scheduler'])
        self.loss_scaler.load_state_dict(saved['loss_scaler'])
        if self.lambda_optimizer is not None:
            self.lambda_optimizer.load_state_dict(saved['lambda_optimizer'])
        if self.lambda_scheduler is not None:
            self.lambda_scheduler.load_state_dict(saved['lambda_scheduler'])
        self.it, self.start_epoch = saved['it'], saved['epoch']
        self.epoch = self.start_epoch
        self.best_it, self.best_eval_acc = saved['best_it'], saved['best_eval_acc']
        self.best_eval_f1, self.best_epoch = saved['best_eval_f1'], saved['best_epoch']
        self.optimizer_steps = saved['optimizer_steps']
        device = next(self.model.parameters()).device
        for key, value in saved.get('legacy_metaexpert', {}).items():
            setattr(self, key, value.to(device) if isinstance(value, torch.Tensor) else value)
        self._resume_rng, self._epoch_rng = saved['rng'], saved['epoch_start_rng']
        if self.distributed:
            states = saved.get('rank_resume_states')
            if states is None or len(states) != self.world_size:
                raise ValueError('Checkpoint missing per-rank RNG for this world size')
            self._resume_rng = states[self.rank]['rng']
            self._epoch_rng = states[self.rank]['epoch_start_rng']
        for key, state in saved['sampler_state'].items():
            self.loader_dict[key].sampler.seed = state['seed']
            self.loader_dict[key].sampler.set_epoch(state['epoch'])
        if local_rank_state is not None:
            for key, value in local_rank_state['legacy_metaexpert'].items():
                setattr(self, key, value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in local_rank_state['selection'].items():
                setattr(self, key, value)
            if self.cola is not None:
                saved = dict(saved, cola_two_kl=local_rank_state['cola_two_kl'])
        self.restore_ddde_state(saved)
        self._sync_tail_tau()
        restore_rng(self._resume_rng)
        return saved

    def collect_rank_resume_states(self):
        self._rank_resume_states = None
        if self.distributed:
            local = dict(rng=capture_rng(), epoch_start_rng=self._epoch_rng,
                model_buffers={k:v.detach().cpu().clone() for k,v in self.model.named_buffers()},
                ema_buffers={k:v.detach().cpu().clone() for k,v in self.ema_model.named_buffers()},
                legacy_metaexpert={k:copy.deepcopy(getattr(self,k)) for k in
                    ('record_w1','record_w3','current_epoch','current_est_step','current_mask',
                     'est_class_dist','count_kl','weight_kl') if hasattr(self,k)},
                selection={k:getattr(self,k) for k in
                    ('best_it','best_eval_acc','best_eval_f1','best_epoch')},
                cola_two_kl=None if self.cola is None else copy.deepcopy(self.cola.state_dict()))
            self._rank_resume_states = [None] * self.world_size
            torch.distributed.all_gather_object(self._rank_resume_states, local)

    def save_model(self, save_name, save_path):
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        temporary = path / (save_name + '.partial')
        torch.save(self.get_save_dict(), temporary)
        temporary.replace(path / save_name)

    def train(self, stop_after=None):
        """Resume at the exact batch cursor; replay skipped worker augmentation.

        Mid-epoch iterator seeds are reconstructed from epoch-start RNG, then
        checkpoint RNG is restored before obtaining the next real batch. With
        workers>0, prefetch/worker streams are replayed by consuming the prefix.
        """
        self.model.train()
        self.call_hook('before_run')
        for epoch in range(self.start_epoch, self.epochs):
            if self.it >= self.num_train_iter:
                break
            self.epoch = epoch
            self.call_hook('before_train_epoch')
            skip = self.it % self.num_iter_per_epoch
            pending = self._resume_rng
            if pending is not None and skip:
                if self._epoch_rng is None:
                    raise RuntimeError('Mid-epoch resume requires epoch-start RNG')
                restore_rng(self._epoch_rng)
            else:
                if pending is not None:
                    restore_rng(pending)
                self._epoch_rng = capture_rng()
            batches = iter(zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']))
            for _ in range(skip):
                next(batches)
            if pending is not None:
                restore_rng(pending)
                self._resume_rng = None
            for data_lb, data_ulb in batches:
                if self.it >= self.num_train_iter:
                    break
                self.call_hook('before_train_step')
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self._checkpoint_during_step = True
                try:
                    self.call_hook('after_train_step')
                finally:
                    self._checkpoint_during_step = False
                self.it += 1
                if stop_after is not None and self.it >= stop_after:
                    self.collect_rank_resume_states()
                    if not self.distributed or self.rank == 0:
                        self.save_model('interrupted_model.pth', Path(self.save_dir) / self.save_name)
                    return
            self.call_hook('after_train_epoch')
        self.collect_rank_resume_states()
        if not self.distributed or self.rank == 0:
            self.save_model('latest_model.pth', Path(self.save_dir) / self.save_name)
        self.call_hook('after_run')

    @staticmethod
    def get_argument():
        return BacnMetaExpert.get_argument() + [
            SSL_Argument('--e3_adjust_mode', str, 'lla_hard'),
            SSL_Argument('--lla_init', float, 0.),
            SSL_Argument('--lla_epsilon', float, 1e-8),
            SSL_Argument('--lla_q_stop_gradient', str2bool, True),
            SSL_Argument('--lla_optimizer', str, 'Adam'),
            SSL_Argument('--lla_lr', float, .0005),
            SSL_Argument('--lla_weight_decay', float, 0.),
            SSL_Argument('--lla_betas', list, [.9, .999]),
            SSL_Argument('--lla_optimizer_epsilon', float, 1e-8),
            SSL_Argument('--lla_scheduler', str, 'constant'),
            SSL_Argument('--lla_clip_grad', float, 0.),
            SSL_Argument('--lla_tail_class_ids', list, [6, 7, 8]),
        ]
