"""Isolated update/selection hooks and diagnostics for the LLA protocol."""
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from semilearn.core.hooks import Hook


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def branch_metrics(y, prediction, classes, tail_ids):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=list(range(classes)), zero_division=0)
    return dict(accuracy=float(accuracy_score(y, prediction)), macro_f1=float(f1.mean()),
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
                support=support.tolist(), tail_macro_f1=float(f1[tail_ids].mean()),
                C5=dict(recall=float(recall[5]), f1=float(f1[5])) if classes > 5 else None,
                C8=dict(recall=float(recall[8]), f1=float(f1[8])) if classes > 8 else None,
                confusion_matrix=confusion_matrix(y, prediction, labels=list(range(classes))).tolist())


class IsolatedUpdateHook(Hook):
    def after_train_step(self, a):
        base = a.out_dict['loss']
        optimizers = [a.optimizer]
        losses = [base]
        if a.lambda_optimizer is not None:
            optimizers.append(a.lambda_optimizer)
            losses.append(a.out_dict['lambda_loss'])
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if a.use_amp:
            for loss in losses:
                a.loss_scaler.scale(loss).backward()
            for opt in optimizers:
                a.loss_scaler.unscale_(opt)
        else:
            for loss in losses:
                loss.backward()
        theta = [p for g in a.optimizer.param_groups for p in g['params']]
        if a.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(theta, a.clip_grad)
        lambda_before = None
        if a.lambda_optimizer is not None:
            p = a.module.lla.lambda_logits
            lambda_before = p.detach().clone()
            if p.grad is None:
                raise RuntimeError('Lambda gradient missing')
            if a.args.lla_clip_grad > 0:
                torch.nn.utils.clip_grad_norm_([p], a.args.lla_clip_grad)
            a.log_dict['train/lambda_grad_norm'] = float(p.grad.detach().norm())
            # Lambda is consumed outside a DDP forward graph; synchronize its
            # gradient explicitly. theta retains native DDP reduction.
            if a.distributed:
                torch.distributed.all_reduce(p.grad)
                p.grad.div_(torch.distributed.get_world_size())
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for opt in optimizers for g in opt.param_groups for p in g['params'])
        if a.distributed:
            flag = torch.tensor(int(finite), device=base.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            finite = bool(flag.item())
        if not finite:
            if not a.use_amp:
                raise FloatingPointError('Non-finite optimizer gradient')
            # Skip BOTH optimizers atomically; avoid lambda advancing alone.
            a.loss_scaler.update(new_scale=a.loss_scaler.get_scale() / 2)
            a.log_dict['train/optimizer_step_skipped'] = 1
        else:
            for opt in optimizers:
                if a.use_amp:
                    a.loss_scaler.step(opt)
                else:
                    opt.step()
            if a.use_amp:
                a.loss_scaler.update()
            if a.scheduler is not None:
                a.scheduler.step()
            if a.lambda_scheduler is not None:
                a.lambda_scheduler.step()
            a.optimizer_steps += 1
            a.log_dict['train/optimizer_step_skipped'] = 0
        if lambda_before is not None:
            a.log_dict['train/lambda_update_norm'] = float((a.module.lla.lambda_logits.detach()-lambda_before).norm())
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        a.log_dict['lr'] = a.optimizer.param_groups[-1]['lr']
        a.record_trajectory()


class MacroF1SelectionHook(Hook):
    def after_train_step(self, a):
        if self.every_n_iters(a, a.num_eval_iter) or self.is_last_iter(a):
            metrics = a.evaluate('eval')
            a.log_dict.update(metrics)
            if metrics['eval/F1'] > a.best_eval_f1:
                a.best_eval_f1 = metrics['eval/F1']
                a.best_eval_acc = metrics['eval/top-1-acc']
                a.best_it = a.it
                a.best_epoch = a.epoch

    def after_run(self, a):
        # Test evaluation is intentionally separate, after all stage-1 models
        # and the validation-only gate have been locked.
        a.results_dict = {'eval/best_F1': a.best_eval_f1, 'eval/best_acc': a.best_eval_acc,
                          'eval/best_it': a.best_it, 'eval/best_epoch': a.best_epoch}
        a.write_json('validation_selection.json', dict(a.results_dict, criterion='validation_macro_f1'))


class ProtocolCheckpointHook(Hook):
    def after_train_step(self, a):
        if self.every_n_iters(a, a.num_eval_iter) or self.is_last_iter(a):
            a.collect_rank_resume_states()
            if not a.distributed or a.rank == 0:
                folder = __import__('pathlib').Path(a.save_dir) / a.save_name
                a.save_model('latest_model.pth', folder)
                if a.it == a.best_it:
                    a.save_model('model_best.pth', folder)
