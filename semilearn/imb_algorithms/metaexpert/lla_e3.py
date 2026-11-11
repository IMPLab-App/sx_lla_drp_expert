"""LLA-E3 Hard adaptation of Lee et al., ICCV 2025, equations (9)-(15).

No EFCC. Network logits are detached in the controller objective; adjustment
parameters are detached whenever used by the base network objective.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LLAE3Controller(nn.Module):
    def __init__(self, num_classes, init=0.0, epsilon=1e-8, q_stop_gradient=True):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        if not 0 < epsilon < 1:
            raise ValueError('epsilon must be in (0, 1)')
        values = torch.as_tensor(init, dtype=torch.float32)
        if values.ndim == 0:
            values = values.expand(num_classes).clone()
        if values.shape != (num_classes,) or not torch.isfinite(values).all():
            raise ValueError('lambda initialization must be finite and match the class count')
        self.lambda_logits = nn.Parameter(values.clone())
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.q_stop_gradient = q_stop_gradient

    def adjustment(self, detach=False):
        value = self.lambda_logits.detach() if detach else self.lambda_logits
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite lambda')
        return -F.log_softmax(value.float(), dim=0)

    def adjust(self, logits, detach_lambda=True):
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError('E3 logits dimension differs from lambda / actual class count')
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite E3 logits')
        value = logits.float() + self.adjustment(detach=detach_lambda)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite adjusted E3 logits')
        return value

    def class_average_loss(self, labeled_logits, labels, weak_logits, strong_logits):
        # Float32 log_softmax and reductions even under AMP. No theta graph.
        labeled = self.adjust(labeled_logits.detach(), detach_lambda=False)
        weak = self.adjust(weak_logits.detach(), detach_lambda=False)
        strong = self.adjust(strong_logits.detach(), detach_lambda=False)
        if labels.ndim != 1 or len(labels) != len(labeled) or labels.dtype != torch.long:
            raise ValueError('Aligned int64 labeled targets required')
        if len(labels) and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError('Labeled target outside class range')
        if weak.shape != strong.shape:
            raise ValueError('Weak/strong unlabeled batches must align')
        one_hot = F.one_hot(labels, self.num_classes).float()
        ll = (-(one_hot * labeled.log_softmax(-1)).sum(0) /
              (one_hot.sum(0) + self.epsilon)).mean()
        q = weak.softmax(-1)
        # Explicit adaptation choice: teacher probabilities and their class
        # normalizers are stop-gradient. The paper does not specify this edge.
        if self.q_stop_gradient:
            q = q.detach()
        lu = (-(q * strong.log_softmax(-1)).sum(0) /
              (q.sum(0) + self.epsilon)).mean()
        total = ll + lu
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite LLA class-average loss')
        return ll, lu, total


def assert_parameter_partition(model, network_optimizer, lambda_optimizer=None):
    required = {id(p) for p in model.parameters() if p.requires_grad}
    theta = [id(p) for g in network_optimizer.param_groups for p in g['params']]
    lam = ([] if lambda_optimizer is None else
           [id(p) for g in lambda_optimizer.param_groups for p in g['params']])
    if (len(theta) != len(set(theta)) or len(lam) != len(set(lam)) or
            set(theta) & set(lam) or set(theta + lam) != required):
        raise RuntimeError('Optimizers must partition all trainable parameters without overlap')
    return dict(network_tensors=len(theta), lambda_tensors=len(lam))
