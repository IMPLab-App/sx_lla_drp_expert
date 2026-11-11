"""Native 1-D WRN-28-2 variants preserving the imported baselines' head APIs.

Residual block topology is reused from the image WRN. All convolutions and
normalization operate on [B, C, L]; signals are never reshaped into images.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .wrn.wrn import WideResNet


def _convert_blocks(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            replacement = nn.Conv1d(child.in_channels, child.out_channels,
                                    child.kernel_size[0], child.stride[0],
                                    child.padding[0], bias=child.bias is not None)
            nn.init.kaiming_normal_(replacement.weight, mode='fan_out', nonlinearity='leaky_relu')
            if replacement.bias is not None:
                nn.init.zeros_(replacement.bias)
            setattr(module, name, replacement)
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.BatchNorm1d(child.num_features, eps=child.eps,
                                               momentum=child.momentum))
        else:
            _convert_blocks(child)


class SignalWideResNet(WideResNet):
    def __init__(self, num_classes=9, variant='plain', **kwargs):
        super().__init__(first_stride=1, num_classes=num_classes, depth=28,
                         widen_factor=2, **kwargs)
        _convert_blocks(self)
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.conv1.bias)
        self.variant = variant
        if variant == 'bacon':
            self.aux_classifier = nn.Linear(self.num_features, num_classes)
            self.projection = nn.Linear(self.num_features, 32)
        elif variant == 'cpe':
            self.BNH = nn.BatchNorm1d(self.num_features)
            self.BNM = nn.BatchNorm1d(self.num_features)
            self.BNT = nn.BatchNorm1d(self.num_features)
            self.aux_classifier1 = nn.Linear(self.num_features, num_classes)
            self.aux_classifier2 = nn.Linear(self.num_features, num_classes)
        for name, child in self.named_children():
            if name in ('aux_classifier', 'projection', 'aux_classifier1', 'aux_classifier2'):
                nn.init.xavier_normal_(child.weight)
                nn.init.zeros_(child.bias)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.classifier(x)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f'Expected [B, 1, L], got {tuple(x.shape)}')
        feature_map = self.extract(x)
        if self.variant == 'cpe':
            features = [F.adaptive_avg_pool1d(bn(feature_map), 1).flatten(1)
                        for bn in (self.BNH, self.BNM, self.BNT)]
            feat = torch.stack(features).mean(0)
            if only_feat:
                return feat
            result = {'feat': feat}
            for classifier, prefix, suffix in ((self.classifier, 'logits', ''),
                                              (self.aux_classifier1, 'aux_logits', '1'),
                                              (self.aux_classifier2, 'aux_logits', '2')):
                outputs = [classifier(f) for f in features]
                for group, logits in zip(('H', 'M', 'T'), outputs):
                    result[prefix + group + suffix] = logits
                result[prefix + suffix] = torch.stack(outputs).mean(0)
            return result
        feat = F.adaptive_avg_pool1d(feature_map, 1).flatten(1)
        if only_feat:
            return feat
        result = {'logits': self.classifier(feat), 'feat': feat}
        if self.variant == 'bacon':
            result.update(aux_logits=self.aux_classifier(feat), pro_out=self.projection(feat))
        return result


def _build(variant, pretrained=False, pretrained_path=None, **kwargs):
    model = SignalWideResNet(variant=variant, **kwargs)
    if pretrained:
        if not pretrained_path:
            raise ValueError('A compatible 1-D checkpoint is required')
        from .utils import load_checkpoint
        model = load_checkpoint(model, pretrained_path)
    return model


def wrn_28_2_signal(**kwargs):
    return _build('plain', **kwargs)


def wrn_28_2_signal_bacon(**kwargs):
    return _build('bacon', **kwargs)


def wrn_28_2_signal_cpe(**kwargs):
    return _build('cpe', **kwargs)
