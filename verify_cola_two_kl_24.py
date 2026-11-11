"""Real GPU train_step/backward/step/checkpoint checks without formal outputs."""
import argparse
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import yaml

from prepare_cola_two_kl_24 import ROOT, VARIANTS, RUNS, PROTECTED
from train import get_config
from semilearn.core.utils import get_net_builder, send_model_cuda
from semilearn.imb_algorithms import get_imb_algorithm
from semilearn.imb_algorithms.metaexpert.cola_ddde import TwoKLDDDE
from semilearn.imb_algorithms.metaexpert.metaexpert import MetaExpertNet
from semilearn.nets.wrn.wrn_metaexpert import WideResNet
from eval_confusion_matrix_new import get_eval_logits


def fail(*args, **kwargs):
    raise AssertionError('Disabled DDDE/MFF/routing path executed')


def verify():
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA GPU verification is required'
    manifest = json.loads((ROOT / 'config/cola_two_kl_24/manifest.json').read_text())
    for task in manifest['tasks']:
        cfg = yaml.safe_load((ROOT / task['config']).read_text())
        baseline = yaml.safe_load((ROOT / task['source']).read_text())
        assert all(cfg[k] == baseline[k] for k in PROTECTED)
    report = dict(device=torch.cuda.get_device_name(0), config_count=len(manifest['tasks']), variants=[])
    with tempfile.TemporaryDirectory(prefix='cola_smoke_') as temporary:
        temp = Path(temporary)
        data = temp / 'data'; data.mkdir()
        for name in ('raw_signal_dataset_G_1024.npz',):
            (data / name).symlink_to(ROOT / 'data' / name)
        for variant in VARIANTS:
            cfg_path = ROOT / 'config/cola_two_kl_24' / variant / (RUNS[0] + '.yaml')
            argv = sys.argv
            try:
                sys.argv = ['verify', '--c', str(cfg_path)]
                args = get_config()
            finally:
                sys.argv = argv
            args.data_dir = str(data)
            args.save_dir = str(temp / 'models')
            args.num_workers = 0
            args.distributed = False
            args.gpu = 0
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            with contextlib.redirect_stdout(io.StringIO()):
                algorithm = get_imb_algorithm(args, get_net_builder(args.net, args.net_from_name),
                                               None, logging.getLogger('smoke'))
            algorithm.model = send_model_cuda(args, algorithm.model)
            algorithm.ema_model = send_model_cuda(args, algorithm.ema_model, clip_batch=False)
            algorithm.finalize_model_setup()
            # Use real augmented signal dataset samples, with deterministic labels.
            batch_lb = next(iter(torch.utils.data.DataLoader(algorithm.dataset_dict['train_lb'], batch_size=4)))
            batch_u = next(iter(torch.utils.data.DataLoader(algorithm.dataset_dict['train_ulb'], batch_size=4)))
            inputs = algorithm.process_batch(**batch_lb, **batch_u)
            algorithm.it = 0
            algorithm.epoch = 25  # Fusion losses are active, past original warmup.
            module = algorithm.model
            hooks = []
            if variant in ('no_mff', 'no_dea_aggregator'):
                hooks.append(module.mff.register_forward_pre_hook(fail))
            if variant == 'no_dea_aggregator':
                hooks.append(module.predict.register_forward_pre_hook(fail))
            # Check classifiers really see the very same shared tensor [B,D].
            classifier_inputs = []
            def capture(m, a):
                classifier_inputs.append(a[0])
            h1 = module.backbone.classifier.register_forward_pre_hook(capture)
            h3 = module.backbone.aux_classifier2.register_forward_pre_hook(capture)
            x = inputs['x_lb']
            out = module(x)
            assert out['feat'].shape == (4, 192)
            assert out['logits'].shape == out['aux_logits2'].shape == (4, args.num_classes)
            assert classifier_inputs[0] is classifier_inputs[1]
            torch.testing.assert_close(out['logits'], module.backbone.classifier(out['feat']))
            torch.testing.assert_close(out['aux_logits2'], module.backbone.aux_classifier2(out['feat']))
            h1.remove(); h3.remove()
            if variant == 'no_dea_aggregator':
                torch.testing.assert_close(get_eval_logits(out), 0.5 * out['logits'] + 0.5 * out['aux_logits2'])
                torch.testing.assert_close(out['fuse_logit_l'], get_eval_logits(out))
            if algorithm.cola is not None:
                algorithm.cola.begin_epoch(24)
                algorithm.cola.observe(torch.eye(9, device='cuda').repeat_interleave(3, 0) * 20,
                                       torch.randn(27, 192, device='cuda', dtype=torch.float16))
                assert algorithm.cola.pi_u.dtype == torch.float32 and algorithm.cola.pi_u.device.type == 'cuda'
            forward_outputs = []
            capture_output = module.register_forward_hook(lambda m, a, o: forward_outputs.append(o))
            observer = patch.object(TwoKLDDDE, 'observe', fail) if variant == 'no_ddde_kl' else patch.object(algorithm.cola, 'observe', wraps=algorithm.cola.observe)
            with observer as spy:
                result, logs = algorithm.train_step(**inputs)
                if variant != 'no_ddde_kl':
                    observed_logits, observed_features = spy.call_args[0]
                    torch.testing.assert_close(observed_logits, forward_outputs[0]['aux_logits2'][4:8])
                    torch.testing.assert_close(observed_features, forward_outputs[0]['feat'][4:8])
            capture_output.remove()
            del forward_outputs
            assert torch.isfinite(result['loss'])
            before = {n: p.detach().clone() for n, p in module.named_parameters() if p.requires_grad}
            result['loss'].backward()
            expected = dict(backbone=module.backbone.patch_embed.signal_proj,
                            head=module.backbone.classifier, tail=module.backbone.aux_classifier2)
            if variant not in ('no_mff', 'no_dea_aggregator'):
                expected['mff'] = module.mff
            if variant != 'no_dea_aggregator':
                expected['routing'] = module.predict
            for name, model in expected.items():
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0, name
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
            algorithm.optimizer.step()
            for name, model in expected.items():
                ids = {id(p) for p in model.parameters()}
                assert any(not torch.equal(before[n], p) for n, p in module.named_parameters() if id(p) in ids and n in before), name
            if variant == 'no_ddde_kl':
                assert algorithm.cola is None and logs['train/tau_T'] == 4. and module.tau3.item() == 4.
            else:
                assert 0 <= module.tau3.item() <= 4
                assert module.tau1.item() == args.la_tau_lb1
                assert 'train/KL_con' not in logs
            if variant == 'no_dea_aggregator':
                assert logs['train/total_fuse_loss'] == 0
            for h in hooks:
                h.remove()
            # Actual save/load restores network, optimizer, EMA, pending KL window and tau.
            algorithm.it = 123
            algorithm.save_model('smoke.pth', str(temp / variant))
            saved = torch.load(temp / variant / 'smoke.pth', map_location='cpu', weights_only=False)
            with torch.no_grad():
                module.backbone.classifier.weight.add_(1)
                if algorithm.cola is not None:
                    algorithm.cola.pi_u.zero_()
            algorithm.load_model(str(temp / variant / 'smoke.pth'))
            torch.testing.assert_close(module.backbone.classifier.weight.cpu(), saved['model']['backbone.classifier.weight'])
            if algorithm.cola is not None:
                for key, value in saved['cola_two_kl'].items():
                    torch.testing.assert_close(algorithm.cola.state_dict()[key].cpu(), value)
            assert algorithm.optimizer.state
            assert algorithm.it == 124
            record = dict(variant=variant, loss=result['loss'].item(),
                          optimizer_parameter_tensors=algorithm.assert_optimizer_coverage(),
                          gradient_modules=list(expected), checkpoint='passed',
                          tau_T=logs['train/tau_T'], shape=list(out['feat'].shape))
            report['variants'].append(record)
            print('PASS', json.dumps(record), flush=True)
            del algorithm, module, result, out, before, expected, saved, classifier_inputs
            torch.cuda.empty_cache()
        # WRN has B shared features (never a 2B BN-concatenated feature tensor).
        wrn = MetaExpertNet(WideResNet(depth=10, widen_factor=1, num_classes=9), 9,
                           torch.ones(9) / 9, 0., 4., 1, 1).cuda()
        x = torch.randn(4, 3, 32, 32, device='cuda')
        out = wrn(x)
        assert out['feat'].shape == (4, 64)
        assert out['logits'].shape == out['aux_logits2'].shape == (4, 9)
        (out['logits'].sum() + out['aux_logits2'].sum() + out['fuse_logit_l'].sum()).backward()
        report['wrn_shape'] = list(out['feat'].shape)
        print('PASS WRN shared-feature forward/backward', report['wrn_shape'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='logs/cola_two_kl_24/setup/gpu_smoke.json')
    args = parser.parse_args()
    report = verify()
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
