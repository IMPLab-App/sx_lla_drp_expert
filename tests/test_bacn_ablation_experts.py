"""Removing DEA must retain the original two expert objectives and updates."""
from pathlib import Path
import tempfile
import unittest

import torch

from verify_bacn_ablation import assert_state_equal, make_algorithm


@unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA comparison required')
class ExpertObjectiveTest(unittest.TestCase):
    def test_no_dea_retains_exact_expert_losses_and_backbone_updates(self):
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        config = Path(__file__).resolve().parents[1] / 'logs/bacn_11_20261105/configs/run_1_r1_pic_lb35_5_ulb315_5_exp.yaml'
        with tempfile.TemporaryDirectory(prefix='bacn_expert_oracle_') as temporary:
            original = make_algorithm(config, 'metaexpert', True, False, Path(temporary))
            removed = make_algorithm(config, 'metaexpert_bacn', True, True, Path(temporary))
            # Force confident Head and Tail pseudo-labels to exercise both H/T
            # unsupervised masks, rather than only the supervised objective.
            with torch.no_grad():
                for algorithm in (original, removed):
                    algorithm.model.backbone.classifier.bias[0] = 8
                    algorithm.model.backbone.aux_classifier2.bias[8] = 8
            assert_state_equal(original.model.backbone.state_dict(), removed.model.backbone.state_dict())
            labeled = next(iter(torch.utils.data.DataLoader(original.dataset_dict['train_lb'], batch_size=8)))
            unlabeled = next(iter(torch.utils.data.DataLoader(original.dataset_dict['train_ulb'], batch_size=8)))
            inputs = original.process_batch(**labeled, **unlabeled)
            original.epoch = removed.epoch = 0  # Original fusion objective is off.
            original.it = removed.it = 0
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            a, logs_a = original.train_step(**inputs)
            a['loss'].backward()
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng)
            b, logs_b = removed.train_step(**inputs)
            b['loss'].backward()
            self.assertGreater(logs_a['train/unsup_loss'], 0)
            self.assertEqual(logs_a['train/util_ratio'], 1)
            torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
            self.assertEqual(logs_a['train/sup_loss'], logs_b['train/sup_loss'])
            self.assertEqual(logs_a['train/unsup_loss'], logs_b['train/unsup_loss'])
            for (name,p), (other,q) in zip(original.model.backbone.named_parameters(), removed.model.backbone.named_parameters()):
                self.assertEqual(name, other)
                self.assertEqual(p.grad is None, q.grad is None)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
            original.optimizer.step(); removed.optimizer.step()
            assert_state_equal(original.model.backbone.state_dict(), removed.model.backbone.state_dict())


if __name__ == '__main__':
    unittest.main()
