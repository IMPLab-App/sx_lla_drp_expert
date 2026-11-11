import json
from pathlib import Path
import tempfile
import unittest

import torch

from semilearn.imb_algorithms.metaexpert.cola_ddde import TwoKLDDDE
from semilearn.imb_algorithms.metaexpert.five_variants import DistributionController
from semilearn.imb_algorithms.metaexpert.temperature_audit import AuditedController


class TemperatureAuditTests(unittest.TestCase):
    def test_instrumentation_preserves_v2_v5_and_rng(self):
        for version in ('V2', 'V5'):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                prior = torch.arange(9, 0, -1).float()
                original = (DistributionController(prior, 'V2', est_epoch=0) if version == 'V2'
                            else TwoKLDDDE(prior, est_epoch=0))
                audited = AuditedController(prior, version, 1., Path(tmp)/'epochs.jsonl', est_epoch=0)
                features = torch.arange(27 * 8).reshape(27, 8).float().sin()
                logits = torch.eye(9).repeat_interleave(3, 0) * 20
                for epoch in (0, 1):
                    original.begin_epoch(epoch); audited.begin_epoch(epoch)
                    before = torch.get_rng_state().clone()
                    original.observe(logits, features); audited.observe(logits, features)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                    for key, value in original.state_dict().items():
                        torch.testing.assert_close(value, audited.state_dict()[key], rtol=0, atol=0)
                audited.flush_epoch()
                rows = [json.loads(s) for s in (Path(tmp)/'epochs.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['high_confidence_pseudo_counts'], [3]*9)
                self.assertEqual(rows[0]['valid_batches'], 1)
                self.assertTrue(all(x > 0 for x in rows[0]['rank_mean']))

    def test_temperature_changes_only_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            taus = []
            for temperature in (.25, .5, 1., 2., 4.):
                controller = AuditedController(torch.arange(9,0,-1).float(), 'V5', temperature,
                    Path(tmp)/f'{temperature}.jsonl', est_epoch=0)
                controller.begin_epoch(0)
                controller.kl_sum.copy_(torch.tensor([.2, 2.2]))
                controller.valid_batches.fill_(1)
                controller._previous_valid = True
                controller.begin_epoch(1)
                expected = 4 * torch.softmax(-torch.tensor([.2, 2.2])/temperature, 0)[1]
                torch.testing.assert_close(controller.tau_dyn, expected)
                taus.append(float(controller.tau_dyn))
            self.assertEqual(taus, sorted(taus))

    def test_no_valid_batch_keeps_previous_tau(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller=AuditedController(torch.ones(9),'V5',2.,Path(tmp)/'e.jsonl',est_epoch=0)
            controller.begin_epoch(0)
            controller.observe(torch.zeros(3,9), torch.ones(3,4))
            controller.begin_epoch(1)
            self.assertEqual(float(controller.tau_dyn),4.)


if __name__ == '__main__':
    unittest.main()
