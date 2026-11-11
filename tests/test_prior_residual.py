"""Prior correction: sign, clipping, missing classes and delayed state restore."""
import inspect
import unittest
from unittest.mock import patch

import torch

from semilearn.imb_algorithms.metaexpert.prior_residual import (
    PriorResidualController, PriorResidualMetaExpert, prior_difference_residual)


class PriorResidualTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_ratio_direction_and_clipping(self):
        p, q = torch.tensor([.8, .2]), torch.tensor([.2, .8])
        expected = .5 * torch.log(p / q).clamp(-1, 1)
        torch.testing.assert_close(prior_difference_residual(p, q, .5, 1), expected)
        self.assertGreater(float(expected[0]), 0)
        self.assertLess(float(expected[1]), 0)
        torch.testing.assert_close(prior_difference_residual(p, p), torch.zeros(2))

    def test_zero_strength_does_not_require_unlabeled_prior(self):
        p = torch.tensor([.8, .2])
        torch.testing.assert_close(prior_difference_residual(p, None, strength=0), torch.zeros(2))

    def test_missing_classes_are_finite_and_bounded(self):
        r = prior_difference_residual(torch.tensor([.9, .1]), torch.tensor([1., 0.]))
        self.assertTrue(torch.isfinite(r).all())
        self.assertLessEqual(float(r.abs().max()), 1.)
        for q in (torch.tensor([float('nan'), 0.]), torch.tensor([-1., 2.])):
            with self.assertRaises(ValueError):
                prior_difference_residual(torch.tensor([.9, .1]), q)

    def test_histogram_never_calls_rank_or_distance_and_uses_previous_batch(self):
        p = torch.tensor([.8, .2])
        c = PriorResidualController(p, 'histogram', est_epoch=1)
        logits = torch.tensor([[20., -20.], [20., -20.], [-20., 20.]])
        features = torch.tensor([[1., 2.], [2., 1.], [0., 3.]])
        c.begin_step(0)
        self.assertFalse(c.observe(logits, features))
        c.begin_step(1)
        with patch('semilearn.imb_algorithms.metaexpert.prior_residual.effective_rank', side_effect=AssertionError('ER called')):
            self.assertTrue(c.observe(logits, features))
        torch.testing.assert_close(c.pi_raw, torch.tensor([2/3, 1/3]))
        torch.testing.assert_close(c.pi_u, .9 * torch.ones(2)/2 + .1 * c.pi_raw)
        torch.testing.assert_close(c.applied_residual, torch.zeros(2))
        c.begin_step(1)
        torch.testing.assert_close(c.applied_residual, prior_difference_residual(p, c.pi_u))
        applied = c.applied_residual.clone()
        self.assertFalse(c.observe(torch.zeros(3, 2), features))
        torch.testing.assert_close(c.applied_residual, applied)

    def test_rank_and_histogram_are_distinct_estimators(self):
        logits = torch.tensor([[20., -20.], [20., -20.], [-20., 20.]])
        features = torch.tensor([[1., 2.], [2., 1.], [0., 3.]])
        c = PriorResidualController(torch.tensor([.8, .2]), 'effective_rank', est_epoch=0)
        c.begin_step(0)
        self.assertTrue(c.observe(logits, features))
        torch.testing.assert_close(c.last_counts, torch.tensor([2., 1.]))
        torch.testing.assert_close(c.last_rank, torch.tensor([1., 1.]))
        torch.testing.assert_close(c.pi_raw, torch.tensor([.5, .5]))

    def test_checkpoint_keeps_applied_and_next_prior_separate(self):
        a = PriorResidualController(torch.tensor([.8, .2]), est_epoch=0)
        logits, features = torch.tensor([[20., -20.]]), torch.ones(1, 2)
        a.begin_step(0)
        a.observe(logits, features)
        a.begin_step(0)
        applied, used = a.applied_residual.clone(), a.pi_used.clone()
        a.observe(logits, features)
        self.assertFalse(torch.equal(a.pi_u, used))
        b = PriorResidualController(torch.tensor([.8, .2]), est_epoch=0)
        b.load_state_dict(a.state_dict())
        torch.testing.assert_close(b.applied_residual, applied, rtol=0, atol=0)
        a.begin_step(1)
        b.begin_step(1)
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)

    def test_framework_signature_and_detached_statistics(self):
        self.assertEqual(list(inspect.signature(PriorResidualMetaExpert.train_step).parameters),
            ['self', 'x_lb', 'y_lb', 'y_lb_noised', 'x_ulb_w', 'x_ulb_s', 'y_ulb'])
        p = torch.tensor([.8, .2], requires_grad=True)
        self.assertFalse(prior_difference_residual(p, p).requires_grad)


if __name__ == '__main__':
    unittest.main()
