"""Numerical edge cases and the project's two-reference (not original CoLA) map."""
import io
import unittest
from unittest.mock import patch
import torch
from semilearn.imb_algorithms.metaexpert.cola_ddde import (
    TwoKLDDDE, effective_rank, symmetric_kl, tau_from_kl, normalize_distribution,
)


class TestDDDE(unittest.TestCase):
    def test_effective_rank(self):
        self.assertEqual(effective_rank(torch.empty(0, 4)).item(), 0)
        self.assertEqual(effective_rank(torch.ones(1, 4)).item(), 1)
        self.assertEqual(effective_rank(torch.zeros(4, 4)).item(), 0)
        self.assertAlmostEqual(effective_rank(torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])).item(), 2., places=5)
        self.assertTrue(torch.isfinite(effective_rank(torch.full((4, 4), float('nan')))))
        self.assertTrue(torch.isfinite(effective_rank(torch.randn(4, 4) * 1e37)))
        with patch('torch.linalg.svdvals', side_effect=RuntimeError('SVD failed')):
            self.assertEqual(effective_rank(torch.randn(3, 4)).item(), 0)

    def test_kl_symmetry_and_zeros(self):
        p, q = torch.tensor([1., 0., 0.]), torch.tensor([0., 0., 1.])
        self.assertTrue(torch.isfinite(symmetric_kl(p, q)))
        torch.testing.assert_close(symmetric_kl(p, q), symmetric_kl(q, p))
        self.assertEqual(symmetric_kl(p, p).item(), 0)
        self.assertTrue(torch.isfinite(normalize_distribution(torch.tensor([float('inf'), float('nan'), 0.]))).all())

    def test_formula_range_monotonicity(self):
        for a, b in ((0., 0.), (1., 3.), (10000., 0.), (0., 10000.)):
            a, b = torch.tensor(a), torch.tensor(b)
            actual = tau_from_kl(a, b, 4.)
            torch.testing.assert_close(actual, 4 * torch.softmax(torch.stack([-a, -b]), 0)[1])
            self.assertTrue(0 <= actual <= 4)
        values = [tau_from_kl(torch.tensor(1.), torch.tensor(b), 4).item() for b in [0., 1., 2., 10.]]
        self.assertEqual(values, sorted(values, reverse=True))
        rev = torch.tensor([.01, .09, .9]); uni = torch.ones(3) / 3
        near_rev = tau_from_kl(symmetric_kl(rev, uni), symmetric_kl(rev, rev), 4)
        near_uni = tau_from_kl(symmetric_kl(uni, uni), symmetric_kl(uni, rev), 4)
        self.assertGreater(near_rev, near_uni)

    def test_empty_confidence_window_and_timing(self):
        c = TwoKLDDDE(torch.tensor([.8, .15, .05]))
        c.begin_epoch(17)
        self.assertIsNone(c.observe(torch.eye(3) * 20, torch.eye(3)))
        c.begin_epoch(18)
        initial = c.pi_u.clone()
        self.assertIsNone(c.observe(torch.zeros(3, 3), torch.randn(3, 4)))
        torch.testing.assert_close(c.pi_u, initial)
        c.begin_epoch(19)
        self.assertEqual(c.tau_dyn.item(), 4)
        kl = c.observe(torch.eye(3) * 20, torch.eye(3))
        self.assertIsNotNone(kl)
        self.assertEqual(c.tau_dyn.item(), 4)  # No within-epoch tau change.
        expected = tau_from_kl(*kl, 4)
        c.begin_epoch(20)
        torch.testing.assert_close(c.tau_dyn, expected)
        c.begin_epoch(21)
        torch.testing.assert_close(c.tau_dyn, expected)  # Empty window keeps tau.

    def test_missing_classes_invalid_rows_and_resume(self):
        c = TwoKLDDDE(torch.tensor([.8, .15, .05])); c.begin_epoch(18)
        logits = torch.tensor([[20., 0., 0.], [float('inf'), 0., 0.], [0., 0., 20.]])
        features = torch.tensor([[1., 2.], [1., 2.], [float('nan'), 0.]])
        self.assertIsNotNone(c.observe(logits, features))
        torch.testing.assert_close(c.pi_ddde, torch.tensor([1., 0., 0.]))
        clone = TwoKLDDDE(c.pi_l); clone.load_state_dict(c.state_dict())
        for key in c.state_dict():
            torch.testing.assert_close(c.state_dict()[key], clone.state_dict()[key])
        c.begin_epoch(19); clone.begin_epoch(19)
        torch.testing.assert_close(c.tau_dyn, clone.tau_dyn)


if __name__ == '__main__':
    unittest.main()
