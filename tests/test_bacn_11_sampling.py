"""Protect the experiment's sample identity and budget invariants."""
import unittest

import numpy as np

from prepare_bacn_11 import SCENARIOS
from semilearn.datasets.cv_datasets.raw_signal import _fixed_count_split


class FixedCountsTest(unittest.TestCase):
    def setUp(self):
        self.targets = np.repeat(np.arange(9), 350)

    def test_exact_counts_disjoint_and_global_rng_independent(self):
        for name, _, full, labeled in SCENARIOS:
            with self.subTest(name=name):
                lb, ulb = _fixed_count_split(self.targets, 9, full, labeled, 2)
                np.testing.assert_array_equal(np.bincount(self.targets[lb]), labeled)
                np.testing.assert_array_equal(np.bincount(self.targets[ulb]), np.array(full)-labeled)
                self.assertEqual(len(np.intersect1d(lb, ulb)), 0)
                np.random.seed(5678)
                other_lb, other_ulb = _fixed_count_split(self.targets, 9, full, labeled, 2)
                np.testing.assert_array_equal(lb, other_lb)
                np.testing.assert_array_equal(ulb, other_ulb)

    def test_fixed_full_membership_and_nested_labels(self):
        selected, labels = [], []
        for _, _, full, labeled in SCENARIOS:
            lb, ulb = _fixed_count_split(self.targets, 9, full, labeled, 2)
            selected.append(set(lb) | set(ulb))
            labels.append(set(lb))
        self.assertEqual(selected[5], selected[6])
        self.assertEqual(selected[6], selected[7])
        for case in (8,9,10):
            self.assertEqual(selected[4], selected[case])
        self.assertTrue(labels[10] < labels[9] < labels[8])

    def test_invalid_budget_rejected(self):
        for full, labeled in (([350]*8, [35]*9), ([351]*9, [35]*9),
                              ([10]*9, [35]*9), ([350]*9, [0]*9),
                              ([350.0]*9, [35]*9), ([True]*9, [1]*9)):
            with self.subTest(full=full, labeled=labeled), self.assertRaises(ValueError):
                _fixed_count_split(self.targets, 9, full, labeled, 2)


if __name__ == '__main__':
    unittest.main()
