"""Tests for the plain sentiment-decodability check (no GPU).

Run: python -B -m unittest evaluation.test_sentiment_decodability_check
"""
import unittest

import numpy as np

from evaluation.sentiment_decodability_check import plain_decodability

D = 32
N_GROUPS = 8
PER_GROUP = 12


def _grouped_world(seed, signal):
    """N_GROUPS subjects x PER_GROUP trials, 3 balanced classes. When ``signal`` is
    True each class gets a distinct mean so it is decodable across held-out groups;
    when False X is pure noise (labels unrelated to X)."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    centers = rng.normal(size=(3, D)) * (3.0 if signal else 0.0)
    for grp in range(N_GROUPS):
        for i in range(PER_GROUP):
            c = i % 3
            X.append(centers[c] + rng.normal(size=D))
            y.append(c)
            g.append(f"S{grp}")
    return np.array(X), np.array(y), np.array(g)


class DecodabilityTests(unittest.TestCase):
    def test_signal_is_decodable_and_beats_permutation(self):
        X, y, g = _grouped_world(0, signal=True)
        out = plain_decodability(X, y, g, n_splits=4, n_permutations=50, seed=1, n_jobs=1)
        self.assertTrue(out["above_chance"])
        self.assertLess(out["p_value_permutation"], 0.05)
        self.assertGreater(out["real"]["balanced_accuracy"],
                           out["permutation_chance_balanced_accuracy"])
        self.assertTrue(out["verdict"].startswith("DECODABLE"))

    def test_noise_is_at_chance(self):
        X, y, g = _grouped_world(1, signal=False)
        out = plain_decodability(X, y, g, n_splits=4, n_permutations=50, seed=2, n_jobs=1)
        self.assertFalse(out["above_chance"])
        self.assertGreater(out["p_value_permutation"], 0.05)
        self.assertTrue(out["verdict"].startswith("AT CHANCE"))

    def test_permutation_chance_near_analytic(self):
        X, y, g = _grouped_world(2, signal=False)
        out = plain_decodability(X, y, g, n_splits=4, n_permutations=80, seed=3, n_jobs=1)
        # empirical chance for balanced accuracy should sit near 1/3
        self.assertAlmostEqual(out["permutation_chance_balanced_accuracy"], 1.0 / 3, delta=0.08)
        self.assertEqual(out["analytic_chance_balanced_accuracy"], round(1.0 / 3, 6))

    def test_parallel_matches_sequential(self):
        # pre-generated perms make the null order-independent: n_jobs must not change results
        X, y, g = _grouped_world(3, signal=True)
        seq = plain_decodability(X, y, g, n_splits=4, n_permutations=40, seed=7, n_jobs=1)
        par = plain_decodability(X, y, g, n_splits=4, n_permutations=40, seed=7, n_jobs=2)
        self.assertEqual(seq["p_value_permutation"], par["p_value_permutation"])
        self.assertEqual(seq["permutation_chance_balanced_accuracy"],
                         par["permutation_chance_balanced_accuracy"])
        self.assertEqual(seq["real"], par["real"])

    def test_feasible_splits_capped_by_group_support(self):
        X, y, g = _grouped_world(4, signal=True)
        out = plain_decodability(X, y, g, n_splits=99, n_permutations=10, seed=5, n_jobs=1)
        self.assertLessEqual(out["n_splits_used"], N_GROUPS)
        self.assertGreaterEqual(out["n_splits_used"], 2)

    def test_length_mismatch_raises(self):
        X, y, g = _grouped_world(0, signal=True)
        with self.assertRaises(ValueError):
            plain_decodability(X, y[:-1], g, n_permutations=5, n_jobs=1)


if __name__ == "__main__":
    unittest.main()
