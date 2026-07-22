"""Local tests for the clustered bootstrap + the Gate-2B primary decision rule.

Run: python -B -m unittest evaluation.test_token_decision
"""
import unittest

import numpy as np

from evaluation.clustered_bootstrap import (
    clustered_delta,
    lower_bound,
    paired_bootstrap,
    task_macro_mean,
)
from evaluation.token_decision import (
    DELTA_SUP,
    apply_primary_decision,
    paired_delta,
    pool_size_robust,
    seed_averaged_effect,
    seed_consistent,
    subject_consistent,
    top1_indicator,
)


def synthetic(n_subj=12, n_text=40, per=3, effect=0.03, noise=0.01, seed=0):
    """Paired per-trial effect with a known task-macro mean ~= `effect`."""
    rng = np.random.default_rng(seed)
    subjects, texts, tasks, eff = [], [], [], []
    for s in range(n_subj):
        for t in range(n_text):
            for _ in range(per):
                subjects.append(f"S{s}")
                texts.append(f"T{t}")
                tasks.append("NR" if t % 2 == 0 else "TSR")
                eff.append(effect + rng.normal(0, noise))
    return np.array(eff), subjects, texts, tasks


class BootstrapTests(unittest.TestCase):
    def test_reproducible_and_clusters_supported(self):
        eff, subj, text, tasks = synthetic()
        for cluster in ("subject_id", "normalized_text_sha256", "two_way_subject_by_text"):
            a = paired_bootstrap(eff, subj, text, cluster, 200, 17, tasks)
            b = paired_bootstrap(eff, subj, text, cluster, 200, 17, tasks)
            self.assertTrue(np.array_equal(a, b))
        with self.assertRaises(ValueError):
            paired_bootstrap(eff, subj, text, "bad_cluster", 10, 0, tasks)

    def test_point_and_lower_bound_track_effect(self):
        eff, subj, text, tasks = synthetic(effect=0.03, noise=0.005, seed=1)
        d = clustered_delta(eff, subj, text, tasks, replicates=400, seed=3)
        self.assertAlmostEqual(d["point"], task_macro_mean(eff, tasks), places=9)
        self.assertLess(d["two_way_lower_bound"], d["point"])          # lower bound below point
        self.assertGreater(d["two_way_lower_bound"], 0.0)              # clearly positive effect
        # two-way clustering is the widest -> its lower bound is the most conservative
        self.assertLessEqual(d["lower_bounds"]["two_way_subject_by_text"],
                             d["lower_bounds"]["subject_id"] + 1e-9)

    def test_lower_bound_percentile(self):
        draws = np.linspace(0.0, 1.0, 1001)
        self.assertAlmostEqual(lower_bound(draws, 0.05), 0.05, places=3)


class DecisionTests(unittest.TestCase):
    def _delta(self, lb, point=None):
        return {"two_way_lower_bound": lb, "point": point if point is not None else lb + 0.01}

    def _controls(self, ok=True):
        return {"matched_wrong_at_chance": ok, "shuffled_at_chance": ok,
                "mean_collapse_gap_positive": ok}

    def test_win_requires_all_five(self):
        v = apply_primary_decision(
            self._delta(0.025), self._delta(0.01), self._controls(True),
            seed_consistent=True, subject_consistent=True, pool_size_robust=True)
        self.assertEqual(v["verdict"], "maxsim_method_win")
        self.assertEqual(v["failed_conditions"], [])

    def test_margin_not_cleared_is_null(self):
        v = apply_primary_decision(
            self._delta(0.015), self._delta(0.01), self._controls(True),
            seed_consistent=True, subject_consistent=True, pool_size_robust=True)
        self.assertEqual(v["verdict"], "null_or_insufficient")
        self.assertIn("mrr_lower_bound_ge_delta_sup", v["failed_conditions"])

    def test_control_failure_blocks_win(self):
        v = apply_primary_decision(
            self._delta(0.05), self._delta(0.03),
            {"matched_wrong_at_chance": False, "shuffled_at_chance": True,
             "mean_collapse_gap_positive": True},
            seed_consistent=True, subject_consistent=True, pool_size_robust=True)
        self.assertEqual(v["verdict"], "null_or_insufficient")
        self.assertIn("controls_pass", v["failed_conditions"])

    def test_second_metric_must_also_win(self):
        v = apply_primary_decision(
            self._delta(0.03), self._delta(-0.005, point=-0.002), self._controls(True),
            seed_consistent=True, subject_consistent=True, pool_size_robust=True)
        self.assertEqual(v["verdict"], "null_or_insufficient")
        self.assertIn("both_metrics_advantage", v["failed_conditions"])

    def test_inconsistency_blocks_win(self):
        v = apply_primary_decision(
            self._delta(0.03), self._delta(0.02), self._controls(True),
            seed_consistent=False, subject_consistent=True, pool_size_robust=True)
        self.assertIn("seed_and_subject_consistent", v["failed_conditions"])

    def test_missing_control_raises(self):
        with self.assertRaises(ValueError):
            apply_primary_decision(self._delta(0.03), self._delta(0.02), {"shuffled_at_chance": True},
                                   True, True, True)

    def test_paired_delta_and_top1(self):
        rr_maxsim = [1.0, 0.5, 1.0, 0.25, 1.0, 0.5]
        rr_pooled = [0.5, 0.5, 0.5, 0.25, 1.0, 0.33]
        subj = ["S0", "S0", "S1", "S1", "S2", "S2"]
        text = ["T0", "T1", "T0", "T1", "T0", "T1"]
        tasks = ["NR", "TSR", "NR", "TSR", "NR", "TSR"]
        d = paired_delta(rr_maxsim, rr_pooled, subj, text, tasks, replicates=100, seed=0)
        self.assertGreater(d["point"], 0.0)                           # maxsim >= pooled here
        self.assertTrue(np.array_equal(top1_indicator(rr_maxsim),
                                       np.array([1, 0, 1, 0, 1, 0], dtype=float)))
        with self.assertRaises(ValueError):
            paired_delta(rr_maxsim, rr_pooled[:-1], subj, text, tasks)

    def test_delta_sup_is_locked_value(self):
        self.assertEqual(DELTA_SUP, 0.02)


class OperationalizationTests(unittest.TestCase):
    def test_seed_averaged_effect(self):
        rr = seed_averaged_effect([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [0.5, 0.5, 0.5]])
        self.assertTrue(np.allclose(rr, [0.5, 0.5, 0.5]))
        with self.assertRaises(ValueError):
            seed_averaged_effect([])

    def test_seed_consistent_requires_all_positive(self):
        self.assertTrue(seed_consistent([0.03, 0.01, 0.005]))
        self.assertFalse(seed_consistent([0.03, -0.001, 0.02]))   # one seed reverses
        with self.assertRaises(ValueError):
            seed_consistent([])

    def test_subject_consistent(self):
        self.assertTrue(subject_consistent({"two_way_lower_bound": 0.004}))
        self.assertFalse(subject_consistent({"two_way_lower_bound": -0.001}))
        self.assertFalse(subject_consistent({"two_way_lower_bound": 0.0}))

    def test_pool_size_robust(self):
        good = {24: {"two_way_lower_bound": 0.02}, 48: {"two_way_lower_bound": 0.01},
                96: {"two_way_lower_bound": 0.005}, 192: {"two_way_lower_bound": 0.001}}
        self.assertTrue(pool_size_robust(good))
        bad = dict(good); bad[192] = {"two_way_lower_bound": -0.001}  # vanishes at large pool
        self.assertFalse(pool_size_robust(bad))
        with self.assertRaises(ValueError):
            pool_size_robust({})


if __name__ == "__main__":
    unittest.main()
