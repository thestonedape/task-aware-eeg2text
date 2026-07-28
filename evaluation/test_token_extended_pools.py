"""Local tests for the deterministic extended-pool generators (no GPU).

Run: python -B -m unittest evaluation.test_token_extended_pools
"""
import unittest

from evaluation.token_extended_pools import (
    build_length_matched_pool,
    build_pool_sweep,
    catalog_task_texts,
    pools_sha256,
    shuffle_labels,
    subject_partition,
)


def make_catalog():
    # 6 NR + 6 TSR trials, distinct texts, varied lengths
    rows = []
    for task in ("NR", "TSR"):
        for j in range(6):
            rows.append({
                "trial_id": f"{task}{j}", "reading_task": task,
                "text_target_id": f"{task}txt{j}", "subject_id": f"S{j % 3}",
                "length_words_whitespace_v1": 10 + j,     # 10..15
            })
    return rows


class LengthMatchedPoolTests(unittest.TestCase):
    def test_closest_length_and_positive(self):
        task = [("t0", 10), ("t1", 11), ("t2", 9), ("t3", 15), ("t4", 30)]
        pool = build_length_matched_pool("t0", 10, task, size=4)
        self.assertEqual(len(pool), 4)
        self.assertEqual(sum(c["is_positive"] for c in pool), 1)
        picked = {c["candidate_text_target_id"] for c in pool}
        self.assertEqual(picked, {"t0", "t1", "t2", "t3"})   # 3 closest to len 10 + positive
        self.assertNotIn("t4", picked)                        # farthest excluded

    def test_deterministic_tie_break(self):
        task = [("b", 10), ("a", 12), ("c", 12), ("d", 8), ("pos", 10)]
        a = build_length_matched_pool("pos", 10, task, size=3)
        b = build_length_matched_pool("pos", 10, task, size=3)
        self.assertEqual(a, b)

    def test_rejects_insufficient_distractors(self):
        with self.assertRaises(ValueError):
            build_length_matched_pool("t0", 10, [("t0", 10), ("t1", 11)], size=24)


class SweepTests(unittest.TestCase):
    def test_sweep_sizes_and_within_task(self):
        catalog = make_catalog()
        targets = [{"trial_id": "NR0"}, {"trial_id": "TSR0"}]
        sweep = build_pool_sweep(targets, catalog, sizes=(3, 5))
        self.assertEqual(set(sweep), {3, 5})
        for size, pools in sweep.items():
            for tid, pool in pools.items():
                self.assertEqual(len(pool), size)
                self.assertEqual(sum(c["is_positive"] for c in pool), 1)
                task = "NR" if tid.startswith("NR") else "TSR"
                self.assertTrue(all(c["candidate_text_target_id"].startswith(task) for c in pool))

    def test_catalog_task_texts(self):
        tt = catalog_task_texts(make_catalog())
        self.assertEqual(set(tt), {"NR", "TSR"})
        self.assertEqual(len(tt["NR"]), 6)


class SplitAndShuffleTests(unittest.TestCase):
    def test_subject_partition_disjoint_deterministic(self):
        catalog = make_catalog()                             # subjects S0,S1,S2
        fit, held = subject_partition(catalog, held_out_fraction=0.34, seed=1)
        self.assertTrue(fit.isdisjoint(held))
        self.assertEqual(fit | held, {"S0", "S1", "S2"})
        self.assertTrue(held)
        self.assertEqual((fit, held), subject_partition(catalog, 0.34, 1))
        with self.assertRaises(ValueError):
            subject_partition(catalog, 0.0, 1)

    def test_shuffle_labels_is_derangement(self):
        targets = [{"trial_id": f"t{i}", "text_target_id": f"txt{i}"} for i in range(6)]
        shuffled = shuffle_labels(targets, seed=3)
        self.assertEqual(set(shuffled), {t["trial_id"] for t in targets})
        for t in targets:
            self.assertNotEqual(shuffled[t["trial_id"]], t["text_target_id"])  # no fixed point
        self.assertEqual(shuffled, shuffle_labels(targets, seed=3))            # deterministic

    def test_pools_sha256_stable_and_sensitive(self):
        # a picks t2 (close), b picks t3 (close) -> different membership -> different hash
        a = build_length_matched_pool("t0", 10, [("t0", 10), ("t1", 11), ("t2", 9), ("t3", 50)], size=3)
        b = build_length_matched_pool("t0", 10, [("t0", 10), ("t1", 11), ("t2", 50), ("t3", 9)], size=3)
        self.assertEqual(pools_sha256(a), pools_sha256(a))
        self.assertNotEqual({c["candidate_text_target_id"] for c in a},
                            {c["candidate_text_target_id"] for c in b})
        self.assertNotEqual(pools_sha256(a), pools_sha256(b))


if __name__ == "__main__":
    unittest.main()
