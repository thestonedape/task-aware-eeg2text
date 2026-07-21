"""Local tests for the Gate-2 pooled-vector identity comparator (no GPU/IO)."""
import unittest

import numpy as np

from evaluation.pooled_vector_identity import align_by_id, compare_pooled_vectors

RNG = np.random.default_rng(0)


def make(n=64, d=1024):
    ids = [f"t{i:04d}" for i in range(n)]
    vecs = RNG.standard_normal((n, d)).astype(np.float32) * 5.0
    return ids, vecs


class PooledVectorIdentityTests(unittest.TestCase):
    def test_float16_roundtrip_matches(self):
        ids, vecs = make()
        half = vecs.astype(np.float16).astype(np.float32)  # simulate co-extracted float16
        report = compare_pooled_vectors(ids, half, ids, vecs)
        self.assertTrue(report["match"], report)
        self.assertEqual(report["shared_count"], len(ids))
        self.assertGreater(report["min_per_row_cosine"], 0.9995)

    def test_detects_mismatch(self):
        ids, vecs = make()
        corrupted = vecs.copy()
        corrupted[3] += 1.0  # a whole row perturbed -> should fail
        report = compare_pooled_vectors(ids, corrupted, ids, vecs)
        self.assertFalse(report["match"], report)

    def test_detects_prompt_mode_style_divergence(self):
        # A systematic offset (e.g. task-prompted vs all_masked) breaks the match.
        ids, vecs = make()
        shifted = vecs + 0.5
        self.assertFalse(compare_pooled_vectors(ids, shifted, ids, vecs)["match"])

    def test_aligns_on_intersection_and_reorders(self):
        ids, vecs = make(n=10)
        # side B is a shuffled subset; alignment must pair by id, not by position
        order = list(range(10))
        RNG.shuffle(order)
        ids_b = [ids[i] for i in order][:7]
        vecs_b = np.stack([vecs[ids.index(t)] for t in ids_b])
        a, b, shared = align_by_id(ids, vecs, ids_b, vecs_b)
        self.assertEqual(len(shared), 7)
        for row, tid in zip(a, [t for t in ids if t in set(ids_b)]):
            np.testing.assert_allclose(row, vecs[ids.index(tid)])
        report = compare_pooled_vectors(ids, vecs, ids_b, vecs_b)
        self.assertTrue(report["match"])
        self.assertEqual(report["shared_count"], 7)

    def test_rejects_duplicates_and_empty_intersection(self):
        ids, vecs = make(n=4)
        with self.assertRaises(ValueError):
            align_by_id(["x", "x", "y", "z"], vecs, ids, vecs)
        with self.assertRaises(ValueError):
            align_by_id(ids, vecs, ["p", "q", "r", "s"], vecs)


if __name__ == "__main__":
    unittest.main()
