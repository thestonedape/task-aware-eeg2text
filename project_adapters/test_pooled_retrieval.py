"""Local tests for the pooled arm of the primary causal pair (no GPU).

The load-bearing test is capacity parity: the pooled adapter must have the exact
same trainable-parameter count as the token (MaxSim) adapter, so any retrieval Δ
between the arms cannot be attributed to capacity.

Run: python -B -m unittest project_adapters.test_pooled_retrieval
"""
import unittest

import torch

from project_adapters.pooled_retrieval import (
    P4B_TRAINABLE_PARAMETERS,
    PooledContrastiveAdapter,
    pooled_reciprocal_rank_of_positive,
    pooled_similarity_matrix,
    symmetric_pooled_loss,
)
from project_adapters.token_late_interaction import TokenLateInteractionAdapter

DIM = 1024


class PooledArmTests(unittest.TestCase):
    def test_capacity_matches_token_arm_exactly(self):
        pooled = PooledContrastiveAdapter()
        token = TokenLateInteractionAdapter()
        self.assertEqual(pooled.trainable_parameter_count, P4B_TRAINABLE_PARAMETERS)
        self.assertEqual(pooled.trainable_parameter_count, token.trainable_parameter_count)
        self.assertEqual(pooled.trainable_parameter_count, 196_608)

    def test_rejects_nonmatching_config(self):
        with self.assertRaises(ValueError):
            PooledContrastiveAdapter(vector_dim=512, rank=96)
        with self.assertRaises(ValueError):
            PooledContrastiveAdapter(vector_dim=1024, rank=32)

    def test_project_shape_and_residual(self):
        adapter = PooledContrastiveAdapter()
        x = torch.randn(8, DIM)
        out = adapter.project(x)
        self.assertEqual(out.shape, (8, DIM))
        # zero-init residual (LowRankResidual starts at zero) => identity at init
        self.assertTrue(torch.allclose(out, x, atol=1e-6))
        with self.assertRaises(ValueError):
            adapter.project(torch.randn(8, DIM, 1))  # wrong ndim

    def test_similarity_is_cosine(self):
        e = torch.randn(4, DIM)
        t = torch.randn(6, DIM)
        sim = pooled_similarity_matrix(e, t)
        self.assertEqual(sim.shape, (4, 6))
        en = e / e.norm(dim=-1, keepdim=True)
        tn = t / t.norm(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(sim, en @ tn.t(), atol=1e-5))
        self.assertLessEqual(float(sim.abs().max()), 1.0 + 1e-5)

    def test_symmetric_loss_low_when_aligned(self):
        v = torch.randn(16, DIM)
        aligned = symmetric_pooled_loss(v, v.clone())               # positives = diagonal
        g = torch.Generator().manual_seed(0)
        misaligned = symmetric_pooled_loss(v, torch.randn(16, DIM, generator=g))
        self.assertLess(float(aligned), float(misaligned))
        with self.assertRaises(ValueError):
            symmetric_pooled_loss(v, v, temperature=0.0)
        with self.assertRaises(ValueError):
            symmetric_pooled_loss(v, torch.randn(15, DIM))          # non-square

    def test_reciprocal_rank_ranks_positive(self):
        pool = torch.randn(24, DIM)
        pos = 7
        # a query identical to the positive candidate must rank it first
        rr = pooled_reciprocal_rank_of_positive(pool[pos], pool, pos)
        self.assertEqual(rr, 1.0)
        # a query orthogonal-ish to its "positive" should not always rank 1
        rr2 = pooled_reciprocal_rank_of_positive(torch.randn(DIM), pool, pos)
        self.assertGreaterEqual(rr2, 1.0 / 24)
        self.assertLessEqual(rr2, 1.0)


if __name__ == "__main__":
    unittest.main()
