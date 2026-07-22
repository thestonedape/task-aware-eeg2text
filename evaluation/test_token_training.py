"""Local tests for the arm-symmetric contrastive trainer (no GPU).

Verifies the properties the primary causal pair depends on: the batch schedule is
deterministic and arm-independent, both arms initialize identically from a seed,
and each arm actually learns (loss decreases) on a learnable synthetic task.

Run: python -B -m unittest evaluation.test_token_training
"""
import unittest

import torch

from evaluation.token_training import TrainConfig, deterministic_batches, train_arm
from project_adapters.pooled_retrieval import PooledContrastiveAdapter
from project_adapters.token_late_interaction import TokenLateInteractionAdapter

DIM = 1024
T = 32
LATENT = 16


def shared_latent_task(n, seed, tokens=None):
    """EEG and text share a low-dim latent through DIFFERENT linear maps, so the
    zero-init (identity) adapter is misaligned (high loss) but a rank-96 residual
    can learn the alignment. Returns (eeg, text)."""
    g = torch.Generator().manual_seed(seed)
    shape = (n, LATENT) if tokens is None else (n, tokens, LATENT)
    z = torch.randn(*shape, generator=g)
    A = torch.randn(LATENT, DIM, generator=g)
    B = torch.randn(LATENT, DIM, generator=g)
    return z @ B, z @ A                                     # eeg = z@B, text = z@A


class BatchScheduleTests(unittest.TestCase):
    def test_deterministic_and_arm_independent(self):
        a = deterministic_batches(200, 64, 3, seed=7)
        b = deterministic_batches(200, 64, 3, seed=7)
        self.assertEqual(a, b)                                   # reproducible
        self.assertNotEqual(a, deterministic_batches(200, 64, 3, seed=8))
        self.assertTrue(all(len(batch) == 64 for batch in a))   # full batches only
        self.assertEqual(len(a), 3 * (200 // 64))               # remainder dropped
        # a full epoch's batches partition distinct trials (no repeats within epoch)
        first_epoch = [i for batch in a[:200 // 64] for i in batch]
        self.assertEqual(len(first_epoch), len(set(first_epoch)))

    def test_requires_full_batch(self):
        with self.assertRaises(ValueError):
            deterministic_batches(10, 64, 1, seed=0)

    def test_text_unique_batches_have_no_repeated_texts(self):
        # 300 trials over 60 texts (5 repeats each) -> naive batches would collide
        text_ids = [f"T{i % 60}" for i in range(300)]
        batches = deterministic_batches(300, 32, epochs=2, seed=3, text_ids=text_ids)
        self.assertTrue(batches)
        for batch in batches:
            self.assertEqual(len(batch), 32)                       # full batches
            texts = [text_ids[i] for i in batch]
            self.assertEqual(len(texts), len(set(texts)))          # all distinct
        # deterministic
        again = deterministic_batches(300, 32, epochs=2, seed=3, text_ids=text_ids)
        self.assertEqual(batches, again)


class TrainerTests(unittest.TestCase):
    def test_both_arms_initialize_identically(self):
        torch.manual_seed(123)
        pooled = PooledContrastiveAdapter()
        torch.manual_seed(123)
        token = TokenLateInteractionAdapter()
        for p, t in zip(pooled.shared.parameters(), token.shared.parameters()):
            self.assertTrue(torch.equal(p, t))

    def test_pooled_arm_learns(self):
        eeg, text = shared_latent_task(64, seed=1)
        features = {"eeg_vectors": eeg, "text_vectors": text}
        batches = deterministic_batches(64, 32, epochs=40, seed=3)
        _, trace = train_arm("pooled", features, batches, TrainConfig(lr=1e-2), seed=5)
        self.assertGreater(sum(trace[:5]) / 5, 0.1)             # non-trivial start
        self.assertLess(sum(trace[-5:]) / 5, 0.7 * (sum(trace[:5]) / 5))

    def test_maxsim_arm_learns(self):
        eeg, text = shared_latent_task(48, seed=2, tokens=T)
        features = {
            "eeg_tokens": eeg, "text_tokens": text,
            "text_masks": torch.ones(48, T, dtype=torch.int8),
        }
        batches = deterministic_batches(48, 24, epochs=40, seed=4)
        _, trace = train_arm("maxsim", features, batches, TrainConfig(lr=1e-2), seed=5)
        self.assertGreater(sum(trace[:5]) / 5, 0.1)             # non-trivial start
        self.assertLess(sum(trace[-5:]) / 5, 0.7 * (sum(trace[:5]) / 5))

    def test_missing_features_rejected(self):
        batches = deterministic_batches(64, 32, 1, seed=0)
        with self.assertRaises(ValueError):
            train_arm("maxsim", {"eeg_vectors": torch.randn(64, DIM)}, batches, TrainConfig(), seed=0)

    def test_trained_adapter_keeps_capacity(self):
        text = torch.randn(64, DIM)
        features = {"eeg_vectors": text.clone(), "text_vectors": text}
        batches = deterministic_batches(64, 32, epochs=2, seed=1)
        adapter, _ = train_arm("pooled", features, batches, TrainConfig(), seed=0)
        self.assertEqual(sum(p.numel() for p in adapter.parameters() if p.requires_grad), 196_608)


if __name__ == "__main__":
    unittest.main()
