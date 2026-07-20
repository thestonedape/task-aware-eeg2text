"""End-to-end synthetic test of the token retrieval evaluation path.

Builds trials with a hidden per-trial direction shared by the EEG tokens and the
positive text, random distractors, and a matched-wrong donor that is another
trial's EEG. Verifies the measurement path produces the paper's endpoints with
the right qualitative behaviour: correct EEG well above 24-way chance, matched-
wrong near chance, and a positive signal gap. This validates the scoring ->
pooling -> macro-MRR -> controls pipeline before any real tokens exist.

Run: python -B -m unittest evaluation.test_token_retrieval_eval
"""
import unittest

import torch

from project_adapters.token_late_interaction import TokenLateInteractionAdapter, l2_normalize
from evaluation.token_retrieval_eval import TokenTrial, evaluate, macro_mrr

CHANCE_MRR_24 = 0.157331591  # H_24 / 24


def build_trials(num_per_task=40, pool=24, te=6, tt=5, dim=32, noise=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    directions = l2_normalize(torch.randn(2 * num_per_task, dim, generator=g))
    trials = []
    idx = 0
    for task in ("NR", "TSR"):
        for _ in range(num_per_task):
            h = directions[idx]
            eeg = l2_normalize(h[None, :] + noise * torch.randn(te, dim, generator=g), dim=-1)
            # positive text shares the hidden direction; distractors are random.
            cand = l2_normalize(torch.randn(pool, tt, dim, generator=g), dim=-1)
            pos = int(torch.randint(0, pool, (1,), generator=g).item())
            cand[pos] = l2_normalize(h[None, :] + noise * torch.randn(tt, dim, generator=g), dim=-1)
            # matched-wrong donor: a different trial's hidden direction.
            donor = directions[(idx + 1) % directions.shape[0]]
            wrong = l2_normalize(donor[None, :] + noise * torch.randn(te, dim, generator=g), dim=-1)
            trials.append(TokenTrial(
                task=task, positive_index=pos,
                eeg_tokens=eeg, candidate_text_tokens=cand,
                wrong_eeg_tokens=wrong,
            ))
            idx += 1
    return trials


class TokenRetrievalEvalTests(unittest.TestCase):
    def _identity_model(self, dim):
        # zero-init adapter => project is identity; isolates the eval path.
        model = TokenLateInteractionAdapter.__new__(TokenLateInteractionAdapter)
        torch.nn.Module.__init__(model)
        model.vector_dim, model.rank = dim, 4
        from project_adapters.task_treatment_pilots import LowRankResidual
        model.shared = LowRankResidual(dim, 4)
        return model

    def test_macro_mrr_averages_over_tasks(self):
        # NR mean 0.5, TSR mean 0.1 -> macro 0.3 (not sample-weighted).
        rr = {"NR": [0.5, 0.5, 0.5], "TSR": [0.1]}
        self.assertAlmostEqual(macro_mrr(rr), 0.3, places=6)

    def test_correct_above_chance_wrong_at_chance(self):
        dim = 32
        trials = build_trials(dim=dim, seed=1)
        model = self._identity_model(dim)
        result = evaluate(model, trials)

        correct = result["macro_mrr"]["correct"]
        wrong = result["macro_mrr"]["matched_wrong"]
        # correct retrieval is well above 24-way chance...
        self.assertGreater(correct, 0.5)
        # ...matched-wrong EEG collapses toward chance...
        self.assertLess(wrong, 0.25)
        # ...and the signal gap is clearly positive.
        self.assertGreater(result["signal_gap"], 0.3)
        self.assertAlmostEqual(result["signal_gap"], correct - wrong, places=6)

    def test_all_endpoints_present_and_shaped(self):
        dim = 24
        trials = build_trials(num_per_task=12, dim=dim, seed=2)
        result = evaluate(model=self._identity_model(dim), trials=trials)
        self.assertEqual(result["num_trials"], 24)
        for source in ("correct", "matched_wrong", "mean_collapse"):
            self.assertIn(source, result["macro_mrr"])
            self.assertTrue(0.0 <= result["macro_mrr"][source] <= 1.0)
            for task in ("NR", "TSR"):
                self.assertIsNotNone(result["per_task_mrr"][source][task])
        self.assertIn("token_vs_mean_collapse", result)

    def test_evaluate_without_wrong_donor_omits_gap(self):
        dim = 16
        trials = build_trials(num_per_task=8, dim=dim, seed=3)
        for t in trials:
            t.wrong_eeg_tokens = None
        result = evaluate(self._identity_model(dim), trials)
        self.assertNotIn("signal_gap", result)
        self.assertNotIn("matched_wrong", result["macro_mrr"])
        self.assertIn("correct", result["macro_mrr"])


if __name__ == "__main__":
    unittest.main()
