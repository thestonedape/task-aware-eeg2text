"""Local tests for the primary-pair retrieval evaluation (no GPU).

Run: python -B -m unittest evaluation.test_primary_pair_eval
"""
import unittest

import torch

from evaluation.primary_pair_eval import (
    PairTrial,
    arm_reciprocal_ranks,
    macro_mrr,
    maxsim_reciprocal_rank,
    pooled_reciprocal_rank,
    select_best_checkpoint,
)
from project_adapters.pooled_retrieval import PooledContrastiveAdapter
from project_adapters.token_late_interaction import TokenLateInteractionAdapter

D = 1024
TT = 12
C = 24


def make_trial(task="NR", subject="S0", text="T0", positive=3, aligned=True, seed=0, wrong=None):
    g = torch.Generator().manual_seed(seed)
    cand_vec = torch.randn(C, D, generator=g)
    cand_tok = torch.randn(C, TT, D, generator=g)
    # correct EEG == the positive candidate, so an identity adapter ranks it first
    eeg_vec = cand_vec[positive].clone() if aligned else torch.randn(D, generator=g)
    eeg_tok = cand_tok[positive].clone() if aligned else torch.randn(TT, D, generator=g)
    kw = {}
    if wrong is not None:  # matched-wrong donor aligned to a DIFFERENT candidate
        kw = {"wrong_eeg_vector": cand_vec[wrong].clone(),
              "wrong_eeg_tokens": cand_tok[wrong].clone()}
    return PairTrial(
        task=task, subject_id=subject, text_id=text, positive_index=positive,
        eeg_vector=eeg_vec, candidate_text_vectors=cand_vec,
        eeg_tokens=eeg_tok, candidate_text_tokens=cand_tok,
        candidate_text_mask=torch.ones(C, TT, dtype=torch.int8), **kw,
    )


class PairTrialTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            make_trial(task="SR")                       # bad task -> PairTrial rejects
        with self.assertRaises(ValueError):             # positive_index out of pool range
            PairTrial(task="NR", subject_id="S", text_id="T", positive_index=99,
                      eeg_vector=torch.randn(D), candidate_text_vectors=torch.randn(C, D),
                      eeg_tokens=torch.randn(TT, D), candidate_text_tokens=torch.randn(C, TT, D))


class ScoringTests(unittest.TestCase):
    def test_identity_adapter_ranks_aligned_positive_first(self):
        pooled = PooledContrastiveAdapter()     # zero-init residual => identity
        maxsim = TokenLateInteractionAdapter()
        trial = make_trial(positive=7, aligned=True)
        self.assertEqual(pooled_reciprocal_rank(pooled, trial), 1.0)
        self.assertEqual(maxsim_reciprocal_rank(maxsim, trial), 1.0)

    def test_matched_wrong_source_uses_donor(self):
        pooled = PooledContrastiveAdapter()
        maxsim = TokenLateInteractionAdapter()
        # correct aligns to positive=5; wrong donor aligns to candidate 12
        trial = make_trial(positive=5, aligned=True, wrong=12)
        self.assertEqual(pooled_reciprocal_rank(pooled, trial, "correct"), 1.0)
        self.assertLess(pooled_reciprocal_rank(pooled, trial, "matched_wrong"), 1.0)
        self.assertLess(maxsim_reciprocal_rank(maxsim, trial, "matched_wrong"), 1.0)

    def test_mean_collapse_source_runs(self):
        maxsim = TokenLateInteractionAdapter()
        trial = make_trial(positive=2, aligned=True)
        rr = maxsim_reciprocal_rank(maxsim, trial, "mean_collapse")
        self.assertGreaterEqual(rr, 1.0 / C)
        self.assertLessEqual(rr, 1.0)
        with self.assertRaises(ValueError):
            maxsim_reciprocal_rank(maxsim, trial, "shuffled")

    def test_pooled_rejects_missing_donor(self):
        pooled = PooledContrastiveAdapter()
        with self.assertRaises(ValueError):
            pooled_reciprocal_rank(pooled, make_trial(), "matched_wrong")


class AggregationTests(unittest.TestCase):
    def _trials(self):
        trials = []
        for i in range(6):
            trials.append(make_trial(task="NR" if i % 2 == 0 else "TSR",
                                     subject=f"S{i%3}", text=f"T{i}", positive=i, seed=i))
        return trials

    def test_labels_align_with_trials(self):
        adapter = PooledContrastiveAdapter()
        trials = self._trials()
        out = arm_reciprocal_ranks(adapter, "pooled", trials)
        self.assertEqual(out["subjects"], [t.subject_id for t in trials])
        self.assertEqual(out["texts"], [t.text_id for t in trials])
        self.assertEqual(out["tasks"], [t.task for t in trials])
        self.assertEqual(len(out["rr"]), len(trials))

    def test_macro_mrr_is_task_macro_and_high_for_aligned(self):
        adapter = PooledContrastiveAdapter()
        trials = self._trials()                 # all aligned => every RR == 1.0
        self.assertAlmostEqual(macro_mrr(adapter, "pooled", trials), 1.0, places=6)

    def test_select_best_checkpoint_picks_higher_dev_mrr(self):
        good = PooledContrastiveAdapter()       # identity => ranks aligned positives first
        bad = PooledContrastiveAdapter()
        with torch.no_grad():                   # distort the bad checkpoint's projection
            bad.shared.up.weight.copy_(torch.randn_like(bad.shared.up.weight) * 5.0)
        trials = self._trials()
        self.assertEqual(select_best_checkpoint([bad, good], "pooled", trials), 1)
        self.assertEqual(select_best_checkpoint([good, bad], "pooled", trials), 0)
        with self.assertRaises(ValueError):
            select_best_checkpoint([], "pooled", trials)


if __name__ == "__main__":
    unittest.main()
