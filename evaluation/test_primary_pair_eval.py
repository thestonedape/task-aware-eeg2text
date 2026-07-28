"""Local tests for the primary-pair retrieval evaluation (no GPU).

Candidate pools are text ids resolved from a shared text_lookup cache (no per-trial
tensor duplication). Run: python -B -m unittest evaluation.test_primary_pair_eval
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


def make_world(seed=0):
    """A shared text_lookup of C candidate texts + a factory for trials over it."""
    g = torch.Generator().manual_seed(seed)
    ids = [f"c{c}" for c in range(C)]
    text_lookup = {
        cid: {"vector": torch.randn(D, generator=g),
              "tokens": torch.randn(TT, D, generator=g),
              "mask": torch.ones(TT, dtype=torch.int8)}
        for cid in ids
    }

    def make_trial(task="NR", subject="S0", text_id="T0", positive=3, aligned=True, wrong=None):
        pos = ids[positive]
        eeg_vec = text_lookup[pos]["vector"].clone() if aligned else torch.randn(D, generator=g)
        eeg_tok = text_lookup[pos]["tokens"].clone() if aligned else torch.randn(TT, D, generator=g)
        kw = {}
        if wrong is not None:
            wid = ids[wrong]
            kw = {"wrong_eeg_vector": text_lookup[wid]["vector"].clone(),
                  "wrong_eeg_tokens": text_lookup[wid]["tokens"].clone()}
        return PairTrial(task=task, subject_id=subject, text_id=text_id, positive_index=positive,
                         eeg_vector=eeg_vec, eeg_tokens=eeg_tok, candidate_text_ids=list(ids), **kw)

    return text_lookup, make_trial


class PairTrialTests(unittest.TestCase):
    def test_validation(self):
        _, make_trial = make_world()
        with self.assertRaises(ValueError):
            make_trial(task="SR")
        with self.assertRaises(ValueError):
            PairTrial(task="NR", subject_id="S", text_id="T", positive_index=99,
                      eeg_vector=torch.randn(D), eeg_tokens=torch.randn(TT, D),
                      candidate_text_ids=[f"c{i}" for i in range(C)])


class ScoringTests(unittest.TestCase):
    def test_identity_adapter_ranks_aligned_positive_first(self):
        text, make_trial = make_world()
        pooled, maxsim = PooledContrastiveAdapter(), TokenLateInteractionAdapter()
        trial = make_trial(positive=7, aligned=True)
        self.assertEqual(pooled_reciprocal_rank(pooled, trial, text), 1.0)
        self.assertEqual(maxsim_reciprocal_rank(maxsim, trial, text), 1.0)

    def test_matched_wrong_source_uses_donor(self):
        text, make_trial = make_world()
        pooled, maxsim = PooledContrastiveAdapter(), TokenLateInteractionAdapter()
        trial = make_trial(positive=5, aligned=True, wrong=12)
        self.assertEqual(pooled_reciprocal_rank(pooled, trial, text, "correct"), 1.0)
        self.assertLess(pooled_reciprocal_rank(pooled, trial, text, "matched_wrong"), 1.0)
        self.assertLess(maxsim_reciprocal_rank(maxsim, trial, text, "matched_wrong"), 1.0)

    def test_mean_collapse_source_runs(self):
        text, make_trial = make_world()
        maxsim = TokenLateInteractionAdapter()
        rr = maxsim_reciprocal_rank(maxsim, make_trial(positive=2), text, "mean_collapse")
        self.assertGreaterEqual(rr, 1.0 / C)
        self.assertLessEqual(rr, 1.0)
        with self.assertRaises(ValueError):
            maxsim_reciprocal_rank(maxsim, make_trial(), text, "shuffled")

    def test_pooled_rejects_missing_donor(self):
        text, make_trial = make_world()
        with self.assertRaises(ValueError):
            pooled_reciprocal_rank(PooledContrastiveAdapter(), make_trial(), text, "matched_wrong")


class AggregationTests(unittest.TestCase):
    def _world(self):
        text, make_trial = make_world()
        trials = [make_trial(task="NR" if i % 2 == 0 else "TSR", subject=f"S{i%3}",
                             text_id=f"T{i}", positive=i) for i in range(6)]
        return text, trials

    def test_labels_align_with_trials(self):
        text, trials = self._world()
        out = arm_reciprocal_ranks(PooledContrastiveAdapter(), "pooled", trials, text)
        self.assertEqual(out["subjects"], [t.subject_id for t in trials])
        self.assertEqual(out["tasks"], [t.task for t in trials])
        self.assertEqual(len(out["rr"]), len(trials))

    def test_macro_mrr_high_for_aligned(self):
        text, trials = self._world()
        self.assertAlmostEqual(macro_mrr(PooledContrastiveAdapter(), "pooled", trials, text), 1.0, places=6)

    def test_select_best_checkpoint_picks_higher_dev_mrr(self):
        text, trials = self._world()
        good = PooledContrastiveAdapter()
        bad = PooledContrastiveAdapter()
        with torch.no_grad():
            bad.shared.up.weight.copy_(torch.randn_like(bad.shared.up.weight) * 5.0)
        self.assertEqual(select_best_checkpoint([bad, good], "pooled", trials, text), 1)
        self.assertEqual(select_best_checkpoint([good, bad], "pooled", trials, text), 0)
        with self.assertRaises(ValueError):
            select_best_checkpoint([], "pooled", trials, text)


if __name__ == "__main__":
    unittest.main()
