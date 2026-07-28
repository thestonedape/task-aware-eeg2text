"""Local tests for the campaign assembly join (no GPU).

Run: python -B -m unittest evaluation.test_token_campaign
"""
import unittest

import torch

from evaluation.primary_pair_eval import (
    MAXSIM_SOURCES,
    POOLED_SOURCES,
    maxsim_reciprocal_rank,
    pooled_reciprocal_rank,
)
from evaluation.token_campaign import assemble_pair_trials, train_and_confirm
from evaluation.token_training import TrainConfig
from project_adapters.pooled_retrieval import PooledContrastiveAdapter
from project_adapters.token_late_interaction import TokenLateInteractionAdapter

D = 1024
TT = 8
N_TEXT = 40


def build_world(seed=0):
    """Synthetic contracts + representations: 40 texts, a few trials, 24-way pools."""
    g = torch.Generator().manual_seed(seed)
    text_lookup = {
        f"txt{j:03d}": {
            "tokens": torch.randn(TT, D, generator=g),
            "mask": torch.ones(TT, dtype=torch.int8),
            "vector": torch.randn(D, generator=g),
        }
        for j in range(N_TEXT)
    }
    # trials t0..t5, each mapped to its own text; EEG aligned to that text so an
    # identity adapter ranks the positive first.
    eeg_lookup, targets, pools, donors = {}, [], {}, {}
    for i in range(6):
        tid, pos_text = f"t{i}", f"txt{i:03d}"
        eeg_lookup[tid] = {
            "tokens": text_lookup[pos_text]["tokens"].clone(),   # aligned -> RR 1.0
            "vector": text_lookup[pos_text]["vector"].clone(),
        }
        targets.append({"trial_id": tid, "reading_task": "NR" if i % 2 == 0 else "TSR",
                        "subject_id": f"S{i%3}", "text_target_id": pos_text})
        # 24-way pool: positive at a non-trivial rank, filled with distinct distractors
        distractors = [f"txt{j:03d}" for j in range(6, 6 + 23)]
        ordered = distractors[:5] + [pos_text] + distractors[5:]        # positive at rank 5
        pools[tid] = [
            {"candidate_text_target_id": c, "is_positive": (c == pos_text)}
            for c in ordered
        ]
        donors[tid] = f"t{(i + 1) % 6}"                                  # a different trial's EEG
    return targets, pools, donors, eeg_lookup, text_lookup


class AssemblyTests(unittest.TestCase):
    def test_positive_index_and_pool_order(self):
        targets, pools, donors, eeg, text = build_world()
        trials = assemble_pair_trials(targets, pools, donors, eeg, text, require_donor=True)
        self.assertEqual(len(trials), 6)
        for trial in trials:
            self.assertEqual(trial.positive_index, 5)                    # placed at rank 5
            self.assertEqual(len(trial.candidate_text_ids), 24)
            self.assertTrue(all(cid in text for cid in trial.candidate_text_ids))
            self.assertIsNotNone(trial.wrong_eeg_vector)

    def test_aligned_positive_ranks_first_both_arms(self):
        targets, pools, donors, eeg, text = build_world()
        trials = assemble_pair_trials(targets, pools, donors, eeg, text, require_donor=True)
        pooled, maxsim = PooledContrastiveAdapter(), TokenLateInteractionAdapter()
        for trial in trials:
            self.assertEqual(pooled_reciprocal_rank(pooled, trial, text), 1.0)
            self.assertEqual(maxsim_reciprocal_rank(maxsim, trial, text), 1.0)
            # matched-wrong donor is a different trial -> should not rank the positive first
            self.assertLess(pooled_reciprocal_rank(pooled, trial, text, "matched_wrong"), 1.0)

    def test_checkpoint_partition_needs_no_donor(self):
        targets, pools, donors, eeg, text = build_world()
        trials = assemble_pair_trials(targets, pools, {}, eeg, text, require_donor=False)
        self.assertTrue(all(t.wrong_eeg_vector is None for t in trials))

    def test_rejects_missing_donor_when_required(self):
        targets, pools, donors, eeg, text = build_world()
        with self.assertRaises(KeyError):
            assemble_pair_trials(targets, pools, {}, eeg, text, require_donor=True)

    def test_rejects_wrong_pool_size(self):
        targets, pools, donors, eeg, text = build_world()
        pools[targets[0]["trial_id"]] = pools[targets[0]["trial_id"]][:23]
        with self.assertRaises(ValueError):
            assemble_pair_trials(targets, pools, donors, eeg, text, require_donor=True)

    def test_rejects_multiple_or_zero_positives(self):
        targets, pools, donors, eeg, text = build_world()
        tid = targets[0]["trial_id"]
        for c in pools[tid]:
            c["is_positive"] = False                                    # zero positives
        with self.assertRaises(ValueError):
            assemble_pair_trials(targets, pools, donors, eeg, text, require_donor=True)

    def test_string_bool_positive_flag(self):
        # CSV-loaded flags arrive as strings; "True"/"false" must be honored
        targets, pools, donors, eeg, text = build_world()
        for tid, pool in pools.items():
            for c in pool:
                c["is_positive"] = "True" if c["is_positive"] else "False"
        trials = assemble_pair_trials(targets, pools, donors, eeg, text, require_donor=True)
        self.assertTrue(all(t.positive_index == 5 for t in trials))


class OrchestrationTests(unittest.TestCase):
    def _fit_features(self, targets, eeg, text):
        fit_text_ids = [t["text_target_id"] for t in targets]
        pooled = {
            "eeg_vectors": torch.stack([eeg[t["trial_id"]]["vector"] for t in targets]),
            "text_vectors": torch.stack([text[t["text_target_id"]]["vector"] for t in targets]),
        }
        maxsim = {
            "eeg_tokens": torch.stack([eeg[t["trial_id"]]["tokens"] for t in targets]),
            "text_tokens": torch.stack([text[t["text_target_id"]]["tokens"] for t in targets]),
            "text_masks": torch.stack([text[t["text_target_id"]]["mask"] for t in targets]),
        }
        return fit_text_ids, pooled, maxsim

    def test_train_and_confirm_runs_both_arms_with_checkpoint_selection(self):
        targets, pools, donors, eeg, text = build_world()
        confirmation = assemble_pair_trials(targets, pools, donors, eeg, text, require_donor=True)
        checkpoint = assemble_pair_trials(targets, pools, {}, eeg, text, require_donor=False)
        fit_text_ids, pooled_feats, maxsim_feats = self._fit_features(targets, eeg, text)
        config = TrainConfig(epochs=4, batch_size=3, lr=1e-2)

        for arm, feats in (("pooled", pooled_feats), ("maxsim", maxsim_feats)):
            out = train_and_confirm(
                arm, feats, fit_text_ids, checkpoint, confirmation, text, config, seed=0, select_every=2,
            )
            expected_sources = MAXSIM_SOURCES if arm == "maxsim" else POOLED_SOURCES
            self.assertEqual(set(out), set(expected_sources))
            for source in expected_sources:
                rr = out[source]["rr"]
                self.assertEqual(len(rr), len(confirmation))
                self.assertTrue(all(1.0 / 24 <= x <= 1.0 for x in rr))
            # labels come back aligned to the confirmation trials
            self.assertEqual(out["correct"]["tasks"], [t.task for t in confirmation])
            self.assertEqual(out["correct"]["subjects"], [t.subject_id for t in confirmation])


if __name__ == "__main__":
    unittest.main()
