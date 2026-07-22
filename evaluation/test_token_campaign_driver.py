"""End-to-end smoke test for the campaign driver (no GPU).

Builds a small synthetic multi-fold contract with fit/checkpoint/confirmation
roles and drives run_campaign -> aggregate_primary, checking the stitch produces
well-formed primary decision inputs. Value correctness of the underlying pieces is
covered by their own tests; this verifies the orchestration and aggregation.

Run: python -B -m unittest evaluation.test_token_campaign_driver
"""
import unittest

import torch

from evaluation.token_campaign_driver import aggregate_primary, run_campaign
from evaluation.token_training import TrainConfig

D, TT, N_TEXT, POOL = 1024, 8, 40, 6


def build_world(seed=0):
    g = torch.Generator().manual_seed(seed)
    text_ids = [f"txt{j:03d}" for j in range(N_TEXT)]
    text_lookup = {
        t: {"tokens": torch.randn(TT, D, generator=g),
            "mask": torch.ones(TT, dtype=torch.int8),
            "vector": torch.randn(D, generator=g)}
        for t in text_ids
    }
    eeg_lookup, assignments, pools, donors = {}, [], {}, {}
    counter = {"n": 0}
    ti = {"n": 0}

    def add(fold, role, task, subj, text_id):
        tid = f"tr{counter['n']}"; counter["n"] += 1
        eeg_lookup[tid] = {"tokens": text_lookup[text_id]["tokens"].clone(),
                           "vector": text_lookup[text_id]["vector"].clone()}   # aligned
        assignments.append({"outer_fold": fold, "role": role, "trial_id": tid,
                            "reading_task": task, "subject_id": subj, "text_target_id": text_id})
        return tid

    for fold in ("0", "1"):
        fit_ids = [add(fold, "fit", "NR" if k % 2 == 0 else "TSR", f"S{k % 2}",
                       text_ids[ti["n"] % N_TEXT]) or ti.__setitem__("n", ti["n"] + 1)
                   for k in range(4)]
        first_fit = [a["trial_id"] for a in assignments if a["outer_fold"] == fold and a["role"] == "fit"][0]
        for role in ("checkpoint", "confirmation"):
            for k in range(2):                                   # one NR, one TSR per fold/role
                pos = text_ids[ti["n"] % N_TEXT]; ti["n"] += 1
                tid = add(fold, role, "NR" if k == 0 else "TSR", f"S{k}", pos)
                distractors = [t for t in text_ids if t != pos][:POOL - 1]
                ordered = distractors[:POOL // 2] + [pos] + distractors[POOL // 2:]
                pools[(fold, role, tid)] = [
                    {"candidate_text_target_id": c, "is_positive": (c == pos)} for c in ordered
                ]
                if role == "confirmation":
                    donors[(fold, tid)] = first_fit             # a real trial's EEG as donor
    return assignments, pools, donors, eeg_lookup, text_lookup


class DriverTests(unittest.TestCase):
    def test_run_and_aggregate(self):
        assignments, pools, donors, eeg, text = build_world()
        config = TrainConfig(epochs=3, batch_size=2, lr=1e-2)
        seeds = [0, 1]
        records, labels = run_campaign(
            outer_folds=["0", "1"], seeds=seeds, assignments=assignments,
            candidate_pools=pools, donors=donors, eeg_lookup=eeg, text_lookup=text,
            config=config, select_every=2, pool_size=POOL,
        )
        # 2 folds x 2 confirmation trials = 4 evaluated trials
        self.assertEqual(len(labels), 4)
        self.assertEqual(set(records), {"pooled", "maxsim"})
        for arm in ("pooled", "maxsim"):
            self.assertIn("correct", records[arm][0])

        out = aggregate_primary(records, labels, seeds, replicates=100, seed=1)
        self.assertEqual(out["n_trials"], 4)
        for key in ("point", "two_way_lower_bound"):
            self.assertIn(key, out["mrr_delta"])
            self.assertIn(key, out["top1_delta"])
        self.assertIn(out["seed_consistent"], (True, False))
        self.assertEqual(len(out["per_seed_delta"]), 2)
        self.assertEqual(set(out["controls"]),
                         {"mean_collapse_gap_positive", "matched_wrong_at_chance"})
        self.assertIn("maxsim_correct_macro_mrr", out["summary"])


if __name__ == "__main__":
    unittest.main()
