"""Local tests for the MaxSim convergence/sensitivity diagnostic (no GPU).

Run: python -B -m unittest evaluation.test_maxsim_convergence_diag
"""
import unittest

import torch

from evaluation.maxsim_convergence_diag import prepare_fold, run_convergence_diagnostic
from evaluation.token_training import TrainConfig

D = 1024
TT = 6
N_TEXT = 12
POOL = 8


def build_world(seed=0):
    """Synthetic P4b-shaped contracts for one fold: fit/checkpoint/confirmation
    assignments, 8-way pools keyed (fold, partition, trial), donor map, and aligned
    EEG so an identity adapter ranks the positive first."""
    g = torch.Generator().manual_seed(seed)
    text_lookup = {
        f"txt{j:03d}": {
            "tokens": torch.randn(TT, D, generator=g),
            "mask": torch.ones(TT, dtype=torch.int8),
            "vector": torch.randn(D, generator=g),
        }
        for j in range(N_TEXT)
    }
    eeg_lookup, assignments = {}, []
    candidate_pools, donors = {}, {}

    def add(trial_id, text_id, task, role):
        eeg_lookup[trial_id] = {
            "tokens": text_lookup[text_id]["tokens"].clone(),
            "vector": text_lookup[text_id]["vector"].clone(),
        }
        assignments.append({
            "outer_fold": "0", "role": role, "trial_id": trial_id,
            "reading_task": task, "subject_id": f"S{len(assignments) % 3}",
            "text_target_id": text_id,
        })

    def pool_for(text_id):
        distract = [f"txt{j:03d}" for j in range(N_TEXT) if f"txt{j:03d}" != text_id][:POOL - 1]
        ordered = distract[:3] + [text_id] + distract[3:]          # positive at rank 3
        return [{"candidate_rank": r,
                 "candidate_text_target_id": c,
                 "is_positive": "True" if c == text_id else "False"}
                for r, c in enumerate(ordered)]

    for i in range(8):                                             # fit trials (>= batch_size)
        add(f"f{i}", f"txt{i:03d}", "NR" if i % 2 == 0 else "TSR", "fit")
    for i in range(4):                                             # checkpoint pools (no donor)
        tid, text_id = f"k{i}", f"txt{i:03d}"
        add(tid, text_id, "NR" if i % 2 == 0 else "TSR", "checkpoint")
        candidate_pools[("0", "checkpoint", tid)] = pool_for(text_id)
    conf_ids = []
    for i in range(4):                                            # confirmation pools + donors
        tid, text_id = f"c{i}", f"txt{i:03d}"
        add(tid, text_id, "NR" if i % 2 == 0 else "TSR", "confirmation")
        candidate_pools[("0", "confirmation", tid)] = pool_for(text_id)
        conf_ids.append(tid)
    for i, tid in enumerate(conf_ids):
        donors[("0", tid)] = conf_ids[(i + 1) % len(conf_ids)]   # a different confirmation trial
    return assignments, candidate_pools, donors, eeg_lookup, text_lookup


class PrepareFoldTests(unittest.TestCase):
    def test_partitions_and_pool_sizes(self):
        assignments, pools, donors, eeg, text = build_world()
        confirmation, checkpoint, maxsim_feats, fit_text_ids, pooled_feats = prepare_fold(
            "0", assignments, pools, donors, eeg, text, pool_size=POOL)
        self.assertEqual(len(confirmation), 4)
        self.assertEqual(len(checkpoint), 4)
        self.assertEqual(len(fit_text_ids), 8)
        self.assertTrue(all(len(t.candidate_text_ids) == POOL for t in confirmation))
        self.assertTrue(all(t.wrong_eeg_vector is not None for t in confirmation))
        self.assertTrue(all(t.wrong_eeg_vector is None for t in checkpoint))
        self.assertEqual(maxsim_feats["eeg_tokens"].shape, (8, TT, D))


class DiagnosticTests(unittest.TestCase):
    def test_grid_shape_and_reference(self):
        assignments, pools, donors, eeg, text = build_world()
        config = TrainConfig(epochs=2, batch_size=4, lr=1e-3, temperature=0.05)
        out = run_convergence_diagnostic(
            "0", 20260722, assignments, pools, donors, eeg, text,
            base_config=config, temperatures=(0.05, 0.2), lrs=(None, 3e-3),
            select_every=1, pool_size=POOL)
        self.assertEqual(len(out["grid"]), 4)                    # 2 temps x 2 lrs
        self.assertEqual(out["locked_temperature"], 0.05)
        self.assertGreaterEqual(out["pooled_reference_mrr"], 1.0 / POOL)
        for cell in out["grid"]:
            self.assertIn("maxsim_confirmation_mrr", cell)
            self.assertIn("mean_collapse_gap", cell)
            self.assertEqual(cell["delta_vs_pooled"],
                             round(cell["maxsim_confirmation_mrr"] - out["pooled_reference_mrr"], 5))
            self.assertTrue(len(cell["dev_mrr_trace"]) >= 1)
            self.assertGreaterEqual(cell["loss"]["steps"], 1)
        # best_cell is the max-MRR grid cell; verdict names the two outcomes
        self.assertEqual(out["best_cell"]["maxsim_confirmation_mrr"],
                         max(c["maxsim_confirmation_mrr"] for c in out["grid"]))
        self.assertTrue(out["verdict"].startswith(("NULL_ROBUST", "PARITY_REACHABLE")))

    def test_lr_none_uses_base(self):
        assignments, pools, donors, eeg, text = build_world()
        config = TrainConfig(epochs=1, batch_size=4, lr=2e-3, temperature=0.05)
        out = run_convergence_diagnostic(
            "0", 1, assignments, pools, donors, eeg, text,
            base_config=config, temperatures=(0.1,), lrs=(None,),
            select_every=1, pool_size=POOL)
        self.assertEqual(out["grid"][0]["lr"], 2e-3)


if __name__ == "__main__":
    unittest.main()
