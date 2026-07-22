"""Local tests for the campaign loaders (no GPU); synthetic files per schema.

Run: python -B -m unittest evaluation.test_token_campaign_io
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.token_campaign_io import (
    load_assignments,
    load_candidate_pools,
    load_donors,
    load_eeg_lookup,
    load_text_lookup,
    select_partition,
)

TD, DD = 4, 8      # small token-count / dim for the synthetic representations


def _write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_eeg_run(root):
    (root / "tokens").mkdir(parents=True)
    tokens = np.arange(2 * TD * DD, dtype=np.float16).reshape(2, TD, DD)
    vectors = np.arange(2 * DD, dtype=np.float16).reshape(2, DD)
    np.savez(root / "tokens" / "tokens_00000.npz", tokens=tokens, vectors=vectors)
    (root / "tokens" / "tokens_00000.json").write_text(
        json.dumps({"trial_ids": ["t0", "t1"]}), encoding="utf-8")
    (root / "token_index.json").write_text(
        json.dumps({"chunks": [{"token_file": "tokens/tokens_00000.npz"}]}), encoding="utf-8")


def build_text_run(root):
    (root / "tokens").mkdir(parents=True)
    tokens = np.arange(2 * TD * DD, dtype=np.float16).reshape(2, TD, DD)
    masks = np.ones((2, TD), dtype=np.int8)
    vectors = np.arange(2 * DD, dtype=np.float32).reshape(2, DD)
    np.savez(root / "tokens" / "text_00000.npz", tokens=tokens, masks=masks, vectors=vectors)
    _write_csv(root / "text_token_index.csv",
               ["text_target_id", "token_file", "token_offset"],
               [{"text_target_id": "txt0", "token_file": "tokens/text_00000.npz", "token_offset": 0},
                {"text_target_id": "txt1", "token_file": "tokens/text_00000.npz", "token_offset": 1}])


class LoaderTests(unittest.TestCase):
    def test_eeg_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); build_eeg_run(root)
            lut = load_eeg_lookup(root)
            self.assertEqual(set(lut), {"t0", "t1"})
            self.assertEqual(lut["t0"]["tokens"].shape, (TD, DD))
            self.assertEqual(lut["t1"]["vector"].shape, (DD,))
            self.assertEqual(lut["t1"]["vector"].dtype, __import__("torch").float32)

    def test_text_lookup_uses_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); build_text_run(root)
            lut = load_text_lookup(root)
            self.assertEqual(set(lut), {"txt0", "txt1"})
            self.assertEqual(lut["txt0"]["tokens"].shape, (TD, DD))
            self.assertEqual(lut["txt0"]["mask"].shape, (TD,))
            # offset 1 must load the second row, not the first
            self.assertAlmostEqual(float(lut["txt1"]["vector"][0]), float(DD))

    def test_contract_loaders_and_partition_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(root / "assignments.csv",
                       ["outer_fold", "role", "trial_id", "reading_task", "subject_id", "text_target_id"],
                       [{"outer_fold": "0", "role": "confirmation", "trial_id": "t0",
                         "reading_task": "NR", "subject_id": "S0", "text_target_id": "txt0"},
                        {"outer_fold": "0", "role": "checkpoint", "trial_id": "t1",
                         "reading_task": "TSR", "subject_id": "S1", "text_target_id": "txt1"}])
            _write_csv(root / "candidate_pools.csv",
                       ["outer_fold", "partition", "target_trial_id", "candidate_rank",
                        "candidate_text_target_id", "is_positive"],
                       [{"outer_fold": "0", "partition": "confirmation", "target_trial_id": "t0",
                         "candidate_rank": "1", "candidate_text_target_id": "txt0", "is_positive": "True"},
                        {"outer_fold": "0", "partition": "confirmation", "target_trial_id": "t0",
                         "candidate_rank": "0", "candidate_text_target_id": "txt1", "is_positive": "False"}])
            _write_csv(root / "confirmation_donors.csv",
                       ["outer_fold", "partition", "target_trial_id", "donor_trial_id"],
                       [{"outer_fold": "0", "partition": "confirmation",
                         "target_trial_id": "t0", "donor_trial_id": "t1"}])

            assignments = load_assignments(root / "assignments.csv")
            pools = load_candidate_pools(root / "candidate_pools.csv")
            donors = load_donors(root / "confirmation_donors.csv")

            # candidate rank order is respected (rank 0 first)
            self.assertEqual([c["candidate_text_target_id"]
                              for c in pools[("0", "confirmation", "t0")]], ["txt1", "txt0"])
            self.assertEqual(donors[("0", "t0")], "t1")

            targets, pool_slice, donor_slice = select_partition(
                assignments, pools, donors, outer_fold="0", partition="confirmation")
            self.assertEqual([t["trial_id"] for t in targets], ["t0"])
            self.assertEqual(targets[0]["reading_task"], "NR")
            self.assertIn("t0", pool_slice)
            self.assertEqual(donor_slice, {"t0": "t1"})


if __name__ == "__main__":
    unittest.main()
