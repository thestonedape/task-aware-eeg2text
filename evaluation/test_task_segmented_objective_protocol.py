"""Regression tests for the prospective P4b protocol freezer."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.freeze_task_segmented_objective_protocol import (  # noqa: E402
    ARTIFACTS,
    REPORT_NAME,
    build_confirmation_donors,
    freeze_rows,
    load_contract,
    sha256,
    validate_rows,
    verify_frozen_output,
)


CONTRACT_PATH = Path(__file__).with_name("task_segmented_objective_contract.json")


def fixture_rows(identities_per_task: int = 130) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in ("NR", "TSR"):
        for identity_number in range(identities_per_task):
            identity = f"{task.lower()}-text-{identity_number:04d}"
            length = 6 + identity_number % 31
            for subject in ("S1", "S2"):
                rows.append(
                    {
                        "trial_id": f"ZuCo2::{task}::{subject}::{identity_number:04d}",
                        "split": "train",
                        "cohort": "primary_zuco2_nr_tsr",
                        "dataset_version": "ZuCo2",
                        "reading_task": task,
                        "subject_id": subject,
                        "normalized_text_sha256": identity,
                        "text_target_id": identity,
                        "text": " ".join([task, "fixture"] + [f"w{identity_number}"] * (length - 2)),
                        "length_words_whitespace_v1": length,
                        "eeg_vector_file": f"vectors/correct_train_{identity_number // 128:05d}.npz",
                        "eeg_vector_offset": identity_number % 128,
                        "eeg_vector_dim": 1024,
                    }
                )
    return rows


class TaskSegmentedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract(CONTRACT_PATH)
        cls.contract_sha256 = sha256(CONTRACT_PATH)

    def test_deterministic_freeze_and_semantics(self) -> None:
        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            report = freeze_rows(rows, first, self.contract, self.contract_sha256)
            second_report = freeze_rows(rows, second, self.contract, self.contract_sha256)

            self.assertEqual(report, second_report)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["counts"]["eligible_rows"], 520)
            self.assertEqual(report["counts"]["candidate_pools"], 1040)
            self.assertEqual(report["counts"]["candidate_pool_rows"], 1040 * 24)
            self.assertFalse(report["checks"]["official_validation_used"])
            self.assertFalse(report["checks"]["held_out_test_accessed"])
            self.assertFalse(report["checks"]["full_training_batch_schedule_frozen"])
            self.assertTrue(report["checks"]["common_64_identity_grid_feasible_in_every_outer_fit"])
            self.assertFalse(report["execution_readiness"]["training_authorized"])
            for name in (*ARTIFACTS, REPORT_NAME):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

            verified = verify_frozen_output(first)
            self.assertEqual(verified["contract_sha256"], self.contract_sha256)
            registry = json.loads((first / "protocol_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["training"]["total_runs"], 45)
            self.assertEqual(registry["shared_model"]["vector_dim"], 1024)
            self.assertEqual(registry["shared_model"]["trainable_parameters"], 196608)
            self.assertEqual(
                registry["shared_model"]["active_trainable_parameters_per_example"], 196608
            )
            self.assertEqual(registry["batches"]["batch_size"], 64)
            self.assertEqual(registry["batches"]["negatives_per_anchor"], 31)
            self.assertEqual(registry["uncertainty"]["per_task_noninferiority_margin_mrr"], 0.01)
            self.assertEqual(registry["uncertainty"]["signal_gap_noninferiority_margin_mrr"], 0.005)
            self.assertFalse(registry["training_authorized"])

            with (first / "outer_split_assignments.csv").open(encoding="utf-8", newline="") as handle:
                assignments = list(csv.DictReader(handle))
            for outer_fold in range(5):
                identities: dict[str, set[str]] = {}
                for row in assignments:
                    if int(row["outer_fold"]) == outer_fold:
                        identities.setdefault(row["role"], set()).add(row["normalized_text_sha256"])
                self.assertFalse(identities["fit"] & identities["checkpoint"])
                self.assertFalse(identities["fit"] & identities["confirmation"])
                self.assertFalse(identities["checkpoint"] & identities["confirmation"])

            with (first / "candidate_pools.csv").open(encoding="utf-8", newline="") as handle:
                pools = list(csv.DictReader(handle))
            grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
            for row in pools:
                grouped.setdefault(
                    (row["outer_fold"], row["partition"], row["target_trial_id"]), []
                ).append(row)
            self.assertTrue(all(len(members) == 24 for members in grouped.values()))
            for (_, partition, _), members in grouped.items():
                self.assertEqual(sum(int(row["is_positive"]) for row in members), 1)
                self.assertEqual(
                    sum(int(row["is_designated_donor_text"]) for row in members),
                    int(partition == "confirmation"),
                )

    def test_non_train_row_is_rejected(self) -> None:
        row = fixture_rows(1)[0]
        row["split"] = "val"
        with self.assertRaisesRegex(ValueError, "non-train"):
            validate_rows([row])

    def test_missing_same_subject_donor_is_rejected(self) -> None:
        row = fixture_rows(1)[0]
        folds = {str(row["normalized_text_sha256"]): 0}
        with self.assertRaisesRegex(ValueError, "no same-fold/task/subject donor"):
            build_confirmation_donors([row], folds, folds=1, seed=7)

    def test_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            freeze_rows(fixture_rows(), output, self.contract, self.contract_sha256)
            path = output / "candidate_pools.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_frozen_output(output)


if __name__ == "__main__":
    unittest.main()
