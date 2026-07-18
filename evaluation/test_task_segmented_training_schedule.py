"""Regression tests for the deterministic P4b common schedule freezer."""

from __future__ import annotations

import copy
import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.freeze_task_segmented_training_schedule import (  # noqa: E402
    INDEX_NAME,
    MANIFEST_NAME,
    PARENT_ARTIFACTS,
    PARENT_CONTRACT_NAME,
    PARENT_REPORT_NAME,
    REPORT_NAME,
    SCHEDULE_ARTIFACTS,
    build_unit_schedule,
    canonical_trial_catalog,
    freeze_assignments,
    global_mask_key,
    initialization_seed,
    load_schedule_contract,
    read_csv,
    select_batch_identities,
    sha256,
    verify_frozen_output,
    verify_parent,
)


CONTRACT_PATH = Path(__file__).with_name("task_segmented_training_schedule_contract.json")


def fixture_assignments(identities_per_cell: int = 40, rows_per_identity: int = 2) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in ("NR", "TSR"):
        for pseudo in (0, 1):
            for identity_number in range(identities_per_cell):
                identity = f"{task.lower()}-p{pseudo}-text-{identity_number:04d}"
                text_fold = identity_number % 5
                for row_number in range(rows_per_identity):
                    trial_id = f"ZuCo2::{task}::P{pseudo}::T{identity_number:04d}::R{row_number}"
                    for outer_fold in range(5):
                        role = (
                            "confirmation" if text_fold == outer_fold
                            else "checkpoint" if text_fold == (outer_fold + 1) % 5
                            else "fit"
                        )
                        rows.append({
                            "outer_fold": str(outer_fold),
                            "role": role,
                            "text_fold": str(text_fold),
                            "trial_id": trial_id,
                            "dataset_version": "ZuCo2",
                            "reading_task": task,
                            "subject_id": f"S{row_number}",
                            "normalized_text_sha256": identity,
                            "text_target_id": identity,
                            "pseudo_group": str(pseudo),
                            "length_words_whitespace_v1": str(8 + identity_number % 10),
                            "eeg_vector_file": "eeg/correct_train_00000.npz",
                            "eeg_vector_offset": str(identity_number * rows_per_identity + row_number),
                            "eeg_vector_dim": "1024",
                        })
    return rows


class TaskSegmentedTrainingScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_schedule_contract(CONTRACT_PATH)
        cls.contract_sha = sha256(CONTRACT_PATH)

    def test_contract_freezes_execution_ambiguities_without_full_authorization(self) -> None:
        self.assertEqual(self.contract["optimization"]["temperature"], 1.0)
        self.assertFalse(self.contract["optimization"]["automatic_mixed_precision"])
        self.assertFalse(self.contract["optimization"]["foreach"])
        self.assertFalse(self.contract["optimization"]["fused"])
        self.assertEqual(self.contract["bootstrap"]["replicates"], 5000)
        self.assertEqual(
            self.contract["bootstrap"]["superiority_familywise_lower_quantile"], 0.0125
        )
        self.assertTrue(
            self.contract["authorization"]["bounded_smoke_authorized_after_schedule_freeze"]
        )
        self.assertFalse(self.contract["authorization"]["full_training_authorized"])
        self.assertEqual(self.contract["randomness"]["global_mask_seed"], 2026071806)

    def test_initialization_and_mask_key_derivations_are_exact_and_sensitive(self) -> None:
        first_seed = initialization_seed(0, 20260717)
        self.assertEqual(first_seed, 414155286793290345)
        self.assertEqual(first_seed, initialization_seed(0, 20260717))
        self.assertNotEqual(first_seed, initialization_seed(1, 20260717))
        first_key = global_mask_key("a" * 64, "b" * 64, 0, 20260717, 1, 0)
        self.assertEqual(first_key, "8e0c1cb6ef4cee71982c406a4fbe7b5447c75809e22619e9afaf5e52a572f0cc")
        self.assertEqual(first_key, global_mask_key("a" * 64, "b" * 64, 0, 20260717, 1, 0))
        self.assertNotEqual(first_key, global_mask_key("a" * 64, "b" * 64, 0, 20260717, 1, 1))

    def test_freeze_is_byte_deterministic_order_independent_and_balanced(self) -> None:
        rows = fixture_assignments()
        shuffled = list(rows)
        random.Random(91).shuffle(shuffled)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            report = freeze_assignments(rows, first, self.contract, self.contract_sha, "a" * 64)
            repeated = freeze_assignments(shuffled, second, self.contract, self.contract_sha, "a" * 64)
            self.assertEqual(report, repeated)
            self.assertEqual(report["counts"]["schedule_units"], 15)
            self.assertEqual(report["counts"]["epochs_per_unit"], 40)
            self.assertEqual(report["counts"]["global_batches_per_epoch"], 3)
            self.assertFalse(report["authorization"]["full_training_authorized"])
            for name in (*SCHEDULE_ARTIFACTS, REPORT_NAME):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            manifest = json.loads((first / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["shape"], [15, 40, 3, 64])
            self.assertEqual(manifest["dtype"], "<u4")
            self.assertEqual(
                manifest["applicable_arms"],
                ["global_mixed", "true_task_segmented", "pseudo_task_segmented"],
            )
            self.assertTrue(manifest["arm_axis_absent_from_binary"])
            self.assertTrue(manifest["same_schedule_across_arms"])
            self.assertEqual((first / INDEX_NAME).stat().st_size, 15 * 40 * 3 * 64 * 4)
            audits = read_csv(first / "schedule_audit.csv")
            self.assertEqual(len(audits), 15 * 4)
            self.assertTrue(all(int(row["identity_use_gap"]) <= 1 for row in audits))
            self.assertTrue(all(int(row["conditional_row_use_gap_max"]) <= 1 for row in audits))
            self.assertTrue(all(row["fit_trial_rows"] == row["covered_fit_rows"] for row in audits))
            units = read_csv(first / "schedule_units.csv")
            for fold in range(5):
                hashes = {row["unit_sha256"] for row in units if int(row["outer_fold"]) == fold}
                self.assertEqual(len(hashes), 3)
            self.assertEqual(verify_frozen_output(first)["status"], "pass")

    def test_catalog_rejects_cross_fit_role_drift(self) -> None:
        rows = fixture_assignments()
        rows[0] = dict(rows[0], role="fit")
        with self.assertRaisesRegex(ValueError, "role mismatch"):
            canonical_trial_catalog(rows, [0, 1, 2, 3, 4])

    def test_impossible_global_identity_matching_is_rejected(self) -> None:
        catalog = []
        index = 0
        for task in ("NR", "TSR"):
            for pseudo in (0, 1):
                for identity_number in range(16):
                    catalog.append({
                        "trial_index": index,
                        "trial_id": f"{task}-{pseudo}-{identity_number}",
                        "text_fold": 2,
                        "dataset_version": "ZuCo2",
                        "reading_task": task,
                        "subject_id": "S1",
                        "normalized_text_sha256": f"shared-{identity_number}",
                        "text_target_id": f"shared-{identity_number}",
                        "pseudo_group": pseudo,
                        "length_words_whitespace_v1": 10,
                        "eeg_vector_file": "x.npz",
                        "eeg_vector_offset": index,
                        "eeg_vector_dim": 1024,
                    })
                    index += 1
        with self.assertRaisesRegex(ValueError, "future-feasibility global identity bound"):
            build_unit_schedule(catalog, 0, 20260717, 1, 1, "fixture")

    def test_augmenting_matcher_repairs_a_solvable_cross_cell_overlap(self) -> None:
        from collections import Counter

        cells = (("NR", 0), ("NR", 1), ("TSR", 0), ("TSR", 1))
        identities_by_cell = {}
        for cell_number, (task, pseudo) in enumerate(cells):
            # Adjacent cells share eight identities, but each also has enough
            # private identities for a globally unique 64-row batch.
            identities = [f"shared-{cell_number}-{value}" for value in range(8)]
            identities += [f"shared-{cell_number - 1}-{value}" for value in range(8)]
            identities += [f"private-{cell_number}-{value}" for value in range(16)]
            identities_by_cell[(task, pseudo)] = set(identities)
        selected = select_batch_identities(
            identities_by_cell,
            {cell: Counter({identity: 0 for identity in identities_by_cell[cell]}) for cell in cells},
            ("overlap", 0),
        )
        flattened = [identity for cell in cells for identity in selected[cell]]
        self.assertEqual(len(flattened), 64)
        self.assertEqual(len(set(flattened)), 64)
        self.assertTrue(all(len(selected[cell]) == 16 for cell in cells))

    def test_saturated_identity_tail_trap_preserves_future_feasibility(self) -> None:
        from collections import Counter

        cells = (("NR", 0), ("NR", 1), ("TSR", 0), ("TSR", 1))
        shared = [f"shared-{value}" for value in range(16)]
        nr_private = [f"nr-private-{value}" for value in range(16)]
        tsr_private = [f"tsr-private-{value}" for value in range(16)]
        nr1 = [f"nr1-{value}" for value in range(16)]
        tsr1 = [f"tsr1-{value}" for value in range(16)]
        identities = {
            ("NR", 0): set(shared + nr_private),
            ("NR", 1): set(nr1),
            ("TSR", 0): set(shared + tsr_private),
            ("TSR", 1): set(tsr1),
        }
        remaining = {
            ("NR", 0): Counter({identity: 1 for identity in shared + nr_private}),
            ("NR", 1): Counter({identity: 2 for identity in nr1}),
            ("TSR", 0): Counter({identity: 1 for identity in shared + tsr_private}),
            ("TSR", 1): Counter({identity: 2 for identity in tsr1}),
        }
        uses = {cell: Counter({identity: 0 for identity in identities[cell]}) for cell in cells}
        first = select_batch_identities(
            identities, uses, ("tail", 0), remaining_quotas=remaining, remaining_batches=2
        )
        selected_first = {identity for cell in cells for identity in first[cell]}
        # The old myopic matcher could omit shared identities here and leave a
        # 16-per-cell final residual with only 61 distinct identities.
        self.assertTrue(set(shared + nr1 + tsr1).issubset(selected_first))
        for cell in cells:
            for identity in first[cell]:
                remaining[cell][identity] -= 1
                uses[cell][identity] += 1
            self.assertEqual(sum(remaining[cell].values()), 16)
        global_residual = Counter()
        for cell in cells:
            global_residual.update(remaining[cell])
        self.assertLessEqual(max(global_residual.values()), 1)
        second = select_batch_identities(
            identities, uses, ("tail", 1), remaining_quotas=remaining, remaining_batches=1
        )
        self.assertEqual(len({identity for cell in cells for identity in second[cell]}), 64)
        for cell in cells:
            for identity in second[cell]:
                remaining[cell][identity] -= 1
            self.assertTrue(all(value == 0 for value in remaining[cell].values()))

    def test_nonempty_output_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                freeze_assignments(
                    fixture_assignments(), output, self.contract, self.contract_sha, "a" * 64
                )

    def test_parent_hash_binding_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / PARENT_CONTRACT_NAME).write_text("{}\n", encoding="utf-8")
            for name in PARENT_ARTIFACTS:
                (root / name).write_text(name + "\n", encoding="utf-8")
            contract = copy.deepcopy(self.contract)
            parent = contract["parent"]
            parent["task_segmented_objective_contract_sha256"] = sha256(root / PARENT_CONTRACT_NAME)
            parent["artifact_sha256"] = {name: sha256(root / name) for name in PARENT_ARTIFACTS}
            report = {
                "status": "pass",
                "contract_sha256": parent["task_segmented_objective_contract_sha256"],
                "artifact_sha256": parent["artifact_sha256"],
                "input_verification": {"preserved_source_id": parent["upstream_input_source_id"]},
                "execution_readiness": {"training_authorized": False},
            }
            (root / PARENT_REPORT_NAME).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            parent["task_segmented_protocol_report_sha256"] = sha256(root / PARENT_REPORT_NAME)
            self.assertEqual(verify_parent(root, contract)["report"]["status"], "pass")
            injected = verify_parent(
                root,
                contract,
                artifact_verifier=lambda artifact_root, source_id: {
                    "status": "pass",
                    "preserved_source_id": source_id,
                    "protocol_report_sha256": parent["task_segmented_protocol_report_sha256"],
                },
            )
            self.assertEqual(injected["clean_remount_summary"]["status"], "pass")
            (root / "pseudo_groups.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_parent(root, contract)

    def test_output_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            freeze_assignments(
                fixture_assignments(), output, self.contract, self.contract_sha, "a" * 64
            )
            with (output / INDEX_NAME).open("ab") as handle:
                handle.write(b"\x00\x00\x00\x00")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_frozen_output(output)


if __name__ == "__main__":
    unittest.main()
