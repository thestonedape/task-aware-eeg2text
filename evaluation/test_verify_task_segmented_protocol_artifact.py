from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evaluation.verify_task_segmented_protocol_artifact import (
    ARTIFACT_FILES,
    CONTRACT_NAME,
    EXPECTED_CHECKS,
    METADATA_NAME,
    REPORT_NAME,
    VerificationExpectations,
    sha256,
    verify,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id"])
        writer.writeheader()
        writer.writerows({"row_id": index} for index in range(count))


def build_fixture(root: Path) -> VerificationExpectations:
    root.mkdir()
    artifact_source = "fixture-task-segmented-protocol-version-1"
    input_source = "fixture-prompt-neutral-input-version-2"
    input_manifest = "b" * 64
    commit = "a" * 40
    folds = 2
    pool_size = 2
    counts = {
        "eligible_rows": 2,
        "normalized_text_groups": 2,
        "outer_split_assignment_rows": 4,
        "pseudo_task_text_identities": 2,
        "confirmation_donors": 2,
        "candidate_pools": 4,
        "candidate_pool_rows": 8,
        "task_rows": {"NR": 1, "TSR": 1},
        "fold_rows": {"0": 1, "1": 1},
    }
    contract = {
        "schema_version": 1,
        "status": "prospectively_frozen_before_p4b_training",
        "input": {
            "preserved_source_id": input_source,
            "combined_manifest_sha256": input_manifest,
        },
        "eligibility": {"split": "train", "reading_tasks": ["NR", "TSR"]},
        "cross_fitting": {"folds": folds, "held_out_test_accessed": False},
        "evaluation": {"pool_size": pool_size},
    }
    write_json(root / CONTRACT_NAME, contract)
    contract_sha = sha256(root / CONTRACT_NAME)

    write_rows(root / "text_group_folds.csv", counts["normalized_text_groups"])
    write_rows(root / "outer_split_assignments.csv", counts["outer_split_assignment_rows"])
    write_rows(root / "pseudo_groups.csv", counts["pseudo_task_text_identities"])
    write_rows(root / "batch_grid_feasibility.csv", folds * 2 * 2)
    write_rows(root / "confirmation_donors.csv", counts["confirmation_donors"])
    write_rows(root / "candidate_pools.csv", counts["candidate_pool_rows"])
    registry = {
        "schema_version": 1,
        "status": "partition_and_evaluation_protocol_frozen_batch_schedule_pending",
        "contract_sha256": contract_sha,
        "training_authorized": False,
        "held_out_test_accessed": False,
    }
    write_json(root / "protocol_registry.json", registry)
    artifact_hashes = {name: sha256(root / name) for name in ARTIFACT_FILES}
    report = {
        "status": "pass",
        "schema_version": 1,
        "contract_sha256": contract_sha,
        "input_verification": {
            "status": "pass",
            "preserved_source_id": input_source,
            "combined_manifest_sha256": input_manifest,
            "all_215_chunk_hashes_revalidated": True,
            "held_out_test_accessed": False,
        },
        "counts": counts,
        "execution_readiness": {
            "partition_pseudo_donor_and_pool_manifests_frozen": True,
            "batch_grid_cell_feasibility_checked": True,
            "full_40_epoch_batch_schedule_frozen": False,
            "training_authorized": False,
        },
        "checks": dict(EXPECTED_CHECKS),
        "artifact_sha256": artifact_hashes,
    }
    write_json(root / REPORT_NAME, report)
    report_sha = sha256(root / REPORT_NAME)
    metadata = {
        "status": "pass",
        "protocol_commit": commit,
        "preserved_source_id": input_source,
        "input_manifest_sha256": input_manifest,
        "contract_sha256": contract_sha,
        "protocol_report_sha256": report_sha,
        "artifact_sha256": artifact_hashes,
        "training_authorized": False,
        "held_out_test_accessed": False,
    }
    write_json(root / METADATA_NAME, metadata)
    return VerificationExpectations(
        artifact_source_id=artifact_source,
        input_source_id=input_source,
        input_manifest_sha256=input_manifest,
        protocol_commit=commit,
        contract_sha256=contract_sha,
        report_sha256=report_sha,
        metadata_sha256=sha256(root / METADATA_NAME),
        artifact_sha256=artifact_hashes,
        counts=counts,
        checks=EXPECTED_CHECKS,
        folds=folds,
        pool_size=pool_size,
    )


def rebind_report(root: Path, expectations: VerificationExpectations) -> VerificationExpectations:
    report_sha = sha256(root / REPORT_NAME)
    metadata = json.loads((root / METADATA_NAME).read_text(encoding="utf-8"))
    metadata["protocol_report_sha256"] = report_sha
    write_json(root / METADATA_NAME, metadata)
    return replace(
        expectations,
        report_sha256=report_sha,
        metadata_sha256=sha256(root / METADATA_NAME),
    )


class TaskSegmentedProtocolArtifactVerifierTests(unittest.TestCase):
    def fixture(self, parent: Path) -> tuple[Path, VerificationExpectations]:
        root = parent / "artifact"
        return root, build_fixture(root)

    def test_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, expected = self.fixture(Path(temporary))
            first = verify(root, expected.artifact_source_id, expected)
            second = verify(root, expected.artifact_source_id, expected)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "pass")
            self.assertFalse(first["training_authorized"])

    def test_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, expected = self.fixture(Path(temporary))
            with (root / "candidate_pools.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "production artifact hashes"):
                verify(root, expected.artifact_source_id, expected)

    def test_extra_and_missing_files_are_rejected(self) -> None:
        for mode in ("extra", "missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root, expected = self.fixture(Path(temporary))
                if mode == "extra":
                    (root / "extra.txt").write_text("extra\n", encoding="utf-8")
                else:
                    (root / "pseudo_groups.csv").unlink()
                with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                    verify(root, expected.artifact_source_id, expected)

    def test_wrong_source_or_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, expected = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "preserved artifact source"):
                verify(root, "fixture-task-segmented-protocol-version-2", expected)

    def test_count_binding_status_and_test_access_are_rejected(self) -> None:
        mutations = ("count", "binding", "status", "test_access")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, expected = self.fixture(Path(temporary))
                if mutation in {"count", "status"}:
                    report = json.loads((root / REPORT_NAME).read_text(encoding="utf-8"))
                    if mutation == "count":
                        report["counts"]["eligible_rows"] += 1
                    else:
                        report["status"] = "failed"
                    write_json(root / REPORT_NAME, report)
                    expected = rebind_report(root, expected)
                else:
                    metadata = json.loads((root / METADATA_NAME).read_text(encoding="utf-8"))
                    if mutation == "binding":
                        metadata["contract_sha256"] = "0" * 64
                    else:
                        metadata["held_out_test_accessed"] = True
                    write_json(root / METADATA_NAME, metadata)
                    expected = replace(expected, metadata_sha256=sha256(root / METADATA_NAME))
                with self.assertRaises(ValueError):
                    verify(root, expected.artifact_source_id, expected)


if __name__ == "__main__":
    unittest.main()
