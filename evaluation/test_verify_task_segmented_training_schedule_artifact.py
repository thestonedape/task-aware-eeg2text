"""Focused tests for the sealed task-segmented schedule verifier."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evaluation.freeze_task_segmented_training_schedule import (
    freeze_assignments,
    load_schedule_contract,
)
from evaluation.test_task_segmented_training_schedule import fixture_assignments
from evaluation.verify_task_segmented_training_schedule_artifact import (
    CORE_FILES,
    METADATA_NAME,
    PARENT_VERIFICATION_NAME,
    SCHEDULE_CONTRACT_NAME,
    VerificationExpectations,
    sha256,
    verify,
    write_report,
)


PARENT_CONTRACT_SHA256 = "b" * 64
PARENT_REPORT_SHA256 = "c" * 64
PARENT_METADATA_SHA256 = "d" * 64
PARENT_SOURCE_ID = (
    "kaggle-dataset-thestonedape-task-aware-eeg2text-"
    "task-segmented-protocol-version-1"
)
UPSTREAM_SOURCE_ID = "kaggle-dataset-thestonedape-task-aware-eegtotext-version-2"
PARENT_SLUG = "thestonedape/task-aware-eeg2text-task-segmented-protocol"
PARENT_COMMIT = "1" * 40
SCHEDULE_COMMIT = "2" * 40
ARTIFACT_SOURCE_ID = "fixture-schedule-version-1"
PYTHON_VERSION = "fixture-python"
CATALOG_ROWS = 320
BATCHES = 3
SHAPE = (15, 40, 3, 64)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_fixture(root: Path, contract_path: Path) -> VerificationExpectations:
    source_contract = Path(__file__).with_name("task_segmented_training_schedule_contract.json")
    contract = load_schedule_contract(source_contract)
    contract["parent"] = dict(contract["parent"])
    contract["parent"].update({
        "preserved_dataset_slug": PARENT_SLUG,
        "preserved_dataset_version": 1,
        "preserved_protocol_artifact_source_id": PARENT_SOURCE_ID,
        "upstream_input_source_id": UPSTREAM_SOURCE_ID,
        "task_segmented_objective_contract_sha256": PARENT_CONTRACT_SHA256,
        "task_segmented_protocol_report_sha256": PARENT_REPORT_SHA256,
    })
    write_json(contract_path, contract)
    contract_sha256 = sha256(contract_path)
    parent_report = {
        "status": "pass",
        "preserved_source_id": PARENT_SOURCE_ID,
        "protocol_commit": PARENT_COMMIT,
        "contract_sha256": PARENT_CONTRACT_SHA256,
        "protocol_report_sha256": PARENT_REPORT_SHA256,
        "protocol_freeze_run_metadata_sha256": PARENT_METADATA_SHA256,
        "training_authorized": False,
        "held_out_test_accessed": False,
        "counts": {"eligible_rows": CATALOG_ROWS},
        "checks": {
            "training_authorized": False,
            "held_out_test_accessed": False,
        },
        "verified_artifact_sha256": {
            "task_segmented_objective_contract.json": PARENT_CONTRACT_SHA256,
            "task_segmented_protocol_report.json": PARENT_REPORT_SHA256,
            "protocol_freeze_run_metadata.json": PARENT_METADATA_SHA256,
        },
    }
    freeze_assignments(
        fixture_assignments(),
        root,
        contract,
        contract_sha256,
        PARENT_REPORT_SHA256,
        parent_verification_summary=parent_report,
    )
    shutil.copy2(contract_path, root / SCHEDULE_CONTRACT_NAME)
    write_json(root / PARENT_VERIFICATION_NAME, parent_report)
    parent_verification_sha256 = sha256(root / PARENT_VERIFICATION_NAME)

    core_hashes = {name: sha256(root / name) for name in sorted(CORE_FILES)}
    metadata = {
        "status": "pass",
        "schema_version": 1,
        "python": PYTHON_VERSION,
        "schedule_commit": SCHEDULE_COMMIT,
        "parent_dataset_slug": PARENT_SLUG,
        "parent_dataset_version": 1,
        "parent_preserved_source_id": PARENT_SOURCE_ID,
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "parent_protocol_report_sha256": PARENT_REPORT_SHA256,
        "parent_protocol_metadata_sha256": PARENT_METADATA_SHA256,
        "parent_verification_report_sha256": parent_verification_sha256,
        "schedule_contract_sha256": contract_sha256,
        "schedule_manifest_sha256": core_hashes[
            "task_segmented_training_schedule_manifest.json"
        ],
        "schedule_report_sha256": core_hashes[
            "task_segmented_training_schedule_report.json"
        ],
        "core_file_sha256": core_hashes,
        "catalog_rows": CATALOG_ROWS,
        "global_batches_per_epoch": BATCHES,
        "shape": list(SHAPE),
        "bounded_smoke_authorized": True,
        "full_training_authorized": False,
        "held_out_test_accessed": False,
    }
    write_json(root / METADATA_NAME, metadata)
    artifact_hashes = {
        path.name: sha256(path)
        for path in root.iterdir()
    }
    return VerificationExpectations(
        artifact_source_id=ARTIFACT_SOURCE_ID,
        artifact_sha256=artifact_hashes,
        python_version=PYTHON_VERSION,
        schedule_commit=SCHEDULE_COMMIT,
        parent_dataset_slug=PARENT_SLUG,
        parent_dataset_version=1,
        parent_source_id=PARENT_SOURCE_ID,
        upstream_source_id=UPSTREAM_SOURCE_ID,
        parent_protocol_commit=PARENT_COMMIT,
        parent_contract_sha256=PARENT_CONTRACT_SHA256,
        parent_protocol_report_sha256=PARENT_REPORT_SHA256,
        parent_protocol_metadata_sha256=PARENT_METADATA_SHA256,
        parent_verification_report_sha256=parent_verification_sha256,
        schedule_contract_sha256=contract_sha256,
        schedule_manifest_sha256=core_hashes[
            "task_segmented_training_schedule_manifest.json"
        ],
        schedule_report_sha256=core_hashes[
            "task_segmented_training_schedule_report.json"
        ],
        catalog_rows=CATALOG_ROWS,
        global_batches_per_epoch=BATCHES,
        shape=SHAPE,
        require_parent_clean_remount=True,
    )


class TaskSegmentedTrainingScheduleArtifactVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        base = Path(cls.temporary.name)
        cls.root = base / "artifact"
        cls.expectations = build_fixture(cls.root, base / SCHEDULE_CONTRACT_NAME)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def copy(self, name: str) -> Path:
        target = Path(self.temporary.name) / name
        shutil.copytree(self.root, target)
        return target

    def test_valid_sealed_fixture_is_deeply_verified(self) -> None:
        result = verify(self.root, ARTIFACT_SOURCE_ID, self.expectations)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scheduled_uint32_indices_deeply_verified"], 115200)
        self.assertTrue(result["bounded_smoke_authorized"])
        self.assertFalse(result["full_training_authorized"])

        report_path = Path(self.temporary.name) / "verification" / "report.json"
        write_report(report_path, result)
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), result)
        self.assertFalse(report_path.with_suffix(".json.tmp").exists())

    def test_wrong_child_dataset_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "preserved schedule artifact source"):
            verify(self.root, "fixture-schedule-version-2", self.expectations)

    def test_byte_tamper_is_rejected_by_external_hashes(self) -> None:
        target = self.copy("tampered")
        with (target / "schedule_indices.u32le").open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\xff\xff\xff\xff")
        with self.assertRaisesRegex(ValueError, "production schedule artifact hashes"):
            verify(target, ARTIFACT_SOURCE_ID, self.expectations)

    def test_extra_missing_and_symlink_inventory_are_rejected(self) -> None:
        for mode in ("extra", "missing"):
            with self.subTest(mode=mode):
                target = self.copy(f"inventory-{mode}")
                if mode == "extra":
                    (target / "extra.txt").write_text("extra\n", encoding="utf-8")
                else:
                    (target / METADATA_NAME).unlink()
                with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                    verify(target, ARTIFACT_SOURCE_ID, self.expectations)

        target = self.copy("inventory-symlink")
        link = target / METADATA_NAME
        link.unlink()
        try:
            os.symlink(self.root / METADATA_NAME, link)
        except OSError as exc:
            return
        with self.assertRaisesRegex(ValueError, "symlink is forbidden"):
            verify(target, ARTIFACT_SOURCE_ID, self.expectations)

    def test_rehashed_metadata_cannot_expand_authorization(self) -> None:
        target = self.copy("authorization-drift")
        metadata = json.loads((target / METADATA_NAME).read_text(encoding="utf-8"))
        metadata["full_training_authorized"] = True
        write_json(target / METADATA_NAME, metadata)
        hashes = dict(self.expectations.artifact_sha256)
        hashes[METADATA_NAME] = sha256(target / METADATA_NAME)
        drifted = replace(self.expectations, artifact_sha256=hashes)
        with self.assertRaisesRegex(ValueError, "full-training authorization"):
            verify(target, ARTIFACT_SOURCE_ID, drifted)


if __name__ == "__main__":
    unittest.main()
