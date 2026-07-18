"""Regression tests for the deep frozen-schedule output verifier."""

from __future__ import annotations

import csv
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.freeze_task_segmented_training_schedule import (  # noqa: E402
    MANIFEST_NAME,
    REPORT_NAME,
    freeze_assignments,
    load_schedule_contract,
    sha256,
)
from evaluation.test_task_segmented_training_schedule import fixture_assignments  # noqa: E402
from evaluation.verify_task_segmented_training_schedule_output import (  # noqa: E402
    AUDIT_NAME,
    INDEX_NAME,
    MANIFEST_ARTIFACTS,
    REPORT_ARTIFACTS,
    UNITS_NAME,
    verify,
)


CONTRACT_PATH = Path(__file__).with_name("task_segmented_training_schedule_contract.json")
PARENT_REPORT_SHA256 = "a" * 64
FIXTURE_CATALOG_ROWS = 320
FIXTURE_BATCHES = 3
FIXTURE_SHAPE = (15, 40, 3, 64)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resign(root: Path) -> None:
    """Simulate an attacker updating all self-declared file hashes."""

    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        name: sha256(root / name) for name in MANIFEST_ARTIFACTS
    }
    _write_json(root / MANIFEST_NAME, manifest)
    report = json.loads((root / REPORT_NAME).read_text(encoding="utf-8"))
    report["artifact_sha256"] = {
        name: sha256(root / name) for name in REPORT_ARTIFACTS
    }
    _write_json(root / REPORT_NAME, report)


def _repair_first_unit_hash(root: Path) -> None:
    fields, units = _read_csv(root / UNITS_NAME)
    first = units[0]
    offset = int(first["byte_offset"])
    length = int(first["byte_length"])
    payload = (root / INDEX_NAME).read_bytes()[offset:offset + length]
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    first["unit_sha256"] = digest
    _write_csv(root / UNITS_NAME, fields, units)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["unit_order"][0]["unit_sha256"] = digest
    _write_json(root / MANIFEST_NAME, manifest)


class TaskSegmentedScheduleOutputVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name) / "schedule"
        cls.contract = load_schedule_contract(CONTRACT_PATH)
        cls.contract_sha256 = sha256(CONTRACT_PATH)
        freeze_assignments(
            fixture_assignments(),
            cls.base,
            cls.contract,
            cls.contract_sha256,
            PARENT_REPORT_SHA256,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _copy(self, name: str) -> Path:
        target = Path(self.temporary.name) / name
        shutil.copytree(self.base, target)
        return target

    def _verify(self, root: Path) -> dict[str, object]:
        return verify(
            root,
            self.contract_sha256,
            PARENT_REPORT_SHA256,
            expected_catalog_rows=FIXTURE_CATALOG_ROWS,
            expected_batches_per_epoch=FIXTURE_BATCHES,
            expected_shape=FIXTURE_SHAPE,
            require_parent_clean_remount=False,
        )

    def test_valid_fixture_is_fully_decoded_and_verified(self) -> None:
        result = self._verify(self.base)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["shape"], list(FIXTURE_SHAPE))
        self.assertEqual(result["catalog_rows"], FIXTURE_CATALOG_ROWS)
        self.assertEqual(result["audit_rows"], 60)
        self.assertTrue(result["bounded_smoke_authorized"])
        self.assertFalse(result["full_training_authorized"])

    def test_exact_six_file_inventory_is_enforced(self) -> None:
        target = self._copy("extra-file")
        (target / "unbound.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self._verify(target)

    def test_explicit_provenance_extra_does_not_weaken_default_inventory(self) -> None:
        target = self._copy("declared-extra-file")
        (target / "provenance.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self._verify(target)
        result = verify(
            target,
            self.contract_sha256,
            PARENT_REPORT_SHA256,
            expected_catalog_rows=FIXTURE_CATALOG_ROWS,
            expected_batches_per_epoch=FIXTURE_BATCHES,
            expected_shape=FIXTURE_SHAPE,
            require_parent_clean_remount=False,
            allowed_extra_files=("provenance.json",),
        )
        self.assertEqual(result["status"], "pass")

        (target / "undeclared.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            verify(
                target,
                self.contract_sha256,
                PARENT_REPORT_SHA256,
                expected_catalog_rows=FIXTURE_CATALOG_ROWS,
                expected_batches_per_epoch=FIXTURE_BATCHES,
                expected_shape=FIXTURE_SHAPE,
                require_parent_clean_remount=False,
                allowed_extra_files=("provenance.json",),
            )

    def test_symbolic_link_cannot_impersonate_a_core_file(self) -> None:
        target = self._copy("symlink-core")
        link = target / REPORT_NAME
        link.unlink()
        try:
            os.symlink(self.base / REPORT_NAME, link)
        except OSError as exc:
            self.skipTest(f"symbolic links unavailable on this platform: {exc}")
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            self._verify(target)

    def test_plain_byte_tamper_fails_report_hash_binding(self) -> None:
        target = self._copy("plain-tamper")
        with (target / INDEX_NAME).open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\xff\xff\xff\xff")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self._verify(target)

    def test_rehashed_out_of_range_index_still_fails_deep_decode(self) -> None:
        target = self._copy("rehashed-index")
        with (target / INDEX_NAME).open("r+b") as handle:
            handle.seek(0)
            handle.write(struct.pack("<I", FIXTURE_CATALOG_ROWS))
        _repair_first_unit_hash(target)
        _resign(target)
        with self.assertRaisesRegex(ValueError, "out-of-range catalog index"):
            self._verify(target)

    def test_rehashed_audit_lie_is_recomputed_and_rejected(self) -> None:
        target = self._copy("audit-lie")
        fields, rows = _read_csv(target / AUDIT_NAME)
        rows[0]["covered_fit_rows"] = str(int(rows[0]["covered_fit_rows"]) - 1)
        _write_csv(target / AUDIT_NAME, fields, rows)
        _resign(target)
        with self.assertRaisesRegex(ValueError, "schedule audit mismatch"):
            self._verify(target)

    def test_rehashed_noncanonical_unit_offset_is_rejected(self) -> None:
        target = self._copy("unit-offset")
        fields, rows = _read_csv(target / UNITS_NAME)
        rows[0]["byte_offset"] = "4"
        _write_csv(target / UNITS_NAME, fields, rows)
        _resign(target)
        with self.assertRaisesRegex(ValueError, "byte offsets"):
            self._verify(target)

    def test_rehashed_arm_or_authorization_drift_is_rejected(self) -> None:
        target = self._copy("arm-drift")
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest["applicable_arms"] = ["global_mixed", "unplanned_arm"]
        manifest["full_training_authorized"] = True
        _write_json(target / MANIFEST_NAME, manifest)
        _resign(target)
        with self.assertRaisesRegex(ValueError, "applicable-arm"):
            self._verify(target)


if __name__ == "__main__":
    unittest.main()
