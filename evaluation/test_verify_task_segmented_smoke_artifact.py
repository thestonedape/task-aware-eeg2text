"""Synthetic and tamper tests for the bounded P4b smoke verifier."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from evaluation.verify_task_segmented_smoke_artifact import (
    ARM_IDS,
    AUTHORIZATION,
    BOUNDED_SMOKE,
    EXECUTION_CONTRACT_SHA256,
    IMMUTABLE_INPUTS,
    MANIFEST_NAME,
    METADATA_NAME,
    NON_MANIFEST_FILES,
    sha256,
    verify,
    write_report,
)


PROJECT_COMMIT = "1" * 40
RUNNER_SHA256 = "2" * 64
SOURCE_ID = "fixture-task-segmented-smoke-version-1"


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hash_token(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def base_binding() -> dict[str, object]:
    return {
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "input_bindings": IMMUTABLE_INPUTS,
        "bounded_smoke": BOUNDED_SMOKE,
        "authorization": AUTHORIZATION,
    }


def reseal(root: Path, manifest: dict[str, object] | None = None) -> str:
    path = root / MANIFEST_NAME
    if manifest is None:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        relative: sha256(root / Path(*relative.split("/")))
        for relative in sorted(NON_MANIFEST_FILES)
    }
    write_json(path, manifest)
    return sha256(path)


def build_fixture(root: Path) -> str:
    root.mkdir()
    (root / "runs").mkdir()

    common_fields = [
        "outer_fold",
        "training_seed",
        "epoch",
        "batch_index",
        "batch_position",
        "catalog_index",
        "trial_id",
        "subject_id",
        "reading_task",
        "pseudo_task_id",
        "normalized_text_sha256",
        "text_target_id",
        "eeg_vector_file",
        "eeg_vector_offset",
        "text_vector_file",
        "text_vector_offset",
    ]
    common_rows: list[dict[str, object]] = []
    for batch in (0, 1):
        position = 0
        for task in ("NR", "TSR"):
            for pseudo in (0, 1):
                for within in range(16):
                    catalog = batch * 1000 + position
                    common_rows.append({
                        "outer_fold": 0,
                        "training_seed": 20260717,
                        "epoch": 1,
                        "batch_index": batch,
                        "batch_position": position,
                        "catalog_index": catalog,
                        "trial_id": f"trial-{catalog}",
                        "subject_id": f"subject-{catalog % 12}",
                        "reading_task": task,
                        "pseudo_task_id": pseudo,
                        "normalized_text_sha256": hash_token(f"text target {catalog}"),
                        "text_target_id": hash_token(f"text target {catalog}"),
                        "eeg_vector_file": f"eeg/chunk-{batch}.npz",
                        "eeg_vector_offset": position,
                        "text_vector_file": f"text/chunk-{batch}.npz",
                        "text_vector_offset": position,
                    })
                    position += 1
    write_csv(root / "common_batch_trace.csv", common_fields, common_rows)
    common_catalog_hashes = {
        batch: hashlib.sha256(
            b"".join(
                (batch * 1000 + position).to_bytes(4, "little", signed=False)
                for position in range(64)
            )
        ).hexdigest()
        for batch in (0, 1)
    }
    common_trial_hashes = {
        batch: hashlib.sha256(
            ("\n".join(f"trial-{batch * 1000 + position}" for position in range(64))
             + "\n").encode("utf-8")
        ).hexdigest()
        for batch in (0, 1)
    }

    step_fields = [
        "outer_fold",
        "training_seed",
        "arm_id",
        "epoch",
        "batch_index",
        "global_step",
        "schedule_unit_sha256",
        "batch_catalog_indices_sha256",
        "batch_trial_ids_sha256",
        "global_mask_key",
        "eeg_to_text_mask_sha256",
        "text_to_eeg_mask_sha256",
        "initial_state_sha256",
        "pre_step_state_sha256",
        "loss",
        "gradient_norm_preclip",
        "post_step_state_sha256",
        "optimizer_state_sha256",
    ]
    arm_rows: list[dict[str, object]] = []
    common_initial = hash_token("paired initial state")
    for arm in ARM_IDS:
        arm_root = root / "runs" / arm
        arm_root.mkdir()
        # Intentionally not a valid torch/pickle payload: verification must not load it.
        (arm_root / "resume_checkpoint.pt").write_bytes(
            b"opaque-not-a-pickle\x00" + arm.encode("ascii")
        )
        after_one = hash_token(f"{arm} after one")
        after_two = hash_token(f"{arm} after two")
        steps: list[dict[str, object]] = []
        for batch in (0, 1):
            steps.append({
                "outer_fold": 0,
                "training_seed": 20260717,
                "arm_id": arm,
                "epoch": 1,
                "batch_index": batch,
                "global_step": batch + 1,
                "schedule_unit_sha256": hash_token("schedule unit"),
                "batch_catalog_indices_sha256": common_catalog_hashes[batch],
                "batch_trial_ids_sha256": common_trial_hashes[batch],
                "global_mask_key": hash_token(f"global mask key {batch}"),
                "eeg_to_text_mask_sha256": hash_token(f"{arm} e2t {batch}"),
                "text_to_eeg_mask_sha256": hash_token(f"{arm} t2e {batch}"),
                "initial_state_sha256": common_initial,
                "pre_step_state_sha256": common_initial if batch == 0 else after_one,
                "loss": f"{1.0 + batch / 10.0:.6f}",
                "gradient_norm_preclip": f"{0.25 + batch / 10.0:.6f}",
                "post_step_state_sha256": after_one if batch == 0 else after_two,
                "optimizer_state_sha256": hash_token(f"{arm} optimizer {batch}"),
            })
        write_csv(arm_root / "step_trace.csv", step_fields, steps)
        summary: dict[str, object] = {
            "schema_version": 1,
            "status": "pass",
            "run_mode": "bounded_smoke",
            "arm_id": arm,
            "outer_fold": 0,
            "training_seed": 20260717,
            "epoch": 1,
            "batch_indices": [0, 1],
            "optimizer_steps": 2,
            "trainable_parameters": 196608,
            "initial_state_sha256": common_initial,
            "final_state_sha256": after_two,
            "final_optimizer_state_sha256": hash_token(f"{arm} optimizer 1"),
            "full_training_authorized": False,
            "scientific_decision_permitted": False,
            "held_out_test_accessed": False,
            "artifact_sha256": {
                "resume_checkpoint.pt": sha256(arm_root / "resume_checkpoint.pt"),
                "step_trace.csv": sha256(arm_root / "step_trace.csv"),
            },
        }
        write_json(arm_root / "run_summary.json", summary)
        arm_rows.append({
            "arm_id": arm,
            "status": "pass",
            "optimizer_steps": 2,
            "initial_state_sha256": common_initial,
            "final_state_sha256": after_two,
            "final_optimizer_state_sha256": hash_token(f"{arm} optimizer 1"),
        })

    write_csv(
        root / "arm_summary.csv",
        [
            "arm_id",
            "status",
            "optimizer_steps",
            "initial_state_sha256",
            "final_state_sha256",
            "final_optimizer_state_sha256",
        ],
        arm_rows,
    )
    binding = base_binding()
    metadata: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "run_mode": "bounded_smoke",
        "project_commit": PROJECT_COMMIT,
        "runner_source_sha256": RUNNER_SHA256,
        **binding,
        "completed_arms": list(ARM_IDS),
        "optimizer_steps_completed": 6,
        "full_training_authorized": False,
        "scientific_decision_permitted": False,
        "held_out_test_accessed": False,
    }
    write_json(root / METADATA_NAME, metadata)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "run_mode": "bounded_smoke",
        "project_commit": PROJECT_COMMIT,
        "runner_source_sha256": RUNNER_SHA256,
        **binding,
        "artifact_sha256": {},
    }
    return reseal(root, manifest)


class TaskSegmentedSmokeArtifactVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "artifact"
        self.manifest_sha256 = build_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def copy(self, name: str) -> Path:
        target = Path(self.temporary.name) / name
        shutil.copytree(self.root, target)
        return target

    def test_valid_fixture_verifies_without_deserializing_checkpoint(self) -> None:
        result = verify(
            self.root,
            self.manifest_sha256,
            preserved_source_id=SOURCE_ID,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["arms"], list(ARM_IDS))
        self.assertEqual(result["total_optimizer_steps"], 6)
        self.assertEqual(result["common_trace_rows"], 128)
        self.assertFalse(result["checkpoint_deserialized"])
        self.assertFalse(result["scientific_decision_permitted"])

        report_path = Path(self.temporary.name) / "verification" / "report.json"
        write_report(report_path, result)
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), result)
        self.assertFalse(report_path.with_suffix(".json.tmp").exists())

    def test_external_manifest_digest_and_nonmanifest_tamper_are_rejected(self) -> None:
        target = self.copy("manifest-tamper")
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest["status"] = "tampered"
        write_json(target / MANIFEST_NAME, manifest)
        with self.assertRaisesRegex(ValueError, "externally expected smoke manifest"):
            verify(target, self.manifest_sha256)

        target = self.copy("file-tamper")
        with (target / "runs" / ARM_IDS[0] / "resume_checkpoint.pt").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "manifest/non-manifest file hashes"):
            verify(target, self.manifest_sha256)

    def test_inventory_symlink_and_path_escape_are_rejected(self) -> None:
        target = self.copy("extra")
        (target / "extra.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "top-level inventory"):
            verify(target, self.manifest_sha256)

        target = self.copy("symlink")
        link = target / METADATA_NAME
        link.unlink()
        try:
            os.symlink(self.root / METADATA_NAME, link)
        except OSError:
            pass
        else:
            with self.assertRaisesRegex(ValueError, "symlink is forbidden"):
                verify(target, self.manifest_sha256)

        target = self.copy("escape")
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        declared = manifest["artifact_sha256"]
        assert isinstance(declared, dict)
        value = declared.pop("arm_summary.csv")
        declared["../arm_summary.csv"] = value
        write_json(target / MANIFEST_NAME, manifest)
        with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
            verify(target, sha256(target / MANIFEST_NAME))

    def test_resealed_nonfinite_diagnostic_is_rejected(self) -> None:
        target = self.copy("nonfinite")
        arm = ARM_IDS[0]
        trace = target / "runs" / arm / "step_trace.csv"
        fields, rows = _read_csv_for_test(trace)
        rows[0]["loss"] = "nan"
        write_csv(trace, fields, rows)
        _refresh_run_summary_hash(target, arm)
        expected = reseal(target)
        with self.assertRaisesRegex(ValueError, "non-finite .* loss"):
            verify(target, expected)

    def test_resealed_unbalanced_batch_and_initial_state_drift_are_rejected(self) -> None:
        target = self.copy("balance")
        trace = target / "common_batch_trace.csv"
        fields, rows = _read_csv_for_test(trace)
        rows[0]["pseudo_task_id"] = "1"
        write_csv(trace, fields, rows)
        expected = reseal(target)
        with self.assertRaisesRegex(ValueError, "task/pseudo balance"):
            verify(target, expected)

        target = self.copy("initial-state")
        arm = ARM_IDS[-1]
        trace = target / "runs" / arm / "step_trace.csv"
        fields, rows = _read_csv_for_test(trace)
        changed = hash_token("unpaired initial")
        rows[0]["initial_state_sha256"] = changed
        rows[0]["pre_step_state_sha256"] = changed
        rows[1]["initial_state_sha256"] = changed
        write_csv(trace, fields, rows)
        summary_path = target / "runs" / arm / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["initial_state_sha256"] = changed
        summary["artifact_sha256"]["step_trace.csv"] = sha256(trace)
        write_json(summary_path, summary)
        aggregate = target / "arm_summary.csv"
        aggregate_fields, aggregate_rows = _read_csv_for_test(aggregate)
        next(row for row in aggregate_rows if row["arm_id"] == arm)[
            "initial_state_sha256"
        ] = changed
        write_csv(aggregate, aggregate_fields, aggregate_rows)
        expected = reseal(target)
        with self.assertRaisesRegex(ValueError, "byte-identical initial state"):
            verify(target, expected)

    def test_resealed_cross_arm_batch_binding_and_authorization_drift_are_rejected(self) -> None:
        target = self.copy("batch-binding")
        arm = ARM_IDS[-1]
        trace = target / "runs" / arm / "step_trace.csv"
        fields, rows = _read_csv_for_test(trace)
        rows[0]["batch_catalog_indices_sha256"] = hash_token("wrong catalog batch")
        write_csv(trace, fields, rows)
        _refresh_run_summary_hash(target, arm)
        expected = reseal(target)
        with self.assertRaisesRegex(ValueError, "step/common row binding"):
            verify(target, expected)

        target = self.copy("authorization")
        metadata_path = target / METADATA_NAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["full_training_authorized"] = True
        write_json(metadata_path, metadata)
        expected = reseal(target)
        with self.assertRaisesRegex(ValueError, "full-training authorization"):
            verify(target, expected)


def _read_csv_for_test(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _refresh_run_summary_hash(root: Path, arm: str) -> None:
    arm_root = root / "runs" / arm
    path = arm_root / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["artifact_sha256"]["step_trace.csv"] = sha256(arm_root / "step_trace.csv")
    write_json(path, summary)


if __name__ == "__main__":
    unittest.main()
