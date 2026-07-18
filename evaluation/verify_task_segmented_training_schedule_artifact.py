"""Verify the preserved P4b task-segmented training-schedule artifact.

This standard-library entrypoint treats a clean Kaggle remount as a sealed
nine-file artifact.  It binds the immutable child dataset version and every
file digest externally, validates all parent/child provenance and
authorization fields, then delegates to the independent schedule verifier to
decode every one of the 4,032,000 production indices.  No artifact content is
allowed to select a path outside the supplied root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Package import in tests.
    from .verify_task_segmented_training_schedule_output import (
        CORE_FILES,
        MANIFEST_NAME,
        REPORT_NAME,
        verify as deep_verify_schedule,
    )
except ImportError:  # Direct ``python evaluation/<script>.py`` execution.
    from verify_task_segmented_training_schedule_output import (  # type: ignore
        CORE_FILES,
        MANIFEST_NAME,
        REPORT_NAME,
        verify as deep_verify_schedule,
    )


SCHEDULE_CONTRACT_NAME = "task_segmented_training_schedule_contract.json"
PARENT_VERIFICATION_NAME = "parent_protocol_verification_report.json"
METADATA_NAME = "schedule_freeze_run_metadata.json"
PROVENANCE_FILES = (
    SCHEDULE_CONTRACT_NAME,
    PARENT_VERIFICATION_NAME,
    METADATA_NAME,
)
ROOT_FILES = frozenset((*CORE_FILES, *PROVENANCE_FILES))

PARENT_CONTRACT_NAME = "task_segmented_objective_contract.json"
PARENT_REPORT_NAME = "task_segmented_protocol_report.json"
PARENT_METADATA_NAME = "protocol_freeze_run_metadata.json"

PRODUCTION_ARTIFACT_SHA256 = {
    "parent_protocol_verification_report.json": (
        "376fedd2ca89189653a6a4e784195411bea061c63b220a8dfce33cba0b8b4b32"
    ),
    "schedule_audit.csv": (
        "f18299fbba6725f7eb33c4686bdd496a83de139c76f3494f48f6a1148d169fbf"
    ),
    "schedule_freeze_run_metadata.json": (
        "8487422b900573e8b44d7e83d78eb32beb9786f8496a972c5585291c3576dc1b"
    ),
    "schedule_indices.u32le": (
        "79543e72f496ee3f7a8140556b274c15ecc5992e900a07bed5e5c74a2ddd7cbc"
    ),
    "schedule_units.csv": (
        "e2644a51ac578b18388ce82d07d910714182cf674e60b0a5f2c547067b8720aa"
    ),
    "task_segmented_training_schedule_contract.json": (
        "a6ea34388cd98380654f413b1440d0d5cee0b8065555b0c04a53d0db6ea12287"
    ),
    "task_segmented_training_schedule_manifest.json": (
        "0cf2a752b5f0a67e7282bc0b4551b4792ceb3d346dd441fef6c359515885270a"
    ),
    "task_segmented_training_schedule_report.json": (
        "5e7d953a8ca31e48d6acbd6168b146d695af449ef0ad5bc573a37d306cb7250f"
    ),
    "trial_catalog.csv": (
        "3d93e0cea4290ac22e8111760241d04109392e6f024abae85a4d9504fc4f8fc9"
    ),
}

METADATA_FIELDS = frozenset({
    "status",
    "schema_version",
    "python",
    "schedule_commit",
    "parent_dataset_slug",
    "parent_dataset_version",
    "parent_preserved_source_id",
    "parent_contract_sha256",
    "parent_protocol_report_sha256",
    "parent_protocol_metadata_sha256",
    "parent_verification_report_sha256",
    "schedule_contract_sha256",
    "schedule_manifest_sha256",
    "schedule_report_sha256",
    "core_file_sha256",
    "catalog_rows",
    "global_batches_per_epoch",
    "shape",
    "bounded_smoke_authorized",
    "full_training_authorized",
    "held_out_test_accessed",
})


@dataclass(frozen=True)
class VerificationExpectations:
    artifact_source_id: str
    artifact_sha256: Mapping[str, str]
    python_version: str
    schedule_commit: str
    parent_dataset_slug: str
    parent_dataset_version: int
    parent_source_id: str
    upstream_source_id: str
    parent_protocol_commit: str
    parent_contract_sha256: str
    parent_protocol_report_sha256: str
    parent_protocol_metadata_sha256: str
    parent_verification_report_sha256: str
    schedule_contract_sha256: str
    schedule_manifest_sha256: str
    schedule_report_sha256: str
    catalog_rows: int
    global_batches_per_epoch: int
    shape: tuple[int, int, int, int]
    parent_verification_bytes: int | None = None
    require_parent_clean_remount: bool = True


PRODUCTION_EXPECTATIONS = VerificationExpectations(
    artifact_source_id=(
        "kaggle-dataset-thestonedape-task-aware-eeg2text-"
        "task-segmented-schedule-version-1"
    ),
    artifact_sha256=PRODUCTION_ARTIFACT_SHA256,
    python_version="3.12.13",
    schedule_commit="93a5bdcdd4a1fc6b140097921906968459b10fe9",
    parent_dataset_slug="thestonedape/task-aware-eeg2text-task-segmented-protocol",
    parent_dataset_version=1,
    parent_source_id=(
        "kaggle-dataset-thestonedape-task-aware-eeg2text-"
        "task-segmented-protocol-version-1"
    ),
    upstream_source_id="kaggle-dataset-thestonedape-task-aware-eegtotext-version-2",
    parent_protocol_commit="51ec9352b611b7abb057d610cb52c23ebd54c88e",
    parent_contract_sha256=(
        "396670afc0244cb601364ff89df53944c4f63402191a9d120e6e2648e5baed3b"
    ),
    parent_protocol_report_sha256=(
        "f99ff4ad371e30b86dc9582bb4c35854f6bba863d7868a5f424f93776eab8116"
    ),
    parent_protocol_metadata_sha256=(
        "766c836c96c2025805d81bbbad7d3378faf14b8eca4fc1885653add5a1f35dc9"
    ),
    parent_verification_report_sha256=(
        "376fedd2ca89189653a6a4e784195411bea061c63b220a8dfce33cba0b8b4b32"
    ),
    schedule_contract_sha256=(
        "a6ea34388cd98380654f413b1440d0d5cee0b8065555b0c04a53d0db6ea12287"
    ),
    schedule_manifest_sha256=(
        "0cf2a752b5f0a67e7282bc0b4551b4792ceb3d346dd441fef6c359515885270a"
    ),
    schedule_report_sha256=(
        "5e7d953a8ca31e48d6acbd6168b146d695af449ef0ad5bc573a37d306cb7250f"
    ),
    catalog_rows=9011,
    global_batches_per_epoch=105,
    shape=(15, 40, 105, 64),
    parent_verification_bytes=2851,
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path.name}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def exact_inventory(root: Path) -> None:
    names: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            require(
                not entry.is_symlink(),
                f"symlink is forbidden in schedule artifact root: {entry.name}",
            )
            require(
                entry.is_file(follow_symlinks=False),
                f"non-file schedule artifact entry: {entry.name}",
            )
            names.add(entry.name)
    missing = sorted(ROOT_FILES - names)
    extra = sorted(names - ROOT_FILES)
    require(
        not missing and not extra,
        f"schedule artifact inventory mismatch; missing={missing}, extra={extra}",
    )


def _verify_contract(
    contract: Mapping[str, Any], expectations: VerificationExpectations,
) -> None:
    require_equal(contract.get("schema_version"), 1, "schedule contract schema")
    require_equal(
        contract.get("status"),
        "frozen_before_p4b_schedule_generation",
        "schedule contract status",
    )
    parent = contract.get("parent")
    require(isinstance(parent, dict), "schedule contract lacks parent provenance")
    require_equal(parent.get("preserved_dataset_slug"), expectations.parent_dataset_slug,
                  "contract parent dataset slug")
    require_equal(parent.get("preserved_dataset_version"), expectations.parent_dataset_version,
                  "contract parent dataset version")
    require_equal(parent.get("preserved_protocol_artifact_source_id"), expectations.parent_source_id,
                  "contract parent source")
    require_equal(parent.get("upstream_input_source_id"), expectations.upstream_source_id,
                  "contract upstream source")
    require_equal(parent.get("task_segmented_objective_contract_sha256"),
                  expectations.parent_contract_sha256, "contract parent contract")
    require_equal(parent.get("task_segmented_protocol_report_sha256"),
                  expectations.parent_protocol_report_sha256, "contract parent report")

    schedule = contract.get("schedule")
    require(isinstance(schedule, dict), "schedule contract lacks schedule section")
    require_equal(schedule.get("outer_folds"), [0, 1, 2, 3, 4], "contract folds")
    require_equal(schedule.get("training_seeds"), [20260717, 20260718, 20260719],
                  "contract training seeds")
    require_equal(schedule.get("epochs"), expectations.shape[1], "contract epochs")
    require_equal(schedule.get("batch_size"), expectations.shape[3], "contract batch size")
    require_equal(schedule.get("cells"), ["NR::0", "NR::1", "TSR::0", "TSR::1"],
                  "contract task/pseudo cells")
    require_equal(schedule.get("examples_per_cell"), 16, "contract examples per cell")
    require_equal(schedule.get("same_schedule_across_arms"), True,
                  "contract same-schedule flag")
    require_equal(schedule.get("arm_id_in_schedule"), False, "contract arm-axis flag")
    output = schedule.get("output")
    require(isinstance(output, dict), "schedule contract lacks output section")
    require_equal(output.get("index_dtype"), "little-endian unsigned 32-bit integer",
                  "contract index dtype")
    require_equal(output.get("index_order"), "C order", "contract index order")
    require_equal(output.get("index_shape"),
                  [expectations.shape[0], expectations.shape[1], "B", expectations.shape[3]],
                  "contract index shape")

    optimization = contract.get("optimization")
    require(isinstance(optimization, dict), "schedule contract lacks optimization section")
    require_equal(optimization.get("paired_initialization_within_outer_fold_and_seed"), True,
                  "contract paired initialization")
    require_equal(optimization.get("initial_state_hash_must_match_across_arms"), True,
                  "contract initial-state binding")

    authorization = contract.get("authorization")
    require(isinstance(authorization, dict), "schedule contract lacks authorization section")
    require_equal(authorization.get("bounded_smoke_authorized_after_schedule_freeze"), True,
                  "contract bounded-smoke authorization")
    require_equal(authorization.get("full_training_authorized"), False,
                  "contract full-training authorization")
    require_equal(authorization.get("held_out_test_accessed"), False,
                  "contract held-out test access")
    bounded = authorization.get("bounded_smoke")
    require(isinstance(bounded, dict), "schedule contract lacks bounded-smoke definition")
    require_equal(bounded, {
        "outer_fold": 0,
        "training_seed": 20260717,
        "epoch": 1,
        "batches": [0, 1],
        "arms": ["global_mixed", "true_task_segmented", "pseudo_task_segmented"],
        "scientific_decision_permitted": False,
    }, "contract bounded-smoke definition")


def _verify_parent_report(
    parent_report: Mapping[str, Any], expectations: VerificationExpectations,
) -> None:
    require_equal(parent_report.get("status"), "pass", "parent verification status")
    require_equal(parent_report.get("preserved_source_id"), expectations.parent_source_id,
                  "parent verification source")
    require_equal(parent_report.get("protocol_commit"), expectations.parent_protocol_commit,
                  "parent protocol commit")
    require_equal(parent_report.get("contract_sha256"), expectations.parent_contract_sha256,
                  "parent verification contract")
    require_equal(parent_report.get("protocol_report_sha256"),
                  expectations.parent_protocol_report_sha256,
                  "parent verification report")
    require_equal(parent_report.get("protocol_freeze_run_metadata_sha256"),
                  expectations.parent_protocol_metadata_sha256,
                  "parent verification metadata")
    require_equal(parent_report.get("training_authorized"), False,
                  "parent training authorization")
    require_equal(parent_report.get("held_out_test_accessed"), False,
                  "parent held-out test access")
    counts = parent_report.get("counts")
    require(isinstance(counts, dict), "parent verification lacks counts")
    require_equal(counts.get("eligible_rows"), expectations.catalog_rows,
                  "parent/catalog row binding")
    checks = parent_report.get("checks")
    require(isinstance(checks, dict), "parent verification lacks checks")
    require_equal(checks.get("training_authorized"), False,
                  "parent check training authorization")
    require_equal(checks.get("held_out_test_accessed"), False,
                  "parent check held-out test access")
    verified = parent_report.get("verified_artifact_sha256")
    require(isinstance(verified, dict), "parent verification lacks artifact hashes")
    require_equal(verified.get(PARENT_CONTRACT_NAME), expectations.parent_contract_sha256,
                  "parent verified contract")
    require_equal(verified.get(PARENT_REPORT_NAME), expectations.parent_protocol_report_sha256,
                  "parent verified report")
    require_equal(verified.get(PARENT_METADATA_NAME), expectations.parent_protocol_metadata_sha256,
                  "parent verified metadata")


def _verify_metadata(
    metadata: Mapping[str, Any],
    actual_hashes: Mapping[str, str],
    expectations: VerificationExpectations,
) -> None:
    require_equal(set(metadata), METADATA_FIELDS, "schedule metadata field inventory")
    require_equal(metadata.get("status"), "pass", "schedule metadata status")
    require_equal(metadata.get("schema_version"), 1, "schedule metadata schema")
    require_equal(metadata.get("python"), expectations.python_version,
                  "schedule metadata Python version")
    require_equal(metadata.get("schedule_commit"), expectations.schedule_commit,
                  "schedule implementation commit")
    require_equal(metadata.get("parent_dataset_slug"), expectations.parent_dataset_slug,
                  "metadata parent dataset slug")
    require_equal(metadata.get("parent_dataset_version"), expectations.parent_dataset_version,
                  "metadata parent dataset version")
    require_equal(metadata.get("parent_preserved_source_id"), expectations.parent_source_id,
                  "metadata parent source")
    require_equal(metadata.get("parent_contract_sha256"), expectations.parent_contract_sha256,
                  "metadata parent contract")
    require_equal(metadata.get("parent_protocol_report_sha256"),
                  expectations.parent_protocol_report_sha256, "metadata parent report")
    require_equal(metadata.get("parent_protocol_metadata_sha256"),
                  expectations.parent_protocol_metadata_sha256, "metadata parent metadata")
    require_equal(metadata.get("parent_verification_report_sha256"),
                  expectations.parent_verification_report_sha256,
                  "metadata parent verification")
    require_equal(metadata.get("schedule_contract_sha256"), expectations.schedule_contract_sha256,
                  "metadata schedule contract")
    require_equal(metadata.get("schedule_manifest_sha256"), expectations.schedule_manifest_sha256,
                  "metadata schedule manifest")
    require_equal(metadata.get("schedule_report_sha256"), expectations.schedule_report_sha256,
                  "metadata schedule report")
    require_equal(metadata.get("core_file_sha256"),
                  {name: actual_hashes[name] for name in sorted(CORE_FILES)},
                  "metadata core-file hashes")
    require_equal(metadata.get("catalog_rows"), expectations.catalog_rows,
                  "metadata catalog rows")
    require_equal(metadata.get("global_batches_per_epoch"),
                  expectations.global_batches_per_epoch, "metadata global B")
    require_equal(metadata.get("shape"), list(expectations.shape), "metadata shape")
    require_equal(metadata.get("bounded_smoke_authorized"), True,
                  "metadata bounded-smoke authorization")
    require_equal(metadata.get("full_training_authorized"), False,
                  "metadata full-training authorization")
    require_equal(metadata.get("held_out_test_accessed"), False,
                  "metadata held-out test access")


def verify(
    artifact_root: Path,
    preserved_source_id: str,
    expectations: VerificationExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, Any]:
    """Verify the exact mounted child artifact and deeply decode its schedule."""

    require_equal(preserved_source_id, expectations.artifact_source_id,
                  "preserved schedule artifact source")
    root = Path(os.path.abspath(os.fspath(artifact_root)))
    require(root.is_dir(), f"schedule artifact root is not a directory: {root}")
    exact_inventory(root)

    require_equal(set(expectations.artifact_sha256), ROOT_FILES,
                  "expected schedule artifact hash inventory")
    require(all(is_sha256(value) for value in expectations.artifact_sha256.values()),
            "expected schedule artifact hashes must be lowercase SHA-256")
    actual_hashes = {name: sha256(root / name) for name in sorted(ROOT_FILES)}
    require_equal(actual_hashes, dict(sorted(expectations.artifact_sha256.items())),
                  "production schedule artifact hashes")
    require_equal(actual_hashes[SCHEDULE_CONTRACT_NAME], expectations.schedule_contract_sha256,
                  "schedule contract SHA256")
    require_equal(actual_hashes[PARENT_VERIFICATION_NAME],
                  expectations.parent_verification_report_sha256,
                  "parent verification SHA256")
    require_equal(actual_hashes[MANIFEST_NAME], expectations.schedule_manifest_sha256,
                  "schedule manifest SHA256")
    require_equal(actual_hashes[REPORT_NAME], expectations.schedule_report_sha256,
                  "schedule report SHA256")
    if expectations.parent_verification_bytes is not None:
        require_equal((root / PARENT_VERIFICATION_NAME).stat().st_size,
                      expectations.parent_verification_bytes,
                      "parent verification byte length")

    contract = read_json(root / SCHEDULE_CONTRACT_NAME)
    parent_report = read_json(root / PARENT_VERIFICATION_NAME)
    manifest = read_json(root / MANIFEST_NAME)
    metadata = read_json(root / METADATA_NAME)
    _verify_contract(contract, expectations)
    _verify_parent_report(parent_report, expectations)
    require_equal(
        manifest.get("parent_clean_remount_verification"),
        parent_report,
        "manifest/preserved parent-verification equality",
    )
    _verify_metadata(metadata, actual_hashes, expectations)

    deep = deep_verify_schedule(
        root,
        expectations.schedule_contract_sha256,
        expectations.parent_protocol_report_sha256,
        expected_catalog_rows=expectations.catalog_rows,
        expected_batches_per_epoch=expectations.global_batches_per_epoch,
        expected_shape=expectations.shape,
        require_parent_clean_remount=expectations.require_parent_clean_remount,
        allowed_extra_files=PROVENANCE_FILES,
    )
    require_equal(deep.get("status"), "pass", "deep schedule verification")
    require_equal(deep.get("schedule_manifest_sha256"),
                  expectations.schedule_manifest_sha256, "deep manifest binding")
    require_equal(deep.get("schedule_report_sha256"),
                  expectations.schedule_report_sha256, "deep report binding")
    require_equal(deep.get("bounded_smoke_authorized"), True,
                  "deep bounded-smoke authorization")
    require_equal(deep.get("full_training_authorized"), False,
                  "deep full-training authorization")
    require_equal(deep.get("held_out_test_accessed"), False,
                  "deep held-out test access")

    scheduled_indices = 1
    for dimension in expectations.shape:
        scheduled_indices *= dimension
    return {
        "status": "pass",
        "preserved_source_id": preserved_source_id,
        "schedule_commit": expectations.schedule_commit,
        "parent_dataset_slug": expectations.parent_dataset_slug,
        "parent_dataset_version": expectations.parent_dataset_version,
        "parent_preserved_source_id": expectations.parent_source_id,
        "schedule_contract_sha256": expectations.schedule_contract_sha256,
        "schedule_manifest_sha256": expectations.schedule_manifest_sha256,
        "schedule_report_sha256": expectations.schedule_report_sha256,
        "schedule_freeze_run_metadata_sha256": actual_hashes[METADATA_NAME],
        "verified_artifact_sha256": actual_hashes,
        "catalog_rows": expectations.catalog_rows,
        "global_batches_per_epoch": expectations.global_batches_per_epoch,
        "shape": list(expectations.shape),
        "scheduled_uint32_indices_deeply_verified": scheduled_indices,
        "applicable_arms": deep["applicable_arms"],
        "bounded_smoke_authorized": True,
        "full_training_authorized": False,
        "held_out_test_accessed": False,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write a canonical verification report with atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--preserved-source-id", required=True)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    report = verify(args.artifact_root, args.preserved_source_id)
    if args.output_report is not None:
        write_report(args.output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TASK-SEGMENTED TRAINING SCHEDULE ARTIFACT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
