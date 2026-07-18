"""Verify the preserved P4b protocol artifact without loading vectors or models.

The verifier is deliberately standard-library only.  It treats the supplied
directory as a sealed artifact root, rejects any extra/missing/non-regular
entry, binds the preserved Kaggle dataset version, and independently checks
the report, metadata, contract, hashes, cardinalities, and no-training/no-test
flags.  It never follows a path declared by artifact content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_FILES = (
    "batch_grid_feasibility.csv",
    "candidate_pools.csv",
    "confirmation_donors.csv",
    "outer_split_assignments.csv",
    "protocol_registry.json",
    "pseudo_groups.csv",
    "text_group_folds.csv",
)
REPORT_NAME = "task_segmented_protocol_report.json"
METADATA_NAME = "protocol_freeze_run_metadata.json"
CONTRACT_NAME = "task_segmented_objective_contract.json"
ROOT_FILES = frozenset((*ARTIFACT_FILES, REPORT_NAME, METADATA_NAME, CONTRACT_NAME))

EXPECTED_CHECKS = {
    "all_rows_canonical_train_only": True,
    "candidate_pool_size_exact": True,
    "candidate_selection_uses_model_scores": False,
    "checkpoint_and_confirmation_candidates_partition_local": True,
    "common_64_identity_grid_feasible_in_every_outer_fit": True,
    "confirmation_donors_always_different_text": True,
    "confirmation_donors_same_fold_dataset_task_subject": True,
    "designated_donor_text_in_every_confirmation_pool": True,
    "full_training_batch_schedule_frozen": False,
    "global_text_groups_cross_fitted_without_leakage": True,
    "held_out_test_accessed": False,
    "model_or_vector_array_loaded": False,
    "official_validation_used": False,
    "one_positive_per_pool": True,
    "only_primary_zuco2_nr_tsr": True,
    "training_authorized": False,
}

PRODUCTION_ARTIFACT_SHA256 = {
    "batch_grid_feasibility.csv": "91356aad08bfabf003852e643df11d2589aaed6216da342251067d26751e9ff5",
    "candidate_pools.csv": "a0b4102acf88d956fc494a5d4237fbdac0e083bf38e5f2dbf621d40b0942ca0c",
    "confirmation_donors.csv": "fa599beb26fe6ffe1e3b5e849a8fefc0cddeef71bba4554adfd310bba203f87c",
    "outer_split_assignments.csv": "ef709d162961b423ccf9065c01b21725acd17f9a99ed99d66152be50bb5fe911",
    "protocol_registry.json": "38106f433ca39af83aac04b17fe06b96c8f4b005fbb32734ec8346994ee24120",
    "pseudo_groups.csv": "12b77333eebe2ea5f8cba45307fedbdf46e0a921934d58d992462bf5577a7085",
    "text_group_folds.csv": "0849e048f7670ae74b06ab3fde74628361efba5c328e8a1743a792e787fad472",
}

PRODUCTION_COUNTS = {
    "eligible_rows": 9011,
    "normalized_text_groups": 515,
    "outer_split_assignment_rows": 45055,
    "pseudo_task_text_identities": 573,
    "confirmation_donors": 9011,
    "candidate_pools": 18022,
    "candidate_pool_rows": 432528,
    "task_rows": {"NR": 4126, "TSR": 4885},
    "fold_rows": {"0": 1810, "1": 1785, "2": 1790, "3": 1814, "4": 1812},
}


@dataclass(frozen=True)
class VerificationExpectations:
    artifact_source_id: str
    input_source_id: str
    input_manifest_sha256: str
    protocol_commit: str
    contract_sha256: str
    report_sha256: str
    metadata_sha256: str
    artifact_sha256: Mapping[str, str]
    counts: Mapping[str, Any]
    checks: Mapping[str, bool]
    folds: int = 5
    pool_size: int = 24
    tasks: tuple[str, ...] = ("NR", "TSR")


PRODUCTION_EXPECTATIONS = VerificationExpectations(
    artifact_source_id=(
        "kaggle-dataset-thestonedape-task-aware-eeg2text-"
        "task-segmented-protocol-version-1"
    ),
    input_source_id="kaggle-dataset-thestonedape-task-aware-eegtotext-version-2",
    input_manifest_sha256="6c1fff8d2e89e33a72d03c39651e8ecce678c3b93cdb66747dd6dcc00538cddb",
    protocol_commit="51ec9352b611b7abb057d610cb52c23ebd54c88e",
    contract_sha256="396670afc0244cb601364ff89df53944c4f63402191a9d120e6e2648e5baed3b",
    report_sha256="f99ff4ad371e30b86dc9582bb4c35854f6bba863d7868a5f424f93776eab8116",
    metadata_sha256="766c836c96c2025805d81bbbad7d3378faf14b8eca4fc1885653add5a1f35dc9",
    artifact_sha256=PRODUCTION_ARTIFACT_SHA256,
    counts=PRODUCTION_COUNTS,
    checks=EXPECTED_CHECKS,
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def csv_row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header is missing: {path.name}")
        return sum(1 for _ in reader)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def exact_inventory(root: Path) -> None:
    names: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            require(not entry.is_symlink(), f"symlink is forbidden in artifact root: {entry.name}")
            require(entry.is_file(follow_symlinks=False), f"non-file artifact entry: {entry.name}")
            names.add(entry.name)
    missing = sorted(ROOT_FILES - names)
    extra = sorted(names - ROOT_FILES)
    require(not missing and not extra, f"artifact file inventory mismatch; missing={missing}, extra={extra}")


def _normal_counts(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "eligible_rows", "normalized_text_groups", "outer_split_assignment_rows",
        "pseudo_task_text_identities", "confirmation_donors", "candidate_pools",
        "candidate_pool_rows", "task_rows", "fold_rows",
    }
    require(isinstance(value, dict), "report counts must be an object")
    require_equal(set(value), required, "report count fields")
    require(isinstance(value["task_rows"], dict), "task_rows must be an object")
    require(isinstance(value["fold_rows"], dict), "fold_rows must be an object")
    result = dict(value)
    result["task_rows"] = {str(key): int(item) for key, item in value["task_rows"].items()}
    result["fold_rows"] = {str(key): int(item) for key, item in value["fold_rows"].items()}
    for key in set(result) - {"task_rows", "fold_rows"}:
        result[key] = int(result[key])
    return result


def verify(
    artifact_root: Path,
    preserved_source_id: str,
    expectations: VerificationExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, Any]:
    require_equal(preserved_source_id, expectations.artifact_source_id, "preserved artifact source")
    root = Path(os.path.abspath(os.fspath(artifact_root)))
    require(root.is_dir(), f"artifact root is not a directory: {root}")
    exact_inventory(root)

    require_equal(set(expectations.artifact_sha256), set(ARTIFACT_FILES), "expected artifact hash set")
    artifact_verified = {
        name: sha256(root / name)
        for name in ARTIFACT_FILES
    }
    require_equal(artifact_verified, dict(expectations.artifact_sha256), "production artifact hashes")
    contract_digest = sha256(root / CONTRACT_NAME)
    report_digest = sha256(root / REPORT_NAME)
    metadata_digest = sha256(root / METADATA_NAME)
    require_equal(contract_digest, expectations.contract_sha256, "contract SHA256")
    require_equal(report_digest, expectations.report_sha256, "protocol report SHA256")
    require_equal(metadata_digest, expectations.metadata_sha256, "run metadata SHA256")
    verified = {
        **artifact_verified,
        CONTRACT_NAME: contract_digest,
        REPORT_NAME: report_digest,
        METADATA_NAME: metadata_digest,
    }

    contract = read_json(root / CONTRACT_NAME)
    report = read_json(root / REPORT_NAME)
    metadata = read_json(root / METADATA_NAME)
    registry = read_json(root / "protocol_registry.json")

    require_equal(contract.get("status"), "prospectively_frozen_before_p4b_training", "contract status")
    require_equal(contract.get("input", {}).get("preserved_source_id"), expectations.input_source_id, "contract input source")
    require_equal(contract.get("input", {}).get("combined_manifest_sha256"), expectations.input_manifest_sha256, "contract input manifest")
    require_equal(contract.get("cross_fitting", {}).get("folds"), expectations.folds, "contract folds")
    require_equal(contract.get("evaluation", {}).get("pool_size"), expectations.pool_size, "contract pool size")
    require_equal(contract.get("eligibility", {}).get("split"), "train", "contract eligible split")
    require_equal(tuple(contract.get("eligibility", {}).get("reading_tasks", [])), expectations.tasks, "contract tasks")
    require_equal(contract.get("cross_fitting", {}).get("held_out_test_accessed"), False, "contract test access")

    require_equal(report.get("status"), "pass", "report status")
    require_equal(report.get("schema_version"), 1, "report schema")
    require_equal(report.get("contract_sha256"), expectations.contract_sha256, "report contract")
    require_equal(report.get("artifact_sha256"), dict(expectations.artifact_sha256), "report artifact hashes")
    require_equal(report.get("checks"), dict(expectations.checks), "report checks")
    counts = _normal_counts(report.get("counts", {}))
    expected_counts = _normal_counts(expectations.counts)
    require_equal(counts, expected_counts, "report counts")

    eligible = counts["eligible_rows"]
    require_equal(sum(counts["task_rows"].values()), eligible, "task-row arithmetic")
    require_equal(sum(counts["fold_rows"].values()), eligible, "fold-row arithmetic")
    require_equal(counts["outer_split_assignment_rows"], eligible * expectations.folds, "outer-assignment arithmetic")
    require_equal(counts["confirmation_donors"], eligible, "donor arithmetic")
    require_equal(counts["candidate_pools"], eligible * 2, "candidate-pool arithmetic")
    require_equal(counts["candidate_pool_rows"], counts["candidate_pools"] * expectations.pool_size, "candidate-row arithmetic")

    csv_counts = {
        "text_group_folds.csv": counts["normalized_text_groups"],
        "outer_split_assignments.csv": counts["outer_split_assignment_rows"],
        "pseudo_groups.csv": counts["pseudo_task_text_identities"],
        "batch_grid_feasibility.csv": expectations.folds * len(expectations.tasks) * 2,
        "confirmation_donors.csv": counts["confirmation_donors"],
        "candidate_pools.csv": counts["candidate_pool_rows"],
    }
    for name, expected in csv_counts.items():
        require_equal(csv_row_count(root / name), expected, f"{name} row count")

    input_verification = report.get("input_verification", {})
    require_equal(input_verification.get("status"), "pass", "input verification status")
    require_equal(input_verification.get("preserved_source_id"), expectations.input_source_id, "report input source")
    require_equal(input_verification.get("combined_manifest_sha256"), expectations.input_manifest_sha256, "report input manifest")
    require_equal(input_verification.get("all_215_chunk_hashes_revalidated"), True, "input chunk validation")
    require_equal(input_verification.get("held_out_test_accessed"), False, "input test access")

    readiness = report.get("execution_readiness", {})
    require_equal(readiness.get("partition_pseudo_donor_and_pool_manifests_frozen"), True, "partition freeze")
    require_equal(readiness.get("batch_grid_cell_feasibility_checked"), True, "batch-grid feasibility")
    require_equal(readiness.get("full_40_epoch_batch_schedule_frozen"), False, "full schedule flag")
    require_equal(readiness.get("training_authorized"), False, "report training authorization")

    require_equal(metadata.get("status"), "pass", "metadata status")
    require_equal(metadata.get("protocol_commit"), expectations.protocol_commit, "metadata commit")
    require_equal(metadata.get("preserved_source_id"), expectations.input_source_id, "metadata input source")
    require_equal(metadata.get("input_manifest_sha256"), expectations.input_manifest_sha256, "metadata input manifest")
    require_equal(metadata.get("contract_sha256"), expectations.contract_sha256, "metadata contract")
    require_equal(metadata.get("protocol_report_sha256"), expectations.report_sha256, "metadata report")
    require_equal(metadata.get("artifact_sha256"), dict(expectations.artifact_sha256), "metadata artifact hashes")
    require_equal(metadata.get("training_authorized"), False, "metadata training authorization")
    require_equal(metadata.get("held_out_test_accessed"), False, "metadata test access")

    require_equal(registry.get("status"), "partition_and_evaluation_protocol_frozen_batch_schedule_pending", "registry status")
    require_equal(registry.get("contract_sha256"), expectations.contract_sha256, "registry contract")
    require_equal(registry.get("training_authorized"), False, "registry training authorization")
    require_equal(registry.get("held_out_test_accessed"), False, "registry test access")

    return {
        "status": "pass",
        "preserved_source_id": preserved_source_id,
        "protocol_commit": expectations.protocol_commit,
        "contract_sha256": expectations.contract_sha256,
        "protocol_report_sha256": expectations.report_sha256,
        "protocol_freeze_run_metadata_sha256": expectations.metadata_sha256,
        "counts": counts,
        "checks": dict(expectations.checks),
        "verified_artifact_sha256": dict(sorted(verified.items())),
        "training_authorized": False,
        "held_out_test_accessed": False,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)


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
    print("TASK-SEGMENTED PROTOCOL ARTIFACT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
