"""Verify a preserved frozen factor-probe artifact without refitting any model.

The verifier is intentionally standard-library only.  It re-hashes every file
declared by the probe manifest, validates the copied frozen protocol, enforces
the predeclared safety and admission decisions, and writes a compact report for
the exact immutable Kaggle notebook-output version used as its input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIMARY_ARTIFACTS = (
    "probe_metrics.csv",
    "probe_selection.csv",
    "probe_predictions.csv",
    "planned_contrasts.csv",
    "factor_admission.csv",
)
PROTOCOL_ARTIFACTS = (
    "recoverability_rows.csv",
    "subject_folds.csv",
    "factor_fold_coverage.csv",
    "duplicated_sentence_consistency.csv",
    "recoverability_registry.json",
)
EXPECTED_FACTORS = (
    "length_words_whitespace_v1",
    "nr_relation_content",
    "sr_sentiment_3",
    "tsr_instruction_relation",
)
EXPECTED_CHECKS = {
    "held_out_test_accessed": False,
    "validation_tuning_permitted": False,
    "target_label_as_probe_input_permitted": False,
    "matched_wrong_model_fit_on_correct_training_vectors": True,
    "zero_and_gaussian_model_fit_on_correct_training_vectors": True,
    "subject_folds_refit_without_held_out_subject": True,
    "null_results_retained": True,
}
EXPECTED_DECISIONS: dict[str, dict[str, Any]] = {
    "length_words_whitespace_v1": {
        "validation_rows": 2200,
        "validation_subjects": 30,
        "ordinary_correct_minus_metadata": 1.2757545020363548,
        "ordinary_correct_minus_matched_wrong": 0.02320589412342411,
        "all_cluster_holm_lower_correct_minus_metadata_positive": 1,
        "all_cluster_holm_lower_correct_minus_matched_wrong_positive": 0,
        "subject_held_out_correct_minus_metadata": 1.3234099509499284,
        "subject_held_out_correct_minus_matched_wrong": 0.02601344542069839,
        "subject_held_out_same_direction": 1,
        "admitted": 0,
        "decision": "reject_retain_null",
    },
    "nr_relation_content": {
        "validation_rows": 290,
        "validation_subjects": 12,
        "ordinary_correct_minus_metadata": -0.013567203975470225,
        "ordinary_correct_minus_matched_wrong": -0.002452192436549268,
        "all_cluster_holm_lower_correct_minus_metadata_positive": 0,
        "all_cluster_holm_lower_correct_minus_matched_wrong_positive": 0,
        "subject_held_out_correct_minus_metadata": "",
        "subject_held_out_correct_minus_matched_wrong": "",
        "subject_held_out_same_direction": 0,
        "admitted": 0,
        "decision": "reject_retain_null",
    },
    "sr_sentiment_3": {
        "validation_rows": 406,
        "validation_subjects": 12,
        "ordinary_correct_minus_metadata": 0.0196890832457835,
        "ordinary_correct_minus_matched_wrong": -0.01038158918431814,
        "all_cluster_holm_lower_correct_minus_metadata_positive": 0,
        "all_cluster_holm_lower_correct_minus_matched_wrong_positive": 0,
        "subject_held_out_correct_minus_metadata": "",
        "subject_held_out_correct_minus_matched_wrong": "",
        "subject_held_out_same_direction": 0,
        "admitted": 0,
        "decision": "reject_retain_null",
    },
    "tsr_instruction_relation": {
        "validation_rows": 970,
        "validation_subjects": 30,
        "ordinary_correct_minus_metadata": 0.21150562332012296,
        "ordinary_correct_minus_matched_wrong": 0.18359389786311134,
        "all_cluster_holm_lower_correct_minus_metadata_positive": 1,
        "all_cluster_holm_lower_correct_minus_matched_wrong_positive": 1,
        "subject_held_out_correct_minus_metadata": 0.25693814738050613,
        "subject_held_out_correct_minus_matched_wrong": 0.1784560294983573,
        "subject_held_out_same_direction": 1,
        "admitted": 1,
        "decision": "admit",
    },
}


@dataclass(frozen=True)
class VerificationExpectations:
    source_index_sha256: str
    vector_index_sha256: str
    vector_manifest_sha256: str
    recoverability_registry_sha256: str
    recoverability_rows_sha256: str
    project_commit: str
    admitted_factors: tuple[str, ...] = ("tsr_instruction_relation",)
    point_gain_candidates: tuple[str, ...] = (
        "length_words_whitespace_v1",
        "tsr_instruction_relation",
    )
    seeds: tuple[int, ...] = (20260716, 20260717, 20260718, 20260719, 20260720)
    bootstrap_replicates: int = 5000
    numpy_version: str = "2.2.6"
    scipy_version: str = "1.15.3"
    sklearn_version: str = "1.7.2"


PRODUCTION_EXPECTATIONS = VerificationExpectations(
    source_index_sha256="bdaaaf5c91d3c9eec16a0727825da996fd2186867245951bfdfdc92aab7738b0",
    vector_index_sha256="65d4a1f38f0df801f17ccb5504b4ee5d2f43aedcbc051e1ae1f405684ccad0e2",
    vector_manifest_sha256="4861d9439a8a1253d87f38524a3720ea61d0934934589434f869b73263d36eba",
    recoverability_registry_sha256="e4afa8a1bc859b86ad786016400035bec6d4a5280350abe271d0d74186af6517",
    recoverability_rows_sha256="4cdb7899088d32c37671bbd2088f650213ba001a771738a66a85660feee67f7d",
    project_commit="d782daa4183aec4df1057f2605535cfdc182bb83",
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
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_hash(path: Path, expected: str, label: str) -> str:
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch: {actual} != {expected}")
    return actual


def compare_value(factor_id: str, field: str, actual: str, expected: Any) -> None:
    label = f"{factor_id}.{field}"
    if isinstance(expected, float):
        try:
            value = float(actual)
        except ValueError as error:
            raise ValueError(f"{label} is not numeric: {actual!r}") from error
        require(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12), f"{label} mismatch")
    elif isinstance(expected, int):
        try:
            value = int(actual)
        except ValueError as error:
            raise ValueError(f"{label} is not an integer: {actual!r}") from error
        require(value == expected, f"{label} mismatch: {value} != {expected}")
    else:
        require(actual == expected, f"{label} mismatch: {actual!r} != {expected!r}")


def verify(
    artifact_root: Path,
    expectations: VerificationExpectations = PRODUCTION_EXPECTATIONS,
    preserved_source_id: str = "",
) -> dict[str, Any]:
    root = artifact_root.resolve()
    require(root.is_dir(), f"artifact root is not a directory: {root}")
    required = {
        "probe_manifest.json", "run_metadata.json", "frozen_protocol", *PRIMARY_ARTIFACTS
    }
    missing = sorted(name for name in required if not (root / name).exists())
    require(not missing, f"artifact is incomplete; missing {missing}")

    manifest_path = root / "probe_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "pass", "probe manifest status is not pass")
    require(manifest.get("run_mode") == "full_development", "unexpected probe run mode")
    require(manifest.get("source_index_sha256") == expectations.source_index_sha256, "source index mismatch")
    require(manifest.get("vector_index_sha256") == expectations.vector_index_sha256, "vector index mismatch")
    require(manifest.get("vector_manifest_sha256") == expectations.vector_manifest_sha256, "vector manifest mismatch")
    require(
        manifest.get("recoverability_registry_sha256") == expectations.recoverability_registry_sha256,
        "recoverability registry mismatch",
    )
    require(
        manifest.get("recoverability_rows_sha256") == expectations.recoverability_rows_sha256,
        "recoverability rows mismatch",
    )
    require(tuple(manifest.get("factors", [])) == EXPECTED_FACTORS, "factor ordering/content mismatch")
    require(tuple(manifest.get("ordinary_point_gain_candidates", [])) == expectations.point_gain_candidates, "point-gain candidates mismatch")
    require(tuple(manifest.get("admitted_factors", [])) == expectations.admitted_factors, "admitted factors mismatch")
    require(tuple(manifest.get("seeds", [])) == expectations.seeds, "probe seeds mismatch")
    require(manifest.get("bootstrap_replicates") == expectations.bootstrap_replicates, "bootstrap count mismatch")
    require(manifest.get("checks") == EXPECTED_CHECKS, "probe safety checks mismatch")

    declared_hashes = manifest.get("artifact_sha256")
    require(isinstance(declared_hashes, dict), "manifest artifact hashes are missing")
    require(set(declared_hashes) == set(PRIMARY_ARTIFACTS), "manifest artifact set mismatch")
    verified_hashes: dict[str, str] = {}
    for name in PRIMARY_ARTIFACTS:
        verified_hashes[name] = require_hash(root / name, str(declared_hashes[name]), name)
        require(read_csv(root / name), f"{name} is empty")

    admission_rows = read_csv(root / "factor_admission.csv")
    require(len(admission_rows) == len(EXPECTED_DECISIONS), "factor admission row count mismatch")
    admissions = {row.get("factor_id", ""): row for row in admission_rows}
    require(set(admissions) == set(EXPECTED_DECISIONS), "factor admission identities mismatch")
    for factor_id, expected_fields in EXPECTED_DECISIONS.items():
        row = admissions[factor_id]
        for field, expected in expected_fields.items():
            require(field in row, f"missing admission field {factor_id}.{field}")
            compare_value(factor_id, field, row[field], expected)
    admitted_from_csv = tuple(
        factor_id for factor_id in EXPECTED_FACTORS if int(admissions[factor_id]["admitted"])
    )
    require(admitted_from_csv == expectations.admitted_factors, "CSV and manifest admissions differ")

    metadata_path = root / "run_metadata.json"
    metadata = read_json(metadata_path)
    require(metadata.get("status") == "pass", "run metadata status is not pass")
    require(metadata.get("project_commit") == expectations.project_commit, "project commit mismatch")
    require(metadata.get("dataset_index_sha256") == expectations.source_index_sha256, "metadata dataset hash mismatch")
    require(metadata.get("vector_index_sha256") == expectations.vector_index_sha256, "metadata vector hash mismatch")
    require(metadata.get("test_accessed") is False, "run metadata reports test access")
    require(metadata.get("numpy") == expectations.numpy_version, "NumPy version mismatch")
    require(metadata.get("scipy") == expectations.scipy_version, "SciPy version mismatch")
    require(metadata.get("scikit_learn") == expectations.sklearn_version, "scikit-learn version mismatch")

    protocol_root = root / "frozen_protocol"
    protocol_report_path = protocol_root / "recoverability_contract_report.json"
    protocol_report = read_json(protocol_report_path)
    require(protocol_report.get("status") == "pass", "frozen protocol status is not pass")
    require(protocol_report.get("source", {}).get("index_sha256") == expectations.source_index_sha256, "protocol source mismatch")
    require(protocol_report.get("checks", {}).get("held_out_test_accessed") is False, "protocol reports test access")
    protocol_hashes = protocol_report.get("artifact_sha256")
    require(isinstance(protocol_hashes, dict), "protocol artifact hashes are missing")
    require(set(protocol_hashes) == set(PROTOCOL_ARTIFACTS), "protocol artifact set mismatch")
    for name in PROTOCOL_ARTIFACTS:
        relative = f"frozen_protocol/{name}"
        verified_hashes[relative] = require_hash(protocol_root / name, str(protocol_hashes[name]), relative)
    require(
        verified_hashes["frozen_protocol/recoverability_registry.json"] == expectations.recoverability_registry_sha256,
        "copied registry does not match the frozen probe input",
    )
    require(
        verified_hashes["frozen_protocol/recoverability_rows.csv"] == expectations.recoverability_rows_sha256,
        "copied recoverability rows do not match the frozen probe input",
    )
    protocol_rows = read_csv(protocol_root / "recoverability_rows.csv")
    require(protocol_rows and all(row.get("split") in {"train", "val"} for row in protocol_rows), "test or invalid split entered frozen protocol")

    verified_hashes["probe_manifest.json"] = sha256(manifest_path)
    verified_hashes["run_metadata.json"] = sha256(metadata_path)
    verified_hashes["frozen_protocol/recoverability_contract_report.json"] = sha256(protocol_report_path)
    return {
        "status": "pass",
        "schema_version": 1,
        "preserved_source_id": preserved_source_id,
        "project_commit": expectations.project_commit,
        "source_index_sha256": expectations.source_index_sha256,
        "vector_index_sha256": expectations.vector_index_sha256,
        "probe_manifest_sha256": verified_hashes["probe_manifest.json"],
        "run_metadata_sha256": verified_hashes["run_metadata.json"],
        "admitted_factors": list(expectations.admitted_factors),
        "null_factors": [factor for factor in EXPECTED_FACTORS if factor not in expectations.admitted_factors],
        "checks": {
            "all_declared_hashes_revalidated": True,
            "frozen_protocol_revalidated": True,
            "held_out_test_accessed": False,
            "admission_decisions_match_frozen_log": True,
        },
        "verified_artifact_sha256": dict(sorted(verified_hashes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--preserved-source-id", default="")
    args = parser.parse_args()
    report = verify(args.artifact_root, preserved_source_id=args.preserved_source_id)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
