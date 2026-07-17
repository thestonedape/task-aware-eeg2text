"""Deterministic pass and tamper regression for the preserved-probe verifier."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.verify_frozen_factor_probe_artifact import (
    EXPECTED_CHECKS,
    EXPECTED_DECISIONS,
    EXPECTED_FACTORS,
    PRIMARY_ARTIFACTS,
    PRODUCTION_EXPECTATIONS,
    PROTOCOL_ARTIFACTS,
    sha256,
    verify,
)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_fixture(root: Path):
    root.mkdir()
    generic = {name: [{"factor_id": "tsr_instruction_relation", "value": "1"}] for name in PRIMARY_ARTIFACTS[:-1]}
    for name, rows in generic.items():
        write_csv(root / name, ["factor_id", "value"], rows)
    admission_rows = []
    for factor_id in EXPECTED_FACTORS:
        admission_rows.append({"factor_id": factor_id, **EXPECTED_DECISIONS[factor_id]})
    admission_fields = list(admission_rows[0])
    write_csv(root / "factor_admission.csv", admission_fields, admission_rows)

    protocol = root / "frozen_protocol"
    protocol.mkdir()
    write_csv(
        protocol / "recoverability_rows.csv",
        ["trial_id", "split"],
        [{"trial_id": "train-1", "split": "train"}, {"trial_id": "val-1", "split": "val"}],
    )
    for name in ("subject_folds.csv", "factor_fold_coverage.csv", "duplicated_sentence_consistency.csv"):
        write_csv(protocol / name, ["value"], [{"value": "fixture"}])
    (protocol / "recoverability_registry.json").write_text('{"status":"fixture"}\n', encoding="utf-8")
    protocol_hashes = {name: sha256(protocol / name) for name in PROTOCOL_ARTIFACTS}
    source_hash = "1" * 64
    vector_hash = "2" * 64
    vector_manifest_hash = "3" * 64
    protocol_report = {
        "status": "pass",
        "source": {"index_sha256": source_hash},
        "checks": {"held_out_test_accessed": False},
        "artifact_sha256": protocol_hashes,
    }
    (protocol / "recoverability_contract_report.json").write_text(
        json.dumps(protocol_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expectations = replace(
        PRODUCTION_EXPECTATIONS,
        source_index_sha256=source_hash,
        vector_index_sha256=vector_hash,
        vector_manifest_sha256=vector_manifest_hash,
        recoverability_registry_sha256=protocol_hashes["recoverability_registry.json"],
        recoverability_rows_sha256=protocol_hashes["recoverability_rows.csv"],
        project_commit="fixture-commit",
    )
    manifest = {
        "status": "pass",
        "run_mode": "full_development",
        "source_index_sha256": source_hash,
        "vector_index_sha256": vector_hash,
        "vector_manifest_sha256": vector_manifest_hash,
        "recoverability_registry_sha256": expectations.recoverability_registry_sha256,
        "recoverability_rows_sha256": expectations.recoverability_rows_sha256,
        "seeds": list(expectations.seeds),
        "bootstrap_replicates": expectations.bootstrap_replicates,
        "factors": list(EXPECTED_FACTORS),
        "ordinary_point_gain_candidates": list(expectations.point_gain_candidates),
        "admitted_factors": list(expectations.admitted_factors),
        "checks": EXPECTED_CHECKS,
        "artifact_sha256": {name: sha256(root / name) for name in PRIMARY_ARTIFACTS},
    }
    (root / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_metadata = {
        "status": "pass",
        "project_commit": expectations.project_commit,
        "dataset_index_sha256": source_hash,
        "vector_index_sha256": vector_hash,
        "test_accessed": False,
        "numpy": expectations.numpy_version,
        "scipy": expectations.scipy_version,
        "scikit_learn": expectations.sklearn_version,
    }
    (root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return expectations


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        artifact = Path(temporary) / "artifact"
        expectations = write_fixture(artifact)
        first = verify(artifact, expectations, preserved_source_id="fixture-version")
        second = verify(artifact, expectations, preserved_source_id="fixture-version")
        assert first == second and first["status"] == "pass"
        with (artifact / "probe_metrics.csv").open("a", encoding="utf-8") as handle:
            handle.write("tamper,1\n")
        try:
            verify(artifact, expectations, preserved_source_id="fixture-version")
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("tampered artifact unexpectedly passed")
    print("FROZEN FACTOR PROBE ARTIFACT VERIFICATION REGRESSION: PASS")


if __name__ == "__main__":
    main()
