"""Freeze the separately reviewed authorization for the complete P4b run.

The prospectively frozen full-execution contract is intentionally deny-by-
default.  This module is the only bridge from that inert contract to a launch
artifact accepted by :mod:`evaluation.run_task_segmented_full_shard`.  It
therefore refuses to infer trust from the working tree: the caller must supply
the independently reviewed SHA-256 for every executable source and notebook,
and every digest is compared with the bytes on disk before one JSON file is
written atomically.

No model, vector, schedule, checkpoint, validation row, or test row is read.
The module never invokes Git and never modifies a source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FULL_CONTRACT_PATH = Path(__file__).with_name(
    "task_segmented_full_execution_contract.json"
)
RUNNER_PATH = Path(__file__).with_name("run_task_segmented_full_shard.py")
ADAPTER_PATH = ROOT / "project_adapters" / "task_segmented_objective.py"
TASK_TREATMENT_PILOTS_PATH = (
    ROOT / "project_adapters" / "task_treatment_pilots.py"
)
SHARD_VERIFIER_PATH = Path(__file__).with_name(
    "verify_task_segmented_full_shard_artifact.py"
)
AGGREGATOR_PATH = Path(__file__).with_name(
    "aggregate_task_segmented_full_shards.py"
)
DECISION_ENGINE_PATH = Path(__file__).with_name(
    "decide_task_segmented_objective.py"
)
EXECUTION_NOTEBOOK_PATH = ROOT / "kaggle" / "run_task_segmented_full_shard.ipynb"
SHARD_VERIFICATION_NOTEBOOK_PATH = (
    ROOT / "kaggle" / "verify_task_segmented_full_shard_artifact.ipynb"
)
COMPLETE_MATRIX_AGGREGATION_NOTEBOOK_PATH = (
    ROOT / "kaggle" / "aggregate_task_segmented_full_shards.ipynb"
)
OUTPUT_NAME = "task_segmented_full_launch_authorization.json"
FULL_EXECUTION_CONTRACT_SHA256 = (
    "99c1235d21ce0dd9eb80b1c1c0c3930b3b7347007ebc35be3385f0bc253a837c"
)

EXPECTED_SHARD_IDS = tuple(
    f"p4b-f{fold}-s{seed}"
    for fold in range(5)
    for seed in (20260717, 20260718, 20260719)
)

PIN_PATH_KEYS = (
    "runner_source_sha256",
    "adapter_source_sha256",
    "task_treatment_pilots_source_sha256",
    "shard_verifier_source_sha256",
    "aggregator_source_sha256",
    "decision_engine_source_sha256",
    "execution_notebook_sha256",
    "shard_clean_remount_verification_notebook_sha256",
    "complete_matrix_aggregation_notebook_sha256",
)

EXACT_RUNTIME_ENVIRONMENT = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cu128",
    "torch_cuda": "12.8",
    "device": "cuda:0",
    "minimum_cuda_device_count": 1,
    "selected_cuda_device_index": 0,
    "selected_cuda_device_name": "Tesla T4",
    "selected_cuda_compute_capability": [7, 5],
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms_required": True,
    "full_scientific_cpu_execution_permitted": False,
    "runtime_fingerprint_bound_to_shard_and_resume": True,
}


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_project_commit(value: str) -> None:
    if (
        len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("project commit must be a full lowercase Git SHA")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_contract_payload(contract: Mapping[str, object]) -> None:
    """Recheck the launch-critical deny flags and exact shard matrix."""

    if contract.get("schema_version") != 1:
        raise ValueError("full-execution contract schema drifted")
    if contract.get("status") != "frozen_before_full_p4b_launch":
        raise ValueError("full-execution contract status drifted")

    authorization = contract.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("full-execution contract lacks authorization section")
    for key in (
        "full_training_authorized",
        "checkpoint_evaluation_authorized",
        "confirmation_evaluation_authorized",
        "scientific_decision_permitted",
        "held_out_test_accessed",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"base full contract must deny {key}")
    required = authorization.get("required_launch_authorization")
    if not isinstance(required, Mapping):
        raise ValueError("full-execution contract lacks launch requirement")
    exact_launch_requirement = {
        "artifact_name": OUTPUT_NAME,
        "sha256": None,
        "status": "not_supplied",
        "must_bind_full_execution_contract_sha256": True,
        "must_bind_runner_verifier_aggregator_and_notebook_hashes": True,
        "must_bind_transitive_task_treatment_adapter_source_sha256": True,
        "must_bind_scientific_decision_engine_source_sha256": True,
        "must_require_exact_clean_project_commit_at_execution": True,
        "must_bind_exact_runtime_environment": True,
        "activation_rule": (
            "full training remains false unless the separately reviewed "
            "launch artifact exists and its exact non-null SHA-256 is supplied "
            "out of band"
        ),
    }
    if dict(required) != exact_launch_requirement:
        raise ValueError("base full contract launch requirement drifted")

    if contract.get("runtime_environment") != EXACT_RUNTIME_ENVIRONMENT:
        raise ValueError("full-execution exact runtime environment drifted")

    sharding = contract.get("sharding")
    if not isinstance(sharding, Mapping):
        raise ValueError("full-execution contract lacks sharding section")
    shards = sharding.get("shards")
    if not isinstance(shards, list):
        raise ValueError("full-execution contract lacks shard list")
    observed_ids = tuple(
        row.get("shard_id") if isinstance(row, Mapping) else None
        for row in shards
    )
    if (
        sharding.get("shard_count") != 15
        or sharding.get("fits_per_shard") != 3
        or sharding.get("total_fits") != 45
        or observed_ids != EXPECTED_SHARD_IDS
    ):
        raise ValueError("full-execution shard matrix drifted")

    completion = contract.get("non_adaptive_completion")
    if not isinstance(completion, Mapping):
        raise ValueError("full-execution contract lacks completion policy")
    expected_completion = {
        "all_15_shards_required": True,
        "all_45_fits_required": True,
        "partial_result_scientific_inspection_permitted": False,
        "aggregation_before_all_shards_independently_verified": False,
        "scientific_decision_before_complete_matrix": False,
    }
    for key, value in expected_completion.items():
        if completion.get(key) is not value:
            raise ValueError(f"non-adaptive completion flag drifted: {key}")

    access = contract.get("data_access")
    if not isinstance(access, Mapping):
        raise ValueError("full-execution contract lacks data-access policy")
    for key in (
        "official_validation_rows_read",
        "official_validation_used_for_selection_confirmation_or_rescue",
        "held_out_test_rows_read",
        "held_out_test_accessed",
    ):
        if access.get(key) is not False:
            raise ValueError(f"base full contract must deny data access: {key}")


def load_exact_contract(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"full-execution contract is missing: {path}")
    actual = sha256(path)
    if actual != FULL_EXECUTION_CONTRACT_SHA256:
        raise ValueError(
            "full-execution contract SHA256 drifted: "
            f"expected {FULL_EXECUTION_CONTRACT_SHA256}, got {actual}"
        )
    contract = _read_json(path)
    validate_contract_payload(contract)
    return contract


def _validate_pins(
    source_paths: Mapping[str, Path], expected_sha256: Mapping[str, str]
) -> dict[str, str]:
    if tuple(source_paths) != PIN_PATH_KEYS:
        raise ValueError("source-path pin inventory drifted")
    if set(expected_sha256) != set(PIN_PATH_KEYS):
        missing = sorted(set(PIN_PATH_KEYS) - set(expected_sha256))
        extra = sorted(set(expected_sha256) - set(PIN_PATH_KEYS))
        raise ValueError(f"expected SHA256 pin inventory drifted: missing={missing}, extra={extra}")

    actual_sha256: dict[str, str] = {}
    for key in PIN_PATH_KEYS:
        expected = expected_sha256[key]
        if not _is_sha256(expected):
            raise ValueError(f"invalid externally pinned SHA256: {key}")
        path = source_paths[key]
        if not path.is_file():
            raise FileNotFoundError(f"required reviewed source is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"reviewed source drifted: {key}; expected {expected}, got {actual}"
            )
        actual_sha256[key] = actual
    return actual_sha256


def _atomic_write_new(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish one new JSON file; never replace prior evidence."""

    if path.name != OUTPUT_NAME:
        raise ValueError(f"launch authorization output must be named {OUTPUT_NAME}")
    if path.exists():
        raise FileExistsError(f"refusing to replace existing launch authorization: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {path.parent}")

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(
                f"refusing to replace existing launch authorization: {path}"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze_launch_authorization(
    *,
    contract_path: Path,
    project_commit: str,
    source_paths: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
    output_path: Path,
) -> dict[str, object]:
    """Validate every review pin and atomically emit the inert-to-live gate."""

    contract = load_exact_contract(contract_path)
    validate_project_commit(project_commit)
    actual_sha256 = _validate_pins(source_paths, expected_sha256)

    launch: dict[str, object] = {
        "schema_version": 1,
        "status": "authorized_for_full_p4b_launch",
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "project_commit": project_commit,
        "runtime_environment": dict(contract["runtime_environment"]),
        "authorized_shard_ids": list(EXPECTED_SHARD_IDS),
        "full_training_authorized": True,
        "checkpoint_evaluation_authorized": True,
        "confirmation_evaluation_authorized": True,
        "scientific_decision_permitted_after_complete_matrix_only": True,
        "partial_result_scientific_inspection_permitted": False,
        "official_validation_rows_read": False,
        "official_validation_used_for_confirmation": False,
        "held_out_test_rows_read": False,
        "held_out_test_accessed": False,
        **actual_sha256,
    }
    _atomic_write_new(output_path, launch)
    if sha256(output_path) != hashlib.sha256(
        (json.dumps(launch, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest():
        raise RuntimeError("published launch authorization failed its final hash check")
    return launch


def default_source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "runner_source_sha256": args.runner,
        "adapter_source_sha256": args.adapter,
        "task_treatment_pilots_source_sha256": args.task_treatment_pilots,
        "shard_verifier_source_sha256": args.shard_verifier,
        "aggregator_source_sha256": args.aggregator,
        "decision_engine_source_sha256": args.decision_engine,
        "execution_notebook_sha256": args.execution_notebook,
        "shard_clean_remount_verification_notebook_sha256": (
            args.shard_verification_notebook
        ),
        "complete_matrix_aggregation_notebook_sha256": (
            args.complete_matrix_aggregation_notebook
        ),
    }


def expected_pins_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        key: getattr(args, f"expected_{key}")
        for key in PIN_PATH_KEYS
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=FULL_CONTRACT_PATH)
    parser.add_argument("--runner", type=Path, default=RUNNER_PATH)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_PATH)
    parser.add_argument(
        "--task-treatment-pilots",
        type=Path,
        default=TASK_TREATMENT_PILOTS_PATH,
    )
    parser.add_argument("--shard-verifier", type=Path, default=SHARD_VERIFIER_PATH)
    parser.add_argument("--aggregator", type=Path, default=AGGREGATOR_PATH)
    parser.add_argument(
        "--decision-engine", type=Path, default=DECISION_ENGINE_PATH
    )
    parser.add_argument(
        "--execution-notebook", type=Path, default=EXECUTION_NOTEBOOK_PATH
    )
    parser.add_argument(
        "--shard-verification-notebook",
        type=Path,
        default=SHARD_VERIFICATION_NOTEBOOK_PATH,
    )
    parser.add_argument(
        "--complete-matrix-aggregation-notebook",
        type=Path,
        default=COMPLETE_MATRIX_AGGREGATION_NOTEBOOK_PATH,
    )
    for key in PIN_PATH_KEYS:
        parser.add_argument(
            "--expected-" + key.replace("_", "-"),
            dest=f"expected_{key}",
            required=True,
            help=f"externally reviewed lowercase SHA-256 for {key}",
        )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    launch = freeze_launch_authorization(
        contract_path=args.contract,
        project_commit=args.project_commit,
        source_paths=default_source_paths(args),
        expected_sha256=expected_pins_from_args(args),
        output_path=args.output,
    )
    print(json.dumps({
        "status": launch["status"],
        "output": str(args.output),
        "launch_authorization_sha256": sha256(args.output),
        "authorized_shard_count": len(launch["authorized_shard_ids"]),
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
