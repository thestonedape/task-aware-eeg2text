"""Run one authorized P4b full-scientific fold/seed shard.

One invocation is exactly one frozen outer-fold/training-seed unit and always
runs all three paired arms.  This module is intentionally separate from the
bounded-smoke runner.  It refuses to touch the vector/schedule inputs or create
an output directory until both the prospectively frozen full-execution
contract and a separately supplied launch-authorization artifact validate.

The held-out test and official validation splits are never inputs.  Fitting,
checkpoint selection, and confirmation all use the frozen canonical-training
cross-fit roles.  Confirmation runs once, after epoch 40 and final checkpoint
selection, under the correct and frozen matched-wrong real-EEG conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FULL_CONTRACT_PATH = Path(__file__).with_name(
    "task_segmented_full_execution_contract.json"
)
RUNNER_PATH = Path(__file__)
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
LOCAL_LAUNCH_PIN_PATHS = {
    "runner_source_sha256": RUNNER_PATH,
    "adapter_source_sha256": ADAPTER_PATH,
    "task_treatment_pilots_source_sha256": TASK_TREATMENT_PILOTS_PATH,
    "shard_verifier_source_sha256": SHARD_VERIFIER_PATH,
    "aggregator_source_sha256": AGGREGATOR_PATH,
    "decision_engine_source_sha256": DECISION_ENGINE_PATH,
}
NOTEBOOK_LAUNCH_PIN_PATHS = {
    "execution_notebook_sha256": EXECUTION_NOTEBOOK_PATH,
    "shard_clean_remount_verification_notebook_sha256": (
        SHARD_VERIFICATION_NOTEBOOK_PATH
    ),
    "complete_matrix_aggregation_notebook_sha256": (
        COMPLETE_MATRIX_AGGREGATION_NOTEBOOK_PATH
    ),
}
FULL_EXECUTION_CONTRACT_SHA256 = (
    "99c1235d21ce0dd9eb80b1c1c0c3930b3b7347007ebc35be3385f0bc253a837c"
)
FULL_MODE = "full_scientific"
SHARD_MODE = "full_scientific_shard"
ARMS = (
    "global_mixed",
    "true_task_segmented",
    "pseudo_task_segmented",
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

CATALOG_FIELDS = (
    "trial_index", "trial_id", "text_fold", "dataset_version", "reading_task",
    "subject_id", "normalized_text_sha256", "text_target_id", "pseudo_group",
    "length_words_whitespace_v1", "eeg_vector_file", "eeg_vector_offset",
    "eeg_vector_dim",
)
UNIT_FIELDS = (
    "unit_index", "outer_fold", "training_seed", "epochs",
    "batches_per_epoch", "batch_size", "uint32_count", "byte_offset",
    "byte_length", "unit_sha256", "initialization_seed",
)
ASSIGNMENT_FIELDS = (
    "outer_fold", "role", "text_fold", "trial_id", "dataset_version",
    "reading_task", "subject_id", "normalized_text_sha256", "text_target_id",
    "pseudo_group", "length_words_whitespace_v1", "eeg_vector_file",
    "eeg_vector_offset", "eeg_vector_dim",
)
EEG_INDEX_FIELDS = (
    "condition", "phase", "target_trial_id", "signal_trial_id",
    "target_source_dataframe_row_index", "signal_source_dataframe_row_index",
    "dataset_version", "reading_task", "subject_id", "text_uid", "vector_file",
    "vector_offset", "vector_dim", "prompt_mode", "checkpoint_sha256",
    "source_index_sha256",
)
TEXT_INDEX_FIELDS = (
    "text_target_id", "normalized_text_sha256", "representative_trial_id",
    "representative_text", "vector_file", "vector_offset", "vector_dim",
    "checkpoint_sha256", "source_index_sha256",
)
TEXT_MAPPING_FIELDS = (
    "trial_id", "split", "cohort", "dataset_version", "reading_task",
    "subject_id", "source_dataframe_row_index", "text_target_id",
    "normalized_text_sha256",
)
CANDIDATE_FIELDS = (
    "outer_fold", "partition", "target_trial_id", "candidate_rank",
    "candidate_normalized_text_sha256", "candidate_text_target_id",
    "is_positive", "is_designated_donor_text", "dataset_version",
    "reading_task", "target_length", "candidate_length",
    "absolute_length_difference", "selection_rule",
)
DONOR_FIELDS = (
    "outer_fold", "partition", "target_trial_id", "donor_trial_id",
    "dataset_version", "reading_task", "subject_id",
    "target_normalized_text_sha256", "donor_normalized_text_sha256",
    "target_length", "donor_length", "absolute_length_difference",
    "selection_rule",
)

RUN_MANIFEST_FIELDS = (
    "arm_id", "outer_fold", "training_seed", "status", "run_mode",
    "full_training_authorized", "scientific_decision_permitted",
    "official_validation_used_for_confirmation", "held_out_test_accessed",
)
CONFIRMATION_PREDICTION_FIELDS = (
    "arm_id", "training_seed", "outer_fold", "trial_id", "reading_task",
    "subject_id", "normalized_text_sha256", "signal_condition",
    "positive_rank", "candidate_pool_size", "scientific_decision_permitted",
)
HISTORY_FIELDS = (
    "arm_id", "outer_fold", "training_seed", "epoch", "optimizer_steps",
    "mean_train_loss", "checkpoint_macro_mrr", "checkpoint_nr_mrr",
    "checkpoint_tsr_mrr", "is_best",
)
STEP_TRACE_FIELDS = (
    "outer_fold", "training_seed", "arm_id", "epoch", "batch_index",
    "global_step", "schedule_unit_sha256", "batch_catalog_indices_sha256",
    "batch_trial_ids_sha256", "global_mask_key", "initial_state_sha256",
    "pre_step_state_sha256", "loss", "gradient_norm_preclip",
    "post_step_state_sha256", "optimizer_state_sha256",
)
CHECKPOINT_RANK_FIELDS = (
    "arm_id", "outer_fold", "training_seed", "epoch", "trial_id",
    "reading_task", "positive_rank", "candidate_pool_size",
)
CONFIRMATION_SCORE_FIELDS = (
    "arm_id", "training_seed", "outer_fold", "trial_id", "reading_task",
    "subject_id", "normalized_text_sha256", "signal_condition",
    "signal_trial_id", "signal_subject_id", "signal_normalized_text_sha256",
    "candidate_rank", "candidate_normalized_text_sha256",
    "candidate_text_target_id", "is_positive", "is_designated_donor_text",
    "score",
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"unexpected CSV header: {path.name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"malformed CSV: {path.name}")
    return rows


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _atomic_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    _atomic_bytes(path, _csv_bytes(fields, rows))


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _as_int(value: object, label: str) -> int:
    try:
        parsed = int(str(value), 10)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer {label}: {value!r}") from exc
    return parsed


def _as_float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float {label}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite float {label}: {value!r}")
    return parsed


def validate_project_commit(value: str) -> None:
    if (
        len(value) != 40
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("project commit must be a full lowercase Git SHA")


def load_full_contract(path: Path = FULL_CONTRACT_PATH) -> dict[str, Any]:
    if sha256(path) != FULL_EXECUTION_CONTRACT_SHA256:
        raise ValueError("full-execution contract SHA256 drifted")
    contract = _json(path)
    validate_full_contract(contract)
    return contract


def validate_full_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != 1:
        raise ValueError("full-execution contract schema drifted")
    if contract.get("status") != "frozen_before_full_p4b_launch":
        raise ValueError("full-execution contract is not prospectively frozen")
    sharding = contract.get("sharding")
    execution = contract.get("execution")
    evaluation = contract.get("checkpoint_and_confirmation")
    access = contract.get("data_access")
    resume = contract.get("resume")
    output = contract.get("output")
    runtime = contract.get("runtime_environment")
    authorization = contract.get("authorization")
    if not all(isinstance(item, dict) for item in (
        sharding, execution, evaluation, access, resume, output, runtime,
        authorization,
    )):
        raise ValueError("full-execution contract sections are missing")
    if runtime != EXACT_RUNTIME_ENVIRONMENT:
        raise ValueError("full-execution exact runtime environment drifted")
    expected_pairs = [
        (fold, seed)
        for fold in range(5)
        for seed in (20260717, 20260718, 20260719)
    ]
    shards = sharding.get("shards")
    if not isinstance(shards, list) or len(shards) != 15:
        raise ValueError("full execution requires exactly 15 shards")
    observed = []
    for index, row in enumerate(shards):
        fold, seed = expected_pairs[index]
        expected = {
            "shard_id": f"p4b-f{fold}-s{seed}",
            "schedule_unit_index": index,
            "outer_fold": fold,
            "training_seed": seed,
            "confirmation_fold": fold,
            "checkpoint_fold": (fold + 1) % 5,
            "fit_folds": [candidate for candidate in range(5)
                          if candidate not in {fold, (fold + 1) % 5}],
        }
        if row != expected:
            raise ValueError(f"full shard definition drifted at unit {index}")
        observed.append((fold, seed))
    if observed != expected_pairs or sharding.get("arms_per_shard") != list(ARMS):
        raise ValueError("full shard matrix/arms drifted")
    exact_execution = {
        "epochs": 40, "batches_per_epoch": 105, "batch_size": 64,
        "optimizer_steps_per_arm": 4200, "optimizer_steps_per_shard": 12600,
        "total_optimizer_steps": 189000, "trainable_parameters_per_arm": 196608,
    }
    for key, value in exact_execution.items():
        if execution.get(key) != value:
            raise ValueError(f"full execution setting drifted: {key}")
    if (
        execution.get("vector_dtype") != "float32"
        or execution.get("automatic_mixed_precision") is not False
        or execution.get("early_stopping") is not False
        or execution.get("hyperparameter_search") is not False
    ):
        raise ValueError("full precision/adaptivity settings drifted")
    if evaluation.get("checkpoint_evaluation_epochs") != list(range(41)):
        raise ValueError("checkpoint epochs drifted")
    if evaluation.get("confirmation_signal_conditions") != ["correct", "matched_wrong"]:
        raise ValueError("confirmation conditions drifted")
    if evaluation.get("candidate_pool_size") != 24:
        raise ValueError("candidate pool size drifted")
    if any(access.get(key) is not False for key in (
        "official_validation_rows_read",
        "official_validation_used_for_selection_confirmation_or_rescue",
        "held_out_test_rows_read", "held_out_test_accessed",
    )):
        raise ValueError("forbidden validation/test access entered full contract")
    if resume.get("checkpoint_boundary") != "after every completed epoch and at arm completion":
        raise ValueError("resume boundary drifted")
    if output.get("exact_top_level_files") != [
        "full_shard_manifest.json", "shard_run_metadata.json",
        "run_manifest.csv", "confirmation_predictions.csv",
    ]:
        raise ValueError("top-level full-shard inventory drifted")
    if output.get("exact_top_level_directories") != ["runs"]:
        raise ValueError("full-shard directory inventory drifted")
    if output.get("exact_per_arm_files") != [
        "resume_checkpoint.pt", "best_checkpoint.pt", "training_history.csv",
        "step_trace.csv", "checkpoint_positive_ranks.csv",
        "confirmation_candidate_scores.csv", "run_summary.json",
    ]:
        raise ValueError("per-arm full-shard inventory drifted")
    if output.get("run_manifest_required_fields") != list(RUN_MANIFEST_FIELDS):
        raise ValueError("run-manifest schema drifted")
    if output.get("confirmation_predictions_required_fields") != list(
        CONFIRMATION_PREDICTION_FIELDS
    ):
        raise ValueError("confirmation-prediction schema drifted")
    if output.get("confirmation_candidate_scores_identity_fields") != [
        "signal_trial_id", "signal_subject_id", "signal_normalized_text_sha256"
    ]:
        raise ValueError("confirmation provenance schema drifted")
    if any(authorization.get(key) is not False for key in (
        "full_training_authorized", "checkpoint_evaluation_authorized",
        "confirmation_evaluation_authorized", "scientific_decision_permitted",
        "held_out_test_accessed",
    )):
        raise ValueError("base full contract must remain deny-by-default")
    required = authorization.get("required_launch_authorization")
    exact_launch_requirement = {
        "artifact_name": "task_segmented_full_launch_authorization.json",
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
    if not isinstance(required, dict) or required != exact_launch_requirement:
        raise ValueError("base contract unexpectedly embeds launch authorization")


def authorize_launch(
    contract: Mapping[str, Any],
    launch_path: Path,
    supplied_sha256: str,
    project_commit: str,
    shard_id: str,
) -> dict[str, Any]:
    """Validate the separate launch artifact before any scientific input I/O."""

    validate_full_contract(contract)
    validate_project_commit(project_commit)
    if not _is_sha256(supplied_sha256):
        raise PermissionError("a lowercase launch-authorization SHA256 is required")
    if not launch_path.is_file() or sha256(launch_path) != supplied_sha256:
        raise PermissionError("launch-authorization artifact/hash mismatch")
    launch = _json(launch_path)
    expected_ids = [row["shard_id"] for row in contract["sharding"]["shards"]]
    exact = {
        "schema_version": 1,
        "status": "authorized_for_full_p4b_launch",
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "project_commit": project_commit,
        "runtime_environment": dict(contract["runtime_environment"]),
        "authorized_shard_ids": expected_ids,
        "full_training_authorized": True,
        "checkpoint_evaluation_authorized": True,
        "confirmation_evaluation_authorized": True,
        "scientific_decision_permitted_after_complete_matrix_only": True,
        "partial_result_scientific_inspection_permitted": False,
        "official_validation_rows_read": False,
        "official_validation_used_for_confirmation": False,
        "held_out_test_rows_read": False,
        "held_out_test_accessed": False,
    }
    exact_fields = (
        set(exact) | set(LOCAL_LAUNCH_PIN_PATHS)
        | set(NOTEBOOK_LAUNCH_PIN_PATHS)
    )
    if set(launch) != exact_fields:
        raise PermissionError("launch authorization field inventory drifted")
    for key, value in exact.items():
        if launch.get(key) != value:
            raise PermissionError(f"launch authorization drifted: {key}")
    for key, path in LOCAL_LAUNCH_PIN_PATHS.items():
        if not _is_sha256(launch.get(key)):
            raise PermissionError(f"launch authorization lacks pinned {key}")
        if launch[key] != sha256(path):
            raise PermissionError(f"launch authorization {key} source drifted")
    for key, path in NOTEBOOK_LAUNCH_PIN_PATHS.items():
        if not _is_sha256(launch.get(key)):
            raise PermissionError(f"launch authorization lacks pinned {key}")
        if launch[key] != sha256(path):
            raise PermissionError(f"launch authorization {key} source drifted")
    if shard_id not in expected_ids:
        raise PermissionError(f"unfrozen full shard: {shard_id}")
    return launch


def _git_output(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify the Git execution boundary") from exc
    return completed.stdout.rstrip("\r\n")


def verify_git_execution_boundary(
    project_commit: str, project_root: Path = ROOT,
) -> dict[str, object]:
    """Require the authorized checkout with no tracked, untracked, or ignored drift."""

    validate_project_commit(project_commit)
    actual_root = Path(
        _git_output(project_root, "rev-parse", "--show-toplevel")
    ).resolve()
    if actual_root != project_root.resolve():
        raise PermissionError("Git top-level directory drifted")
    actual_commit = _git_output(project_root, "rev-parse", "HEAD")
    if actual_commit != project_commit:
        raise PermissionError("Git HEAD does not equal the authorized project commit")
    worktree_status = _git_output(
        project_root, "status", "--porcelain=v1", "--untracked-files=all",
        "--ignored=matching",
    )
    if worktree_status:
        raise PermissionError("Git worktree is not clean")
    submodule_status = _git_output(
        project_root, "submodule", "status", "--recursive"
    )
    if any(not line.startswith(" ") for line in submodule_status.splitlines()):
        raise PermissionError("Git submodule state is missing or drifted")
    return {
        "project_commit": actual_commit,
        "git_top_level": actual_root.as_posix(),
        "git_worktree_clean": True,
        "git_submodules_clean": True,
    }


def _runtime_fingerprint_sha256(runtime_fingerprint: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(runtime_fingerprint), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _require_exact_runtime_fingerprint(
    expected: Mapping[str, Any], observed: Mapping[str, Any],
    *, requested_device: str, current_cuda_device: int | None,
) -> dict[str, Any]:
    if requested_device != "cuda:0":
        raise RuntimeError("full scientific execution requires exactly cuda:0")
    if current_cuda_device != 0:
        raise RuntimeError("full scientific execution requires CUDA device index 0")
    exact_observed_fields = {
        "python", "numpy", "torch", "torch_cuda", "device",
        "cuda_device_count", "selected_cuda_device_index",
        "selected_cuda_device_name", "selected_cuda_compute_capability",
        "cublas_workspace_config", "deterministic_algorithms_enabled",
    }
    if set(observed) != exact_observed_fields:
        raise RuntimeError("runtime fingerprint field inventory drifted")
    comparisons = {
        "python": expected.get("python"),
        "numpy": expected.get("numpy"),
        "torch": expected.get("torch"),
        "torch_cuda": expected.get("torch_cuda"),
        "device": expected.get("device"),
        "selected_cuda_device_index": expected.get(
            "selected_cuda_device_index"
        ),
        "selected_cuda_device_name": expected.get("selected_cuda_device_name"),
        "selected_cuda_compute_capability": expected.get(
            "selected_cuda_compute_capability"
        ),
        "cublas_workspace_config": expected.get("cublas_workspace_config"),
        "deterministic_algorithms_enabled": expected.get(
            "deterministic_algorithms_required"
        ),
    }
    drift = sorted(
        key for key, value in comparisons.items()
        if observed.get(key) != value
    )
    count = observed.get("cuda_device_count")
    minimum_count = expected.get("minimum_cuda_device_count")
    if (
        not isinstance(count, int) or isinstance(count, bool)
        or not isinstance(minimum_count, int) or isinstance(minimum_count, bool)
        or count < minimum_count
    ):
        drift.append("cuda_device_count")
    if drift:
        raise RuntimeError(
            "exact Kaggle runtime fingerprint drifted: " + ", ".join(drift)
        )
    # Normalize away the count of unused devices.  A Kaggle T4x2 and T4x1
    # session have the same computational boundary because only cuda:0 is ever
    # selected; the observed count is retained separately as provenance.
    return dict(expected)


def inspect_exact_runtime(
    contract: Mapping[str, Any], requested_device: str,
) -> tuple[dict[str, Any], int]:
    """Inspect and require the frozen selected-Tesla-T4 Kaggle runtime."""

    _configure_cuda_determinism_environment()

    import numpy as np
    import torch

    if requested_device != "cuda:0":
        raise RuntimeError("full scientific CPU/non-cuda:0 execution is prohibited")
    if not torch.cuda.is_available():
        raise RuntimeError("exact full-scientific CUDA runtime is unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    count = int(torch.cuda.device_count())
    current_device = int(torch.cuda.current_device()) if count else None
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": requested_device,
        "cuda_device_count": count,
        "selected_cuda_device_index": current_device,
        "selected_cuda_device_name": (
            torch.cuda.get_device_name(0) if count else None
        ),
        "selected_cuda_compute_capability": (
            list(torch.cuda.get_device_capability(0)) if count else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    expected = contract.get("runtime_environment")
    if not isinstance(expected, dict):
        raise RuntimeError("full contract lacks an exact runtime fingerprint")
    normalized = _require_exact_runtime_fingerprint(
        expected, observed, requested_device=requested_device,
        current_cuda_device=current_device,
    )
    return normalized, count


def shard_definition(contract: Mapping[str, Any], shard_id: str) -> dict[str, Any]:
    matches = [row for row in contract["sharding"]["shards"]
               if row["shard_id"] == shard_id]
    if len(matches) != 1:
        raise ValueError(f"unknown/duplicate full shard: {shard_id}")
    return dict(matches[0])


def initialization_seed(outer_fold: int, training_seed: int) -> int:
    return int(stable_hash(
        "p4b-adapter-init-v1", outer_fold, training_seed
    )[:16], 16) & ((1 << 63) - 1)


def global_mask_key(
    schedule_contract_sha256: str,
    schedule_unit_sha256: str,
    outer_fold: int,
    training_seed: int,
    epoch: int,
    batch_index: int,
) -> str:
    return stable_hash(
        "p4b-global-mask-key-v1", 2026071806, schedule_contract_sha256,
        schedule_unit_sha256, outer_fold, training_seed, epoch, batch_index,
    )


def positive_rank_from_scores(
    scores: Sequence[float], candidate_ranks: Sequence[int], positive_offset: int
) -> int:
    """Rank a positive by descending score and frozen candidate rank on ties."""

    if len(scores) != len(candidate_ranks) or not scores:
        raise ValueError("score/candidate-rank cardinality mismatch")
    if sorted(candidate_ranks) != list(range(len(candidate_ranks))):
        raise ValueError("candidate ranks must be exactly 0..pool_size-1")
    if not 0 <= positive_offset < len(scores):
        raise ValueError("positive candidate offset is invalid")
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("candidate scores must be finite")
    order = sorted(
        range(len(scores)), key=lambda index: (-float(scores[index]), candidate_ranks[index])
    )
    return order.index(positive_offset) + 1


def strict_best_epoch(
    incumbent_epoch: int, incumbent_metric: float, epoch: int, metric: float
) -> tuple[int, float, bool]:
    """Apply strict-greater replacement; exact ties retain the earlier epoch."""

    if epoch < incumbent_epoch or not all(map(math.isfinite, (incumbent_metric, metric))):
        raise ValueError("invalid checkpoint comparison")
    if metric > incumbent_metric:
        return epoch, metric, True
    return incumbent_epoch, incumbent_metric, False


def _unique(
    rows: Sequence[dict[str, str]], key: str, label: str
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        identity = row[key]
        if identity in output:
            raise ValueError(f"duplicate {label}: {identity}")
        output[identity] = row
    return output


def _verify_input_artifacts(
    vector_root: Path, protocol_root: Path, schedule_root: Path,
    smoke_root: Path, contract: Mapping[str, Any],
) -> dict[str, Any]:
    from evaluation.verify_prompt_neutral_pilot_inputs import verify as verify_vectors
    from evaluation.verify_task_segmented_protocol_artifact import verify as verify_protocol
    from evaluation.verify_task_segmented_smoke_artifact import verify as verify_smoke
    from evaluation.verify_task_segmented_training_schedule_artifact import (
        verify as verify_schedule,
    )

    inputs = contract["immutable_inputs"]
    vectors_expected = inputs["prompt_neutral_vectors"]
    protocol_expected = inputs["task_segmented_protocol"]
    schedule_expected = inputs["training_schedule"]
    smoke_expected = inputs["preserved_smoke"]
    vectors = verify_vectors(vector_root, vectors_expected["preserved_source_id"])
    protocol = verify_protocol(protocol_root, protocol_expected["preserved_source_id"])
    schedule = verify_schedule(schedule_root, schedule_expected["preserved_source_id"])
    smoke = verify_smoke(
        smoke_root, smoke_expected["manifest_sha256"],
        preserved_source_id=smoke_expected["preserved_source_id"],
    )
    vector_binding = {
        "preserved_source_id": vectors["preserved_source_id"],
        "combined_manifest_sha256": vectors["combined_manifest_sha256"],
        "eeg_vector_index_sha256": vectors["eeg"]["vector_index_sha256"],
        "text_vector_index_sha256": vectors["text"]["text_vector_index_sha256"],
        "trial_text_targets_sha256": vectors["text"]["trial_text_targets_sha256"],
    }
    for key, value in vector_binding.items():
        if vectors_expected.get(key) != value:
            raise ValueError(f"prompt-neutral vector binding drifted: {key}")
    protocol_binding = {
        "preserved_source_id": protocol["preserved_source_id"],
        "contract_sha256": protocol["contract_sha256"],
        "report_sha256": protocol["protocol_report_sha256"],
    }
    for key, value in protocol_binding.items():
        if protocol_expected.get(key) != value:
            raise ValueError(f"protocol binding drifted: {key}")
    schedule_binding = {
        "preserved_source_id": schedule["preserved_source_id"],
        "contract_sha256": schedule["schedule_contract_sha256"],
        "manifest_sha256": schedule["schedule_manifest_sha256"],
        "report_sha256": schedule["schedule_report_sha256"],
        "shape": schedule["shape"],
    }
    for key, value in schedule_binding.items():
        if schedule_expected.get(key) != value:
            raise ValueError(f"schedule binding drifted: {key}")
    if smoke["verified_artifact_sha256"] != smoke_expected["artifact_sha256"]:
        raise ValueError("preserved smoke artifact hashes drifted")
    if any(report.get("held_out_test_accessed") is not False for report in (
        protocol, schedule, smoke
    )) or vectors["checks"]["held_out_test_accessed"] is not False:
        raise ValueError("an immutable input reported held-out-test access")
    return {"vectors": vectors, "protocol": protocol, "schedule": schedule, "smoke": smoke}


@dataclass
class ShardData:
    shard: dict[str, Any]
    unit: dict[str, str]
    catalog: list[dict[str, str]]
    assignments: dict[str, dict[str, str]]
    schedule_indices: list[int]
    eeg_by_trial: dict[str, Any]
    text_by_target: dict[str, Any]
    pools: dict[tuple[str, str], list[dict[str, str]]]
    donors: dict[str, dict[str, str]]

    @property
    def epochs(self) -> int:
        return 40

    @property
    def batches_per_epoch(self) -> int:
        return 105

    def batch_indices(self, epoch: int, batch_index: int) -> list[int]:
        start = ((epoch - 1) * self.batches_per_epoch + batch_index) * 64
        return self.schedule_indices[start:start + 64]


def _safe_vector_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe vector path: {relative!r}")
    resolved_root = root.resolve(strict=True)
    candidate = root / relative
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"vector path escapes input root: {relative!r}") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"vector path is not a regular file: {relative!r}")
    return resolved


def _load_vector_map(
    root: Path, specs: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    import numpy as np

    grouped: dict[str, list[tuple[str, int]]] = {}
    for identity, spec in specs.items():
        grouped.setdefault(spec["vector_file"], []).append(
            (identity, _as_int(spec["vector_offset"], "vector offset"))
        )
    output: dict[str, Any] = {}
    for relative, members in sorted(grouped.items()):
        path = _safe_vector_path(root, relative)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"vectors"}:
                raise ValueError(f"unexpected NPZ members: {relative}")
            vectors = archive["vectors"]
            for identity, offset in members:
                if not 0 <= offset < len(vectors):
                    raise ValueError(f"vector offset out of range: {relative}:{offset}")
                vector = np.asarray(vectors[offset], dtype=np.float32)
                if vector.shape != (1024,) or not np.isfinite(vector).all():
                    raise ValueError(f"invalid vector: {relative}:{offset}")
                output[identity] = vector.copy()
    if set(output) != set(specs):
        raise AssertionError("vector loader lost an identity")
    return output


def _decode_schedule_unit(
    schedule_root: Path, unit: Mapping[str, str]
) -> list[int]:
    count = _as_int(unit["uint32_count"], "schedule uint32 count")
    if count != 40 * 105 * 64:
        raise ValueError("full schedule unit size drifted")
    offset = _as_int(unit["byte_offset"], "schedule byte offset")
    path = schedule_root / "schedule_indices.u32le"
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(count * 4)
    if len(payload) != count * 4:
        raise ValueError("short full schedule-unit read")
    if hashlib.sha256(payload).hexdigest() != unit["unit_sha256"]:
        raise ValueError("full schedule-unit SHA256 drifted")
    return [value[0] for value in struct.iter_unpack("<I", payload)]


def _validate_correct_train_eeg_binding(
    catalog_row: Mapping[str, str], eeg_row: Mapping[str, str], trial_id: str,
) -> None:
    expected = {
        "condition": "correct_train",
        "phase": "train",
        "target_trial_id": trial_id,
        "signal_trial_id": trial_id,
        "dataset_version": catalog_row["dataset_version"],
        "reading_task": catalog_row["reading_task"],
        "subject_id": catalog_row["subject_id"],
        "vector_file": catalog_row["eeg_vector_file"],
        "vector_offset": catalog_row["eeg_vector_offset"],
        "vector_dim": catalog_row["eeg_vector_dim"],
        "prompt_mode": "all_masked",
    }
    if any(eeg_row.get(field) != value for field, value in expected.items()):
        raise ValueError(f"correct_train EEG binding mismatch: {trial_id}")
    if eeg_row["vector_dim"] != "1024":
        raise ValueError(f"correct_train EEG dimension mismatch: {trial_id}")


def load_shard_data(
    vector_root: Path, protocol_root: Path, schedule_root: Path,
    shard: Mapping[str, Any],
) -> ShardData:
    """Strictly join all 9,011 canonical-training rows and one schedule unit."""

    catalog = _csv(schedule_root / "trial_catalog.csv", CATALOG_FIELDS)
    if len(catalog) != 9011:
        raise ValueError("full catalog must contain exactly 9011 trials")
    for index, row in enumerate(catalog):
        if _as_int(row["trial_index"], "catalog trial index") != index:
            raise ValueError("catalog indices are not contiguous and ordered")

    assignment_rows = _csv(
        protocol_root / "outer_split_assignments.csv", ASSIGNMENT_FIELDS
    )
    fold_rows = [row for row in assignment_rows
                 if _as_int(row["outer_fold"], "outer fold") == shard["outer_fold"]]
    assignments = _unique(fold_rows, "trial_id", "outer-fold assignment")
    if len(assignments) != 9011:
        raise ValueError("outer-fold assignment does not cover 9011 trials")

    eeg_rows = _csv(vector_root / "eeg" / "vector_index.csv", EEG_INDEX_FIELDS)
    correct_train = [row for row in eeg_rows if row["condition"] == "correct_train"]
    eeg_specs = _unique(correct_train, "target_trial_id", "correct_train EEG trial")
    mapping_rows = _csv(
        vector_root / "text" / "trial_text_targets.csv", TEXT_MAPPING_FIELDS
    )
    mappings = _unique(mapping_rows, "trial_id", "trial-text mapping")
    text_rows = _csv(
        vector_root / "text" / "text_vector_index.csv", TEXT_INDEX_FIELDS
    )
    text_specs = _unique(text_rows, "text_target_id", "text target")

    catalog_ids = {row["trial_id"] for row in catalog}
    if not catalog_ids <= set(eeg_specs) or not catalog_ids <= set(mappings):
        raise ValueError("catalog trials do not resolve to frozen vectors/mappings")
    for row in catalog:
        trial_id = row["trial_id"]
        assignment = assignments[trial_id]
        eeg = eeg_specs[trial_id]
        mapping = mappings[trial_id]
        if any(row[key] != assignment[key] for key in (
            "text_fold", "dataset_version", "reading_task", "subject_id",
            "normalized_text_sha256", "text_target_id", "pseudo_group",
            "length_words_whitespace_v1", "eeg_vector_file", "eeg_vector_offset",
            "eeg_vector_dim",
        )):
            raise ValueError(f"protocol/schedule metadata mismatch: {trial_id}")
        _validate_correct_train_eeg_binding(row, eeg, trial_id)
        if not (
            mapping["split"] == "train"
            and mapping["cohort"] == "primary_zuco2_nr_tsr"
            and mapping["dataset_version"] == row["dataset_version"] == "ZuCo2"
            and mapping["reading_task"] == row["reading_task"]
            and mapping["subject_id"] == row["subject_id"]
            and mapping["text_target_id"] == row["text_target_id"]
            and mapping["normalized_text_sha256"] == row["normalized_text_sha256"]
        ):
            raise ValueError(f"canonical-training vector binding mismatch: {trial_id}")
        text = text_specs.get(row["text_target_id"])
        if text is None or text["normalized_text_sha256"] != row["normalized_text_sha256"]:
            raise ValueError(f"text vector binding mismatch: {trial_id}")

    units = _csv(schedule_root / "schedule_units.csv", UNIT_FIELDS)
    matches = [row for row in units
               if _as_int(row["unit_index"], "unit index") == shard["schedule_unit_index"]]
    if len(matches) != 1:
        raise ValueError("full shard schedule unit is not unique")
    unit = matches[0]
    for field, expected in (
        ("outer_fold", shard["outer_fold"]),
        ("training_seed", shard["training_seed"]),
        ("epochs", 40), ("batches_per_epoch", 105), ("batch_size", 64),
    ):
        if _as_int(unit[field], field) != expected:
            raise ValueError(f"schedule unit field drifted: {field}")
    if _as_int(unit["initialization_seed"], "initialization seed") != initialization_seed(
        shard["outer_fold"], shard["training_seed"]
    ):
        raise ValueError("schedule initialization seed drifted")
    schedule_indices = _decode_schedule_unit(schedule_root, unit)
    for start in range(0, len(schedule_indices), 64):
        indices = schedule_indices[start:start + 64]
        if len(indices) != 64 or len(set(indices)) != 64:
            raise ValueError("full schedule batch repeats a catalog index")
        if any(not 0 <= index < len(catalog) for index in indices):
            raise ValueError("full schedule batch has out-of-range index")
        rows = [catalog[index] for index in indices]
        if any(assignments[row["trial_id"]]["role"] != "fit" for row in rows):
            raise ValueError("full schedule addressed checkpoint/confirmation trial")
        if len({row["text_target_id"] for row in rows}) != 64:
            raise ValueError("full schedule batch repeats a text identity")
        cells = Counter((row["reading_task"], row["pseudo_group"]) for row in rows)
        if cells != Counter({("NR", "0"): 16, ("NR", "1"): 16,
                             ("TSR", "0"): 16, ("TSR", "1"): 16}):
            raise ValueError("full schedule batch cell balance drifted")

    pool_rows = _csv(protocol_root / "candidate_pools.csv", CANDIDATE_FIELDS)
    selected_pool_rows = [row for row in pool_rows
                          if _as_int(row["outer_fold"], "pool outer fold") == shard["outer_fold"]]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in selected_pool_rows:
        grouped.setdefault((row["partition"], row["target_trial_id"]), []).append(row)
    expected_targets = {
        role: {trial_id for trial_id, row in assignments.items() if row["role"] == role}
        for role in ("checkpoint", "confirmation")
    }
    if set(grouped) != {
        (role, trial_id) for role, ids in expected_targets.items() for trial_id in ids
    }:
        raise ValueError("frozen checkpoint/confirmation pool coverage drifted")
    for key, members in grouped.items():
        members.sort(key=lambda row: _as_int(row["candidate_rank"], "candidate rank"))
        if [_as_int(row["candidate_rank"], "candidate rank") for row in members] != list(range(24)):
            raise ValueError(f"candidate ranks drifted: {key}")
        if sum(_as_int(row["is_positive"], "positive") for row in members) != 1:
            raise ValueError(f"candidate pool lacks one positive: {key}")
        if len({row["candidate_normalized_text_sha256"] for row in members}) != 24:
            raise ValueError(f"candidate pool repeats text identity: {key}")
        if any(row["candidate_text_target_id"] not in text_specs for row in members):
            raise ValueError(f"candidate pool text vector is absent: {key}")

    donor_rows = _csv(protocol_root / "confirmation_donors.csv", DONOR_FIELDS)
    selected_donors = [row for row in donor_rows
                       if _as_int(row["outer_fold"], "donor outer fold") == shard["outer_fold"]]
    donors = _unique(selected_donors, "target_trial_id", "confirmation donor")
    if set(donors) != expected_targets["confirmation"]:
        raise ValueError("confirmation donor coverage drifted")
    catalog_by_trial = _unique(catalog, "trial_id", "catalog trial")
    for target_id, donor in donors.items():
        target = catalog_by_trial[target_id]
        signal = catalog_by_trial.get(donor["donor_trial_id"])
        if signal is None or not (
            target_id != signal["trial_id"]
            and target["reading_task"] == signal["reading_task"] == donor["reading_task"]
            and target["subject_id"] == signal["subject_id"] == donor["subject_id"]
            and target["normalized_text_sha256"] != signal["normalized_text_sha256"]
            and donor["target_normalized_text_sha256"] == target["normalized_text_sha256"]
            and donor["donor_normalized_text_sha256"] == signal["normalized_text_sha256"]
            and assignments[signal["trial_id"]]["role"] == "confirmation"
        ):
            raise ValueError(f"invalid matched-wrong donor: {target_id}")

    eeg_needed = {trial_id: eeg_specs[trial_id] for trial_id in catalog_ids}
    # Training addresses every fit catalog row, while evaluation addresses the
    # frozen pool candidates.  Pool membership alone does not guarantee that
    # every fit target is present, so bind and load the union explicitly.
    text_needed_ids = (
        {row["text_target_id"] for row in catalog}
        | {row["candidate_text_target_id"] for row in selected_pool_rows}
    )
    text_needed = {identity: text_specs[identity] for identity in text_needed_ids}
    eeg_vectors = _load_vector_map(vector_root / "eeg", eeg_needed)
    text_vectors = _load_vector_map(vector_root / "text", text_needed)
    return ShardData(
        shard=dict(shard), unit=unit, catalog=catalog, assignments=assignments,
        schedule_indices=schedule_indices, eeg_by_trial=eeg_vectors,
        text_by_target=text_vectors, pools=grouped, donors=donors,
    )


def _macro_mrr(rank_rows: Sequence[Mapping[str, object]]) -> tuple[float, float, float]:
    by_task: dict[str, list[float]] = {"NR": [], "TSR": []}
    for row in rank_rows:
        task = str(row["reading_task"])
        if task not in by_task:
            raise ValueError(f"unexpected checkpoint task: {task}")
        by_task[task].append(1.0 / _as_int(row["positive_rank"], "positive rank"))
    if not by_task["NR"] or not by_task["TSR"]:
        raise ValueError("checkpoint metric requires NR and TSR")
    nr = sum(by_task["NR"]) / len(by_task["NR"])
    tsr = sum(by_task["TSR"]) / len(by_task["TSR"])
    return 0.5 * (nr + tsr), nr, tsr


def confirmation_predictions_from_scores(
    score_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Reconstruct decision-summary ranks from sealed candidate-score rows."""

    grouped: dict[tuple[str, int, int, str, str], list[Mapping[str, object]]] = {}
    for row in score_rows:
        key = (
            str(row["arm_id"]), _as_int(row["training_seed"], "training seed"),
            _as_int(row["outer_fold"], "outer fold"), str(row["trial_id"]),
            str(row["signal_condition"]),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        members = grouped[key]
        if len(members) != 24:
            raise ValueError(f"confirmation candidate-score pool is not 24-way: {key}")
        ranks = [_as_int(row["candidate_rank"], "candidate rank") for row in members]
        if sorted(ranks) != list(range(24)):
            raise ValueError(f"confirmation candidate ranks drifted: {key}")
        members = sorted(members, key=lambda row: _as_int(
            row["candidate_rank"], "candidate rank"
        ))
        positives = [index for index, row in enumerate(members)
                     if _as_int(row["is_positive"], "positive") == 1]
        if len(positives) != 1:
            raise ValueError(f"confirmation score pool lacks one positive: {key}")
        condition = key[-1]
        if condition not in {"correct", "matched_wrong"}:
            raise ValueError(f"unexpected confirmation signal condition: {condition}")
        identity_fields = (
            "reading_task", "subject_id", "normalized_text_sha256",
            "signal_trial_id", "signal_subject_id",
            "signal_normalized_text_sha256",
        )
        for field in identity_fields:
            if len({str(row[field]) for row in members}) != 1:
                raise ValueError(f"confirmation query provenance changed within pool: {field}")
        first = members[0]
        target_trial = str(first["trial_id"])
        signal_trial = str(first["signal_trial_id"])
        target_subject = str(first["subject_id"])
        signal_subject = str(first["signal_subject_id"])
        target_text = str(first["normalized_text_sha256"])
        signal_text = str(first["signal_normalized_text_sha256"])
        if condition == "correct":
            if not (
                signal_trial == target_trial
                and signal_subject == target_subject
                and signal_text == target_text
            ):
                raise ValueError("correct confirmation query changed EEG identity")
        elif not (
            signal_trial != target_trial
            and signal_subject == target_subject
            and signal_text != target_text
        ):
            raise ValueError("matched-wrong confirmation provenance is invalid")
        positive_rank = positive_rank_from_scores(
            [_as_float(row["score"], "candidate score") for row in members],
            list(range(24)), positives[0],
        )
        output.append({
            "arm_id": key[0], "training_seed": key[1], "outer_fold": key[2],
            "trial_id": target_trial, "reading_task": first["reading_task"],
            "subject_id": target_subject,
            "normalized_text_sha256": target_text,
            "signal_condition": condition, "positive_rank": positive_rank,
            "candidate_pool_size": 24,
            "scientific_decision_permitted": "false",
        })
    return output


def _score_partition(
    model: Any, data: ShardData, arm_id: str, partition: str,
    signal_condition: str, device: Any, epoch: int | None,
    evaluation_batch_size: int = 256,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import numpy as np
    import torch
    from torch.nn import functional as F

    target_ids = sorted(
        trial_id for trial_id, row in data.assignments.items()
        if row["role"] == partition
    )
    if signal_condition not in {"correct", "matched_wrong"}:
        raise ValueError("unsupported full-shard signal condition")
    if partition != "confirmation" and signal_condition != "correct":
        raise ValueError("checkpoint scoring admits only correct EEG")
    catalog_by_trial = {row["trial_id"]: row for row in data.catalog}
    ranks: list[dict[str, object]] = []
    scores_output: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(target_ids), evaluation_batch_size):
            ids = target_ids[start:start + evaluation_batch_size]
            signal_ids = [
                trial_id if signal_condition == "correct"
                else data.donors[trial_id]["donor_trial_id"]
                for trial_id in ids
            ]
            eeg = torch.from_numpy(np.stack([
                data.eeg_by_trial[signal_id] for signal_id in signal_ids
            ])).to(device=device, dtype=torch.float32)
            pool_rows = [data.pools[(partition, trial_id)] for trial_id in ids]
            text = torch.from_numpy(np.stack([
                np.stack([data.text_by_target[row["candidate_text_target_id"]]
                          for row in members])
                for members in pool_rows
            ])).to(device=device, dtype=torch.float32)
            adapted = F.normalize(model(eeg), dim=1)
            text = F.normalize(text, dim=2)
            score_matrix = torch.einsum("bd,bkd->bk", adapted, text).cpu().numpy()
            for offset, trial_id in enumerate(ids):
                members = pool_rows[offset]
                row_scores = [float(value) for value in score_matrix[offset]]
                positive_offsets = [index for index, row in enumerate(members)
                                    if _as_int(row["is_positive"], "positive") == 1]
                if len(positive_offsets) != 1:
                    raise ValueError("candidate pool lacks exactly one positive")
                rank = positive_rank_from_scores(
                    row_scores,
                    [_as_int(row["candidate_rank"], "candidate rank") for row in members],
                    positive_offsets[0],
                )
                target = catalog_by_trial[trial_id]
                signal = catalog_by_trial[signal_ids[offset]]
                common = {
                    "arm_id": arm_id,
                    "training_seed": data.shard["training_seed"],
                    "outer_fold": data.shard["outer_fold"],
                    "trial_id": trial_id,
                    "reading_task": target["reading_task"],
                    "subject_id": target["subject_id"],
                    "normalized_text_sha256": target["normalized_text_sha256"],
                }
                if partition == "checkpoint":
                    ranks.append({
                        "arm_id": arm_id,
                        "outer_fold": data.shard["outer_fold"],
                        "training_seed": data.shard["training_seed"],
                        "epoch": epoch,
                        "trial_id": trial_id,
                        "reading_task": target["reading_task"],
                        "positive_rank": rank,
                        "candidate_pool_size": 24,
                    })
                else:
                    ranks.append({
                        **common,
                        "signal_condition": signal_condition,
                        "positive_rank": rank,
                        "candidate_pool_size": 24,
                        "scientific_decision_permitted": "false",
                    })
                    for candidate, score in zip(members, row_scores):
                        scores_output.append({
                            **common,
                            "signal_condition": signal_condition,
                            "signal_trial_id": signal["trial_id"],
                            "signal_subject_id": signal["subject_id"],
                            "signal_normalized_text_sha256": signal[
                                "normalized_text_sha256"
                            ],
                            "candidate_rank": candidate["candidate_rank"],
                            "candidate_normalized_text_sha256": candidate[
                                "candidate_normalized_text_sha256"
                            ],
                            "candidate_text_target_id": candidate[
                                "candidate_text_target_id"
                            ],
                            "is_positive": candidate["is_positive"],
                            "is_designated_donor_text": candidate[
                                "is_designated_donor_text"
                            ],
                            "score": format(score, ".17g"),
                        })
    return ranks, scores_output


def _set_determinism(seed: int) -> None:
    _configure_cuda_determinism_environment()

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _configure_cuda_determinism_environment() -> None:
    """Configure deterministic cuBLAS before PyTorch creates a CUDA context."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _rng_state() -> dict[str, object]:
    import numpy as np
    import torch

    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": name, "keys": keys.tolist(), "position": int(position),
            "has_gauss": int(has_gauss), "cached_gaussian": float(cached),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_device_0": (
            torch.cuda.get_rng_state(0) if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    import numpy as np
    import torch

    random.setstate(tuple(state["python"]))
    numpy = state["numpy"]
    if not isinstance(numpy, dict):
        raise ValueError("invalid NumPy resume RNG state")
    np.random.set_state((
        str(numpy["name"]), np.asarray(numpy["keys"], dtype=np.uint32),
        int(numpy["position"]), int(numpy["has_gauss"]),
        float(numpy["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda_device_0")
    if torch.cuda.is_available():
        if cuda_state is None:
            raise ValueError("resume lacks selected cuda:0 RNG state")
        torch.cuda.set_rng_state(cuda_state, device=0)
    elif cuda_state is not None:
        raise ValueError("cannot restore selected cuda:0 RNG state without CUDA")


def _canonical_tensor_sha256(tensor: Any) -> str:
    import numpy as np

    array = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\x00" + np.ascontiguousarray(array).tobytes()).hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    state = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        state.update(name.encode("utf-8") + b"\x00")
        state.update(_canonical_tensor_sha256(tensor).encode("ascii"))
    return state.hexdigest()


def _model_sha256(model: Any) -> str:
    return _state_dict_sha256(model.state_dict())


def _optimizer_sha256(optimizer: Any) -> str:
    def canonical(value: Any) -> Any:
        import torch

        if isinstance(value, torch.Tensor):
            return {"tensor_sha256": _canonical_tensor_sha256(value)}
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"unsupported optimizer-state value: {type(value).__name__}")

    payload = json.dumps(
        canonical(optimizer.state_dict()), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optimizer(model: Any, execution: Mapping[str, Any]) -> Any:
    import torch

    return torch.optim.AdamW(
        model.parameters(), lr=execution["learning_rate"],
        betas=tuple(execution["betas"]), eps=execution["eps"],
        weight_decay=execution["weight_decay"], amsgrad=execution["amsgrad"],
        maximize=execution["maximize"], foreach=execution["foreach"],
        capturable=execution["capturable"],
        differentiable=execution["differentiable"], fused=execution["fused"],
    )


def _schedule_prefix_sha256(data: ShardData, completed_epoch: int) -> str:
    count = completed_epoch * data.batches_per_epoch * 64
    return hashlib.sha256(b"".join(
        struct.pack("<I", index) for index in data.schedule_indices[:count]
    )).hexdigest()


def _save_resume(
    path: Path, *, model: Any, optimizer: Any, best_state: Mapping[str, Any],
    best_epoch: int, best_metric: float, completed_epoch: int,
    initial_state_sha256: str, binding_sha256: str, data: ShardData,
    arm_id: str, trace_path: Path, history_path: Path, ranks_path: Path,
    launch_authorization_sha256: str,
    runtime_fingerprint: Mapping[str, Any],
) -> None:
    hashes = {
        "step_trace_sha256": sha256(trace_path),
        "training_history_sha256": sha256(history_path),
        "checkpoint_positive_ranks_sha256": sha256(ranks_path),
    }
    _atomic_torch_save(path, {
        "schema_version": 1, "run_mode": FULL_MODE,
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "launch_authorization_sha256": launch_authorization_sha256,
        "shard_id": data.shard["shard_id"],
        "schedule_unit_index": data.shard["schedule_unit_index"],
        "outer_fold": data.shard["outer_fold"],
        "training_seed": data.shard["training_seed"], "arm_id": arm_id,
        "binding_sha256": binding_sha256,
        "runtime_fingerprint": dict(runtime_fingerprint),
        "runtime_fingerprint_sha256": _runtime_fingerprint_sha256(
            runtime_fingerprint
        ),
        "schedule_unit_sha256": data.unit["unit_sha256"],
        "initial_model_state_sha256": initial_state_sha256,
        "current_model_state_sha256": _model_sha256(model),
        "optimizer_state_sha256": _optimizer_sha256(optimizer),
        "completed_epoch": completed_epoch,
        "completed_batch_index": data.batches_per_epoch - 1 if completed_epoch else -1,
        "completed_steps": completed_epoch * data.batches_per_epoch,
        "schedule_prefix_sha256": _schedule_prefix_sha256(data, completed_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_state_dict": dict(best_state), "best_epoch": best_epoch,
        "best_checkpoint_macro_mrr": best_metric,
        "rng_state": _rng_state(), **hashes,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
    })


def _save_best_checkpoint(
    path: Path, *, best_state: Mapping[str, Any], best_epoch: int,
    best_metric: float, binding_sha256: str, arm_id: str,
) -> None:
    """Atomically materialize the best state named by an authoritative journal."""

    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in best_state.items()
    }
    _atomic_torch_save(path, {
        "schema_version": 1, "run_mode": FULL_MODE,
        "binding_sha256": binding_sha256, "arm_id": arm_id,
        "best_epoch": best_epoch,
        "best_checkpoint_macro_mrr": best_metric,
        "model_state_sha256": _state_dict_sha256(state),
        "model_state_dict": state,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
    })


def _truncate_bound_csv_suffix(
    path: Path, fields: Sequence[str], expected_rows: int,
    checkpoint_sha256: object,
) -> list[dict[str, str]]:
    """Validate a checkpointed CSV prefix and discard a crash-window suffix."""

    if not path.is_file():
        raise ValueError(f"resume CSV is absent: {path.name}")
    rows = _csv(path, fields)
    if len(rows) < expected_rows:
        raise ValueError(f"resume CSV is shorter than checkpoint: {path.name}")
    prefix = rows[:expected_rows]
    if hashlib.sha256(_csv_bytes(fields, prefix)).hexdigest() != checkpoint_sha256:
        raise ValueError(f"resume CSV prefix binding drifted: {path.name}")
    if len(rows) != expected_rows:
        _atomic_csv(path, fields, prefix)
    return prefix


def _load_resume(
    path: Path, *, model: Any, optimizer: Any, initial_state_sha256: str,
    binding_sha256: str, data: ShardData, arm_id: str, trace_path: Path,
    history_path: Path, ranks_path: Path, best_path: Path, device: Any,
    launch_authorization_sha256: str,
    runtime_fingerprint: Mapping[str, Any],
) -> tuple[int, dict[str, Any], int, float, bool]:
    import torch

    if not path.exists():
        return 0, {name: tensor.detach().cpu().clone()
                   for name, tensor in model.state_dict().items()}, 0, float("-inf"), False
    # Always deserialize on CPU.  In particular, torch CPU/CUDA RNG states are
    # serialized as CPU ByteTensors; mapping them to CUDA makes
    # ``set_rng_state``/``set_rng_state_all`` reject them.  Model and optimizer
    # ``load_state_dict`` calls below migrate their tensors to parameter devices.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version": 1, "run_mode": FULL_MODE,
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "launch_authorization_sha256": launch_authorization_sha256,
        "shard_id": data.shard["shard_id"],
        "schedule_unit_index": data.shard["schedule_unit_index"],
        "outer_fold": data.shard["outer_fold"],
        "training_seed": data.shard["training_seed"], "arm_id": arm_id,
        "binding_sha256": binding_sha256,
        "runtime_fingerprint": dict(runtime_fingerprint),
        "runtime_fingerprint_sha256": _runtime_fingerprint_sha256(
            runtime_fingerprint
        ),
        "schedule_unit_sha256": data.unit["unit_sha256"],
        "initial_model_state_sha256": initial_state_sha256,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume binding/tamper failure: {arm_id}.{key}")
    completed_epoch = int(checkpoint["completed_epoch"])
    if not 0 <= completed_epoch <= 40:
        raise ValueError("resume epoch is outside 0..40")
    if checkpoint.get("completed_batch_index") != (
        data.batches_per_epoch - 1 if completed_epoch else -1
    ):
        raise ValueError("resume batch boundary drifted")
    if checkpoint.get("completed_steps") != completed_epoch * data.batches_per_epoch:
        raise ValueError("resume optimizer-step count drifted")
    if checkpoint.get("schedule_prefix_sha256") != _schedule_prefix_sha256(
        data, completed_epoch
    ):
        raise ValueError("resume schedule-prefix hash drifted")
    expected_steps = completed_epoch * data.batches_per_epoch
    checkpoint_count = sum(
        row["role"] == "checkpoint" for row in data.assignments.values()
    )
    # Any of the three CSVs may have been atomically replaced immediately
    # before a crash but before the new resume checkpoint was installed.  The
    # old checkpoint is authoritative: validate its exact prefix, then discard
    # the uncommitted suffix and replay the partial epoch deterministically.
    _truncate_bound_csv_suffix(
        trace_path, STEP_TRACE_FIELDS, expected_steps,
        checkpoint.get("step_trace_sha256"),
    )
    _truncate_bound_csv_suffix(
        history_path, HISTORY_FIELDS, completed_epoch + 1,
        checkpoint.get("training_history_sha256"),
    )
    _truncate_bound_csv_suffix(
        ranks_path, CHECKPOINT_RANK_FIELDS,
        (completed_epoch + 1) * checkpoint_count,
        checkpoint.get("checkpoint_positive_ranks_sha256"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if _model_sha256(model) != checkpoint.get("current_model_state_sha256"):
        raise ValueError("resume model-state hash drifted")
    if _optimizer_sha256(optimizer) != checkpoint.get("optimizer_state_sha256"):
        raise ValueError("resume optimizer-state hash drifted")
    _restore_rng_state(checkpoint["rng_state"])
    # The resume checkpoint is the sole committed epoch-boundary journal.  A
    # crash can occur after replacing ``best_checkpoint.pt`` for a newly
    # improved epoch but before installing that epoch's resume journal.  Rebuild
    # the materialized best file from the old authoritative resume state so an
    # uncommitted future best can never survive replay.
    _save_best_checkpoint(
        best_path, best_state=checkpoint["best_state_dict"],
        best_epoch=int(checkpoint["best_epoch"]),
        best_metric=float(checkpoint["best_checkpoint_macro_mrr"]),
        binding_sha256=binding_sha256, arm_id=arm_id,
    )
    return (
        completed_epoch, dict(checkpoint["best_state_dict"]),
        int(checkpoint["best_epoch"]),
        float(checkpoint["best_checkpoint_macro_mrr"]), True,
    )


def _reported_resume_status(
    *, resume_preexisted: bool, journal_loaded: bool,
) -> bool:
    if not journal_loaded:
        raise RuntimeError("epoch-zero resume journal was not loaded")
    return resume_preexisted


def _run_arm(
    arm_root: Path, arm_id: str, data: ShardData,
    contract: Mapping[str, Any], binding_sha256: str, device: Any,
    launch_authorization_sha256: str,
    runtime_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from project_adapters.task_segmented_objective import (
        SharedResidualAdapter, build_partition_masks,
        masked_symmetric_alignment_loss,
    )

    summary_path = arm_root / "run_summary.json"
    if summary_path.exists():
        summary = _json(summary_path)
        required_summary = {
            "schema_version": 1, "status": "complete", "run_mode": FULL_MODE,
            "shard_id": data.shard["shard_id"],
            "schedule_unit_index": data.shard["schedule_unit_index"],
            "outer_fold": data.shard["outer_fold"],
            "training_seed": data.shard["training_seed"], "arm_id": arm_id,
            "binding_sha256": binding_sha256,
            "schedule_unit_sha256": data.unit["unit_sha256"],
            "epochs": 40, "batches_per_epoch": 105, "optimizer_steps": 4200,
            "trainable_parameters": 196608,
            "full_training_authorized": True,
            "scientific_decision_permitted": False,
            "official_validation_used_for_confirmation": False,
            "held_out_test_accessed": False,
        }
        if any(summary.get(key) != value for key, value in required_summary.items()):
            raise ValueError(f"completed arm binding drifted: {arm_id}")
        expected_files = set(contract["output"]["exact_per_arm_files"])
        if {entry.name for entry in arm_root.iterdir()} != expected_files:
            raise ValueError(f"completed arm inventory drifted: {arm_id}")
        expected_artifacts = {
            "resume_checkpoint.pt", "best_checkpoint.pt", "training_history.csv",
            "step_trace.csv", "checkpoint_positive_ranks.csv",
            "confirmation_candidate_scores.csv",
        }
        declared = summary.get("artifact_sha256")
        if not isinstance(declared, dict) or set(declared) != expected_artifacts:
            raise ValueError(f"completed arm artifact declaration drifted: {arm_id}")
        for name, expected in declared.items():
            if not _is_sha256(expected) or sha256(arm_root / name) != expected:
                raise ValueError(f"completed arm artifact drifted: {arm_id}/{name}")
        checkpoint_count = sum(
            row["role"] == "checkpoint" for row in data.assignments.values()
        )
        confirmation_count = sum(
            row["role"] == "confirmation" for row in data.assignments.values()
        )
        exact_counts = {
            "checkpoint_trial_count": checkpoint_count,
            "checkpoint_rank_rows": 41 * checkpoint_count,
            "confirmation_trial_count": confirmation_count,
            "confirmation_prediction_rows": 2 * confirmation_count,
            "confirmation_candidate_score_rows": 2 * confirmation_count * 24,
        }
        if any(summary.get(key) != value for key, value in exact_counts.items()):
            raise ValueError(f"completed arm cardinality drifted: {arm_id}")
        score_rows = _csv(
            arm_root / "confirmation_candidate_scores.csv",
            CONFIRMATION_SCORE_FIELDS,
        )
        predictions = confirmation_predictions_from_scores(score_rows)
        if len(predictions) != 2 * confirmation_count:
            raise ValueError(f"completed arm prediction reconstruction drifted: {arm_id}")
        return {
            **summary, "resumed": True,
            "confirmation_predictions": predictions,
        }

    arm_root.mkdir(parents=True, exist_ok=True)
    trace_path = arm_root / "step_trace.csv"
    history_path = arm_root / "training_history.csv"
    ranks_path = arm_root / "checkpoint_positive_ranks.csv"
    resume_path = arm_root / "resume_checkpoint.pt"
    best_path = arm_root / "best_checkpoint.pt"
    scores_path = arm_root / "confirmation_candidate_scores.csv"
    init_seed = initialization_seed(
        data.shard["outer_fold"], data.shard["training_seed"]
    )
    _set_determinism(init_seed)
    model = SharedResidualAdapter().to(device=device, dtype=torch.float32)
    if model.trainable_parameter_count != 196608:
        raise ValueError("full-shard adapter parameter count drifted")
    initial_hash = _model_sha256(model)
    optimizer = _optimizer(model, contract["execution"])

    resume_preexisted = resume_path.exists()
    if not resume_path.exists():
        # Epoch-zero initialization is one deterministic transaction rooted in
        # the resume checkpoint.  If a crash left any earlier files behind,
        # rebuild all of them rather than inferring state from an uncommitted
        # history/rank/best artifact.
        _atomic_csv(trace_path, STEP_TRACE_FIELDS, [])
        checkpoint_rows, _ = _score_partition(
            model, data, arm_id, "checkpoint", "correct", device, 0
        )
        macro, nr, tsr = _macro_mrr(checkpoint_rows)
        history = [{
            "arm_id": arm_id, "outer_fold": data.shard["outer_fold"],
            "training_seed": data.shard["training_seed"], "epoch": 0,
            "optimizer_steps": 0, "mean_train_loss": "",
            "checkpoint_macro_mrr": format(macro, ".17g"),
            "checkpoint_nr_mrr": format(nr, ".17g"),
            "checkpoint_tsr_mrr": format(tsr, ".17g"), "is_best": 1,
        }]
        _atomic_csv(history_path, HISTORY_FIELDS, history)
        _atomic_csv(ranks_path, CHECKPOINT_RANK_FIELDS, checkpoint_rows)
        best_state = {name: tensor.detach().cpu().clone()
                      for name, tensor in model.state_dict().items()}
        _save_best_checkpoint(
            best_path, best_state=best_state, best_epoch=0,
            best_metric=macro, binding_sha256=binding_sha256, arm_id=arm_id,
        )
        _save_resume(
            resume_path, model=model, optimizer=optimizer, best_state=best_state,
            best_epoch=0, best_metric=macro, completed_epoch=0,
            initial_state_sha256=initial_hash, binding_sha256=binding_sha256,
            data=data, arm_id=arm_id, trace_path=trace_path,
            history_path=history_path, ranks_path=ranks_path,
            launch_authorization_sha256=launch_authorization_sha256,
            runtime_fingerprint=runtime_fingerprint,
        )
    elif not all(path.is_file() for path in (trace_path, history_path, ranks_path, best_path)):
        raise ValueError(f"resume artifact inventory is incomplete: {arm_id}")

    completed_epoch, best_state, best_epoch, best_metric, journal_loaded = _load_resume(
        resume_path, model=model, optimizer=optimizer,
        initial_state_sha256=initial_hash, binding_sha256=binding_sha256,
        data=data, arm_id=arm_id, trace_path=trace_path,
        history_path=history_path, ranks_path=ranks_path,
        best_path=best_path, device=device,
        launch_authorization_sha256=launch_authorization_sha256,
        runtime_fingerprint=runtime_fingerprint,
    )
    # Loading the epoch-zero journal is part of a fresh transaction, not a
    # resumed run.  Report resume only when that journal existed on entry.
    resumed = _reported_resume_status(
        resume_preexisted=resume_preexisted, journal_loaded=journal_loaded,
    )
    schedule_contract_sha = contract["immutable_inputs"]["training_schedule"][
        "contract_sha256"
    ]
    trace_rows = _csv(trace_path, STEP_TRACE_FIELDS)
    history_rows = _csv(history_path, HISTORY_FIELDS)
    checkpoint_rank_rows = _csv(ranks_path, CHECKPOINT_RANK_FIELDS)
    catalog = data.catalog
    for epoch in range(completed_epoch + 1, 41):
        model.train()
        losses: list[float] = []
        for batch_index in range(data.batches_per_epoch):
            indices = data.batch_indices(epoch, batch_index)
            rows = [catalog[index] for index in indices]
            eeg = torch.from_numpy(np.stack([
                data.eeg_by_trial[row["trial_id"]] for row in rows
            ])).to(device=device, dtype=torch.float32)
            text = torch.from_numpy(np.stack([
                data.text_by_target[row["text_target_id"]] for row in rows
            ])).to(device=device, dtype=torch.float32)
            key = global_mask_key(
                schedule_contract_sha, data.unit["unit_sha256"],
                data.shard["outer_fold"], data.shard["training_seed"],
                epoch, batch_index,
            )
            masks = build_partition_masks(
                arm_id, [row["reading_task"] for row in rows],
                [row["pseudo_group"] for row in rows], key=key, device=device,
            )
            pre_hash = _model_sha256(model)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_symmetric_alignment_loss(model(eeg), text, *masks)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite full-shard loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=contract["execution"]["gradient_clip_norm"],
                norm_type=contract["execution"]["gradient_clip_norm_type"],
                error_if_nonfinite=contract["execution"][
                    "gradient_clip_error_if_nonfinite"
                ],
            )
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all())
                   for parameter in model.parameters()):
                raise FloatingPointError("non-finite full-shard model state")
            losses.append(float(loss.item()))
            trial_ids = [row["trial_id"] for row in rows]
            trace_rows.append({
                "outer_fold": data.shard["outer_fold"],
                "training_seed": data.shard["training_seed"], "arm_id": arm_id,
                "epoch": epoch, "batch_index": batch_index,
                "global_step": (epoch - 1) * data.batches_per_epoch + batch_index + 1,
                "schedule_unit_sha256": data.unit["unit_sha256"],
                "batch_catalog_indices_sha256": hashlib.sha256(b"".join(
                    struct.pack("<I", index) for index in indices
                )).hexdigest(),
                "batch_trial_ids_sha256": hashlib.sha256(
                    ("\n".join(trial_ids) + "\n").encode("utf-8")
                ).hexdigest(),
                "global_mask_key": key, "initial_state_sha256": initial_hash,
                "pre_step_state_sha256": pre_hash,
                "loss": format(float(loss.item()), ".17g"),
                "gradient_norm_preclip": format(float(gradient_norm.item()), ".17g"),
                "post_step_state_sha256": _model_sha256(model),
                "optimizer_state_sha256": _optimizer_sha256(optimizer),
            })
        epoch_ranks, _ = _score_partition(
            model, data, arm_id, "checkpoint", "correct", device, epoch
        )
        checkpoint_rank_rows.extend(epoch_ranks)
        macro, nr, tsr = _macro_mrr(epoch_ranks)
        selected_epoch, selected_metric, replaced = strict_best_epoch(
            best_epoch, best_metric, epoch, macro
        )
        if replaced:
            best_epoch, best_metric = selected_epoch, selected_metric
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
            _save_best_checkpoint(
                best_path, best_state=best_state, best_epoch=best_epoch,
                best_metric=best_metric, binding_sha256=binding_sha256,
                arm_id=arm_id,
            )
        history_rows.append({
            "arm_id": arm_id, "outer_fold": data.shard["outer_fold"],
            "training_seed": data.shard["training_seed"], "epoch": epoch,
            "optimizer_steps": epoch * data.batches_per_epoch,
            "mean_train_loss": format(sum(losses) / len(losses), ".17g"),
            "checkpoint_macro_mrr": format(macro, ".17g"),
            "checkpoint_nr_mrr": format(nr, ".17g"),
            "checkpoint_tsr_mrr": format(tsr, ".17g"),
            "is_best": int(replaced),
        })
        _atomic_csv(history_path, HISTORY_FIELDS, history_rows)
        _atomic_csv(ranks_path, CHECKPOINT_RANK_FIELDS, checkpoint_rank_rows)
        # Contract resume boundary is a completed epoch.  Buffering step rows
        # in memory avoids O(n^2) whole-file rewrites on all 189,000 steps.
        _atomic_csv(trace_path, STEP_TRACE_FIELDS, trace_rows)
        _save_resume(
            resume_path, model=model, optimizer=optimizer, best_state=best_state,
            best_epoch=best_epoch, best_metric=best_metric,
            completed_epoch=epoch, initial_state_sha256=initial_hash,
            binding_sha256=binding_sha256, data=data, arm_id=arm_id,
            trace_path=trace_path, history_path=history_path, ranks_path=ranks_path,
            launch_authorization_sha256=launch_authorization_sha256,
            runtime_fingerprint=runtime_fingerprint,
        )

    if len(history_rows) != 41 or len(trace_rows) != 4200:
        raise ValueError("full arm did not complete the frozen 40x105 execution")
    final_training_state_sha256 = _model_sha256(model)
    final_optimizer_state_sha256 = _optimizer_sha256(optimizer)
    # ``is_best`` is an append-only audit event: epoch 0 and every epoch that
    # strictly replaced the incumbent are marked.  The final marked epoch must
    # equal ``best_epoch``.  Avoiding an in-place epoch-40 history rewrite also
    # removes a finalization crash window after the sealed epoch-40 resume.
    model.load_state_dict(best_state, strict=True)
    confirmation_predictions: list[dict[str, object]] = []
    confirmation_scores: list[dict[str, object]] = []
    for condition in ("correct", "matched_wrong"):
        ranks, scores = _score_partition(
            model, data, arm_id, "confirmation", condition, device, None
        )
        confirmation_predictions.extend(ranks)
        confirmation_scores.extend(scores)
    _atomic_csv(scores_path, CONFIRMATION_SCORE_FIELDS, confirmation_scores)
    checkpoint_count = sum(
        row["role"] == "checkpoint" for row in data.assignments.values()
    )
    confirmation_count = sum(
        row["role"] == "confirmation" for row in data.assignments.values()
    )
    if len(checkpoint_rank_rows) != 41 * checkpoint_count:
        raise ValueError("checkpoint-rank cardinality drifted")
    if len(confirmation_predictions) != 2 * confirmation_count:
        raise ValueError("confirmation-prediction cardinality drifted")
    if len(confirmation_scores) != 2 * confirmation_count * 24:
        raise ValueError("confirmation-score cardinality drifted")
    artifacts = {
        name: sha256(arm_root / name)
        for name in (
            "resume_checkpoint.pt", "best_checkpoint.pt", "training_history.csv",
            "step_trace.csv", "checkpoint_positive_ranks.csv",
            "confirmation_candidate_scores.csv",
        )
    }
    summary = {
        "schema_version": 1, "status": "complete", "run_mode": FULL_MODE,
        "shard_id": data.shard["shard_id"],
        "schedule_unit_index": data.shard["schedule_unit_index"],
        "outer_fold": data.shard["outer_fold"],
        "training_seed": data.shard["training_seed"], "arm_id": arm_id,
        "binding_sha256": binding_sha256,
        "schedule_unit_sha256": data.unit["unit_sha256"],
        "epochs": 40, "batches_per_epoch": 105, "optimizer_steps": 4200,
        "trainable_parameters": 196608,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_training_state_sha256,
        "final_optimizer_state_sha256": final_optimizer_state_sha256,
        "best_epoch": best_epoch,
        "best_checkpoint_macro_mrr": best_metric,
        "checkpoint_trial_count": checkpoint_count,
        "checkpoint_rank_rows": len(checkpoint_rank_rows),
        "confirmation_trial_count": confirmation_count,
        "confirmation_prediction_rows": len(confirmation_predictions),
        "confirmation_candidate_score_rows": len(confirmation_scores),
        "resumed": resumed, "artifact_sha256": artifacts,
        "full_training_authorized": True,
        "scientific_decision_permitted": False,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
    }
    _atomic_json(summary_path, summary)
    return {**summary, "confirmation_predictions": confirmation_predictions}


def _regular_file_hashes(root: Path, excluded: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"full-shard output contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                output[relative] = sha256(path)
    return output


def _binding_sha256(
    contract: Mapping[str, Any], launch_sha256: str, project_commit: str,
    shard: Mapping[str, Any], unit_sha256: str,
    runtime_fingerprint: Mapping[str, Any], git_worktree_clean: bool,
    git_submodules_clean: bool,
) -> str:
    payload = {
        "schema_version": 1, "run_mode": SHARD_MODE,
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "launch_authorization_sha256": launch_sha256,
        "project_commit": project_commit,
        "runner_source_sha256": sha256(RUNNER_PATH),
        "adapter_source_sha256": sha256(ADAPTER_PATH),
        "task_treatment_pilots_source_sha256": sha256(
            TASK_TREATMENT_PILOTS_PATH
        ),
        "input_bindings": contract["immutable_inputs"],
        "shard": dict(shard), "schedule_unit_sha256": unit_sha256,
        "runtime_fingerprint": dict(runtime_fingerprint),
        "runtime_fingerprint_sha256": _runtime_fingerprint_sha256(
            runtime_fingerprint
        ),
        "git_worktree_clean": git_worktree_clean,
        "git_submodules_clean": git_submodules_clean,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def run(
    vector_root: Path, protocol_root: Path, schedule_root: Path,
    smoke_root: Path, output_root: Path, project_commit: str,
    shard_id: str, launch_authorization: Path,
    launch_authorization_sha256: str, device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Execute one exact authorized fold/seed shard containing all three arms."""

    # This must precede the first PyTorch import or CUDA availability query in
    # this process so deterministic cuBLAS is effective for the CLI itself.
    _configure_cuda_determinism_environment()
    contract = load_full_contract()
    # Authorization precedes all scientific inputs and output creation.
    authorize_launch(
        contract, launch_authorization, launch_authorization_sha256,
        project_commit, shard_id,
    )
    git_boundary = verify_git_execution_boundary(project_commit)
    runtime_fingerprint, observed_cuda_device_count = inspect_exact_runtime(
        contract, device_name
    )
    shard = shard_definition(contract, shard_id)
    _verify_input_artifacts(
        vector_root, protocol_root, schedule_root, smoke_root, contract
    )
    data = load_shard_data(vector_root, protocol_root, schedule_root, shard)

    import torch

    device = torch.device(runtime_fingerprint["device"])
    binding = _binding_sha256(
        contract, launch_authorization_sha256, project_commit,
        shard, data.unit["unit_sha256"], runtime_fingerprint,
        bool(git_boundary["git_worktree_clean"]),
        bool(git_boundary["git_submodules_clean"]),
    )
    if output_root.exists() and (output_root / "full_shard_manifest.json").exists():
        raise ValueError(
            "completed full shard must pass independent verification before reuse"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_arm(
            output_root / "runs" / arm, arm, data, contract, binding, device,
            launch_authorization_sha256, runtime_fingerprint,
        )
        for arm in ARMS
    ]
    if len({result["initial_state_sha256"] for result in results}) != 1:
        raise ValueError("paired initial state differs across full-shard arms")
    predictions: list[dict[str, object]] = []
    for result in results:
        predictions.extend(result.pop("confirmation_predictions"))
    _atomic_csv(
        output_root / "run_manifest.csv", RUN_MANIFEST_FIELDS,
        [{
            "arm_id": arm, "outer_fold": shard["outer_fold"],
            "training_seed": shard["training_seed"], "status": "complete",
            "run_mode": FULL_MODE, "full_training_authorized": "true",
            "scientific_decision_permitted": "false",
            "official_validation_used_for_confirmation": "false",
            "held_out_test_accessed": "false",
        } for arm in ARMS],
    )
    _atomic_csv(
        output_root / "confirmation_predictions.csv",
        CONFIRMATION_PREDICTION_FIELDS, predictions,
    )
    metadata = {
        "schema_version": 1, "status": "complete", "run_mode": SHARD_MODE,
        "project_commit": project_commit,
        "runner_source_sha256": sha256(RUNNER_PATH),
        "adapter_source_sha256": sha256(ADAPTER_PATH),
        "task_treatment_pilots_source_sha256": sha256(
            TASK_TREATMENT_PILOTS_PATH
        ),
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "launch_authorization_sha256": launch_authorization_sha256,
        "binding_sha256": binding, "input_bindings": contract["immutable_inputs"],
        "shard_id": shard_id, "schedule_unit_index": shard["schedule_unit_index"],
        "outer_fold": shard["outer_fold"],
        "training_seed": shard["training_seed"],
        "schedule_unit_sha256": data.unit["unit_sha256"], "arms": list(ARMS),
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": _runtime_fingerprint_sha256(
            runtime_fingerprint
        ),
        "observed_cuda_device_count": observed_cuda_device_count,
        "git_worktree_clean": git_boundary["git_worktree_clean"],
        "git_submodules_clean": git_boundary["git_submodules_clean"],
        "epochs": 40, "batches_per_epoch": 105, "batch_size": 64,
        "completed_arms": list(ARMS), "optimizer_steps_completed": 12600,
        "full_training_authorized": True,
        "scientific_decision_permitted_after_complete_matrix_only": True,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False, "device": str(device),
        "python": sys.version.split()[0], "torch": torch.__version__,
    }
    _atomic_json(output_root / "shard_run_metadata.json", metadata)
    artifacts = _regular_file_hashes(output_root, {"full_shard_manifest.json"})
    if len(artifacts) != 24:
        raise ValueError(f"full-shard output must bind 24 files, got {len(artifacts)}")
    manifest = {
        "schema_version": 1, "status": "complete", "run_mode": SHARD_MODE,
        "project_commit": project_commit,
        "runner_source_sha256": sha256(RUNNER_PATH),
        "adapter_source_sha256": sha256(ADAPTER_PATH),
        "task_treatment_pilots_source_sha256": sha256(
            TASK_TREATMENT_PILOTS_PATH
        ),
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "launch_authorization_sha256": launch_authorization_sha256,
        "binding_sha256": binding, "input_bindings": contract["immutable_inputs"],
        "shard_id": shard_id, "schedule_unit_index": shard["schedule_unit_index"],
        "outer_fold": shard["outer_fold"],
        "training_seed": shard["training_seed"],
        "schedule_unit_sha256": data.unit["unit_sha256"],
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": _runtime_fingerprint_sha256(
            runtime_fingerprint
        ),
        "observed_cuda_device_count": observed_cuda_device_count,
        "git_worktree_clean": git_boundary["git_worktree_clean"],
        "git_submodules_clean": git_boundary["git_submodules_clean"],
        "arms": list(ARMS), "epochs": 40, "batches_per_epoch": 105,
        "batch_size": 64, "optimizer_steps_per_arm": 4200,
        "optimizer_steps_per_shard": 12600,
        "full_training_authorized": True,
        "scientific_decision_permitted_after_complete_matrix_only": True,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False, "artifact_sha256": artifacts,
    }
    _atomic_json(output_root / "full_shard_manifest.json", manifest)
    return {**manifest, "output": str(output_root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--launch-authorization", type=Path, required=True)
    parser.add_argument("--launch-authorization-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run(
        args.vector_root, args.protocol_root, args.schedule_root,
        args.smoke_root, args.output_root, args.project_commit, args.shard_id,
        args.launch_authorization, args.launch_authorization_sha256, args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("P4B TASK-SEGMENTED FULL SHARD: PASS")


if __name__ == "__main__":
    main()
