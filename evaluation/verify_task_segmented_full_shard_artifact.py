"""Independently verify one sealed P4b full-scientific fold/seed shard.

This verifier is deliberately standard-library only.  In particular, it
never imports NumPy or PyTorch and never deserializes either checkpoint.  The
checkpoint files are opaque bytes bound by the top-level manifest.

The production verifier accepts the expected full-execution-contract and
one-time launch-authorization digests as required inputs.  Those values are
not scientific tuning switches: they bind the artifact to the prospectively
frozen execution plan and the separately issued launch authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


MANIFEST_NAME = "full_shard_manifest.json"
METADATA_NAME = "shard_run_metadata.json"
ARM_IDS = (
    "global_mixed",
    "true_task_segmented",
    "pseudo_task_segmented",
)
TOP_LEVEL_ENTRIES = frozenset({
    MANIFEST_NAME,
    METADATA_NAME,
    "run_manifest.csv",
    "confirmation_predictions.csv",
    "runs",
})
PER_ARM_FILES = frozenset({
    "resume_checkpoint.pt",
    "best_checkpoint.pt",
    "training_history.csv",
    "step_trace.csv",
    "checkpoint_positive_ranks.csv",
    "confirmation_candidate_scores.csv",
    "run_summary.json",
})
NON_MANIFEST_FILES = frozenset(
    {
        METADATA_NAME,
        "run_manifest.csv",
        "confirmation_predictions.csv",
    }
    | {
        f"runs/{arm}/{name}"
        for arm in ARM_IDS
        for name in PER_ARM_FILES
    }
)

RUN_FIELDS = (
    "arm_id", "outer_fold", "training_seed", "status", "run_mode",
    "full_training_authorized", "scientific_decision_permitted",
    "official_validation_used_for_confirmation", "held_out_test_accessed",
)
PREDICTION_FIELDS = (
    "arm_id", "training_seed", "outer_fold", "trial_id", "reading_task",
    "subject_id", "normalized_text_sha256", "signal_condition",
    "positive_rank", "candidate_pool_size", "scientific_decision_permitted",
)
HISTORY_FIELDS = (
    "arm_id", "outer_fold", "training_seed", "epoch", "optimizer_steps",
    "mean_train_loss", "checkpoint_macro_mrr", "checkpoint_nr_mrr",
    "checkpoint_tsr_mrr", "is_best",
)
STEP_FIELDS = (
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

IMMUTABLE_INPUTS: dict[str, dict[str, Any]] = {
    "prompt_neutral_vectors": {
        "dataset_slug": "thestonedape/task-aware-eegtotext",
        "dataset_version": 2,
        "preserved_source_id": (
            "kaggle-dataset-thestonedape-task-aware-eegtotext-version-2"
        ),
        "combined_manifest_sha256": (
            "6c1fff8d2e89e33a72d03c39651e8ecce678c3b93cdb66747dd6dcc00538cddb"
        ),
        "eeg_vector_index_sha256": (
            "373e49da7d3d6d00aaae414437886033e7f2c6a938a92662f76a0973744e0ae9"
        ),
        "text_vector_index_sha256": (
            "5b0a9579cbc586a7eaab353cee7ba65ab538a5e9cbd333e51ccab9855bde98ae"
        ),
        "trial_text_targets_sha256": (
            "1cde6365bdfe523d2497cba56375d1a27d917cfab02d7d6a407564ce32ca2b21"
        ),
        "clean_remount_verification_report_sha256": (
            "e5a01e8342bd44a87e512c8251cefa930e62258a591f52bff9f3fd4bc682b1bf"
        ),
    },
    "task_segmented_protocol": {
        "dataset_slug": (
            "thestonedape/task-aware-eeg2text-task-segmented-protocol"
        ),
        "dataset_version": 1,
        "preserved_source_id": (
            "kaggle-dataset-thestonedape-task-aware-eeg2text-"
            "task-segmented-protocol-version-1"
        ),
        "contract_sha256": (
            "396670afc0244cb601364ff89df53944c4f63402191a9d120e6e2648e5baed3b"
        ),
        "report_sha256": (
            "f99ff4ad371e30b86dc9582bb4c35854f6bba863d7868a5f424f93776eab8116"
        ),
        "run_metadata_sha256": (
            "766c836c96c2025805d81bbbad7d3378faf14b8eca4fc1885653add5a1f35dc9"
        ),
        "artifact_sha256": {
            "batch_grid_feasibility.csv": "91356aad08bfabf003852e643df11d2589aaed6216da342251067d26751e9ff5",
            "candidate_pools.csv": "a0b4102acf88d956fc494a5d4237fbdac0e083bf38e5f2dbf621d40b0942ca0c",
            "confirmation_donors.csv": "fa599beb26fe6ffe1e3b5e849a8fefc0cddeef71bba4554adfd310bba203f87c",
            "outer_split_assignments.csv": "ef709d162961b423ccf9065c01b21725acd17f9a99ed99d66152be50bb5fe911",
            "protocol_registry.json": "38106f433ca39af83aac04b17fe06b96c8f4b005fbb32734ec8346994ee24120",
            "pseudo_groups.csv": "12b77333eebe2ea5f8cba45307fedbdf46e0a921934d58d992462bf5577a7085",
            "text_group_folds.csv": "0849e048f7670ae74b06ab3fde74628361efba5c328e8a1743a792e787fad472",
        },
    },
    "training_schedule": {
        "dataset_slug": (
            "thestonedape/task-aware-eeg2text-task-segmented-schedule"
        ),
        "dataset_version": 1,
        "preserved_source_id": (
            "kaggle-dataset-thestonedape-task-aware-eeg2text-"
            "task-segmented-schedule-version-1"
        ),
        "contract_sha256": (
            "a6ea34388cd98380654f413b1440d0d5cee0b8065555b0c04a53d0db6ea12287"
        ),
        "manifest_sha256": (
            "0cf2a752b5f0a67e7282bc0b4551b4792ceb3d346dd441fef6c359515885270a"
        ),
        "report_sha256": (
            "5e7d953a8ca31e48d6acbd6168b146d695af449ef0ad5bc573a37d306cb7250f"
        ),
        "clean_remount_verification_report_sha256": (
            "65609e0dc9ae48403a16ddf39f08403fbe0578e96ce8b01255c966252498d193"
        ),
        "shape": [15, 40, 105, 64],
        "artifact_sha256": {
            "parent_protocol_verification_report.json": "376fedd2ca89189653a6a4e784195411bea061c63b220a8dfce33cba0b8b4b32",
            "schedule_audit.csv": "f18299fbba6725f7eb33c4686bdd496a83de139c76f3494f48f6a1148d169fbf",
            "schedule_freeze_run_metadata.json": "8487422b900573e8b44d7e83d78eb32beb9786f8496a972c5585291c3576dc1b",
            "schedule_indices.u32le": "79543e72f496ee3f7a8140556b274c15ecc5992e900a07bed5e5c74a2ddd7cbc",
            "schedule_units.csv": "e2644a51ac578b18388ce82d07d910714182cf674e60b0a5f2c547067b8720aa",
            "task_segmented_training_schedule_contract.json": "a6ea34388cd98380654f413b1440d0d5cee0b8065555b0c04a53d0db6ea12287",
            "task_segmented_training_schedule_manifest.json": "0cf2a752b5f0a67e7282bc0b4551b4792ceb3d346dd441fef6c359515885270a",
            "task_segmented_training_schedule_report.json": "5e7d953a8ca31e48d6acbd6168b146d695af449ef0ad5bc573a37d306cb7250f",
            "trial_catalog.csv": "3d93e0cea4290ac22e8111760241d04109392e6f024abae85a4d9504fc4f8fc9",
        },
    },
}

PRESERVED_SMOKE: dict[str, Any] = {
    "dataset_slug": "thestonedape/task-aware-eeg2text-task-segmented-smoke",
    "dataset_version": 1,
    "preserved_source_id": (
        "kaggle-dataset-thestonedape-task-aware-eeg2text-"
        "task-segmented-smoke-version-1"
    ),
    "manifest_sha256": "2cf38c78bdc25815fbb16ad17832ac4062fe48dbc814bcf82c6999b2950ec2a3",
    "run_metadata_sha256": "91aa75ca77889627d4d2c42a104dd7063627d5b1996a67a81d65046d165b7f94",
    "execution_contract_sha256": "9fd862b970ab95ded6f5efa0eb5290f8687bad7fd3a7f251f9ca66176de9f813",
    "runner_source_sha256": "62c246f6f8979b8a2f58b6b36072c7ac238d8c86a2fcdd0e3f2843af90cfef7c",
    "clean_remount_verification_report_sha256": "a2537e4a10de659705c956ec74105239e2046b5d42ca01a3524e1480b1081187",
    "artifact_sha256": {
        "arm_summary.csv": "351e9849644a08ee673911ec11d904390fb8f77ce83c99cfb282efc4c44003d6",
        "common_batch_trace.csv": "c4280ab853a0470a2a7d5381c2f2737289d8257fe22e84561436224a0cccd95a",
        "runs/global_mixed/resume_checkpoint.pt": "008ac974edf360543b55e1eb796b2bfae0a459f12cf1ded6fed8db66e8a34fef",
        "runs/global_mixed/run_summary.json": "a86c28152c58618b3488e1cd6469f91f09e7ec098e19e3ad79b6e8bca1f95f4d",
        "runs/global_mixed/step_trace.csv": "3a41a7221dc34c28d013beb0840e786ce10379b33cb27511b413f5b733a13168",
        "runs/pseudo_task_segmented/resume_checkpoint.pt": "d2e7c995d6f7a62a9cd534183a90eb6fb66577efde62b98e3fd7403a09a4c44c",
        "runs/pseudo_task_segmented/run_summary.json": "fb90a8e6eb41a75fb3979ff981b26b8dd494f3c65ae86fa604a508122b5b35f8",
        "runs/pseudo_task_segmented/step_trace.csv": "31ea0d55b05ff5c9684207a0406488188e6487e8d4d439fd4aa2d08523543339",
        "runs/true_task_segmented/resume_checkpoint.pt": "9d173be21715fa25d8653cc4eee14cd0eb43b714c2131a85ca206f10d38ad429",
        "runs/true_task_segmented/run_summary.json": "278f5666938540f16be3c1db3ee9f9aeb0a38e001daf03fb81502092e410128d",
        "runs/true_task_segmented/step_trace.csv": "dbec21a47612554fef2399021856d0b1f4c6d845b7f5c710cae25b0ee6c35fb7",
        "smoke_run_metadata.json": "91aa75ca77889627d4d2c42a104dd7063627d5b1996a67a81d65046d165b7f94",
    },
}

FULL_INPUT_BINDINGS: dict[str, dict[str, Any]] = {
    **IMMUTABLE_INPUTS,
    "preserved_smoke": PRESERVED_SMOKE,
}

EXACT_RUNTIME_ENVIRONMENT: dict[str, Any] = {
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

METADATA_FIELDS = frozenset({
    "schema_version", "status", "run_mode", "project_commit",
    "runner_source_sha256", "adapter_source_sha256",
    "task_treatment_pilots_source_sha256",
    "full_execution_contract_sha256", "launch_authorization_sha256",
    "binding_sha256", "input_bindings", "shard_id", "schedule_unit_index",
    "outer_fold", "training_seed", "schedule_unit_sha256",
    "runtime_fingerprint", "runtime_fingerprint_sha256",
    "observed_cuda_device_count", "git_worktree_clean",
    "git_submodules_clean", "arms", "epochs",
    "batches_per_epoch", "batch_size", "completed_arms",
    "optimizer_steps_completed", "full_training_authorized",
    "scientific_decision_permitted_after_complete_matrix_only",
    "official_validation_used_for_confirmation", "held_out_test_accessed",
    "device", "python", "torch",
})

SUMMARY_FIELDS = frozenset({
    "schema_version", "status", "run_mode", "shard_id",
    "schedule_unit_index", "outer_fold", "training_seed", "arm_id",
    "binding_sha256", "schedule_unit_sha256", "epochs",
    "batches_per_epoch", "optimizer_steps", "trainable_parameters",
    "initial_state_sha256", "final_state_sha256",
    "final_optimizer_state_sha256", "best_epoch",
    "best_checkpoint_macro_mrr", "checkpoint_trial_count",
    "checkpoint_rank_rows", "confirmation_trial_count",
    "confirmation_prediction_rows", "confirmation_candidate_score_rows",
    "resumed", "artifact_sha256", "full_training_authorized",
    "scientific_decision_permitted",
    "official_validation_used_for_confirmation", "held_out_test_accessed",
})

REPORT_FIELDS = frozenset({
    "status", "schema_version", "preserved_source_id",
    "full_shard_manifest_sha256", "full_execution_contract_sha256",
    "launch_authorization_sha256", "outer_fold", "training_seed",
    "schedule_unit_sha256", "arms", "epochs_per_arm",
    "optimizer_steps_per_arm", "total_optimizer_steps",
    "checkpoint_rows_per_arm", "confirmation_prediction_rows",
    "confirmation_candidate_score_rows", "best_epochs",
    "paired_initial_state", "epoch_zero_eligible",
    "earliest_tie_selection_verified",
    "append_only_strict_incumbent_history_verified",
    "partial_scientific_decision_permitted",
    "correct_and_matched_wrong_provenance_verified",
    "preserved_protocol_evidence_verified",
    "preserved_schedule_evidence_verified", "recomputed_binding_sha256",
    "preserved_evidence_sha256", "runtime_fingerprint_verified",
    "runtime_fingerprint_sha256", "git_execution_boundary_verified",
    "checkpoint_deserialized",
    "official_validation_used_for_confirmation", "held_out_test_accessed",
    "verified_artifact_sha256",
})


@dataclass(frozen=True)
class VerificationSpec:
    arms: tuple[str, ...] = ARM_IDS
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (20260717, 20260718, 20260719)
    fold_counts: tuple[int, ...] = (1810, 1785, 1790, 1814, 1812)
    epochs: int = 40
    batches_per_epoch: int = 105
    pool_size: int = 24

    def confirmation_count(self, fold: int) -> int:
        return self.fold_counts[fold]

    def checkpoint_count(self, fold: int) -> int:
        return self.fold_counts[(fold + 1) % len(self.fold_counts)]

    @property
    def optimizer_steps(self) -> int:
        return self.epochs * self.batches_per_epoch


PRODUCTION_SPEC = VerificationSpec()


@dataclass(frozen=True)
class EvidenceHashes:
    candidate_pools_sha256: str
    confirmation_donors_sha256: str
    schedule_indices_sha256: str
    schedule_units_sha256: str
    trial_catalog_sha256: str


PRODUCTION_EVIDENCE_HASHES = EvidenceHashes(
    candidate_pools_sha256=FULL_INPUT_BINDINGS["task_segmented_protocol"]
        ["artifact_sha256"]["candidate_pools.csv"],
    confirmation_donors_sha256=FULL_INPUT_BINDINGS["task_segmented_protocol"]
        ["artifact_sha256"]["confirmation_donors.csv"],
    schedule_indices_sha256=FULL_INPUT_BINDINGS["training_schedule"]
        ["artifact_sha256"]["schedule_indices.u32le"],
    schedule_units_sha256=FULL_INPUT_BINDINGS["training_schedule"]
        ["artifact_sha256"]["schedule_units.csv"],
    trial_catalog_sha256=FULL_INPUT_BINDINGS["training_schedule"]
        ["artifact_sha256"]["trial_catalog.csv"],
)


@dataclass(frozen=True)
class PreservedEvidence:
    checkpoint_targets: Mapping[str, str]
    confirmation_targets: Mapping[str, str]
    confirmation_pools: Mapping[str, tuple[Mapping[str, str], ...]]
    confirmation_donors: Mapping[str, Mapping[str, str]]
    schedule_unit_sha256: str
    batch_bindings: Mapping[tuple[int, int], tuple[str, str]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def runtime_fingerprint_sha256(runtime_fingerprint: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(runtime_fingerprint), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def is_commit(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def parse_bool(value: object, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label}: expected true/false, got {value!r}")


def parse_int(value: object, label: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected integer, got {value!r}") from exc
    return parsed


def parse_float(value: object, label: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected float, got {value!r}") from exc
    require(math.isfinite(parsed), f"{label}: non-finite value")
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require_equal(reader.fieldnames, list(fields), f"columns in {path.name}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot read CSV: {path}") from exc
    require(
        all(None not in row and all(value is not None for value in row.values())
            for row in rows),
        f"malformed CSV row in {path}",
    )
    return rows


def _preserved_file(root: Path, name: str, expected_sha256: str) -> Path:
    resolved_root = root.resolve(strict=True)
    require(resolved_root.is_dir() and not root.is_symlink(),
            f"preserved root is not a real directory: {root}")
    path = root / name
    require(path.is_file() and not path.is_symlink(),
            f"preserved input is not a regular file: {name}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"preserved input escapes its root: {name}") from exc
    require_equal(sha256(resolved), expected_sha256,
                  f"preserved input SHA256 for {name}")
    return resolved


def _unique_rows(
    rows: Sequence[Mapping[str, str]], key: str, label: str,
) -> dict[str, Mapping[str, str]]:
    output: dict[str, Mapping[str, str]] = {}
    for row in rows:
        identity = row[key]
        require(identity and identity not in output,
                f"duplicate/empty {label}: {identity!r}")
        output[identity] = row
    return output


def _load_preserved_evidence(
    protocol_root: Path, schedule_root: Path, fold: int, seed: int,
    spec: VerificationSpec, hashes: EvidenceHashes,
) -> PreservedEvidence:
    for value in (
        hashes.candidate_pools_sha256, hashes.confirmation_donors_sha256,
        hashes.schedule_indices_sha256, hashes.schedule_units_sha256,
        hashes.trial_catalog_sha256,
    ):
        require(is_sha256(value), "invalid preserved-evidence SHA256")
    candidate_path = _preserved_file(
        protocol_root, "candidate_pools.csv", hashes.candidate_pools_sha256
    )
    donor_path = _preserved_file(
        protocol_root, "confirmation_donors.csv",
        hashes.confirmation_donors_sha256,
    )
    indices_path = _preserved_file(
        schedule_root, "schedule_indices.u32le", hashes.schedule_indices_sha256
    )
    units_path = _preserved_file(
        schedule_root, "schedule_units.csv", hashes.schedule_units_sha256
    )
    catalog_path = _preserved_file(
        schedule_root, "trial_catalog.csv", hashes.trial_catalog_sha256
    )

    catalog_rows = read_csv(catalog_path, CATALOG_FIELDS)
    catalog_by_trial: dict[str, Mapping[str, str]] = {}
    for index, row in enumerate(catalog_rows):
        require_equal(parse_int(row["trial_index"], "catalog trial index"), index,
                      "ordered catalog trial index")
        trial = row["trial_id"]
        require(trial and trial not in catalog_by_trial,
                f"duplicate/empty catalog trial: {trial!r}")
        require(row["reading_task"] in {"NR", "TSR"},
                f"invalid catalog task: {trial}")
        require(is_sha256(row["normalized_text_sha256"]),
                f"invalid catalog text identity: {trial}")
        catalog_by_trial[trial] = row

    selected_candidates = [
        row for row in read_csv(candidate_path, CANDIDATE_FIELDS)
        if parse_int(row["outer_fold"], "candidate outer fold") == fold
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in selected_candidates:
        partition = row["partition"]
        require(partition in {"checkpoint", "confirmation"},
                f"invalid frozen candidate partition: {partition}")
        grouped[(partition, row["target_trial_id"])].append(row)
    targets: dict[str, dict[str, str]] = {"checkpoint": {}, "confirmation": {}}
    confirmation_pools: dict[str, tuple[Mapping[str, str], ...]] = {}
    for (partition, trial), members in grouped.items():
        require(trial in catalog_by_trial,
                f"frozen candidate target is absent from catalog: {trial}")
        ordered = tuple(sorted(
            members, key=lambda row: parse_int(row["candidate_rank"], "candidate rank")
        ))
        require_equal(
            [parse_int(row["candidate_rank"], "candidate rank") for row in ordered],
            list(range(spec.pool_size)), f"frozen candidate ranks: {(partition, trial)}",
        )
        tasks = {row["reading_task"] for row in ordered}
        require_equal(len(tasks), 1, f"frozen candidate task consistency: {trial}")
        task = next(iter(tasks))
        require_equal(task, catalog_by_trial[trial]["reading_task"],
                      f"frozen candidate/catalog task: {trial}")
        require_equal(sum(parse_bool(row["is_positive"], "frozen positive")
                          for row in ordered), 1,
                      f"frozen positive count: {(partition, trial)}")
        expected_donors = 1 if partition == "confirmation" else 0
        require_equal(sum(parse_bool(row["is_designated_donor_text"],
                                     "frozen donor flag") for row in ordered),
                      expected_donors,
                      f"frozen designated-donor count: {(partition, trial)}")
        require_equal(len({row["candidate_normalized_text_sha256"] for row in ordered}),
                      spec.pool_size, f"frozen candidate text uniqueness: {trial}")
        for row in ordered:
            require(is_sha256(row["candidate_normalized_text_sha256"]),
                    f"invalid frozen candidate text identity: {trial}")
            require(bool(row["candidate_text_target_id"]),
                    f"empty frozen candidate text target: {trial}")
        positive = next(row for row in ordered
                        if parse_bool(row["is_positive"], "frozen positive"))
        require_equal(positive["candidate_normalized_text_sha256"],
                      catalog_by_trial[trial]["normalized_text_sha256"],
                      f"frozen positive/catalog text: {trial}")
        require(trial not in targets[partition], f"duplicate frozen target: {trial}")
        targets[partition][trial] = task
        if partition == "confirmation":
            confirmation_pools[trial] = ordered
    require_equal(len(targets["checkpoint"]), spec.checkpoint_count(fold),
                  "frozen checkpoint target count")
    require_equal(len(targets["confirmation"]), spec.confirmation_count(fold),
                  "frozen confirmation target count")

    selected_donors = [
        row for row in read_csv(donor_path, DONOR_FIELDS)
        if parse_int(row["outer_fold"], "donor outer fold") == fold
    ]
    donors = _unique_rows(selected_donors, "target_trial_id", "confirmation donor")
    require_equal(set(donors), set(targets["confirmation"]),
                  "frozen confirmation-donor targets")
    for target_id, donor in donors.items():
        target = catalog_by_trial[target_id]
        donor_id = donor["donor_trial_id"]
        require(donor_id in catalog_by_trial and donor_id != target_id,
                f"invalid frozen donor trial: {target_id}")
        signal = catalog_by_trial[donor_id]
        require_equal(donor["partition"], "confirmation",
                      f"frozen donor partition: {target_id}")
        require_equal(donor["reading_task"], target["reading_task"],
                      f"frozen donor target task: {target_id}")
        require_equal(signal["reading_task"], target["reading_task"],
                      f"frozen donor signal task: {target_id}")
        require_equal(donor["subject_id"], target["subject_id"],
                      f"frozen donor target subject: {target_id}")
        require_equal(signal["subject_id"], target["subject_id"],
                      f"frozen donor signal subject: {target_id}")
        require_equal(donor["target_normalized_text_sha256"],
                      target["normalized_text_sha256"],
                      f"frozen donor target text: {target_id}")
        require_equal(donor["donor_normalized_text_sha256"],
                      signal["normalized_text_sha256"],
                      f"frozen donor signal text: {target_id}")
        require(signal["normalized_text_sha256"] != target["normalized_text_sha256"],
                f"frozen donor reuses target text: {target_id}")
        designated = [row for row in confirmation_pools[target_id]
                      if parse_bool(row["is_designated_donor_text"],
                                    "frozen donor flag")]
        require_equal(designated[0]["candidate_normalized_text_sha256"],
                      signal["normalized_text_sha256"],
                      f"frozen pool/donor text: {target_id}")

    unit_index = spec.folds.index(fold) * len(spec.seeds) + spec.seeds.index(seed)
    unit_rows = read_csv(units_path, UNIT_FIELDS)
    matches = [row for row in unit_rows
               if parse_int(row["unit_index"], "schedule unit index") == unit_index]
    require_equal(len(matches), 1, "frozen schedule-unit uniqueness")
    unit = matches[0]
    for field, expected in (
        ("outer_fold", fold), ("training_seed", seed),
        ("epochs", spec.epochs), ("batches_per_epoch", spec.batches_per_epoch),
        ("batch_size", 64),
    ):
        require_equal(parse_int(unit[field], f"schedule unit {field}"), expected,
                      f"frozen schedule-unit {field}")
    count = spec.optimizer_steps * 64
    require_equal(parse_int(unit["uint32_count"], "schedule uint32 count"), count,
                  "frozen schedule-unit uint32 count")
    offset = parse_int(unit["byte_offset"], "schedule byte offset")
    byte_length = count * 4
    require_equal(parse_int(unit["byte_length"], "schedule byte length"), byte_length,
                  "frozen schedule-unit byte length")
    require(is_sha256(unit["unit_sha256"]), "invalid frozen schedule-unit SHA256")
    with indices_path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(byte_length)
    require_equal(len(payload), byte_length, "frozen schedule-unit payload length")
    require_equal(hashlib.sha256(payload).hexdigest(), unit["unit_sha256"],
                  "frozen schedule-unit payload SHA256")
    indices = [value[0] for value in struct.iter_unpack("<I", payload)]
    batch_bindings: dict[tuple[int, int], tuple[str, str]] = {}
    for step in range(spec.optimizer_steps):
        epoch = step // spec.batches_per_epoch + 1
        batch = step % spec.batches_per_epoch
        members = indices[step * 64:(step + 1) * 64]
        require_equal(len(members), 64, "frozen schedule batch size")
        require_equal(len(set(members)), 64, "frozen schedule batch uniqueness")
        require(all(0 <= index < len(catalog_rows) for index in members),
                "frozen schedule contains out-of-range catalog index")
        index_sha = hashlib.sha256(
            b"".join(struct.pack("<I", index) for index in members)
        ).hexdigest()
        trial_sha = hashlib.sha256(
            ("\n".join(catalog_rows[index]["trial_id"] for index in members)
             + "\n").encode("utf-8")
        ).hexdigest()
        batch_bindings[(epoch, batch)] = (index_sha, trial_sha)
    return PreservedEvidence(
        checkpoint_targets=targets["checkpoint"],
        confirmation_targets=targets["confirmation"],
        confirmation_pools=confirmation_pools,
        confirmation_donors=donors,
        schedule_unit_sha256=unit["unit_sha256"],
        batch_bindings=batch_bindings,
    )


def _safe_relative(value: object) -> str:
    require(isinstance(value, str), "manifest artifact path must be a string")
    path = PurePosixPath(value)
    require(
        value == path.as_posix() and not path.is_absolute()
        and value not in {"", "."} and ".." not in path.parts,
        f"unsafe manifest artifact path: {value!r}",
    )
    return value


def _exact_inventory(root: Path) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "artifact root is not a real directory")
    require_equal(set(path.name for path in root.iterdir()), set(TOP_LEVEL_ENTRIES),
                  "top-level artifact inventory")
    runs = root / "runs"
    require(runs.is_dir() and not runs.is_symlink(), "runs is not a real directory")
    require_equal(set(path.name for path in runs.iterdir()), set(ARM_IDS),
                  "run-arm inventory")
    for arm in ARM_IDS:
        arm_root = runs / arm
        require(arm_root.is_dir() and not arm_root.is_symlink(),
                f"arm directory is invalid: {arm}")
        require_equal(set(path.name for path in arm_root.iterdir()), set(PER_ARM_FILES),
                      f"{arm} inventory")
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"artifact contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            require(relative == MANIFEST_NAME or relative in NON_MANIFEST_FILES,
                    f"unexpected regular file: {relative}")
            if relative != MANIFEST_NAME:
                actual[relative] = sha256(path)
    require_equal(set(actual), set(NON_MANIFEST_FILES), "regular-file inventory")
    for arm in ARM_IDS:
        for name in ("resume_checkpoint.pt", "best_checkpoint.pt"):
            require((runs / arm / name).stat().st_size > 0,
                    f"empty opaque checkpoint: {arm}/{name}")
    return actual


def _validate_preserved_smoke(value: object) -> dict[str, Any]:
    require(isinstance(value, dict), "preserved_smoke must be an object")
    require_equal(value, PRESERVED_SMOKE, "preserved smoke binding")
    return dict(value)


def _check_common_binding(
    document: Mapping[str, Any], *, label: str, manifest: Mapping[str, Any],
    expected_contract_sha256: str, expected_launch_sha256: str,
) -> None:
    for field in (
        "project_commit", "runner_source_sha256", "adapter_source_sha256",
        "task_treatment_pilots_source_sha256",
        "full_execution_contract_sha256", "launch_authorization_sha256",
        "binding_sha256", "input_bindings", "shard_id", "schedule_unit_index",
        "outer_fold", "training_seed", "schedule_unit_sha256",
        "runtime_fingerprint", "runtime_fingerprint_sha256",
        "observed_cuda_device_count", "git_worktree_clean",
        "git_submodules_clean",
    ):
        require_equal(document.get(field), manifest.get(field), f"{label} {field}")
    require_equal(document.get("full_execution_contract_sha256"),
                  expected_contract_sha256, f"{label} full contract")
    require_equal(document.get("launch_authorization_sha256"),
                  expected_launch_sha256, f"{label} launch authorization")


def shard_definition(
    fold: int, seed: int, spec: VerificationSpec = PRODUCTION_SPEC,
) -> dict[str, Any]:
    require(fold in spec.folds and seed in spec.seeds,
            "cannot construct a shard outside the frozen matrix")
    fold_position = spec.folds.index(fold)
    unit_index = fold_position * len(spec.seeds) + spec.seeds.index(seed)
    checkpoint_fold = spec.folds[(fold_position + 1) % len(spec.folds)]
    fit_folds = [value for value in spec.folds
                 if value not in {fold, checkpoint_fold}]
    return {
        "shard_id": f"p4b-f{fold}-s{seed}",
        "schedule_unit_index": unit_index,
        "outer_fold": fold,
        "training_seed": seed,
        "confirmation_fold": fold,
        "checkpoint_fold": checkpoint_fold,
        "fit_folds": fit_folds,
    }


def recompute_binding_sha256(
    manifest: Mapping[str, Any], runtime_fingerprint: Mapping[str, Any],
    spec: VerificationSpec = PRODUCTION_SPEC,
) -> str:
    fold = parse_int(manifest.get("outer_fold"), "binding outer fold")
    seed = parse_int(manifest.get("training_seed"), "binding training seed")
    payload = {
        "schema_version": 1,
        "run_mode": "full_scientific_shard",
        "full_execution_contract_sha256": manifest.get(
            "full_execution_contract_sha256"
        ),
        "launch_authorization_sha256": manifest.get(
            "launch_authorization_sha256"
        ),
        "project_commit": manifest.get("project_commit"),
        "runner_source_sha256": manifest.get("runner_source_sha256"),
        "adapter_source_sha256": manifest.get("adapter_source_sha256"),
        "task_treatment_pilots_source_sha256": manifest.get(
            "task_treatment_pilots_source_sha256"
        ),
        "input_bindings": manifest.get("input_bindings"),
        "shard": shard_definition(fold, seed, spec),
        "schedule_unit_sha256": manifest.get("schedule_unit_sha256"),
        "runtime_fingerprint": dict(runtime_fingerprint),
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256(
            runtime_fingerprint
        ),
        "git_worktree_clean": manifest.get("git_worktree_clean"),
        "git_submodules_clean": manifest.get("git_submodules_clean"),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _verify_run_manifest(
    path: Path, fold: int, seed: int, spec: VerificationSpec,
) -> None:
    rows = read_csv(path, RUN_FIELDS)
    require_equal(len(rows), len(spec.arms), "run-manifest row count")
    by_arm: dict[str, dict[str, str]] = {}
    for row in rows:
        arm = row["arm_id"]
        require(arm in spec.arms and arm not in by_arm, f"invalid/duplicate run arm: {arm}")
        require_equal(parse_int(row["outer_fold"], "run outer_fold"), fold,
                      "run outer fold")
        require_equal(parse_int(row["training_seed"], "run training_seed"), seed,
                      "run seed")
        require_equal(row["status"], "complete", f"{arm} run status")
        require_equal(row["run_mode"], "full_scientific", f"{arm} run mode")
        require(parse_bool(row["full_training_authorized"], "full authorization"),
                f"{arm} is not full-training authorized")
        require(not parse_bool(row["scientific_decision_permitted"],
                               "decision permission"),
                f"{arm} partial shard improperly permits a scientific decision")
        require(not parse_bool(row["official_validation_used_for_confirmation"],
                               "official validation flag"),
                f"{arm} used official validation for confirmation")
        require(not parse_bool(row["held_out_test_accessed"], "test flag"),
                f"{arm} accessed held-out test")
        by_arm[arm] = row
    require_equal(set(by_arm), set(spec.arms), "run-manifest arms")


def _verify_step_trace(
    path: Path, arm: str, fold: int, seed: int, schedule_sha: str,
    batch_bindings: Mapping[tuple[int, int], tuple[str, str]],
    spec: VerificationSpec,
) -> tuple[str, str, str, tuple[tuple[int, int, str, str, str], ...]]:
    rows = read_csv(path, STEP_FIELDS)
    require_equal(len(rows), spec.optimizer_steps, f"{arm} step count")
    initial: str | None = None
    prior_post: str | None = None
    prior_optimizer: str | None = None
    schedule_trace: list[tuple[int, int, str, str, str]] = []
    for offset, row in enumerate(rows):
        epoch = offset // spec.batches_per_epoch + 1
        batch = offset % spec.batches_per_epoch
        require_equal(parse_int(row["outer_fold"], "step outer_fold"), fold,
                      f"{arm} step fold")
        require_equal(parse_int(row["training_seed"], "step seed"), seed,
                      f"{arm} step seed")
        require_equal(row["arm_id"], arm, f"{arm} step arm")
        require_equal(parse_int(row["epoch"], "step epoch"), epoch,
                      f"{arm} epoch ordering")
        require_equal(parse_int(row["batch_index"], "step batch"), batch,
                      f"{arm} batch ordering")
        require_equal(parse_int(row["global_step"], "global step"), offset + 1,
                      f"{arm} global-step ordering")
        require_equal(row["schedule_unit_sha256"], schedule_sha,
                      f"{arm} schedule-unit binding")
        require((epoch, batch) in batch_bindings,
                f"{arm} step is absent from preserved schedule")
        expected_indices_sha, expected_trials_sha = batch_bindings[(epoch, batch)]
        require_equal(row["batch_catalog_indices_sha256"], expected_indices_sha,
                      f"{arm} preserved catalog-index batch at step {offset + 1}")
        require_equal(row["batch_trial_ids_sha256"], expected_trials_sha,
                      f"{arm} preserved trial-ID batch at step {offset + 1}")
        expected_key = stable_hash(
            "p4b-global-mask-key-v1", 2026071806,
            FULL_INPUT_BINDINGS["training_schedule"]["contract_sha256"],
            schedule_sha, fold, seed, epoch, batch,
        )
        require_equal(row["global_mask_key"], expected_key,
                      f"{arm} global-mask key at step {offset + 1}")
        for field in (
            "batch_catalog_indices_sha256", "batch_trial_ids_sha256",
            "global_mask_key", "initial_state_sha256", "pre_step_state_sha256",
            "post_step_state_sha256", "optimizer_state_sha256",
        ):
            require(is_sha256(row[field]), f"{arm} invalid {field} at step {offset + 1}")
        parse_float(row["loss"], f"{arm} loss at step {offset + 1}")
        require(parse_float(row["gradient_norm_preclip"],
                            f"{arm} gradient norm at step {offset + 1}") >= 0.0,
                f"{arm} negative gradient norm")
        if initial is None:
            initial = row["initial_state_sha256"]
            require_equal(row["pre_step_state_sha256"], initial,
                          f"{arm} initial pre-step state")
        else:
            require_equal(row["initial_state_sha256"], initial,
                          f"{arm} initial-state drift")
            require_equal(row["pre_step_state_sha256"], prior_post,
                          f"{arm} model-state continuity")
        prior_post = row["post_step_state_sha256"]
        prior_optimizer = row["optimizer_state_sha256"]
        schedule_trace.append((
            epoch, batch, row["batch_catalog_indices_sha256"],
            row["batch_trial_ids_sha256"], row["global_mask_key"],
        ))
    require(initial is not None and prior_post is not None and prior_optimizer is not None,
            f"{arm} trace is empty")
    return initial, prior_post, prior_optimizer, tuple(schedule_trace)


def _verify_checkpoint_ranks(
    path: Path, arm: str, fold: int, seed: int, checkpoint_count: int,
    frozen_targets: Mapping[str, str], spec: VerificationSpec,
) -> dict[int, tuple[float, float, float]]:
    rows = read_csv(path, CHECKPOINT_RANK_FIELDS)
    require_equal(len(rows), (spec.epochs + 1) * checkpoint_count,
                  f"{arm} checkpoint-rank row count")
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    identity_metadata: dict[str, str] = {}
    for row in rows:
        require_equal(row["arm_id"], arm, f"{arm} checkpoint arm")
        require_equal(parse_int(row["outer_fold"], "checkpoint outer_fold"), fold,
                      f"{arm} checkpoint fold")
        require_equal(parse_int(row["training_seed"], "checkpoint seed"), seed,
                      f"{arm} checkpoint seed")
        epoch = parse_int(row["epoch"], "checkpoint epoch")
        require(0 <= epoch <= spec.epochs, f"{arm} invalid checkpoint epoch")
        task = row["reading_task"]
        require(task in {"NR", "TSR"}, f"{arm} invalid checkpoint task")
        rank = parse_int(row["positive_rank"], "checkpoint positive rank")
        require(1 <= rank <= spec.pool_size, f"{arm} invalid checkpoint rank")
        require_equal(parse_int(row["candidate_pool_size"], "checkpoint pool size"),
                      spec.pool_size, f"{arm} checkpoint pool size")
        trial = row["trial_id"]
        require(trial in frozen_targets, f"{arm} non-frozen checkpoint trial: {trial}")
        require_equal(task, frozen_targets[trial],
                      f"{arm} frozen checkpoint task: {trial}")
        if trial in identity_metadata:
            require_equal(task, identity_metadata[trial],
                          f"{arm} checkpoint task drift: {trial}")
        else:
            identity_metadata[trial] = task
        by_epoch[epoch].append(row)
    require_equal(set(by_epoch), set(range(spec.epochs + 1)),
                  f"{arm} checkpoint epochs")
    reference: set[str] | None = None
    metrics: dict[int, tuple[float, float, float]] = {}
    for epoch in range(spec.epochs + 1):
        epoch_rows = by_epoch[epoch]
        require_equal(len(epoch_rows), checkpoint_count,
                      f"{arm} checkpoint rows at epoch {epoch}")
        trials = [row["trial_id"] for row in epoch_rows]
        require_equal(len(set(trials)), checkpoint_count,
                      f"{arm} duplicate checkpoint trial at epoch {epoch}")
        current = set(trials)
        require_equal(current, set(frozen_targets),
                      f"{arm} frozen checkpoint target set at epoch {epoch}")
        if reference is None:
            reference = current
        else:
            require_equal(current, reference,
                          f"{arm} checkpoint identity drift at epoch {epoch}")
        task_values: dict[str, list[float]] = {"NR": [], "TSR": []}
        for row in epoch_rows:
            task_values[row["reading_task"]].append(
                1.0 / parse_int(row["positive_rank"], "checkpoint rank")
            )
        require(all(task_values.values()), f"{arm} missing checkpoint task at epoch {epoch}")
        nr = sum(task_values["NR"]) / len(task_values["NR"])
        tsr = sum(task_values["TSR"]) / len(task_values["TSR"])
        metrics[epoch] = ((nr + tsr) / 2.0, nr, tsr)
    return metrics


def _close(actual: float, expected: float, label: str) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12),
            f"{label}: expected {expected!r}, got {actual!r}")


def _verify_history(
    path: Path, arm: str, fold: int, seed: int,
    metrics: Mapping[int, tuple[float, float, float]], spec: VerificationSpec,
) -> tuple[int, float]:
    rows = read_csv(path, HISTORY_FIELDS)
    require_equal(len(rows), spec.epochs + 1, f"{arm} history row count")
    strict_improvement_epochs: list[int] = []
    incumbent = -math.inf
    for epoch in range(spec.epochs + 1):
        metric = metrics[epoch][0]
        if metric > incumbent:
            strict_improvement_epochs.append(epoch)
            incumbent = metric
    require(strict_improvement_epochs and strict_improvement_epochs[0] == 0,
            f"{arm} epoch zero is not the initial eligible incumbent")
    best_epoch = strict_improvement_epochs[-1]
    marked: list[int] = []
    for epoch, row in enumerate(rows):
        require_equal(row["arm_id"], arm, f"{arm} history arm")
        require_equal(parse_int(row["outer_fold"], "history outer_fold"), fold,
                      f"{arm} history fold")
        require_equal(parse_int(row["training_seed"], "history seed"), seed,
                      f"{arm} history seed")
        require_equal(parse_int(row["epoch"], "history epoch"), epoch,
                      f"{arm} history epoch ordering")
        require_equal(parse_int(row["optimizer_steps"], "history steps"),
                      epoch * spec.batches_per_epoch, f"{arm} history step count")
        if epoch == 0:
            require(row["mean_train_loss"] == "",
                    f"{arm} epoch-zero train loss must be empty")
        else:
            parse_float(row["mean_train_loss"], f"{arm} train loss epoch {epoch}")
        macro, nr, tsr = metrics[epoch]
        _close(parse_float(row["checkpoint_macro_mrr"], "checkpoint macro MRR"),
               macro, f"{arm} macro MRR epoch {epoch}")
        _close(parse_float(row["checkpoint_nr_mrr"], "checkpoint NR MRR"),
               nr, f"{arm} NR MRR epoch {epoch}")
        _close(parse_float(row["checkpoint_tsr_mrr"], "checkpoint TSR MRR"),
               tsr, f"{arm} TSR MRR epoch {epoch}")
        if parse_bool(row["is_best"], "history is_best"):
            marked.append(epoch)
    require_equal(marked, strict_improvement_epochs,
                  f"{arm} append-only strict-improvement history audit")
    return best_epoch, metrics[best_epoch][0]


def _score_positive_rank(rows: Sequence[Mapping[str, str]]) -> int:
    require(rows, "empty candidate-score pool")
    ranked: list[tuple[float, int, bool]] = []
    for row in rows:
        ranked.append((
            parse_float(row["score"], "candidate score"),
            parse_int(row["candidate_rank"], "candidate rank"),
            parse_bool(row["is_positive"], "is_positive"),
        ))
    ordered = sorted(ranked, key=lambda item: (-item[0], item[1]))
    return next(index for index, item in enumerate(ordered, start=1) if item[2])


def _verify_confirmation(
    root: Path, fold: int, seed: int, evidence: PreservedEvidence,
    spec: VerificationSpec,
) -> None:
    confirmation_count = len(evidence.confirmation_targets)
    prediction_rows = read_csv(root / "confirmation_predictions.csv", PREDICTION_FIELDS)
    expected_predictions = len(spec.arms) * 2 * confirmation_count
    require_equal(len(prediction_rows), expected_predictions,
                  "confirmation prediction row count")
    predictions: dict[tuple[str, str, str], dict[str, str]] = {}
    targets: dict[str, dict[str, str]] = {}
    for row in prediction_rows:
        arm = row["arm_id"]
        require(arm in spec.arms, f"unknown prediction arm: {arm}")
        require_equal(parse_int(row["training_seed"], "prediction seed"), seed,
                      "prediction seed")
        require_equal(parse_int(row["outer_fold"], "prediction fold"), fold,
                      "prediction fold")
        trial = row["trial_id"]
        condition = row["signal_condition"]
        require(condition in {"correct", "matched_wrong"},
                f"invalid prediction signal condition: {condition}")
        key = (arm, trial, condition)
        require(key not in predictions, f"duplicate prediction: {key}")
        require(trial in evidence.confirmation_targets,
                f"prediction uses non-frozen confirmation target: {key}")
        require_equal(row["reading_task"], evidence.confirmation_targets[trial],
                      f"frozen prediction target task: {key}")
        require(is_sha256(row["normalized_text_sha256"]),
                f"invalid prediction text identity: {key}")
        stable = {
            "reading_task": row["reading_task"],
            "subject_id": row["subject_id"],
            "normalized_text_sha256": row["normalized_text_sha256"],
        }
        if trial in targets:
            require_equal(stable, targets[trial], f"prediction target drift: {trial}")
        else:
            targets[trial] = stable
        rank = parse_int(row["positive_rank"], "prediction positive rank")
        require(1 <= rank <= spec.pool_size, f"invalid prediction rank: {key}")
        require_equal(parse_int(row["candidate_pool_size"], "prediction pool size"),
                      spec.pool_size, f"prediction pool size: {key}")
        require(not parse_bool(row["scientific_decision_permitted"],
                               "prediction decision permission"),
                f"partial-shard prediction improperly permits a decision: {key}")
        predictions[key] = row
    require_equal(len(targets), confirmation_count, "unique confirmation targets")
    require_equal(set(targets), set(evidence.confirmation_targets),
                  "frozen confirmation prediction target set")
    for arm in spec.arms:
        for trial in targets:
            for condition in ("correct", "matched_wrong"):
                require((arm, trial, condition) in predictions,
                        f"missing confirmation prediction: {(arm, trial, condition)}")

    common_pools: dict[str, tuple[tuple[int, str, str, bool, bool], ...]] = {}
    common_queries: dict[tuple[str, str], tuple[str, str, str]] = {}
    for arm in spec.arms:
        score_rows = read_csv(
            root / "runs" / arm / "confirmation_candidate_scores.csv",
            CONFIRMATION_SCORE_FIELDS,
        )
        require_equal(len(score_rows), 2 * confirmation_count * spec.pool_size,
                      f"{arm} confirmation-score row count")
        pools: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in score_rows:
            require_equal(row["arm_id"], arm, f"{arm} score arm")
            require_equal(parse_int(row["training_seed"], "score seed"), seed,
                          f"{arm} score seed")
            require_equal(parse_int(row["outer_fold"], "score fold"), fold,
                          f"{arm} score fold")
            trial = row["trial_id"]
            condition = row["signal_condition"]
            require(trial in targets and condition in {"correct", "matched_wrong"},
                    f"{arm} score has invalid target/condition: {(trial, condition)}")
            stable = targets[trial]
            require_equal(row["reading_task"], stable["reading_task"],
                          f"{arm} score task")
            require_equal(row["subject_id"], stable["subject_id"],
                          f"{arm} score subject")
            require_equal(row["normalized_text_sha256"], stable["normalized_text_sha256"],
                          f"{arm} score target text")
            require_equal(row["signal_subject_id"], stable["subject_id"],
                          f"{arm} signal subject provenance")
            require(is_sha256(row["signal_normalized_text_sha256"]),
                    f"{arm} invalid signal text identity")
            query = (
                row["signal_trial_id"], row["signal_subject_id"],
                row["signal_normalized_text_sha256"],
            )
            query_key = (trial, condition)
            if query_key in common_queries:
                require_equal(query, common_queries[query_key],
                              f"query provenance drift across candidate/arm: {query_key}")
            else:
                common_queries[query_key] = query
            if condition == "correct":
                require_equal(row["signal_trial_id"], trial,
                              f"{arm} correct signal trial provenance")
                require_equal(row["signal_normalized_text_sha256"],
                              stable["normalized_text_sha256"],
                              f"{arm} correct signal text provenance")
            else:
                require(row["signal_trial_id"] != trial,
                        f"{arm} matched-wrong signal reuses target trial")
                require(row["signal_normalized_text_sha256"]
                        != stable["normalized_text_sha256"],
                        f"{arm} matched-wrong signal reuses target text")
            donor = evidence.confirmation_donors[trial]
            require_equal(stable["reading_task"], donor["reading_task"],
                          f"{arm} frozen confirmation target task")
            require_equal(stable["subject_id"], donor["subject_id"],
                          f"{arm} frozen confirmation target subject")
            require_equal(stable["normalized_text_sha256"],
                          donor["target_normalized_text_sha256"],
                          f"{arm} frozen confirmation target text")
            if condition == "correct":
                expected_signal = (
                    trial, donor["subject_id"],
                    donor["target_normalized_text_sha256"],
                )
            else:
                expected_signal = (
                    donor["donor_trial_id"], donor["subject_id"],
                    donor["donor_normalized_text_sha256"],
                )
            require_equal(query, expected_signal,
                          f"{arm} frozen {condition} signal provenance: {trial}")
            require(is_sha256(row["candidate_normalized_text_sha256"]),
                    f"{arm} invalid candidate text hash")
            require(bool(row["candidate_text_target_id"]),
                    f"{arm} empty candidate text target id")
            parse_float(row["score"], f"{arm} candidate score")
            pools[(trial, condition)].append(row)
        require_equal(
            set(pools),
            {(trial, condition) for trial in targets
             for condition in ("correct", "matched_wrong")},
            f"{arm} confirmation pools",
        )
        for (trial, condition), pool in pools.items():
            require_equal(len(pool), spec.pool_size, f"{arm} pool size")
            ordered = sorted(pool, key=lambda row: parse_int(row["candidate_rank"],
                                                              "candidate rank"))
            require_equal(
                [parse_int(row["candidate_rank"], "candidate rank") for row in ordered],
                list(range(spec.pool_size)), f"{arm} candidate rank inventory",
            )
            candidate_texts = [row["candidate_normalized_text_sha256"] for row in ordered]
            require_equal(len(set(candidate_texts)), spec.pool_size,
                          f"{arm} duplicate candidate text")
            positives = [row for row in ordered if parse_bool(row["is_positive"],
                                                               "is_positive")]
            donors = [row for row in ordered if parse_bool(
                row["is_designated_donor_text"], "is_designated_donor_text")]
            require_equal(len(positives), 1, f"{arm} positive candidate count")
            require_equal(len(donors), 1, f"{arm} designated donor count")
            require_equal(positives[0]["candidate_normalized_text_sha256"],
                          targets[trial]["normalized_text_sha256"],
                          f"{arm} positive candidate identity")
            donor_text = common_queries[(trial, "matched_wrong")][2]
            require_equal(donors[0]["candidate_normalized_text_sha256"], donor_text,
                          f"{arm} designated donor identity")
            require(not parse_bool(donors[0]["is_positive"], "donor positive flag"),
                    f"{arm} designated donor is positive")
            frozen_pool = evidence.confirmation_pools[trial]
            require_equal(len(frozen_pool), spec.pool_size,
                          f"frozen pool size: {trial}")
            for actual, frozen in zip(ordered, frozen_pool):
                require_equal(parse_int(actual["candidate_rank"], "candidate rank"),
                              parse_int(frozen["candidate_rank"],
                                        "frozen candidate rank"),
                              f"{arm} frozen candidate rank: {(trial, condition)}")
                for field in (
                    "candidate_normalized_text_sha256", "candidate_text_target_id",
                ):
                    require_equal(actual[field], frozen[field],
                                  f"{arm} frozen {field}: {(trial, condition)}")
                require_equal(parse_bool(actual["is_positive"], "is_positive"),
                              parse_bool(frozen["is_positive"], "frozen positive"),
                              f"{arm} frozen positive flag: {(trial, condition)}")
                require_equal(
                    parse_bool(actual["is_designated_donor_text"], "is donor"),
                    parse_bool(frozen["is_designated_donor_text"],
                               "frozen donor flag"),
                    f"{arm} frozen donor flag: {(trial, condition)}",
                )
            pool_binding = tuple((
                parse_int(row["candidate_rank"], "candidate rank"),
                row["candidate_normalized_text_sha256"],
                row["candidate_text_target_id"],
                parse_bool(row["is_positive"], "is_positive"),
                parse_bool(row["is_designated_donor_text"], "is donor"),
            ) for row in ordered)
            if trial in common_pools:
                require_equal(pool_binding, common_pools[trial],
                              f"candidate pool drift across arm/condition: {trial}")
            else:
                common_pools[trial] = pool_binding
            predicted = predictions[(arm, trial, condition)]
            require_equal(parse_int(predicted["positive_rank"], "prediction rank"),
                          _score_positive_rank(ordered),
                          f"{arm} prediction rank recomputation")
    require_equal(len(predictions), expected_predictions, "complete predictions")
    require_equal(
        set(common_queries),
        {(trial, condition) for trial in targets
         for condition in ("correct", "matched_wrong")},
        "complete confirmation query provenance",
    )


def _verify_summary(
    path: Path, arm: str, manifest: Mapping[str, Any], best_epoch: int,
    best_macro: float, initial_state: str, final_state: str,
    final_optimizer_state: str, expected_hashes: Mapping[str, str],
    confirmation_count: int, checkpoint_count: int, spec: VerificationSpec,
) -> dict[str, Any]:
    summary = read_json(path)
    require_equal(set(summary), set(SUMMARY_FIELDS), f"{arm} summary fields")
    require_equal(summary.get("schema_version"), 1, f"{arm} summary schema")
    require_equal(summary.get("status"), "complete", f"{arm} summary status")
    require_equal(summary.get("run_mode"), "full_scientific",
                  f"{arm} summary mode")
    for field in (
        "shard_id", "schedule_unit_index", "outer_fold", "training_seed",
        "binding_sha256", "schedule_unit_sha256",
    ):
        require_equal(summary.get(field), manifest.get(field),
                      f"{arm} summary {field}")
    require_equal(summary.get("arm_id"), arm, f"{arm} summary arm")
    require_equal(summary.get("epochs"), spec.epochs, f"{arm} summary epochs")
    require_equal(summary.get("batches_per_epoch"), spec.batches_per_epoch,
                  f"{arm} summary batches per epoch")
    require_equal(summary.get("optimizer_steps"), spec.optimizer_steps,
                  f"{arm} summary optimizer steps")
    require_equal(summary.get("trainable_parameters"), 196608,
                  f"{arm} trainable parameters")
    require_equal(summary.get("best_epoch"), best_epoch, f"{arm} best epoch")
    _close(float(summary.get("best_checkpoint_macro_mrr")), best_macro,
           f"{arm} best checkpoint MRR")
    require_equal(summary.get("checkpoint_trial_count"), checkpoint_count,
                  f"{arm} checkpoint trial count")
    require_equal(summary.get("checkpoint_rank_rows"),
                  (spec.epochs + 1) * checkpoint_count,
                  f"{arm} checkpoint rank rows")
    require_equal(summary.get("confirmation_trial_count"), confirmation_count,
                  f"{arm} confirmation trial count")
    require_equal(summary.get("confirmation_prediction_rows"),
                  2 * confirmation_count, f"{arm} prediction rows")
    require_equal(summary.get("confirmation_candidate_score_rows"),
                  2 * confirmation_count * spec.pool_size,
                  f"{arm} score rows")
    require_equal(summary.get("initial_state_sha256"), initial_state,
                  f"{arm} initial state")
    require_equal(summary.get("final_state_sha256"), final_state,
                  f"{arm} final state")
    require_equal(summary.get("final_optimizer_state_sha256"), final_optimizer_state,
                  f"{arm} final optimizer state")
    require(isinstance(summary.get("resumed"), bool), f"{arm} invalid resumed flag")
    require_equal(summary.get("full_training_authorized"), True,
                  f"{arm} full authorization")
    require_equal(summary.get("scientific_decision_permitted"), False,
                  f"{arm} decision permission")
    require_equal(summary.get("official_validation_used_for_confirmation"), False,
                  f"{arm} official-validation flag")
    require_equal(summary.get("held_out_test_accessed"), False,
                  f"{arm} held-out-test flag")
    nested = summary.get("artifact_sha256")
    require(isinstance(nested, dict), f"{arm} summary artifact hashes are invalid")
    expected_names = set(PER_ARM_FILES) - {"run_summary.json"}
    require_equal(set(nested), expected_names, f"{arm} nested artifact inventory")
    for name in expected_names:
        require_equal(nested[name], expected_hashes[f"runs/{arm}/{name}"],
                      f"{arm} nested artifact SHA256 for {name}")
    return summary


def verify(
    artifact_root: Path,
    expected_manifest_sha256: str,
    expected_contract_sha256: str,
    expected_launch_authorization_sha256: str,
    *,
    protocol_root: Path,
    schedule_root: Path,
    spec: VerificationSpec = PRODUCTION_SPEC,
    evidence_hashes: EvidenceHashes = PRODUCTION_EVIDENCE_HASHES,
    preserved_source_id: str | None = None,
) -> dict[str, Any]:
    for value, label in (
        (expected_manifest_sha256, "expected manifest SHA256"),
        (expected_contract_sha256, "expected full contract SHA256"),
        (expected_launch_authorization_sha256, "expected launch authorization SHA256"),
    ):
        require(is_sha256(value), f"invalid {label}")
    root = artifact_root.resolve(strict=True)
    actual_hashes = _exact_inventory(root)
    manifest_path = root / MANIFEST_NAME
    require_equal(sha256(manifest_path), expected_manifest_sha256,
                  "full-shard manifest SHA256")
    manifest = read_json(manifest_path)
    expected_manifest_fields = {
        "schema_version", "status", "run_mode", "project_commit",
        "runner_source_sha256", "adapter_source_sha256",
        "task_treatment_pilots_source_sha256",
        "full_execution_contract_sha256", "launch_authorization_sha256",
        "binding_sha256", "input_bindings", "shard_id", "schedule_unit_index",
        "outer_fold", "training_seed", "schedule_unit_sha256",
        "runtime_fingerprint", "runtime_fingerprint_sha256",
        "observed_cuda_device_count", "git_worktree_clean",
        "git_submodules_clean", "arms", "epochs",
        "batches_per_epoch", "batch_size", "optimizer_steps_per_arm",
        "optimizer_steps_per_shard", "full_training_authorized",
        "scientific_decision_permitted_after_complete_matrix_only",
        "official_validation_used_for_confirmation", "held_out_test_accessed",
        "artifact_sha256",
    }
    require_equal(set(manifest), expected_manifest_fields, "manifest fields")
    require_equal(manifest["schema_version"], 1, "manifest schema")
    require_equal(manifest["status"], "complete", "manifest status")
    require_equal(manifest["run_mode"], "full_scientific_shard", "manifest mode")
    require(is_commit(manifest["project_commit"]), "invalid project commit")
    for field in (
        "runner_source_sha256", "adapter_source_sha256",
        "task_treatment_pilots_source_sha256", "binding_sha256",
        "schedule_unit_sha256",
    ):
        require(is_sha256(manifest[field]), f"invalid manifest {field}")
    require_equal(manifest["full_execution_contract_sha256"],
                  expected_contract_sha256, "manifest full contract")
    require_equal(manifest["launch_authorization_sha256"],
                  expected_launch_authorization_sha256,
                  "manifest launch authorization")
    require_equal(manifest["input_bindings"], FULL_INPUT_BINDINGS,
                  "manifest immutable inputs")
    fold = parse_int(manifest["outer_fold"], "manifest outer_fold")
    seed = parse_int(manifest["training_seed"], "manifest training_seed")
    require(fold in spec.folds, "manifest outer fold is outside frozen domain")
    require(seed in spec.seeds, "manifest seed is outside frozen domain")
    expected_shard_id = f"p4b-f{fold}-s{seed}"
    expected_unit_index = spec.folds.index(fold) * len(spec.seeds) + spec.seeds.index(seed)
    require_equal(manifest["shard_id"], expected_shard_id, "manifest shard id")
    require_equal(manifest["schedule_unit_index"], expected_unit_index,
                  "manifest schedule-unit index")
    require_equal(manifest["arms"], list(spec.arms), "manifest arms")
    require_equal(manifest["epochs"], spec.epochs, "manifest epochs")
    require_equal(manifest["batches_per_epoch"], spec.batches_per_epoch,
                  "manifest batches per epoch")
    require_equal(manifest["batch_size"], 64, "manifest batch size")
    require_equal(manifest["optimizer_steps_per_arm"], spec.optimizer_steps,
                  "manifest optimizer steps per arm")
    require_equal(manifest["optimizer_steps_per_shard"],
                  len(spec.arms) * spec.optimizer_steps,
                  "manifest optimizer steps per shard")
    require_equal(manifest["full_training_authorized"], True,
                  "manifest full authorization")
    require_equal(manifest["scientific_decision_permitted_after_complete_matrix_only"],
                  True, "manifest complete-matrix decision condition")
    require_equal(manifest["official_validation_used_for_confirmation"], False,
                  "manifest official-validation flag")
    require_equal(manifest["held_out_test_accessed"], False,
                  "manifest held-out-test flag")
    require_equal(manifest["runtime_fingerprint"], EXACT_RUNTIME_ENVIRONMENT,
                  "manifest exact runtime fingerprint")
    runtime_sha256 = runtime_fingerprint_sha256(EXACT_RUNTIME_ENVIRONMENT)
    require_equal(manifest["runtime_fingerprint_sha256"], runtime_sha256,
                  "manifest runtime-fingerprint SHA256")
    observed_cuda_count = manifest["observed_cuda_device_count"]
    require(isinstance(observed_cuda_count, int)
            and not isinstance(observed_cuda_count, bool)
            and observed_cuda_count >= EXACT_RUNTIME_ENVIRONMENT[
                "minimum_cuda_device_count"
            ], "manifest observed CUDA device count is below frozen minimum")
    require_equal(manifest["git_worktree_clean"], True,
                  "manifest Git worktree boundary")
    require_equal(manifest["git_submodules_clean"], True,
                  "manifest Git submodule boundary")
    evidence = _load_preserved_evidence(
        protocol_root, schedule_root, fold, seed, spec, evidence_hashes
    )
    require_equal(manifest["schedule_unit_sha256"],
                  evidence.schedule_unit_sha256,
                  "manifest preserved schedule-unit SHA256")
    declared = manifest["artifact_sha256"]
    require(isinstance(declared, dict), "manifest artifact hashes must be an object")
    require_equal({_safe_relative(key) for key in declared}, set(NON_MANIFEST_FILES),
                  "manifest artifact inventory")
    require(all(is_sha256(value) for value in declared.values()),
            "manifest contains invalid artifact SHA256")
    require_equal(dict(declared), actual_hashes, "manifest artifact hashes")

    metadata = read_json(root / METADATA_NAME)
    require_equal(set(metadata), set(METADATA_FIELDS), "metadata fields")
    _check_common_binding(
        metadata, label="metadata", manifest=manifest,
        expected_contract_sha256=expected_contract_sha256,
        expected_launch_sha256=expected_launch_authorization_sha256,
    )
    require_equal(metadata.get("schema_version"), 1, "metadata schema")
    require_equal(metadata.get("status"), "complete", "metadata status")
    require_equal(metadata.get("run_mode"), "full_scientific_shard", "metadata mode")
    require_equal(metadata.get("arms"), list(spec.arms), "metadata arms")
    require_equal(metadata.get("epochs"), spec.epochs, "metadata epochs")
    require_equal(metadata.get("batches_per_epoch"), spec.batches_per_epoch,
                  "metadata batches per epoch")
    require_equal(metadata.get("batch_size"), 64, "metadata batch size")
    require_equal(metadata.get("completed_arms"), list(spec.arms), "metadata arms")
    require_equal(metadata.get("optimizer_steps_completed"),
                  len(spec.arms) * spec.optimizer_steps, "metadata optimizer steps")
    require_equal(metadata.get("full_training_authorized"), True,
                  "metadata full authorization")
    require_equal(metadata.get("scientific_decision_permitted_after_complete_matrix_only"),
                  True, "metadata complete-matrix decision condition")
    require_equal(metadata.get("official_validation_used_for_confirmation"), False,
                  "metadata official-validation flag")
    require_equal(metadata.get("held_out_test_accessed"), False,
                  "metadata held-out-test flag")
    require_equal(metadata.get("runtime_fingerprint"), EXACT_RUNTIME_ENVIRONMENT,
                  "metadata exact runtime fingerprint")
    require_equal(metadata.get("runtime_fingerprint_sha256"), runtime_sha256,
                  "metadata runtime-fingerprint SHA256")
    require_equal(metadata.get("observed_cuda_device_count"),
                  observed_cuda_count, "metadata observed CUDA device count")
    require_equal(metadata.get("git_worktree_clean"), True,
                  "metadata Git worktree boundary")
    require_equal(metadata.get("git_submodules_clean"), True,
                  "metadata Git submodule boundary")
    require_equal(metadata.get("device"), EXACT_RUNTIME_ENVIRONMENT["device"],
                  "metadata runtime device")
    require_equal(metadata.get("python"), EXACT_RUNTIME_ENVIRONMENT["python"],
                  "metadata Python version")
    require_equal(metadata.get("torch"), EXACT_RUNTIME_ENVIRONMENT["torch"],
                  "metadata Torch version")
    require_equal(manifest["binding_sha256"],
                  recompute_binding_sha256(
                      manifest, EXACT_RUNTIME_ENVIRONMENT, spec
                  ),
                  "recomputed shard binding SHA256")

    _verify_run_manifest(root / "run_manifest.csv", fold, seed, spec)
    _verify_confirmation(root, fold, seed, evidence, spec)

    initial_states: set[str] = set()
    best_epochs: dict[str, int] = {}
    common_schedule_trace: tuple[tuple[int, int, str, str, str], ...] | None = None
    for arm in spec.arms:
        arm_root = root / "runs" / arm
        initial, final, final_optimizer, schedule_trace = _verify_step_trace(
            arm_root / "step_trace.csv", arm, fold, seed,
            manifest["schedule_unit_sha256"], evidence.batch_bindings, spec,
        )
        if common_schedule_trace is None:
            common_schedule_trace = schedule_trace
        else:
            require_equal(schedule_trace, common_schedule_trace,
                          f"{arm} schedule/catalog trace differs across arms")
        initial_states.add(initial)
        metrics = _verify_checkpoint_ranks(
            arm_root / "checkpoint_positive_ranks.csv", arm, fold, seed,
            spec.checkpoint_count(fold), evidence.checkpoint_targets, spec,
        )
        best_epoch, best_macro = _verify_history(
            arm_root / "training_history.csv", arm, fold, seed, metrics, spec,
        )
        best_epochs[arm] = best_epoch
        _verify_summary(
            arm_root / "run_summary.json", arm, manifest, best_epoch, best_macro,
            initial, final, final_optimizer, actual_hashes,
            spec.confirmation_count(fold), spec.checkpoint_count(fold), spec,
        )
    require_equal(len(initial_states), 1, "paired initial model state across arms")

    return {
        "status": "pass",
        "schema_version": 1,
        "preserved_source_id": preserved_source_id,
        "full_shard_manifest_sha256": expected_manifest_sha256,
        "full_execution_contract_sha256": expected_contract_sha256,
        "launch_authorization_sha256": expected_launch_authorization_sha256,
        "outer_fold": fold,
        "training_seed": seed,
        "schedule_unit_sha256": manifest["schedule_unit_sha256"],
        "arms": list(spec.arms),
        "epochs_per_arm": spec.epochs,
        "optimizer_steps_per_arm": spec.optimizer_steps,
        "total_optimizer_steps": len(spec.arms) * spec.optimizer_steps,
        "checkpoint_rows_per_arm": (
            (spec.epochs + 1) * spec.checkpoint_count(fold)
        ),
        "confirmation_prediction_rows": (
            len(spec.arms) * 2 * spec.confirmation_count(fold)
        ),
        "confirmation_candidate_score_rows": (
            len(spec.arms) * 2 * spec.confirmation_count(fold) * spec.pool_size
        ),
        "best_epochs": best_epochs,
        "paired_initial_state": True,
        "epoch_zero_eligible": True,
        "earliest_tie_selection_verified": True,
        "append_only_strict_incumbent_history_verified": True,
        "partial_scientific_decision_permitted": False,
        "correct_and_matched_wrong_provenance_verified": True,
        "preserved_protocol_evidence_verified": True,
        "preserved_schedule_evidence_verified": True,
        "recomputed_binding_sha256": manifest["binding_sha256"],
        "preserved_evidence_sha256": {
            "candidate_pools.csv": evidence_hashes.candidate_pools_sha256,
            "confirmation_donors.csv": evidence_hashes.confirmation_donors_sha256,
            "schedule_indices.u32le": evidence_hashes.schedule_indices_sha256,
            "schedule_units.csv": evidence_hashes.schedule_units_sha256,
            "trial_catalog.csv": evidence_hashes.trial_catalog_sha256,
        },
        "runtime_fingerprint_verified": True,
        "runtime_fingerprint_sha256": runtime_sha256,
        "git_execution_boundary_verified": True,
        "checkpoint_deserialized": False,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
        "verified_artifact_sha256": actual_hashes,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
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
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-launch-authorization-sha256", required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--preserved-source-id")
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    report = verify(
        args.artifact_root,
        args.expected_manifest_sha256,
        args.expected_contract_sha256,
        args.expected_launch_authorization_sha256,
        protocol_root=args.protocol_root,
        schedule_root=args.schedule_root,
        preserved_source_id=args.preserved_source_id,
    )
    if args.output_report is not None:
        write_report(args.output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TASK-SEGMENTED FULL SCIENTIFIC SHARD ARTIFACT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
