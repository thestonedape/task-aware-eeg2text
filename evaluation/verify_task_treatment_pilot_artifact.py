"""Verify and ingest the preserved frozen task-treatment pilot artifact.

This verifier is deliberately standard-library only. It never imports the
training runner, NumPy, Torch, or any checkpoint payload. It first binds the
immutable pilot manifest, then re-hashes every declared top-level artifact,
all 12 run summaries, and their 48 nested artifacts. After integrity checks,
it recomputes the frozen continuation booleans from the preserved CSV/JSON
tables and emits exact read-only result summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONFIGURATIONS = (
    "generic_pooled",
    "separate_per_task",
    "task_token",
    "masked_shared_private",
)
SEEDS = (20260717, 20260718, 20260719)
CONDITIONS = (
    ("correct_val", "correct"),
    ("zero_val", "correct"),
    ("gaussian_val", "correct"),
    ("matched_wrong_val", "correct"),
    ("correct_val", "masked"),
    ("correct_val", "shuffled"),
)
CLUSTERS = (
    "subject_id",
    "normalized_text_sha256",
    "two_way_subject_by_text",
)
CONTRASTS = (
    ("model_comparison", "masked_shared_private", "generic_pooled"),
    ("model_comparison", "masked_shared_private", "task_token"),
    ("signal_specificity", "masked_shared_private", "matched_wrong_eeg"),
    ("task_control", "masked_shared_private", "masked_task"),
    ("task_control", "masked_shared_private", "shuffled_task"),
)
TOP_LEVEL_ARTIFACTS = (
    "all_seed_metrics.csv",
    "continuation_decision.json",
    "frozen_candidate_pools.csv",
    "frozen_protocol/run_task_treatment_pilots.py",
    "frozen_protocol/task_treatment_pilot_contract.json",
    "frozen_protocol/task_treatment_pilot_execution_protocol.json",
    "frozen_protocol/task_treatment_pilots.py",
    "paired_model_comparisons.csv",
    "seed_averaged_predictions.csv",
    "seedwise_contrasts.csv",
    "validation_partition.csv",
)
RUN_ARTIFACTS = (
    "best_checkpoint.pt",
    "training_history.csv",
    "predictions.csv",
    "metrics.csv",
)
POOL_FIELDS = (
    "target_trial_id", "candidate_rank", "candidate_id", "is_positive",
    "dataset_version", "reading_task", "target_evaluation_partition",
    "candidate_catalog_scope", "target_length", "candidate_length",
    "absolute_length_difference", "selection_rule",
)
PARTITION_FIELDS = (
    "trial_id", "cohort", "dataset_version", "reading_task", "subject_id",
    "normalized_text_sha256", "evaluation_partition",
)
PREDICTION_FIELDS = (
    "config_id", "seed", "signal_condition", "task_condition", "trial_id",
    "cohort", "evaluation_partition", "dataset_version", "reading_task",
    "subject_id", "normalized_text_sha256", "positive_rank",
    "reciprocal_rank", "top1", "top5",
    "positive_minus_best_negative_margin",
)
METRIC_FIELDS = (
    "config_id", "seed", "signal_condition", "task_condition", "scope",
    "reading_task", "rows", "top1", "top5", "mrr",
    "mean_positive_rank", "mean_positive_minus_best_negative_margin",
)
HISTORY_FIELDS = (
    "config_id", "seed", "epoch", "train_loss", "headline_macro_mrr",
)
SEED_AVERAGE_FIELDS = (
    "config_id", "signal_condition", "task_condition", "trial_id", "cohort",
    "evaluation_partition", "reading_task", "subject_id",
    "normalized_text_sha256", "mean_reciprocal_rank",
)
COMPARISON_FIELDS = (
    "contrast_type", "model", "reference", "cluster", "rows",
    "mean_mrr_delta", "ci95_lower", "ci95_upper",
    "decision_lower_quantile", "decision_lower", "bootstrap_replicates",
)
SEEDWISE_FIELDS = (
    "contrast_type", "model", "reference", "seed", "macro_mrr_delta",
    "direction_positive",
)
STORED_PARAMETERS = {
    "generic_pooled": 196608,
    "separate_per_task": 196608,
    "task_token": 196896,
    "masked_shared_private": 196608,
}
ACTIVE_PARAMETERS = {
    "generic_pooled": 196608,
    "separate_per_task": 65536,
    "task_token": 196896,
    "masked_shared_private": 131072,
}
EXPECTED_REQUIREMENTS = {
    "masked_model_signal_gap_positive": True,
    "model_all_cluster_familywise_lower_bounds_positive": False,
    "model_delta_positive_in_all_training_seeds": False,
    "mrr_above_both_planned_baselines": False,
    "signal_all_cluster_lower_bounds_positive": False,
    "signal_gap_not_below_either_planned_baseline": True,
    "task_controls_all_cluster_familywise_lower_bounds_positive": False,
}
POOL_SELECTION_RULE = (
    "same_dataset_task_training_or_same_partition_then_length_then_seeded_hash"
)
PRODUCTION_TOP_LEVEL_SHA256 = {
    "all_seed_metrics.csv": "5daa35db141fc82c77cd8018b7c3cf02f929461395b4dbd3a6f467e33ea65158",
    "continuation_decision.json": "6a814ce0ad14fd297a3b3403dfa03a25c9e9794b75fb853762915a33d4b3ccfd",
    "frozen_candidate_pools.csv": "171726752491413c29c8f63c3847bcb2e9afa5e15b695f7805cc0a161947669c",
    "frozen_protocol/run_task_treatment_pilots.py": "f9e833c7e34be81a373e687bb0a3726eaf2c4c7dce57d2807a06cf080f92c613",
    "frozen_protocol/task_treatment_pilot_contract.json": "a7370a61921803fbeaaab874dcfef77d38d6dceb35b05e249b6f5548e8a2921e",
    "frozen_protocol/task_treatment_pilot_execution_protocol.json": "35519ef55af615e593000bc353ad1e6d7043238352ad7f83b8f23c3aa1ddd9b7",
    "frozen_protocol/task_treatment_pilots.py": "67089942c60a89d8f371e02c3f2786c31ff0be5a423690916144573e17d0650f",
    "paired_model_comparisons.csv": "f058d84c9f0308553118af813c7fec6858e2b02bfb22186415e2fecef5fbdab5",
    "seed_averaged_predictions.csv": "74100c0b964a4490cea66ee03be957ac5d71526e7fc0c84ce9fc222453f483f3",
    "seedwise_contrasts.csv": "f87c9e8742faecc5caa3d08cfbfd456c88ee11a363acec35cc3bf8998222d027",
    "validation_partition.csv": "885c37944c76fc993125afaa33de7e044ac6e806af39f442b81273754801baa0",
}
PRODUCTION_RUN_SUMMARY_SHA256 = {
    "runs/generic_pooled/20260717/run_summary.json": "4100f7ad027c9c1b4f4fea9c38e56cfbb67ee770a412f7bd3ed867df46a4efe0",
    "runs/generic_pooled/20260718/run_summary.json": "9b04f51970bbc7470d6d720bb7ae24290f7514e03abe1736dd66387aa1d07f9f",
    "runs/generic_pooled/20260719/run_summary.json": "9b1d808a2d3078cc467062d2bbaccbd9f28dcc8358384d1bb948bccc248a3042",
    "runs/separate_per_task/20260717/run_summary.json": "8ae739fb5566305485428e92647cf26150d5ea87f935e5063276e9f06c98bcd3",
    "runs/separate_per_task/20260718/run_summary.json": "26d3058274960a206ed284ad453794c4804f22eee3259c93ecb9e111a8ab1d0c",
    "runs/separate_per_task/20260719/run_summary.json": "4ff2fc697df5101c73e376cce2bc1023fbfe1bbd7a8b849e93742f0a33358a11",
    "runs/task_token/20260717/run_summary.json": "7e2c35d4dad41ed12ea30989195b5373821d00ee46178f91295b91de1edb8ef8",
    "runs/task_token/20260718/run_summary.json": "c1996950bb87092e42a0b205766366723a49cebda8ed9d943ecf0bc13797a9eb",
    "runs/task_token/20260719/run_summary.json": "a56b4169b8c035f2ac00e7f4a731e69b581f76c42423e9ca1ef9017a7fa4d9da",
    "runs/masked_shared_private/20260717/run_summary.json": "e3f33acac3a1685d0859959396a91cbee8014f9ba58da99d449f86e89f37dd1f",
    "runs/masked_shared_private/20260718/run_summary.json": "45251a9ad185773f7ea2a79048c48c1cd53abf2d11a183aeeef2b578e8e56624",
    "runs/masked_shared_private/20260719/run_summary.json": "040945701b80e5551990658985784ec3e77f20747db2cbc9c412c5063c5411a8",
}


@dataclass(frozen=True)
class VerificationExpectations:
    pilot_manifest_sha256: str
    run_metadata_sha256: str
    top_level_sha256: dict[str, str]
    run_summary_sha256: dict[str, str]
    project_commit: str
    pilot_contract_sha256: str
    execution_protocol_sha256: str
    input_manifest_sha256: str
    input_preserved_source_id: str
    runner_source_sha256: str
    adapter_source_sha256: str
    candidate_pool_sha256: str
    validation_partition_sha256: str
    output_preserved_source_prefix: str = (
        "kaggle-dataset-thestonedape-task-aware-eeg2text-"
        "task-treatment-pilots-version-"
    )
    validation_rows: int = 2200
    candidate_pool_size: int = 24
    candidate_rows: int = 52800
    seed_average_rows: int = 52800
    all_seed_metric_rows: int = 576
    comparison_rows: int = 15
    seedwise_rows: int = 15
    history_rows_per_run: int = 41
    predictions_per_run: int = 13200
    metrics_per_run: int = 48
    decision_rows: int = 529
    epochs: int = 40
    bootstrap_replicates: int = 5000
    partition_counts: tuple[tuple[str, int], ...] = (
        ("checkpoint", 574),
        ("decision", 529),
        ("auxiliary_sr", 406),
        ("zuco1_noncausal_diagnostic", 691),
    )


PRODUCTION_EXPECTATIONS = VerificationExpectations(
    pilot_manifest_sha256="a8cc82e59e078b3a05e21c1aeb046ed91070e59e754030b2d63cd4ff1091329a",
    run_metadata_sha256="e59917d7e2e9edbff67b1f51de61acc0bc01015d5fb36986abe9cd7f17f2b3ae",
    top_level_sha256=PRODUCTION_TOP_LEVEL_SHA256,
    run_summary_sha256=PRODUCTION_RUN_SUMMARY_SHA256,
    project_commit="8c6a0065fcc34f73d354c9d8ca0dbddae801b99b",
    pilot_contract_sha256="a7370a61921803fbeaaab874dcfef77d38d6dceb35b05e249b6f5548e8a2921e",
    execution_protocol_sha256="35519ef55af615e593000bc353ad1e6d7043238352ad7f83b8f23c3aa1ddd9b7",
    input_manifest_sha256="6c1fff8d2e89e33a72d03c39651e8ecce678c3b93cdb66747dd6dcc00538cddb",
    input_preserved_source_id="kaggle-dataset-thestonedape-task-aware-eegtotext-version-2",
    runner_source_sha256="f9e833c7e34be81a373e687bb0a3726eaf2c4c7dce57d2807a06cf080f92c613",
    adapter_source_sha256="67089942c60a89d8f371e02c3f2786c31ff0be5a423690916144573e17d0650f",
    candidate_pool_sha256="171726752491413c29c8f63c3847bcb2e9afa5e15b695f7805cc0a161947669c",
    validation_partition_sha256="885c37944c76fc993125afaa33de7e044ac6e806af39f442b81273754801baa0",
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        return fields, list(reader)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def require_bool(actual: Any, expected: bool, label: str) -> None:
    require(
        type(actual) is bool and actual is expected,
        f"{label}: expected boolean {expected!r}, got {actual!r}",
    )


def require_float(actual: Any, expected: float, label: str, tolerance: float = 1e-12) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {actual!r}") from error
    require(math.isfinite(value), f"{label} is not finite")
    require(math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance), f"{label}: {value} != {expected}")


def require_fields(actual: tuple[str, ...], expected: tuple[str, ...], label: str) -> None:
    require_equal(actual, expected, f"{label} columns")


def safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"absolute manifest path is forbidden: {relative}")
    require(".." not in candidate.parts, f"parent traversal is forbidden: {relative}")
    resolved = (root / candidate).resolve()
    require(resolved == root or root in resolved.parents, f"path escapes artifact root: {relative}")
    return resolved


def require_hash(path: Path, expected: str, label: str) -> str:
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256(path)
    require_equal(actual, expected, f"{label} SHA256")
    return actual


def task_macro(values: dict[str, float], partitions: dict[str, dict[str, str]], trial_ids: Iterable[str]) -> float:
    by_task: dict[str, list[float]] = {"NR": [], "TSR": []}
    for trial_id in trial_ids:
        task = partitions[trial_id]["reading_task"]
        require(task in by_task, f"task macro received non-headline task {task!r}")
        by_task[task].append(values[trial_id])
    require(all(by_task.values()), "task macro requires nonempty NR and TSR rows")
    return sum(sum(rows) / len(rows) for rows in by_task.values()) / 2.0


def numeric_comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append({
            "contrast_type": row["contrast_type"],
            "model": row["model"],
            "reference": row["reference"],
            "cluster": row["cluster"],
            "rows": int(row["rows"]),
            "mean_mrr_delta": float(row["mean_mrr_delta"]),
            "ci95_lower": float(row["ci95_lower"]),
            "ci95_upper": float(row["ci95_upper"]),
            "decision_lower_quantile": float(row["decision_lower_quantile"]),
            "decision_lower": float(row["decision_lower"]),
            "bootstrap_replicates": int(row["bootstrap_replicates"]),
        })
    return output


def numeric_seedwise_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [{
        "contrast_type": row["contrast_type"],
        "model": row["model"],
        "reference": row["reference"],
        "seed": int(row["seed"]),
        "macro_mrr_delta": float(row["macro_mrr_delta"]),
        "direction_positive": bool(int(row["direction_positive"])),
    } for row in rows]


def verify_metric_rows(
    metric_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    config: str,
    seed: int,
) -> None:
    """Recompute every per-run aggregate emitted by the frozen runner."""

    metric_lookup: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in metric_rows:
        require_equal(row["config_id"], config, f"{config}/{seed} metric config")
        require_equal(int(row["seed"]), seed, f"{config}/{seed} metric seed")
        key = (
            row["signal_condition"], row["task_condition"],
            row["scope"], row["reading_task"],
        )
        require(key not in metric_lookup, f"duplicate metric row {config}/{seed}/{key}")
        metric_lookup[key] = row

    expected: dict[tuple[str, str, str, str], dict[str, float | int]] = {}
    for signal_condition, task_condition in CONDITIONS:
        condition_rows = [
            row for row in prediction_rows
            if row["signal_condition"] == signal_condition
            and row["task_condition"] == task_condition
        ]
        group_specs = [
            (
                "headline_decision_task", task,
                [
                    row for row in condition_rows
                    if row["evaluation_partition"] == "decision"
                    and row["reading_task"] == task
                ],
            )
            for task in ("NR", "TSR")
        ]
        group_specs.extend([
            (
                "checkpoint_task_diagnostic", task,
                [
                    row for row in condition_rows
                    if row["evaluation_partition"] == "checkpoint"
                    and row["reading_task"] == task
                ],
            )
            for task in ("NR", "TSR")
        ])
        group_specs.append((
            "auxiliary_sr", "SR",
            [row for row in condition_rows if row["cohort"] == "auxiliary_sr"],
        ))
        group_specs.extend([
            (
                "zuco1_noncausal_diagnostic", task,
                [
                    row for row in condition_rows
                    if row["cohort"] == "zuco1_nr_tsr_noncausal"
                    and row["reading_task"] == task
                ],
            )
            for task in ("NR", "TSR")
        ])
        aggregates: dict[tuple[str, str], dict[str, float | int]] = {}
        for scope, task, members in group_specs:
            require(members, f"empty metric group {config}/{seed}/{scope}/{task}")
            count = len(members)
            aggregate: dict[str, float | int] = {
                "rows": count,
                "top1": sum(int(row["top1"]) for row in members) / count,
                "top5": sum(int(row["top5"]) for row in members) / count,
                "mrr": sum(float(row["reciprocal_rank"]) for row in members) / count,
                "mean_positive_rank": (
                    sum(int(row["positive_rank"]) for row in members) / count
                ),
                "mean_positive_minus_best_negative_margin": (
                    sum(float(row["positive_minus_best_negative_margin"]) for row in members)
                    / count
                ),
            }
            aggregates[(scope, task)] = aggregate
            expected[(signal_condition, task_condition, scope, task)] = aggregate
        headline_members = [
            aggregates[("headline_decision_task", task)]
            for task in ("NR", "TSR")
        ]
        expected[(
            signal_condition, task_condition, "headline_macro", "NR|TSR"
        )] = {
            "rows": sum(int(row["rows"]) for row in headline_members),
            **{
                field: sum(float(row[field]) for row in headline_members) / 2.0
                for field in (
                    "top1", "top5", "mrr", "mean_positive_rank",
                    "mean_positive_minus_best_negative_margin",
                )
            },
        }

    require_equal(set(metric_lookup), set(expected), f"{config}/{seed} metric key set")
    for key, expected_row in expected.items():
        actual = metric_lookup[key]
        require_equal(int(actual["rows"]), int(expected_row["rows"]), f"metric rows {key}")
        for field in (
            "top1", "top5", "mrr", "mean_positive_rank",
            "mean_positive_minus_best_negative_margin",
        ):
            require_float(actual[field], float(expected_row[field]), f"metric {key}.{field}")


def verify(
    artifact_root: Path,
    output_preserved_source_id: str,
    expectations: VerificationExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, Any]:
    root = artifact_root.resolve()
    require(root.is_dir(), f"artifact root is not a directory: {root}")
    verified: dict[str, str] = {}

    manifest_path = root / "pilot_manifest.json"
    verified["pilot_manifest.json"] = require_hash(
        manifest_path, expectations.pilot_manifest_sha256, "pilot manifest"
    )
    manifest = read_json(manifest_path)
    require_equal(manifest.get("status"), "pass", "manifest status")
    require_equal(manifest.get("schema_version"), 1, "manifest schema")
    require_equal(manifest.get("run_mode"), "full", "manifest run mode")
    require_equal(manifest.get("project_commit"), expectations.project_commit, "manifest project commit")
    require_equal(manifest.get("pilot_contract_sha256"), expectations.pilot_contract_sha256, "manifest contract")
    require_equal(manifest.get("execution_protocol_sha256"), expectations.execution_protocol_sha256, "manifest protocol")
    require_equal(manifest.get("input_manifest_sha256"), expectations.input_manifest_sha256, "manifest input")
    require_equal(manifest.get("preserved_source_id"), expectations.input_preserved_source_id, "manifest input source")
    require_equal(manifest.get("runner_source_sha256"), expectations.runner_source_sha256, "manifest runner")
    require_equal(manifest.get("adapter_source_sha256"), expectations.adapter_source_sha256, "manifest adapter")
    require_equal(manifest.get("candidate_pool_sha256"), expectations.candidate_pool_sha256, "manifest pool")
    require_equal(manifest.get("validation_partition_sha256"), expectations.validation_partition_sha256, "manifest partition")
    require_equal(tuple(manifest.get("configurations", [])), CONFIGURATIONS, "manifest configurations")
    require_equal(tuple(manifest.get("seeds", [])), SEEDS, "manifest seeds")
    require_equal(manifest.get("parameter_budget"), {
        **STORED_PARAMETERS,
        "reference": 196608,
        "maximum_relative_deviation": 0.00146484375,
    }, "manifest parameter budget")
    require_equal(manifest.get("active_parameters_per_example"), ACTIVE_PARAMETERS, "manifest active parameters")
    require_equal(manifest.get("auxiliary_factor_losses"), [], "manifest factor losses")
    require_bool(manifest.get("held_out_test_accessed"), False, "manifest test access")
    require_bool(manifest.get("continuation_selected"), False, "manifest continuation")

    declared_top = manifest.get("artifact_sha256")
    require(isinstance(declared_top, dict), "manifest top-level artifact hashes are missing")
    require_equal(set(declared_top), set(TOP_LEVEL_ARTIFACTS), "manifest top-level artifact set")
    require_equal(declared_top, expectations.top_level_sha256, "production top-level artifact map")
    for relative in TOP_LEVEL_ARTIFACTS:
        verified[relative] = require_hash(
            safe_path(root, relative), str(declared_top[relative]), relative
        )

    declared_summaries = manifest.get("run_summary_sha256")
    require(isinstance(declared_summaries, dict), "manifest run-summary hashes are missing")
    expected_summary_paths = {
        f"runs/{config}/{seed}/run_summary.json"
        for config in CONFIGURATIONS for seed in SEEDS
    }
    require_equal(set(declared_summaries), expected_summary_paths, "manifest run-summary set")
    require_equal(declared_summaries, expectations.run_summary_sha256, "production run-summary map")

    metadata_path = root / "run_metadata.json"
    verified["run_metadata.json"] = require_hash(
        metadata_path, expectations.run_metadata_sha256, "run metadata"
    )
    metadata = read_json(metadata_path)
    require_equal(metadata.get("status"), "pass", "metadata status")
    require_equal(metadata.get("project_commit"), expectations.project_commit, "metadata project commit")
    require_equal(metadata.get("execution_protocol_sha256"), expectations.execution_protocol_sha256, "metadata protocol")
    require_equal(metadata.get("input_manifest_sha256"), expectations.input_manifest_sha256, "metadata input")
    require_equal(metadata.get("candidate_pool_sha256"), expectations.candidate_pool_sha256, "metadata pool")
    require_equal(metadata.get("pilot_manifest_sha256"), expectations.pilot_manifest_sha256, "metadata manifest")
    require_bool(metadata.get("continuation_selected"), False, "metadata continuation")
    require_bool(metadata.get("held_out_test_accessed"), False, "metadata test access")

    contract = read_json(root / "frozen_protocol" / "task_treatment_pilot_contract.json")
    protocol = read_json(root / "frozen_protocol" / "task_treatment_pilot_execution_protocol.json")
    require_equal(contract.get("status"), "frozen_before_pilot_training", "frozen contract status")
    require_equal(contract.get("training", {}).get("seeds"), list(SEEDS), "frozen contract seeds")
    require_equal(contract.get("training", {}).get("auxiliary_factor_losses"), [], "frozen contract factor losses")
    require_bool(contract.get("evaluation", {}).get("held_out_test_accessed"), False, "frozen contract test access")
    require_equal(protocol.get("status"), "frozen_before_pilot_execution", "frozen protocol status")
    require_equal(protocol.get("parent_contract_sha256"), expectations.pilot_contract_sha256, "protocol parent contract")
    require_equal(protocol.get("input", {}).get("preserved_source_id"), expectations.input_preserved_source_id, "protocol input source")
    require_equal(protocol.get("input", {}).get("combined_manifest_sha256"), expectations.input_manifest_sha256, "protocol input manifest")
    require_equal(protocol.get("training", {}).get("stored_trainable_parameters"), STORED_PARAMETERS, "protocol stored parameters")
    require_equal(protocol.get("training", {}).get("active_parameters_per_example"), ACTIVE_PARAMETERS, "protocol active parameters")
    require_equal(protocol.get("training", {}).get("auxiliary_factor_losses"), [], "protocol factor losses")
    require_equal(protocol.get("evaluation", {}).get("checkpoint_decision_partition", {}).get("decision_rows"), expectations.decision_rows, "protocol decision rows")
    require_bool(protocol.get("continuation", {}).get("held_out_test_accessed"), False, "protocol test access")

    partition_fields, partition_rows = read_csv(root / "validation_partition.csv")
    require_fields(partition_fields, PARTITION_FIELDS, "validation partition")
    require_equal(len(partition_rows), expectations.validation_rows, "validation partition rows")
    partitions = {row["trial_id"]: row for row in partition_rows}
    require_equal(len(partitions), len(partition_rows), "unique validation trials")
    require_equal(
        Counter(row["evaluation_partition"] for row in partition_rows),
        Counter(dict(expectations.partition_counts)),
        "validation partition counts",
    )
    require(not any("test" in row["evaluation_partition"].lower() for row in partition_rows), "test partition found")
    expected_partition_by_cohort = {
        "primary_zuco2_nr_tsr": {"checkpoint", "decision"},
        "auxiliary_sr": {"auxiliary_sr"},
        "zuco1_nr_tsr_noncausal": {"zuco1_noncausal_diagnostic"},
    }
    for row in partition_rows:
        cohort = row["cohort"]
        require(cohort in expected_partition_by_cohort, f"unknown validation cohort {cohort!r}")
        require(
            row["evaluation_partition"] in expected_partition_by_cohort[cohort],
            f"invalid cohort/partition assignment for {row['trial_id']}",
        )
        if cohort == "primary_zuco2_nr_tsr":
            require_equal(row["dataset_version"], "ZuCo2", "primary dataset version")
            require(row["reading_task"] in {"NR", "TSR"}, "primary row is not NR/TSR")
        elif cohort == "auxiliary_sr":
            require_equal(row["dataset_version"], "ZuCo1", "auxiliary SR dataset version")
            require_equal(row["reading_task"], "SR", "auxiliary SR task")
        else:
            require_equal(row["dataset_version"], "ZuCo1", "noncausal dataset version")
            require(row["reading_task"] in {"NR", "TSR"}, "noncausal row is not NR/TSR")
        require(bool(row["subject_id"]), f"missing subject for {row['trial_id']}")
        require(len(row["normalized_text_sha256"]) == 64, f"invalid text identity for {row['trial_id']}")
    checkpoint_texts = {
        (row["reading_task"], row["normalized_text_sha256"])
        for row in partition_rows if row["evaluation_partition"] == "checkpoint"
    }
    decision_texts = {
        (row["reading_task"], row["normalized_text_sha256"])
        for row in partition_rows if row["evaluation_partition"] == "decision"
    }
    require(checkpoint_texts.isdisjoint(decision_texts), "checkpoint/decision text leakage")

    pool_fields, pool_rows = read_csv(root / "frozen_candidate_pools.csv")
    require_fields(pool_fields, POOL_FIELDS, "candidate pool")
    require_equal(len(pool_rows), expectations.candidate_rows, "candidate rows")
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pool_rows:
        require(row["target_trial_id"] in partitions, "candidate pool references unknown target")
        pools[row["target_trial_id"]].append(row)
    require_equal(set(pools), set(partitions), "candidate pool target set")
    for trial_id, rows in pools.items():
        require_equal(len(rows), expectations.candidate_pool_size, f"pool size {trial_id}")
        target = partitions[trial_id]
        ranks = [int(row["candidate_rank"]) for row in rows]
        require_equal(set(ranks), set(range(expectations.candidate_pool_size)), f"candidate ranks {trial_id}")
        require_equal(len({row["candidate_id"] for row in rows}), len(rows), f"candidate IDs {trial_id}")
        flags = [int(row["is_positive"]) for row in rows]
        require(all(flag in {0, 1} for flag in flags), f"invalid positive flag {trial_id}")
        require_equal(sum(flags), 1, f"positive count {trial_id}")
        for row in rows:
            require_equal(row["dataset_version"], target["dataset_version"], f"pool dataset {trial_id}")
            require_equal(row["reading_task"], target["reading_task"], f"pool task {trial_id}")
            require_equal(
                row["target_evaluation_partition"], target["evaluation_partition"],
                f"pool target partition {trial_id}",
            )
            require(
                row["candidate_catalog_scope"] in {
                    "training_catalog", target["evaluation_partition"]
                },
                f"invalid candidate catalog scope {trial_id}",
            )
            target_length = int(row["target_length"])
            candidate_length = int(row["candidate_length"])
            require(target_length >= 0 and candidate_length >= 0, f"negative length {trial_id}")
            require_equal(
                int(row["absolute_length_difference"]),
                abs(candidate_length - target_length),
                f"candidate length difference {trial_id}",
            )
            require_equal(row["selection_rule"], POOL_SELECTION_RULE, f"selection rule {trial_id}")
            if int(row["is_positive"]) == 1:
                require_equal(
                    row["candidate_id"],
                    f"{target['dataset_version']}::{target['reading_task']}::{target['normalized_text_sha256']}",
                    f"positive candidate identity {trial_id}",
                )

    all_metric_fields, all_metric_rows = read_csv(root / "all_seed_metrics.csv")
    require_fields(all_metric_fields, METRIC_FIELDS, "all-seed metrics")
    require_equal(len(all_metric_rows), expectations.all_seed_metric_rows, "all-seed metric rows")
    seed_average_fields, seed_average_rows = read_csv(root / "seed_averaged_predictions.csv")
    require_fields(seed_average_fields, SEED_AVERAGE_FIELDS, "seed-averaged predictions")
    require_equal(len(seed_average_rows), expectations.seed_average_rows, "seed-averaged prediction rows")
    comparison_fields, comparison_rows = read_csv(root / "paired_model_comparisons.csv")
    require_fields(comparison_fields, COMPARISON_FIELDS, "paired comparisons")
    require_equal(len(comparison_rows), expectations.comparison_rows, "paired comparison rows")
    seedwise_fields, seedwise_rows = read_csv(root / "seedwise_contrasts.csv")
    require_fields(seedwise_fields, SEEDWISE_FIELDS, "seedwise contrasts")
    require_equal(len(seedwise_rows), expectations.seedwise_rows, "seedwise contrast rows")

    accumulated_metrics: list[dict[str, str]] = []
    run_prediction_lookup: dict[tuple[str, int, str, str, str], float] = {}
    checkpoint_diagnostics: list[dict[str, Any]] = []
    fit_row_counts: set[int] = set()
    for config in CONFIGURATIONS:
        for seed in SEEDS:
            relative = f"runs/{config}/{seed}/run_summary.json"
            summary_path = safe_path(root, relative)
            verified[relative] = require_hash(summary_path, str(declared_summaries[relative]), relative)
            summary = read_json(summary_path)
            require_equal(summary.get("status"), "pass", f"{config}/{seed} status")
            binding = summary.get("binding")
            require(isinstance(binding, dict), f"{config}/{seed} binding missing")
            expected_binding = {
                "protocol_sha256": expectations.execution_protocol_sha256,
                "pilot_contract_sha256": expectations.pilot_contract_sha256,
                "input_manifest_sha256": expectations.input_manifest_sha256,
                "preserved_source_id": expectations.input_preserved_source_id,
                "project_commit": expectations.project_commit,
                "candidate_pool_sha256": expectations.candidate_pool_sha256,
                "runner_source_sha256": expectations.runner_source_sha256,
                "adapter_source_sha256": expectations.adapter_source_sha256,
                "config_id": config,
                "seed": seed,
                "run_mode": "full",
            }
            require_equal(binding, expected_binding, f"{config}/{seed} binding")
            require_equal(summary.get("epochs"), expectations.epochs, f"{config}/{seed} epochs")
            require_equal(
                summary.get("checkpoint_partition_rows"),
                Counter(row["evaluation_partition"] for row in partition_rows)["checkpoint"],
                f"{config}/{seed} checkpoint rows",
            )
            require_equal(
                summary.get("evaluation_rows"), expectations.validation_rows,
                f"{config}/{seed} evaluation rows",
            )
            fit_rows = int(summary.get("fit_rows", 0))
            require(fit_rows > 0, f"{config}/{seed} fit rows must be positive")
            fit_row_counts.add(fit_rows)
            require_equal(summary.get("trainable_parameters"), STORED_PARAMETERS[config], f"{config}/{seed} stored parameters")
            require_equal(summary.get("active_parameters_per_example"), ACTIVE_PARAMETERS[config], f"{config}/{seed} active parameters")
            require_equal(summary.get("auxiliary_factor_losses"), [], f"{config}/{seed} factor losses")
            require_bool(summary.get("held_out_test_accessed"), False, f"{config}/{seed} test access")
            declared_nested = summary.get("artifact_sha256")
            require(isinstance(declared_nested, dict), f"{config}/{seed} artifact hashes missing")
            require_equal(set(declared_nested), set(RUN_ARTIFACTS), f"{config}/{seed} artifact set")
            run_root = summary_path.parent
            for name in RUN_ARTIFACTS:
                nested_relative = f"runs/{config}/{seed}/{name}"
                nested_path = safe_path(root, nested_relative)
                verified[nested_relative] = require_hash(
                    nested_path, str(declared_nested[name]), nested_relative
                )
            checkpoint_path = safe_path(root, f"runs/{config}/{seed}/best_checkpoint.pt")
            require(checkpoint_path.stat().st_size > 0, f"empty checkpoint {config}/{seed}")

            history_fields, history_rows = read_csv(run_root / "training_history.csv")
            require_fields(history_fields, HISTORY_FIELDS, f"{config}/{seed} history")
            require_equal(len(history_rows), expectations.history_rows_per_run, f"{config}/{seed} history rows")
            require_equal([int(row["epoch"]) for row in history_rows], list(range(expectations.epochs + 1)), f"{config}/{seed} history epochs")
            require(all(row["config_id"] == config and int(row["seed"]) == seed for row in history_rows), f"{config}/{seed} history identity")
            history_scores = [float(row["headline_macro_mrr"]) for row in history_rows]
            require(all(math.isfinite(value) for value in history_scores), f"{config}/{seed} non-finite history score")
            for row in history_rows:
                loss = row["train_loss"]
                if int(row["epoch"]) == 0 and loss == "":
                    continue
                require(math.isfinite(float(loss)), f"{config}/{seed} non-finite train loss")
            best_epoch = 0
            best_score = history_scores[0]
            for epoch, score in enumerate(history_scores[1:], start=1):
                if score > best_score:
                    best_epoch, best_score = epoch, score
            require_equal(summary.get("best_epoch"), best_epoch, f"{config}/{seed} best epoch")
            require_float(
                summary.get("best_headline_macro_mrr"), best_score,
                f"{config}/{seed} best checkpoint score",
            )

            prediction_fields, prediction_rows = read_csv(run_root / "predictions.csv")
            require_fields(prediction_fields, PREDICTION_FIELDS, f"{config}/{seed} predictions")
            require_equal(len(prediction_rows), expectations.predictions_per_run, f"{config}/{seed} prediction rows")
            condition_counts = Counter((row["signal_condition"], row["task_condition"]) for row in prediction_rows)
            require_equal(condition_counts, Counter({condition: expectations.validation_rows for condition in CONDITIONS}), f"{config}/{seed} condition counts")
            for row in prediction_rows:
                require_equal(row["config_id"], config, f"{config}/{seed} prediction config")
                require_equal(int(row["seed"]), seed, f"{config}/{seed} prediction seed")
                require(row["trial_id"] in partitions, f"{config}/{seed} unknown prediction trial")
                partition = partitions[row["trial_id"]]
                for field in (
                    "cohort", "evaluation_partition", "dataset_version", "reading_task",
                    "subject_id", "normalized_text_sha256",
                ):
                    require_equal(
                        row[field], partition[field],
                        f"{config}/{seed} prediction identity {row['trial_id']}.{field}",
                    )
                key = (config, seed, row["signal_condition"], row["task_condition"], row["trial_id"])
                require(key not in run_prediction_lookup, f"duplicate run prediction {key}")
                positive_rank = int(row["positive_rank"])
                require(
                    1 <= positive_rank <= expectations.candidate_pool_size,
                    f"invalid positive rank {key}",
                )
                value = float(row["reciprocal_rank"])
                require(math.isfinite(value) and 0.0 <= value <= 1.0, f"invalid reciprocal rank {key}")
                require_float(value, 1.0 / positive_rank, f"reciprocal rank {key}")
                require_equal(int(row["top1"]), int(positive_rank <= 1), f"top1 {key}")
                require_equal(int(row["top5"]), int(positive_rank <= 5), f"top5 {key}")
                require(
                    math.isfinite(float(row["positive_minus_best_negative_margin"])),
                    f"non-finite score margin {key}",
                )
                run_prediction_lookup[key] = value

            metric_fields, metric_rows = read_csv(run_root / "metrics.csv")
            require_fields(metric_fields, METRIC_FIELDS, f"{config}/{seed} metrics")
            require_equal(len(metric_rows), expectations.metrics_per_run, f"{config}/{seed} metric rows")
            require(all(row["config_id"] == config and int(row["seed"]) == seed for row in metric_rows), f"{config}/{seed} metric identity")
            if metric_rows:
                verify_metric_rows(
                    metric_rows, prediction_rows, config, seed,
                )
            accumulated_metrics.extend(metric_rows)
            checkpoint_diagnostics.append({
                "config_id": config,
                "seed": seed,
                "best_epoch": int(summary["best_epoch"]),
                "best_checkpoint_macro_mrr": float(summary["best_headline_macro_mrr"]),
                "stored_trainable_parameters": int(summary["trainable_parameters"]),
                "active_parameters_per_example": int(summary["active_parameters_per_example"]),
            })
    require_equal(len(fit_row_counts), 1, "fit row count consistency")
    require_equal(all_metric_rows, accumulated_metrics, "all-seed metrics concatenation")

    averaged_lookup: dict[tuple[str, str, str, str], float] = {}
    for row in seed_average_rows:
        config = row["config_id"]
        signal = row["signal_condition"]
        task_condition = row["task_condition"]
        trial_id = row["trial_id"]
        require(config in CONFIGURATIONS, f"unknown seed-average configuration {config}")
        require((signal, task_condition) in CONDITIONS, "unknown seed-average condition")
        require(trial_id in partitions, "seed-average row references unknown trial")
        key = (config, signal, task_condition, trial_id)
        require(key not in averaged_lookup, f"duplicate seed-average row {key}")
        seed_values = [run_prediction_lookup[(config, seed, signal, task_condition, trial_id)] for seed in SEEDS]
        expected_mean = sum(seed_values) / len(seed_values)
        require_float(row["mean_reciprocal_rank"], expected_mean, f"seed average {key}")
        averaged_lookup[key] = float(row["mean_reciprocal_rank"])
        partition = partitions[trial_id]
        for field in ("cohort", "evaluation_partition", "reading_task", "subject_id", "normalized_text_sha256"):
            require_equal(row[field], partition[field], f"seed-average identity {key}.{field}")
    expected_average_keys = {
        (config, signal, task_condition, trial_id)
        for config in CONFIGURATIONS for signal, task_condition in CONDITIONS
        for trial_id in partitions
    }
    require_equal(set(averaged_lookup), expected_average_keys, "seed-average key set")

    decision = read_json(root / "continuation_decision.json")
    require_equal(decision.get("status"), "pass", "decision status")
    require_equal(decision.get("required_configuration"), "masked_shared_private", "decision configuration")
    require_bool(decision.get("selected_for_richer_stage"), False, "decision continuation")
    require_equal(decision.get("richer_factor_if_selected"), "tsr_instruction_relation", "decision richer factor")
    require_equal(decision.get("prohibited_factor_losses"), [
        "length_words_whitespace_v1", "nr_relation_content", "sr_sentiment_3"
    ], "decision prohibited factors")
    require_bool(decision.get("held_out_test_accessed"), False, "decision test access")
    requirements = decision.get("requirements")
    require(isinstance(requirements, dict), "decision requirements missing")
    require_equal(set(requirements), set(EXPECTED_REQUIREMENTS), "decision requirement keys")
    require(
        all(type(value) is bool for value in requirements.values()),
        "decision requirement values must be JSON booleans",
    )

    decision_ids = sorted(
        trial_id for trial_id, row in partitions.items()
        if row["evaluation_partition"] == "decision"
    )
    require_equal(len(decision_ids), expectations.decision_rows, "decision trial count")
    headline: dict[str, float] = {}
    signal_gaps: dict[str, float] = {}
    for config in CONFIGURATIONS:
        correct = {trial_id: averaged_lookup[(config, "correct_val", "correct", trial_id)] for trial_id in decision_ids}
        wrong = {trial_id: averaged_lookup[(config, "matched_wrong_val", "correct", trial_id)] for trial_id in decision_ids}
        headline[config] = task_macro(correct, partitions, decision_ids)
        differences = {trial_id: correct[trial_id] - wrong[trial_id] for trial_id in decision_ids}
        signal_gaps[config] = task_macro(differences, partitions, decision_ids)
    masked = "masked_shared_private"
    task_control_gaps: dict[str, float] = {}
    for reference, condition in (("masked_task", "masked"), ("shuffled_task", "shuffled")):
        differences = {
            trial_id: averaged_lookup[(masked, "correct_val", "correct", trial_id)]
            - averaged_lookup[(masked, "correct_val", condition, trial_id)]
            for trial_id in decision_ids
        }
        task_control_gaps[reference] = task_macro(differences, partitions, decision_ids)
    for key, value in headline.items():
        require_float(decision.get("headline_macro_mrr", {}).get(key), value, f"decision headline {key}")
    for key, value in signal_gaps.items():
        require_float(decision.get("correct_minus_matched_wrong_macro_mrr_gap", {}).get(key), value, f"decision signal gap {key}")
    for key, value in task_control_gaps.items():
        require_float(decision.get("correct_task_control_macro_mrr_gap", {}).get(key), value, f"decision task gap {key}")

    comparison_keys = {
        (row["contrast_type"], row["model"], row["reference"], row["cluster"])
        for row in comparison_rows
    }
    require_equal(comparison_keys, {
        (kind, model, reference, cluster)
        for kind, model, reference in CONTRASTS for cluster in CLUSTERS
    }, "comparison identities")
    for row in comparison_rows:
        kind, reference = row["contrast_type"], row["reference"]
        require_equal(int(row["rows"]), expectations.decision_rows, "comparison decision rows")
        require_equal(int(row["bootstrap_replicates"]), expectations.bootstrap_replicates, "comparison bootstrap count")
        expected_quantile = 0.025 if kind == "signal_specificity" else 0.0125
        require_float(row["decision_lower_quantile"], expected_quantile, "comparison decision quantile")
        if kind == "model_comparison":
            expected_delta = headline[masked] - headline[reference]
        elif kind == "signal_specificity":
            expected_delta = signal_gaps[masked]
        else:
            expected_delta = task_control_gaps[reference]
        require_float(row["mean_mrr_delta"], expected_delta, f"comparison mean {kind}/{reference}")
        for field in ("ci95_lower", "ci95_upper", "decision_lower"):
            require(math.isfinite(float(row[field])), f"non-finite comparison {field}")

    seedwise_keys = {
        (row["contrast_type"], row["model"], row["reference"], int(row["seed"]))
        for row in seedwise_rows
    }
    require_equal(seedwise_keys, {
        (kind, model, reference, seed)
        for kind, model, reference in CONTRASTS for seed in SEEDS
    }, "seedwise identities")
    for row in seedwise_rows:
        kind, reference, seed = row["contrast_type"], row["reference"], int(row["seed"])
        if kind == "model_comparison":
            left = {trial_id: run_prediction_lookup[(masked, seed, "correct_val", "correct", trial_id)] for trial_id in decision_ids}
            right = {trial_id: run_prediction_lookup[(reference, seed, "correct_val", "correct", trial_id)] for trial_id in decision_ids}
        elif kind == "signal_specificity":
            left = {trial_id: run_prediction_lookup[(masked, seed, "correct_val", "correct", trial_id)] for trial_id in decision_ids}
            right = {trial_id: run_prediction_lookup[(masked, seed, "matched_wrong_val", "correct", trial_id)] for trial_id in decision_ids}
        else:
            condition = "masked" if reference == "masked_task" else "shuffled"
            left = {trial_id: run_prediction_lookup[(masked, seed, "correct_val", "correct", trial_id)] for trial_id in decision_ids}
            right = {trial_id: run_prediction_lookup[(masked, seed, "correct_val", condition, trial_id)] for trial_id in decision_ids}
        differences = {trial_id: left[trial_id] - right[trial_id] for trial_id in decision_ids}
        delta = task_macro(differences, partitions, decision_ids)
        require_float(row["macro_mrr_delta"], delta, f"seedwise delta {kind}/{reference}/{seed}")
        require_equal(int(row["direction_positive"]), int(delta > 0), f"seedwise direction {kind}/{reference}/{seed}")

    model_rows = [row for row in comparison_rows if row["contrast_type"] == "model_comparison"]
    signal_rows = [row for row in comparison_rows if row["contrast_type"] == "signal_specificity"]
    task_rows = [row for row in comparison_rows if row["contrast_type"] == "task_control"]
    recomputed_requirements = {
        "mrr_above_both_planned_baselines": all(
            headline[masked] > headline[baseline]
            for baseline in ("generic_pooled", "task_token")
        ),
        "model_all_cluster_familywise_lower_bounds_positive": all(
            float(row["decision_lower"]) > 0 for row in model_rows
        ),
        "model_delta_positive_in_all_training_seeds": all(
            int(row["direction_positive"]) == 1
            for row in seedwise_rows if row["contrast_type"] == "model_comparison"
        ),
        "masked_model_signal_gap_positive": signal_gaps[masked] > 0,
        "signal_all_cluster_lower_bounds_positive": all(
            float(row["decision_lower"]) > 0 for row in signal_rows
        ),
        "signal_gap_not_below_either_planned_baseline": all(
            signal_gaps[masked] >= signal_gaps[baseline]
            for baseline in ("generic_pooled", "task_token")
        ),
        "task_controls_all_cluster_familywise_lower_bounds_positive": all(
            float(row["decision_lower"]) > 0 for row in task_rows
        ),
    }
    require_equal(recomputed_requirements, requirements, "recomputed decision requirements")
    require_equal(recomputed_requirements, EXPECTED_REQUIREMENTS, "frozen negative decision pattern")
    require_equal(sum(bool(value) for value in requirements.values()), 2, "passed requirement count")
    require_bool(all(requirements.values()), False, "recomputed continuation")

    existing_summary_paths = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in (root / "runs").glob("*/*/run_summary.json")
    }
    require_equal(existing_summary_paths, expected_summary_paths, "filesystem run-summary set")
    source_prefix = expectations.output_preserved_source_prefix
    require(
        output_preserved_source_id.startswith(source_prefix),
        "output preserved source ID must bind the frozen Kaggle dataset and version",
    )
    source_version = output_preserved_source_id[len(source_prefix):]
    require(
        source_version.isdigit() and int(source_version) >= 1,
        "output preserved source ID must end in a positive immutable version number",
    )
    return {
        "status": "pass",
        "schema_version": 1,
        "output_preserved_source_id": output_preserved_source_id,
        "input_preserved_source_id": expectations.input_preserved_source_id,
        "project_commit": expectations.project_commit,
        "pilot_manifest_sha256": expectations.pilot_manifest_sha256,
        "run_metadata_sha256": expectations.run_metadata_sha256,
        "continuation_decision_sha256": verified["continuation_decision.json"],
        "continuation_selected": False,
        "requirements": requirements,
        "requirement_counts": {"passed": 2, "failed": 5},
        "results": {
            "headline_macro_mrr": headline,
            "correct_minus_matched_wrong_macro_mrr_gap": signal_gaps,
            "correct_task_control_macro_mrr_gap": task_control_gaps,
            "paired_model_comparisons": numeric_comparison_rows(comparison_rows),
            "seedwise_contrasts": numeric_seedwise_rows(seedwise_rows),
            "checkpoint_partition_diagnostics": checkpoint_diagnostics,
            "stored_trainable_parameters": STORED_PARAMETERS,
            "active_parameters_per_example": ACTIVE_PARAMETERS,
        },
        "checks": {
            "all_11_top_level_artifacts_rehashed": True,
            "all_12_run_summaries_rehashed": True,
            "all_48_nested_run_artifacts_rehashed": True,
            "run_metadata_independently_bound": True,
            "frozen_protocol_revalidated": True,
            "decision_tables_cross_checked": True,
            "seven_requirements_recomputed": True,
            "uncertainty_tables_hash_bound": True,
            "bootstrap_intervals_rerun": False,
            "held_out_test_accessed": False,
            "checkpoints_loaded": False,
        },
        "verified_file_count": len(verified),
        "verified_artifact_sha256": dict(sorted(verified.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-preserved-source-id", required=True)
    args = parser.parse_args()
    report = verify(
        args.artifact_root,
        args.output_preserved_source_id,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
