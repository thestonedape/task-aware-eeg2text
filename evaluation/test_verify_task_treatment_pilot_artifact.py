"""Regression tests for the preserved task-treatment pilot verifier."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.verify_task_treatment_pilot_artifact import (
    ACTIVE_PARAMETERS,
    CLUSTERS,
    COMPARISON_FIELDS,
    CONDITIONS,
    CONFIGURATIONS,
    CONTRASTS,
    EXPECTED_REQUIREMENTS,
    HISTORY_FIELDS,
    METRIC_FIELDS,
    PARTITION_FIELDS,
    POOL_FIELDS,
    PREDICTION_FIELDS,
    RUN_ARTIFACTS,
    SEEDS,
    SEED_AVERAGE_FIELDS,
    SEEDWISE_FIELDS,
    STORED_PARAMETERS,
    TOP_LEVEL_ARTIFACTS,
    POOL_SELECTION_RULE,
    VerificationExpectations,
    sha256,
    verify,
)


PROJECT_COMMIT = "a" * 40
INPUT_MANIFEST_SHA256 = "b" * 64
INPUT_SOURCE_ID = "fixture-prompt-neutral-input-version-2"
OUTPUT_SOURCE_ID = "fixture-preserved-output-version-1"
CONFIG_BASE = {
    "generic_pooled": {20260717: 0.5, 20260718: 0.5, 20260719: 0.5},
    "separate_per_task": {20260717: 1 / 3, 20260718: 1 / 3, 20260719: 1 / 3},
    "task_token": {20260717: 1 / 3, 20260718: 1 / 3, 20260719: 1 / 3},
    "masked_shared_private": {20260717: 1.0, 20260718: 0.25, 20260719: 0.25},
}
SIGNAL_GAP = {
    "generic_pooled": 0.0,
    "separate_per_task": 0.0,
    "task_token": 0.0,
    "masked_shared_private": 0.2,
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prediction_value(config: str, seed: int, signal: str, task_condition: str) -> float:
    base = CONFIG_BASE[config][seed]
    if signal == "matched_wrong_val":
        if config == "masked_shared_private":
            return 0.5 if seed == SEEDS[0] else 0.2
        return base
    return base


def macro_for_seed(config: str, seed: int) -> float:
    return CONFIG_BASE[config][seed]


def seed_average(config: str) -> float:
    return sum(CONFIG_BASE[config].values()) / len(SEEDS)


def contrast_delta(kind: str, reference: str, seed: int | None = None) -> float:
    masked = "masked_shared_private"
    if kind == "model_comparison":
        if seed is None:
            return seed_average(masked) - seed_average(reference)
        return macro_for_seed(masked, seed) - macro_for_seed(reference, seed)
    if kind == "signal_specificity":
        if seed is None:
            return SIGNAL_GAP[masked]
        return (
            CONFIG_BASE[masked][seed]
            - prediction_value(masked, seed, "matched_wrong_val", "correct")
        )
    return 0.0


def build_fixture(root: Path) -> VerificationExpectations:
    frozen = root / "frozen_protocol"
    runner = frozen / "run_task_treatment_pilots.py"
    adapter = frozen / "task_treatment_pilots.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("# frozen synthetic runner\n", encoding="utf-8")
    adapter.write_text("# frozen synthetic adapter\n", encoding="utf-8")
    runner_sha = sha256(runner)
    adapter_sha = sha256(adapter)

    contract_path = frozen / "task_treatment_pilot_contract.json"
    write_json(contract_path, {
        "status": "frozen_before_pilot_training",
        "training": {"seeds": list(SEEDS), "auxiliary_factor_losses": []},
        "evaluation": {"held_out_test_accessed": False},
    })
    contract_sha = sha256(contract_path)
    protocol_path = frozen / "task_treatment_pilot_execution_protocol.json"
    write_json(protocol_path, {
        "status": "frozen_before_pilot_execution",
        "parent_contract_sha256": contract_sha,
        "input": {
            "preserved_source_id": INPUT_SOURCE_ID,
            "combined_manifest_sha256": INPUT_MANIFEST_SHA256,
        },
        "training": {
            "stored_trainable_parameters": STORED_PARAMETERS,
            "active_parameters_per_example": ACTIVE_PARAMETERS,
            "auxiliary_factor_losses": [],
        },
        "evaluation": {"checkpoint_decision_partition": {"decision_rows": 2}},
        "continuation": {"held_out_test_accessed": False},
    })
    protocol_sha = sha256(protocol_path)

    partition_rows = [
        {
            "trial_id": "trial_nr",
            "cohort": "primary_zuco2_nr_tsr",
            "dataset_version": "ZuCo2",
            "reading_task": "NR",
            "subject_id": "subject_nr",
            "normalized_text_sha256": "1" * 64,
            "evaluation_partition": "decision",
        },
        {
            "trial_id": "trial_tsr",
            "cohort": "primary_zuco2_nr_tsr",
            "dataset_version": "ZuCo2",
            "reading_task": "TSR",
            "subject_id": "subject_tsr",
            "normalized_text_sha256": "2" * 64,
            "evaluation_partition": "decision",
        },
        {
            "trial_id": "trial_checkpoint_nr",
            "cohort": "primary_zuco2_nr_tsr",
            "dataset_version": "ZuCo2",
            "reading_task": "NR",
            "subject_id": "subject_checkpoint_nr",
            "normalized_text_sha256": "3" * 64,
            "evaluation_partition": "checkpoint",
        },
        {
            "trial_id": "trial_checkpoint_tsr",
            "cohort": "primary_zuco2_nr_tsr",
            "dataset_version": "ZuCo2",
            "reading_task": "TSR",
            "subject_id": "subject_checkpoint_tsr",
            "normalized_text_sha256": "4" * 64,
            "evaluation_partition": "checkpoint",
        },
        {
            "trial_id": "trial_sr",
            "cohort": "auxiliary_sr",
            "dataset_version": "ZuCo1",
            "reading_task": "SR",
            "subject_id": "subject_sr",
            "normalized_text_sha256": "5" * 64,
            "evaluation_partition": "auxiliary_sr",
        },
        {
            "trial_id": "trial_diag_nr",
            "cohort": "zuco1_nr_tsr_noncausal",
            "dataset_version": "ZuCo1",
            "reading_task": "NR",
            "subject_id": "subject_diag_nr",
            "normalized_text_sha256": "6" * 64,
            "evaluation_partition": "zuco1_noncausal_diagnostic",
        },
        {
            "trial_id": "trial_diag_tsr",
            "cohort": "zuco1_nr_tsr_noncausal",
            "dataset_version": "ZuCo1",
            "reading_task": "TSR",
            "subject_id": "subject_diag_tsr",
            "normalized_text_sha256": "7" * 64,
            "evaluation_partition": "zuco1_noncausal_diagnostic",
        },
    ]
    partition_path = root / "validation_partition.csv"
    write_csv(partition_path, PARTITION_FIELDS, partition_rows)
    partition_sha = sha256(partition_path)

    pool_rows: list[dict[str, Any]] = []
    for partition in partition_rows:
        for rank in range(5):
            pool_rows.append({
                "target_trial_id": partition["trial_id"],
                "candidate_rank": rank,
                "candidate_id": (
                    f"{partition['dataset_version']}::{partition['reading_task']}::"
                    f"{partition['normalized_text_sha256']}"
                    if rank == 0 else f"negative_{partition['trial_id']}_{rank}"
                ),
                "is_positive": int(rank == 0),
                "dataset_version": partition["dataset_version"],
                "reading_task": partition["reading_task"],
                "target_evaluation_partition": partition["evaluation_partition"],
                "candidate_catalog_scope": partition["evaluation_partition"],
                "target_length": 5,
                "candidate_length": 5,
                "absolute_length_difference": 0,
                "selection_rule": POOL_SELECTION_RULE,
            })
    pool_path = root / "frozen_candidate_pools.csv"
    write_csv(pool_path, POOL_FIELDS, pool_rows)
    pool_sha = sha256(pool_path)

    all_metrics: list[dict[str, Any]] = []
    run_summary_hashes: dict[str, str] = {}
    run_predictions: dict[tuple[str, int, str, str, str], float] = {}
    for config in CONFIGURATIONS:
        for seed in SEEDS:
            run_root = root / "runs" / config / str(seed)
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "best_checkpoint.pt").write_bytes(
                f"opaque checkpoint {config} {seed}\n".encode("utf-8")
            )
            write_csv(run_root / "training_history.csv", HISTORY_FIELDS, [
                {
                    "config_id": config,
                    "seed": seed,
                    "epoch": 0,
                    "train_loss": "",
                    "headline_macro_mrr": CONFIG_BASE[config][seed] - 0.01,
                },
                {
                    "config_id": config,
                    "seed": seed,
                    "epoch": 1,
                    "train_loss": 1.0,
                    "headline_macro_mrr": CONFIG_BASE[config][seed],
                },
                {
                    "config_id": config,
                    "seed": seed,
                    "epoch": 2,
                    "train_loss": 0.9,
                    "headline_macro_mrr": CONFIG_BASE[config][seed],
                },
            ])
            prediction_rows: list[dict[str, Any]] = []
            for signal, task_condition in CONDITIONS:
                for partition in partition_rows:
                    value = prediction_value(config, seed, signal, task_condition)
                    positive_rank = int(round(1.0 / value))
                    run_predictions[(config, seed, signal, task_condition, partition["trial_id"])] = value
                    prediction_rows.append({
                        "config_id": config,
                        "seed": seed,
                        "signal_condition": signal,
                        "task_condition": task_condition,
                        "trial_id": partition["trial_id"],
                        "cohort": partition["cohort"],
                        "evaluation_partition": partition["evaluation_partition"],
                        "dataset_version": partition["dataset_version"],
                        "reading_task": partition["reading_task"],
                        "subject_id": partition["subject_id"],
                        "normalized_text_sha256": partition["normalized_text_sha256"],
                        "positive_rank": positive_rank,
                        "reciprocal_rank": value,
                        "top1": int(positive_rank <= 1),
                        "top5": int(positive_rank <= 5),
                        "positive_minus_best_negative_margin": value - 0.5,
                    })
            write_csv(run_root / "predictions.csv", PREDICTION_FIELDS, prediction_rows)
            metric_rows: list[dict[str, Any]] = []
            for signal, task_condition in CONDITIONS:
                members = [
                    row for row in prediction_rows
                    if row["signal_condition"] == signal
                    and row["task_condition"] == task_condition
                ]
                groups = [
                    (
                        "headline_decision_task", task,
                        [
                            row for row in members
                            if row["evaluation_partition"] == "decision"
                            and row["reading_task"] == task
                        ],
                    )
                    for task in ("NR", "TSR")
                ]
                groups.extend([
                    (
                        "checkpoint_task_diagnostic", task,
                        [
                            row for row in members
                            if row["evaluation_partition"] == "checkpoint"
                            and row["reading_task"] == task
                        ],
                    )
                    for task in ("NR", "TSR")
                ])
                groups.append((
                    "auxiliary_sr", "SR",
                    [row for row in members if row["cohort"] == "auxiliary_sr"],
                ))
                groups.extend([
                    (
                        "zuco1_noncausal_diagnostic", task,
                        [
                            row for row in members
                            if row["cohort"] == "zuco1_nr_tsr_noncausal"
                            and row["reading_task"] == task
                        ],
                    )
                    for task in ("NR", "TSR")
                ])
                condition_metrics: list[dict[str, Any]] = []
                for scope, task, group in groups:
                    count = len(group)
                    condition_metrics.append({
                        "config_id": config,
                        "seed": seed,
                        "signal_condition": signal,
                        "task_condition": task_condition,
                        "scope": scope,
                        "reading_task": task,
                        "rows": count,
                        "top1": sum(row["top1"] for row in group) / count,
                        "top5": sum(row["top5"] for row in group) / count,
                        "mrr": sum(row["reciprocal_rank"] for row in group) / count,
                        "mean_positive_rank": sum(row["positive_rank"] for row in group) / count,
                        "mean_positive_minus_best_negative_margin": (
                            sum(row["positive_minus_best_negative_margin"] for row in group)
                            / count
                        ),
                    })
                headline = [
                    row for row in condition_metrics
                    if row["scope"] == "headline_decision_task"
                ]
                condition_metrics.append({
                    "config_id": config,
                    "seed": seed,
                    "signal_condition": signal,
                    "task_condition": task_condition,
                    "scope": "headline_macro",
                    "reading_task": "NR|TSR",
                    "rows": sum(row["rows"] for row in headline),
                    **{
                        field: sum(row[field] for row in headline) / 2
                        for field in (
                            "top1", "top5", "mrr", "mean_positive_rank",
                            "mean_positive_minus_best_negative_margin",
                        )
                    },
                })
                metric_rows.extend(condition_metrics)
            write_csv(run_root / "metrics.csv", METRIC_FIELDS, metric_rows)
            all_metrics.extend(metric_rows)
            nested_hashes = {name: sha256(run_root / name) for name in RUN_ARTIFACTS}
            summary = {
                "status": "pass",
                "binding": {
                    "protocol_sha256": protocol_sha,
                    "pilot_contract_sha256": contract_sha,
                    "input_manifest_sha256": INPUT_MANIFEST_SHA256,
                    "preserved_source_id": INPUT_SOURCE_ID,
                    "project_commit": PROJECT_COMMIT,
                    "candidate_pool_sha256": pool_sha,
                    "runner_source_sha256": runner_sha,
                    "adapter_source_sha256": adapter_sha,
                    "config_id": config,
                    "seed": seed,
                    "run_mode": "full",
                },
                "epochs": 2,
                "best_epoch": 1,
                "best_headline_macro_mrr": CONFIG_BASE[config][seed],
                "checkpoint_partition_rows": 2,
                "fit_rows": 5,
                "evaluation_rows": 7,
                "trainable_parameters": STORED_PARAMETERS[config],
                "active_parameters_per_example": ACTIVE_PARAMETERS[config],
                "auxiliary_factor_losses": [],
                "held_out_test_accessed": False,
                "artifact_sha256": nested_hashes,
            }
            summary_path = run_root / "run_summary.json"
            write_json(summary_path, summary)
            relative = summary_path.relative_to(root).as_posix()
            run_summary_hashes[relative] = sha256(summary_path)

    write_csv(root / "all_seed_metrics.csv", METRIC_FIELDS, all_metrics)
    averaged_rows: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        for signal, task_condition in CONDITIONS:
            for partition in partition_rows:
                values = [
                    run_predictions[(config, seed, signal, task_condition, partition["trial_id"])]
                    for seed in SEEDS
                ]
                averaged_rows.append({
                    "config_id": config,
                    "signal_condition": signal,
                    "task_condition": task_condition,
                    "trial_id": partition["trial_id"],
                    "cohort": partition["cohort"],
                    "evaluation_partition": partition["evaluation_partition"],
                    "reading_task": partition["reading_task"],
                    "subject_id": partition["subject_id"],
                    "normalized_text_sha256": partition["normalized_text_sha256"],
                    "mean_reciprocal_rank": sum(values) / len(values),
                })
    write_csv(root / "seed_averaged_predictions.csv", SEED_AVERAGE_FIELDS, averaged_rows)

    comparison_rows: list[dict[str, Any]] = []
    for kind, model, reference in CONTRASTS:
        delta = contrast_delta(kind, reference)
        for cluster in CLUSTERS:
            comparison_rows.append({
                "contrast_type": kind,
                "model": model,
                "reference": reference,
                "cluster": cluster,
                "rows": 2,
                "mean_mrr_delta": delta,
                "ci95_lower": delta - 0.10,
                "ci95_upper": delta + 0.10,
                "decision_lower_quantile": 0.025 if kind == "signal_specificity" else 0.0125,
                "decision_lower": -0.10,
                "bootstrap_replicates": 7,
            })
    write_csv(root / "paired_model_comparisons.csv", COMPARISON_FIELDS, comparison_rows)

    seedwise_rows: list[dict[str, Any]] = []
    for kind, model, reference in CONTRASTS:
        for seed in SEEDS:
            delta = contrast_delta(kind, reference, seed)
            seedwise_rows.append({
                "contrast_type": kind,
                "model": model,
                "reference": reference,
                "seed": seed,
                "macro_mrr_delta": delta,
                "direction_positive": int(delta > 0),
            })
    write_csv(root / "seedwise_contrasts.csv", SEEDWISE_FIELDS, seedwise_rows)

    masked_headline = seed_average("masked_shared_private")
    write_json(root / "continuation_decision.json", {
        "status": "pass",
        "required_configuration": "masked_shared_private",
        "selected_for_richer_stage": False,
        "richer_factor_if_selected": "tsr_instruction_relation",
        "prohibited_factor_losses": [
            "length_words_whitespace_v1", "nr_relation_content", "sr_sentiment_3"
        ],
        "held_out_test_accessed": False,
        "headline_macro_mrr": {config: seed_average(config) for config in CONFIGURATIONS},
        "correct_minus_matched_wrong_macro_mrr_gap": SIGNAL_GAP,
        "correct_task_control_macro_mrr_gap": {"masked_task": 0.0, "shuffled_task": 0.0},
        "requirements": EXPECTED_REQUIREMENTS,
        "masked_headline_for_fixture_readability": masked_headline,
    })

    top_hashes = {relative: sha256(root / relative) for relative in TOP_LEVEL_ARTIFACTS}
    manifest = {
        "status": "pass",
        "schema_version": 1,
        "run_mode": "full",
        "project_commit": PROJECT_COMMIT,
        "pilot_contract_sha256": contract_sha,
        "execution_protocol_sha256": protocol_sha,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "preserved_source_id": INPUT_SOURCE_ID,
        "runner_source_sha256": runner_sha,
        "adapter_source_sha256": adapter_sha,
        "candidate_pool_sha256": pool_sha,
        "validation_partition_sha256": partition_sha,
        "configurations": list(CONFIGURATIONS),
        "seeds": list(SEEDS),
        "parameter_budget": {
            **STORED_PARAMETERS,
            "reference": 196608,
            "maximum_relative_deviation": 0.00146484375,
        },
        "active_parameters_per_example": ACTIVE_PARAMETERS,
        "auxiliary_factor_losses": [],
        "held_out_test_accessed": False,
        "continuation_selected": False,
        "artifact_sha256": top_hashes,
        "run_summary_sha256": run_summary_hashes,
    }
    manifest_path = root / "pilot_manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    metadata_path = root / "run_metadata.json"
    write_json(metadata_path, {
        "status": "pass",
        "project_commit": PROJECT_COMMIT,
        "execution_protocol_sha256": protocol_sha,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "candidate_pool_sha256": pool_sha,
        "pilot_manifest_sha256": manifest_sha,
        "continuation_selected": False,
        "held_out_test_accessed": False,
    })
    return VerificationExpectations(
        pilot_manifest_sha256=manifest_sha,
        run_metadata_sha256=sha256(metadata_path),
        top_level_sha256=top_hashes,
        run_summary_sha256=run_summary_hashes,
        project_commit=PROJECT_COMMIT,
        pilot_contract_sha256=contract_sha,
        execution_protocol_sha256=protocol_sha,
        input_manifest_sha256=INPUT_MANIFEST_SHA256,
        input_preserved_source_id=INPUT_SOURCE_ID,
        runner_source_sha256=runner_sha,
        adapter_source_sha256=adapter_sha,
        candidate_pool_sha256=pool_sha,
        validation_partition_sha256=partition_sha,
        output_preserved_source_prefix="fixture-preserved-output-version-",
        validation_rows=7,
        candidate_pool_size=5,
        candidate_rows=35,
        seed_average_rows=168,
        all_seed_metric_rows=576,
        comparison_rows=15,
        seedwise_rows=15,
        history_rows_per_run=3,
        predictions_per_run=42,
        metrics_per_run=48,
        decision_rows=2,
        epochs=2,
        bootstrap_replicates=7,
        partition_counts=(
            ("checkpoint", 2),
            ("decision", 2),
            ("auxiliary_sr", 1),
            ("zuco1_noncausal_diagnostic", 2),
        ),
    )


def rebind_manifest(root: Path, expectations: VerificationExpectations) -> VerificationExpectations:
    manifest_path = root / "pilot_manifest.json"
    manifest_sha = sha256(manifest_path)
    metadata_path = root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pilot_manifest_sha256"] = manifest_sha
    write_json(metadata_path, metadata)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return replace(
        expectations,
        pilot_manifest_sha256=manifest_sha,
        run_metadata_sha256=sha256(metadata_path),
        top_level_sha256=dict(manifest["artifact_sha256"]),
    )


class TaskTreatmentPilotArtifactVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.expectations = build_fixture(self.root)

    def test_valid_artifact_passes_deterministically_twice(self) -> None:
        first = verify(self.root, OUTPUT_SOURCE_ID, self.expectations)
        second = verify(self.root, OUTPUT_SOURCE_ID, self.expectations)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "pass")
        self.assertFalse(first["continuation_selected"])
        self.assertEqual(first["requirement_counts"], {"passed": 2, "failed": 5})
        self.assertEqual(first["verified_file_count"], 73)
        self.assertFalse(first["checks"]["held_out_test_accessed"])
        self.assertFalse(first["checks"]["checkpoints_loaded"])

    def test_rejects_top_level_artifact_tamper(self) -> None:
        with (self.root / "frozen_candidate_pools.csv").open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
        with self.assertRaisesRegex(ValueError, r"frozen_candidate_pools\.csv SHA256"):
            verify(self.root, OUTPUT_SOURCE_ID, self.expectations)

    def test_rejects_nested_run_artifact_tamper(self) -> None:
        checkpoint = self.root / "runs" / "generic_pooled" / str(SEEDS[0]) / "best_checkpoint.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, r"best_checkpoint\.pt SHA256"):
            verify(self.root, OUTPUT_SOURCE_ID, self.expectations)

    def test_rejects_missing_run_summary(self) -> None:
        summary = self.root / "runs" / "generic_pooled" / str(SEEDS[0]) / "run_summary.json"
        summary.unlink()
        with self.assertRaisesRegex(ValueError, r"missing runs/generic_pooled/.*/run_summary\.json"):
            verify(self.root, OUTPUT_SOURCE_ID, self.expectations)

    def test_rejects_held_out_test_access_even_with_updated_metadata_hash(self) -> None:
        metadata_path = self.root / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["held_out_test_accessed"] = True
        write_json(metadata_path, metadata)
        expectations = replace(self.expectations, run_metadata_sha256=sha256(metadata_path))
        with self.assertRaisesRegex(ValueError, r"metadata test access"):
            verify(self.root, OUTPUT_SOURCE_ID, expectations)

    def test_rejects_integer_disguised_as_false_boolean(self) -> None:
        metadata_path = self.root / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["held_out_test_accessed"] = 0
        write_json(metadata_path, metadata)
        expectations = replace(self.expectations, run_metadata_sha256=sha256(metadata_path))
        with self.assertRaisesRegex(ValueError, r"expected boolean False"):
            verify(self.root, OUTPUT_SOURCE_ID, expectations)

    def test_rejects_unversioned_output_source_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must bind the frozen Kaggle dataset"):
            verify(self.root, "fixture-preserved-output", self.expectations)

    def test_rejects_semantic_continuation_decision_mismatch_after_rebinding(self) -> None:
        decision_path = self.root / "continuation_decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["selected_for_richer_stage"] = True
        write_json(decision_path, decision)
        manifest_path = self.root / "pilot_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_sha256"]["continuation_decision.json"] = sha256(decision_path)
        write_json(manifest_path, manifest)
        expectations = rebind_manifest(self.root, self.expectations)
        with self.assertRaisesRegex(ValueError, r"decision continuation"):
            verify(self.root, OUTPUT_SOURCE_ID, expectations)


if __name__ == "__main__":
    unittest.main()
