"""Verify and aggregate the complete frozen P4b full-shard matrix.

The aggregator is intentionally downstream of the independent per-shard
verifier.  It accepts exactly one sealed shard and one separately generated
verification report for every frozen fold/seed unit, re-runs the independent
verifier, then reconstructs the decision inputs from raw candidate scores.

Integrity and the scientific decision are separate states: a scientifically
negative result remains a successfully verified matrix.  No output directory
is published unless all shards, all histories, all predictions, all raw pools,
and the frozen validation/test-access denials pass first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FULL_EXECUTION_CONTRACT_SHA256 = (
    "99c1235d21ce0dd9eb80b1c1c0c3930b3b7347007ebc35be3385f0bc253a837c"
)
ARM_IDS = (
    "global_mixed",
    "true_task_segmented",
    "pseudo_task_segmented",
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
CONFIRMATION_SCORE_FIELDS = (
    "arm_id", "training_seed", "outer_fold", "trial_id", "reading_task",
    "subject_id", "normalized_text_sha256", "signal_condition",
    "signal_trial_id", "signal_subject_id", "signal_normalized_text_sha256",
    "candidate_rank", "candidate_normalized_text_sha256",
    "candidate_text_target_id", "is_positive", "is_designated_donor_text",
    "score",
)
SIGNAL_CONDITIONS = ("correct", "matched_wrong")
PRESERVED_EVIDENCE_SHA256 = {
    "candidate_pools.csv": (
        "a0b4102acf88d956fc494a5d4237fbdac0e083bf38e5f2dbf621d40b0942ca0c"
    ),
    "confirmation_donors.csv": (
        "fa599beb26fe6ffe1e3b5e849a8fefc0cddeef71bba4554adfd310bba203f87c"
    ),
    "schedule_indices.u32le": (
        "79543e72f496ee3f7a8140556b274c15ecc5992e900a07bed5e5c74a2ddd7cbc"
    ),
    "schedule_units.csv": (
        "e2644a51ac578b18388ce82d07d910714182cf674e60b0a5f2c547067b8720aa"
    ),
    "trial_catalog.csv": (
        "3d93e0cea4290ac22e8111760241d04109392e6f024abae85a4d9504fc4f8fc9"
    ),
}
REGISTRY_FIELDS = frozenset({
    "schema_version", "status", "full_execution_contract_sha256",
    "launch_authorization_sha256", "partial_scientific_decision_permitted",
    "held_out_test_accessed", "shards",
})
REGISTRY_SHARD_FIELDS = frozenset({
    "shard_id", "outer_fold", "training_seed", "dataset_slug",
    "dataset_version", "preserved_source_id", "full_shard_manifest_sha256",
    "verification_report_sha256",
})


@dataclass(frozen=True)
class AggregationSpec:
    arms: tuple[str, ...] = ARM_IDS
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (20260717, 20260718, 20260719)
    fold_counts: tuple[tuple[int, int], ...] = (
        (0, 1810), (1, 1785), (2, 1790), (3, 1814), (4, 1812),
    )
    task_counts: tuple[tuple[str, int], ...] = (("NR", 4126), ("TSR", 4885))
    pool_size: int = 24
    epochs: int = 40
    batches_per_epoch: int = 105

    @property
    def expected_shards(self) -> set[tuple[int, int]]:
        return {(fold, seed) for fold in self.folds for seed in self.seeds}

    @property
    def expected_runs(self) -> int:
        return len(self.expected_shards) * len(self.arms)

    @property
    def expected_trials(self) -> int:
        return sum(count for _, count in self.fold_counts)

    @property
    def expected_predictions(self) -> int:
        return (
            len(self.arms) * len(self.seeds) * len(SIGNAL_CONDITIONS)
            * self.expected_trials
        )

    @property
    def optimizer_steps(self) -> int:
        return self.epochs * self.batches_per_epoch


PRODUCTION_SPEC = AggregationSpec()


@dataclass(frozen=True)
class ShardInput:
    artifact_root: Path
    verification_report: Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_equal(actual: object, expected: object, label: str) -> None:
    require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def parse_bool(value: object, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label}: expected true/false, got {value!r}")


def parse_int(value: object, label: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected integer, got {value!r}") from exc


def parse_float(value: object, label: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected float, got {value!r}") from exc
    require(math.isfinite(parsed), f"{label}: non-finite value")
    return parsed


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
            require_equal(reader.fieldnames, list(fields), f"columns in {path}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot read CSV: {path}") from exc
    require(
        all(None not in row and all(value is not None for value in row.values())
            for row in rows),
        f"malformed CSV row in {path}",
    )
    return rows


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _default_reverify(
    root: Path,
    report: Mapping[str, Any],
    launch_authorization_sha256: str,
    spec: AggregationSpec,
    protocol_root: Path,
    schedule_root: Path,
) -> dict[str, Any]:
    from evaluation.verify_task_segmented_full_shard_artifact import (
        VerificationSpec,
        verify,
    )

    verifier_spec = VerificationSpec(
        arms=spec.arms,
        folds=spec.folds,
        seeds=spec.seeds,
        fold_counts=tuple(dict(spec.fold_counts)[fold] for fold in spec.folds),
        epochs=spec.epochs,
        batches_per_epoch=spec.batches_per_epoch,
        pool_size=spec.pool_size,
    )
    return verify(
        root,
        str(report["full_shard_manifest_sha256"]),
        FULL_EXECUTION_CONTRACT_SHA256,
        launch_authorization_sha256,
        spec=verifier_spec,
        preserved_source_id=str(report["preserved_source_id"]),
        protocol_root=protocol_root,
        schedule_root=schedule_root,
    )


Reverify = Callable[
    [Path, Mapping[str, Any], str, AggregationSpec, Path, Path], dict[str, Any]
]
Decision = Callable[
    [Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]], dict[str, Any]
]


def _default_decision(
    run_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    from evaluation.decide_task_segmented_objective import (
        PRODUCTION_SPEC as DECISION_SPEC,
        decide,
        validate_inputs,
    )

    validated = validate_inputs(run_rows, prediction_rows, DECISION_SPEC)
    return decide(validated, DECISION_SPEC)


def _validate_verification_report(
    report: Mapping[str, Any], launch_sha256: str, spec: AggregationSpec,
) -> tuple[int, int]:
    required = {
        "status", "schema_version", "preserved_source_id",
        "full_shard_manifest_sha256", "full_execution_contract_sha256",
        "launch_authorization_sha256", "outer_fold", "training_seed", "arms",
        "epochs_per_arm", "optimizer_steps_per_arm", "paired_initial_state",
        "total_optimizer_steps", "checkpoint_rows_per_arm",
        "confirmation_prediction_rows", "confirmation_candidate_score_rows",
        "best_epochs", "schedule_unit_sha256", "verified_artifact_sha256",
        "epoch_zero_eligible", "earliest_tie_selection_verified",
        "append_only_strict_incumbent_history_verified",
        "partial_scientific_decision_permitted",
        "correct_and_matched_wrong_provenance_verified",
        "preserved_protocol_evidence_verified",
        "preserved_schedule_evidence_verified",
        "recomputed_binding_sha256", "preserved_evidence_sha256",
        "runtime_fingerprint_verified", "runtime_fingerprint_sha256",
        "git_execution_boundary_verified",
        "checkpoint_deserialized", "official_validation_used_for_confirmation",
        "held_out_test_accessed",
    }
    require_equal(set(report), required, "verification-report fields")
    require_equal(report["status"], "pass", "verification status")
    require_equal(report["schema_version"], 1, "verification schema")
    require(
        isinstance(report["preserved_source_id"], str)
        and bool(report["preserved_source_id"].strip()),
        "verification report lacks a preserved source ID",
    )
    require(is_sha256(report["full_shard_manifest_sha256"]),
            "invalid verified shard-manifest SHA256")
    require_equal(report["full_execution_contract_sha256"],
                  FULL_EXECUTION_CONTRACT_SHA256, "verified full contract")
    require_equal(report["launch_authorization_sha256"], launch_sha256,
                  "verified launch authorization")
    fold = parse_int(report["outer_fold"], "verified outer fold")
    seed = parse_int(report["training_seed"], "verified training seed")
    require((fold, seed) in spec.expected_shards,
            f"verified shard is outside frozen matrix: {(fold, seed)}")
    require_equal(report["arms"], list(spec.arms), "verified shard arms")
    require_equal(report["epochs_per_arm"], spec.epochs, "verified epochs")
    require_equal(report["optimizer_steps_per_arm"], spec.optimizer_steps,
                  "verified optimizer steps")
    require_equal(report["total_optimizer_steps"],
                  len(spec.arms) * spec.optimizer_steps,
                  "verified total optimizer steps")
    fold_counts = dict(spec.fold_counts)
    checkpoint_fold = spec.folds[(spec.folds.index(fold) + 1) % len(spec.folds)]
    require_equal(report["checkpoint_rows_per_arm"],
                  (spec.epochs + 1) * fold_counts[checkpoint_fold],
                  "verified checkpoint rows")
    require_equal(report["confirmation_prediction_rows"],
                  len(spec.arms) * len(SIGNAL_CONDITIONS) * fold_counts[fold],
                  "verified confirmation-prediction rows")
    require_equal(
        report["confirmation_candidate_score_rows"],
        len(spec.arms) * len(SIGNAL_CONDITIONS) * fold_counts[fold] * spec.pool_size,
        "verified confirmation-score rows",
    )
    require(is_sha256(report["schedule_unit_sha256"]),
            "invalid verified schedule-unit SHA256")
    require(is_sha256(report["recomputed_binding_sha256"]),
            "invalid independently recomputed runner-binding SHA256")
    require(is_sha256(report["runtime_fingerprint_sha256"]),
            "invalid verified runtime-fingerprint SHA256")
    require_equal(
        report["preserved_evidence_sha256"],
        PRESERVED_EVIDENCE_SHA256,
        "verified preserved protocol/schedule evidence hashes",
    )
    best_epochs = report["best_epochs"]
    require(isinstance(best_epochs, dict) and set(best_epochs) == set(spec.arms),
            "verified best-epoch arms differ")
    require(all(isinstance(epoch, int) and 0 <= epoch <= spec.epochs
                for epoch in best_epochs.values()), "invalid verified best epoch")
    verified_hashes = report["verified_artifact_sha256"]
    require(isinstance(verified_hashes, dict) and bool(verified_hashes)
            and all(isinstance(name, str) and is_sha256(digest)
                    for name, digest in verified_hashes.items()),
            "invalid verified artifact hashes")
    for field in (
        "paired_initial_state", "epoch_zero_eligible",
        "earliest_tie_selection_verified",
        "append_only_strict_incumbent_history_verified",
        "correct_and_matched_wrong_provenance_verified",
        "preserved_protocol_evidence_verified",
        "preserved_schedule_evidence_verified",
        "runtime_fingerprint_verified",
        "git_execution_boundary_verified",
    ):
        require_equal(report[field], True, f"verification {field}")
    require_equal(report["checkpoint_deserialized"], False,
                  "checkpoint deserialization flag")
    require_equal(report["partial_scientific_decision_permitted"], False,
                  "partial-shard scientific-decision permission")
    require_equal(report["official_validation_used_for_confirmation"], False,
                  "official-validation drift")
    require_equal(report["held_out_test_accessed"], False, "held-out-test drift")
    return fold, seed


def _validate_frozen_registry(
    path: Path,
    expected_sha256: str,
    launch_sha256: str,
    spec: AggregationSpec,
) -> dict[tuple[int, int], dict[str, Any]]:
    require(is_sha256(expected_sha256),
            "invalid externally supplied frozen-registry SHA256")
    require_equal(sha256(path), expected_sha256,
                  "out-of-band frozen-registry SHA256")
    registry = read_json(path)
    require_equal(set(registry), set(REGISTRY_FIELDS), "frozen-registry fields")
    require(type(registry["schema_version"]) is int
            and registry["schema_version"] == 1,
            "frozen-registry schema must be integer 1")
    require_equal(registry["status"], "frozen_after_all_p4b_shards_preserved",
                  "frozen-registry status")
    require_equal(registry["full_execution_contract_sha256"],
                  FULL_EXECUTION_CONTRACT_SHA256, "frozen-registry contract")
    require_equal(registry["launch_authorization_sha256"], launch_sha256,
                  "frozen-registry launch authorization")
    require(registry["partial_scientific_decision_permitted"] is False,
            "frozen-registry partial decision permission must be false")
    require(registry["held_out_test_accessed"] is False,
            "frozen-registry held-out-test access must be false")
    entries = registry["shards"]
    require(isinstance(entries, list), "frozen-registry shards must be a list")
    ordered_units = sorted(spec.expected_shards)
    require_equal(len(entries), len(ordered_units),
                  "frozen-registry shard count")
    by_unit: dict[tuple[int, int], dict[str, Any]] = {}
    dataset_versions: set[tuple[str, int]] = set()
    source_ids: set[str] = set()
    manifest_hashes: set[str] = set()
    report_hashes: set[str] = set()
    for index, (entry, expected_unit) in enumerate(zip(entries, ordered_units)):
        require(isinstance(entry, dict),
                f"frozen-registry shard {index} is not an object")
        require_equal(set(entry), set(REGISTRY_SHARD_FIELDS),
                      f"frozen-registry shard {index} fields")
        require(type(entry["outer_fold"]) is int,
                f"registry outer fold is not an integer at shard {index}")
        require(type(entry["training_seed"]) is int,
                f"registry training seed is not an integer at shard {index}")
        fold = entry["outer_fold"]
        seed = entry["training_seed"]
        unit = (fold, seed)
        require_equal(unit, expected_unit,
                      f"frozen-registry shard ordering at index {index}")
        require_equal(entry["shard_id"], f"p4b-f{fold}-s{seed}",
                      f"frozen-registry shard ID at index {index}")
        slug = entry["dataset_slug"]
        version = entry["dataset_version"]
        source_id = entry["preserved_source_id"]
        require(isinstance(slug, str) and bool(slug.strip()),
                f"invalid dataset slug at registry shard {index}")
        require(type(version) is int and version > 0,
                f"invalid dataset version at registry shard {index}")
        require(isinstance(source_id, str) and bool(source_id.strip()),
                f"invalid preserved source ID at registry shard {index}")
        require(is_sha256(entry["full_shard_manifest_sha256"]),
                f"invalid shard-manifest SHA256 at registry shard {index}")
        require(is_sha256(entry["verification_report_sha256"]),
                f"invalid verification-report SHA256 at registry shard {index}")
        require((slug, version) not in dataset_versions,
                f"duplicate registry dataset version: {(slug, version)}")
        require(source_id not in source_ids,
                f"duplicate registry preserved source ID: {source_id}")
        require(entry["full_shard_manifest_sha256"] not in manifest_hashes,
                "duplicate registry shard-manifest SHA256")
        require(entry["verification_report_sha256"] not in report_hashes,
                "duplicate registry verification-report SHA256")
        dataset_versions.add((slug, version))
        source_ids.add(source_id)
        manifest_hashes.add(entry["full_shard_manifest_sha256"])
        report_hashes.add(entry["verification_report_sha256"])
        by_unit[unit] = dict(entry)
    require_equal(set(by_unit), spec.expected_shards,
                  "frozen-registry fold/seed matrix")
    return by_unit


def _validate_run_rows(
    rows: Sequence[Mapping[str, str]], fold: int, seed: int, spec: AggregationSpec,
) -> list[dict[str, str]]:
    require_equal(len(rows), len(spec.arms), "per-shard run-manifest row count")
    observed: set[str] = set()
    result: list[dict[str, str]] = []
    for row in rows:
        arm = row["arm_id"]
        require(arm in spec.arms and arm not in observed,
                f"invalid/duplicate run arm: {arm}")
        observed.add(arm)
        require_equal(parse_int(row["outer_fold"], "run outer fold"), fold,
                      "run outer fold")
        require_equal(parse_int(row["training_seed"], "run seed"), seed,
                      "run seed")
        require_equal(row["status"], "complete", f"{arm} run status")
        require_equal(row["run_mode"], "full_scientific", f"{arm} run mode")
        require(parse_bool(row["full_training_authorized"], "full authorization"),
                f"{arm} lacks full-training authorization")
        require(not parse_bool(row["scientific_decision_permitted"],
                               "decision permission"),
                f"{arm} prematurely permits a partial scientific decision")
        require(not parse_bool(row["official_validation_used_for_confirmation"],
                               "official-validation flag"),
                f"{arm} used official validation")
        require(not parse_bool(row["held_out_test_accessed"], "held-out-test flag"),
                f"{arm} accessed held-out test")
        result.append(dict(row))
    require_equal(observed, set(spec.arms), "per-shard arms")
    return result


def _validate_history(
    path: Path, arm: str, fold: int, seed: int, spec: AggregationSpec,
) -> int:
    rows = read_csv(path, HISTORY_FIELDS)
    require_equal(len(rows), spec.epochs + 1, f"{arm} training-history rows")
    marked_epochs: list[int] = []
    expected_incumbents: list[int] = []
    incumbent_macro = -math.inf
    for epoch, row in enumerate(rows):
        require_equal(row["arm_id"], arm, f"{arm} history arm")
        require_equal(parse_int(row["outer_fold"], "history fold"), fold,
                      f"{arm} history fold")
        require_equal(parse_int(row["training_seed"], "history seed"), seed,
                      f"{arm} history seed")
        require_equal(parse_int(row["epoch"], "history epoch"), epoch,
                      f"{arm} history epoch sequence")
        require_equal(parse_int(row["optimizer_steps"], "history steps"),
                      epoch * spec.batches_per_epoch, f"{arm} history steps")
        if epoch == 0:
            require_equal(row["mean_train_loss"], "", f"{arm} epoch-zero loss")
        else:
            parse_float(row["mean_train_loss"], f"{arm} train loss")
        macro = parse_float(row["checkpoint_macro_mrr"],
                            f"{arm} checkpoint_macro_mrr")
        parse_float(row["checkpoint_nr_mrr"], f"{arm} checkpoint_nr_mrr")
        parse_float(row["checkpoint_tsr_mrr"], f"{arm} checkpoint_tsr_mrr")
        if macro > incumbent_macro:
            expected_incumbents.append(epoch)
            incumbent_macro = macro
        if parse_bool(row["is_best"], f"{arm} history is_best"):
            marked_epochs.append(epoch)
    require_equal(marked_epochs, expected_incumbents,
                  f"{arm} append-only strict-improvement history")
    require(expected_incumbents and expected_incumbents[0] == 0,
            f"{arm} epoch zero is not the initial incumbent")
    return expected_incumbents[-1]


def _pool_positive_rank(rows: Sequence[Mapping[str, str]], pool_size: int) -> int:
    require_equal(len(rows), pool_size, "raw candidate pool size")
    ranked: list[tuple[float, int, bool]] = []
    ranks: list[int] = []
    for row in rows:
        candidate_rank = parse_int(row["candidate_rank"], "candidate rank")
        ranks.append(candidate_rank)
        ranked.append((
            parse_float(row["score"], "candidate score"),
            candidate_rank,
            parse_bool(row["is_positive"], "positive flag"),
        ))
    require_equal(sorted(ranks), list(range(pool_size)), "candidate-rank inventory")
    require_equal(sum(item[2] for item in ranked), 1, "positive candidate count")
    # Frozen rule: descending score, then immutable candidate_rank ascending.
    ordered = sorted(ranked, key=lambda item: (-item[0], item[1]))
    return next(index for index, item in enumerate(ordered, start=1) if item[2])


def _validate_and_recompute_predictions(
    root: Path,
    fold: int,
    seed: int,
    spec: AggregationSpec,
    global_trial_meta: dict[str, tuple[int, str, str, str]],
    global_text_folds: dict[str, int],
    global_provenance: dict[tuple[str, str], tuple[str, str, str]],
    global_pool_binding: dict[
        str, tuple[tuple[int, str, str, bool, bool], ...]
    ],
) -> list[dict[str, object]]:
    predictions = read_csv(root / "confirmation_predictions.csv", PREDICTION_FIELDS)
    expected_targets = dict(spec.fold_counts)[fold]
    require_equal(
        len(predictions), len(spec.arms) * len(SIGNAL_CONDITIONS) * expected_targets,
        "per-shard confirmation-prediction rows",
    )
    indexed_predictions: dict[tuple[str, str, str], dict[str, str]] = {}
    shard_targets: dict[str, tuple[str, str, str]] = {}
    for row in predictions:
        arm = row["arm_id"]
        condition = row["signal_condition"]
        trial = row["trial_id"]
        require(arm in spec.arms, f"unknown prediction arm: {arm}")
        require(condition in SIGNAL_CONDITIONS,
                f"unknown prediction signal condition: {condition}")
        require_equal(parse_int(row["outer_fold"], "prediction fold"), fold,
                      "prediction fold")
        require_equal(parse_int(row["training_seed"], "prediction seed"), seed,
                      "prediction seed")
        require(row["reading_task"] in {"NR", "TSR"}, "invalid prediction task")
        require(trial and row["subject_id"], "blank prediction identity")
        require(is_sha256(row["normalized_text_sha256"]),
                "invalid prediction text identity")
        require_equal(parse_int(row["candidate_pool_size"], "prediction pool size"),
                      spec.pool_size, "prediction pool size")
        require(not parse_bool(row["scientific_decision_permitted"],
                               "prediction decision permission"),
                "prediction row prematurely permits a partial scientific decision")
        rank = parse_int(row["positive_rank"], "prediction positive rank")
        require(1 <= rank <= spec.pool_size, "invalid prediction positive rank")
        key = (arm, trial, condition)
        require(key not in indexed_predictions, f"duplicate prediction: {key}")
        indexed_predictions[key] = row
        stable = (row["reading_task"], row["subject_id"],
                  row["normalized_text_sha256"])
        if trial in shard_targets:
            require_equal(stable, shard_targets[trial], f"shard target drift: {trial}")
        else:
            shard_targets[trial] = stable
        global_stable = (fold, *stable)
        if trial in global_trial_meta:
            require_equal(global_stable, global_trial_meta[trial],
                          f"cross-shard target drift: {trial}")
        else:
            global_trial_meta[trial] = global_stable
        text = row["normalized_text_sha256"]
        if text in global_text_folds:
            require_equal(global_text_folds[text], fold,
                          f"normalized text spans folds: {text}")
        else:
            global_text_folds[text] = fold
    require_equal(len(shard_targets), expected_targets,
                  "unique confirmation targets in shard")

    recomputed: dict[tuple[str, str, str], int] = {}
    for arm in spec.arms:
        rows = read_csv(
            root / "runs" / arm / "confirmation_candidate_scores.csv",
            CONFIRMATION_SCORE_FIELDS,
        )
        require_equal(
            len(rows), len(SIGNAL_CONDITIONS) * expected_targets * spec.pool_size,
            f"{arm} raw confirmation-score rows",
        )
        pools: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            require_equal(row["arm_id"], arm, f"{arm} raw score arm")
            require_equal(parse_int(row["outer_fold"], "raw score fold"), fold,
                          f"{arm} raw score fold")
            require_equal(parse_int(row["training_seed"], "raw score seed"), seed,
                          f"{arm} raw score seed")
            trial = row["trial_id"]
            condition = row["signal_condition"]
            require(trial in shard_targets, f"raw score has unknown target: {trial}")
            require(condition in SIGNAL_CONDITIONS,
                    f"raw score has unknown signal condition: {condition}")
            stable = shard_targets[trial]
            require_equal(
                (row["reading_task"], row["subject_id"],
                 row["normalized_text_sha256"]),
                stable, f"raw target provenance: {(arm, trial, condition)}",
            )
            require(row["signal_trial_id"] and row["signal_subject_id"],
                    "blank signal provenance")
            require(is_sha256(row["signal_normalized_text_sha256"]),
                    "invalid signal text identity")
            require(is_sha256(row["candidate_normalized_text_sha256"]),
                    "invalid candidate text identity")
            require(bool(row["candidate_text_target_id"]),
                    "blank candidate text target ID")
            pools[(trial, condition)].append(row)
        require_equal(
            set(pools),
            {(trial, condition) for trial in shard_targets
             for condition in SIGNAL_CONDITIONS},
            f"{arm} raw confirmation-pool coverage",
        )
        for (trial, condition), pool in pools.items():
            ordered = sorted(pool, key=lambda row: parse_int(
                row["candidate_rank"], "candidate rank"))
            candidate_texts = [row["candidate_normalized_text_sha256"]
                               for row in ordered]
            require_equal(len(set(candidate_texts)), spec.pool_size,
                          f"duplicate candidate text: {(trial, condition)}")
            _target_task, target_subject, target_text = shard_targets[trial]
            signal_bindings = {
                (row["signal_trial_id"], row["signal_subject_id"],
                 row["signal_normalized_text_sha256"])
                for row in ordered
            }
            require_equal(len(signal_bindings), 1,
                          f"signal provenance drifts within pool: {(trial, condition)}")
            signal = next(iter(signal_bindings))
            require_equal(signal[1], target_subject,
                          f"signal subject mismatch: {(trial, condition)}")
            if condition == "correct":
                require_equal(signal, (trial, target_subject, target_text),
                              f"correct-signal provenance: {trial}")
            else:
                require(signal[0] != trial and signal[2] != target_text,
                        f"matched-wrong signal reuses target: {trial}")
            provenance_key = (trial, condition)
            if provenance_key in global_provenance:
                require_equal(signal, global_provenance[provenance_key],
                              f"donor provenance drift: {provenance_key}")
            else:
                global_provenance[provenance_key] = signal

            positives = [row for row in ordered if parse_bool(
                row["is_positive"], "positive flag")]
            donors = [row for row in ordered if parse_bool(
                row["is_designated_donor_text"], "donor flag")]
            require_equal(len(positives), 1, f"positive count: {(trial, condition)}")
            require_equal(len(donors), 1, f"donor count: {(trial, condition)}")
            require_equal(positives[0]["candidate_normalized_text_sha256"],
                          target_text, f"positive identity: {trial}")
            # The matched-wrong provenance may first appear after the correct
            # pool.  Defer its donor-identity check until both are present.
            pool_binding = tuple((
                parse_int(row["candidate_rank"], "candidate rank"),
                row["candidate_normalized_text_sha256"],
                row["candidate_text_target_id"],
                parse_bool(row["is_positive"], "positive flag"),
                parse_bool(row["is_designated_donor_text"], "donor flag"),
            ) for row in ordered)
            if trial in global_pool_binding:
                require_equal(pool_binding, global_pool_binding[trial],
                              f"candidate-pool drift: {trial}")
            else:
                global_pool_binding[trial] = pool_binding
            rank = _pool_positive_rank(ordered, spec.pool_size)
            key = (arm, trial, condition)
            require_equal(parse_int(indexed_predictions[key]["positive_rank"],
                                    "declared positive rank"),
                          rank, f"raw-score rank recomputation: {key}")
            recomputed[key] = rank

    # Now every target has a frozen matched-wrong donor binding; validate the
    # designated candidate against it for the common pool.
    for trial, pool_binding in global_pool_binding.items():
        if trial not in shard_targets:
            continue
        donor_text = global_provenance[(trial, "matched_wrong")][2]
        designated = [candidate for candidate in pool_binding if candidate[4]]
        require_equal(len(designated), 1, f"global donor count: {trial}")
        require_equal(designated[0][1], donor_text,
                      f"designated donor provenance: {trial}")
        require(not designated[0][3], f"designated donor is positive: {trial}")

    result: list[dict[str, object]] = []
    for arm in spec.arms:
        for trial in sorted(shard_targets):
            for condition in SIGNAL_CONDITIONS:
                source = indexed_predictions[(arm, trial, condition)]
                result.append({
                    **source,
                    "positive_rank": recomputed[(arm, trial, condition)],
                })
    return result


def _validate_global_matrix(
    run_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    trial_meta: Mapping[str, tuple[int, str, str, str]],
    spec: AggregationSpec,
) -> None:
    require_equal(len(run_rows), spec.expected_runs, "complete run-matrix rows")
    run_keys = {
        (str(row["arm_id"]), parse_int(row["outer_fold"], "run fold"),
         parse_int(row["training_seed"], "run seed"))
        for row in run_rows
    }
    expected_run_keys = {
        (arm, fold, seed) for fold, seed in spec.expected_shards for arm in spec.arms
    }
    require_equal(run_keys, expected_run_keys, "complete run matrix")
    require_equal(len(prediction_rows), spec.expected_predictions,
                  "complete confirmation-prediction rows")
    require_equal(len(trial_meta), spec.expected_trials,
                  "unique global confirmation trials")
    fold_counts = defaultdict(int)
    task_counts = defaultdict(int)
    for fold, task, _subject, _text in trial_meta.values():
        fold_counts[fold] += 1
        task_counts[task] += 1
    require_equal(dict(sorted(fold_counts.items())), dict(spec.fold_counts),
                  "global confirmation fold counts")
    require_equal(dict(sorted(task_counts.items())), dict(spec.task_counts),
                  "global confirmation task counts")
    keys = {
        (str(row["arm_id"]), parse_int(row["training_seed"], "prediction seed"),
         str(row["signal_condition"]), str(row["trial_id"]))
        for row in prediction_rows
    }
    require_equal(len(keys), spec.expected_predictions,
                  "unique global confirmation predictions")


def aggregate(
    shard_inputs: Sequence[ShardInput],
    launch_authorization_sha256: str,
    output_root: Path,
    *,
    frozen_registry_path: Path,
    expected_frozen_registry_sha256: str,
    protocol_root: Path,
    schedule_root: Path,
    spec: AggregationSpec = PRODUCTION_SPEC,
    contract_path: Path | None = None,
    reverify: Reverify = _default_reverify,
    decision: Decision = _default_decision,
) -> dict[str, Any]:
    require(is_sha256(launch_authorization_sha256),
            "invalid separately supplied launch-authorization SHA256")
    contract = contract_path or Path(__file__).with_name(
        "task_segmented_full_execution_contract.json")
    require_equal(sha256(contract), FULL_EXECUTION_CONTRACT_SHA256,
                  "local full-execution contract SHA256")
    require_equal(len(shard_inputs), len(spec.expected_shards),
                  "exact full-shard input count")
    require(not output_root.exists() and not output_root.is_symlink(),
            f"output root already exists: {output_root}")

    require(not protocol_root.is_symlink(),
            f"preserved protocol root is a symlink: {protocol_root}")
    protocol_root = protocol_root.resolve(strict=True)
    require(protocol_root.is_dir() and not protocol_root.is_symlink(),
            f"invalid preserved protocol root: {protocol_root}")
    require(not schedule_root.is_symlink(),
            f"preserved schedule root is a symlink: {schedule_root}")
    schedule_root = schedule_root.resolve(strict=True)
    require(schedule_root.is_dir() and not schedule_root.is_symlink(),
            f"invalid preserved schedule root: {schedule_root}")
    require(protocol_root != schedule_root,
            "preserved protocol and schedule roots must be distinct")

    require(not frozen_registry_path.is_symlink(),
            f"frozen registry path is a symlink: {frozen_registry_path}")
    registry_path = frozen_registry_path.resolve(strict=True)
    require(registry_path.is_file() and not registry_path.is_symlink(),
            f"invalid out-of-band frozen registry: {registry_path}")
    registry_by_unit = _validate_frozen_registry(
        registry_path, expected_frozen_registry_sha256,
        launch_authorization_sha256, spec,
    )

    seen_roots: set[Path] = set()
    seen_reports: set[Path] = set()
    seen_preserved_sources: set[str] = set()
    seen_manifest_hashes: set[str] = set()
    seen_runtime_fingerprint_hashes: set[str] = set()
    seen_units: set[tuple[int, int]] = set()
    reports_by_unit: dict[tuple[int, int], dict[str, Any]] = {}
    roots_by_unit: dict[tuple[int, int], Path] = {}
    for shard_input in shard_inputs:
        root = shard_input.artifact_root.resolve(strict=True)
        report_path = shard_input.verification_report.resolve(strict=True)
        require(root.is_dir() and not root.is_symlink(),
                f"invalid full-shard root: {root}")
        require(report_path.is_file() and not report_path.is_symlink(),
                f"invalid independent verification report: {report_path}")
        require(not registry_path.is_relative_to(root),
                f"frozen registry is not out-of-band from shard root: {root}")
        require(report_path != registry_path,
                "verification report cannot also be the frozen registry")
        require(root not in seen_roots, f"duplicate shard root: {root}")
        require(report_path not in seen_reports,
                f"duplicate verification report: {report_path}")
        seen_roots.add(root)
        seen_reports.add(report_path)
        report = read_json(report_path)
        unit = _validate_verification_report(
            report, launch_authorization_sha256, spec,
        )
        registry_entry = registry_by_unit[unit]
        require_equal(report["preserved_source_id"],
                      registry_entry["preserved_source_id"],
                      f"registry/report preserved source for shard {unit}")
        require_equal(report["full_shard_manifest_sha256"],
                      registry_entry["full_shard_manifest_sha256"],
                      f"registry/report manifest for shard {unit}")
        require_equal(sha256(report_path),
                      registry_entry["verification_report_sha256"],
                      f"registry verification-report file for shard {unit}")
        preserved_source = str(report["preserved_source_id"])
        manifest_hash = str(report["full_shard_manifest_sha256"])
        require(preserved_source not in seen_preserved_sources,
                f"duplicate preserved shard source: {preserved_source}")
        require(manifest_hash not in seen_manifest_hashes,
                f"duplicate full-shard manifest: {manifest_hash}")
        seen_preserved_sources.add(preserved_source)
        seen_manifest_hashes.add(manifest_hash)
        seen_runtime_fingerprint_hashes.add(
            str(report["runtime_fingerprint_sha256"])
        )
        require(unit not in seen_units, f"duplicate fold/seed shard: {unit}")
        seen_units.add(unit)
        rerun = reverify(
            root,
            report,
            launch_authorization_sha256,
            spec,
            protocol_root,
            schedule_root,
        )
        require_equal(rerun, report,
                      f"independent re-verification report for shard {unit}")
        reports_by_unit[unit] = report
        roots_by_unit[unit] = root
    require_equal(seen_units, spec.expected_shards, "frozen fold/seed shard matrix")
    require_equal(
        len(seen_runtime_fingerprint_hashes),
        1,
        "one normalized runtime fingerprint across all full shards",
    )
    runtime_fingerprint_sha256 = next(iter(seen_runtime_fingerprint_hashes))

    all_runs: list[dict[str, str]] = []
    all_predictions: list[dict[str, object]] = []
    trial_meta: dict[str, tuple[int, str, str, str]] = {}
    text_folds: dict[str, int] = {}
    provenance: dict[tuple[str, str], tuple[str, str, str]] = {}
    pools: dict[str, tuple[tuple[int, str, str, bool, bool], ...]] = {}
    for unit in sorted(spec.expected_shards):
        fold, seed = unit
        root = roots_by_unit[unit]
        run_rows = read_csv(root / "run_manifest.csv", RUN_FIELDS)
        all_runs.extend(_validate_run_rows(run_rows, fold, seed, spec))
        for arm in spec.arms:
            final_incumbent_epoch = _validate_history(
                root / "runs" / arm / "training_history.csv",
                arm, fold, seed, spec,
            )
            require_equal(
                final_incumbent_epoch,
                reports_by_unit[unit]["best_epochs"][arm],
                f"{arm} final incumbent epoch in shard {unit}",
            )
        all_predictions.extend(_validate_and_recompute_predictions(
            root, fold, seed, spec, trial_meta, text_folds, provenance, pools,
        ))
    _validate_global_matrix(all_runs, all_predictions, trial_meta, spec)

    all_runs.sort(key=lambda row: (
        parse_int(row["outer_fold"], "run fold"),
        parse_int(row["training_seed"], "run seed"),
        spec.arms.index(row["arm_id"]),
    ))
    all_predictions.sort(key=lambda row: (
        parse_int(row["outer_fold"], "prediction fold"),
        parse_int(row["training_seed"], "prediction seed"),
        spec.arms.index(str(row["arm_id"])),
        str(row["trial_id"]),
        SIGNAL_CONDITIONS.index(str(row["signal_condition"])),
    ))

    # Individual shards are deliberately non-scientific.  Permission is
    # promoted only in new complete-matrix copies after every integrity gate
    # above has passed; the source shard rows are never mutated.
    decision_runs = [
        {**row, "scientific_decision_permitted": "true"}
        for row in all_runs
    ]
    decision_predictions = [
        {**row, "scientific_decision_permitted": "true"}
        for row in all_predictions
    ]
    permission_transition = {
        "partial_shard_scientific_decision_permitted": False,
        "complete_matrix_scientific_decision_permitted": True,
        "activation_gate": (
            "all frozen shards independently verified and complete-matrix "
            "integrity passed"
        ),
        "partial_run_rows_sha256": canonical_sha256(all_runs),
        "partial_prediction_rows_sha256": canonical_sha256(all_predictions),
        "complete_matrix_run_rows_sha256": canonical_sha256(decision_runs),
        "complete_matrix_prediction_rows_sha256": canonical_sha256(
            decision_predictions),
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.tmp-", dir=output_root.parent,
    ))
    published = False
    try:
        run_path = temporary / "run_manifest.csv"
        prediction_path = temporary / "confirmation_predictions.csv"
        write_csv(run_path, RUN_FIELDS, decision_runs)
        write_csv(prediction_path, PREDICTION_FIELDS, decision_predictions)

        # The frozen decision is invoked only here, after the complete matrix
        # and its raw-score reconstruction have passed every integrity gate.
        scientific = decision(decision_runs, decision_predictions)
        require(isinstance(scientific, dict), "scientific decision is not an object")
        require(scientific.get("status") in {"pass", "fail"},
                "scientific decision has invalid status")
        require_equal(scientific.get("scientific_decision_permitted"), True,
                      "scientific decision permission")
        require_equal(scientific.get("held_out_test_accessed"), False,
                      "scientific decision held-out-test access")
        expected_continuation = (
            "continue_p4b" if scientific["status"] == "pass"
            else "stop_p4b_permanently"
        )
        require_equal(scientific.get("continuation_decision"),
                      expected_continuation,
                      "scientific continuation decision")
        decision_path = temporary / "scientific_decision.json"
        write_json(decision_path, scientific)

        shard_bindings = [
            dict(registry_by_unit[unit]) for unit in sorted(spec.expected_shards)
        ]
        integrity = {
            "schema_version": 1,
            "status": "pass",
            "integrity_status": "pass",
            "scientific_decision_status": scientific["status"],
            "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
            "launch_authorization_sha256": launch_authorization_sha256,
            "frozen_shard_registry_sha256": expected_frozen_registry_sha256,
            "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
            "verified_shards": len(shard_bindings),
            "run_manifest_rows": len(all_runs),
            "confirmation_prediction_rows": len(all_predictions),
            "unique_confirmation_trials": len(trial_meta),
            "candidate_pool_binding_sha256": canonical_sha256([
                [trial, list(binding)] for trial, binding in sorted(pools.items())
            ]),
            "signal_provenance_binding_sha256": canonical_sha256([
                [trial, condition, list(binding)]
                for (trial, condition), binding in sorted(provenance.items())
            ]),
            "official_validation_used_for_confirmation": False,
            "held_out_test_accessed": False,
            "checkpoint_deserialized_by_aggregator": False,
            "scientific_decision_permission_transition": permission_transition,
            "shards": shard_bindings,
        }
        integrity_path = temporary / "integrity_report.json"
        write_json(integrity_path, integrity)
        artifacts = {
            name: sha256(temporary / name)
            for name in (
                "run_manifest.csv", "confirmation_predictions.csv",
                "integrity_report.json", "scientific_decision.json",
            )
        }
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "integrity_status": "pass",
            "scientific_decision_status": scientific["status"],
            "continuation_decision": scientific.get("continuation_decision"),
            "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
            "launch_authorization_sha256": launch_authorization_sha256,
            "frozen_shard_registry_sha256": expected_frozen_registry_sha256,
            "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
            "verified_shard_count": len(shard_bindings),
            "run_manifest_rows": len(all_runs),
            "confirmation_prediction_rows": len(all_predictions),
            "scientific_decision_permission_transition": permission_transition,
            "artifact_sha256": artifacts,
        }
        write_json(temporary / "full_matrix_manifest.json", manifest)
        os.replace(temporary, output_root)
        published = True
        return {
            **manifest,
            "full_matrix_manifest_sha256": sha256(
                output_root / "full_matrix_manifest.json"),
            "output_root": str(output_root),
        }
    finally:
        if not published and temporary.exists():
            # The temporary directory contains only newly produced aggregate
            # files.  Remove individual regular files, then the directory,
            # without traversing or following external paths.
            for path in temporary.iterdir():
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            temporary.rmdir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard", action="append", nargs=2, metavar=("ROOT", "VERIFICATION_REPORT"),
        required=True,
        help="repeat exactly 15 times; the report must be independently produced",
    )
    parser.add_argument("--launch-authorization-sha256", required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--frozen-shard-registry", type=Path, required=True)
    parser.add_argument(
        "--expected-frozen-shard-registry-sha256", required=True,
        help="out-of-band digest; never infer this value from the registry file",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = aggregate(
        [ShardInput(Path(root), Path(report)) for root, report in args.shard],
        args.launch_authorization_sha256,
        args.output_root,
        frozen_registry_path=args.frozen_shard_registry,
        expected_frozen_registry_sha256=(
            args.expected_frozen_shard_registry_sha256),
        protocol_root=args.protocol_root,
        schedule_root=args.schedule_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("TASK-SEGMENTED COMPLETE FULL MATRIX AGGREGATION: PASS")


if __name__ == "__main__":
    main()
