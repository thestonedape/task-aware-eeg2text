"""Run frozen GLIM factor probes under the predeclared recoverability contract.

The runner fits only development probes.  It never reads canonical test rows, never
uses target labels as inputs, and evaluates matched-wrong/zero/Gaussian vectors
through the model fitted on correct training vectors.  Subject-held-out refits are
performed only for factors whose ordinary correct-vector point estimates beat both
admission comparators; factors failing that necessary condition are retained as
null results without unnecessary refits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import ConstantInputWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning


warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=r"Label not .* is present in all training examples\.")
warnings.filterwarnings("ignore", message=r"No positive class found in y_true.*")
warnings.filterwarnings("ignore", message=r"y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message=r"A single label was found in .*confusion matrix.*")


EXPECTED_CHECKS = {
    "held_out_test_accessed": False,
    "target_identity_preserved": True,
    "matched_wrong_changes_signal_only": True,
    "zero_and_gaussian_keep_target_metadata": True,
    "gaussian_uses_training_statistics_only": True,
    "checkpoint_and_source_pinned": True,
    "vectors_finite": True,
}
CONDITIONS = (
    "frozen_glim_correct",
    "metadata_only",
    "task_only",
    "frozen_glim_plus_explicit_metadata",
    "frozen_glim_matched_wrong_eeg",
    "frozen_glim_zero_eeg",
    "frozen_glim_gaussian_eeg",
)
VECTOR_CONDITIONS = {
    "frozen_glim_correct": "correct_val",
    "frozen_glim_matched_wrong_eeg": "matched_wrong_val",
    "frozen_glim_zero_eeg": "zero_val",
    "frozen_glim_gaussian_eeg": "gaussian_val",
}


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class VectorStore:
    def __init__(
        self,
        root: Path,
        expected_index_sha256: str | None = None,
        expected_vector_index_sha256: str | None = None,
    ) -> None:
        self.root = root
        manifest_path = root / "vector_manifest.json"
        index_path = root / "vector_index.csv"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "pass" or self.manifest.get("run_mode") != "full_development":
            raise ValueError("vector artifact is not a passed full-development extraction")
        if self.manifest.get("checks") != EXPECTED_CHECKS:
            raise ValueError(f"unexpected vector safety checks: {self.manifest.get('checks')}")
        if expected_index_sha256 and self.manifest.get("source_index_sha256") != expected_index_sha256:
            raise ValueError("vector artifact does not match the frozen canonical index")
        actual_index = sha256(index_path)
        if actual_index != self.manifest.get("vector_index_sha256"):
            raise ValueError("vector_index.csv hash mismatch")
        if expected_vector_index_sha256 and actual_index != expected_vector_index_sha256:
            raise ValueError("vector index does not match the requested immutable artifact")
        self.rows = read_csv(index_path)
        if len(self.rows) != sum(int(value) for value in self.manifest["condition_counts"].values()):
            raise ValueError("vector index cardinality mismatch")
        if any(row["phase"] == "test" for row in self.rows):
            raise AssertionError("held-out test vector entered the probe store")
        self.lookup: dict[tuple[str, str], dict[str, str]] = {}
        for row in self.rows:
            key = (row["condition"], row["target_trial_id"])
            if key in self.lookup:
                raise ValueError(f"duplicate vector identity: {key}")
            self.lookup[key] = row
        self.chunk_hashes: dict[str, str] = {}
        for item in self.manifest["chunks"]:
            # Schema v1 extraction manifests identify chunks by condition and
            # chunk_number; vector_index.csv carries the concrete vector_file.
            # Accept a future explicit path while remaining compatible with the
            # already-frozen schema-v1 artifact.
            relative = item.get("vector_file")
            if relative is None:
                condition = item.get("condition")
                chunk_number = item.get("chunk_number")
                if not isinstance(condition, str) or not condition:
                    raise ValueError(f"invalid vector chunk condition: {condition!r}")
                if isinstance(chunk_number, bool) or not isinstance(chunk_number, int):
                    raise ValueError(f"invalid vector chunk number: {chunk_number!r}")
                relative = f"vectors/{condition}_{chunk_number:05d}.npz"
            relative = str(relative)
            if relative in self.chunk_hashes:
                raise ValueError(f"duplicate vector chunk manifest entry: {relative}")
            self.chunk_hashes[relative] = str(item["vector_npz_sha256"])
        indexed_files = {row["vector_file"] for row in self.rows}
        if indexed_files != set(self.chunk_hashes):
            raise ValueError("vector index and chunk manifest file sets differ")
        self.cache: dict[str, np.ndarray] = {}

    def _chunk(self, relative: str) -> np.ndarray:
        if relative not in self.cache:
            path = self.root / relative
            expected = self.chunk_hashes.get(relative)
            if expected is None or sha256(path) != expected:
                raise ValueError(f"vector chunk hash mismatch: {relative}")
            with np.load(path, allow_pickle=False) as archive:
                vectors = np.asarray(archive["vectors"], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != int(self.manifest["vector_dim"]):
                raise ValueError(f"invalid vector chunk shape: {relative} {vectors.shape}")
            if not np.isfinite(vectors).all():
                raise ValueError(f"non-finite vector chunk: {relative}")
            self.cache[relative] = vectors
        return self.cache[relative]

    def matrix(self, condition: str, trial_ids: list[str]) -> np.ndarray:
        output = np.empty((len(trial_ids), int(self.manifest["vector_dim"])), dtype=np.float32)
        for index, trial_id in enumerate(trial_ids):
            row = self.lookup.get((condition, trial_id))
            if row is None:
                raise KeyError(f"missing {condition} vector for {trial_id}")
            vectors = self._chunk(row["vector_file"])
            output[index] = vectors[int(row["vector_offset"])]
        return output


def target_arrays(
    factor_id: str, spec: dict[str, Any], rows: list[dict[str, str]]
) -> tuple[np.ndarray, list[str]]:
    target_type = str(spec["target_type"])
    ontology = [str(item) for item in spec.get("ontology", [])]
    if target_type == "regression":
        return np.asarray([float(row["target_value"]) for row in rows], dtype=np.float64), ontology
    if target_type == "multiclass":
        label_to_index = {label: index for index, label in enumerate(ontology)}
        try:
            return np.asarray([label_to_index[row["target_value"]] for row in rows], dtype=np.int64), ontology
        except KeyError as error:
            raise ValueError(f"{factor_id}: target outside ontology: {error}") from error
    if target_type == "multilabel":
        label_to_index = {label: index for index, label in enumerate(ontology)}
        target = np.zeros((len(rows), len(ontology)), dtype=np.int8)
        for row_index, row in enumerate(rows):
            atoms = [item for item in row["target_atoms"].split(";") if item and item != "NO-RELATION"]
            for atom in atoms:
                if atom not in label_to_index:
                    raise ValueError(f"{factor_id}: target atom outside ontology: {atom}")
                target[row_index, label_to_index[atom]] = 1
        return target, ontology
    raise KeyError(target_type)


def metadata_dicts(
    rows: list[dict[str, str]],
    length_by_trial: dict[str, float],
    factor_id: str,
    task_only: bool,
) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    for row in rows:
        if task_only:
            output.append({f"reading_task={row['reading_task']}": 1.0})
            continue
        values = {
            f"dataset_version={row['dataset_version']}": 1.0,
            f"reading_task={row['reading_task']}": 1.0,
            f"subject_id={row['subject_id']}": 1.0,
            f"cohort={row['cohort']}": 1.0,
        }
        if factor_id != "length_words_whitespace_v1":
            values["length_words_whitespace_v1"] = float(length_by_trial[row["trial_id"]])
        output.append(values)
    return output


def build_model(target_type: str, parameter: float, seed: int):
    if target_type in {"multiclass", "multilabel"}:
        estimator = LogisticRegression(
            C=float(parameter), penalty="l2", solver="liblinear", class_weight="balanced",
            max_iter=3000, random_state=seed,
        )
        if target_type == "multilabel":
            estimator = OneVsRestClassifier(estimator)
        return make_pipeline(StandardScaler(), estimator)
    if target_type == "regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(parameter)))
    raise KeyError(target_type)


def predict_scores(model, target_type: str, matrix: np.ndarray, class_count: int) -> np.ndarray:
    if target_type == "regression":
        return np.asarray(model.predict(matrix), dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    if target_type == "multiclass" and probabilities.shape[1] != class_count:
        # One or more ontology classes can be absent from a small subject fold.
        fitted = model[-1]
        classes = np.asarray(getattr(fitted, "classes_", np.arange(probabilities.shape[1])), dtype=int)
        expanded = np.zeros((len(matrix), class_count), dtype=np.float64)
        expanded[:, classes] = probabilities
        probabilities = expanded
    return probabilities


def hard_predictions(scores: np.ndarray, target_type: str) -> np.ndarray:
    if target_type == "multiclass":
        return scores.argmax(axis=1)
    if target_type == "multilabel":
        return (scores >= 0.5).astype(np.int8)
    return scores


def metrics(y: np.ndarray, scores: np.ndarray, target_type: str, class_count: int) -> dict[str, float]:
    predicted = hard_predictions(scores, target_type)
    if target_type == "multiclass":
        labels = list(range(class_count))
        output = {
            "macro_f1": f1_score(y, predicted, labels=labels, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "accuracy": accuracy_score(y, predicted),
        }
        try:
            output["ovr_auroc_when_defined"] = roc_auc_score(
                y, scores, labels=labels, multi_class="ovr", average="macro"
            )
        except ValueError:
            output["ovr_auroc_when_defined"] = math.nan
        return {key: float(value) for key, value in output.items()}
    if target_type == "multilabel":
        output = {
            "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
            "micro_f1": f1_score(y, predicted, average="micro", zero_division=0),
            "exact_match": accuracy_score(y, predicted),
        }
        try:
            output["macro_average_precision"] = average_precision_score(y, scores, average="macro")
        except ValueError:
            output["macro_average_precision"] = math.nan
        return {key: float(value) for key, value in output.items()}
    correlation = spearmanr(y, predicted).statistic
    return {
        "mae": float(mean_absolute_error(y, predicted)),
        "spearman_r": float(correlation) if np.isfinite(correlation) else math.nan,
        "r2": float(r2_score(y, predicted)),
    }


def utility(y: np.ndarray, predicted: np.ndarray, target_type: str, class_count: int, weights=None) -> float:
    if weights is None:
        weights = np.ones(len(y), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        return math.nan
    if target_type == "regression":
        return -float(np.sum(weights * np.abs(y - predicted)) / weights.sum())
    if target_type == "multiclass":
        values = []
        for label in range(class_count):
            truth = y == label
            guess = predicted == label
            tp = weights[truth & guess].sum()
            fp = weights[~truth & guess].sum()
            fn = weights[truth & ~guess].sum()
            denominator = 2 * tp + fp + fn
            values.append(0.0 if denominator == 0 else 2 * tp / denominator)
        return float(np.mean(values))
    values = []
    for label in range(class_count):
        truth = y[:, label].astype(bool)
        guess = predicted[:, label].astype(bool)
        tp = weights[truth & guess].sum()
        fp = weights[~truth & guess].sum()
        fn = weights[truth & ~guess].sum()
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def primary_metric_value(metric_values: dict[str, float], target_type: str) -> float:
    return -metric_values["mae"] if target_type == "regression" else metric_values["macro_f1"]


def select_parameter(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: list[str],
    target_type: str,
    grid: list[float],
    seed: int,
    class_count: int,
) -> tuple[float, float]:
    unique_groups = len(set(groups))
    splits = min(5, unique_groups)
    if splits < 2:
        raise ValueError("train-only group CV needs at least two text groups")
    cv = GroupKFold(n_splits=splits)
    best_parameter = float(grid[0])
    best_score = -math.inf
    groups_array = np.asarray(groups)
    for parameter in grid:
        fold_scores = []
        for train_index, validation_index in cv.split(matrix, groups=groups_array):
            model = build_model(target_type, float(parameter), seed)
            model.fit(matrix[train_index], target[train_index])
            scores = predict_scores(model, target_type, matrix[validation_index], class_count)
            fold_scores.append(
                primary_metric_value(metrics(target[validation_index], scores, target_type, class_count), target_type)
            )
        score = float(np.mean(fold_scores))
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and float(parameter) < best_parameter):
            best_parameter, best_score = float(parameter), score
    return best_parameter, best_score


def bootstrap_weights(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(labels), return_inverse=True)
    counts = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
    return counts[inverse].astype(np.float64)


def paired_bootstrap(
    y: np.ndarray,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    target_type: str,
    class_count: int,
    subjects: list[str],
    texts: list[str],
    cluster: str,
    replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        if cluster == "subject_id":
            weights = bootstrap_weights(subjects, rng)
        elif cluster == "normalized_text_sha256":
            weights = bootstrap_weights(texts, rng)
        elif cluster == "two_way_subject_by_text":
            weights = bootstrap_weights(subjects, rng) * bootstrap_weights(texts, rng)
            if weights.sum() == 0:
                weights = np.ones(len(y), dtype=np.float64)
        else:
            raise KeyError(cluster)
        draws[index] = utility(y, predicted_a, target_type, class_count, weights) - utility(
            y, predicted_b, target_type, class_count, weights
        )
    return draws


def holm_adjust(p_values: list[float]) -> tuple[list[float], list[int]]:
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * len(p_values)
    ranks = [0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for position, index in enumerate(order):
        running = max(running, min(1.0, (total - position) * p_values[index]))
        adjusted[index] = running
        ranks[index] = position + 1
    return adjusted, ranks


def average_scores(items: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(items, axis=0), axis=0)


def run(
    vector_root: Path,
    protocol_root: Path,
    output_root: Path,
    expected_index_sha256: str | None = None,
    expected_vector_index_sha256: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    registry_path = protocol_root / "recoverability_registry.json"
    rows_path = protocol_root / "recoverability_rows.csv"
    coverage_path = protocol_root / "factor_fold_coverage.csv"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("held_out_test_accessed") is not False:
        raise ValueError("recoverability registry does not seal the held-out test")
    if expected_index_sha256 and registry.get("source_index_sha256") != expected_index_sha256:
        raise ValueError("recoverability registry index mismatch")
    store = VectorStore(vector_root, expected_index_sha256, expected_vector_index_sha256)
    if store.manifest["source_index_sha256"] != registry["source_index_sha256"]:
        raise ValueError("vector and recoverability source indexes differ")
    factor_rows = read_csv(rows_path)
    if any(row["split"] not in {"train", "val"} for row in factor_rows):
        raise AssertionError("held-out row entered recoverability rows")
    if any(row["oracle_policy"] != "supervision_only_never_probe_input" for row in factor_rows):
        raise ValueError("target oracle policy mismatch")
    coverage = read_csv(coverage_path)
    included_folds = {
        (row["factor_id"], row["fold_id"])
        for row in coverage
        if row["fold_mode"] == "subject_held_out" and int(row["fold_included"]) == 1
    }
    length_by_trial = {
        row["trial_id"]: float(row["target_value"])
        for row in factor_rows
        if row["factor_id"] == "length_words_whitespace_v1"
    }
    seeds = [int(value) for value in registry["seeds"]]
    bootstrap_replicates = int(registry["uncertainty"]["bootstrap_replicates"])
    factors = list(registry["factors"])
    if smoke:
        seeds = seeds[:1]
        bootstrap_replicates = min(25, bootstrap_replicates)
        factors = ["length_words_whitespace_v1"]

    output_root.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    factor_state: dict[str, dict[str, Any]] = {}

    for factor_id in factors:
        spec = registry["factors"][factor_id]
        target_type = str(spec["target_type"])
        train_rows = sorted(
            [row for row in factor_rows if row["factor_id"] == factor_id and row["split"] == "train"],
            key=lambda row: row["trial_id"],
        )
        val_rows = sorted(
            [row for row in factor_rows if row["factor_id"] == factor_id and row["split"] == "val"],
            key=lambda row: row["trial_id"],
        )
        if smoke:
            train_rows = train_rows[: min(256, len(train_rows))]
            val_rows = val_rows[: min(64, len(val_rows))]
        if len(val_rows) < 1:
            raise ValueError(f"{factor_id}: no validation rows")
        train_ids = [row["trial_id"] for row in train_rows]
        val_ids = [row["trial_id"] for row in val_rows]
        train_target, ontology = target_arrays(factor_id, spec, train_rows)
        val_target, _ = target_arrays(factor_id, spec, val_rows)
        class_count = len(ontology) if target_type != "regression" else 0
        groups = [row["text_uid"] for row in train_rows]
        correct_train = store.matrix("correct_train", train_ids)
        val_vectors = {
            condition: store.matrix(vector_condition, val_ids)
            for condition, vector_condition in VECTOR_CONDITIONS.items()
        }

        vectorizer = DictVectorizer(sparse=False)
        metadata_train = vectorizer.fit_transform(
            metadata_dicts(train_rows, length_by_trial, factor_id, task_only=False)
        ).astype(np.float32)
        metadata_val = vectorizer.transform(
            metadata_dicts(val_rows, length_by_trial, factor_id, task_only=False)
        ).astype(np.float32)
        task_vectorizer = DictVectorizer(sparse=False)
        task_train = task_vectorizer.fit_transform(
            metadata_dicts(train_rows, length_by_trial, factor_id, task_only=True)
        ).astype(np.float32)
        task_val = task_vectorizer.transform(
            metadata_dicts(val_rows, length_by_trial, factor_id, task_only=True)
        ).astype(np.float32)
        plus_train = np.concatenate([correct_train, metadata_train], axis=1)
        plus_val = np.concatenate([val_vectors["frozen_glim_correct"], metadata_val], axis=1)
        grid = [float(value) for value in (
            registry["probe_models"][target_type]["grid_alpha"]
            if target_type == "regression"
            else registry["probe_models"][target_type]["grid_C"]
        )]
        scores_by_condition: dict[str, list[np.ndarray]] = defaultdict(list)

        # Hyperparameters are selected once with the first declared seed.  The
        # selected value is then refit under every declared seed; validation is
        # never used for selection and identical CV work is not repeated five times.
        correct_parameter, correct_cv = select_parameter(
            correct_train, train_target, groups, target_type, grid, seeds[0], class_count
        )
        metadata_parameter, metadata_cv = select_parameter(
            metadata_train, train_target, groups, target_type, grid, seeds[0], class_count
        )
        task_parameter, task_cv = select_parameter(
            task_train, train_target, groups, target_type, grid, seeds[0], class_count
        )
        plus_parameter, plus_cv = select_parameter(
            plus_train, train_target, groups, target_type, grid, seeds[0], class_count
        )

        for seed in seeds:
            correct_model = build_model(target_type, correct_parameter, seed)
            correct_model.fit(correct_train, train_target)
            metadata_model = build_model(target_type, metadata_parameter, seed)
            metadata_model.fit(metadata_train, train_target)
            task_model = build_model(target_type, task_parameter, seed)
            task_model.fit(task_train, train_target)
            plus_model = build_model(target_type, plus_parameter, seed)
            plus_model.fit(plus_train, train_target)
            for condition, parameter, cv_score in (
                ("frozen_glim_correct", correct_parameter, correct_cv),
                ("metadata_only", metadata_parameter, metadata_cv),
                ("task_only", task_parameter, task_cv),
                ("frozen_glim_plus_explicit_metadata", plus_parameter, plus_cv),
            ):
                selection_rows.append({
                    "factor_id": factor_id, "evaluation_mode": "ordinary", "fold_id": "ordinary",
                    "condition": condition, "seed": seed, "selected_parameter": parameter,
                    "train_only_group_cv_utility": cv_score,
                })
            current_scores = {
                "frozen_glim_correct": predict_scores(
                    correct_model, target_type, val_vectors["frozen_glim_correct"], class_count
                ),
                "frozen_glim_matched_wrong_eeg": predict_scores(
                    correct_model, target_type, val_vectors["frozen_glim_matched_wrong_eeg"], class_count
                ),
                "frozen_glim_zero_eeg": predict_scores(
                    correct_model, target_type, val_vectors["frozen_glim_zero_eeg"], class_count
                ),
                "frozen_glim_gaussian_eeg": predict_scores(
                    correct_model, target_type, val_vectors["frozen_glim_gaussian_eeg"], class_count
                ),
                "metadata_only": predict_scores(metadata_model, target_type, metadata_val, class_count),
                "task_only": predict_scores(task_model, target_type, task_val, class_count),
                "frozen_glim_plus_explicit_metadata": predict_scores(
                    plus_model, target_type, plus_val, class_count
                ),
            }
            for condition, scores in current_scores.items():
                scores_by_condition[condition].append(scores)
                for metric, value in metrics(val_target, scores, target_type, class_count).items():
                    metric_rows.append({
                        "factor_id": factor_id, "evaluation_mode": "ordinary", "fold_id": "ordinary",
                        "condition": condition, "seed": seed, "metric": metric, "value": value,
                        "rows": len(val_rows), "subjects": len({row['subject_id'] for row in val_rows}),
                    })

        averaged = {condition: average_scores(values) for condition, values in scores_by_condition.items()}
        hard = {condition: hard_predictions(scores, target_type) for condition, scores in averaged.items()}
        ordinary_utility = {
            condition: utility(val_target, predicted, target_type, class_count)
            for condition, predicted in hard.items()
        }
        for condition, predicted in hard.items():
            for index, row in enumerate(val_rows):
                if target_type == "multiclass":
                    target_text = ontology[int(val_target[index])]
                    prediction_text = ontology[int(predicted[index])]
                elif target_type == "multilabel":
                    target_text = ";".join(
                        ontology[label] for label in range(class_count) if int(val_target[index, label])
                    ) or "NO-RELATION"
                    prediction_text = ";".join(
                        ontology[label] for label in range(class_count) if int(predicted[index, label])
                    ) or "NO-RELATION"
                else:
                    target_text = str(float(val_target[index]))
                    prediction_text = str(float(predicted[index]))
                prediction_rows.append({
                    "factor_id": factor_id, "evaluation_mode": "ordinary", "fold_id": "ordinary",
                    "condition": condition, "trial_id": row["trial_id"], "subject_id": row["subject_id"],
                    "normalized_text_sha256": row["normalized_text_sha256"],
                    "target": target_text, "prediction": prediction_text,
                })
        factor_state[factor_id] = {
            "spec": spec, "target_type": target_type, "ontology": ontology,
            "val_rows": val_rows, "val_target": val_target, "hard": hard,
            "ordinary_utility": ordinary_utility, "train_rows": train_rows,
            "train_target": train_target, "groups": groups, "grid": grid,
            "correct_train": correct_train, "metadata_train": metadata_train,
            "metadata_vectorizer": vectorizer,
        }

    # Bootstrap both planned contrasts.  In smoke mode this validates mechanics only.
    contrast_targets = {
        "frozen_glim_correct_minus_metadata_only": "metadata_only",
        "frozen_glim_correct_minus_frozen_glim_matched_wrong_eeg": "frozen_glim_matched_wrong_eeg",
    }
    bootstrap_state: list[dict[str, Any]] = []
    for factor_id, state in factor_state.items():
        val_rows = state["val_rows"]
        subjects = [row["subject_id"] for row in val_rows]
        texts = [row["normalized_text_sha256"] for row in val_rows]
        correct = state["hard"]["frozen_glim_correct"]
        for contrast, comparator in contrast_targets.items():
            estimate = state["ordinary_utility"]["frozen_glim_correct"] - state["ordinary_utility"][comparator]
            for cluster_index, cluster in enumerate(registry["uncertainty"]["clusters"]):
                draws = paired_bootstrap(
                    state["val_target"], correct, state["hard"][comparator], state["target_type"],
                    len(state["ontology"]), subjects, texts, str(cluster), bootstrap_replicates,
                    seeds[0] + 1000 * cluster_index + len(bootstrap_state),
                )
                bootstrap_state.append({
                    "factor_id": factor_id, "contrast": contrast, "cluster": str(cluster),
                    "estimate": estimate, "draws": draws,
                    "p_one_sided": float((1 + np.sum(draws <= 0)) / (len(draws) + 1)),
                })

    contrast_rows: list[dict[str, Any]] = []
    for contrast in contrast_targets:
        for cluster in registry["uncertainty"]["clusters"]:
            family = [
                item for item in bootstrap_state
                if item["contrast"] == contrast and item["cluster"] == cluster
            ]
            adjusted, ranks = holm_adjust([float(item["p_one_sided"]) for item in family])
            total = len(family)
            for item, holm_p, rank in zip(family, adjusted, ranks):
                alpha = 0.05 / (total - rank + 1)
                draws = item["draws"]
                contrast_rows.append({
                    "factor_id": item["factor_id"], "contrast": contrast, "cluster": cluster,
                    "utility_delta": item["estimate"],
                    "ci95_lower": float(np.quantile(draws, 0.05)),
                    "ci95_upper": float(np.quantile(draws, 0.95)),
                    "p_one_sided": item["p_one_sided"], "holm_adjusted_p": holm_p,
                    "holm_rank": rank, "holm_rank_alpha": alpha,
                    "holm_adjusted_lower": float(np.quantile(draws, alpha)),
                    "bootstrap_replicates": len(draws),
                })

    # Necessary ordinary point-gain screen before expensive per-subject refits.
    candidates = []
    if not smoke:
        for factor_id, state in factor_state.items():
            correct = state["ordinary_utility"]["frozen_glim_correct"]
            if (
                correct > state["ordinary_utility"]["metadata_only"]
                and correct > state["ordinary_utility"]["frozen_glim_matched_wrong_eeg"]
            ):
                candidates.append(factor_id)

    subject_deltas: dict[str, dict[str, float]] = {}
    for factor_id in candidates:
        state = factor_state[factor_id]
        spec = state["spec"]
        target_type = state["target_type"]
        ontology = state["ontology"]
        class_count = len(ontology)
        train_rows = state["train_rows"]
        val_rows = state["val_rows"]
        all_train_target = state["train_target"]
        train_index = {row["trial_id"]: index for index, row in enumerate(train_rows)}
        val_target = state["val_target"]
        val_index = {row["trial_id"]: index for index, row in enumerate(val_rows)}
        fold_scores: dict[str, list[np.ndarray]] = defaultdict(list)
        covered_indices: set[int] = set()
        for subject in sorted({row["subject_id"] for row in val_rows}):
            fold_id = f"loso::{subject}"
            if (factor_id, fold_id) not in included_folds:
                continue
            fold_train_rows = [row for row in train_rows if row["subject_id"] != subject]
            fold_val_rows = [row for row in val_rows if row["subject_id"] == subject]
            if not fold_val_rows:
                continue
            train_positions = np.asarray([train_index[row["trial_id"]] for row in fold_train_rows], dtype=int)
            validation_positions = np.asarray([val_index[row["trial_id"]] for row in fold_val_rows], dtype=int)
            covered_indices.update(validation_positions.tolist())
            fold_train_target = all_train_target[train_positions]
            fold_groups = [row["text_uid"] for row in fold_train_rows]
            fold_correct_train = state["correct_train"][train_positions]
            fold_val_ids = [row["trial_id"] for row in fold_val_rows]
            fold_correct_val = store.matrix("correct_val", fold_val_ids)
            fold_wrong_val = store.matrix("matched_wrong_val", fold_val_ids)
            fold_vectorizer = DictVectorizer(sparse=False)
            fold_metadata_train = fold_vectorizer.fit_transform(
                metadata_dicts(fold_train_rows, length_by_trial, factor_id, task_only=False)
            ).astype(np.float32)
            fold_metadata_val = fold_vectorizer.transform(
                metadata_dicts(fold_val_rows, length_by_trial, factor_id, task_only=False)
            ).astype(np.float32)
            correct_parameter, correct_cv = select_parameter(
                fold_correct_train, fold_train_target, fold_groups, target_type,
                state["grid"], seeds[0], class_count,
            )
            metadata_parameter, metadata_cv = select_parameter(
                fold_metadata_train, fold_train_target, fold_groups, target_type,
                state["grid"], seeds[0], class_count,
            )
            for seed in seeds:
                correct_model = build_model(target_type, correct_parameter, seed)
                correct_model.fit(fold_correct_train, fold_train_target)
                metadata_model = build_model(target_type, metadata_parameter, seed)
                metadata_model.fit(fold_metadata_train, fold_train_target)
                selection_rows.extend([
                    {"factor_id": factor_id, "evaluation_mode": "subject_held_out", "fold_id": fold_id,
                     "condition": "frozen_glim_correct", "seed": seed,
                     "selected_parameter": correct_parameter, "train_only_group_cv_utility": correct_cv},
                    {"factor_id": factor_id, "evaluation_mode": "subject_held_out", "fold_id": fold_id,
                     "condition": "metadata_only", "seed": seed,
                     "selected_parameter": metadata_parameter, "train_only_group_cv_utility": metadata_cv},
                ])
                for condition, scores in (
                    ("frozen_glim_correct", predict_scores(correct_model, target_type, fold_correct_val, class_count)),
                    ("frozen_glim_matched_wrong_eeg", predict_scores(correct_model, target_type, fold_wrong_val, class_count)),
                    ("metadata_only", predict_scores(metadata_model, target_type, fold_metadata_val, class_count)),
                ):
                    key = f"{condition}\x1f{fold_id}"
                    fold_scores[key].append(scores)
        ordered = sorted(covered_indices)
        combined_scores: dict[str, np.ndarray] = {}
        for condition in ("frozen_glim_correct", "frozen_glim_matched_wrong_eeg", "metadata_only"):
            pieces = []
            piece_positions = []
            for subject in sorted({val_rows[index]["subject_id"] for index in ordered}):
                fold_id = f"loso::{subject}"
                key = f"{condition}\x1f{fold_id}"
                if key not in fold_scores:
                    continue
                positions = [index for index in ordered if val_rows[index]["subject_id"] == subject]
                piece_positions.extend(positions)
                pieces.append(average_scores(fold_scores[key]))
            order = np.argsort(piece_positions)
            combined_scores[condition] = np.concatenate(pieces, axis=0)[order]
        held_target = val_target[np.asarray(ordered, dtype=int)]
        held_utility = {
            condition: utility(
                held_target, hard_predictions(scores, target_type), target_type, class_count
            )
            for condition, scores in combined_scores.items()
        }
        subject_deltas[factor_id] = {
            "correct_minus_metadata": held_utility["frozen_glim_correct"] - held_utility["metadata_only"],
            "correct_minus_matched_wrong": held_utility["frozen_glim_correct"] - held_utility["frozen_glim_matched_wrong_eeg"],
            "rows": len(ordered),
            "subjects": len({val_rows[index]["subject_id"] for index in ordered}),
        }
        for condition, scores in combined_scores.items():
            for metric, value in metrics(held_target, scores, target_type, class_count).items():
                metric_rows.append({
                    "factor_id": factor_id, "evaluation_mode": "subject_held_out", "fold_id": "combined",
                    "condition": condition, "seed": "ensemble", "metric": metric, "value": value,
                    "rows": len(ordered), "subjects": subject_deltas[factor_id]["subjects"],
                })

    admission_rows: list[dict[str, Any]] = []
    for factor_id in factors:
        state = factor_state[factor_id]
        rows_for_factor = [row for row in contrast_rows if row["factor_id"] == factor_id]
        contrast_passes = {}
        for contrast in contrast_targets:
            cluster_rows = [row for row in rows_for_factor if row["contrast"] == contrast]
            contrast_passes[contrast] = bool(cluster_rows) and all(
                float(row["holm_adjusted_lower"]) > 0 for row in cluster_rows
            )
        held = subject_deltas.get(factor_id, {})
        held_direction = bool(held) and float(held["correct_minus_metadata"]) > 0 and float(
            held["correct_minus_matched_wrong"]
        ) > 0
        val_rows = state["val_rows"]
        minimums = (
            len(val_rows) >= int(registry["admission_rule"]["minimum_validation_rows"])
            and len({row["subject_id"] for row in val_rows})
            >= int(registry["admission_rule"]["minimum_validation_subjects"])
        )
        admitted = bool(minimums and all(contrast_passes.values()) and held_direction and not smoke)
        admission_rows.append({
            "factor_id": factor_id, "validation_rows": len(val_rows),
            "validation_subjects": len({row['subject_id'] for row in val_rows}),
            "ordinary_correct_minus_metadata": (
                state["ordinary_utility"]["frozen_glim_correct"] - state["ordinary_utility"]["metadata_only"]
            ),
            "ordinary_correct_minus_matched_wrong": (
                state["ordinary_utility"]["frozen_glim_correct"]
                - state["ordinary_utility"]["frozen_glim_matched_wrong_eeg"]
            ),
            "all_cluster_holm_lower_correct_minus_metadata_positive": int(
                contrast_passes["frozen_glim_correct_minus_metadata_only"]
            ),
            "all_cluster_holm_lower_correct_minus_matched_wrong_positive": int(
                contrast_passes["frozen_glim_correct_minus_frozen_glim_matched_wrong_eeg"]
            ),
            "subject_held_out_correct_minus_metadata": held.get("correct_minus_metadata", ""),
            "subject_held_out_correct_minus_matched_wrong": held.get("correct_minus_matched_wrong", ""),
            "subject_held_out_same_direction": int(held_direction),
            "admitted": int(admitted),
            "decision": "smoke_only" if smoke else ("admit" if admitted else "reject_retain_null"),
        })

    metric_fields = [
        "factor_id", "evaluation_mode", "fold_id", "condition", "seed", "metric", "value", "rows", "subjects"
    ]
    selection_fields = [
        "factor_id", "evaluation_mode", "fold_id", "condition", "seed", "selected_parameter",
        "train_only_group_cv_utility",
    ]
    prediction_fields = [
        "factor_id", "evaluation_mode", "fold_id", "condition", "trial_id", "subject_id",
        "normalized_text_sha256", "target", "prediction",
    ]
    contrast_fields = [
        "factor_id", "contrast", "cluster", "utility_delta", "ci95_lower", "ci95_upper",
        "p_one_sided", "holm_adjusted_p", "holm_rank", "holm_rank_alpha",
        "holm_adjusted_lower", "bootstrap_replicates",
    ]
    admission_fields = list(admission_rows[0])
    write_csv(output_root / "probe_metrics.csv", metric_fields, metric_rows)
    write_csv(output_root / "probe_selection.csv", selection_fields, selection_rows)
    write_csv(output_root / "probe_predictions.csv", prediction_fields, prediction_rows)
    write_csv(output_root / "planned_contrasts.csv", contrast_fields, contrast_rows)
    write_csv(output_root / "factor_admission.csv", admission_fields, admission_rows)
    artifacts = [
        "probe_metrics.csv", "probe_selection.csv", "probe_predictions.csv",
        "planned_contrasts.csv", "factor_admission.csv",
    ]
    manifest = {
        "status": "pass",
        "run_mode": "smoke" if smoke else "full_development",
        "source_index_sha256": registry["source_index_sha256"],
        "vector_index_sha256": store.manifest["vector_index_sha256"],
        "vector_manifest_sha256": sha256(vector_root / "vector_manifest.json"),
        "recoverability_registry_sha256": sha256(registry_path),
        "recoverability_rows_sha256": sha256(rows_path),
        "seeds": seeds,
        "bootstrap_replicates": bootstrap_replicates,
        "factors": factors,
        "ordinary_point_gain_candidates": candidates,
        "admitted_factors": [row["factor_id"] for row in admission_rows if int(row["admitted"])],
        "checks": {
            "held_out_test_accessed": False,
            "validation_tuning_permitted": False,
            "target_label_as_probe_input_permitted": False,
            "matched_wrong_model_fit_on_correct_training_vectors": True,
            "zero_and_gaussian_model_fit_on_correct_training_vectors": True,
            "subject_folds_refit_without_held_out_subject": True,
            "null_results_retained": True,
        },
        "artifact_sha256": {name: sha256(output_root / name) for name in artifacts},
    }
    manifest_path = output_root / "probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "pass", "run_mode": manifest["run_mode"], "factors": factors,
        "ordinary_point_gain_candidates": candidates, "admitted_factors": manifest["admitted_factors"],
        "vector_index_sha256": manifest["vector_index_sha256"], "output": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--expected-vector-index-sha256")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run(
        args.vector_root, args.protocol_root, args.output_root,
        args.expected_index_sha256, args.expected_vector_index_sha256, args.smoke,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
