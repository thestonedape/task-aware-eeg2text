"""Run the frozen four-way task-treatment pilot on prompt-neutral GLIM vectors.

The runner cannot address the held-out test split. It verifies the preserved
input artifact, reconstructs the already-frozen 24-way validation pools and
freezes their exact SHA-256 during an all-four real-data smoke, trains only the
declared residual adapters, and evaluates signal/task controls under one common
protocol.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.verify_prompt_neutral_pilot_inputs import (  # noqa: E402
    EXPECTED as INPUT_EXPECTED,
    read_csv,
    read_json,
    sha256,
    verify,
)
from project_adapters.task_treatment_pilots import (  # noqa: E402
    CONFIG_IDS,
    TASKS,
    TaskTreatmentPilot,
    active_parameter_budget,
    parameter_budget,
    symmetric_alignment_loss,
)


PROTOCOL_PATH = Path(__file__).with_name("task_treatment_pilot_execution_protocol.json")
CONTRACT_PATH = Path(__file__).with_name("task_treatment_pilot_contract.json")
RUNNER_PATH = Path(__file__)
ADAPTER_PATH = ROOT / "project_adapters" / "task_treatment_pilots.py"
POOL_FIELDS = (
    "target_trial_id", "candidate_rank", "candidate_id", "is_positive",
    "dataset_version", "reading_task", "target_evaluation_partition",
    "candidate_catalog_scope", "target_length", "candidate_length",
    "absolute_length_difference", "selection_rule",
)
PREDICTION_FIELDS = (
    "config_id", "seed", "signal_condition", "task_condition", "trial_id",
    "cohort", "evaluation_partition", "dataset_version", "reading_task", "subject_id",
    "normalized_text_sha256", "positive_rank", "reciprocal_rank", "top1", "top5",
    "positive_minus_best_negative_margin",
)
METRIC_FIELDS = (
    "config_id", "seed", "signal_condition", "task_condition", "scope",
    "reading_task", "rows", "top1", "top5", "mrr", "mean_positive_rank",
    "mean_positive_minus_best_negative_margin",
)
HISTORY_FIELDS = ("config_id", "seed", "epoch", "train_loss", "headline_macro_mrr")
FIT_COHORTS = {"primary_zuco2_nr_tsr", "auxiliary_sr"}
EXCLUDED_FIT_COHORT = "zuco1_nr_tsr_noncausal"
SIGNAL_CONDITIONS = ("correct_val", "zero_val", "gaussian_val", "matched_wrong_val")
TASK_CONDITIONS = ("correct", "masked", "shuffled")
RUN_ARTIFACTS = frozenset({
    "best_checkpoint.pt", "training_history.csv", "predictions.csv", "metrics.csv",
})


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def normalized_text(text: object) -> str:
    return " ".join(str(text).lower().split())


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_bytes(fields: Sequence[str], rows: Iterable[dict[str, object]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class Trial:
    trial_id: str
    split: str
    cohort: str
    dataset_version: str
    reading_task: str
    subject_id: str
    text_target_id: str


@dataclass
class PilotData:
    artifact_root: Path
    trials: dict[str, Trial]
    text_records: dict[str, dict[str, str]]
    text_vectors: dict[str, np.ndarray]
    eeg_vectors: dict[str, dict[str, np.ndarray]]
    partitions: dict[str, str]
    pools: dict[str, list[str]]
    positive_offsets: dict[str, int]
    pool_csv_sha256: str


def _load_indexed_vectors(
    root: Path,
    rows: list[dict[str, str]],
    identity_field: str,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["vector_file"]].append(row)
    output: dict[str, np.ndarray] = {}
    for relative, members in sorted(grouped.items()):
        path = root / relative
        with np.load(path, allow_pickle=False) as archive:
            vectors = archive["vectors"]
            for row in members:
                identity = row[identity_field]
                if identity in output:
                    raise ValueError(f"duplicate vector identity: {identity}")
                offset = int(row["vector_offset"])
                vector = vectors[offset].astype(np.float32, copy=True)
                if vector.shape != (1024,) or not np.isfinite(vector).all():
                    raise ValueError(f"invalid vector for {identity}: {vector.shape}")
                output[identity] = vector
    return output


def reconstruct_candidate_pools(
    trials: list[Trial],
    text_records: dict[str, dict[str, str]],
    pool_size: int,
    seed: int,
    partitions: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, list[str]], dict[str, int], str]:
    candidates: dict[tuple[str, str, str], dict[str, object]] = {}
    candidate_scopes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    contexts: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    validation = [trial for trial in trials if trial.split == "val"]
    if partitions is not None and any(
        trial.trial_id not in partitions for trial in validation
    ):
        raise ValueError("candidate catalog is missing a validation partition")
    for trial in trials:
        if trial.split not in {"train", "val"}:
            raise ValueError(f"candidate catalog received forbidden split: {trial.split}")
        text_record = text_records[trial.text_target_id]
        text = text_record["representative_text"]
        if hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest() != trial.text_target_id:
            raise ValueError(f"text identity mismatch: {trial.text_target_id}")
        candidate_key = (
            trial.dataset_version, trial.reading_task, trial.text_target_id
        )
        candidate_id = (
            f"{trial.dataset_version}::{trial.reading_task}::{trial.text_target_id}"
        )
        record = {
            "candidate_id": candidate_id,
            "text_target_id": trial.text_target_id,
            "dataset_version": trial.dataset_version,
            "reading_task": trial.reading_task,
            "length": len(str(text).split()),
        }
        previous = candidates.setdefault(candidate_key, record)
        if previous != record:
            raise ValueError(f"candidate collision: {candidate_id}")
        scope = (
            "training_catalog"
            if trial.split == "train"
            else (partitions or {}).get(trial.trial_id, "evaluation")
        )
        candidate_scopes[candidate_key].add(scope)
    for candidate_key, scopes in candidate_scopes.items():
        if "training_catalog" in scopes and len(scopes) > 1:
            raise ValueError(
                "normalized text identity crosses fit and evaluation catalogs: "
                f"{candidate_key}"
            )
    for record in candidates.values():
        contexts[(str(record["dataset_version"]), str(record["reading_task"]))].append(record)

    rows: list[dict[str, object]] = []
    pools: dict[str, list[str]] = {}
    positives: dict[str, int] = {}
    for trial in sorted(validation, key=lambda item: item.trial_id):
        target_partition = (partitions or {}).get(trial.trial_id, "evaluation")
        positive_key = (
            trial.dataset_version, trial.reading_task, trial.text_target_id
        )
        positive_id = (
            f"{trial.dataset_version}::{trial.reading_task}::{trial.text_target_id}"
        )
        positive = candidates[positive_key]
        target_length = int(positive["length"])
        eligible = [
            row
            for row in contexts[(trial.dataset_version, trial.reading_task)]
            if row["candidate_id"] != positive_id
            and (
                "training_catalog" in candidate_scopes[
                    (
                        str(row["dataset_version"]),
                        str(row["reading_task"]),
                        str(row["text_target_id"]),
                    )
                ]
                or target_partition in candidate_scopes[
                    (
                        str(row["dataset_version"]),
                        str(row["reading_task"]),
                        str(row["text_target_id"]),
                    )
                ]
            )
        ]
        eligible.sort(
            key=lambda row: (
                abs(int(row["length"]) - target_length),
                stable_hash(seed, trial.trial_id, row["candidate_id"], "pool-select"),
            )
        )
        if len(eligible) < pool_size - 1:
            raise ValueError(f"{trial.trial_id}: fewer than {pool_size} candidates")
        selected = [positive, *eligible[: pool_size - 1]]
        selected.sort(
            key=lambda row: stable_hash(seed, trial.trial_id, row["candidate_id"], "pool-order")
        )
        pools[trial.trial_id] = [str(row["text_target_id"]) for row in selected]
        positives[trial.trial_id] = next(
            index for index, row in enumerate(selected) if row["candidate_id"] == positive_id
        )
        for rank, row in enumerate(selected):
            row_key = (
                str(row["dataset_version"]),
                str(row["reading_task"]),
                str(row["text_target_id"]),
            )
            scopes = candidate_scopes[row_key]
            candidate_scope = (
                "training_catalog" if "training_catalog" in scopes else target_partition
            )
            rows.append({
                "target_trial_id": trial.trial_id,
                "candidate_rank": rank,
                "candidate_id": row["candidate_id"],
                "is_positive": int(row["candidate_id"] == positive_id),
                "dataset_version": row["dataset_version"],
                "reading_task": row["reading_task"],
                "target_evaluation_partition": target_partition,
                "candidate_catalog_scope": candidate_scope,
                "target_length": target_length,
                "candidate_length": row["length"],
                "absolute_length_difference": abs(int(row["length"]) - target_length),
                "selection_rule": (
                    "same_dataset_task_training_or_same_partition_then_length_then_seeded_hash"
                ),
            })
    payload = csv_bytes(POOL_FIELDS, rows)
    return rows, pools, positives, hashlib.sha256(payload).hexdigest()


def build_validation_partitions(
    trials: Iterable[Trial],
    protocol: dict[str, object],
) -> dict[str, str]:
    evaluation = protocol["evaluation"]
    assert isinstance(evaluation, dict)
    partition = evaluation["checkpoint_decision_partition"]
    assert isinstance(partition, dict)
    seed = int(partition["seed"])
    primary_by_task: dict[str, set[str]] = defaultdict(set)
    trial_list = list(trials)
    for trial in trial_list:
        if trial.cohort == "primary_zuco2_nr_tsr":
            primary_by_task[trial.reading_task].add(trial.text_target_id)
    if set(primary_by_task) != {"NR", "TSR"}:
        raise ValueError("primary validation partition requires NR and TSR")
    checkpoint_texts: dict[str, set[str]] = {}
    for task in ("NR", "TSR"):
        ordered = sorted(
            primary_by_task[task],
            key=lambda identity: stable_hash(
                seed, task, identity, "checkpoint-decision-partition"
            ),
        )
        checkpoint_texts[task] = set(ordered[: len(ordered) // 2])

    assignments: dict[str, str] = {}
    for trial in trial_list:
        if trial.cohort == "primary_zuco2_nr_tsr":
            assignments[trial.trial_id] = (
                "checkpoint"
                if trial.text_target_id in checkpoint_texts[trial.reading_task]
                else "decision"
            )
        elif trial.cohort == "auxiliary_sr":
            assignments[trial.trial_id] = "auxiliary_sr"
        elif trial.cohort == EXCLUDED_FIT_COHORT:
            assignments[trial.trial_id] = "zuco1_noncausal_diagnostic"
        else:
            raise ValueError(f"unknown validation cohort: {trial.cohort}")

    for name in ("checkpoint", "decision"):
        identities = sorted(
            trial_id for trial_id, value in assignments.items() if value == name
        )
        digest = hashlib.sha256(("\n".join(identities) + "\n").encode("utf-8")).hexdigest()
        if len(identities) != int(partition[f"{name}_rows"]):
            raise ValueError(f"{name} partition row-count mismatch")
        if digest != partition[f"{name}_trial_ids_sha256"]:
            raise ValueError(f"{name} partition SHA256 mismatch")
    return assignments


def load_pilot_data(
    artifact_root: Path,
    preserved_source_id: str,
    protocol: dict[str, object],
) -> PilotData:
    expected_source_id = str(protocol["input"]["preserved_source_id"])
    if preserved_source_id != expected_source_id:
        raise ValueError(
            f"preserved source identity mismatch: expected {expected_source_id!r}, "
            f"got {preserved_source_id!r}"
        )
    verify(artifact_root, preserved_source_id)
    text_index = read_csv(artifact_root / "text" / "text_vector_index.csv")
    mapping = read_csv(artifact_root / "text" / "trial_text_targets.csv")
    eeg_index = read_csv(artifact_root / "eeg" / "vector_index.csv")
    text_records = {row["text_target_id"]: row for row in text_index}
    trials = {
        row["trial_id"]: Trial(
            trial_id=row["trial_id"],
            split=row["split"],
            cohort=row["cohort"],
            dataset_version=row["dataset_version"],
            reading_task=row["reading_task"],
            subject_id=row["subject_id"],
            text_target_id=row["text_target_id"],
        )
        for row in mapping
    }
    if len(trials) != len(mapping) or any(trial.split == "test" for trial in trials.values()):
        raise ValueError("invalid development trial mapping")

    text_vectors = _load_indexed_vectors(
        artifact_root / "text", text_index, "text_target_id"
    )
    eeg_vectors: dict[str, dict[str, np.ndarray]] = {}
    for condition in sorted({row["condition"] for row in eeg_index}):
        condition_rows = [row for row in eeg_index if row["condition"] == condition]
        eeg_vectors[condition] = _load_indexed_vectors(
            artifact_root / "eeg", condition_rows, "target_trial_id"
        )
    if set(eeg_vectors) != {
        "correct_train", "correct_val", "zero_val", "gaussian_val", "matched_wrong_val"
    }:
        raise ValueError(f"unexpected EEG conditions: {sorted(eeg_vectors)}")

    pool_contract = protocol["candidate_pool"]
    assert isinstance(pool_contract, dict)
    validation = [trial for trial in trials.values() if trial.split == "val"]
    partitions = build_validation_partitions(validation, protocol)
    pool_rows, pools, positives, pool_sha = reconstruct_candidate_pools(
        list(trials.values()),
        text_records,
        int(pool_contract["size"]),
        int(pool_contract["seed"]),
        partitions,
    )
    if len(pool_rows) != int(pool_contract["expected_rows"]):
        raise ValueError("frozen candidate-pool cardinality mismatch")
    return PilotData(
        artifact_root=artifact_root,
        trials=trials,
        text_records=text_records,
        text_vectors=text_vectors,
        eeg_vectors=eeg_vectors,
        partitions=partitions,
        pools=pools,
        positive_offsets=positives,
        pool_csv_sha256=pool_sha,
    )


def task_batches(
    trials: list[Trial],
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[str]]:
    by_task: dict[str, list[Trial]] = defaultdict(list)
    for trial in trials:
        by_task[trial.reading_task].append(trial)
    if set(by_task) != set(TASKS):
        raise ValueError(f"fit schedule must contain SR/NR/TSR, got {sorted(by_task)}")
    all_batches: list[list[str]] = []
    for task in TASKS:
        rng = np.random.default_rng(
            int(stable_hash(seed, epoch, task, "task-batches")[:16], 16)
        )
        buckets: dict[str, list[str]] = defaultdict(list)
        for trial in by_task[task]:
            buckets[trial.text_target_id].append(trial.trial_id)
        for values in buckets.values():
            rng.shuffle(values)
        identities = sorted(buckets)
        task_batches_list: list[list[str]] = []
        while any(buckets[identity] for identity in identities):
            active = [identity for identity in identities if buckets[identity]]
            rng.shuffle(active)
            cycle = [buckets[identity].pop() for identity in active]
            for start in range(0, len(cycle), batch_size):
                task_batches_list.append(cycle[start:start + batch_size])
        trial_to_text = {
            trial.trial_id: trial.text_target_id for trial in by_task[task]
        }
        for singleton in [batch for batch in task_batches_list if len(batch) == 1]:
            singleton_text = trial_to_text[singleton[0]]
            donor_batch = next(
                (
                    batch for batch in reversed(task_batches_list)
                    if batch is not singleton
                    and len(batch) > 2
                    and any(trial_to_text[trial_id] != singleton_text for trial_id in batch)
                ),
                None,
            )
            if donor_batch is None:
                raise ValueError(f"{task}: cannot repair singleton contrastive batch")
            donor_index = next(
                index
                for index, trial_id in enumerate(donor_batch)
                if trial_to_text[trial_id] != singleton_text
            )
            singleton.append(donor_batch.pop(donor_index))
        if any(len(batch) < 2 for batch in task_batches_list):
            raise ValueError(f"{task}: singleton contrastive batch")
        if any(
            len({trial_to_text[trial_id] for trial_id in batch}) != len(batch)
            for batch in task_batches_list
        ):
            raise AssertionError(f"{task}: duplicate text identity within batch")
        all_batches.extend(task_batches_list)
    order_rng = np.random.default_rng(
        int(stable_hash(seed, epoch, "combined-task-order")[:16], 16)
    )
    order_rng.shuffle(all_batches)
    if sorted(item for batch in all_batches for item in batch) != sorted(
        trial.trial_id for trial in trials
    ):
        raise AssertionError("task schedule lost or duplicated a trial")
    return all_batches


def _batch_arrays(
    data: PilotData,
    trial_ids: Sequence[str],
    signal_condition: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    eeg = np.stack([data.eeg_vectors[signal_condition][trial_id] for trial_id in trial_ids])
    text = np.stack([
        data.text_vectors[data.trials[trial_id].text_target_id] for trial_id in trial_ids
    ])
    tasks = [data.trials[trial_id].reading_task for trial_id in trial_ids]
    return eeg, text, tasks


def score_trials(
    model: TaskTreatmentPilot,
    data: PilotData,
    trial_ids: Sequence[str],
    signal_condition: str,
    task_condition: str,
    config_id: str,
    seed: int,
    device: torch.device,
    batch_size: int = 512,
) -> list[dict[str, object]]:
    model.eval()
    output: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(trial_ids), batch_size):
            ids = list(trial_ids[start:start + batch_size])
            eeg = torch.from_numpy(
                np.stack([data.eeg_vectors[signal_condition][trial_id] for trial_id in ids])
            ).to(device)
            tasks = [data.trials[trial_id].reading_task for trial_id in ids]
            candidates = torch.from_numpy(np.stack([
                np.stack([data.text_vectors[text_id] for text_id in data.pools[trial_id]])
                for trial_id in ids
            ])).to(device)
            candidates = F.normalize(candidates, dim=2)
            if config_id == "separate_per_task" and task_condition == "masked":
                # The frozen contract defines this diagnostic as the mean
                # prediction, not a mean adapter delta followed by normalization.
                private_scores = []
                for adapter in model.private:
                    adapted = F.normalize(eeg + adapter.delta(eeg), dim=1)
                    private_scores.append(
                        torch.einsum("bd,bkd->bk", adapted, candidates)
                    )
                scores = torch.stack(private_scores).mean(0).cpu().numpy()
            else:
                adapted = F.normalize(model(eeg, tasks, task_condition), dim=1)
                scores = torch.einsum("bd,bkd->bk", adapted, candidates).cpu().numpy()
            for row_index, trial_id in enumerate(ids):
                positive_offset = data.positive_offsets[trial_id]
                row_scores = scores[row_index]
                # Stable ordering makes exact-score ties follow the frozen seeded pool order.
                ordering = np.argsort(-row_scores, kind="stable")
                positive_rank = int(np.flatnonzero(ordering == positive_offset)[0]) + 1
                negatives = np.delete(row_scores, positive_offset)
                trial = data.trials[trial_id]
                output.append({
                    "config_id": config_id,
                    "seed": seed,
                    "signal_condition": signal_condition,
                    "task_condition": task_condition,
                    "trial_id": trial_id,
                    "cohort": trial.cohort,
                    "evaluation_partition": data.partitions[trial_id],
                    "dataset_version": trial.dataset_version,
                    "reading_task": trial.reading_task,
                    "subject_id": trial.subject_id,
                    "normalized_text_sha256": trial.text_target_id,
                    "positive_rank": positive_rank,
                    "reciprocal_rank": 1.0 / positive_rank,
                    "top1": int(positive_rank <= 1),
                    "top5": int(positive_rank <= 5),
                    "positive_minus_best_negative_margin": float(
                        row_scores[positive_offset] - negatives.max()
                    ),
                })
    return output


def headline_macro_mrr(predictions: list[dict[str, object]]) -> float:
    task_values: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        if row["cohort"] == "primary_zuco2_nr_tsr":
            task_values[str(row["reading_task"])].append(float(row["reciprocal_rank"]))
    if set(task_values) != {"NR", "TSR"}:
        raise ValueError(f"headline requires NR and TSR, got {sorted(task_values)}")
    return float(np.mean([np.mean(task_values["NR"]), np.mean(task_values["TSR"])]))


def aggregate_metrics(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    if not predictions:
        return []
    first = predictions[0]
    groups: list[tuple[str, str, list[dict[str, object]]]] = []
    for task in ("NR", "TSR"):
        groups.append((
            "headline_decision_task", task,
            [
                row for row in predictions
                if row["evaluation_partition"] == "decision"
                and row["reading_task"] == task
            ],
        ))
        groups.append((
            "checkpoint_task_diagnostic", task,
            [
                row for row in predictions
                if row["evaluation_partition"] == "checkpoint"
                and row["reading_task"] == task
            ],
        ))
    groups.append((
        "auxiliary_sr", "SR",
        [row for row in predictions if row["cohort"] == "auxiliary_sr"],
    ))
    for task in ("NR", "TSR"):
        groups.append((
            "zuco1_noncausal_diagnostic", task,
            [
                row for row in predictions
                if row["cohort"] == EXCLUDED_FIT_COHORT and row["reading_task"] == task
            ],
        ))

    rows: list[dict[str, object]] = []
    for scope, task, members in groups:
        if not members:
            continue
        rows.append({
            "config_id": first["config_id"],
            "seed": first["seed"],
            "signal_condition": first["signal_condition"],
            "task_condition": first["task_condition"],
            "scope": scope,
            "reading_task": task,
            "rows": len(members),
            "top1": float(np.mean([float(row["top1"]) for row in members])),
            "top5": float(np.mean([float(row["top5"]) for row in members])),
            "mrr": float(np.mean([float(row["reciprocal_rank"]) for row in members])),
            "mean_positive_rank": float(
                np.mean([float(row["positive_rank"]) for row in members])
            ),
            "mean_positive_minus_best_negative_margin": float(np.mean([
                float(row["positive_minus_best_negative_margin"]) for row in members
            ])),
        })
    headline = [row for row in rows if row["scope"] == "headline_decision_task"]
    if len(headline) != 2:
        raise ValueError("missing headline task aggregates")
    rows.append({
        "config_id": first["config_id"],
        "seed": first["seed"],
        "signal_condition": first["signal_condition"],
        "task_condition": first["task_condition"],
        "scope": "headline_macro",
        "reading_task": "NR|TSR",
        "rows": sum(int(row["rows"]) for row in headline),
        "top1": float(np.mean([float(row["top1"]) for row in headline])),
        "top5": float(np.mean([float(row["top5"]) for row in headline])),
        "mrr": float(np.mean([float(row["mrr"]) for row in headline])),
        "mean_positive_rank": float(np.mean([
            float(row["mean_positive_rank"]) for row in headline
        ])),
        "mean_positive_minus_best_negative_margin": float(np.mean([
            float(row["mean_positive_minus_best_negative_margin"]) for row in headline
        ])),
    })
    return rows


def _binding(
    protocol_sha: str,
    contract_sha: str,
    preserved_source_id: str,
    project_commit: str,
    candidate_pool_sha: str,
    config_id: str,
    seed: int,
    smoke: bool,
) -> dict[str, object]:
    return {
        "protocol_sha256": protocol_sha,
        "pilot_contract_sha256": contract_sha,
        "input_manifest_sha256": INPUT_EXPECTED["combined_manifest_sha256"],
        "preserved_source_id": preserved_source_id,
        "project_commit": project_commit,
        "candidate_pool_sha256": candidate_pool_sha,
        "runner_source_sha256": sha256(RUNNER_PATH),
        "adapter_source_sha256": sha256(ADAPTER_PATH),
        "config_id": config_id,
        "seed": seed,
        "run_mode": "smoke" if smoke else "full",
    }


def _valid_completed_run(run_root: Path, binding: dict[str, object]) -> bool:
    summary_path = run_root / "run_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        if summary.get("status") != "pass" or summary.get("binding") != binding:
            return False
        artifact_hashes = summary["artifact_sha256"]
        if set(artifact_hashes) != RUN_ARTIFACTS:
            return False
        for name, expected in artifact_hashes.items():
            if sha256(run_root / name) != expected:
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_configuration_seed(
    data: PilotData,
    output_root: Path,
    config_id: str,
    seed: int,
    contract: dict[str, object],
    protocol_sha: str,
    contract_sha: str,
    preserved_source_id: str,
    project_commit: str,
    device: torch.device,
    smoke: bool,
) -> dict[str, object]:
    run_root = output_root / "runs" / config_id / str(seed)
    binding = _binding(
        protocol_sha, contract_sha, preserved_source_id, project_commit,
        data.pool_csv_sha256, config_id, seed, smoke,
    )
    if _valid_completed_run(run_root, binding):
        print(f"{config_id}/{seed}: reused completed run", flush=True)
        return read_json(run_root / "run_summary.json")

    training = contract["training"]
    assert isinstance(training, dict)
    epochs = 1 if smoke else int(training["epochs"])
    batch_size = int(training["batch_size"])
    fit_trials = [
        trial for trial in data.trials.values()
        if trial.split == "train" and trial.cohort in FIT_COHORTS
    ]
    if any(trial.cohort == EXCLUDED_FIT_COHORT for trial in fit_trials):
        raise AssertionError("excluded ZuCo1 NR/TSR cohort entered fit")
    if smoke:
        by_task: dict[str, list[Trial]] = defaultdict(list)
        for trial in sorted(fit_trials, key=lambda item: item.trial_id):
            by_task[trial.reading_task].append(trial)
        fit_trials = [trial for task in TASKS for trial in by_task[task][: min(256, len(by_task[task]))]]

    validation_ids = sorted(
        trial.trial_id for trial in data.trials.values() if trial.split == "val"
    )
    checkpoint_ids = [
        trial_id for trial_id in validation_ids
        if data.partitions[trial_id] == "checkpoint"
    ]
    if smoke:
        checkpoint_by_task: dict[str, list[str]] = defaultdict(list)
        for trial_id in checkpoint_ids:
            checkpoint_by_task[data.trials[trial_id].reading_task].append(trial_id)
        checkpoint_ids = [
            trial_id
            for task in ("NR", "TSR")
            for trial_id in checkpoint_by_task[task][:64]
        ]

    set_determinism(seed)
    model = TaskTreatmentPilot(config_id, vector_dim=1024).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    initial_predictions = score_trials(
        model, data, checkpoint_ids, "correct_val", "correct",
        config_id, seed, device,
    )
    best_score = headline_macro_mrr(initial_predictions)
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    history: list[dict[str, object]] = [{
        "config_id": config_id,
        "seed": seed,
        "epoch": 0,
        "train_loss": "",
        "headline_macro_mrr": best_score,
    }]
    print(
        f"{config_id}/{seed}: epoch 0/{epochs} identity macro_mrr={best_score:.6f}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_ids in task_batches(fit_trials, batch_size, seed, epoch):
            eeg_np, text_np, tasks = _batch_arrays(data, batch_ids, "correct_train")
            eeg = torch.from_numpy(eeg_np).to(device)
            text = torch.from_numpy(text_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = symmetric_alignment_loss(model(eeg, tasks, "correct"), text)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"non-finite loss: {config_id}/{seed}/epoch{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        checkpoint_predictions = score_trials(
            model, data, checkpoint_ids, "correct_val", "correct",
            config_id, seed, device,
        )
        score = headline_macro_mrr(checkpoint_predictions)
        history.append({
            "config_id": config_id,
            "seed": seed,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "headline_macro_mrr": score,
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        print(
            f"{config_id}/{seed}: epoch {epoch}/{epochs} "
            f"loss={history[-1]['train_loss']:.6f} macro_mrr={score:.6f}",
            flush=True,
        )
    if best_state is None:
        raise AssertionError("no checkpoint selected")
    model.load_state_dict(best_state)

    if smoke:
        evaluation_by_task: dict[str, list[str]] = defaultdict(list)
        for trial_id in validation_ids:
            evaluation_by_task[data.trials[trial_id].reading_task].append(trial_id)
        evaluation_ids = [
            trial_id
            for task in TASKS
            for trial_id in evaluation_by_task[task][:40]
        ]
    else:
        evaluation_ids = validation_ids
    combinations = [
        *(condition_task for condition_task in (
            (condition, "correct") for condition in SIGNAL_CONDITIONS
        )),
        ("correct_val", "masked"),
        ("correct_val", "shuffled"),
    ]
    predictions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for signal_condition, task_condition in combinations:
        members = score_trials(
            model, data, evaluation_ids, signal_condition, task_condition,
            config_id, seed, device,
        )
        predictions.extend(members)
        if not smoke:
            metrics.extend(aggregate_metrics(members))

    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_root / "best_checkpoint.pt"
    torch.save({
        "config_id": config_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_headline_macro_mrr": best_score,
        "state_dict": best_state,
        "binding": binding,
    }, checkpoint_path)
    history_path = run_root / "training_history.csv"
    predictions_path = run_root / "predictions.csv"
    metrics_path = run_root / "metrics.csv"
    write_csv(history_path, HISTORY_FIELDS, history)
    write_csv(predictions_path, PREDICTION_FIELDS, predictions)
    write_csv(metrics_path, METRIC_FIELDS, metrics)
    artifact_hashes = {
        path.name: sha256(path)
        for path in (checkpoint_path, history_path, predictions_path, metrics_path)
    }
    summary = {
        "status": "pass",
        "binding": binding,
        "best_epoch": best_epoch,
        "best_headline_macro_mrr": best_score,
        "checkpoint_partition_rows": len(checkpoint_ids),
        "epochs": epochs,
        "fit_rows": len(fit_trials),
        "evaluation_rows": len(evaluation_ids),
        "trainable_parameters": model.trainable_parameter_count,
        "active_parameters_per_example": model.active_parameter_count_per_example,
        "auxiliary_factor_losses": [],
        "held_out_test_accessed": False,
        "artifact_sha256": artifact_hashes,
    }
    atomic_json(run_root / "run_summary.json", summary)
    return summary


def bootstrap_weights(labels: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    unique, inverse = np.unique(np.asarray(labels), return_inverse=True)
    counts = np.bincount(
        rng.integers(0, len(unique), len(unique)), minlength=len(unique)
    )
    return counts[inverse].astype(np.float64)


def paired_bootstrap(
    effects: np.ndarray,
    subjects: Sequence[str],
    texts: Sequence[str],
    cluster: str,
    replicates: int,
    seed: int,
    tasks: Sequence[str] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    task_array = np.asarray(tasks) if tasks is not None else None
    if task_array is not None and set(task_array.tolist()) != {"NR", "TSR"}:
        raise ValueError("task-macro bootstrap requires NR and TSR")
    for index in range(replicates):
        for _attempt in range(1000):
            if cluster == "subject_id":
                weights = bootstrap_weights(subjects, rng)
            elif cluster == "normalized_text_sha256":
                weights = bootstrap_weights(texts, rng)
            elif cluster == "two_way_subject_by_text":
                weights = (
                    bootstrap_weights(subjects, rng)
                    * bootstrap_weights(texts, rng)
                )
            else:
                raise ValueError(cluster)
            if task_array is None:
                denominator = weights.sum()
                if denominator > 0:
                    output[index] = float(np.dot(weights, effects) / denominator)
                    break
                continue
            task_estimates = []
            valid = True
            for task in ("NR", "TSR"):
                mask = task_array == task
                denominator = weights[mask].sum()
                if denominator <= 0:
                    valid = False
                    break
                task_estimates.append(
                    float(np.dot(weights[mask], effects[mask]) / denominator)
                )
            if valid:
                output[index] = float(np.mean(task_estimates))
                break
        else:
            raise RuntimeError(f"unable to draw a nonempty {cluster} bootstrap replicate")
    return output


def _read_run_predictions(output_root: Path, config_id: str, seed: int) -> list[dict[str, str]]:
    return read_csv(output_root / "runs" / config_id / str(seed) / "predictions.csv")


def finalize_full_run(
    output_root: Path,
    protocol: dict[str, object],
    protocol_sha: str,
    contract_sha: str,
    seeds: list[int],
    pool_sha: str,
    partition_sha: str,
    preserved_source_id: str,
    project_commit: str,
) -> dict[str, object]:
    prediction_lookup: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    seed_prediction: dict[tuple[str, int, str, str, str], float] = {}
    identity: dict[str, dict[str, str]] = {}
    metric_rows: list[dict[str, object]] = []
    for config_id in CONFIG_IDS:
        for seed in seeds:
            predictions = _read_run_predictions(output_root, config_id, seed)
            for row in predictions:
                key = (
                    config_id, row["signal_condition"], row["task_condition"], row["trial_id"]
                )
                prediction_lookup[key].append(float(row["reciprocal_rank"]))
                seed_prediction[(
                    config_id, seed, row["signal_condition"],
                    row["task_condition"], row["trial_id"],
                )] = float(row["reciprocal_rank"])
                identity[row["trial_id"]] = row
            metric_rows.extend(
                read_csv(output_root / "runs" / config_id / str(seed) / "metrics.csv")
            )
    write_csv(output_root / "all_seed_metrics.csv", METRIC_FIELDS, metric_rows)

    conditions = sorted({
        (key[1], key[2]) for key in prediction_lookup
        if key[0] == "generic_pooled"
    })
    seed_average_rows: list[dict[str, object]] = []
    seed_average: dict[tuple[str, str, str, str], float] = {}
    for config_id in CONFIG_IDS:
        for signal_condition, task_condition in conditions:
            for trial_id, row in identity.items():
                key = (config_id, signal_condition, task_condition, trial_id)
                values = prediction_lookup.get(key, [])
                if len(values) != len(seeds):
                    raise ValueError(f"missing seed predictions: {key} ({len(values)})")
                value = float(np.mean(values))
                seed_average[key] = value
                seed_average_rows.append({
                    "config_id": config_id,
                    "signal_condition": signal_condition,
                    "task_condition": task_condition,
                    "trial_id": trial_id,
                    "cohort": row["cohort"],
                    "evaluation_partition": row["evaluation_partition"],
                    "reading_task": row["reading_task"],
                    "subject_id": row["subject_id"],
                    "normalized_text_sha256": row["normalized_text_sha256"],
                    "mean_reciprocal_rank": value,
                })
    seed_average_fields = (
        "config_id", "signal_condition", "task_condition", "trial_id", "cohort",
        "evaluation_partition", "reading_task", "subject_id",
        "normalized_text_sha256", "mean_reciprocal_rank",
    )
    write_csv(output_root / "seed_averaged_predictions.csv", seed_average_fields, seed_average_rows)

    primary_ids = sorted(
        trial_id for trial_id, row in identity.items()
        if row["evaluation_partition"] == "decision"
    )
    if len(primary_ids) != int(
        protocol["evaluation"]["checkpoint_decision_partition"]["decision_rows"]
    ):
        raise ValueError("decision partition row count changed")
    primary_tasks = [identity[trial_id]["reading_task"] for trial_id in primary_ids]
    task_array = np.asarray(primary_tasks)

    def task_macro(values: np.ndarray) -> float:
        return float(np.mean([
            values[task_array == task].mean() for task in ("NR", "TSR")
        ]))

    def averaged_effects(
        left: tuple[str, str, str],
        right: tuple[str, str, str],
    ) -> np.ndarray:
        return np.asarray([
            seed_average[(left[0], left[1], left[2], trial_id)]
            - seed_average[(right[0], right[1], right[2], trial_id)]
            for trial_id in primary_ids
        ], dtype=np.float64)

    def seed_effects(
        seed: int,
        left: tuple[str, str, str],
        right: tuple[str, str, str],
    ) -> np.ndarray:
        return np.asarray([
            seed_prediction[(left[0], seed, left[1], left[2], trial_id)]
            - seed_prediction[(right[0], seed, right[1], right[2], trial_id)]
            for trial_id in primary_ids
        ], dtype=np.float64)

    headline: dict[str, float] = {}
    signal_gaps: dict[str, float] = {}
    for config_id in CONFIG_IDS:
        correct = np.asarray([
            seed_average[(config_id, "correct_val", "correct", trial_id)]
            for trial_id in primary_ids
        ])
        wrong = np.asarray([
            seed_average[(config_id, "matched_wrong_val", "correct", trial_id)]
            for trial_id in primary_ids
        ])
        headline[config_id] = task_macro(correct)
        signal_gaps[config_id] = task_macro(correct - wrong)

    uncertainty = protocol["uncertainty"]
    assert isinstance(uncertainty, dict)
    masked = "masked_shared_private"
    planned_baselines = [str(value) for value in uncertainty["planned_baselines"]]
    contrasts = [
        *[
            (
                "model_comparison", masked, baseline,
                (masked, "correct_val", "correct"),
                (baseline, "correct_val", "correct"),
                float(uncertainty["familywise_lower_quantile"]),
            )
            for baseline in planned_baselines
        ],
        (
            "signal_specificity", masked, "matched_wrong_eeg",
            (masked, "correct_val", "correct"),
            (masked, "matched_wrong_val", "correct"),
            float(uncertainty["single_contrast_lower_quantile"]),
        ),
        (
            "task_control", masked, "masked_task",
            (masked, "correct_val", "correct"),
            (masked, "correct_val", "masked"),
            float(uncertainty["familywise_lower_quantile"]),
        ),
        (
            "task_control", masked, "shuffled_task",
            (masked, "correct_val", "correct"),
            (masked, "correct_val", "shuffled"),
            float(uncertainty["familywise_lower_quantile"]),
        ),
    ]
    subjects = [identity[trial_id]["subject_id"] for trial_id in primary_ids]
    texts = [identity[trial_id]["normalized_text_sha256"] for trial_id in primary_ids]
    comparisons: list[dict[str, object]] = []
    seedwise: list[dict[str, object]] = []
    for contrast_index, (
        contrast_type, model, reference, left, right, decision_quantile,
    ) in enumerate(contrasts):
        effects = averaged_effects(left, right)
        macro_effect = task_macro(effects)
        for seed in seeds:
            seed_delta = task_macro(seed_effects(seed, left, right))
            seedwise.append({
                "contrast_type": contrast_type,
                "model": model,
                "reference": reference,
                "seed": seed,
                "macro_mrr_delta": seed_delta,
                "direction_positive": int(seed_delta > 0),
            })
        for cluster_index, cluster in enumerate(uncertainty["clusters"]):
            cluster = str(cluster)
            draws = paired_bootstrap(
                effects, subjects, texts, cluster,
                int(uncertainty["bootstrap_replicates"]),
                20260717 + 1000 * contrast_index + 100 * cluster_index,
                primary_tasks,
            )
            comparisons.append({
                "contrast_type": contrast_type,
                "model": model,
                "reference": reference,
                "cluster": cluster,
                "rows": len(effects),
                "mean_mrr_delta": macro_effect,
                "ci95_lower": float(np.quantile(draws, 0.025)),
                "ci95_upper": float(np.quantile(draws, 0.975)),
                "decision_lower_quantile": decision_quantile,
                "decision_lower": float(np.quantile(draws, decision_quantile)),
                "bootstrap_replicates": len(draws),
            })
    comparison_fields = (
        "contrast_type", "model", "reference", "cluster", "rows",
        "mean_mrr_delta", "ci95_lower", "ci95_upper",
        "decision_lower_quantile", "decision_lower", "bootstrap_replicates",
    )
    write_csv(output_root / "paired_model_comparisons.csv", comparison_fields, comparisons)
    seedwise_fields = (
        "contrast_type", "model", "reference", "seed",
        "macro_mrr_delta", "direction_positive",
    )
    write_csv(output_root / "seedwise_contrasts.csv", seedwise_fields, seedwise)

    mrr_wins = all(headline[masked] > headline[baseline] for baseline in planned_baselines)
    model_rows = [row for row in comparisons if row["contrast_type"] == "model_comparison"]
    model_uncertainty_pass = all(float(row["decision_lower"]) > 0 for row in model_rows)
    model_seed_pass = all(
        int(row["direction_positive"]) == 1
        for row in seedwise if row["contrast_type"] == "model_comparison"
    )
    signal_rows = [row for row in comparisons if row["contrast_type"] == "signal_specificity"]
    signal_positive = signal_gaps[masked] > 0
    signal_uncertainty_pass = all(float(row["decision_lower"]) > 0 for row in signal_rows)
    signal_not_decreased = all(
        signal_gaps[masked] >= signal_gaps[baseline] for baseline in planned_baselines
    )
    task_rows = [row for row in comparisons if row["contrast_type"] == "task_control"]
    task_control_uncertainty_pass = all(float(row["decision_lower"]) > 0 for row in task_rows)
    task_control_gaps = {
        reference: float(next(
            row["mean_mrr_delta"] for row in task_rows
            if row["reference"] == reference
        ))
        for reference in ("masked_task", "shuffled_task")
    }
    selected = all((
        mrr_wins,
        model_uncertainty_pass,
        model_seed_pass,
        signal_positive,
        signal_uncertainty_pass,
        signal_not_decreased,
        task_control_uncertainty_pass,
    ))
    decision = {
        "status": "pass",
        "selected_for_richer_stage": bool(selected),
        "required_configuration": masked,
        "headline_macro_mrr": headline,
        "correct_minus_matched_wrong_macro_mrr_gap": signal_gaps,
        "correct_task_control_macro_mrr_gap": task_control_gaps,
        "requirements": {
            "mrr_above_both_planned_baselines": mrr_wins,
            "model_all_cluster_familywise_lower_bounds_positive": model_uncertainty_pass,
            "model_delta_positive_in_all_training_seeds": model_seed_pass,
            "masked_model_signal_gap_positive": signal_positive,
            "signal_all_cluster_lower_bounds_positive": signal_uncertainty_pass,
            "signal_gap_not_below_either_planned_baseline": signal_not_decreased,
            "task_controls_all_cluster_familywise_lower_bounds_positive": (
                task_control_uncertainty_pass
            ),
        },
        "richer_factor_if_selected": "tsr_instruction_relation",
        "prohibited_factor_losses": [
            "length_words_whitespace_v1", "nr_relation_content", "sr_sentiment_3"
        ],
        "held_out_test_accessed": False,
    }
    atomic_json(output_root / "continuation_decision.json", decision)

    artifact_names = [
        "all_seed_metrics.csv", "seed_averaged_predictions.csv", "seedwise_contrasts.csv",
        "paired_model_comparisons.csv", "continuation_decision.json",
        "frozen_candidate_pools.csv", "validation_partition.csv",
        "frozen_protocol/task_treatment_pilot_contract.json",
        "frozen_protocol/task_treatment_pilot_execution_protocol.json",
        "frozen_protocol/run_task_treatment_pilots.py",
        "frozen_protocol/task_treatment_pilots.py",
    ]
    run_summary_hashes = {
        str(path.relative_to(output_root)).replace("\\", "/"): sha256(path)
        for path in sorted((output_root / "runs").glob("*/*/run_summary.json"))
    }
    if len(run_summary_hashes) != len(CONFIG_IDS) * len(seeds):
        raise ValueError("top-level manifest did not find all run summaries")
    manifest = {
        "status": "pass",
        "schema_version": 1,
        "run_mode": "full",
        "pilot_contract_sha256": contract_sha,
        "execution_protocol_sha256": protocol_sha,
        "input_manifest_sha256": INPUT_EXPECTED["combined_manifest_sha256"],
        "preserved_source_id": preserved_source_id,
        "project_commit": project_commit,
        "runner_source_sha256": sha256(RUNNER_PATH),
        "adapter_source_sha256": sha256(ADAPTER_PATH),
        "candidate_pool_sha256": pool_sha,
        "validation_partition_sha256": partition_sha,
        "configurations": list(CONFIG_IDS),
        "seeds": seeds,
        "parameter_budget": parameter_budget(),
        "active_parameters_per_example": protocol["training"]["active_parameters_per_example"],
        "auxiliary_factor_losses": [],
        "held_out_test_accessed": False,
        "continuation_selected": decision["selected_for_richer_stage"],
        "artifact_sha256": {name: sha256(output_root / name) for name in artifact_names},
        "run_summary_sha256": run_summary_hashes,
    }
    atomic_json(output_root / "pilot_manifest.json", manifest)
    return manifest


def validate_execution_contract(
    contract: dict[str, object],
    protocol: dict[str, object],
    preserved_source_id: str,
    project_commit: str,
) -> None:
    if len(project_commit) != 40 or any(
        character not in "0123456789abcdef" for character in project_commit.lower()
    ):
        raise ValueError("project commit must be a full 40-character Git SHA")
    if preserved_source_id != protocol["input"]["preserved_source_id"]:
        raise ValueError("preserved source ID does not match execution protocol")
    if contract["status"] != "frozen_before_pilot_training":
        raise ValueError("pilot contract is not frozen")
    if protocol["status"] != "frozen_before_pilot_execution":
        raise ValueError("execution protocol is not frozen")
    contract_configs = tuple(
        str(row["id"]) for row in contract["configurations"]
    )
    if contract_configs != CONFIG_IDS:
        raise ValueError("configuration IDs diverge from the frozen contract")
    if set(contract["cohorts"]["fit"]) != FIT_COHORTS:
        raise ValueError("fit cohorts diverge from the frozen contract")
    if set(protocol["training"]["fit_cohorts"]) != FIT_COHORTS:
        raise ValueError("execution fit cohorts diverge from implementation")
    if contract["cohorts"]["excluded_from_fit"] != [EXCLUDED_FIT_COHORT]:
        raise ValueError("excluded fit cohort diverges from the frozen contract")
    if protocol["training"]["excluded_cohort"] != EXCLUDED_FIT_COHORT:
        raise ValueError("execution excluded cohort diverges from implementation")
    contract_signals = tuple(
        f"{condition}_val" for condition in contract["evaluation"]["signal_conditions"]
    )
    if contract_signals != SIGNAL_CONDITIONS:
        raise ValueError("signal conditions diverge from the frozen contract")
    if tuple(protocol["evaluation"]["signal_conditions"]) != SIGNAL_CONDITIONS:
        raise ValueError("execution signal conditions diverge from implementation")
    if tuple(protocol["evaluation"]["task_conditions"]) != TASK_CONDITIONS:
        raise ValueError("execution task conditions diverge from implementation")
    counts = parameter_budget()
    contract_counts = {
        str(row["id"]): int(row["trainable_parameters_at_1024d"])
        for row in contract["configurations"]
    }
    if any(int(counts[config_id]) != contract_counts[config_id] for config_id in CONFIG_IDS):
        raise ValueError("implemented parameter budget diverges from the frozen contract")
    if {
        key: int(value)
        for key, value in protocol["training"]["stored_trainable_parameters"].items()
    } != contract_counts:
        raise ValueError("execution parameter budget diverges from the frozen contract")
    if {
        key: int(value)
        for key, value in protocol["training"]["active_parameters_per_example"].items()
    } != active_parameter_budget():
        raise ValueError("active per-example parameter report diverges from implementation")
    if contract["training"]["auxiliary_factor_losses"]:
        raise ValueError("factor loss entered the frozen task-treatment contract")
    if protocol["training"]["auxiliary_factor_losses"]:
        raise ValueError("factor loss entered the execution protocol")
    if contract["evaluation"]["held_out_test_accessed"]:
        raise ValueError("frozen contract reports held-out test access")
    if protocol["continuation"]["held_out_test_accessed"]:
        raise ValueError("execution protocol reports held-out test access")


def execution_matrix(
    contract: dict[str, object], smoke: bool
) -> tuple[list[str], list[int]]:
    seeds = [int(value) for value in contract["training"]["seeds"]]
    return list(CONFIG_IDS), seeds[:1] if smoke else seeds


def run(
    artifact_root: Path,
    output_root: Path,
    preserved_source_id: str,
    project_commit: str,
    device_name: str,
    smoke: bool = False,
    expected_candidate_pool_sha256: str | None = None,
) -> dict[str, object]:
    contract = read_json(CONTRACT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    contract_sha = sha256(CONTRACT_PATH)
    protocol_sha = sha256(PROTOCOL_PATH)
    if contract_sha != protocol["parent_contract_sha256"]:
        raise ValueError("execution protocol is not bound to the frozen pilot contract")
    if protocol["input"]["combined_manifest_sha256"] != INPUT_EXPECTED["combined_manifest_sha256"]:
        raise ValueError("execution protocol input binding mismatch")
    validate_execution_contract(
        contract, protocol, preserved_source_id, project_commit
    )
    data = load_pilot_data(artifact_root, preserved_source_id, protocol)
    if not smoke and expected_candidate_pool_sha256 is None:
        raise ValueError(
            "full execution requires the candidate-pool SHA256 frozen by the "
            "all-four real-data smoke"
        )
    if (
        expected_candidate_pool_sha256 is not None
        and data.pool_csv_sha256 != expected_candidate_pool_sha256
    ):
        raise ValueError(
            "candidate-pool SHA256 differs from the all-four real-data smoke"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    pool_rows, _, _, pool_sha = reconstruct_candidate_pools(
        list(data.trials.values()),
        data.text_records,
        int(protocol["candidate_pool"]["size"]),
        int(protocol["candidate_pool"]["seed"]),
        data.partitions,
    )
    write_csv(output_root / "frozen_candidate_pools.csv", POOL_FIELDS, pool_rows)
    if sha256(output_root / "frozen_candidate_pools.csv") != pool_sha:
        raise AssertionError("candidate pool changed while writing")
    partition_rows = [
        {
            "trial_id": trial.trial_id,
            "cohort": trial.cohort,
            "dataset_version": trial.dataset_version,
            "reading_task": trial.reading_task,
            "subject_id": trial.subject_id,
            "normalized_text_sha256": trial.text_target_id,
            "evaluation_partition": data.partitions[trial.trial_id],
        }
        for trial in sorted(data.trials.values(), key=lambda item: item.trial_id)
        if trial.split == "val"
    ]
    partition_fields = (
        "trial_id", "cohort", "dataset_version", "reading_task", "subject_id",
        "normalized_text_sha256", "evaluation_partition",
    )
    write_csv(output_root / "validation_partition.csv", partition_fields, partition_rows)
    partition_sha = sha256(output_root / "validation_partition.csv")
    (output_root / "frozen_protocol").mkdir(exist_ok=True)
    frozen_sources = (
        (CONTRACT_PATH, CONTRACT_PATH.name),
        (PROTOCOL_PATH, PROTOCOL_PATH.name),
        (RUNNER_PATH, RUNNER_PATH.name),
        (ADAPTER_PATH, ADAPTER_PATH.name),
    )
    for source, name in frozen_sources:
        destination = output_root / "frozen_protocol" / name
        destination.write_bytes(source.read_bytes())

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    configs, seeds = execution_matrix(contract, smoke)
    summaries = []
    for config_id in configs:
        for seed in seeds:
            summaries.append(run_configuration_seed(
                data, output_root, config_id, seed, contract, protocol_sha,
                contract_sha, preserved_source_id, project_commit, device, smoke,
            ))
    if smoke:
        report = {
            "status": "pass",
            "run_mode": "smoke",
            "runs": len(summaries),
            "configurations": configs,
            "seed": seeds[0],
            "candidate_pool_sha256": pool_sha,
            "validation_partition_sha256": partition_sha,
            "parameter_budget": parameter_budget(),
            "held_out_test_accessed": False,
            "output": str(output_root),
        }
        atomic_json(output_root / "smoke_report.json", report)
        return report
    manifest = finalize_full_run(
        output_root, protocol, protocol_sha, contract_sha, seeds, pool_sha,
        partition_sha, preserved_source_id, project_commit,
    )
    return {**manifest, "output": str(output_root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preserved-source-id", required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expected-candidate-pool-sha256")
    args = parser.parse_args()
    report = run(
        args.artifact_root,
        args.output_root,
        args.preserved_source_id,
        args.project_commit,
        args.device,
        args.smoke,
        args.expected_candidate_pool_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    marker = (
        "TASK-TREATMENT PILOT SMOKE: PASS"
        if args.smoke else "TASK-TREATMENT PILOTS: PASS"
    )
    print(marker)


if __name__ == "__main__":
    main()
