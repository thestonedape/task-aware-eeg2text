"""Apply the prospectively frozen P4b continuation decision.

This module is deliberately separate from training.  It consumes only a
completed full-run manifest and per-trial reciprocal-rank summaries.  It does
not load checkpoints, authorize training, or access the held-out test split.

The production command has no switches for changing trial counts, seeds,
bootstrap replicates, quantiles, margins, or continuation gates.  The small
``DecisionSpec`` injection point exists only so unit tests can exercise the
same validation and bootstrap code without fabricating the full 45-run table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARM_IDS = ("global_mixed", "true_task_segmented", "pseudo_task_segmented")
BASELINE_IDS = ("global_mixed", "pseudo_task_segmented")
TASKS = ("NR", "TSR")
SIGNAL_CONDITIONS = ("correct", "matched_wrong")
CLUSTERS = ("subject_id", "normalized_text_sha256", "two_way_subject_by_text")

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


@dataclass(frozen=True)
class DecisionSpec:
    arms: tuple[str, ...] = ARM_IDS
    baselines: tuple[str, ...] = BASELINE_IDS
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (20260717, 20260718, 20260719)
    task_counts: tuple[tuple[str, int], ...] = (("NR", 4126), ("TSR", 4885))
    fold_counts: tuple[tuple[int, int], ...] = (
        (0, 1810), (1, 1785), (2, 1790), (3, 1814), (4, 1812),
    )
    pool_size: int = 24
    bootstrap_replicates: int = 5000
    bootstrap_base_seed: int = 2026071805
    superiority_q: float = 0.0125
    per_task_noninferiority_q: float = 0.0125
    chance_q: float = 0.025
    signal_positive_q: float = 0.025
    signal_noninferiority_q: float = 0.0125
    per_task_noninferiority_margin: float = 0.01
    signal_noninferiority_margin: float = 0.005
    chance_mrr: float = 0.15733159073972944

    @property
    def expected_trials(self) -> int:
        return sum(count for _, count in self.task_counts)


PRODUCTION_SPEC = DecisionSpec()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_bool(value: object, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{field}: expected true/false, got {value!r}")


def _parse_int(value: object, field: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected integer, got {value!r}") from exc
    return parsed


def read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == list(expected_fields), f"unexpected columns in {path}")
        return list(reader)


def reciprocal_rank_from_candidates(candidates: Sequence[Mapping[str, object]]) -> float:
    """Return RR with score ties broken by the prospectively frozen candidate rank.

    Candidate rank is an immutable pool-order field, not a model-produced rank.
    The summary CSV consumed by the CLI stores the resulting positive rank; a
    run-artifact verifier should call this helper when checking raw scores.
    """

    _require(len(candidates) > 0, "candidate list is empty")
    pool_ranks = [_parse_int(row.get("candidate_rank"), "candidate_rank") for row in candidates]
    _require(sorted(pool_ranks) == list(range(len(candidates))), "candidate_rank must be 0..pool_size-1")
    positives = [row for row in candidates if _parse_bool(row.get("is_positive"), "is_positive")]
    _require(len(positives) == 1, "candidate pool must contain exactly one positive")
    scored: list[tuple[float, int, bool]] = []
    for row in candidates:
        score = float(row.get("score"))
        _require(math.isfinite(score), "candidate score must be finite")
        scored.append((score, _parse_int(row.get("candidate_rank"), "candidate_rank"), _parse_bool(row.get("is_positive"), "is_positive")))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
    positive_rank = next(index for index, item in enumerate(ordered, start=1) if item[2])
    return 1.0 / positive_rank


@dataclass(frozen=True)
class Trial:
    trial_id: str
    outer_fold: int
    reading_task: str
    subject_id: str
    normalized_text_sha256: str


@dataclass
class ValidatedData:
    trials: list[Trial]
    # (arm, seed, signal condition) -> RR values aligned with ``trials``.
    values: dict[tuple[str, int, str], list[float]]


def validate_inputs(
    run_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    spec: DecisionSpec = PRODUCTION_SPEC,
) -> ValidatedData:
    expected_runs = {
        (arm, fold, seed)
        for arm in spec.arms for fold in spec.folds for seed in spec.seeds
    }
    _require(len(run_rows) == len(expected_runs), f"incomplete run matrix: expected {len(expected_runs)} rows")
    observed_runs: set[tuple[str, int, int]] = set()
    for row in run_rows:
        arm = str(row.get("arm_id"))
        fold = _parse_int(row.get("outer_fold"), "outer_fold")
        seed = _parse_int(row.get("training_seed"), "training_seed")
        key = (arm, fold, seed)
        _require(key in expected_runs, f"unexpected run {key}")
        _require(key not in observed_runs, f"duplicate run {key}")
        observed_runs.add(key)
        # ``complete`` is an execution state.  The top-level decision status is
        # independently computed as pass/fail only after every frozen gate.
        _require(str(row.get("status")) == "complete", f"run {key} is incomplete")
        _require(str(row.get("run_mode")) == "full_scientific", f"run {key} is smoke/non-scientific")
        _require(_parse_bool(row.get("full_training_authorized"), "full_training_authorized"), f"run {key} lacks full-training authorization")
        _require(_parse_bool(row.get("scientific_decision_permitted"), "scientific_decision_permitted"), f"run {key} forbids a scientific decision")
        _require(not _parse_bool(row.get("official_validation_used_for_confirmation"), "official_validation_used_for_confirmation"), f"run {key} reused official validation for confirmation")
        _require(not _parse_bool(row.get("held_out_test_accessed"), "held_out_test_accessed"), f"run {key} accessed held-out test")
    _require(observed_runs == expected_runs, "incomplete run matrix")

    task_counts = dict(spec.task_counts)
    expected_prediction_count = (
        len(spec.arms) * len(spec.seeds) * len(SIGNAL_CONDITIONS) * spec.expected_trials
    )
    _require(
        len(prediction_rows) == expected_prediction_count,
        f"unexpected prediction-row count: expected {expected_prediction_count}",
    )

    metadata: dict[str, Trial] = {}
    text_folds: dict[str, int] = {}
    raw: dict[tuple[str, int, str, str], float] = {}
    trial_sets: dict[tuple[str, int, str], set[str]] = {}
    trial_task_counts: dict[tuple[str, int, str], Counter[str]] = {}
    for row in prediction_rows:
        arm = str(row.get("arm_id"))
        seed = _parse_int(row.get("training_seed"), "training_seed")
        fold = _parse_int(row.get("outer_fold"), "outer_fold")
        trial_id = str(row.get("trial_id"))
        task = str(row.get("reading_task"))
        subject = str(row.get("subject_id"))
        text = str(row.get("normalized_text_sha256"))
        condition = str(row.get("signal_condition"))
        _require(arm in spec.arms, f"unexpected arm {arm!r}")
        _require(seed in spec.seeds, f"unexpected training seed {seed}")
        _require(fold in spec.folds, f"unexpected outer fold {fold}")
        _require(task in task_counts, f"unexpected reading task {task!r}")
        _require(condition in SIGNAL_CONDITIONS, f"unexpected signal condition {condition!r}")
        _require(trial_id != "" and subject != "" and text != "", "blank trial metadata")
        _require(_parse_bool(row.get("scientific_decision_permitted"), "scientific_decision_permitted"), "prediction row forbids scientific decision")
        _require(_parse_int(row.get("candidate_pool_size"), "candidate_pool_size") == spec.pool_size, "candidate pool size is not frozen at 24")
        rank = _parse_int(row.get("positive_rank"), "positive_rank")
        _require(1 <= rank <= spec.pool_size, f"invalid positive rank {rank}")

        trial = Trial(trial_id, fold, task, subject, text)
        if trial_id in metadata:
            _require(metadata[trial_id] == trial, f"trial metadata/fold changed for {trial_id}")
        else:
            metadata[trial_id] = trial
        if text in text_folds:
            _require(text_folds[text] == fold, f"normalized text spans outer folds: {text}")
        else:
            text_folds[text] = fold
        key = (arm, seed, condition, trial_id)
        _require(key not in raw, f"duplicate prediction {key}")
        raw[key] = 1.0 / rank
        group = (arm, seed, condition)
        trial_sets.setdefault(group, set()).add(trial_id)
        trial_task_counts.setdefault(group, Counter())[task] += 1

    _require(len(metadata) == spec.expected_trials, f"expected {spec.expected_trials} unique confirmation trials")
    _require(
        Counter(trial.outer_fold for trial in metadata.values()) == Counter(dict(spec.fold_counts)),
        f"confirmation fold counts differ from frozen counts {dict(spec.fold_counts)}",
    )
    expected_ids = set(metadata)
    for arm in spec.arms:
        for seed in spec.seeds:
            for condition in SIGNAL_CONDITIONS:
                group = (arm, seed, condition)
                _require(trial_sets.get(group) == expected_ids, f"incomplete confirmation trials for {group}")
                _require(trial_task_counts.get(group) == Counter(task_counts), f"task counts differ for {group}")

    trials = sorted(metadata.values(), key=lambda row: row.trial_id)
    _require({trial.outer_fold for trial in trials} == set(spec.folds), "not all five outer folds are represented")
    values = {
        (arm, seed, condition): [raw[(arm, seed, condition, trial.trial_id)] for trial in trials]
        for arm in spec.arms for seed in spec.seeds for condition in SIGNAL_CONDITIONS
    }
    return ValidatedData(trials=trials, values=values)


def _macro(values: Sequence[float], trials: Sequence[Trial]) -> tuple[float, dict[str, float]]:
    task_values = {
        task: [value for value, trial in zip(values, trials) if trial.reading_task == task]
        for task in TASKS
    }
    means = {task: sum(items) / len(items) for task, items in task_values.items()}
    return 0.5 * (means["NR"] + means["TSR"]), means


def _stable_seed(base_seed: int, endpoint_id: str, reference_id: str, cluster_id: str) -> int:
    serialized = "\x1f".join(map(str, (base_seed, endpoint_id, reference_id, cluster_id)))
    return int(hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16], 16)


def _clustered_bootstrap(
    values: Sequence[float],
    trials: Sequence[Trial],
    cluster_id: str,
    endpoint_id: str,
    reference_id: str,
    *,
    replicates: int,
    base_seed: int,
) -> dict[str, list[float]]:
    """Fixed-fold/task-stratum clustered bootstrap using shared multiplicities."""

    import numpy as np

    _require(cluster_id in CLUSTERS, f"unknown cluster scheme {cluster_id}")
    array = np.asarray(values, dtype=np.float64)
    _require(array.shape == (len(trials),), "bootstrap value/trial shape mismatch")
    _require(np.isfinite(array).all(), "non-finite bootstrap endpoint")
    fold_index = {fold: index for index, fold in enumerate(sorted({trial.outer_fold for trial in trials}))}
    task_index = {task: index for index, task in enumerate(TASKS)}
    strata = np.asarray([
        2 * fold_index[trial.outer_fold] + task_index[trial.reading_task]
        for trial in trials
    ], dtype=np.int64)
    stratum_count = 2 * len(fold_index)
    subject_labels = sorted({trial.subject_id for trial in trials})
    text_labels = sorted({trial.normalized_text_sha256 for trial in trials})
    subject_lookup = {label: index for index, label in enumerate(subject_labels)}
    text_lookup = {label: index for index, label in enumerate(text_labels)}
    subject_index = np.asarray([subject_lookup[trial.subject_id] for trial in trials], dtype=np.int64)
    text_index = np.asarray([text_lookup[trial.normalized_text_sha256] for trial in trials], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(_stable_seed(base_seed, endpoint_id, reference_id, cluster_id)))

    if cluster_id == "subject_id":
        label_index = subject_index
        label_count = len(subject_labels)
    elif cluster_id == "normalized_text_sha256":
        label_index = text_index
        label_count = len(text_labels)
    else:
        label_index = None
        label_count = 0

    if label_index is not None:
        counts_by_label = np.zeros((label_count, stratum_count), dtype=np.float64)
        sums_by_label = np.zeros((label_count, stratum_count), dtype=np.float64)
        np.add.at(counts_by_label, (label_index, strata), 1.0)
        np.add.at(sums_by_label, (label_index, strata), array)

    collected: list[np.ndarray] = []
    accepted = 0
    attempted = 0
    batch_size = min(256, max(16, replicates))
    while accepted < replicates:
        draw_count = min(batch_size, replicates - accepted + 32)
        attempted += draw_count
        _require(attempted <= replicates * 1000, "bootstrap empty-stratum redraw limit exceeded")
        if cluster_id != "two_way_subject_by_text":
            multiplicities = rng.multinomial(
                label_count,
                np.full(label_count, 1.0 / label_count),
                size=draw_count,
            ).astype(np.float64, copy=False)
            denominators = multiplicities @ counts_by_label
            numerators = multiplicities @ sums_by_label
        else:
            subject_multiplicity = rng.multinomial(
                len(subject_labels),
                np.full(len(subject_labels), 1.0 / len(subject_labels)),
                size=draw_count,
            ).astype(np.float64, copy=False)
            text_multiplicity = rng.multinomial(
                len(text_labels),
                np.full(len(text_labels), 1.0 / len(text_labels)),
                size=draw_count,
            ).astype(np.float64, copy=False)
            row_weights = subject_multiplicity[:, subject_index] * text_multiplicity[:, text_index]
            denominators = np.zeros((draw_count, stratum_count), dtype=np.float64)
            numerators = np.zeros((draw_count, stratum_count), dtype=np.float64)
            for stratum in range(stratum_count):
                mask = strata == stratum
                denominators[:, stratum] = row_weights[:, mask].sum(axis=1)
                numerators[:, stratum] = row_weights[:, mask] @ array[mask]

        valid = (denominators > 0.0).all(axis=1)
        if not valid.any():
            continue
        denominators = denominators[valid]
        numerators = numerators[valid]
        # Fixed folds remain present; aggregate their numerators/denominators
        # within task, then macro-average the two task means.
        nr = numerators[:, 0::2].sum(axis=1) / denominators[:, 0::2].sum(axis=1)
        tsr = numerators[:, 1::2].sum(axis=1) / denominators[:, 1::2].sum(axis=1)
        take = min(replicates - accepted, len(nr))
        collected.append(np.column_stack((0.5 * (nr[:take] + tsr[:take]), nr[:take], tsr[:take])))
        accepted += take

    draws = np.concatenate(collected, axis=0)
    return {"macro": draws[:, 0].tolist(), "NR": draws[:, 1].tolist(), "TSR": draws[:, 2].tolist()}


def _quantile(values: Sequence[float], q: float) -> float:
    import numpy as np

    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def decide(data: ValidatedData, spec: DecisionSpec = PRODUCTION_SPEC) -> dict[str, Any]:
    trials = data.trials
    seed_averages: dict[tuple[str, str], list[float]] = {}
    point_estimates: dict[str, dict[str, Any]] = {}
    seedwise: list[dict[str, Any]] = []
    for arm in spec.arms:
        for condition in SIGNAL_CONDITIONS:
            averaged = [
                sum(data.values[(arm, seed, condition)][index] for seed in spec.seeds) / len(spec.seeds)
                for index in range(len(trials))
            ]
            seed_averages[(arm, condition)] = averaged
            macro, tasks = _macro(averaged, trials)
            point_estimates.setdefault(arm, {})[condition] = {"macro_mrr": macro, "task_mrr": tasks}
        for seed in spec.seeds:
            macro, tasks = _macro(data.values[(arm, seed, "correct")], trials)
            seedwise.append({"arm_id": arm, "training_seed": seed, "macro_mrr": macro, "task_mrr": tasks})

    bootstrap_rows: list[dict[str, Any]] = []

    def interval(
        values: Sequence[float], endpoint: str, reference: str, cluster: str,
        q: float, scope: str,
    ) -> float:
        draws = _clustered_bootstrap(
            values, trials, cluster, endpoint, reference,
            replicates=spec.bootstrap_replicates, base_seed=spec.bootstrap_base_seed,
        )
        lower = _quantile(draws[scope], q)
        ci95_lower = _quantile(draws[scope], 0.025)
        ci95_upper = _quantile(draws[scope], 0.975)
        bootstrap_rows.append({
            "endpoint_id": endpoint,
            "reference_id": reference,
            "cluster": cluster,
            "scope": scope,
            "lower_quantile": q,
            "lower_bound": lower,
            "ci95_lower": ci95_lower,
            "ci95_upper": ci95_upper,
            "replicates": spec.bootstrap_replicates,
        })
        return lower

    true_correct = seed_averages[("true_task_segmented", "correct")]
    headline_point = all(
        point_estimates["true_task_segmented"]["correct"]["macro_mrr"]
        > point_estimates[baseline]["correct"]["macro_mrr"]
        for baseline in spec.baselines
    )

    superiority_checks: dict[str, dict[str, float]] = {}
    task_noninferiority_checks: dict[str, dict[str, dict[str, float]]] = {}
    signal_noninferiority_checks: dict[str, dict[str, float]] = {}
    for baseline in spec.baselines:
        baseline_correct = seed_averages[(baseline, "correct")]
        delta = [left - right for left, right in zip(true_correct, baseline_correct)]
        superiority_checks[baseline] = {
            cluster: interval(delta, "superiority_macro", baseline, cluster, spec.superiority_q, "macro")
            for cluster in CLUSTERS
        }
        task_noninferiority_checks[baseline] = {
            task: {
                cluster: interval(
                    delta, f"per_task_noninferiority_{task}", baseline, cluster,
                    spec.per_task_noninferiority_q, task,
                )
                for cluster in CLUSTERS
            }
            for task in TASKS
        }
        true_gap = [
            correct - wrong
            for correct, wrong in zip(true_correct, seed_averages[("true_task_segmented", "matched_wrong")])
        ]
        baseline_gap = [
            correct - wrong
            for correct, wrong in zip(baseline_correct, seed_averages[(baseline, "matched_wrong")])
        ]
        gap_delta = [left - right for left, right in zip(true_gap, baseline_gap)]
        signal_noninferiority_checks[baseline] = {
            cluster: interval(
                gap_delta, "signal_gap_noninferiority", baseline, cluster,
                spec.signal_noninferiority_q, "macro",
            )
            for cluster in CLUSTERS
        }

    chance_checks = {
        cluster: interval(true_correct, "absolute_true_macro", "chance_mrr_24_way", cluster, spec.chance_q, "macro")
        for cluster in CLUSTERS
    }
    true_signal_gap = [
        correct - wrong
        for correct, wrong in zip(true_correct, seed_averages[("true_task_segmented", "matched_wrong")])
    ]
    signal_positive_checks = {
        cluster: interval(true_signal_gap, "true_signal_gap", "matched_wrong", cluster, spec.signal_positive_q, "macro")
        for cluster in CLUSTERS
    }

    seedwise_positive = True
    seedwise_deltas: list[dict[str, Any]] = []
    for baseline in spec.baselines:
        for seed in spec.seeds:
            true_macro, _ = _macro(data.values[("true_task_segmented", seed, "correct")], trials)
            baseline_macro, _ = _macro(data.values[(baseline, seed, "correct")], trials)
            delta = true_macro - baseline_macro
            seedwise_deltas.append({"reference_id": baseline, "training_seed": seed, "macro_mrr_delta": delta})
            seedwise_positive = seedwise_positive and delta > 0.0

    requirements = {
        "headline_macro_exceeds_both_baselines": headline_point,
        "superiority_all_cluster_familywise_lower_bounds_positive": all(
            lower > 0.0 for checks in superiority_checks.values() for lower in checks.values()
        ),
        "both_deltas_positive_in_all_training_seeds": seedwise_positive,
        "all_task_by_baseline_lower_bounds_above_minus_0_01": all(
            lower > -spec.per_task_noninferiority_margin
            for baseline in task_noninferiority_checks.values()
            for task in baseline.values() for lower in task.values()
        ),
        "absolute_true_macro_lower_bound_above_chance": all(
            lower > spec.chance_mrr for lower in chance_checks.values()
        ),
        "true_signal_gap_all_cluster_lower_bounds_positive": all(
            lower > 0.0 for lower in signal_positive_checks.values()
        ),
        "true_signal_gap_noninferior_to_both_baselines": all(
            lower > -spec.signal_noninferiority_margin
            for checks in signal_noninferiority_checks.values() for lower in checks.values()
        ),
    }
    all_pass = all(requirements.values())
    return {
        "schema_version": 1,
        "status": "pass" if all_pass else "fail",
        "scientific_decision_permitted": True,
        "held_out_test_accessed": False,
        "validated_input": {
            "runs": len(spec.arms) * len(spec.folds) * len(spec.seeds),
            "confirmation_trials": len(trials),
            "task_counts": dict(spec.task_counts),
            "fold_counts": dict(spec.fold_counts),
            "arms": list(spec.arms),
            "training_seeds": list(spec.seeds),
            "outer_folds": list(spec.folds),
        },
        "point_estimates": point_estimates,
        "seedwise_deltas": seedwise_deltas,
        "bootstrap_intervals": bootstrap_rows,
        "requirements": requirements,
        "continuation_decision": "continue_p4b" if all_pass else "stop_p4b_permanently",
        "failure_action": None if all_pass else "no retuning, richer-factor training, generation training, or held-out-test access",
    }


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runs = read_csv(args.run_manifest, RUN_FIELDS)
    predictions = read_csv(args.predictions, PREDICTION_FIELDS)
    data = validate_inputs(runs, predictions, PRODUCTION_SPEC)
    result = decide(data, PRODUCTION_SPEC)
    atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "continuation_decision": result["continuation_decision"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
