"""Freeze the prospective P4b task-segmented-objective protocol.

The production entry point first revalidates exact prompt-neutral Kaggle input
version 2.  It then uses metadata and vector locators only: no vector array,
model weight, official-validation outcome, or held-out-test row is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


CONTRACT_NAME = "task_segmented_objective_contract.json"
REPORT_NAME = "task_segmented_protocol_report.json"
ARTIFACTS = (
    "text_group_folds.csv",
    "outer_split_assignments.csv",
    "pseudo_groups.csv",
    "batch_grid_feasibility.csv",
    "confirmation_donors.csv",
    "candidate_pools.csv",
    "protocol_registry.json",
)
ELIGIBILITY = {
    "split": "train",
    "cohort": "primary_zuco2_nr_tsr",
    "dataset_version": "ZuCo2",
    "reading_tasks": {"NR", "TSR"},
}
LENGTH_STRATA = (
    (0, 5, "0-5"),
    (6, 10, "6-10"),
    (11, 15, "11-15"),
    (16, 20, "16-20"),
    (21, 30, "21-30"),
    (31, None, "31+"),
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: object, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in {field}: {value!r}") from exc


def length_stratum(length: int) -> str:
    if length < 0:
        raise ValueError(f"negative whitespace length: {length}")
    for lower, upper, label in LENGTH_STRATA:
        if length >= lower and (upper is None or length <= upper):
            return label
    raise AssertionError(length)


def load_contract(path: Path) -> dict:
    contract = read_json(path)
    if contract.get("schema_version") != 1:
        raise ValueError("P4b contract schema must be 1")
    if contract.get("status") != "prospectively_frozen_before_p4b_training":
        raise ValueError("P4b contract is not prospectively frozen")
    if contract.get("cross_fitting", {}).get("folds") != 5:
        raise ValueError("P4b requires five outer folds")
    if contract.get("evaluation", {}).get("pool_size") != 24:
        raise ValueError("P4b requires 24-way evaluation pools")
    if contract.get("batches", {}).get("batch_size") != 64:
        raise ValueError("P4b requires common 64-row batches")
    if contract.get("training", {}).get("total_runs") != 45:
        raise ValueError("P4b requires 45 fold/arm/seed fits")
    return contract


def verify_preserved_input(artifact_root: Path, preserved_source_id: str) -> dict:
    try:
        from evaluation.verify_prompt_neutral_pilot_inputs import verify
    except ModuleNotFoundError:  # direct script execution from evaluation/
        from verify_prompt_neutral_pilot_inputs import verify
    return verify(artifact_root, preserved_source_id)


def load_eligible_rows(artifact_root: Path, contract: dict) -> list[dict[str, object]]:
    eeg_rows = read_csv(artifact_root / "eeg" / "vector_index.csv")
    mapping_rows = read_csv(artifact_root / "text" / "trial_text_targets.csv")
    text_rows = read_csv(artifact_root / "text" / "text_vector_index.csv")
    mapping = {row["trial_id"]: row for row in mapping_rows}
    if len(mapping) != len(mapping_rows):
        raise ValueError("duplicate trial ID in trial-text mapping")
    text_catalog = {row["text_target_id"]: row for row in text_rows}
    if len(text_catalog) != len(text_rows):
        raise ValueError("duplicate text target in text-vector index")

    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for eeg in eeg_rows:
        if eeg["condition"] != contract["input"]["eeg_condition"]:
            continue
        if eeg["phase"] != "train":
            raise ValueError("correct_train vector has a non-train phase")
        trial_id = eeg["target_trial_id"]
        if trial_id in seen:
            raise ValueError(f"duplicate correct_train trial: {trial_id}")
        seen.add(trial_id)
        target = mapping.get(trial_id)
        if target is None:
            raise ValueError(f"correct_train trial lacks text target: {trial_id}")
        for field in ("dataset_version", "reading_task", "subject_id"):
            if eeg[field] != target[field]:
                raise ValueError(f"EEG/text {field} mismatch for {trial_id}")
        if target["split"] != "train":
            raise ValueError(f"correct_train text mapping is not train: {trial_id}")
        if not (
            target["cohort"] == ELIGIBILITY["cohort"]
            and target["dataset_version"] == ELIGIBILITY["dataset_version"]
            and target["reading_task"] in ELIGIBILITY["reading_tasks"]
        ):
            continue
        text = text_catalog.get(target["text_target_id"])
        if text is None:
            raise ValueError(f"missing text vector target: {target['text_target_id']}")
        if text["normalized_text_sha256"] != target["normalized_text_sha256"]:
            raise ValueError(f"normalized-text mismatch for {trial_id}")
        representative = text["representative_text"]
        output.append(
            {
                "trial_id": trial_id,
                "split": target["split"],
                "cohort": target["cohort"],
                "dataset_version": target["dataset_version"],
                "reading_task": target["reading_task"],
                "subject_id": target["subject_id"],
                "normalized_text_sha256": target["normalized_text_sha256"],
                "text_target_id": target["text_target_id"],
                "text": representative,
                "length_words_whitespace_v1": len(representative.split()),
                "eeg_vector_file": eeg["vector_file"],
                "eeg_vector_offset": as_int(eeg["vector_offset"], "vector_offset"),
                "eeg_vector_dim": as_int(eeg["vector_dim"], "vector_dim"),
            }
        )
    if not output:
        raise ValueError("no eligible primary ZuCo2 NR/TSR train rows")
    return output


def validate_rows(rows: list[dict[str, object]]) -> None:
    trials: set[str] = set()
    text_values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        trial_id = str(row["trial_id"])
        if trial_id in trials:
            raise ValueError(f"duplicate eligible trial ID: {trial_id}")
        trials.add(trial_id)
        if row["split"] != ELIGIBILITY["split"]:
            raise ValueError("P4b input contains a non-train row")
        if row["cohort"] != ELIGIBILITY["cohort"]:
            raise ValueError("P4b input contains an ineligible cohort")
        if row["dataset_version"] != ELIGIBILITY["dataset_version"]:
            raise ValueError("P4b input contains an ineligible dataset")
        if row["reading_task"] not in ELIGIBILITY["reading_tasks"]:
            raise ValueError("P4b input contains an ineligible task")
        if as_int(row["length_words_whitespace_v1"], "length") < 1:
            raise ValueError("empty text is not eligible")
        if "eeg_vector_dim" in row and as_int(row["eeg_vector_dim"], "eeg_vector_dim") != 1024:
            raise ValueError("P4b requires 1024-D EEG vectors")
        text_values[str(row["normalized_text_sha256"])].add(str(row["text_target_id"]))
    collisions = [identity for identity, values in text_values.items() if len(values) != 1]
    if collisions:
        raise ValueError(f"normalized-text identity collision: {collisions[0]}")
    if {str(row["reading_task"]) for row in rows} != {"NR", "TSR"}:
        raise ValueError("P4b requires both NR and TSR")


def assign_group_folds(
    rows: list[dict[str, object]], folds: int, seed: int
) -> tuple[dict[str, int], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["normalized_text_sha256"])].append(row)
    total_rows = Counter(str(row["reading_task"]) for row in rows)
    total_identities = Counter()
    group_stats: dict[str, dict[str, int]] = {}
    for identity, members in grouped.items():
        row_counts = Counter(str(row["reading_task"]) for row in members)
        stats = {
            "NR_rows": row_counts["NR"],
            "TSR_rows": row_counts["TSR"],
            "NR_identities": int(row_counts["NR"] > 0),
            "TSR_identities": int(row_counts["TSR"] > 0),
        }
        group_stats[identity] = stats
        total_identities["NR"] += stats["NR_identities"]
        total_identities["TSR"] += stats["TSR_identities"]

    fold_stats = [Counter() for _ in range(folds)]
    assignment: dict[str, int] = {}
    order = sorted(
        grouped,
        key=lambda identity: (
            -len(grouped[identity]),
            -(group_stats[identity]["NR_identities"] + group_stats[identity]["TSR_identities"]),
            stable_hash(seed, identity, "group-order"),
        ),
    )

    def total_cost(candidate_fold: int, stats: dict[str, int]) -> float:
        cost = 0.0
        for fold_index, current in enumerate(fold_stats):
            for task in ("NR", "TSR"):
                add_rows = stats[f"{task}_rows"] if fold_index == candidate_fold else 0
                add_ids = stats[f"{task}_identities"] if fold_index == candidate_fold else 0
                row_target = max(total_rows[task] / folds, 1.0)
                id_target = max(total_identities[task] / folds, 1.0)
                cost += ((current[f"{task}_rows"] + add_rows - row_target) / row_target) ** 2
                cost += ((current[f"{task}_identities"] + add_ids - id_target) / id_target) ** 2
        return cost

    for identity in order:
        stats = group_stats[identity]
        selected = min(
            range(folds),
            key=lambda fold: (
                total_cost(fold, stats),
                stable_hash(seed, identity, fold, "fold-tie"),
            ),
        )
        assignment[identity] = selected
        fold_stats[selected].update(stats)

    fold_rows = []
    for identity in sorted(grouped):
        stats = group_stats[identity]
        lengths = {as_int(row["length_words_whitespace_v1"], "length") for row in grouped[identity]}
        if len(lengths) != 1:
            raise ValueError(f"one text identity has inconsistent lengths: {identity}")
        fold_rows.append(
            {
                "normalized_text_sha256": identity,
                "fold": assignment[identity],
                "length_words_whitespace_v1": next(iter(lengths)),
                **stats,
            }
        )
    if set(assignment.values()) != set(range(folds)):
        raise ValueError("greedy partition left an empty outer fold")
    return assignment, fold_rows


def build_pseudo_groups(rows: list[dict[str, object]], seed: int) -> tuple[dict[tuple[str, str], int], list[dict[str, object]]]:
    identities: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["reading_task"]), str(row["normalized_text_sha256"]))
        length = as_int(row["length_words_whitespace_v1"], "length")
        previous = identities.get(key)
        if previous is not None and previous["length"] != length:
            raise ValueError(f"task-text identity has inconsistent length: {key}")
        identities[key] = {"length": length, "stratum": length_stratum(length)}
    by_stratum: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, values in identities.items():
        by_stratum[(key[0], str(values["stratum"]))].append(key)

    assignment: dict[tuple[str, str], int] = {}
    output: list[dict[str, object]] = []
    for (task, stratum), keys in sorted(by_stratum.items()):
        ordered = sorted(keys, key=lambda key: stable_hash(seed, *key, stratum, "pseudo-order"))
        offset = int(stable_hash(seed, task, stratum, "pseudo-offset")[:8], 16) % 2
        for index, key in enumerate(ordered):
            group = (index + offset) % 2
            assignment[key] = group
            output.append(
                {
                    "reading_task": task,
                    "normalized_text_sha256": key[1],
                    "length_words_whitespace_v1": identities[key]["length"],
                    "length_stratum": stratum,
                    "pseudo_group": group,
                }
            )
    output.sort(key=lambda row: (str(row["reading_task"]), str(row["normalized_text_sha256"])))
    return assignment, output


def role_for_fold(text_fold: int, outer_fold: int, folds: int) -> str:
    if text_fold == outer_fold:
        return "confirmation"
    if text_fold == (outer_fold + 1) % folds:
        return "checkpoint"
    return "fit"


def build_outer_assignments(
    rows: list[dict[str, object]], group_folds: dict[str, int], pseudo: dict[tuple[str, str], int], folds: int
) -> list[dict[str, object]]:
    output = []
    for outer_fold in range(folds):
        for row in sorted(rows, key=lambda item: str(item["trial_id"])):
            identity = str(row["normalized_text_sha256"])
            task = str(row["reading_task"])
            text_fold = group_folds[identity]
            output.append(
                {
                    "outer_fold": outer_fold,
                    "role": role_for_fold(text_fold, outer_fold, folds),
                    "text_fold": text_fold,
                    "trial_id": row["trial_id"],
                    "dataset_version": row["dataset_version"],
                    "reading_task": task,
                    "subject_id": row["subject_id"],
                    "normalized_text_sha256": identity,
                    "text_target_id": row["text_target_id"],
                    "pseudo_group": pseudo[(task, identity)],
                    "length_words_whitespace_v1": row["length_words_whitespace_v1"],
                    "eeg_vector_file": row.get("eeg_vector_file", ""),
                    "eeg_vector_offset": row.get("eeg_vector_offset", ""),
                    "eeg_vector_dim": row.get("eeg_vector_dim", 1024),
                }
            )
    return output


def build_batch_feasibility(assignments: list[dict[str, object]], minimum: int) -> list[dict[str, object]]:
    output = []
    for outer_fold in sorted({as_int(row["outer_fold"], "outer_fold") for row in assignments}):
        fit = [row for row in assignments if row["outer_fold"] == outer_fold and row["role"] == "fit"]
        candidates_by_cell: dict[tuple[str, int], set[str]] = {}
        for task in ("NR", "TSR"):
            for group in (0, 1):
                members = [
                    row for row in fit
                    if row["reading_task"] == task and as_int(row["pseudo_group"], "pseudo_group") == group
                ]
                identities = {str(row["normalized_text_sha256"]) for row in members}
                if len(identities) < minimum:
                    raise ValueError(
                        f"outer fold {outer_fold} cell {task}::{group} has {len(identities)} "
                        f"identities; needs {minimum}"
                    )
                candidates_by_cell[(task, group)] = identities
                output.append(
                    {
                        "outer_fold": outer_fold,
                        "reading_task": task,
                        "pseudo_group": group,
                        "fit_trial_rows": len(members),
                        "fit_unique_text_identities": len(identities),
                        "required_unique_identities_per_batch": minimum,
                        "common_batch_unique_text_grid_feasible": 1,
                        "status": "pass",
                    }
                )
        slots = [cell for cell in sorted(candidates_by_cell) for _ in range(minimum)]
        identity_to_slot: dict[str, int] = {}

        def augment(slot_index: int, seen: set[str]) -> bool:
            cell = slots[slot_index]
            for identity in sorted(candidates_by_cell[cell]):
                if identity in seen:
                    continue
                seen.add(identity)
                previous = identity_to_slot.get(identity)
                if previous is None or augment(previous, seen):
                    identity_to_slot[identity] = slot_index
                    return True
            return False

        if not all(augment(slot_index, set()) for slot_index in range(len(slots))):
            raise ValueError(
                f"outer fold {outer_fold} cannot form one 64-row grid with globally unique texts"
            )
    return output


def build_confirmation_donors(
    rows: list[dict[str, object]], group_folds: dict[str, int], folds: int, seed: int
) -> tuple[list[dict[str, object]], dict[tuple[int, str], dict[str, object]]]:
    output = []
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for outer_fold in range(folds):
        targets = [
            row for row in rows
            if group_folds[str(row["normalized_text_sha256"])] == outer_fold
        ]
        for target in sorted(targets, key=lambda item: str(item["trial_id"])):
            candidates = [
                row for row in targets
                if row["dataset_version"] == target["dataset_version"]
                and row["reading_task"] == target["reading_task"]
                and row["subject_id"] == target["subject_id"]
                and row["normalized_text_sha256"] != target["normalized_text_sha256"]
            ]
            if not candidates:
                raise ValueError(f"no same-fold/task/subject donor for {target['trial_id']}")
            target_length = as_int(target["length_words_whitespace_v1"], "length")
            donor = min(
                candidates,
                key=lambda row: (
                    abs(as_int(row["length_words_whitespace_v1"], "length") - target_length),
                    stable_hash(seed, outer_fold, target["trial_id"], row["trial_id"], "donor"),
                ),
            )
            record = {
                "outer_fold": outer_fold,
                "partition": "confirmation",
                "target_trial_id": target["trial_id"],
                "donor_trial_id": donor["trial_id"],
                "dataset_version": target["dataset_version"],
                "reading_task": target["reading_task"],
                "subject_id": target["subject_id"],
                "target_normalized_text_sha256": target["normalized_text_sha256"],
                "donor_normalized_text_sha256": donor["normalized_text_sha256"],
                "target_length": target_length,
                "donor_length": donor["length_words_whitespace_v1"],
                "absolute_length_difference": abs(
                    as_int(donor["length_words_whitespace_v1"], "length") - target_length
                ),
                "selection_rule": "same_fold_partition_dataset_task_subject_different_text_then_length_then_seeded_hash",
            }
            output.append(record)
            lookup[(outer_fold, str(target["trial_id"]))] = donor
    return output, lookup


def _candidate_catalog(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    catalog: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["reading_task"]), str(row["normalized_text_sha256"]))
        previous = catalog.get(key)
        if previous is None or str(row["trial_id"]) < str(previous["trial_id"]):
            catalog[key] = row
    return catalog


def build_candidate_pools(
    rows: list[dict[str, object]], group_folds: dict[str, int], donors: dict[tuple[int, str], dict[str, object]],
    folds: int, pool_size: int, seed: int,
) -> list[dict[str, object]]:
    output = []
    for outer_fold in range(folds):
        for partition, text_fold in (
            ("confirmation", outer_fold),
            ("checkpoint", (outer_fold + 1) % folds),
        ):
            targets = [
                row for row in rows
                if group_folds[str(row["normalized_text_sha256"])] == text_fold
            ]
            catalog = _candidate_catalog(targets)
            by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
            for (task, _), candidate in catalog.items():
                by_task[task].append(candidate)
            for target in sorted(targets, key=lambda item: str(item["trial_id"])):
                task = str(target["reading_task"])
                positive_identity = str(target["normalized_text_sha256"])
                donor = donors.get((outer_fold, str(target["trial_id"]))) if partition == "confirmation" else None
                donor_identity = str(donor["normalized_text_sha256"]) if donor is not None else ""
                target_length = as_int(target["length_words_whitespace_v1"], "length")
                eligible = [
                    candidate for candidate in by_task[task]
                    if candidate["normalized_text_sha256"] not in {positive_identity, donor_identity}
                ]
                eligible.sort(
                    key=lambda candidate: (
                        abs(as_int(candidate["length_words_whitespace_v1"], "length") - target_length),
                        stable_hash(
                            seed, outer_fold, partition, target["trial_id"],
                            candidate["normalized_text_sha256"], "pool-select",
                        ),
                    )
                )
                needed = pool_size - 1 - int(donor is not None)
                if len(eligible) < needed:
                    raise ValueError(
                        f"{partition} target {target['trial_id']} has {len(eligible)} optional candidates; "
                        f"needs {needed}"
                    )
                positive = catalog[(task, positive_identity)]
                selected = [positive]
                if donor is not None:
                    selected.append(catalog[(task, donor_identity)])
                selected.extend(eligible[:needed])
                if len({str(row["normalized_text_sha256"]) for row in selected}) != pool_size:
                    raise AssertionError("candidate text identities are not unique")
                selected.sort(
                    key=lambda candidate: stable_hash(
                        seed, outer_fold, partition, target["trial_id"],
                        candidate["normalized_text_sha256"], "pool-order",
                    )
                )
                for rank, candidate in enumerate(selected):
                    identity = str(candidate["normalized_text_sha256"])
                    output.append(
                        {
                            "outer_fold": outer_fold,
                            "partition": partition,
                            "target_trial_id": target["trial_id"],
                            "candidate_rank": rank,
                            "candidate_normalized_text_sha256": identity,
                            "candidate_text_target_id": candidate["text_target_id"],
                            "is_positive": int(identity == positive_identity),
                            "is_designated_donor_text": int(bool(donor_identity) and identity == donor_identity),
                            "dataset_version": target["dataset_version"],
                            "reading_task": task,
                            "target_length": target_length,
                            "candidate_length": candidate["length_words_whitespace_v1"],
                            "absolute_length_difference": abs(
                                as_int(candidate["length_words_whitespace_v1"], "length") - target_length
                            ),
                            "selection_rule": (
                                "same_partition_dataset_task_positive_donor_then_length_seeded_hash"
                                if partition == "confirmation"
                                else "same_partition_dataset_task_positive_then_length_seeded_hash"
                            ),
                        }
                    )
    return output


def audit_protocol(
    rows: list[dict[str, object]], group_folds: dict[str, int], assignments: list[dict[str, object]],
    donors: list[dict[str, object]], pools: list[dict[str, object]], folds: int, pool_size: int,
) -> dict[str, object]:
    for outer_fold in range(folds):
        roles: dict[str, set[str]] = defaultdict(set)
        for row in assignments:
            if row["outer_fold"] == outer_fold:
                roles[str(row["role"])].add(str(row["normalized_text_sha256"]))
        if set(roles) != {"fit", "checkpoint", "confirmation"}:
            raise ValueError(f"outer fold {outer_fold} lacks a required role")
        if any(roles[left] & roles[right] for left in roles for right in roles if left < right):
            raise ValueError(f"cross-partition normalized-text leakage in outer fold {outer_fold}")

    row_by_trial = {str(row["trial_id"]): row for row in rows}
    for donor in donors:
        target = row_by_trial[str(donor["target_trial_id"])]
        source = row_by_trial[str(donor["donor_trial_id"])]
        if not (
            target["dataset_version"] == source["dataset_version"]
            and target["reading_task"] == source["reading_task"]
            and target["subject_id"] == source["subject_id"]
            and target["normalized_text_sha256"] != source["normalized_text_sha256"]
            and group_folds[str(target["normalized_text_sha256"])]
            == group_folds[str(source["normalized_text_sha256"])]
            == as_int(donor["outer_fold"], "outer_fold")
        ):
            raise ValueError("invalid confirmation donor")

    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pools:
        grouped[(as_int(row["outer_fold"], "outer_fold"), str(row["partition"]), str(row["target_trial_id"]))].append(row)
    expected_target_pools = len(rows) * 2
    if len(grouped) != expected_target_pools:
        raise ValueError(f"expected {expected_target_pools} target pools, got {len(grouped)}")
    for (_, partition, _), members in grouped.items():
        if len(members) != pool_size or sum(as_int(row["is_positive"], "is_positive") for row in members) != 1:
            raise ValueError("candidate-pool size/positive invariant failed")
        donor_count = sum(as_int(row["is_designated_donor_text"], "is_donor") for row in members)
        if donor_count != int(partition == "confirmation"):
            raise ValueError("confirmation donor-text invariant failed")
        if len({str(row["candidate_normalized_text_sha256"]) for row in members}) != pool_size:
            raise ValueError("candidate pool repeats a text identity")
    return {
        "all_rows_canonical_train_only": True,
        "only_primary_zuco2_nr_tsr": True,
        "global_text_groups_cross_fitted_without_leakage": True,
        "checkpoint_and_confirmation_candidates_partition_local": True,
        "confirmation_donors_same_fold_dataset_task_subject": True,
        "confirmation_donors_always_different_text": True,
        "candidate_pool_size_exact": True,
        "one_positive_per_pool": True,
        "designated_donor_text_in_every_confirmation_pool": True,
        "candidate_selection_uses_model_scores": False,
        "model_or_vector_array_loaded": False,
        "full_training_batch_schedule_frozen": False,
        "training_authorized": False,
        "official_validation_used": False,
        "held_out_test_accessed": False,
    }


def protocol_registry(contract: dict, contract_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "partition_and_evaluation_protocol_frozen_batch_schedule_pending",
        "contract_sha256": contract_sha256,
        "input": contract["input"],
        "eligibility": contract["eligibility"],
        "cross_fitting": contract["cross_fitting"],
        "pseudo_groups": contract["pseudo_groups"],
        "arms": contract["arms"],
        "shared_model": contract["shared_model"],
        "batches": contract["batches"],
        "training": contract["training"],
        "evaluation": contract["evaluation"],
        "uncertainty": contract["uncertainty"],
        "continuation": contract["continuation"],
        "training_authorized": False,
        "held_out_test_accessed": False,
    }


def freeze_rows(
    rows: list[dict[str, object]], output_root: Path, contract: dict, contract_sha256: str,
    input_verification: dict[str, object] | None = None,
) -> dict[str, object]:
    validate_rows(rows)
    folds = as_int(contract["cross_fitting"]["folds"], "folds")
    group_folds, fold_rows = assign_group_folds(
        rows, folds, as_int(contract["cross_fitting"]["partition_seed"], "partition_seed")
    )
    pseudo, pseudo_rows = build_pseudo_groups(rows, as_int(contract["pseudo_groups"]["seed"], "pseudo_seed"))
    assignments = build_outer_assignments(rows, group_folds, pseudo, folds)
    batch_feasibility = build_batch_feasibility(
        assignments, as_int(contract["batches"]["examples_per_cell"], "examples_per_cell")
    )
    donor_rows, donor_lookup = build_confirmation_donors(
        rows, group_folds, folds, as_int(contract["evaluation"]["donor_seed"], "donor_seed")
    )
    pool_rows = build_candidate_pools(
        rows, group_folds, donor_lookup, folds,
        as_int(contract["evaluation"]["pool_size"], "pool_size"),
        as_int(contract["evaluation"]["pool_seed"], "pool_seed"),
    )
    checks = audit_protocol(
        rows, group_folds, assignments, donor_rows, pool_rows, folds,
        as_int(contract["evaluation"]["pool_size"], "pool_size"),
    )
    checks["common_64_identity_grid_feasible_in_every_outer_fit"] = True

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "text_group_folds.csv",
        [
            "normalized_text_sha256", "fold", "length_words_whitespace_v1",
            "NR_rows", "TSR_rows", "NR_identities", "TSR_identities",
        ],
        fold_rows,
    )
    write_csv(
        output_root / "outer_split_assignments.csv",
        [
            "outer_fold", "role", "text_fold", "trial_id", "dataset_version", "reading_task",
            "subject_id", "normalized_text_sha256", "text_target_id", "pseudo_group",
            "length_words_whitespace_v1", "eeg_vector_file", "eeg_vector_offset", "eeg_vector_dim",
        ],
        assignments,
    )
    write_csv(
        output_root / "pseudo_groups.csv",
        [
            "reading_task", "normalized_text_sha256", "length_words_whitespace_v1",
            "length_stratum", "pseudo_group",
        ],
        pseudo_rows,
    )
    write_csv(
        output_root / "batch_grid_feasibility.csv",
        [
            "outer_fold", "reading_task", "pseudo_group", "fit_trial_rows",
            "fit_unique_text_identities", "required_unique_identities_per_batch", "status",
            "common_batch_unique_text_grid_feasible",
        ],
        batch_feasibility,
    )
    write_csv(
        output_root / "confirmation_donors.csv",
        [
            "outer_fold", "partition", "target_trial_id", "donor_trial_id", "dataset_version",
            "reading_task", "subject_id", "target_normalized_text_sha256",
            "donor_normalized_text_sha256", "target_length", "donor_length",
            "absolute_length_difference", "selection_rule",
        ],
        donor_rows,
    )
    write_csv(
        output_root / "candidate_pools.csv",
        [
            "outer_fold", "partition", "target_trial_id", "candidate_rank",
            "candidate_normalized_text_sha256", "candidate_text_target_id", "is_positive",
            "is_designated_donor_text", "dataset_version", "reading_task", "target_length",
            "candidate_length", "absolute_length_difference", "selection_rule",
        ],
        pool_rows,
    )
    registry = protocol_registry(contract, contract_sha256)
    (output_root / "protocol_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = {
        "status": "pass",
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "input_verification": input_verification or {"status": "synthetic_test_fixture"},
        "counts": {
            "eligible_rows": len(rows),
            "normalized_text_groups": len(group_folds),
            "outer_split_assignment_rows": len(assignments),
            "pseudo_task_text_identities": len(pseudo_rows),
            "confirmation_donors": len(donor_rows),
            "candidate_pools": len(rows) * 2,
            "candidate_pool_rows": len(pool_rows),
            "task_rows": dict(sorted(Counter(str(row["reading_task"]) for row in rows).items())),
            "fold_rows": dict(sorted(Counter(group_folds[str(row["normalized_text_sha256"])] for row in rows).items())),
        },
        "execution_readiness": {
            "partition_pseudo_donor_and_pool_manifests_frozen": True,
            "batch_grid_cell_feasibility_checked": True,
            "full_40_epoch_batch_schedule_frozen": False,
            "training_authorized": False,
            "next_required_artifact": "one common per-seed per-outer-fold 40-epoch batch schedule, frozen and hashed before any fit",
        },
        "checks": checks,
        "artifact_sha256": {name: sha256(output_root / name) for name in ARTIFACTS},
    }
    (output_root / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def freeze(artifact_root: Path, output_root: Path, contract_path: Path, preserved_source_id: str) -> dict[str, object]:
    contract = load_contract(contract_path)
    if preserved_source_id != contract["input"]["preserved_source_id"]:
        raise ValueError("preserved source ID differs from the frozen contract")
    verification = verify_preserved_input(artifact_root, preserved_source_id)
    if verification.get("status") != "pass":
        raise ValueError("prompt-neutral input verification did not pass")
    if verification.get("combined_manifest_sha256") != contract["input"]["combined_manifest_sha256"]:
        raise ValueError("verified input manifest differs from the frozen contract")
    if verification.get("eeg", {}).get("vector_index_sha256") != contract["input"]["eeg_vector_index_sha256"]:
        raise ValueError("verified EEG index differs from the frozen contract")
    if verification.get("text", {}).get("text_vector_index_sha256") != contract["input"]["text_vector_index_sha256"]:
        raise ValueError("verified text index differs from the frozen contract")
    if verification.get("text", {}).get("trial_text_targets_sha256") != contract["input"]["trial_text_targets_sha256"]:
        raise ValueError("verified trial-text mapping differs from the frozen contract")
    rows = load_eligible_rows(artifact_root, contract)
    verification_summary = {
        "status": "pass",
        "preserved_source_id": preserved_source_id,
        "combined_manifest_sha256": verification["combined_manifest_sha256"],
        "eeg_vector_index_sha256": verification["eeg"]["vector_index_sha256"],
        "text_vector_index_sha256": verification["text"]["text_vector_index_sha256"],
        "trial_text_targets_sha256": verification["text"]["trial_text_targets_sha256"],
        "all_215_chunk_hashes_revalidated": verification["checks"]["all_215_chunk_hashes_revalidated"],
        "held_out_test_accessed": False,
    }
    return freeze_rows(rows, output_root, contract, sha256(contract_path), verification_summary)


def verify_frozen_output(output_root: Path) -> dict[str, object]:
    report = read_json(output_root / REPORT_NAME)
    if report.get("status") != "pass":
        raise ValueError("frozen P4b report is not pass")
    expected = report.get("artifact_sha256")
    if not isinstance(expected, dict) or set(expected) != set(ARTIFACTS):
        raise ValueError("frozen P4b report has an invalid artifact hash inventory")
    for name in ARTIFACTS:
        actual = sha256(output_root / name)
        if actual != expected[name]:
            raise ValueError(f"frozen P4b artifact hash mismatch: {name}")
    registry = read_json(output_root / "protocol_registry.json")
    if registry.get("contract_sha256") != report.get("contract_sha256"):
        raise ValueError("P4b registry/report contract hash mismatch")
    if registry.get("held_out_test_accessed") is not False:
        raise ValueError("P4b registry test-access flag is not false")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name(CONTRACT_NAME),
    )
    parser.add_argument(
        "--preserved-source-id",
        default="kaggle-dataset-thestonedape-task-aware-eegtotext-version-2",
    )
    parser.add_argument("--verify-output-only", action="store_true")
    args = parser.parse_args()
    if args.verify_output_only:
        report = verify_frozen_output(args.output_root)
    else:
        if args.artifact_root is None:
            parser.error("--artifact-root is required unless --verify-output-only is used")
        report = freeze(args.artifact_root, args.output_root, args.contract, args.preserved_source_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TASK-SEGMENTED OBJECTIVE PROTOCOL FREEZE: PASS")


if __name__ == "__main__":
    main()
