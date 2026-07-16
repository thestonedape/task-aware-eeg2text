"""Freeze leakage-safe retrieval and EEG-control manifests from canonical shards.

This builder is deliberately metadata-only: it never reads an EEG array and it
never uses a model score.  The current development gate permits train/validation
manifests only; the held-out test split remains inaccessible until final runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ARTIFACTS = (
    "evaluation_rows.csv",
    "candidate_catalog.csv",
    "candidate_pools.csv",
    "wrong_eeg_donors.csv",
    "feature_masks.csv",
    "protocol_registry.json",
)

REQUIRED_FIELDS = {
    "dataset_version",
    "reading_task",
    "subject_id",
    "text_uid",
    "trial_id",
    "split",
    "cohort",
    "text",
    "length_words_whitespace_v1",
    "mask_sr_sentiment_3",
    "mask_nr_relation_content",
    "mask_tsr_instruction_relation",
    "mask_gpt2_mean_nll_v1",
    "mask_semkey_sentiment_2",
    "mask_topic_label",
}

MASK_FIELDS = (
    "mask_sr_sentiment_3",
    "mask_nr_relation_content",
    "mask_tsr_instruction_relation",
    "mask_gpt2_mean_nll_v1",
    "mask_semkey_sentiment_2",
    "mask_topic_label",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_text(text: object) -> str:
    return " ".join(str(text).lower().split())


def as_int(value: object, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in {field}: {value!r}") from exc


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_index(dataset_root: Path) -> tuple[list[dict[str, str]], dict[str, object], Path]:
    manifest_path = dataset_root / "metadata" / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError(f"expected canonical schema_version 2, got {manifest.get('schema_version')!r}")
    index_path = dataset_root / str(manifest["index"])
    if sha256(index_path) != manifest.get("index_sha256"):
        raise ValueError("canonical index SHA256 does not match shard_manifest.json")
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"canonical index missing fields: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != int(manifest["row_count"]):
        raise ValueError("canonical index row count does not match shard_manifest.json")
    return rows, manifest, index_path


def audit_split_identity(rows: list[dict[str, str]]) -> dict[str, int]:
    trial_ids: set[str] = set()
    uid_splits: dict[str, set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        trial_id = row["trial_id"]
        if trial_id in trial_ids:
            raise ValueError(f"duplicate trial_id: {trial_id}")
        trial_ids.add(trial_id)
        uid_splits[row["text_uid"]].add(row["split"])
        text_splits[normalized_text(row["text"])].add(row["split"])
    uid_overlap = sum(len(splits) > 1 for splits in uid_splits.values())
    text_overlap = sum(len(splits) > 1 for splits in text_splits.values())
    if uid_overlap or text_overlap:
        raise ValueError(
            f"split leakage detected: {uid_overlap} text UIDs and "
            f"{text_overlap} normalized texts cross splits"
        )
    return {
        "unique_trial_id_count": len(trial_ids),
        "cross_split_text_uid_count": uid_overlap,
        "cross_split_normalized_text_count": text_overlap,
    }


def candidate_id(row: dict[str, str]) -> str:
    text_hash = stable_hash(normalized_text(row["text"]))[:20]
    return f"{row['dataset_version']}::{row['reading_task']}::{text_hash}"


def build_candidate_catalog(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[candidate_id(row)].append(row)
    catalog: list[dict[str, object]] = []
    for cid, members in sorted(grouped.items()):
        texts = {normalized_text(row["text"]) for row in members}
        datasets = {row["dataset_version"] for row in members}
        tasks = {row["reading_task"] for row in members}
        if len(texts) != 1 or len(datasets) != 1 or len(tasks) != 1:
            raise ValueError(f"candidate identity collision: {cid}")
        representative = min(members, key=lambda row: row["trial_id"])
        catalog.append(
            {
                "candidate_id": cid,
                "dataset_version": representative["dataset_version"],
                "reading_task": representative["reading_task"],
                "text": representative["text"],
                "normalized_text_sha256": stable_hash(normalized_text(representative["text"])),
                "length_words_whitespace_v1": as_int(
                    representative["length_words_whitespace_v1"], "length_words_whitespace_v1"
                ),
                "text_uids": "|".join(sorted({row["text_uid"] for row in members})),
                "trial_count": len(members),
            }
        )
    return catalog


def build_candidate_pools(
    targets: list[dict[str, str]], catalog: list[dict[str, object]], pool_size: int, seed: int
) -> list[dict[str, object]]:
    by_context: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_id = {str(row["candidate_id"]): row for row in catalog}
    for row in catalog:
        by_context[(str(row["dataset_version"]), str(row["reading_task"]))].append(row)

    output: list[dict[str, object]] = []
    for target in sorted(targets, key=lambda row: row["trial_id"]):
        positive_id = candidate_id(target)
        positive = by_id[positive_id]
        target_length = as_int(target["length_words_whitespace_v1"], "length_words_whitespace_v1")
        eligible = [
            row
            for row in by_context[(target["dataset_version"], target["reading_task"])]
            if row["candidate_id"] != positive_id
        ]
        eligible.sort(
            key=lambda row: (
                abs(int(row["length_words_whitespace_v1"]) - target_length),
                stable_hash(seed, target["trial_id"], row["candidate_id"], "pool-select"),
            )
        )
        if len(eligible) < pool_size - 1:
            raise ValueError(
                f"target {target['trial_id']} has only {len(eligible) + 1} unique matched "
                f"candidates; pool_size={pool_size}"
            )
        selected = [positive, *eligible[: pool_size - 1]]
        selected.sort(
            key=lambda row: stable_hash(seed, target["trial_id"], row["candidate_id"], "pool-order")
        )
        for rank, row in enumerate(selected):
            output.append(
                {
                    "target_trial_id": target["trial_id"],
                    "candidate_rank": rank,
                    "candidate_id": row["candidate_id"],
                    "is_positive": int(row["candidate_id"] == positive_id),
                    "dataset_version": row["dataset_version"],
                    "reading_task": row["reading_task"],
                    "target_length": target_length,
                    "candidate_length": row["length_words_whitespace_v1"],
                    "absolute_length_difference": abs(
                        int(row["length_words_whitespace_v1"]) - target_length
                    ),
                    "selection_rule": "same_dataset_task_then_length_then_seeded_hash",
                }
            )
    return output


def choose_donor(
    target: dict[str, str], candidates: Iterable[dict[str, str]], seed: int, rule: str,
    use_length: bool,
) -> dict[str, str] | None:
    different = [
        row
        for row in candidates
        if row["trial_id"] != target["trial_id"]
        and normalized_text(row["text"]) != normalized_text(target["text"])
    ]
    if not different:
        return None
    target_length = as_int(target["length_words_whitespace_v1"], "length_words_whitespace_v1")
    return min(
        different,
        key=lambda row: (
            abs(as_int(row["length_words_whitespace_v1"], "length_words_whitespace_v1") - target_length)
            if use_length
            else 0,
            stable_hash(seed, target["trial_id"], row["trial_id"], rule),
        ),
    )


def build_wrong_eeg_donors(
    targets: list[dict[str, str]], seed: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for target in sorted(targets, key=lambda row: row["trial_id"]):
        matched_pool = [
            row
            for row in targets
            if row["dataset_version"] == target["dataset_version"]
            and row["reading_task"] == target["reading_task"]
        ]
        subject_pool = [row for row in matched_pool if row["subject_id"] == target["subject_id"]]
        matched = choose_donor(target, matched_pool, seed, "task-dataset-wrong", True)
        subject = choose_donor(target, subject_pool, seed, "subject-length-wrong", True)
        random_valid = choose_donor(target, targets, seed, "random-valid-wrong", False)
        if matched is None or random_valid is None:
            raise ValueError(f"no valid wrong-EEG donor for {target['trial_id']}")
        output.append(
            {
                "target_trial_id": target["trial_id"],
                "target_dataset_version": target["dataset_version"],
                "target_reading_task": target["reading_task"],
                "target_subject_id": target["subject_id"],
                "random_valid_trial_id": random_valid["trial_id"],
                "task_dataset_wrong_trial_id": matched["trial_id"],
                "task_dataset_length_difference": abs(
                    as_int(matched["length_words_whitespace_v1"], "length_words_whitespace_v1")
                    - as_int(target["length_words_whitespace_v1"], "length_words_whitespace_v1")
                ),
                "subject_length_wrong_trial_id": subject["trial_id"] if subject else "",
                "subject_length_difference": (
                    abs(
                        as_int(subject["length_words_whitespace_v1"], "length_words_whitespace_v1")
                        - as_int(target["length_words_whitespace_v1"], "length_words_whitespace_v1")
                    )
                    if subject
                    else ""
                ),
                "subject_matched_available": int(subject is not None),
            }
        )
    return output


def build_feature_masks(targets: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in sorted(targets, key=lambda item: item["trial_id"]):
        masks = {field: as_int(row[field], field) for field in MASK_FIELDS}
        if any(value not in {0, 1} for value in masks.values()):
            raise ValueError(f"non-binary feature mask for {row['trial_id']}")
        if masks["mask_sr_sentiment_3"] and row["reading_task"] != "SR":
            raise ValueError("SR feature admitted outside SR")
        if masks["mask_nr_relation_content"] and row["reading_task"] != "NR":
            raise ValueError("NR feature admitted outside NR")
        if masks["mask_tsr_instruction_relation"] and row["reading_task"] != "TSR":
            raise ValueError("TSR feature admitted outside TSR")
        output.append(
            {
                "trial_id": row["trial_id"],
                "dataset_version": row["dataset_version"],
                "reading_task": row["reading_task"],
                "cohort": row["cohort"],
                "mask_length_words_whitespace_v1": int(
                    as_int(row["length_words_whitespace_v1"], "length_words_whitespace_v1") >= 0
                ),
                **masks,
            }
        )
    return output


def protocol_registry(seed: int, pool_size: int, phase: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": phase,
        "seed": seed,
        "candidate_pool": {
            "size": pool_size,
            "positive_count": 1,
            "matching": ["dataset_version", "reading_task"],
            "negative_selection": "closest whitespace-token length, seeded SHA256 tie-break",
            "ordering": "seeded SHA256",
            "model_scores_used": False,
        },
        "task_conditions": [
            "correct_task",
            "task_masked",
            "task_shuffled",
            "task_only",
            "metadata_only",
        ],
        "signal_conditions": [
            "correct_eeg",
            "zero_eeg",
            "gaussian_eeg",
            "random_valid_wrong_eeg",
            "task_dataset_matched_wrong_eeg",
            "subject_length_matched_wrong_eeg_when_available",
        ],
        "feature_policy": {
            "native_task_labels": "supervision_only_never_inference_input",
            "task_masks": "a factor may be active only where its provenance is valid",
            "admission": "incremental EEG factor retained only with EEG-specific gain and control survival",
        },
        "cohort_policy": {
            "primary": "primary_zuco2_nr_tsr",
            "auxiliary": ["auxiliary_sr", "zuco1_nr_tsr_noncausal"],
            "causal_cross_task_claims": False,
        },
        "held_out_test_accessed": False,
    }


def build(dataset_root: Path, output_root: Path, phase: str, pool_size: int, seed: int) -> dict[str, object]:
    if phase not in {"train", "val"}:
        raise ValueError("protocol freeze permits only train or val; held-out test is sealed")
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")
    all_rows, source_manifest, index_path = read_index(dataset_root)
    split_audit = audit_split_identity(all_rows)
    targets = [row for row in all_rows if row["split"] == phase]
    if not targets:
        raise ValueError(f"no rows for phase {phase!r}")
    if any(row["split"] == "test" for row in targets):
        raise AssertionError("held-out test row entered development manifest")

    catalog = build_candidate_catalog(targets)
    pools = build_candidate_pools(targets, catalog, pool_size, seed)
    donors = build_wrong_eeg_donors(targets, seed)
    feature_masks = build_feature_masks(targets)
    evaluation_rows = [
        {
            "trial_id": row["trial_id"],
            "split": row["split"],
            "cohort": row["cohort"],
            "dataset_version": row["dataset_version"],
            "reading_task": row["reading_task"],
            "subject_id": row["subject_id"],
            "text_uid": row["text_uid"],
            "candidate_id": candidate_id(row),
            "length_words_whitespace_v1": row["length_words_whitespace_v1"],
        }
        for row in sorted(targets, key=lambda item: item["trial_id"])
    ]

    pool_counts = Counter(str(row["target_trial_id"]) for row in pools)
    positive_counts = Counter(
        str(row["target_trial_id"]) for row in pools if int(row["is_positive"]) == 1
    )
    if set(pool_counts.values()) != {pool_size} or set(positive_counts.values()) != {1}:
        raise AssertionError("candidate-pool cardinality invariant failed")

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "evaluation_rows.csv",
        [
            "trial_id", "split", "cohort", "dataset_version", "reading_task", "subject_id",
            "text_uid", "candidate_id", "length_words_whitespace_v1",
        ],
        evaluation_rows,
    )
    write_csv(
        output_root / "candidate_catalog.csv",
        [
            "candidate_id", "dataset_version", "reading_task", "text", "normalized_text_sha256",
            "length_words_whitespace_v1", "text_uids", "trial_count",
        ],
        catalog,
    )
    write_csv(
        output_root / "candidate_pools.csv",
        [
            "target_trial_id", "candidate_rank", "candidate_id", "is_positive",
            "dataset_version", "reading_task", "target_length", "candidate_length",
            "absolute_length_difference", "selection_rule",
        ],
        pools,
    )
    write_csv(
        output_root / "wrong_eeg_donors.csv",
        [
            "target_trial_id", "target_dataset_version", "target_reading_task", "target_subject_id",
            "random_valid_trial_id", "task_dataset_wrong_trial_id",
            "task_dataset_length_difference", "subject_length_wrong_trial_id",
            "subject_length_difference", "subject_matched_available",
        ],
        donors,
    )
    write_csv(
        output_root / "feature_masks.csv",
        [
            "trial_id", "dataset_version", "reading_task", "cohort",
            "mask_length_words_whitespace_v1", *MASK_FIELDS,
        ],
        feature_masks,
    )
    registry = protocol_registry(seed, pool_size, phase)
    (output_root / "protocol_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    subject_available = sum(int(row["subject_matched_available"]) for row in donors)
    report = {
        "status": "pass",
        "schema_version": 1,
        "phase": phase,
        "seed": seed,
        "pool_size": pool_size,
        "source": {
            "schema_version": source_manifest["schema_version"],
            "index": str(index_path.relative_to(dataset_root)).replace("\\", "/"),
            "index_sha256": source_manifest["index_sha256"],
            "source_dataframe_sha256": source_manifest.get("source_dataframe_sha256", ""),
        },
        "counts": {
            "rows": len(targets),
            "candidate_identities": len(catalog),
            "candidate_pool_rows": len(pools),
            "wrong_eeg_rows": len(donors),
            "subject_matched_wrong_eeg_available": subject_available,
            "task_counts": dict(sorted(Counter(row["reading_task"] for row in targets).items())),
            "cohort_counts": dict(sorted(Counter(row["cohort"] for row in targets).items())),
        },
        "checks": {
            **split_audit,
            "held_out_test_accessed": False,
            "candidate_pool_size_exact": True,
            "one_positive_per_pool": True,
            "candidate_pool_dataset_task_matched": True,
            "candidate_selection_uses_model_scores": False,
            "wrong_eeg_never_same_text": True,
            "task_dataset_wrong_eeg_matched": True,
            "task_specific_feature_masks_valid": True,
        },
        "limitations": {
            "session_block": "unavailable_in_frozen_metadata",
            "subject_matched_wrong_eeg_not_universal": subject_available != len(donors),
        },
        "artifact_sha256": {
            name: sha256(output_root / name) for name in ARTIFACTS
        },
    }
    report_path = output_root / "evaluation_contract_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "val"), default="val")
    parser.add_argument("--pool-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    report = build(args.dataset_root, args.output_root, args.phase, args.pool_size, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
