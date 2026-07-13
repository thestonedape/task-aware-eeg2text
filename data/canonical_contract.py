"""Canonical task/label contract for the task-aware ZuCo shard dataset."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping


TASK_TO_READING = {"task1": "SR", "task2": "NR", "task3": "TSR"}
SR_SENTIMENTS = {"negative", "neutral", "positive"}
RELATIONS = {
    "AWARD", "EDUCATION", "EMPLOYER", "FOUNDER", "JOB_TITLE",
    "NATIONALITY", "POLITICAL_AFFILIATION", "VISITED", "WIFE",
}
TSR_RELATIONS = RELATIONS | {"CONTROL"}
RELATION_NORMALIZATION = {
    "awarding": "AWARD", "award": "AWARD", "education": "EDUCATION",
    "employment": "EMPLOYER", "employer": "EMPLOYER", "founder": "FOUNDER",
    "job title": "JOB_TITLE", "job_title": "JOB_TITLE", "nationality": "NATIONALITY",
    "political affiliation": "POLITICAL_AFFILIATION",
    "political_affiliation": "POLITICAL_AFFILIATION", "visit": "VISITED",
    "visited": "VISITED", "marriage": "WIFE", "wife": "WIFE",
}
RELEASED_TYPOBOOK = {
    "emp11111ty": "empty", "film.1": "film.", "–": "-", "’s": "'s", "�s": "'s",
    "`s": "'s", "Maria": "Marić", "1Universidad": "Universidad", "1902—19": "1902 - 19",
    "Wuerttemberg": "Württemberg", "long -time": "long-time", "Jose": "José", "Bucher": "Bôcher",
    "1839 ? May": "1839 - May", "G�n�ration": "Generation", "Bragança": "Bragana",
    "1837?October": "1837 - October", "nVera-Ellen": "Vera-Ellen", "write Ethics": "wrote Ethics",
    "Adams-Onis": "Adams-Onís", "(40 km?)": "(40 km²)", "(40 km˝)": "(40 km²)",
    " (IPA: /?g?nz?b?g/) ": " ", '""Canes""': '"Canes"',
}

CANONICAL_FIELDS = [
    "dataset_version", "reading_task", "raw_task", "subject_id", "text_uid", "trial_id",
    "split", "session_block", "session_block_status", "cohort", "text",
    "source_dataframe_row_index", "source_dataframe_sha256", "eeg_locator", "shard_locator",
    "sr_sentiment_3", "nr_relation_content", "tsr_instruction_relation",
    "mask_sr_sentiment_3", "mask_nr_relation_content", "mask_tsr_instruction_relation",
    "length_words_whitespace_v1", "gpt2_mean_nll_v1", "mask_gpt2_mean_nll_v1",
    "semkey_sentiment_2", "mask_semkey_sentiment_2", "topic_label", "mask_topic_label",
    "length", "surprisal", "oracle_policy",
]
CANONICAL_MANIFEST_FIELDS = [
    field for field in CANONICAL_FIELDS
    if field not in {
        "semkey_sentiment_2", "mask_semkey_sentiment_2", "topic_label", "mask_topic_label",
        "length", "surprisal",
    }
]

VALIDATION_COMPARISON_FIELDS = [
    "dataset_version", "reading_task", "raw_task", "subject_id", "text_uid", "trial_id", "split",
    "cohort", "text", "source_dataframe_row_index", "source_dataframe_sha256", "eeg_locator",
    "sr_sentiment_3", "nr_relation_content", "tsr_instruction_relation",
    "mask_sr_sentiment_3", "mask_nr_relation_content", "mask_tsr_instruction_relation",
    "oracle_policy",
]


def clean(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na"} else text


def normalize_text(text: object) -> str:
    value = clean(text).replace("\ufeff", "")
    for source, target in RELEASED_TYPOBOOK.items():
        value = value.replace(source, target)
    return " ".join(value.split())


def canonical_tsr(value: object) -> str:
    label = clean(value)
    if not label:
        return ""
    upper = label.upper().replace("-", "_")
    if upper in TSR_RELATIONS:
        return upper
    return RELATION_NORMALIZATION.get(label.lower(), "")


def _read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_nr_relation_map(label_dir: str | Path) -> dict[str, str]:
    path = Path(label_dir) / "relations_labels_task2.csv"
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in _read_catalog(path):
        text = normalize_text(row.get("sentence", ""))
        label = clean(row.get("relation_types", ""))
        if text and label:
            grouped[text].add(label)
    conflicts = {text: labels for text, labels in grouped.items() if len(labels) > 1}
    if conflicts:
        raise ValueError(f"conflicting NR catalog labels for {len(conflicts)} normalized texts")
    return {text: next(iter(labels)) for text, labels in grouped.items()}


def canonical_record(
    row: Mapping[str, object],
    row_index: int,
    source_sha256: str,
    shard_name: str,
    offset: int,
    nr_relation_map: Mapping[str, str],
) -> dict[str, object]:
    dataset = clean(row.get("dataset"))
    task = clean(row.get("task"))
    subject = clean(row.get("subject"))
    phase = clean(row.get("phase"))
    if dataset not in {"ZuCo1", "ZuCo2"}:
        raise ValueError(f"row {row_index}: invalid dataset {dataset!r}")
    if task not in TASK_TO_READING:
        raise ValueError(f"row {row_index}: invalid task {task!r}")
    if phase not in {"train", "val", "test"}:
        raise ValueError(f"row {row_index}: invalid phase {phase!r}")
    if not subject:
        raise ValueError(f"row {row_index}: missing subject")

    raw_text = clean(row.get("input text"))
    text = normalize_text(raw_text)
    if not text:
        raise ValueError(f"row {row_index}: empty input text")
    text_uid = clean(row.get("text uid"))
    if not text_uid:
        raise ValueError(f"row {row_index}: missing text uid")
    trial_id = f"{dataset}::{task}::{subject}::row{row_index:06d}"
    cohort = (
        "primary_zuco2_nr_tsr" if dataset == "ZuCo2" and task in {"task2", "task3"}
        else "auxiliary_sr" if task == "task1"
        else "zuco1_nr_tsr_noncausal"
    )

    sentiment = clean(row.get("sentiment label")) if task == "task1" else ""
    if task == "task1" and sentiment not in SR_SENTIMENTS:
        raise ValueError(f"row {row_index}: invalid/missing native SR sentiment {sentiment!r}")
    nr_relation = nr_relation_map.get(text, "") if dataset == "ZuCo1" and task == "task2" else ""
    tsr_relation = canonical_tsr(row.get("relation label")) if task == "task3" else ""
    if task == "task3" and tsr_relation not in TSR_RELATIONS:
        raise ValueError(f"row {row_index}: invalid/missing TSR relation {row.get('relation label')!r}")

    semkey_sentiment = clean(row.get("semkey_sentiment_2"))
    topic = clean(row.get("topic_label"))
    surprisal = clean(row.get("surprisal"))
    length = len(raw_text.split())
    return {
        "dataset_version": dataset,
        "reading_task": TASK_TO_READING[task],
        "raw_task": task,
        "subject_id": subject,
        "text_uid": text_uid,
        "trial_id": trial_id,
        "split": phase,
        "session_block": "",
        "session_block_status": "unavailable_in_frozen_metadata",
        "cohort": cohort,
        "text": text,
        "source_dataframe_row_index": row_index,
        "source_dataframe_sha256": source_sha256,
        "eeg_locator": f"GLIM/data/tmp/zuco_eeg_label_8variants.df#row={row_index}",
        "shard_locator": f"shards/{shard_name}#offset={offset}",
        "sr_sentiment_3": sentiment,
        "nr_relation_content": nr_relation,
        "tsr_instruction_relation": tsr_relation,
        "mask_sr_sentiment_3": int(bool(sentiment)),
        "mask_nr_relation_content": int(bool(nr_relation)),
        "mask_tsr_instruction_relation": int(bool(tsr_relation)),
        "length_words_whitespace_v1": length,
        "gpt2_mean_nll_v1": surprisal,
        "mask_gpt2_mean_nll_v1": int(bool(surprisal)),
        "semkey_sentiment_2": semkey_sentiment,
        "mask_semkey_sentiment_2": int(bool(semkey_sentiment)),
        "topic_label": topic,
        "mask_topic_label": int(bool(topic)),
        "length": length,
        "surprisal": surprisal,
        "oracle_policy": "labels_for_supervision_only",
    }


def audit_records(records: list[dict[str, object]]) -> dict[str, object]:
    phase_counts = Counter(str(row["split"]) for row in records)
    task_counts = Counter(str(row["reading_task"]) for row in records)
    cohort_counts = Counter(str(row["cohort"]) for row in records)
    text_uid_splits: dict[str, set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    trial_ids: set[str] = set()
    for row in records:
        trial_id = str(row["trial_id"])
        if trial_id in trial_ids:
            raise ValueError(f"duplicate trial id: {trial_id}")
        trial_ids.add(trial_id)
        text_uid_splits[str(row["text_uid"])].add(str(row["split"]))
        text_splits[str(row["text"]).lower()].add(str(row["split"]))
    uid_overlap = sum(len(parts) > 1 for parts in text_uid_splits.values())
    text_overlap = sum(len(parts) > 1 for parts in text_splits.values())
    if uid_overlap or text_overlap:
        raise ValueError(
            f"split leakage: {uid_overlap} text UIDs and {text_overlap} normalized texts cross phases"
        )
    return {
        "row_count": len(records),
        "phase_counts": dict(sorted(phase_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "cross_split_text_uid_count": uid_overlap,
        "cross_split_normalized_text_count": text_overlap,
        "unique_trial_id_count": len(trial_ids),
    }


def compare_frozen_validation(
    records: list[dict[str, object]], expected_path: str | Path
) -> dict[str, object]:
    actual = {str(row["trial_id"]): row for row in records if row["split"] == "val"}
    with Path(expected_path).open(encoding="utf-8-sig", newline="") as handle:
        expected = {row["trial_id"]: row for row in csv.DictReader(handle)}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise ValueError(f"frozen validation identity mismatch; missing={missing}, extra={extra}")
    mismatches = []
    for trial_id, expected_row in expected.items():
        actual_row = actual[trial_id]
        for field in VALIDATION_COMPARISON_FIELDS:
            if clean(actual_row.get(field)) != clean(expected_row.get(field)):
                mismatches.append((trial_id, field, clean(expected_row.get(field)), clean(actual_row.get(field))))
                if len(mismatches) == 5:
                    break
        if len(mismatches) == 5:
            break
    if mismatches:
        raise ValueError(f"frozen validation content mismatch: {mismatches}")
    return {"status": "pass", "row_count": len(expected), "compared_fields": VALIDATION_COMPARISON_FIELDS}
