"""Deterministic regression for the recoverability protocol supplement."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.build_recoverability_protocol import ARTIFACTS, build


FIELDS = [
    "dataset_version", "reading_task", "subject_id", "text_uid", "trial_id", "split",
    "cohort", "text", "length_words_whitespace_v1", "mask_sr_sentiment_3",
    "sr_sentiment_3", "mask_nr_relation_content", "nr_relation_content",
    "mask_tsr_instruction_relation", "tsr_instruction_relation", "mask_gpt2_mean_nll_v1",
    "gpt2_mean_nll_v1", "mask_semkey_sentiment_2", "semkey_sentiment_2",
    "mask_topic_label", "topic_label",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_row(
    split: str, task: str, subject: str, number: int, label: str, text: str | None = None
) -> dict[str, object]:
    dataset = "ZuCo1" if task in {"SR", "NR"} else "ZuCo2"
    return {
        "dataset_version": dataset,
        "reading_task": task,
        "subject_id": subject,
        "text_uid": f"{split}-{task}-{number}",
        "trial_id": f"{dataset}::{task}::{subject}::{split}{number:02d}",
        "split": split,
        "cohort": "auxiliary_sr" if task == "SR" else (
            "zuco1_nr_tsr_noncausal" if task == "NR" else "primary_zuco2_nr_tsr"
        ),
        "text": text or f"Unique {split} {task} sentence {number}",
        "length_words_whitespace_v1": 5 + number % 4,
        "mask_sr_sentiment_3": int(task == "SR"),
        "sr_sentiment_3": label if task == "SR" else "",
        "mask_nr_relation_content": int(task == "NR"),
        "nr_relation_content": label if task == "NR" else "",
        "mask_tsr_instruction_relation": int(task == "TSR"),
        "tsr_instruction_relation": label if task == "TSR" else "",
        "mask_gpt2_mean_nll_v1": 0,
        "gpt2_mean_nll_v1": "",
        "mask_semkey_sentiment_2": 0,
        "semkey_sentiment_2": "",
        "mask_topic_label": 0,
        "topic_label": "",
    }


def write_fixture(root: Path) -> None:
    train: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    labels = {
        "SR": ("negative", "positive"),
        "NR": ("NO-RELATION", "EDUCATION;JOB_TITLE"),
        "TSR": ("AWARD", "CONTROL"),
    }
    for task, pair in labels.items():
        for number in range(4):
            text = "Shared train relation sentence" if task in {"NR", "TSR"} and number == 0 else None
            train.append(make_row("train", task, "S1" if number < 2 else "S2", number, pair[number % 2], text))
        for number in range(2):
            validation.append(make_row("val", task, "S1" if number == 0 else "S2", number + 10, pair[number]))
    test_row = make_row("test", "SR", "S3", 99, "neutral")
    rows = [*train, *validation, test_row]

    shard_dir = root / "shards"
    metadata_dir = root / "metadata"
    shard_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    index_path = shard_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 2,
        "row_count": len(rows),
        "index": "shards/index.csv",
        "index_sha256": digest(index_path),
        "source_dataframe_sha256": "fixture",
    }
    (metadata_dir / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        first = root / "first"
        second = root / "second"
        write_fixture(dataset)
        report = build(dataset, first)
        second_report = build(dataset, second)
        assert report == second_report
        assert report["status"] == "pass"
        assert report["counts"]["train_rows"] == 12
        assert report["counts"]["validation_rows"] == 6
        assert report["counts"]["subject_folds"] == 2
        assert report["counts"]["cross_task_duplicate_groups"] == 1
        assert report["counts"]["validation_duplicate_groups"] == 0
        assert report["duplicate_consistency_status"] == "unavailable_under_frozen_validation_split"
        assert report["checks"]["held_out_test_accessed"] is False
        for name in (*ARTIFACTS, "recoverability_contract_report.json"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

        with (first / "subject_folds.csv").open(encoding="utf-8", newline="") as handle:
            folds = list(csv.DictReader(handle))
        assert all(int(row["training_rows_excluded"]) > 0 for row in folds)
        assert all(int(row["evaluation_rows_eligible"]) == 3 for row in folds)

        registry = json.loads((first / "recoverability_registry.json").read_text(encoding="utf-8"))
        assert registry["held_out_test_accessed"] is False
        assert registry["unavailable_factors"]["gpt2_mean_nll_v1"] == "not_fabricated"
        assert registry["conditions"]["frozen_glim_correct"]["warning"].startswith("contains_released")
        print("RECOVERABILITY PROTOCOL REGRESSION: PASS")


if __name__ == "__main__":
    main()
