"""Regression test for deterministic protocol-manifest construction."""

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

from evaluation.build_protocol_manifests import ARTIFACTS, build


FIELDS = [
    "dataset_version", "reading_task", "subject_id", "text_uid", "trial_id", "split",
    "cohort", "text", "length_words_whitespace_v1", "mask_sr_sentiment_3",
    "mask_nr_relation_content", "mask_tsr_instruction_relation", "mask_gpt2_mean_nll_v1",
    "mask_semkey_sentiment_2", "mask_topic_label",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(dataset: str, task: str, subject: str, number: int, split: str = "val") -> dict[str, object]:
    return {
        "dataset_version": dataset,
        "reading_task": task,
        "subject_id": subject,
        "text_uid": f"{dataset}-{task}-{split}-{number}",
        "trial_id": f"{dataset}::{task}::{subject}::{split}{number:02d}",
        "split": split,
        "cohort": "auxiliary_sr" if task == "SR" else "primary_zuco2_nr_tsr",
        "text": f"Unique {split} sentence number {number} for {dataset} {task}",
        "length_words_whitespace_v1": 7 + number % 3,
        "mask_sr_sentiment_3": int(task == "SR"),
        "mask_nr_relation_content": 0,
        "mask_tsr_instruction_relation": int(task == "TSR"),
        "mask_gpt2_mean_nll_v1": 0,
        "mask_semkey_sentiment_2": 0,
        "mask_topic_label": 0,
    }


def write_fixture(root: Path) -> None:
    shard_dir = root / "shards"
    metadata_dir = root / "metadata"
    shard_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    rows = [
        row("ZuCo1", "SR", "S1" if number < 2 else "S2", number)
        for number in range(4)
    ] + [
        row("ZuCo2", "NR", "S1" if number < 2 else "S2", number + 10)
        for number in range(4)
    ] + [row("ZuCo2", "NR", "S3", 99, split="train")]
    index_path = shard_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 2,
        "row_count": len(rows),
        "index": "shards/index.csv",
        "index_sha256": file_sha256(index_path),
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
        report = build(dataset, first, phase="val", pool_size=3, seed=17)
        second_report = build(dataset, second, phase="val", pool_size=3, seed=17)

        assert report == second_report
        assert report["status"] == "pass"
        assert report["counts"]["rows"] == 8
        assert report["counts"]["candidate_pool_rows"] == 24
        assert report["checks"]["held_out_test_accessed"] is False
        for name in (*ARTIFACTS, "evaluation_contract_report.json"):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

        with (first / "candidate_pools.csv").open(encoding="utf-8", newline="") as handle:
            pools = list(csv.DictReader(handle))
        grouped: dict[str, list[dict[str, str]]] = {}
        for candidate in pools:
            grouped.setdefault(candidate["target_trial_id"], []).append(candidate)
        assert all(len(items) == 3 for items in grouped.values())
        assert all(sum(int(item["is_positive"]) for item in items) == 1 for items in grouped.values())

        with (first / "wrong_eeg_donors.csv").open(encoding="utf-8", newline="") as handle:
            donors = list(csv.DictReader(handle))
        assert all(item["target_trial_id"] != item["task_dataset_wrong_trial_id"] for item in donors)
        assert all(item["subject_matched_available"] == "1" for item in donors)
        print("PROTOCOL MANIFEST REGRESSION: PASS")


if __name__ == "__main__":
    main()
