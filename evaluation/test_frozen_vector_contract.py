"""Regression for resumable frozen-vector identities and controls."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.build_protocol_manifests import sha256
from evaluation.extract_frozen_glim_vectors import run_extraction


FIELDS = [
    "dataset_version", "reading_task", "subject_id", "text_uid", "trial_id", "sample_id",
    "split", "phase", "cohort", "text", "length_words_whitespace_v1",
    "mask_sr_sentiment_3", "mask_nr_relation_content", "mask_tsr_instruction_relation",
    "mask_gpt2_mean_nll_v1", "mask_semkey_sentiment_2", "mask_topic_label",
    "source_dataframe_row_index", "shard", "offset",
]


def fixture_row(split: str, subject: str, number: int) -> dict[str, object]:
    trial_id = f"ZuCo1::task1::{subject}::row{number:06d}"
    return {
        "dataset_version": "ZuCo1",
        "reading_task": "SR",
        "subject_id": subject,
        "text_uid": f"{split}-{number}",
        "trial_id": trial_id,
        "sample_id": trial_id,
        "split": split,
        "phase": split,
        "cohort": "auxiliary_sr",
        "text": f"Unique {split} sentence {number}",
        "length_words_whitespace_v1": 4,
        "mask_sr_sentiment_3": 1,
        "mask_nr_relation_content": 0,
        "mask_tsr_instruction_relation": 0,
        "mask_gpt2_mean_nll_v1": 0,
        "mask_semkey_sentiment_2": 0,
        "mask_topic_label": 0,
        "source_dataframe_row_index": number,
        "shard": "shard_00000.npz",
        "offset": number,
    }


def write_fixture(root: Path) -> None:
    rows = [
        *(fixture_row("train", "S1" if i < 2 else "S2", i) for i in range(4)),
        *(fixture_row("val", "S1" if i < 6 else "S2", i) for i in range(4, 8)),
        fixture_row("test", "S3", 8),
    ]
    shard_dir = root / "shards"
    metadata_dir = root / "metadata"
    shard_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    eeg = np.stack([
        np.asarray(
            [[i + 1, i + 2, i + 3], [i + 2, i + 3, i + 4],
             [i + 3, i + 4, i + 5], [0, 0, 0]],
            dtype=np.float32,
        )
        for i in range(len(rows))
    ])
    mask = np.tile(np.asarray([1, 1, 1, 0], dtype=np.int8), (len(rows), 1))
    np.savez_compressed(shard_dir / "shard_00000.npz", eeg=eeg, mask=mask)
    index_path = shard_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 2,
        "row_count": len(rows),
        "index": "shards/index.csv",
        "index_sha256": sha256(index_path),
        "source_dataframe_sha256": "fixture",
    }
    (metadata_dir / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class FakeEmbedder:
    vector_dim = 4

    def __call__(self, eeg, mask, prompts, sample_ids, source_rows):
        valid = mask.astype(bool)
        rows = []
        for index in range(len(eeg)):
            values = eeg[index][valid[index]]
            rows.append([
                float(values.mean()) if values.size else 0.0,
                float(values.std()) if values.size else 0.0,
                float(values.sum()),
                float({"<SR>": 1, "<NR>": 2, "<TSR>": 3}[prompts[0][index]]),
            ])
        return np.asarray(rows, dtype=np.float32)


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        output = root / "output"
        smoke = root / "smoke"
        write_fixture(dataset)
        index_sha = json.loads(
            (dataset / "metadata" / "shard_manifest.json").read_text(encoding="utf-8")
        )["index_sha256"]
        common = dict(
            dataset_root=dataset,
            embed_batch=FakeEmbedder(),
            vector_dim=4,
            checkpoint_sha256="fake-checkpoint",
            glim_commit="fake-glim",
            expected_index_sha256=index_sha,
            batch_size=2,
            chunk_size=2,
            expected_signal_shape=None,
        )
        first = run_extraction(output_root=output, **common)
        before = tree_hashes(output)
        second = run_extraction(output_root=output, **common)
        after = tree_hashes(output)
        assert before == after
        assert second["chunks_reused_this_invocation"] == second["chunks"]
        assert first["condition_counts"] == {
            "correct_train": 4,
            "correct_val": 4,
            "gaussian_val": 4,
            "matched_wrong_val": 4,
            "zero_val": 4,
        }
        with (output / "vector_index.csv").open(encoding="utf-8", newline="") as handle:
            index = list(csv.DictReader(handle))
        assert len(index) == 20
        assert all(row["phase"] != "test" for row in index)
        wrong = [row for row in index if row["condition"] == "matched_wrong_val"]
        assert all(row["target_trial_id"] != row["signal_trial_id"] for row in wrong)
        manifest = json.loads((output / "vector_manifest.json").read_text(encoding="utf-8"))
        assert manifest["checks"]["held_out_test_accessed"] is False
        assert manifest["gaussian_stats"]["training_rows"] == 4

        smoke_report = run_extraction(output_root=smoke, smoke_limit=1, **common)
        assert set(smoke_report["condition_counts"].values()) == {1}
        assert smoke_report["run_mode"] == "smoke"
        print("FROZEN VECTOR CONTRACT REGRESSION: PASS")


if __name__ == "__main__":
    main()
