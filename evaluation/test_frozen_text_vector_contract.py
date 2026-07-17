"""Regression for frozen GLIM text-target identities, sealing, and resume."""

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

from evaluation.extract_frozen_glim_text_vectors import run_extraction, sha256


FIELDS = [
    "trial_id", "split", "cohort", "dataset_version", "reading_task", "subject_id",
    "source_dataframe_row_index", "text",
]


def write_fixture(root: Path) -> str:
    rows = [
        {"trial_id": "train-1", "split": "train", "cohort": "primary_zuco2_nr_tsr", "dataset_version": "ZuCo2", "reading_task": "NR", "subject_id": "S1", "source_dataframe_row_index": 1, "text": "Repeated text"},
        {"trial_id": "train-2", "split": "train", "cohort": "primary_zuco2_nr_tsr", "dataset_version": "ZuCo2", "reading_task": "NR", "subject_id": "S2", "source_dataframe_row_index": 2, "text": " repeated   TEXT "},
        {"trial_id": "val-1", "split": "val", "cohort": "auxiliary_sr", "dataset_version": "ZuCo1", "reading_task": "SR", "subject_id": "S1", "source_dataframe_row_index": 3, "text": "Validation text"},
        {"trial_id": "test-1", "split": "test", "cohort": "primary_zuco2_nr_tsr", "dataset_version": "ZuCo2", "reading_task": "TSR", "subject_id": "S3", "source_dataframe_row_index": 4, "text": "Sealed text"},
    ]
    (root / "metadata").mkdir(parents=True)
    (root / "shards").mkdir()
    index = root / "shards" / "index.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    digest = sha256(index)
    (root / "metadata" / "shard_manifest.json").write_text(
        json.dumps({"schema_version": 2, "index": "shards/index.csv", "index_sha256": digest}),
        encoding="utf-8",
    )
    return digest


class FakeTextEmbedder:
    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[len(text), sum(map(ord, text)) % 17, text.count(" "), 1.0] for text in texts],
            dtype=np.float32,
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        output = root / "output"
        digest = write_fixture(dataset)
        kwargs = dict(
            dataset_root=dataset,
            output_root=output,
            embed_texts=FakeTextEmbedder(),
            vector_dim=4,
            checkpoint_sha256="checkpoint",
            glim_commit="glim",
            text_model_id="fake-t5",
            text_dtype="float32",
            expected_index_sha256=digest,
            batch_size=1,
            chunk_size=1,
        )
        first = run_extraction(**kwargs)
        second = run_extraction(**kwargs)
        assert first["unique_text_identities"] == 2
        assert first["mapped_trials"] == 3
        assert second["chunks_reused_this_invocation"] == second["chunks"] == 2
        manifest = json.loads((output / "text_vector_manifest.json").read_text(encoding="utf-8"))
        assert manifest["split_counts"] == {"train": 2, "val": 1}
        assert manifest["checks"]["held_out_test_accessed"] is False
        with (output / "trial_text_targets.csv").open(encoding="utf-8", newline="") as handle:
            mappings = list(csv.DictReader(handle))
        assert all(row["split"] != "test" for row in mappings)
        assert mappings[0]["text_target_id"] == mappings[1]["text_target_id"]
        print("FROZEN TEXT VECTOR CONTRACT REGRESSION: PASS")


if __name__ == "__main__":
    main()
