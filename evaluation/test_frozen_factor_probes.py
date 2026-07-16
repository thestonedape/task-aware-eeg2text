"""Tiny deterministic regression for the frozen factor-probe runner."""

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
from evaluation.build_recoverability_protocol import build
from evaluation.run_frozen_factor_probes import EXPECTED_CHECKS, run
from evaluation.test_recoverability_protocol import write_fixture


VECTOR_FIELDS = [
    "condition", "phase", "target_trial_id", "signal_trial_id",
    "target_source_dataframe_row_index", "signal_source_dataframe_row_index",
    "dataset_version", "reading_task", "subject_id", "text_uid", "vector_file",
    "vector_offset", "vector_dim", "prompt_mode", "checkpoint_sha256", "source_index_sha256",
]


def write_vector_fixture(root: Path, protocol_root: Path) -> None:
    factor_rows = list(csv.DictReader((protocol_root / "recoverability_rows.csv").open(encoding="utf-8")))
    registry = json.loads((protocol_root / "recoverability_registry.json").read_text(encoding="utf-8"))
    trial_rows: dict[str, dict[str, str]] = {}
    lengths: dict[str, float] = {}
    for row in factor_rows:
        trial_rows[row["trial_id"]] = row
        if row["factor_id"] == "length_words_whitespace_v1":
            lengths[row["trial_id"]] = float(row["target_value"])
    train = sorted((row for row in trial_rows.values() if row["split"] == "train"), key=lambda row: row["trial_id"])
    validation = sorted((row for row in trial_rows.values() if row["split"] == "val"), key=lambda row: row["trial_id"])
    root.mkdir(parents=True)
    vector_dir = root / "vectors"
    vector_dir.mkdir()
    index_rows = []
    chunks = []
    task_value = {"SR": 1.0, "NR": 2.0, "TSR": 3.0}

    def correct(row: dict[str, str]) -> np.ndarray:
        return np.asarray([
            lengths[row["trial_id"]], task_value[row["reading_task"]],
            1.0 if row["subject_id"] == "S1" else 2.0,
            float(int(row["trial_id"][-2:])),
        ], dtype=np.float32)

    conditions = {
        "correct_train": (train, [correct(row) for row in train]),
        "correct_val": (validation, [correct(row) for row in validation]),
        "matched_wrong_val": (validation, [correct(row) for row in reversed(validation)]),
        "zero_val": (validation, [np.zeros(4, dtype=np.float32) for _ in validation]),
        "gaussian_val": (
            validation,
            [np.random.default_rng(100 + index).normal(size=4).astype(np.float32) for index in range(len(validation))],
        ),
    }
    for condition, (rows, values) in conditions.items():
        relative = f"vectors/{condition}_00000.npz"
        path = root / relative
        vectors = np.stack(values)
        np.savez(path, vectors=vectors)
        chunks.append({
            # Match the frozen schema-v1 extractor: the concrete vector path is
            # carried by vector_index.csv, not repeated in each chunk entry.
            "condition": condition, "chunk_number": 0,
            "vector_npz_sha256": sha256(path),
        })
        for offset, row in enumerate(rows):
            index_rows.append({
                "condition": condition, "phase": row["split"], "target_trial_id": row["trial_id"],
                "signal_trial_id": row["trial_id"] if condition.startswith("correct") else f"{condition}::{offset}",
                "target_source_dataframe_row_index": offset, "signal_source_dataframe_row_index": offset,
                "dataset_version": row["dataset_version"], "reading_task": row["reading_task"],
                "subject_id": row["subject_id"], "text_uid": row["text_uid"], "vector_file": relative,
                "vector_offset": offset, "vector_dim": 4, "prompt_mode": "canonical",
                "checkpoint_sha256": "fixture", "source_index_sha256": registry["source_index_sha256"],
            })
    index_path = root / "vector_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VECTOR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(index_rows)
    manifest = {
        "status": "pass", "run_mode": "full_development", "source_index_sha256": registry["source_index_sha256"],
        "vector_dim": 4, "vector_index_sha256": sha256(index_path), "chunks": chunks,
        "condition_counts": {condition: len(rows) for condition, (rows, _) in conditions.items()},
        "checks": EXPECTED_CHECKS,
    }
    (root / "vector_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        protocol = root / "protocol"
        vectors = root / "vectors"
        first = root / "first"
        second = root / "second"
        write_fixture(dataset)
        build(dataset, protocol)
        registry_path = protocol / "recoverability_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["seeds"] = [7]
        registry["uncertainty"]["bootstrap_replicates"] = 25
        for spec in registry["probe_models"].values():
            if "grid_C" in spec:
                spec["grid_C"] = [0.1]
            if "grid_alpha" in spec:
                spec["grid_alpha"] = [1.0]
        registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_vector_fixture(vectors, protocol)

        first_report = run(vectors, protocol, first)
        second_report = run(vectors, protocol, second)
        assert first_report["status"] == second_report["status"] == "pass"
        assert tree_hashes(first) == tree_hashes(second)
        manifest = json.loads((first / "probe_manifest.json").read_text(encoding="utf-8"))
        assert manifest["checks"]["held_out_test_accessed"] is False
        assert manifest["checks"]["target_label_as_probe_input_permitted"] is False
        assert set(manifest["factors"]) == {
            "sr_sentiment_3", "nr_relation_content", "tsr_instruction_relation",
            "length_words_whitespace_v1",
        }
        admissions = list(csv.DictReader((first / "factor_admission.csv").open(encoding="utf-8")))
        assert len(admissions) == 4
        assert all(row["decision"] in {"admit", "reject_retain_null"} for row in admissions)
        print("FROZEN FACTOR PROBE REGRESSION: PASS")


if __name__ == "__main__":
    main()
