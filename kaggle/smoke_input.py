"""Discover and validate a canonical Kaggle input; optionally load one real shard batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_INDEX_FIELDS = {
    "source_dataframe_row_index", "sample_id", "shard", "offset", "dataset", "task",
    "subject", "phase", "text_uid", "input text", "raw text", "raw label", "control",
    "label id", "sentiment label", "relation label",
    "dataset_version", "reading_task", "raw_task", "subject_id", "trial_id", "split",
    "cohort", "text", "source_dataframe_sha256", "eeg_locator", "shard_locator",
    "sr_sentiment_3", "nr_relation_content", "tsr_instruction_relation",
    "mask_sr_sentiment_3", "mask_nr_relation_content", "mask_tsr_instruction_relation",
    "length_words_whitespace_v1", "oracle_policy",
    "lexical simplification (v0)", "lexical simplification (v1)",
    "semantic clarity (v0)", "semantic clarity (v1)", "syntax simplification (v0)",
    "syntax simplification (v1)", "naive rewritten", "naive simplified",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root: Path, relative: str) -> Path:
    matches = sorted(path for path in root.glob(f"**/{relative}") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one recursive {relative} below {root}, found {matches}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    dataset_root = args.dataset_root
    manifest_path = dataset_root / "metadata/shard_manifest.json" if dataset_root else discover(
        args.input_root, "metadata/shard_manifest.json"
    )
    dataset_root = manifest_path.parents[1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("canonical shard schema_version 2 is required")
    index_path = dataset_root / manifest["index"]
    if sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("shard index hash mismatch")
    with index_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_INDEX_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"shard index missing trainable fields: {sorted(missing)}")
        index_rows = list(reader)
    if len(index_rows) != manifest["row_count"]:
        raise ValueError("index row count mismatch")
    canonical_path = dataset_root / manifest["canonical_manifest"]
    if sha256(canonical_path) != manifest["canonical_manifest_sha256"]:
        raise ValueError("canonical full manifest hash mismatch")
    contract_path = dataset_root / manifest["canonical_contract_report"]
    if sha256(contract_path) != manifest["canonical_contract_report_sha256"]:
        raise ValueError("canonical contract report hash mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "pass" or int(contract.get("row_count", -1)) != len(index_rows):
        raise ValueError("canonical contract report did not pass")
    sample_ids = [row["sample_id"] for row in index_rows]
    if any(not sample_id for sample_id in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be non-empty and unique")
    shard_rows = {item["name"]: int(item["rows"]) for item in manifest["shards"]}
    for row in index_rows:
        if row["shard"] not in shard_rows or not 0 <= int(row["offset"]) < shard_rows[row["shard"]]:
            raise ValueError(f"invalid shard/offset locator for {row['sample_id']}")
    report = {
        "status": "metadata_pass", "dataset_root": str(dataset_root),
        "rows": len(index_rows), "shards": len(manifest["shards"]),
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "phase_counts": contract["phase_counts"],
        "task_counts": contract["task_counts"],
        "semkey_generated_labels": contract["semkey_generated_labels"]["status"],
    }
    if not args.metadata_only:
        if args.batch_size != 1:
            raise ValueError("the gate requires batch-size 1")
        first = index_rows[0]
        shard_path = dataset_root / "shards" / first["shard"]
        expected = next(item["sha256"] for item in manifest["shards"] if item["name"] == first["shard"])
        if sha256(shard_path) != expected:
            raise ValueError("shard hash mismatch")
        archive = np.load(shard_path, allow_pickle=False)
        offset = int(first["offset"])
        eeg = archive["eeg"][offset : offset + 1]
        mask = archive["mask"][offset : offset + 1]
        source_row = int(archive["source_dataframe_row_index"][offset])
        if eeg.shape[0] != 1 or mask.shape[0] != 1 or not np.isfinite(eeg).all():
            raise ValueError("invalid real batch-1 shard payload")
        if source_row != int(first["source_dataframe_row_index"]):
            raise ValueError("source dataframe row mismatch between index and shard")
        report.update({
            "status": "batch1_pass", "sample_id": first["sample_id"],
            "eeg_shape": list(eeg.shape), "mask_shape": list(mask.shape),
            "eeg_std": float(eeg.std()),
        })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
