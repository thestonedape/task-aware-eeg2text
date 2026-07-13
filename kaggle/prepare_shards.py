"""Convert the monolithic GLIM dataframe into canonical row-addressable NPZ shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.canonical_contract import (  # noqa: E402
    CANONICAL_FIELDS,
    CANONICAL_MANIFEST_FIELDS,
    audit_records,
    canonical_record,
    clean,
    compare_frozen_validation,
    load_nr_relation_map,
)


PT_TARGET_KEYS = [
    "lexical simplification (v0)", "lexical simplification (v1)",
    "semantic clarity (v0)", "semantic clarity (v1)",
    "syntax simplification (v0)", "syntax simplification (v1)",
    "naive rewritten", "naive simplified",
]
RAW_METADATA = [
    "label id", "raw text", "raw label", "control", "relation label", "sentiment label",
]
OPTIONAL_SOURCE_METADATA = [
    "semkey_sentiment_2", "topic_label", "surprisal", "keyword_1", "keyword_2", "keyword_3",
]
REQUIRED = {
    "eeg", "mask", "dataset", "task", "subject", "phase", "text uid", "input text",
    "sentiment label", "relation label", *PT_TARGET_KEYS,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/semkey_sharded"))
    parser.add_argument("--rows-per-shard", type=int, default=128)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--label-dir", type=Path, default=ROOT / "preprocess" / "resource" / "revised_csv"
    )
    parser.add_argument("--expected-validation-manifest", type=Path)
    args = parser.parse_args()
    if args.rows_per_shard <= 0:
        raise ValueError("rows-per-shard must be positive")
    source_hash = sha256(args.dataframe)
    if args.expected_sha256 and source_hash != args.expected_sha256:
        raise ValueError("source dataframe hash mismatch")

    frame = pd.read_pickle(args.dataframe)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"source dataframe missing canonical source columns: {sorted(missing)}")
    if not frame.index.is_unique:
        raise ValueError("source dataframe index must be unique")
    try:
        source_indices = frame.index.to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source dataframe index must be integer-valued") from exc

    output = args.output_root
    shard_dir = output / "shards"
    metadata_dir = output / "metadata"
    manifest_dir = output / "manifests"
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    nr_relation_map = load_nr_relation_map(args.label_dir)
    canonical_records: list[dict[str, object]] = []
    index_records: list[dict[str, object]] = []
    shard_records = []
    raw_fields = [field for field in RAW_METADATA if field in frame.columns]
    optional_fields = [field for field in OPTIONAL_SOURCE_METADATA if field in frame.columns]

    for shard_number, start in enumerate(range(0, len(frame), args.rows_per_shard)):
        stop = min(start + args.rows_per_shard, len(frame))
        part = frame.iloc[start:stop]
        eeg = np.stack(part["eeg"].to_list()).astype(np.float32, copy=False)
        mask = np.stack(part["mask"].to_list()).astype(np.int8, copy=False)
        row_indices = source_indices[start:stop]
        shard_name = f"shard_{shard_number:05d}.npz"
        shard_path = shard_dir / shard_name
        np.savez(shard_path, eeg=eeg, mask=mask, source_dataframe_row_index=row_indices)
        shard_records.append({"name": shard_name, "rows": len(part), "sha256": sha256(shard_path)})

        for offset, (row_index, row) in enumerate(part.iterrows()):
            canonical = canonical_record(
                row, int(row_index), source_hash, shard_name, offset, nr_relation_map
            )
            canonical_records.append(canonical)
            record = {
                "source_dataframe_row_index": int(row_index),
                "sample_id": canonical["trial_id"],
                "shard": shard_name,
                "offset": offset,
                "dataset": canonical["dataset_version"],
                "task": canonical["raw_task"],
                "subject": canonical["subject_id"],
                "phase": canonical["split"],
                "text_uid": canonical["text_uid"],
                "input text": canonical["text"],
            }
            for field in raw_fields:
                record[field] = clean(row.get(field))
            for field in optional_fields:
                record[field] = clean(row.get(field))
            for field in PT_TARGET_KEYS:
                record[field] = clean(row.get(field))
            record.update(canonical)
            index_records.append(record)

    audit = audit_records(canonical_records)
    validation_check = None
    if args.expected_validation_manifest:
        validation_check = compare_frozen_validation(canonical_records, args.expected_validation_manifest)
    if source_hash == "2fefc942859a0af06cb2058df33c5681783e13266920d65ffe5e772b7a2fcfea":
        expected_counts = {"train": 17908, "val": 2200, "test": 2227}
        if audit["row_count"] != 22335 or audit["phase_counts"] != expected_counts:
            raise ValueError(f"frozen source count mismatch: {audit}")

    canonical_path = manifest_dir / "canonical_full_manifest.csv"
    write_csv(canonical_path, CANONICAL_MANIFEST_FIELDS, canonical_records)
    index_fields = list(index_records[0])
    index_path = shard_dir / "index.csv"
    write_csv(index_path, index_fields, index_records)

    contract_report = {
        "status": "pass",
        "source_dataframe_sha256": source_hash,
        **audit,
        "frozen_validation_check": validation_check or {"status": "not_requested"},
        "semkey_generated_labels": {
            "status": "not_fabricated",
            "note": "topic/surprisal/binary sentiment require a separate versioned text-label enrichment",
        },
    }
    contract_path = metadata_dir / "canonical_full_contract_report.json"
    contract_path.write_text(json.dumps(contract_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "source_dataframe": str(args.dataframe),
        "source_dataframe_sha256": source_hash,
        "row_count": len(frame),
        "rows_per_shard": args.rows_per_shard,
        "index": "shards/index.csv",
        "index_sha256": sha256(index_path),
        "canonical_manifest": "manifests/canonical_full_manifest.csv",
        "canonical_manifest_sha256": sha256(canonical_path),
        "canonical_contract_report": "metadata/canonical_full_contract_report.json",
        "canonical_contract_report_sha256": sha256(contract_path),
        "shards": shard_records,
    }
    manifest_path = metadata_dir / "shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", **audit, "shards": len(shard_records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
