"""Convert the monolithic GLIM-compatible pickle into row-addressable NPZ shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PT_TARGET_KEYS = [
    "lexical simplification (v0)", "lexical simplification (v1)",
    "semantic clarity (v0)", "semantic clarity (v1)",
    "syntax simplification (v0)", "syntax simplification (v1)",
    "naive rewritten", "naive simplified",
]
MODEL_METADATA = [
    "dataset", "task", "subject", "phase", "text uid", "input text",
    "sentiment label", "topic_label", "length", "surprisal", *PT_TARGET_KEYS,
]
OPTIONAL_METADATA = ["keyword_1", "keyword_2", "keyword_3"]
REQUIRED = {"eeg", "mask", *MODEL_METADATA}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/semkey_sharded"))
    parser.add_argument("--rows-per-shard", type=int, default=128)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.rows_per_shard <= 0:
        raise ValueError("rows-per-shard must be positive")
    source_hash = sha256(args.dataframe)
    if args.expected_sha256 and source_hash != args.expected_sha256:
        raise ValueError("source dataframe hash mismatch")

    frame = pd.read_pickle(args.dataframe)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"source dataframe missing columns: {sorted(missing)}")
    output = args.output_root
    shard_dir = output / "shards"
    metadata_dir = output / "metadata"
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    index_path = shard_dir / "index.csv"
    if not frame.index.is_unique:
        raise ValueError("source dataframe index must be unique")
    try:
        source_indices = frame.index.to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("source dataframe index must be integer-valued") from exc
    metadata_fields = MODEL_METADATA + [name for name in OPTIONAL_METADATA if name in frame.columns]
    index_fields = [
        "source_dataframe_row_index", "sample_id", "shard", "offset", "dataset", "task",
        "subject", "phase", "text_uid", *[name for name in metadata_fields if name not in {"dataset", "task", "subject", "phase", "text uid"}],
    ]
    shard_records = []
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fields)
        writer.writeheader()
        for shard_number, start in enumerate(range(0, len(frame), args.rows_per_shard)):
            stop = min(start + args.rows_per_shard, len(frame))
            part = frame.iloc[start:stop]
            eeg = np.stack(part["eeg"].to_list()).astype(np.float32, copy=False)
            mask = np.stack(part["mask"].to_list()).astype(np.int8, copy=False)
            row_indices = source_indices[start:stop]
            shard_name = f"shard_{shard_number:05d}.npz"
            shard_path = shard_dir / shard_name
            np.savez(shard_path, eeg=eeg, mask=mask, source_dataframe_row_index=row_indices)
            shard_hash = sha256(shard_path)
            shard_records.append({"name": shard_name, "rows": len(part), "sha256": shard_hash})
            for offset, (row_index, row) in enumerate(part.iterrows()):
                sample_id = f"{row['dataset']}::{row['task']}::{row['subject']}::row{int(row_index):06d}"
                record = {
                    "source_dataframe_row_index": int(row_index), "sample_id": sample_id,
                    "shard": shard_name, "offset": offset, "dataset": row["dataset"],
                    "task": row["task"], "subject": row["subject"], "phase": row["phase"],
                    "text_uid": row["text uid"],
                }
                for name in metadata_fields:
                    if name not in {"dataset", "task", "subject", "phase", "text uid"}:
                        record[name] = row[name]
                writer.writerow(record)
    manifest = {
        "schema_version": 1,
        "source_dataframe": str(args.dataframe),
        "source_dataframe_sha256": source_hash,
        "row_count": len(frame),
        "rows_per_shard": args.rows_per_shard,
        "metadata_columns": metadata_fields,
        "index": "shards/index.csv",
        "index_sha256": sha256(index_path),
        "shards": shard_records,
    }
    manifest_path = metadata_dir / "shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(frame), "shards": len(shard_records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
