"""Discover and validate a canonical Kaggle input; optionally load one real shard batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root: Path, relative: str) -> Path:
    matches = sorted(root.glob(f"*/{relative}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one */{relative} below {root}, found {matches}")
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
    index_path = dataset_root / manifest["index"]
    if sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("shard index hash mismatch")
    with index_path.open(encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    if len(index_rows) != manifest["row_count"]:
        raise ValueError("index row count mismatch")
    report = {
        "status": "metadata_pass", "dataset_root": str(dataset_root),
        "rows": len(index_rows), "shards": len(manifest["shards"]),
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
        if eeg.shape[0] != 1 or mask.shape[0] != 1 or not np.isfinite(eeg).all():
            raise ValueError("invalid real batch-1 shard payload")
        report.update({
            "status": "batch1_pass", "sample_id": first["sample_id"],
            "eeg_shape": list(eeg.shape), "mask_shape": list(mask.shape),
            "eeg_std": float(eeg.std()),
        })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
