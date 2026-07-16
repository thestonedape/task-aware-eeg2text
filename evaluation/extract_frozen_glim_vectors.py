"""Extract resumable identity-safe frozen GLIM global vectors and controls.

Correct vectors are extracted for canonical train and validation trials. Matched
wrong-real, zero, and train-scale-matched Gaussian vectors are validation-only.
The held-out test split is never addressable through this CLI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.build_protocol_manifests import (  # noqa: E402
    audit_split_identity,
    build_wrong_eeg_donors,
    read_index,
    sha256,
    stable_hash,
    write_csv,
)


CONDITIONS = (
    "correct_train",
    "correct_val",
    "matched_wrong_val",
    "zero_val",
    "gaussian_val",
)
TASK_PROMPTS = {"SR": "<SR>", "NR": "<NR>", "TSR": "<TSR>"}
DONOR_FIELDS = [
    "target_trial_id", "target_dataset_version", "target_reading_task", "target_subject_id",
    "random_valid_trial_id", "task_dataset_wrong_trial_id", "task_dataset_length_difference",
    "subject_length_wrong_trial_id", "subject_length_difference", "subject_matched_available",
]
VECTOR_INDEX_FIELDS = [
    "condition", "phase", "target_trial_id", "signal_trial_id",
    "target_source_dataframe_row_index", "signal_source_dataframe_row_index",
    "dataset_version", "reading_task", "subject_id", "text_uid", "vector_file",
    "vector_offset", "vector_dim", "prompt_mode", "checkpoint_sha256", "source_index_sha256",
]


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def identity_sha256(records: list[dict[str, object]]) -> str:
    state = hashlib.sha256()
    for record in records:
        state.update(
            (
                f"{record['condition']}\x1f{record['target']['trial_id']}\x1f"
                f"{record.get('signal_trial_id', '')}\n"
            ).encode("utf-8")
        )
    return state.hexdigest()


class ShardSignalStore:
    """Small LRU cache over read-only canonical NPZ shards."""

    def __init__(self, dataset_root: Path, max_open: int = 3):
        self.dataset_root = dataset_root
        self.max_open = max_open
        self.archives: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def load(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        shard = row["shard"]
        arrays = self.archives.pop(shard, None)
        if arrays is None:
            with np.load(self.dataset_root / "shards" / shard, allow_pickle=False) as archive:
                arrays = (archive["eeg"], archive["mask"])
        self.archives[shard] = arrays
        while len(self.archives) > self.max_open:
            self.archives.popitem(last=False)
        offset = int(row["offset"])
        eeg = arrays[0][offset].astype(np.float32, copy=False)
        mask = arrays[1][offset].astype(np.int8, copy=False)
        if eeg.ndim != 2 or mask.ndim != 1 or eeg.shape[0] != mask.shape[0]:
            raise ValueError(f"{row['trial_id']}: invalid EEG/mask shapes {eeg.shape}, {mask.shape}")
        if not np.isfinite(eeg).all():
            raise ValueError(f"{row['trial_id']}: non-finite EEG input")
        return eeg, mask

    def close(self) -> None:
        self.archives.clear()


def load_development_rows(
    dataset_root: Path, expected_index_sha256: str | None
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    all_rows, manifest, _ = read_index(dataset_root)
    required = {
        "shard", "offset", "sample_id", "source_dataframe_row_index", "dataset_version",
        "reading_task", "subject_id", "text_uid", "trial_id", "split",
    }
    missing = required - set(all_rows[0])
    if missing:
        raise ValueError(f"canonical index missing vector fields: {sorted(missing)}")
    if expected_index_sha256 and manifest["index_sha256"] != expected_index_sha256:
        raise ValueError("canonical index does not match the frozen protocol")
    audit_split_identity(all_rows)
    train = sorted(
        (row for row in all_rows if row["split"] == "train"),
        key=lambda row: int(row["source_dataframe_row_index"]),
    )
    validation = sorted(
        (row for row in all_rows if row["split"] == "val"),
        key=lambda row: int(row["source_dataframe_row_index"]),
    )
    if not train or not validation:
        raise ValueError("canonical train and validation rows are required")
    if any(row["split"] == "test" for row in [*train, *validation]):
        raise AssertionError("held-out test row entered vector extraction")
    if any(row["sample_id"] != row["trial_id"] for row in [*train, *validation]):
        raise ValueError("sample_id and canonical trial_id diverge")
    return train, validation, manifest


def freeze_donors(
    validation: list[dict[str, str]], output_root: Path, expected_sha256: str | None
) -> tuple[dict[str, str], str]:
    donors = build_wrong_eeg_donors(validation, seed=20260716)
    if not all(int(row["subject_matched_available"]) == 1 for row in donors):
        raise ValueError("subject-matched wrong EEG is not available for every validation target")
    path = output_root / "frozen_wrong_eeg_donors.csv"
    write_csv(path, DONOR_FIELDS, donors)
    digest = sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"wrong-EEG donor hash mismatch: {digest}")
    return {
        str(row["target_trial_id"]): str(row["subject_length_wrong_trial_id"])
        for row in donors
    }, digest


def stats_identity(rows: list[dict[str, str]]) -> str:
    return stable_hash(*(row["trial_id"] for row in rows))


def compute_train_signal_stats(
    store: ShardSignalStore,
    rows: list[dict[str, str]],
    output_root: Path,
    source_index_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    npz_path = output_root / "train_signal_stats.npz"
    metadata_path = output_root / "train_signal_stats.json"
    expected_identity = stats_identity(rows)
    if npz_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("row_identity_sha256") == expected_identity
            and metadata.get("source_index_sha256") == source_index_sha256
            and metadata.get("stats_npz_sha256") == sha256(npz_path)
        ):
            with np.load(npz_path, allow_pickle=False) as archive:
                mean = archive["channel_mean"].astype(np.float32)
                std = archive["channel_std"].astype(np.float32)
            return mean, std, metadata

    total = None
    total_sq = None
    valid_timepoints = 0
    expected_shape = None
    for row_number, row in enumerate(rows, start=1):
        eeg, mask = store.load(row)
        expected_shape = expected_shape or tuple(eeg.shape)
        if tuple(eeg.shape) != expected_shape:
            raise ValueError("inconsistent EEG shape while computing training statistics")
        valid = mask.astype(bool)
        if not valid.any():
            raise ValueError(f"{row['trial_id']}: no valid EEG timepoints")
        values = eeg[valid].astype(np.float64, copy=False)
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            total_sq = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        valid_timepoints += values.shape[0]
        if row_number % 256 == 0 or row_number == len(rows):
            print(
                f"training signal statistics: {row_number}/{len(rows)} rows",
                flush=True,
            )
    if total is None or total_sq is None or valid_timepoints <= 0:
        raise ValueError("no training signal statistics were accumulated")
    mean64 = total / valid_timepoints
    variance = np.maximum(total_sq / valid_timepoints - np.square(mean64), 1e-12)
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance).astype(np.float32)
    atomic_npz(
        npz_path,
        channel_mean=mean,
        channel_std=std,
        valid_timepoints=np.asarray([valid_timepoints], dtype=np.int64),
    )
    metadata = {
        "status": "pass",
        "source_index_sha256": source_index_sha256,
        "row_identity_sha256": expected_identity,
        "training_rows": len(rows),
        "valid_timepoints": valid_timepoints,
        "channels": len(mean),
        "stats_npz_sha256": sha256(npz_path),
        "algorithm": "per_channel_mean_std_over_masked_valid_training_timepoints_float64_accumulation",
    }
    atomic_json(metadata_path, metadata)
    return mean, std, metadata


def build_condition_records(
    train: list[dict[str, str]],
    validation: list[dict[str, str]],
    donor_map: dict[str, str],
    smoke_limit: int | None,
) -> dict[str, list[dict[str, object]]]:
    by_id = {row["trial_id"]: row for row in [*train, *validation]}
    train_targets = train[:smoke_limit] if smoke_limit else train
    val_targets = validation[:smoke_limit] if smoke_limit else validation
    result: dict[str, list[dict[str, object]]] = {condition: [] for condition in CONDITIONS}
    for row in train_targets:
        result["correct_train"].append(
            {"condition": "correct_train", "target": row, "signal": row, "signal_trial_id": row["trial_id"]}
        )
    for row in val_targets:
        result["correct_val"].append(
            {"condition": "correct_val", "target": row, "signal": row, "signal_trial_id": row["trial_id"]}
        )
        donor_id = donor_map[row["trial_id"]]
        donor = by_id.get(donor_id)
        if donor is None or donor["split"] != "val":
            raise ValueError(f"invalid frozen validation donor {donor_id!r}")
        result["matched_wrong_val"].append(
            {"condition": "matched_wrong_val", "target": row, "signal": donor, "signal_trial_id": donor_id}
        )
        result["zero_val"].append(
            {"condition": "zero_val", "target": row, "signal": None, "signal_trial_id": f"zero::{row['trial_id']}"}
        )
        result["gaussian_val"].append(
            {
                "condition": "gaussian_val", "target": row, "signal": None,
                "signal_trial_id": f"gaussian::{row['trial_id']}",
            }
        )
    return result


def record_arrays(
    record: dict[str, object],
    store: ShardSignalStore,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    gaussian_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = record["target"]
    assert isinstance(target, dict)
    condition = str(record["condition"])
    if condition in {"correct_train", "correct_val", "matched_wrong_val"}:
        signal = record["signal"]
        assert isinstance(signal, dict)
        return store.load(signal)
    target_eeg, target_mask = store.load(target)
    if condition == "zero_val":
        return np.zeros_like(target_eeg, dtype=np.float32), target_mask
    if condition == "gaussian_val":
        seed = int(stable_hash(gaussian_seed, target["trial_id"], "gaussian")[:16], 16)
        rng = np.random.default_rng(seed)
        eeg = rng.normal(
            loc=channel_mean,
            scale=channel_std,
            size=target_eeg.shape,
        ).astype(np.float32)
        eeg[~target_mask.astype(bool)] = 0.0
        return eeg, target_mask
    raise ValueError(condition)


def chunk_base_metadata(
    condition: str,
    chunk_number: int,
    records: list[dict[str, object]],
    vector_dim: int,
    checkpoint_sha256: str,
    source_index_sha256: str,
    gaussian_seed: int,
    train_signal_stats_sha256: str,
    batch_size: int,
) -> dict[str, object]:
    return {
        "condition": condition,
        "chunk_number": chunk_number,
        "rows": len(records),
        "vector_dim": vector_dim,
        "identity_sha256": identity_sha256(records),
        "checkpoint_sha256": checkpoint_sha256,
        "source_index_sha256": source_index_sha256,
        "prompt_mode": "canonical",
        "dtype": "float32",
        "gaussian_seed": gaussian_seed,
        "train_signal_stats_sha256": train_signal_stats_sha256,
        "batch_size": batch_size,
        "extraction_contract_version": 1,
    }


def valid_existing_chunk(npz_path: Path, meta_path: Path, expected: dict[str, object]) -> bool:
    if not npz_path.is_file() or not meta_path.is_file():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if metadata.get("vector_npz_sha256") != sha256(npz_path):
            return False
        with np.load(npz_path, allow_pickle=False) as archive:
            vectors = archive["vectors"]
            return (
                vectors.shape == (int(expected["rows"]), int(expected["vector_dim"]))
                and vectors.dtype == np.float32
                and np.isfinite(vectors).all()
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def extract_condition(
    condition: str,
    records: list[dict[str, object]],
    output_root: Path,
    store: ShardSignalStore,
    embed_batch: Callable[[np.ndarray, np.ndarray, tuple[list[str], list[str], list[str]], list[str], list[int]], np.ndarray],
    vector_dim: int,
    batch_size: int,
    chunk_size: int,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    gaussian_seed: int,
    checkpoint_sha256: str,
    source_index_sha256: str,
    train_signal_stats_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    vector_dir = output_root / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    master_rows: list[dict[str, object]] = []
    chunk_manifest: list[dict[str, object]] = []
    reused = 0
    total_chunks = (len(records) + chunk_size - 1) // chunk_size
    print(f"{condition}: {len(records)} rows in {total_chunks} chunks", flush=True)
    for chunk_number, start in enumerate(range(0, len(records), chunk_size)):
        chunk_records = records[start:start + chunk_size]
        relative_npz = f"vectors/{condition}_{chunk_number:05d}.npz"
        npz_path = output_root / relative_npz
        meta_path = output_root / f"vectors/{condition}_{chunk_number:05d}.json"
        expected = chunk_base_metadata(
            condition, chunk_number, chunk_records, vector_dim, checkpoint_sha256,
            source_index_sha256, gaussian_seed, str(train_signal_stats_sha256), batch_size
        )
        was_reused = valid_existing_chunk(npz_path, meta_path, expected)
        if was_reused:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            reused += 1
        else:
            for path in (npz_path, meta_path):
                if path.exists():
                    path.unlink()
            parts: list[np.ndarray] = []
            for batch_start in range(0, len(chunk_records), batch_size):
                batch_records = chunk_records[batch_start:batch_start + batch_size]
                eeg_items = []
                mask_items = []
                tasks = []
                datasets = []
                subjects = []
                target_ids = []
                target_rows = []
                for record in batch_records:
                    eeg, mask = record_arrays(
                        record, store, channel_mean, channel_std, gaussian_seed
                    )
                    target = record["target"]
                    assert isinstance(target, dict)
                    eeg_items.append(eeg)
                    mask_items.append(mask)
                    tasks.append(TASK_PROMPTS[target["reading_task"]])
                    datasets.append(target["dataset_version"])
                    subjects.append(target["subject_id"])
                    target_ids.append(target["trial_id"])
                    target_rows.append(int(target["source_dataframe_row_index"]))
                vectors = embed_batch(
                    np.stack(eeg_items).astype(np.float32, copy=False),
                    np.stack(mask_items).astype(np.int8, copy=False),
                    (tasks, datasets, subjects),
                    target_ids,
                    target_rows,
                )
                if vectors.shape != (len(batch_records), vector_dim) or not np.isfinite(vectors).all():
                    raise ValueError(f"invalid vector batch for {condition}: {vectors.shape}")
                parts.append(vectors.astype(np.float32, copy=False))
            chunk_vectors = np.concatenate(parts, axis=0)
            atomic_npz(npz_path, vectors=chunk_vectors)
            metadata = {**expected, "vector_npz_sha256": sha256(npz_path)}
            atomic_json(meta_path, metadata)
        print(
            f"{condition}: chunk {chunk_number + 1}/{total_chunks} "
            f"{'reused' if was_reused else 'written'}",
            flush=True,
        )
        chunk_manifest.append(metadata)
        for offset, record in enumerate(chunk_records):
            target = record["target"]
            signal = record["signal"]
            assert isinstance(target, dict)
            master_rows.append(
                {
                    "condition": condition,
                    "phase": target["split"],
                    "target_trial_id": target["trial_id"],
                    "signal_trial_id": record["signal_trial_id"],
                    "target_source_dataframe_row_index": target["source_dataframe_row_index"],
                    "signal_source_dataframe_row_index": (
                        signal["source_dataframe_row_index"] if isinstance(signal, dict) else ""
                    ),
                    "dataset_version": target["dataset_version"],
                    "reading_task": target["reading_task"],
                    "subject_id": target["subject_id"],
                    "text_uid": target["text_uid"],
                    "vector_file": relative_npz,
                    "vector_offset": offset,
                    "vector_dim": vector_dim,
                    "prompt_mode": "canonical",
                    "checkpoint_sha256": checkpoint_sha256,
                    "source_index_sha256": source_index_sha256,
                }
            )
    return master_rows, chunk_manifest, reused


def run_extraction(
    dataset_root: Path,
    output_root: Path,
    embed_batch: Callable,
    vector_dim: int,
    checkpoint_sha256: str,
    glim_commit: str,
    expected_index_sha256: str | None = None,
    expected_donor_sha256: str | None = None,
    batch_size: int = 8,
    chunk_size: int = 128,
    gaussian_seed: int = 20260716,
    smoke_limit: int | None = None,
    expected_signal_shape: tuple[int, int] | None = (1280, 128),
) -> dict[str, object]:
    if batch_size <= 0 or chunk_size <= 0 or chunk_size < batch_size:
        raise ValueError("batch_size and chunk_size must be positive; chunk_size >= batch_size")
    output_root.mkdir(parents=True, exist_ok=True)
    train, validation, source_manifest = load_development_rows(dataset_root, expected_index_sha256)
    donor_map, donor_sha = freeze_donors(validation, output_root, expected_donor_sha256)
    records = build_condition_records(train, validation, donor_map, smoke_limit)
    stats_rows = train[: min(len(train), 128)] if smoke_limit else train
    store = ShardSignalStore(dataset_root)
    try:
        first_eeg, _ = store.load(train[0])
        if expected_signal_shape and tuple(first_eeg.shape) != expected_signal_shape:
            raise ValueError(f"expected signal shape {expected_signal_shape}, got {first_eeg.shape}")
        channel_mean, channel_std, stats_metadata = compute_train_signal_stats(
            store, stats_rows, output_root, str(source_manifest["index_sha256"])
        )
        if channel_mean.shape != (first_eeg.shape[1],) or channel_std.shape != channel_mean.shape:
            raise ValueError("training channel statistics have invalid shape")
        all_index_rows: list[dict[str, object]] = []
        all_chunks: list[dict[str, object]] = []
        reused_chunks = 0
        for condition in CONDITIONS:
            index_rows, chunks, reused = extract_condition(
                condition=condition,
                records=records[condition],
                output_root=output_root,
                store=store,
                embed_batch=embed_batch,
                vector_dim=vector_dim,
                batch_size=batch_size,
                chunk_size=chunk_size,
                channel_mean=channel_mean,
                channel_std=channel_std,
                gaussian_seed=gaussian_seed,
                checkpoint_sha256=checkpoint_sha256,
                source_index_sha256=str(source_manifest["index_sha256"]),
                train_signal_stats_sha256=str(stats_metadata["stats_npz_sha256"]),
            )
            all_index_rows.extend(index_rows)
            all_chunks.extend(chunks)
            reused_chunks += reused
    finally:
        store.close()

    vector_index_path = output_root / "vector_index.csv"
    write_csv(vector_index_path, VECTOR_INDEX_FIELDS, all_index_rows)
    condition_counts = dict(sorted(Counter(row["condition"] for row in all_index_rows).items()))
    expected_counts = {
        "correct_train": min(smoke_limit, len(train)) if smoke_limit else len(train),
        "correct_val": min(smoke_limit, len(validation)) if smoke_limit else len(validation),
        "matched_wrong_val": min(smoke_limit, len(validation)) if smoke_limit else len(validation),
        "zero_val": min(smoke_limit, len(validation)) if smoke_limit else len(validation),
        "gaussian_val": min(smoke_limit, len(validation)) if smoke_limit else len(validation),
    }
    if condition_counts != expected_counts:
        raise AssertionError((condition_counts, expected_counts))
    if any(row["phase"] == "test" for row in all_index_rows):
        raise AssertionError("held-out test row entered vector index")
    manifest = {
        "status": "pass",
        "schema_version": 1,
        "run_mode": "smoke" if smoke_limit else "full_development",
        "source_index_sha256": source_manifest["index_sha256"],
        "source_dataframe_sha256": source_manifest.get("source_dataframe_sha256", ""),
        "checkpoint_sha256": checkpoint_sha256,
        "glim_commit": glim_commit,
        "prompt_mode": "canonical",
        "vector_dim": vector_dim,
        "dtype": "float32",
        "gaussian_seed": gaussian_seed,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "gaussian_stats": stats_metadata,
        "wrong_eeg_donor_sha256": donor_sha,
        "condition_counts": condition_counts,
        "vector_index": "vector_index.csv",
        "vector_index_sha256": sha256(vector_index_path),
        "chunks": all_chunks,
        "checks": {
            "held_out_test_accessed": False,
            "target_identity_preserved": True,
            "matched_wrong_changes_signal_only": True,
            "zero_and_gaussian_keep_target_metadata": True,
            "gaussian_uses_training_statistics_only": True,
            "checkpoint_and_source_pinned": True,
            "vectors_finite": True,
        },
    }
    manifest_path = output_root / "vector_manifest.json"
    atomic_json(manifest_path, manifest)
    report = {
        "status": "pass",
        "run_mode": manifest["run_mode"],
        "condition_counts": condition_counts,
        "chunks": len(all_chunks),
        "chunks_reused_this_invocation": reused_chunks,
        "vector_dim": vector_dim,
        "vector_index_sha256": manifest["vector_index_sha256"],
        "wrong_eeg_donor_sha256": donor_sha,
        "train_signal_stats_sha256": stats_metadata["stats_npz_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "held_out_test_accessed": False,
        "output": str(output_root),
    }
    return report


class GLIMVectorEmbedder:
    def __init__(self, glim_root: Path, checkpoint: Path, device: str):
        import torch
        from project_adapters.glim_representation import (
            CanonicalGLIMRepresentationAdapter,
            load_upstream_glim_class,
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA extraction requested but unavailable")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        GLIM = load_upstream_glim_class(glim_root.resolve())
        model = GLIM.load_from_checkpoint(str(checkpoint), map_location="cpu", strict=False)
        model.eval().to(self.device)
        self.adapter = CanonicalGLIMRepresentationAdapter(model).eval().to(self.device)
        self.vector_dim = 1024

    def __call__(
        self,
        eeg: np.ndarray,
        mask: np.ndarray,
        prompts: tuple[list[str], list[str], list[str]],
        sample_ids: list[str],
        source_rows: list[int],
    ) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            output = self.adapter(
                torch.from_numpy(eeg).to(self.device),
                torch.from_numpy(mask).to(self.device),
                prompts,
                sample_ids=sample_ids,
                source_dataframe_row_indices=source_rows,
                mode="canonical",
            )
            if output["sample_id"] != sample_ids or output["source_dataframe_row_index"] != source_rows:
                raise ValueError("GLIM adapter changed batch identities")
            vectors = output["eeg_vector"].detach().float().cpu().numpy()
        return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--glim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--glim-commit", required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-donor-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--gaussian-seed", type=int, default=20260716)
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("GLIM checkpoint SHA256 mismatch")
    embedder = GLIMVectorEmbedder(args.glim_root, args.checkpoint, args.device)
    report = run_extraction(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        embed_batch=embedder,
        vector_dim=embedder.vector_dim,
        checkpoint_sha256=checkpoint_sha,
        glim_commit=args.glim_commit,
        expected_index_sha256=args.expected_index_sha256,
        expected_donor_sha256=args.expected_donor_sha256,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        gaussian_seed=args.gaussian_seed,
        smoke_limit=args.smoke_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
