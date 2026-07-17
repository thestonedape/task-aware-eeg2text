"""Extract resumable frozen GLIM text vectors for canonical train/validation text identities.

The output stores one vector per normalized text identity plus an identity-safe
trial-to-text mapping.  Held-out test rows are never embedded or exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INDEX_FIELDS = (
    "text_target_id", "normalized_text_sha256", "representative_trial_id",
    "representative_text", "vector_file", "vector_offset", "vector_dim",
    "checkpoint_sha256", "source_index_sha256",
)
MAPPING_FIELDS = (
    "trial_id", "split", "cohort", "dataset_version", "reading_task", "subject_id",
    "source_dataframe_row_index", "text_target_id", "normalized_text_sha256",
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def normalized_text(text: object) -> str:
    return " ".join(str(text).lower().split())


def text_target_id(text: object) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(dataset_root: Path, expected_index_sha256: str) -> tuple[list[dict[str, str]], dict]:
    source_manifest = json.loads(
        (dataset_root / "metadata" / "shard_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("schema_version") != 2:
        raise ValueError("expected canonical schema version 2")
    index_path = dataset_root / str(source_manifest["index"])
    actual_index_sha = sha256(index_path)
    if actual_index_sha != source_manifest.get("index_sha256") or actual_index_sha != expected_index_sha256:
        raise ValueError("canonical index SHA256 mismatch")
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "trial_id", "split", "cohort", "dataset_version", "reading_task", "subject_id",
        "source_dataframe_row_index", "text",
    }
    if not rows or required - set(rows[0]):
        raise ValueError(f"canonical index missing fields: {sorted(required - set(rows[0] if rows else ())) }")
    development = [row for row in rows if row["split"] in {"train", "val"}]
    if not development or any(row["split"] == "test" for row in development):
        raise AssertionError("held-out test row entered text-vector extraction")
    if len({row["trial_id"] for row in development}) != len(development):
        raise ValueError("duplicate development trial identity")
    return development, source_manifest


def build_text_records(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    by_identity: dict[str, list[dict[str, str]]] = {}
    mapping: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["trial_id"]):
        identity = text_target_id(row["text"])
        by_identity.setdefault(identity, []).append(row)
        mapping.append({
            "trial_id": row["trial_id"],
            "split": row["split"],
            "cohort": row["cohort"],
            "dataset_version": row["dataset_version"],
            "reading_task": row["reading_task"],
            "subject_id": row["subject_id"],
            "source_dataframe_row_index": row["source_dataframe_row_index"],
            "text_target_id": identity,
            "normalized_text_sha256": identity,
        })
    records = []
    for identity, members in sorted(by_identity.items()):
        representative = min(members, key=lambda item: item["trial_id"])
        normalized = {normalized_text(row["text"]) for row in members}
        if len(normalized) != 1:
            raise AssertionError("normalized-text SHA collision")
        records.append({
            "text_target_id": identity,
            "normalized_text_sha256": identity,
            "representative_trial_id": representative["trial_id"],
            "representative_text": representative["text"],
        })
    return records, mapping


def chunk_expected(
    number: int,
    records: list[dict[str, str]],
    vector_dim: int,
    checkpoint_sha256: str,
    source_index_sha256: str,
    text_model_id: str,
    text_dtype: str,
    batch_size: int,
) -> dict[str, object]:
    identity_payload = "\n".join(row["text_target_id"] for row in records).encode("utf-8")
    return {
        "chunk_number": number,
        "rows": len(records),
        "identity_sha256": hashlib.sha256(identity_payload).hexdigest(),
        "vector_dim": vector_dim,
        "dtype": "float32",
        "checkpoint_sha256": checkpoint_sha256,
        "source_index_sha256": source_index_sha256,
        "text_model_id": text_model_id,
        "text_dtype": text_dtype,
        "batch_size": batch_size,
        "extraction_contract_version": 1,
    }


def valid_chunk(npz_path: Path, metadata_path: Path, expected: dict[str, object]) -> bool:
    if not npz_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if metadata.get("vector_npz_sha256") != sha256(npz_path):
            return False
        with np.load(npz_path, allow_pickle=False) as archive:
            vectors = archive["vectors"]
            return bool(
                vectors.shape == (int(expected["rows"]), int(expected["vector_dim"]))
                and vectors.dtype == np.float32
                and np.isfinite(vectors).all()
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_extraction(
    dataset_root: Path,
    output_root: Path,
    embed_texts: Callable[[list[str]], np.ndarray],
    vector_dim: int,
    checkpoint_sha256: str,
    glim_commit: str,
    text_model_id: str,
    expected_index_sha256: str,
    text_dtype: str = "float32",
    batch_size: int = 64,
    chunk_size: int = 512,
    smoke_limit: int | None = None,
) -> dict[str, object]:
    if batch_size <= 0 or chunk_size < batch_size:
        raise ValueError("batch_size must be positive and chunk_size >= batch_size")
    rows, source_manifest = load_rows(dataset_root, expected_index_sha256)
    records, mapping = build_text_records(rows)
    if smoke_limit is not None:
        if smoke_limit <= 0:
            raise ValueError("smoke_limit must be positive")
        records = records[:smoke_limit]
        allowed = {row["text_target_id"] for row in records}
        mapping = [row for row in mapping if row["text_target_id"] in allowed]
    output_root.mkdir(parents=True, exist_ok=True)
    vector_root = output_root / "vectors"
    vector_root.mkdir(exist_ok=True)
    index_rows: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []
    reused = 0
    for number, start in enumerate(range(0, len(records), chunk_size)):
        chunk_records = records[start:start + chunk_size]
        relative_npz = f"vectors/text_{number:05d}.npz"
        npz_path = output_root / relative_npz
        metadata_path = output_root / f"vectors/text_{number:05d}.json"
        expected = chunk_expected(
            number, chunk_records, vector_dim, checkpoint_sha256,
            str(source_manifest["index_sha256"]), text_model_id, text_dtype, batch_size,
        )
        was_reused = valid_chunk(npz_path, metadata_path, expected)
        if was_reused:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reused += 1
        else:
            parts = []
            for batch_start in range(0, len(chunk_records), batch_size):
                texts = [row["representative_text"] for row in chunk_records[batch_start:batch_start + batch_size]]
                vectors = embed_texts(texts)
                if vectors.shape != (len(texts), vector_dim) or not np.isfinite(vectors).all():
                    raise ValueError(f"invalid text vector batch: {vectors.shape}")
                parts.append(vectors.astype(np.float32, copy=False))
            atomic_npz(npz_path, vectors=np.concatenate(parts, axis=0))
            metadata = {**expected, "vector_npz_sha256": sha256(npz_path)}
            atomic_json(metadata_path, metadata)
        chunks.append(metadata)
        for offset, row in enumerate(chunk_records):
            index_rows.append({
                **row,
                "vector_file": relative_npz,
                "vector_offset": offset,
                "vector_dim": vector_dim,
                "checkpoint_sha256": checkpoint_sha256,
                "source_index_sha256": source_manifest["index_sha256"],
            })
        print(f"text: chunk {number + 1}/{(len(records) + chunk_size - 1) // chunk_size} {'reused' if was_reused else 'written'}", flush=True)

    index_path = output_root / "text_vector_index.csv"
    mapping_path = output_root / "trial_text_targets.csv"
    write_csv(index_path, INDEX_FIELDS, index_rows)
    write_csv(mapping_path, MAPPING_FIELDS, mapping)
    if set(row["text_target_id"] for row in mapping) - set(row["text_target_id"] for row in index_rows):
        raise AssertionError("trial mapping references a missing text vector")
    split_counts = dict(sorted(Counter(row["split"] for row in mapping).items()))
    manifest = {
        "status": "pass",
        "schema_version": 1,
        "run_mode": "smoke" if smoke_limit else "full_development",
        "source_index_sha256": source_manifest["index_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "glim_commit": glim_commit,
        "text_model_id": text_model_id,
        "text_dtype": text_dtype,
        "vector_dim": vector_dim,
        "dtype": "float32",
        "unique_text_identities": len(index_rows),
        "mapped_trials": len(mapping),
        "split_counts": split_counts,
        "text_vector_index_sha256": sha256(index_path),
        "trial_text_targets_sha256": sha256(mapping_path),
        "chunks": chunks,
        "checks": {
            "held_out_test_accessed": False,
            "one_vector_per_normalized_text_identity": True,
            "all_mapped_trials_resolve": True,
            "vectors_finite": True,
        },
    }
    manifest_path = output_root / "text_vector_manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        "status": "pass",
        "run_mode": manifest["run_mode"],
        "unique_text_identities": len(index_rows),
        "mapped_trials": len(mapping),
        "chunks": len(chunks),
        "chunks_reused_this_invocation": reused,
        "manifest_sha256": sha256(manifest_path),
        "held_out_test_accessed": False,
        "output": str(output_root),
    }


class GLIMTextEmbedder:
    def __init__(self, glim_root: Path, checkpoint: Path, device: str):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from project_adapters.glim_representation import load_upstream_glim_class

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA text extraction requested but unavailable")
        GLIM = load_upstream_glim_class(glim_root.resolve())
        model = GLIM.load_from_checkpoint(str(checkpoint), map_location="cpu", strict=False)
        model.eval().to(self.device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model.tokenizer = AutoTokenizer.from_pretrained(model.text_model_id)
        model.text_model = T5ForConditionalGeneration.from_pretrained(
            model.text_model_id, torch_dtype=dtype
        ).requires_grad_(False).eval().to(self.device)
        model.eval()
        self.model = model
        self.vector_dim = int(model.embed_dim)
        self.text_model_id = str(model.text_model_id)
        self.text_dtype = str(dtype).replace("torch.", "")

    def __call__(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            ids, mask = self.model.tokenize(
                texts, int(self.model.input_text_len - self.model.prompt_tuning_len)
            )
            hidden, hidden_mask = self.model.encode_text(ids, mask)
            vectors = self.model.aligner.embed_text(hidden, hidden_mask)
            if vectors.ndim == 1:
                vectors = vectors.unsqueeze(0)
        return vectors.detach().float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--glim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--glim-commit", required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("GLIM checkpoint SHA256 mismatch")
    embedder = GLIMTextEmbedder(args.glim_root, args.checkpoint, args.device)
    report = run_extraction(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        embed_texts=embedder,
        vector_dim=embedder.vector_dim,
        checkpoint_sha256=checkpoint_sha,
        glim_commit=args.glim_commit,
        text_model_id=embedder.text_model_id,
        text_dtype=embedder.text_dtype,
        expected_index_sha256=args.expected_index_sha256,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        smoke_limit=args.smoke_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
