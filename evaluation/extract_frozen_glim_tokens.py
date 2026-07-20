"""Extract and hash-bind GLIM's unpooled EEG token sequences [96, 1024].

Companion to ``extract_frozen_glim_vectors.py`` (which stores only the pooled
[1024] ``eeg_vector`` and is left untouched). This extractor stores the
``eeg_tokens`` [96, 1024] sequence that the same GLIM forward pass already
produces, for the token-level late-interaction retrieval experiment.

Only one condition is stored: correct EEG per trial. The matched-wrong-EEG
control is realised at evaluation time by looking up a donor trial's stored
tokens, so no separate wrong-EEG extraction is needed. Tokens are stored as
float16 (MaxSim L2-normalizes, so half precision is ample and halves the
~3.5 GB float32 footprint).

The extraction driver takes an ``embed`` callable and an ``eeg`` loader, so the
chunking/hashing/identity/resume logic is unit-testable with a fake embedder and
no GPU. ``GLIMTokenEmbedder`` supplies the real forward pass on Kaggle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from evaluation.extract_frozen_glim_vectors import atomic_json, atomic_npz

TOKEN_COUNT = 96
TOKEN_DIM = 1024
EmbedFn = Callable[[list, list, list, list, list], np.ndarray]
LoadEegFn = Callable[[dict], "tuple[np.ndarray, np.ndarray]"]


def token_chunk_sha256(
    tokens: np.ndarray, trial_ids: Sequence[str], sample_ids: Sequence[str],
    source_rows: Sequence[int],
) -> str:
    state = hashlib.sha256()
    state.update(np.ascontiguousarray(tokens).tobytes())
    state.update(f"{tokens.dtype}\x1f{tokens.shape}\n".encode("utf-8"))
    for tid, sid, row in zip(trial_ids, sample_ids, source_rows):
        state.update(f"{tid}\x1f{sid}\x1f{row}\n".encode("utf-8"))
    return state.hexdigest()


def _valid_existing_chunk(npz_path: Path, meta_path: Path, expected_sha: str) -> bool:
    if not npz_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("sha256") == expected_sha


def run_token_extraction(
    rows: list[dict],
    embed: EmbedFn,
    output_root: Path,
    *,
    checkpoint_sha256: str,
    glim_commit: str,
    prompt_mode: str,
    load_eeg: LoadEegFn,
    chunk_size: int = 64,
    dtype: str = "float16",
) -> dict:
    """Store correct-EEG token chunks with per-chunk hashes and an index.

    ``rows`` items require: ``trial_id``, ``sample_id``, ``source_dataframe_row_index``,
    ``reading_task``, plus whatever ``load_eeg`` needs. Resumable: a chunk whose
    npz+meta exist and whose recorded hash matches is not recomputed.
    """
    if prompt_mode not in {"canonical", "all_masked"}:
        raise ValueError("prompt_mode must be 'canonical' or 'all_masked'")
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be float16 or float32")
    np_dtype = np.float16 if dtype == "float16" else np.float32
    tokens_dir = output_root / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    chunk_entries: list[dict] = []
    total = 0
    for chunk_number, start in enumerate(range(0, len(rows), chunk_size)):
        batch = rows[start:start + chunk_size]
        trial_ids = [str(r["trial_id"]) for r in batch]
        sample_ids = [str(r["sample_id"]) for r in batch]
        source_rows = [int(r["source_dataframe_row_index"]) for r in batch]
        tasks = [str(r["reading_task"]) for r in batch]

        npz_path = tokens_dir / f"tokens_{chunk_number:05d}.npz"
        meta_path = tokens_dir / f"tokens_{chunk_number:05d}.json"

        eeg_list, mask_list = [], []
        for row in batch:
            eeg, mask = load_eeg(row)
            eeg_list.append(eeg)
            mask_list.append(mask)
        tokens = embed(eeg_list, mask_list, sample_ids, source_rows, tasks)
        tokens = np.ascontiguousarray(tokens).astype(np_dtype, copy=False)
        if tokens.shape != (len(batch), TOKEN_COUNT, TOKEN_DIM):
            raise ValueError(
                f"chunk {chunk_number}: expected {(len(batch), TOKEN_COUNT, TOKEN_DIM)}, got {tokens.shape}"
            )
        if not np.isfinite(tokens.astype(np.float32)).all():
            raise ValueError(f"chunk {chunk_number}: non-finite tokens")

        sha = token_chunk_sha256(tokens, trial_ids, sample_ids, source_rows)
        entry = {
            "chunk_number": chunk_number,
            "rows": len(batch),
            "offset": start,
            "token_file": f"tokens/tokens_{chunk_number:05d}.npz",
            "sha256": sha,
        }

        if not _valid_existing_chunk(npz_path, meta_path, sha):
            atomic_npz(npz_path, tokens=tokens)
            atomic_json(meta_path, {
                "chunk_number": chunk_number,
                "rows": len(batch),
                "offset": start,
                "token_shape": [TOKEN_COUNT, TOKEN_DIM],
                "dtype": dtype,
                "prompt_mode": prompt_mode,
                "checkpoint_sha256": checkpoint_sha256,
                "glim_commit": glim_commit,
                "trial_ids": trial_ids,
                "sample_ids": sample_ids,
                "source_dataframe_row_indices": source_rows,
                "reading_tasks": tasks,
                "sha256": sha,
            })
        chunk_entries.append(entry)
        total += len(batch)

    index = {
        "schema_version": 1,
        "total_rows": total,
        "num_chunks": len(chunk_entries),
        "token_shape": [TOKEN_COUNT, TOKEN_DIM],
        "dtype": dtype,
        "prompt_mode": prompt_mode,
        "checkpoint_sha256": checkpoint_sha256,
        "glim_commit": glim_commit,
        "chunk_size": chunk_size,
        "chunks": chunk_entries,
    }
    combined = hashlib.sha256()
    for entry in chunk_entries:
        combined.update(f"{entry['chunk_number']}\x1f{entry['sha256']}\n".encode("utf-8"))
    index["combined_chunk_sha256"] = combined.hexdigest()
    atomic_json(output_root / "token_index.json", index)
    return index


class GLIMTokenEmbedder:
    """Real GLIM forward pass returning ``eeg_tokens`` [B, 96, 1024] (Kaggle/GPU).

    Mirrors ``GLIMVectorEmbedder`` but returns the unpooled token sequence. Not
    unit-tested locally (needs the GLIM model and checkpoint); exercised on Kaggle.
    """

    def __init__(self, glim_root: Path, checkpoint: Path, device: str, prompt_mode: str = "all_masked"):
        import torch
        from project_adapters.glim_representation import (
            CanonicalGLIMRepresentationAdapter,
            load_upstream_glim_class,
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        glim_cls = load_upstream_glim_class(glim_root.resolve())
        model = glim_cls.load_from_checkpoint(str(checkpoint), map_location="cpu", strict=False)
        model.eval().to(self.device)
        self.adapter = CanonicalGLIMRepresentationAdapter(model).eval().to(self.device)
        if prompt_mode not in {"canonical", "all_masked"}:
            raise ValueError("prompt_mode must be 'canonical' or 'all_masked'")
        self.prompt_mode = prompt_mode

    def __call__(self, eeg_list, mask_list, sample_ids, source_rows, tasks) -> np.ndarray:
        torch = self.torch
        max_t = max(int(e.shape[0]) for e in eeg_list)
        channels = eeg_list[0].shape[1]
        batch = len(eeg_list)
        eeg = np.zeros((batch, max_t, channels), dtype=np.float32)
        mask = np.zeros((batch, max_t), dtype=np.int8)
        for i, (e, m) in enumerate(zip(eeg_list, mask_list)):
            eeg[i, : e.shape[0]] = e
            mask[i, : m.shape[0]] = m
        prompts = ([f"<{t}>" for t in tasks], ["<UNK>"] * batch, ["<UNK>"] * batch)
        with torch.inference_mode():
            output = self.adapter(
                torch.from_numpy(eeg).to(self.device),
                torch.from_numpy(mask).to(self.device),
                prompts,
                sample_ids=sample_ids,
                source_dataframe_row_indices=source_rows,
                mode=self.prompt_mode,
            )
            if output["sample_id"] != sample_ids or output["source_dataframe_row_index"] != source_rows:
                raise ValueError("GLIM adapter changed batch identities")
            tokens = output["eeg_tokens"].detach().float().cpu().numpy()
        if tokens.shape != (batch, TOKEN_COUNT, TOKEN_DIM):
            raise ValueError(f"GLIM returned tokens {tokens.shape}, expected {(batch, TOKEN_COUNT, TOKEN_DIM)}")
        return tokens
