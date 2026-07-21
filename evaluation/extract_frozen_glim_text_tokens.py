"""Extract and hash-bind GLIM's unpooled text token sequences [L, 1024].

Companion to ``extract_frozen_glim_text_vectors.py`` (which stores only the
pooled [1024] text vector and is left untouched). This extractor stores the
unpooled ``encode_text`` hidden states [L, 1024] plus their attention mask, for
the text side of the token-level MaxSim retrieval experiment.

Why these are the right text tokens. GLIM's aligner (``model/modules.py``
``Aligner``) pools text with a single learned query ``q_y`` cross-attention
(``embed_text``); the *unpooled* input to that pool is exactly ``encode_text``'s
``last_hidden_state`` [n, L, e]. GLIM's commitment loss trains
``in_proj(eeg_hidden) approx text_hidden`` token-wise (``align_embeds``, MSE), so
the stored EEG tokens (``in_proj``-projected) and these text tokens already live
in the same 1024-d space -- which is what makes token-level MaxSim well posed.

One forward pass per text identity produces BOTH the unpooled tokens (+ mask) and
the pooled ``embed_text`` vector; all three are stored (tokens+mask feed MaxSim,
the pooled vector feeds the pooled-cosine arm and lets us verify against the
existing frozen text vectors). Dedup is by normalized-text identity, exactly as
the vector extractor, and the held-out test split is never embedded. Stored as
float16 (MaxSim/cosine L2-normalize).

The extraction driver takes an ``embed`` callable, so the chunking / hashing /
identity / resume logic is unit-testable with a fake embedder and no GPU.
``GLIMTextTokenEmbedder`` supplies the real forward pass on Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from evaluation.extract_frozen_glim_text_vectors import (
    atomic_json,
    atomic_npz,
    build_text_records,
    load_rows,
    sha256,
    write_csv,
)

TOKEN_DIM = 1024
MAPPING_FIELDS = (
    "trial_id", "split", "cohort", "dataset_version", "reading_task", "subject_id",
    "source_dataframe_row_index", "text_target_id", "normalized_text_sha256",
)
INDEX_FIELDS = (
    "text_target_id", "normalized_text_sha256", "representative_trial_id",
    "token_file", "token_offset", "token_len", "token_dim",
    "checkpoint_sha256", "source_index_sha256",
)

TextEmbedFn = Callable[[list], "tuple[np.ndarray, np.ndarray, np.ndarray]"]


def text_token_chunk_sha256(
    tokens: np.ndarray, masks: np.ndarray, vectors: np.ndarray, text_target_ids: list[str]
) -> str:
    state = hashlib.sha256()
    state.update(np.ascontiguousarray(tokens).tobytes())
    state.update(np.ascontiguousarray(masks).tobytes())
    state.update(np.ascontiguousarray(vectors).tobytes())
    state.update(
        f"{tokens.dtype}\x1f{tokens.shape}\x1f{masks.dtype}\x1f{vectors.shape}\n".encode("utf-8")
    )
    for tid in text_target_ids:
        state.update(f"{tid}\n".encode("utf-8"))
    return state.hexdigest()


def _valid_existing_chunk(npz_path: Path, meta_path: Path, expected_sha: str) -> bool:
    if not npz_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("sha256") == expected_sha


def run_text_token_extraction(
    records: list[dict],
    mapping: list[dict],
    output_root: Path,
    embed: TextEmbedFn,
    *,
    checkpoint_sha256: str,
    glim_commit: str,
    text_model_id: str,
    source_index_sha256: str,
    token_len: int,
    dtype: str = "float16",
    chunk_size: int = 256,
    run_mode: str = "full_development",
) -> dict:
    """Store one unpooled token chunk per group of text identities, hash-bound.

    ``records`` (deduped text identities) and ``mapping`` (trial->text) come from
    ``build_text_records``; keeping them as inputs makes this unit-testable with a
    fake embedder and no dataset. Resumable: a chunk whose npz+meta exist and
    whose recorded hash matches is not recomputed.
    """
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be float16 or float32")
    np_dtype = np.float16 if dtype == "float16" else np.float32
    source_index_sha = str(source_index_sha256)
    tokens_dir = output_root / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    chunk_entries: list[dict] = []
    index_rows: list[dict] = []
    reused = 0
    for chunk_number, start in enumerate(range(0, len(records), chunk_size)):
        batch = records[start:start + chunk_size]
        target_ids = [r["text_target_id"] for r in batch]
        texts = [r["representative_text"] for r in batch]

        npz_path = tokens_dir / f"text_{chunk_number:05d}.npz"
        meta_path = tokens_dir / f"text_{chunk_number:05d}.json"

        tokens, masks, vectors = embed(texts)
        tokens = np.ascontiguousarray(tokens).astype(np_dtype, copy=False)
        masks = np.ascontiguousarray(masks).astype(np.int8, copy=False)
        # Pooled vector kept float32 (matches the frozen text-vector storage) so the
        # Gate-2 identity comparison measures pure COMPUTE agreement, not conflated
        # with this side's float16 quantization. Tokens stay float16 for MaxSim.
        vectors = np.ascontiguousarray(vectors).astype(np.float32, copy=False)
        if tokens.shape != (len(batch), token_len, TOKEN_DIM):
            raise ValueError(
                f"chunk {chunk_number}: expected tokens {(len(batch), token_len, TOKEN_DIM)}, got {tokens.shape}"
            )
        if masks.shape != (len(batch), token_len):
            raise ValueError(f"chunk {chunk_number}: expected masks {(len(batch), token_len)}, got {masks.shape}")
        if vectors.shape != (len(batch), TOKEN_DIM):
            raise ValueError(f"chunk {chunk_number}: expected vectors {(len(batch), TOKEN_DIM)}, got {vectors.shape}")
        if not np.isfinite(tokens.astype(np.float32)).all() or not np.isfinite(vectors.astype(np.float32)).all():
            raise ValueError(f"chunk {chunk_number}: non-finite tokens or vectors")
        if (masks.sum(axis=1) < 1).any():
            raise ValueError(f"chunk {chunk_number}: a text has zero valid tokens")

        sha = text_token_chunk_sha256(tokens, masks, vectors, target_ids)
        if not _valid_existing_chunk(npz_path, meta_path, sha):
            atomic_npz(npz_path, tokens=tokens, masks=masks, vectors=vectors)
            atomic_json(meta_path, {
                "chunk_number": chunk_number,
                "rows": len(batch),
                "offset": start,
                "token_len": token_len,
                "token_dim": TOKEN_DIM,
                "dtype": dtype,
                "checkpoint_sha256": checkpoint_sha256,
                "source_index_sha256": source_index_sha,
                "text_model_id": text_model_id,
                "text_target_ids": target_ids,
                "representative_trial_ids": [r["representative_trial_id"] for r in batch],
                "sha256": sha,
            })
        else:
            reused += 1
        chunk_entries.append({
            "chunk_number": chunk_number,
            "rows": len(batch),
            "offset": start,
            "token_file": f"tokens/text_{chunk_number:05d}.npz",
            "sha256": sha,
        })
        for offset, record in enumerate(batch):
            index_rows.append({
                "text_target_id": record["text_target_id"],
                "normalized_text_sha256": record["normalized_text_sha256"],
                "representative_trial_id": record["representative_trial_id"],
                "token_file": f"tokens/text_{chunk_number:05d}.npz",
                "token_offset": offset,
                "token_len": token_len,
                "token_dim": TOKEN_DIM,
                "checkpoint_sha256": checkpoint_sha256,
                "source_index_sha256": source_index_sha,
            })

    index_path = output_root / "text_token_index.csv"
    mapping_path = output_root / "trial_text_targets.csv"
    write_csv(index_path, INDEX_FIELDS, index_rows)
    write_csv(mapping_path, MAPPING_FIELDS, mapping)
    if {m["text_target_id"] for m in mapping} - {r["text_target_id"] for r in index_rows}:
        raise AssertionError("trial mapping references a missing text token identity")

    combined = hashlib.sha256()
    for entry in chunk_entries:
        combined.update(f"{entry['chunk_number']}\x1f{entry['sha256']}\n".encode("utf-8"))
    manifest = {
        "status": "pass",
        "schema_version": 1,
        "run_mode": run_mode,
        "source_index_sha256": source_index_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "glim_commit": glim_commit,
        "text_model_id": text_model_id,
        "token_len": token_len,
        "token_dim": TOKEN_DIM,
        "dtype": dtype,
        "vector_dtype": "float32",
        "unique_text_identities": len(index_rows),
        "mapped_trials": len(mapping),
        "num_chunks": len(chunk_entries),
        "split_counts": dict(sorted(Counter(m["split"] for m in mapping).items())),
        "text_token_index_sha256": sha256(index_path),
        "trial_text_targets_sha256": sha256(mapping_path),
        "combined_chunk_sha256": combined.hexdigest(),
        "chunks": chunk_entries,
        "checks": {"held_out_test_accessed": False, "all_mapped_trials_resolve": True},
    }
    manifest_path = output_root / "text_token_manifest.json"
    atomic_json(manifest_path, manifest)
    manifest["chunks_reused_this_invocation"] = reused
    return manifest


class GLIMTextTokenEmbedder:
    """Real GLIM text path returning (tokens [B,L,e], mask [B,L], vectors [B,e]).

    Mirrors ``GLIMTextEmbedder`` but also returns the unpooled ``encode_text``
    hidden states and their mask. Not unit-tested locally (needs the model +
    checkpoint); exercised on Kaggle.
    """

    def __init__(self, glim_root: Path, checkpoint: Path, device: str, text_batch_size: int = 64):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        from project_adapters.glim_representation import load_upstream_glim_class

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA text extraction requested but unavailable")
            torch.backends.cuda.matmul.allow_tf32 = False
        GLIM = load_upstream_glim_class(glim_root.resolve())
        model = GLIM.load_from_checkpoint(str(checkpoint), map_location="cpu", strict=False)
        model.eval().to(self.device)
        # Match the frozen text-VECTOR extractor's precision path exactly: fp16 T5 +
        # fp16 autocast on CUDA. FLAN-T5-large is numerically unstable in fp16, so the
        # emitted hidden states differ materially from fp32; to keep these unpooled
        # tokens the true counterpart of the established pooled text vectors (and the
        # 0.32 anchor), the token path must reproduce that same fp16 computation.
        self.text_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model.tokenizer = AutoTokenizer.from_pretrained(model.text_model_id)
        model.text_model = T5ForConditionalGeneration.from_pretrained(
            model.text_model_id, torch_dtype=self.text_dtype
        ).requires_grad_(False).eval().to(self.device)
        model.eval()
        self.model = model
        self.token_len = int(model.input_text_len - model.prompt_tuning_len)
        self.vector_dim = int(model.embed_dim)
        self.text_model_id = str(model.text_model_id)
        self.text_batch_size = int(text_batch_size)

    def _forward(self, texts: list[str]):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            ids, mask = self.model.tokenize(texts, self.token_len)
            hidden, hidden_mask = self.model.encode_text(ids, mask)
            vectors = self.model.aligner.embed_text(hidden, hidden_mask)
            if vectors.ndim == 1:
                vectors = vectors.unsqueeze(0)
        return (hidden.detach().float().cpu().numpy(),
                hidden_mask.detach().to(torch.int8).cpu().numpy(),
                vectors.detach().float().cpu().numpy())

    def __call__(self, texts: list[str]):
        # Sub-batch the T5 forward at ``text_batch_size`` to match the frozen text-vector
        # extractor's batch construction (fp16 GEMM kernel selection is shape-dependent,
        # so batch size can perturb outputs at the fp16 level).
        bs = self.text_batch_size
        tok_parts, mask_parts, vec_parts = [], [], []
        for i in range(0, len(texts), bs):
            t, m, v = self._forward(texts[i:i + bs])
            tok_parts.append(t); mask_parts.append(m); vec_parts.append(v)
        tokens = np.concatenate(tok_parts, axis=0)
        masks = np.concatenate(mask_parts, axis=0)
        vecs = np.concatenate(vec_parts, axis=0)
        if tokens.shape != (len(texts), self.token_len, self.vector_dim):
            raise ValueError(f"GLIM returned text tokens {tokens.shape}")
        return tokens, masks, vecs


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
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=64,
                        help="T5 forward batch size; match the frozen extractor (64)")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()

    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("GLIM checkpoint SHA256 mismatch")

    rows, source_manifest = load_rows(args.dataset_root, args.expected_index_sha256)
    records, mapping = build_text_records(rows)
    run_mode = "full_development"
    if args.smoke_limit is not None:
        if args.smoke_limit <= 0:
            raise ValueError("smoke_limit must be positive")
        records = records[: args.smoke_limit]
        allowed = {r["text_target_id"] for r in records}
        mapping = [m for m in mapping if m["text_target_id"] in allowed]
        run_mode = "smoke"

    embedder = GLIMTextTokenEmbedder(
        args.glim_root, args.checkpoint, args.device, text_batch_size=args.text_batch_size)
    manifest = run_text_token_extraction(
        records, mapping, args.output_root, embedder,
        checkpoint_sha256=checkpoint_sha, glim_commit=args.glim_commit,
        text_model_id=embedder.text_model_id,
        source_index_sha256=str(source_manifest["index_sha256"]),
        token_len=embedder.token_len, dtype=args.dtype, chunk_size=args.chunk_size,
        run_mode=run_mode,
    )
    print(f"TEXT TOKEN EXTRACTION: {manifest['unique_text_identities']} identities, "
          f"{manifest['num_chunks']} chunks, token_len={manifest['token_len']}, "
          f"combined_chunk_sha256={manifest['combined_chunk_sha256']}")
    print(json.dumps({k: manifest[k] for k in
                      ("run_mode", "unique_text_identities", "mapped_trials", "split_counts")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
