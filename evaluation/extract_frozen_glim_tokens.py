"""Extract and hash-bind GLIM's unpooled EEG token sequences [96, 1024].

Companion to ``extract_frozen_glim_vectors.py`` (which stores only the pooled
[1024] ``eeg_vector`` and is left untouched). This extractor stores the
``eeg_tokens`` [96, 1024] sequence that the same GLIM forward pass already
produces, for the token-level late-interaction retrieval experiment.

One forward pass per trial produces BOTH the 96 unpooled tokens and GLIM's
learned pooled vector; both are stored (tokens feed the MaxSim arm, the pooled
vector feeds the pooled-cosine arm of the primary pair, and having both from the
identical pass guarantees they are consistent). Only correct EEG is stored; the
matched-wrong-EEG control is realised at eval time from a donor trial's stored
tokens. Stored as float16 (MaxSim/cosine L2-normalize, so half precision is ample
and halves the footprint).

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
    tokens: np.ndarray, vectors: np.ndarray, trial_ids: Sequence[str],
    sample_ids: Sequence[str], source_rows: Sequence[int],
) -> str:
    state = hashlib.sha256()
    state.update(np.ascontiguousarray(tokens).tobytes())
    state.update(np.ascontiguousarray(vectors).tobytes())
    state.update(f"{tokens.dtype}\x1f{tokens.shape}\x1f{vectors.shape}\n".encode("utf-8"))
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
        tokens, vectors = embed(eeg_list, mask_list, sample_ids, source_rows, tasks)
        tokens = np.ascontiguousarray(tokens).astype(np_dtype, copy=False)
        vectors = np.ascontiguousarray(vectors).astype(np_dtype, copy=False)
        if tokens.shape != (len(batch), TOKEN_COUNT, TOKEN_DIM):
            raise ValueError(
                f"chunk {chunk_number}: expected {(len(batch), TOKEN_COUNT, TOKEN_DIM)}, got {tokens.shape}"
            )
        if vectors.shape != (len(batch), TOKEN_DIM):
            raise ValueError(
                f"chunk {chunk_number}: expected vectors {(len(batch), TOKEN_DIM)}, got {vectors.shape}"
            )
        if not np.isfinite(tokens.astype(np.float32)).all() or not np.isfinite(vectors.astype(np.float32)).all():
            raise ValueError(f"chunk {chunk_number}: non-finite tokens or vectors")

        sha = token_chunk_sha256(tokens, vectors, trial_ids, sample_ids, source_rows)
        entry = {
            "chunk_number": chunk_number,
            "rows": len(batch),
            "offset": start,
            "token_file": f"tokens/tokens_{chunk_number:05d}.npz",
            "sha256": sha,
        }

        if not _valid_existing_chunk(npz_path, meta_path, sha):
            atomic_npz(npz_path, tokens=tokens, vectors=vectors)
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

    def __call__(self, eeg_list, mask_list, sample_ids, source_rows, tasks):
        """Return (tokens [B,96,1024], vectors [B,1024]) from one forward pass."""
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
            vectors = output["eeg_vector"].detach().float().cpu().numpy()
        if tokens.shape != (batch, TOKEN_COUNT, TOKEN_DIM):
            raise ValueError(f"GLIM returned tokens {tokens.shape}, expected {(batch, TOKEN_COUNT, TOKEN_DIM)}")
        if vectors.shape != (batch, TOKEN_DIM):
            raise ValueError(f"GLIM returned vectors {vectors.shape}, expected {(batch, TOKEN_DIM)}")
        return tokens, vectors


def select_primary_cohort(train_rows: list[dict]) -> list[dict]:
    """Filter canonical train rows to the P4b primary_zuco2_nr_tsr cohort."""
    cohort = [
        r for r in train_rows
        if str(r["dataset_version"]) == "ZuCo2" and str(r["reading_task"]) in {"NR", "TSR"}
    ]
    cohort.sort(key=lambda r: int(r["source_dataframe_row_index"]))
    return cohort


def main() -> None:
    import argparse
    import json as _json

    import torch

    from evaluation.extract_frozen_glim_vectors import (
        ShardSignalStore,
        load_development_rows,
        sha256,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--glim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--glim-commit", required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()

    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("GLIM checkpoint SHA256 mismatch")

    train, _validation, _manifest = load_development_rows(
        args.dataset_root, args.expected_index_sha256
    )
    cohort = select_primary_cohort(train)
    nr = sum(1 for r in cohort if str(r["reading_task"]) == "NR")
    tsr = sum(1 for r in cohort if str(r["reading_task"]) == "TSR")
    print({"primary_cohort_rows": len(cohort), "NR": nr, "TSR": tsr})
    # Expected P4b cohort: 9,011 = 4,126 NR + 4,885 TSR. Assert unless smoke-limited.
    if args.smoke_limit is None and (len(cohort), nr, tsr) != (9011, 4126, 4885):
        raise ValueError(f"cohort != frozen P4b inventory: {(len(cohort), nr, tsr)}")
    if args.smoke_limit is not None:
        cohort = cohort[: args.smoke_limit]

    store = ShardSignalStore(args.dataset_root)
    embedder = GLIMTokenEmbedder(
        args.glim_root, args.checkpoint, args.device, prompt_mode="all_masked"
    )

    def embed(eeg_list, mask_list, sample_ids, source_rows, tasks):
        parts = []
        for i in range(0, len(eeg_list), args.batch_size):
            j = i + args.batch_size
            parts.append(embedder(
                eeg_list[i:j], mask_list[i:j], sample_ids[i:j], source_rows[i:j], tasks[i:j]
            ))
        return np.concatenate(parts, axis=0)

    index = run_token_extraction(
        cohort, embed, args.output_root,
        checkpoint_sha256=checkpoint_sha, glim_commit=args.glim_commit,
        prompt_mode="all_masked", load_eeg=store.load,
        chunk_size=args.chunk_size, dtype=args.dtype,
    )
    store.close()
    print(f"TOKEN EXTRACTION: {index['total_rows']} rows, {index['num_chunks']} chunks, "
          f"combined_chunk_sha256={index['combined_chunk_sha256']}")

    # Gate-1 diagnostic over the FULL cohort, streamed chunk-by-chunk (memory-safe:
    # per-trial redundancy and effective rank are cheap scalars; we never hold all
    # 9,011 x 96 x 1024 tokens at once).
    from evaluation.token_diagnostics import (
        effective_rank, position_liveness, verdict_from_stats, within_trial_redundancy,
    )
    red_sum = eff_sum = norm_sum = count = 0.0
    norm_min, norm_max, red_max = float("inf"), 0.0, 0.0
    finite = True
    live_sample = []
    for entry in index["chunks"]:
        with np.load(args.output_root / entry["token_file"]) as _arch:
            arr = _arch["tokens"].astype(np.float32)
        t = torch.from_numpy(arr)
        if not bool(torch.isfinite(t).all()):
            finite = False
        norms = t.norm(dim=-1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum()); count += norms.numel()
        red = within_trial_redundancy(t)
        red_sum += float(red.sum()); red_max = max(red_max, float(red.max()))
        eff_sum += float(effective_rank(t).sum())
        if len(live_sample) < 2048:
            live_sample.append(arr)
    trials = index["total_rows"]
    mean_red = red_sum / trials
    mean_eff = eff_sum / trials
    live = position_liveness(torch.from_numpy(np.concatenate(live_sample, axis=0)[:2048]))
    report = {
        "cohort_trials": trials, "tokens_per_trial": TOKEN_COUNT, "dim": TOKEN_DIM,
        "finite": finite,
        "token_norm_mean": norm_sum / count, "token_norm_min": norm_min, "token_norm_max": norm_max,
        "within_trial_redundancy_mean": mean_red, "within_trial_redundancy_max": red_max,
        "effective_rank_mean": mean_eff, "effective_rank_fraction_of_T": mean_eff / TOKEN_COUNT,
        **{f"{k}_sample2048": v for k, v in live.items()},
        "verdict": verdict_from_stats(mean_red, mean_eff, TOKEN_COUNT, finite, norm_min),
    }
    with open(args.output_root / "gate1_token_diagnostic.json", "w", encoding="utf-8") as handle:
        _json.dump(report, handle, indent=2, sort_keys=True); handle.write("\n")
    print("GATE-1 TOKEN DIAGNOSTIC (full cohort):")
    print(_json.dumps(report, indent=2, sort_keys=True))
    print("VERDICT:", report["verdict"])


if __name__ == "__main__":
    main()
