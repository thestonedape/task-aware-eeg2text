"""Local tests for the GLIM text-token extraction driver (no GPU, fake embedder).

Covers what a GPU cannot help with: chunk shaping, per-chunk hashing over
tokens+masks+vectors+identity, the index + trial mapping, resume idempotence,
and rejection of malformed embedder output.

Run: python -B -m unittest evaluation.test_extract_frozen_glim_text_tokens
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.extract_frozen_glim_text_tokens import (
    TOKEN_DIM,
    run_text_token_extraction,
    text_token_chunk_sha256,
)

CHECKPOINT_SHA = "a" * 64
GLIM_COMMIT = "b" * 40
SOURCE_INDEX_SHA = "c" * 64
LEN = 12


def make_records(n):
    return [
        {
            "text_target_id": f"{i:064d}",
            "normalized_text_sha256": f"{i:064d}",
            "representative_trial_id": f"trial{i:04d}",
            "representative_text": f"sentence number {i}",
        }
        for i in range(n)
    ]


def make_mapping(records):
    return [
        {
            "trial_id": r["representative_trial_id"], "split": "train" if i % 2 else "val",
            "cohort": "primary", "dataset_version": "ZuCo2", "reading_task": "NR",
            "subject_id": "S1", "source_dataframe_row_index": str(i),
            "text_target_id": r["text_target_id"], "normalized_text_sha256": r["normalized_text_sha256"],
        }
        for i, r in enumerate(records)
    ]


def fake_embed(texts):
    """Deterministic (tokens, masks, vectors) from text so hashes reproduce."""
    n = len(texts)
    tokens = np.zeros((n, LEN, TOKEN_DIM), dtype=np.float32)
    masks = np.ones((n, LEN), dtype=np.int8)
    vectors = np.zeros((n, TOKEN_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        tokens[i] = rng.standard_normal((LEN, TOKEN_DIM), dtype=np.float32)
        vectors[i] = rng.standard_normal(TOKEN_DIM, dtype=np.float32)
        masks[i, LEN - (i % 3):] = 0  # some padding, at least one valid token
    return tokens, masks, vectors


class TextTokenExtractionTests(unittest.TestCase):
    def _run(self, tmp, records, mapping, chunk_size=4, dtype="float16"):
        return run_text_token_extraction(
            records, mapping, Path(tmp), fake_embed,
            checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
            text_model_id="google/flan-t5-large", source_index_sha256=SOURCE_INDEX_SHA,
            token_len=LEN, dtype=dtype, chunk_size=chunk_size,
        )

    def test_index_shapes_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = make_records(10)
            manifest = self._run(tmp, records, make_mapping(records), chunk_size=4)
            self.assertEqual(manifest["unique_text_identities"], 10)
            self.assertEqual(manifest["num_chunks"], 3)  # 4 + 4 + 2
            self.assertEqual(manifest["token_len"], LEN)
            self.assertEqual(manifest["mapped_trials"], 10)
            with np.load(Path(tmp) / "tokens" / "text_00002.npz") as arch:
                self.assertEqual(set(arch.files), {"tokens", "masks", "vectors"})
                self.assertEqual(arch["tokens"].shape, (2, LEN, TOKEN_DIM))
                self.assertEqual(arch["tokens"].dtype, np.float16)
                self.assertEqual(arch["masks"].shape, (2, LEN))
                self.assertEqual(arch["vectors"].shape, (2, TOKEN_DIM))
                self.assertEqual(arch["vectors"].dtype, np.float32)  # pooled vector kept fp32
            index = (Path(tmp) / "text_token_index.csv").read_text(encoding="utf-8")
            self.assertEqual(index.count("\n"), 11)  # header + 10 rows

    def test_hash_binds_tokens_masks_vectors_and_identity(self):
        toks = np.zeros((2, LEN, TOKEN_DIM), dtype=np.float16)
        masks = np.ones((2, LEN), dtype=np.int8)
        vecs = np.zeros((2, TOKEN_DIM), dtype=np.float16)
        base = text_token_chunk_sha256(toks, masks, vecs, ["t0", "t1"])
        self.assertNotEqual(base, text_token_chunk_sha256(toks + 1, masks, vecs, ["t0", "t1"]))
        m2 = masks.copy(); m2[0, 0] = 0
        self.assertNotEqual(base, text_token_chunk_sha256(toks, m2, vecs, ["t0", "t1"]))
        self.assertNotEqual(base, text_token_chunk_sha256(toks, masks, vecs + 1, ["t0", "t1"]))
        self.assertNotEqual(base, text_token_chunk_sha256(toks, masks, vecs, ["t0", "tX"]))

    def test_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = make_records(8)
            mapping = make_mapping(records)
            first = self._run(tmp, records, mapping, chunk_size=4)
            mtimes = {p.name: p.stat().st_mtime_ns for p in (Path(tmp) / "tokens").glob("*.npz")}
            second = self._run(tmp, records, mapping, chunk_size=4)
            self.assertEqual(first["combined_chunk_sha256"], second["combined_chunk_sha256"])
            self.assertEqual(second["chunks_reused_this_invocation"], first["num_chunks"])
            for p in (Path(tmp) / "tokens").glob("*.npz"):
                self.assertEqual(mtimes[p.name], p.stat().st_mtime_ns, f"{p.name} rewritten")

    def test_rejects_wrong_token_shape(self):
        def bad_embed(texts):
            n = len(texts)
            return (np.zeros((n, LEN, 512), np.float32), np.ones((n, LEN), np.int8),
                    np.zeros((n, TOKEN_DIM), np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            records = make_records(2)
            with self.assertRaises(ValueError):
                run_text_token_extraction(
                    records, make_mapping(records), Path(tmp), bad_embed,
                    checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
                    text_model_id="m", source_index_sha256=SOURCE_INDEX_SHA, token_len=LEN,
                )

    def test_rejects_zero_valid_tokens(self):
        def dead_embed(texts):
            n = len(texts)
            return (np.zeros((n, LEN, TOKEN_DIM), np.float32), np.zeros((n, LEN), np.int8),
                    np.zeros((n, TOKEN_DIM), np.float32))

        with tempfile.TemporaryDirectory() as tmp:
            records = make_records(2)
            with self.assertRaises(ValueError):
                run_text_token_extraction(
                    records, make_mapping(records), Path(tmp), dead_embed,
                    checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
                    text_model_id="m", source_index_sha256=SOURCE_INDEX_SHA, token_len=LEN,
                )

    def test_mapping_must_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = make_records(4)
            mapping = make_mapping(records)
            mapping[0]["text_target_id"] = "d" * 64  # dangling reference
            with self.assertRaises(AssertionError):
                self._run(tmp, records, mapping, chunk_size=4)


if __name__ == "__main__":
    unittest.main()
