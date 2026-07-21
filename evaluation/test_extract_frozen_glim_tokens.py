"""Local tests for the GLIM token extraction driver (no GPU, fake embedder).

Covers the parts where correctness matters and a GPU cannot help: chunk
shaping, per-chunk hashing, identity binding, the index, resume idempotence,
and rejection of malformed embedder output.

Run: python -B -m unittest evaluation.test_extract_frozen_glim_tokens
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.extract_frozen_glim_tokens import (
    TOKEN_COUNT,
    TOKEN_DIM,
    run_token_extraction,
    select_primary_cohort,
    subbatched_embed,
    token_chunk_sha256,
)

CHECKPOINT_SHA = "a" * 64
GLIM_COMMIT = "b" * 40


def make_rows(n):
    return [
        {
            "trial_id": f"trial{i:04d}",
            "sample_id": f"ZuCo2::task2::S{i%3}::row{i:06d}",
            "source_dataframe_row_index": i,
            "reading_task": "NR" if i % 2 == 0 else "TSR",
            "shard": "s0.npz",
            "offset": i,
        }
        for i in range(n)
    ]


def fake_load_eeg(row):
    # deterministic, shape-valid EEG/mask; contents irrelevant to the fake embedder
    t = 40
    return np.zeros((t, 8), dtype=np.float32), np.ones((t,), dtype=np.int8)


def fake_embed(eeg_list, mask_list, sample_ids, source_rows, tasks):
    """Deterministic (tokens, vectors) from sample_id so hashes are reproducible."""
    tokens = np.zeros((len(sample_ids), TOKEN_COUNT, TOKEN_DIM), dtype=np.float32)
    vectors = np.zeros((len(sample_ids), TOKEN_DIM), dtype=np.float32)
    for i, sid in enumerate(sample_ids):
        rng = np.random.default_rng(abs(hash(sid)) % (2**31))
        tokens[i] = rng.standard_normal((TOKEN_COUNT, TOKEN_DIM), dtype=np.float32)
        vectors[i] = rng.standard_normal(TOKEN_DIM, dtype=np.float32)
    return tokens, vectors


class TokenExtractionTests(unittest.TestCase):
    def _run(self, tmp, rows, chunk_size=4, dtype="float16"):
        return run_token_extraction(
            rows, fake_embed, Path(tmp),
            checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
            prompt_mode="all_masked", load_eeg=fake_load_eeg,
            chunk_size=chunk_size, dtype=dtype,
        )

    def test_index_and_chunk_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = make_rows(10)
            index = self._run(tmp, rows, chunk_size=4)
            self.assertEqual(index["total_rows"], 10)
            self.assertEqual(index["num_chunks"], 3)          # 4 + 4 + 2
            self.assertEqual(index["token_shape"], [TOKEN_COUNT, TOKEN_DIM])
            self.assertEqual(index["dtype"], "float16")
            written = json.loads((Path(tmp) / "token_index.json").read_text())
            self.assertEqual(written["combined_chunk_sha256"], index["combined_chunk_sha256"])
            with np.load(Path(tmp) / "tokens" / "tokens_00002.npz") as arch:
                self.assertEqual(set(arch.files), {"tokens", "vectors"})
                self.assertEqual(arch["tokens"].shape, (2, TOKEN_COUNT, TOKEN_DIM))
                self.assertEqual(arch["tokens"].dtype, np.float16)
                self.assertEqual(arch["vectors"].shape, (2, TOKEN_DIM))

    def test_identity_binding_recorded_per_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = make_rows(6)
            self._run(tmp, rows, chunk_size=4)
            meta = json.loads((Path(tmp) / "tokens" / "tokens_00000.json").read_text())
            self.assertEqual(meta["trial_ids"], [r["trial_id"] for r in rows[:4]])
            self.assertEqual(meta["sample_ids"], [r["sample_id"] for r in rows[:4]])
            self.assertEqual(meta["source_dataframe_row_indices"], [0, 1, 2, 3])
            self.assertEqual(meta["reading_tasks"], ["NR", "TSR", "NR", "TSR"])
            self.assertEqual(meta["checkpoint_sha256"], CHECKPOINT_SHA)
            self.assertEqual(meta["prompt_mode"], "all_masked")

    def test_deterministic_and_resume_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = make_rows(8)
            first = self._run(tmp, rows, chunk_size=4)
            mtimes = {
                p.name: p.stat().st_mtime_ns
                for p in (Path(tmp) / "tokens").glob("*.npz")
            }
            second = self._run(tmp, rows, chunk_size=4)   # resume: nothing recomputed
            self.assertEqual(first["combined_chunk_sha256"], second["combined_chunk_sha256"])
            for p in (Path(tmp) / "tokens").glob("*.npz"):
                self.assertEqual(mtimes[p.name], p.stat().st_mtime_ns, f"{p.name} rewritten")

    def test_hash_changes_when_identity_changes(self):
        tokens = np.zeros((2, TOKEN_COUNT, TOKEN_DIM), dtype=np.float16)
        vectors = np.zeros((2, TOKEN_DIM), dtype=np.float16)
        base = token_chunk_sha256(tokens, vectors, ["t0", "t1"], ["s0", "s1"], [0, 1])
        moved = token_chunk_sha256(tokens, vectors, ["t0", "t1"], ["s0", "s1"], [0, 2])
        relabel = token_chunk_sha256(tokens, vectors, ["t0", "tX"], ["s0", "s1"], [0, 1])
        vecdiff = token_chunk_sha256(tokens, vectors + 1, ["t0", "t1"], ["s0", "s1"], [0, 1])
        self.assertNotEqual(base, moved)
        self.assertNotEqual(base, relabel)
        self.assertNotEqual(base, vecdiff)

    def test_rejects_wrong_token_shape(self):
        def bad_embed(eeg_list, mask_list, sample_ids, source_rows, tasks):
            n = len(sample_ids)
            return np.zeros((n, TOKEN_COUNT, 512), dtype=np.float32), np.zeros((n, TOKEN_DIM), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_token_extraction(
                    make_rows(2), bad_embed, Path(tmp),
                    checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
                    prompt_mode="all_masked", load_eeg=fake_load_eeg, chunk_size=2,
                )

    def test_rejects_non_finite_tokens(self):
        def nan_embed(eeg_list, mask_list, sample_ids, source_rows, tasks):
            n = len(sample_ids)
            out = np.zeros((n, TOKEN_COUNT, TOKEN_DIM), dtype=np.float32)
            out[0, 0, 0] = np.nan
            return out, np.zeros((n, TOKEN_DIM), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_token_extraction(
                    make_rows(2), nan_embed, Path(tmp),
                    checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
                    prompt_mode="all_masked", load_eeg=fake_load_eeg, chunk_size=2,
                )

    def test_select_primary_cohort(self):
        rows = [
            {"dataset_version": "ZuCo2", "reading_task": "NR", "source_dataframe_row_index": 5},
            {"dataset_version": "ZuCo2", "reading_task": "TSR", "source_dataframe_row_index": 1},
            {"dataset_version": "ZuCo2", "reading_task": "SR", "source_dataframe_row_index": 2},  # drop SR
            {"dataset_version": "ZuCo1", "reading_task": "NR", "source_dataframe_row_index": 3},   # drop ZuCo1
            {"dataset_version": "ZuCo2", "reading_task": "NR", "source_dataframe_row_index": 0},
        ]
        cohort = select_primary_cohort(rows)
        self.assertEqual([r["source_dataframe_row_index"] for r in cohort], [0, 1, 5])  # sorted, filtered
        self.assertTrue(all(r["dataset_version"] == "ZuCo2" for r in cohort))
        self.assertTrue(all(r["reading_task"] in {"NR", "TSR"} for r in cohort))

    def test_subbatched_embed_concatenates_both_arrays_in_order(self):
        # Regression: main()'s wrapper must return (tokens, vectors) — not a
        # concatenation of tuples — when a chunk spans multiple GPU sub-batches.
        n = 10  # > batch_size, so the sub-batching loop runs several times

        def embedder(eeg_list, mask_list, sample_ids, source_rows, tasks):
            b = len(sample_ids)
            tok = np.stack([np.full((TOKEN_COUNT, TOKEN_DIM), r, np.float32) for r in source_rows])
            vec = np.stack([np.full((TOKEN_DIM,), r, np.float32) for r in source_rows])
            return tok, vec

        rows = make_rows(n)
        tokens, vectors = subbatched_embed(
            embedder, [fake_load_eeg(r)[0] for r in rows], [fake_load_eeg(r)[1] for r in rows],
            [r["sample_id"] for r in rows], [r["source_dataframe_row_index"] for r in rows],
            [r["reading_task"] for r in rows], batch_size=4,
        )
        self.assertEqual(tokens.shape, (n, TOKEN_COUNT, TOKEN_DIM))
        self.assertEqual(vectors.shape, (n, TOKEN_DIM))
        # order preserved across sub-batch boundaries: row i is tagged with value i
        self.assertTrue(np.array_equal(tokens[:, 0, 0], np.arange(n, dtype=np.float32)))
        self.assertTrue(np.array_equal(vectors[:, 0], np.arange(n, dtype=np.float32)))

    def test_rejects_bad_prompt_mode_and_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_token_extraction(
                    make_rows(2), fake_embed, Path(tmp),
                    checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
                    prompt_mode="nonsense", load_eeg=fake_load_eeg,
                )
            with self.assertRaises(ValueError):
                run_token_extraction(
                    make_rows(2), fake_embed, Path(tmp),
                    checkpoint_sha256=CHECKPOINT_SHA, glim_commit=GLIM_COMMIT,
                    prompt_mode="all_masked", load_eeg=fake_load_eeg, dtype="int8",
                )


if __name__ == "__main__":
    unittest.main()
