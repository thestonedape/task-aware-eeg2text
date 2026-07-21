"""Gate-2 identity check: do co-extracted pooled vectors match the frozen ones?

The token extractors co-store GLIM's pooled vector alongside the unpooled tokens,
from the *same* forward pass. This module confirms those pooled vectors reproduce
the separately frozen pooled vectors the pooled pipeline already uses -- proving
the token arm and the pooled arm share one representation, which the primary
causal pair depends on. Frozen vectors were stored float32; the co-extracted ones
are float16, so the comparison uses a half-precision tolerance, not bit-equality.

Pure array/dict logic; unit-tested with synthetic vectors, no GPU and no I/O.
"""

from __future__ import annotations

import numpy as np


def align_by_id(
    ids_a: list[str], vectors_a: np.ndarray, ids_b: list[str], vectors_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (A_rows, B_rows, shared_ids) row-aligned on the intersection of ids.

    Raises if either side has duplicate ids or the intersection is empty.
    """
    if vectors_a.shape[0] != len(ids_a) or vectors_b.shape[0] != len(ids_b):
        raise ValueError("id/vector row-count mismatch")
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise ValueError("duplicate ids on one side")
    index_b = {tid: i for i, tid in enumerate(ids_b)}
    shared = [tid for tid in ids_a if tid in index_b]
    if not shared:
        raise ValueError("no shared ids between the two vector sets")
    rows_a = np.stack([vectors_a[i] for i, tid in enumerate(ids_a) if tid in index_b])
    rows_b = np.stack([vectors_b[index_b[tid]] for tid in shared])
    return rows_a.astype(np.float64), rows_b.astype(np.float64), shared


def compare_pooled_vectors(
    ids_a: list[str], vectors_a: np.ndarray, ids_b: list[str], vectors_b: np.ndarray,
    *, atol: float = 2e-2, rtol: float = 2e-3, min_cosine: float = 0.9995,
) -> dict:
    """Row-align on shared ids and quantify agreement.

    Defaults are float16-appropriate: a co-extracted float16 vector vs a frozen
    float32 vector agrees to ~1e-3 relative. Reports max abs/rel diff, the worst
    per-row cosine, and a boolean ``match`` (allclose AND worst cosine >= min).
    """
    a, b, shared = align_by_id(ids_a, vectors_a, ids_b, vectors_b)
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-8)
    rel_diff = abs_diff / denom
    an = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    per_row_cosine = (an * bn).sum(axis=1)
    allclose = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    worst_cosine = float(per_row_cosine.min())
    return {
        "shared_count": len(shared),
        "compared_count_a": len(ids_a),
        "compared_count_b": len(ids_b),
        "max_abs_diff": float(abs_diff.max()),
        "max_rel_diff": float(rel_diff.max()),
        "min_per_row_cosine": worst_cosine,
        "mean_per_row_cosine": float(per_row_cosine.mean()),
        "allclose": allclose,
        "atol": atol,
        "rtol": rtol,
        "min_cosine": min_cosine,
        "match": bool(allclose and worst_cosine >= min_cosine),
    }
