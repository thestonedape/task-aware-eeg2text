"""Clustered paired bootstrap for the token study's decision statistics.

Mirrors the P4b implementation (``run_task_treatment_pilots.paired_bootstrap`` /
``run_frozen_factor_probes.paired_bootstrap``) exactly, so the pooled-vs-token
Δ inherits the same clustered-inference method the established results used:

* ``subject_id`` and ``normalized_text_sha256`` one-way cluster resampling;
* ``two_way_subject_by_text`` = the product of subject and text resample weights
  (Cameron–Gelbach–Miller multiplicative two-way clustering);
* task-macro estimates (NR and TSR estimated separately, then averaged), matching
  the paper's macro-MRR endpoint.

Pure numpy; no torch, no I/O; unit-tested for reproducibility.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

CLUSTERS = ("subject_id", "normalized_text_sha256", "two_way_subject_by_text")


def bootstrap_weights(labels: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    """Cluster resample weights: draw len(unique) clusters with replacement."""
    unique, inverse = np.unique(np.asarray(labels), return_inverse=True)
    counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
    return counts[inverse].astype(np.float64)


def paired_bootstrap(
    effects: np.ndarray,
    subjects: Sequence[str],
    texts: Sequence[str],
    cluster: str,
    replicates: int,
    seed: int,
    tasks: Sequence[str] | None = None,
) -> np.ndarray:
    """Clustered bootstrap distribution of the (task-macro) mean of ``effects``.

    ``effects`` are per-trial paired quantities (e.g. RR_maxsim - RR_pooled).
    With ``tasks`` given, each replicate averages the NR and TSR weighted means
    (task-macro); without, it is a single weighted mean.
    """
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {CLUSTERS}")
    effects = np.asarray(effects, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(int(replicates), dtype=np.float64)
    task_array = np.asarray(tasks) if tasks is not None else None
    if task_array is not None and set(task_array.tolist()) != {"NR", "TSR"}:
        raise ValueError("task-macro bootstrap requires NR and TSR")
    for index in range(int(replicates)):
        for _attempt in range(1000):
            if cluster == "subject_id":
                weights = bootstrap_weights(subjects, rng)
            elif cluster == "normalized_text_sha256":
                weights = bootstrap_weights(texts, rng)
            else:  # two_way_subject_by_text
                weights = bootstrap_weights(subjects, rng) * bootstrap_weights(texts, rng)
            if task_array is None:
                denom = weights.sum()
                if denom > 0:
                    out[index] = float(np.dot(weights, effects) / denom)
                    break
                continue
            estimates, valid = [], True
            for task in ("NR", "TSR"):
                mask = task_array == task
                denom = weights[mask].sum()
                if denom <= 0:
                    valid = False
                    break
                estimates.append(float(np.dot(weights[mask], effects[mask]) / denom))
            if valid:
                out[index] = float(np.mean(estimates))
                break
        else:
            raise RuntimeError(f"unable to draw a nonempty {cluster} bootstrap replicate")
    return out


def task_macro_mean(effects: np.ndarray, tasks: Sequence[str]) -> float:
    """Observed task-macro mean (NR and TSR means averaged) — the point estimate."""
    effects = np.asarray(effects, dtype=np.float64)
    task_array = np.asarray(tasks)
    per_task = []
    for task in ("NR", "TSR"):
        mask = task_array == task
        if mask.sum() == 0:
            raise ValueError(f"no trials for task {task}")
        per_task.append(float(effects[mask].mean()))
    return float(np.mean(per_task))


def lower_bound(distribution: np.ndarray, alpha: float = 0.05) -> float:
    """One-sided lower confidence bound = the alpha percentile of the distribution."""
    return float(np.percentile(np.asarray(distribution, dtype=np.float64), 100.0 * alpha))


def clustered_delta(
    effects: np.ndarray,
    subjects: Sequence[str],
    texts: Sequence[str],
    tasks: Sequence[str],
    *,
    replicates: int = 5000,
    seed: int = 20260722,
    alpha: float = 0.05,
) -> dict:
    """Point estimate + one-sided lower bounds of a paired task-macro effect under
    all three cluster schemes. The primary decision uses the two-way lower bound."""
    point = task_macro_mean(effects, tasks)
    bounds = {}
    for cluster in CLUSTERS:
        draws = paired_bootstrap(effects, subjects, texts, cluster, replicates, seed, tasks)
        bounds[cluster] = lower_bound(draws, alpha)
    return {
        "point": point,
        "lower_bounds": bounds,
        "two_way_lower_bound": bounds["two_way_subject_by_text"],
        "replicates": int(replicates),
        "seed": int(seed),
        "alpha": alpha,
    }


__all__ = [
    "CLUSTERS", "bootstrap_weights", "paired_bootstrap",
    "task_macro_mean", "lower_bound", "clustered_delta",
]
