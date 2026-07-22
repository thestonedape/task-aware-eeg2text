"""The Gate-2B primary decision rule for the pooled-vs-token diagnostic.

Turns paired per-trial retrieval outcomes for the two arms into the single,
pre-registered verdict from the lock (§1): MaxSim is a *method win* iff

1. the two-way clustered lower bound of the macro-MRR Δ (MaxSim - pooled) ≥ δ_sup
   (locked at 0.02);
2. every control passes (matched-wrong at chance, shuffled at chance, MaxSim beats
   its own mean-collapse);
3. the advantage holds on BOTH metrics (macro-MRR and macro Top-1: each two-way
   lower bound > 0);
4. it is consistent across seeds and held-out subjects;
5. it does not vanish as pool size grows.

Otherwise the result is a NULL and the equivalence/breadth battery governs it. The
rule is applied ONCE, on the frozen aggregated predictions. Pure functions over
arrays + booleans, so the whole decision is unit-tested without a GPU.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from evaluation.clustered_bootstrap import clustered_delta

DELTA_SUP = 0.02          # locked MaxSim superiority margin (D-030)


def top1_indicator(reciprocal_ranks: Sequence[float]) -> np.ndarray:
    """Per-trial Recall@1 indicator: 1.0 iff the positive was ranked first (RR==1)."""
    rr = np.asarray(reciprocal_ranks, dtype=np.float64)
    return (rr >= 1.0 - 1e-9).astype(np.float64)


def paired_delta(
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    subjects: Sequence[str],
    texts: Sequence[str],
    tasks: Sequence[str],
    **kwargs,
) -> dict:
    """Clustered task-macro Δ of a per-trial quantity between two arms (a - b)."""
    a = np.asarray(arm_a, dtype=np.float64)
    b = np.asarray(arm_b, dtype=np.float64)
    if not (len(a) == len(b) == len(subjects) == len(texts) == len(tasks)):
        raise ValueError("arms and cluster labels must be the same length")
    return clustered_delta(a - b, subjects, texts, tasks, **kwargs)


def apply_primary_decision(
    mrr_delta: dict,
    top1_delta: dict,
    controls: dict,
    seed_consistent: bool,
    subject_consistent: bool,
    pool_size_robust: bool,
    delta_sup: float = DELTA_SUP,
) -> dict:
    """Apply the five-part §1 rule; return the verdict with a per-condition trace.

    ``mrr_delta`` / ``top1_delta`` are ``clustered_delta`` outputs for the paired
    MaxSim-minus-pooled effect. ``controls`` must contain booleans
    ``matched_wrong_at_chance``, ``shuffled_at_chance``, ``mean_collapse_gap_positive``.
    """
    required = {"matched_wrong_at_chance", "shuffled_at_chance", "mean_collapse_gap_positive"}
    missing = required - set(controls)
    if missing:
        raise ValueError(f"controls missing: {sorted(missing)}")

    conditions = {
        "mrr_lower_bound_ge_delta_sup": mrr_delta["two_way_lower_bound"] >= delta_sup,
        "controls_pass": all(bool(controls[k]) for k in required),
        "both_metrics_advantage": (mrr_delta["two_way_lower_bound"] > 0.0
                                   and top1_delta["two_way_lower_bound"] > 0.0),
        "seed_and_subject_consistent": bool(seed_consistent) and bool(subject_consistent),
        "pool_size_robust": bool(pool_size_robust),
    }
    win = all(conditions.values())
    return {
        "delta_sup": float(delta_sup),
        "mrr_delta_point": mrr_delta["point"],
        "mrr_delta_two_way_lower_bound": mrr_delta["two_way_lower_bound"],
        "top1_delta_point": top1_delta["point"],
        "top1_delta_two_way_lower_bound": top1_delta["two_way_lower_bound"],
        "conditions": conditions,
        "verdict": "maxsim_method_win" if win else "null_or_insufficient",
        "failed_conditions": [k for k, ok in conditions.items() if not ok],
        "note": ("A win requires ALL five conditions. Otherwise the result is a NULL "
                 "and the equivalence/breadth battery governs it (lock §13); the null "
                 "is preserved, never reframed as a win."),
    }


__all__ = ["DELTA_SUP", "top1_indicator", "paired_delta", "apply_primary_decision"]
