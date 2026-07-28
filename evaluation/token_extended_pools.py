"""Deterministic, hash-bound extended pools for the three lock conditions that
need data beyond the frozen 24-way contract (Gate 3).

The primary comparison stays on P4b's frozen 24-way pools (identity to the
established results). These generators build the *additional* evaluation data,
reusing P4b's rules so they are comparable:

* ``build_pool_sweep`` (16.4) -- larger within-task, sentence-disjoint,
  closest-length pools at sizes {48, 96, 192};
* ``subject_partition`` (16.3) -- a subject-disjoint fit / held-out split;
* ``shuffle_labels`` (9) -- a derangement of the EEG->text assignment, which a
  genuinely EEG-specific model must send to chance.

All are pure, seeded, and hashable (``pools_sha256``) so every extended artifact
is provenance-bound like the frozen ones. Unit-tested; no GPU, no I/O.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict

DEFAULT_SWEEP = (48, 96, 192)


def catalog_task_texts(catalog: list[dict]) -> dict:
    """reading_task -> sorted list of unique (text_target_id, length) in that task."""
    seen: dict = defaultdict(dict)
    for row in catalog:
        seen[str(row["reading_task"])][str(row["text_target_id"])] = int(
            row["length_words_whitespace_v1"])
    return {task: sorted(texts.items()) for task, texts in seen.items()}


def build_length_matched_pool(
    positive_text_id: str, positive_length: int,
    task_text_lengths: list, size: int,
) -> list[dict]:
    """Positive + (size-1) closest-length, sentence-disjoint, same-task distractors.

    Ranking is by absolute length difference with a deterministic text-id tie-break
    (matches P4b's 24-way rule, just larger). The positive is placed at a fixed
    interior rank so ``is_positive`` is unambiguous.
    """
    if size < 2:
        raise ValueError("pool size must be >= 2")
    others = [(tid, L) for tid, L in task_text_lengths if tid != positive_text_id]
    if len(others) < size - 1:
        raise ValueError(
            f"only {len(others)} same-task distractors available for a {size}-way pool")
    others.sort(key=lambda x: (abs(x[1] - positive_length), x[0]))
    chosen = others[: size - 1]
    pool = [{"candidate_text_target_id": tid, "is_positive": False} for tid, _ in chosen]
    pool.insert(size // 2, {"candidate_text_target_id": positive_text_id, "is_positive": True})
    return pool


def build_pool_sweep(
    targets: list[dict], catalog: list[dict], sizes=DEFAULT_SWEEP,
) -> dict:
    """size -> {trial_id -> candidate pool}. ``targets`` need ``trial_id``; their
    text/length/task are looked up in ``catalog``."""
    task_texts = catalog_task_texts(catalog)
    by_trial = {str(r["trial_id"]): r for r in catalog}
    sweep: dict = {}
    for size in sizes:
        pools: dict = {}
        for target in targets:
            tid = str(target["trial_id"])
            row = by_trial[tid]
            pools[tid] = build_length_matched_pool(
                str(row["text_target_id"]), int(row["length_words_whitespace_v1"]),
                task_texts[str(row["reading_task"])], size)
        sweep[size] = pools
    return sweep


def subject_partition(catalog: list[dict], held_out_fraction: float, seed: int) -> tuple:
    """Deterministic subject-disjoint split -> (fit_subjects, held_out_subjects)."""
    subjects = sorted({str(r["subject_id"]) for r in catalog})
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError("held_out_fraction must be in (0, 1)")
    order = subjects[:]
    random.Random(seed).shuffle(order)
    n_held = max(1, int(round(len(subjects) * held_out_fraction)))
    if n_held >= len(subjects):
        raise ValueError("held-out fraction leaves no fit subjects")
    held = set(order[:n_held])
    return set(order[n_held:]), held


def shuffle_labels(targets: list[dict], seed: int) -> dict:
    """Derange the EEG->text assignment: trial_id -> a DIFFERENT target's text id.

    A derangement (no trial keeps its own text) so every confirmation pair is
    wrong; a truly EEG-specific model then retrieves at chance.
    """
    texts = [str(t["text_target_id"]) for t in targets]
    n = len(texts)
    if n < 2:
        raise ValueError("need >= 2 targets to derange")
    if len(set(texts)) < 2:
        raise ValueError("targets must reference >= 2 distinct texts to derange")
    rng = random.Random(seed)
    order = list(range(n))
    for _ in range(1000):
        rng.shuffle(order)
        if all(texts[order[i]] != texts[i] for i in range(n)):    # no fixed text
            break
    else:
        raise RuntimeError("failed to derange labels")
    return {str(targets[i]["trial_id"]): texts[order[i]] for i in range(n)}


def pools_sha256(obj) -> str:
    """Stable hash of a generated pool/split/label structure for provenance."""
    payload = json.dumps(obj, sort_keys=True, default=sorted).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DEFAULT_SWEEP", "catalog_task_texts", "build_length_matched_pool",
    "build_pool_sweep", "subject_partition", "shuffle_labels", "pools_sha256",
]
