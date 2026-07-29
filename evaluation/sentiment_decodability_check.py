"""Plain SR-sentiment decodability check (D-032 follow-up; the "did we test it
wrong?" question).

The frozen factor screen rejected SR three-class sentiment because it did not beat
the matched-wrong-EEG decoy (correct-minus-decoy = -0.010), even though it DID beat
the ordinary metadata baseline (correct-minus-metadata = +0.020). For a coarse
3-class label the decoy (same-task, same-subject, different-text EEG) very often
shares the sentiment, so that control is unfair to the attribute -- it can suppress
a real-but-weak signal. This check removes the decoy entirely and asks the simple
question the screen never answered directly: can a plain classifier read SR
sentiment off the frozen GLIM ``correct`` vector *above chance at all*?

It fits the SAME estimator family as the screen (standardized L2 logistic
regression, balanced classes) under subject-grouped cross-validation, and scores it
two ways: against analytic chance (1/3 balanced accuracy) and against a
label-permutation empirical null (the honest chance estimate, since metadata can
make classes guessable without any EEG). A clearly-above-chance result with a small
permutation p means the signal is there and the decoy control was the wrong
instrument; a near-chance result means it is genuinely absent from this frozen
representation. Pure/CPU-testable; the Kaggle runner wires it to the frozen vectors.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:                                                    # sklearn >= 1.0
    from sklearn.model_selection import StratifiedGroupKFold
    _HAVE_SGKF = True
except ImportError:                                     # pragma: no cover
    from sklearn.model_selection import GroupKFold
    _HAVE_SGKF = False


def _splitter(n_splits: int, seed: int):
    if _HAVE_SGKF:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)                # pragma: no cover


def _feasible_splits(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """Largest workable fold count: bounded by group count and the rarest class's
    group support, so every fold can hold every class."""
    n_groups = len(np.unique(groups))
    rarest_class_groups = min(
        len(np.unique(groups[y == c])) for c in np.unique(y)
    )
    return max(2, min(requested, n_groups, rarest_class_groups))


def _oof_balanced_accuracy(X, y, groups, n_splits, seed, C):
    """Out-of-fold balanced accuracy for one label vector (used for both the real
    labels and each permutation)."""
    predicted = np.full(len(y), -1, dtype=np.int64)
    cv = _splitter(n_splits, seed)
    for train_idx, val_idx in cv.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, penalty="l2", solver="liblinear",
                               class_weight="balanced", max_iter=3000, random_state=seed),
        )
        model.fit(X[train_idx], y[train_idx])
        predicted[val_idx] = model.predict(X[val_idx])
    scored = predicted >= 0
    return balanced_accuracy_score(y[scored], predicted[scored]), predicted, scored


def plain_decodability(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 20260729,
    C: float = 1.0,
    n_permutations: int = 200,
) -> dict:
    """Subject-grouped CV decodability of ``y`` from ``X`` with a label-permutation
    null. Returns the real out-of-fold scores plus the empirical chance level and a
    one-sided permutation p-value on balanced accuracy.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, groups must be the same length")
    n_classes = len(np.unique(y))
    splits = _feasible_splits(y, groups, n_splits)

    bal, predicted, scored = _oof_balanced_accuracy(X, y, groups, splits, seed, C)
    real = {
        "balanced_accuracy": float(bal),
        "accuracy": float(accuracy_score(y[scored], predicted[scored])),
        "macro_f1": float(f1_score(y[scored], predicted[scored],
                                   labels=list(range(n_classes)), average="macro", zero_division=0)),
    }

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(len(y))                  # break the X<->y link, keep groups fixed
        null[i], _, _ = _oof_balanced_accuracy(X, y[perm], groups, splits, seed, C)
    p_value = float((1 + np.sum(null >= real["balanced_accuracy"])) / (n_permutations + 1))

    _, counts = np.unique(y, return_counts=True)
    return {
        "n": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "n_classes": int(n_classes),
        "class_counts": counts.astype(int).tolist(),
        "n_splits_used": int(splits),
        "analytic_chance_balanced_accuracy": round(1.0 / n_classes, 6),
        "real": {k: round(v, 6) for k, v in real.items()},
        "permutation_chance_balanced_accuracy": round(float(null.mean()), 6),
        "permutation_chance_sd": round(float(null.std()), 6),
        "p_value_permutation": p_value,
        "above_chance": bool(real["balanced_accuracy"] > null.mean() and p_value < 0.05),
        "verdict": (
            "DECODABLE (signal present; the decoy control was the wrong instrument)"
            if (real["balanced_accuracy"] > null.mean() and p_value < 0.05)
            else "AT CHANCE (genuinely not recoverable from this frozen representation)"
        ),
    }


__all__ = ["plain_decodability"]
