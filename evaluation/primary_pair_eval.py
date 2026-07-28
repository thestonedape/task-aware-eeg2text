"""Retrieval evaluation for the primary causal pair (Gate 3, increment 3b).

Turns a trained arm into per-trial reciprocal ranks carrying the subject / text /
task labels the clustered bootstrap and the decision rule need. One ``PairTrial``
holds BOTH representations of the same physical trial and its one 24-way pool, so
the pooled and MaxSim arms are scored on identical data and differ only in the
scorer:

* pooled arm  -- project the pooled EEG vector, cosine against candidate vectors;
* MaxSim arm  -- project the 96 EEG tokens, MaxSim against candidate text tokens.

Sources: ``correct`` (real EEG), ``matched_wrong`` (donor EEG, the specificity
control), and ``mean_collapse`` (MaxSim only -- the 96 tokens replaced by their
mean, the token-structure control). ``select_best_checkpoint`` implements the
locked checkpoint rule (best development macro-MRR). Pure tensor logic, CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.clustered_bootstrap import task_macro_mean
from project_adapters.pooled_retrieval import PooledContrastiveAdapter, pooled_similarity_matrix
from project_adapters.token_late_interaction import (
    TokenLateInteractionAdapter,
    maxsim_matrix,
    mean_collapsed,
)

TASKS = ("NR", "TSR")
POOLED_SOURCES = ("correct", "matched_wrong")
MAXSIM_SOURCES = ("correct", "matched_wrong", "mean_collapse")


@dataclass
class PairTrial:
    """One evaluation target: both representations, one 24-way pool, and labels."""

    task: str
    subject_id: str
    text_id: str
    positive_index: int
    eeg_vector: torch.Tensor                    # [D]           pooled arm
    candidate_text_vectors: torch.Tensor        # [C, D]        pooled arm
    eeg_tokens: torch.Tensor                     # [Te, D]       MaxSim arm
    candidate_text_tokens: torch.Tensor          # [C, Tt, D]    MaxSim arm
    candidate_text_mask: torch.Tensor | None = None   # [C, Tt]
    eeg_mask: torch.Tensor | None = None              # [Te]
    wrong_eeg_vector: torch.Tensor | None = None      # [D]      matched-wrong donor
    wrong_eeg_tokens: torch.Tensor | None = None      # [Te, D]
    wrong_eeg_mask: torch.Tensor | None = None        # [Te]
    trial_id: str = ""                                # for aggregation + artifacts

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        c = self.candidate_text_vectors.shape[0]
        if not (0 <= self.positive_index < c):
            raise ValueError("positive_index outside candidate pool")
        if self.candidate_text_tokens.shape[0] != c:
            raise ValueError("vector and token pools must have the same size")
        if self.eeg_vector.ndim != 1 or self.eeg_tokens.ndim != 2:
            raise ValueError("eeg_vector must be [D], eeg_tokens [Te, D]")


def _reciprocal_rank(scores: torch.Tensor, positive_index: int) -> float:
    positive = scores[positive_index]
    rank = int((scores > positive).sum().item()) + int((scores == positive).sum().item())
    return 1.0 / rank


def _adapter_device(adapter) -> torch.device:
    return next(adapter.parameters()).device


def pooled_reciprocal_rank(
    adapter: PooledContrastiveAdapter, trial: PairTrial, source: str = "correct"
) -> float:
    if source == "correct":
        vector = trial.eeg_vector
    elif source == "matched_wrong":
        if trial.wrong_eeg_vector is None:
            raise ValueError("trial has no matched-wrong pooled donor")
        vector = trial.wrong_eeg_vector
    else:
        raise ValueError(f"pooled source must be one of {POOLED_SOURCES}")
    dev = _adapter_device(adapter)
    projected = adapter.project(vector.unsqueeze(0).to(dev))          # [1, D]
    scores = pooled_similarity_matrix(projected, trial.candidate_text_vectors.to(dev))[0]
    return _reciprocal_rank(scores, trial.positive_index)


def maxsim_reciprocal_rank(
    adapter: TokenLateInteractionAdapter, trial: PairTrial, source: str = "correct"
) -> float:
    if source == "correct":
        tokens, mask = trial.eeg_tokens, trial.eeg_mask
    elif source == "matched_wrong":
        if trial.wrong_eeg_tokens is None:
            raise ValueError("trial has no matched-wrong token donor")
        tokens, mask = trial.wrong_eeg_tokens, trial.wrong_eeg_mask
    elif source == "mean_collapse":
        tokens = mean_collapsed(
            trial.eeg_tokens.unsqueeze(0),
            None if trial.eeg_mask is None else trial.eeg_mask.unsqueeze(0),
        )[0]
        mask = None
    else:
        raise ValueError(f"maxsim source must be one of {MAXSIM_SOURCES}")
    dev = _adapter_device(adapter)
    projected = adapter.project(tokens.unsqueeze(0).to(dev))          # [1, Te, D]
    scores = maxsim_matrix(
        projected, trial.candidate_text_tokens.to(dev),
        eeg_mask=None if mask is None else mask.unsqueeze(0).to(dev),
        text_mask=None if trial.candidate_text_mask is None else trial.candidate_text_mask.to(dev),
    )[0]
    return _reciprocal_rank(scores, trial.positive_index)


def arm_reciprocal_ranks(adapter, arm: str, trials: list[PairTrial], source: str = "correct") -> dict:
    """Per-trial RR + aligned subject/text/task labels for one arm and source."""
    if arm not in ("pooled", "maxsim"):
        raise ValueError("arm must be 'pooled' or 'maxsim'")
    if not trials:
        raise ValueError("no trials to evaluate")
    score = pooled_reciprocal_rank if arm == "pooled" else maxsim_reciprocal_rank
    rr, subjects, texts, tasks, trial_ids = [], [], [], [], []
    with torch.inference_mode():
        for trial in trials:
            rr.append(score(adapter, trial, source))
            subjects.append(trial.subject_id)
            texts.append(trial.text_id)
            tasks.append(trial.task)
            trial_ids.append(trial.trial_id)
    return {"rr": rr, "subjects": subjects, "texts": texts, "tasks": tasks, "trial_ids": trial_ids}


def macro_mrr(adapter, arm_name: str, trials: list[PairTrial], source: str = "correct") -> float:
    """Task-macro mean reciprocal rank (the checkpoint-selection / headline metric)."""
    result = arm_reciprocal_ranks(adapter, arm_name, trials, source)
    return task_macro_mean(result["rr"], result["tasks"])


def select_best_checkpoint(adapters: list, arm_name: str, selection_trials: list[PairTrial]) -> int:
    """Locked checkpoint rule (§11): pick the checkpoint with the highest
    development macro-MRR on the selection fold. Ties resolve to the earliest."""
    if not adapters:
        raise ValueError("no candidate checkpoints")
    scores = [macro_mrr(a, arm_name, selection_trials) for a in adapters]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return best


__all__ = [
    "PairTrial", "pooled_reciprocal_rank", "maxsim_reciprocal_rank",
    "arm_reciprocal_ranks", "macro_mrr", "select_best_checkpoint",
    "TASKS", "POOLED_SOURCES", "MAXSIM_SOURCES",
]
