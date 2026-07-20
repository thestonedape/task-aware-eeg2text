"""Token-level retrieval evaluation: turn MaxSim scores into the paper's endpoints.

Given a trained ``TokenLateInteractionAdapter`` and a set of query trials --- each
with EEG tokens, a reading task, a designated positive, and a 24-way candidate
pool of text tokens --- compute per-task and macro reciprocal rank plus the two
controls the spec requires:

* ``matched_wrong`` --- score the same pools with a matched-wrong-EEG donor's
  tokens; a genuine EEG-specific model drops toward chance here.
* ``mean_collapse`` --- replace the query's tokens by their single mean; if the
  full-token score does not beat this, the 96 tokens add nothing over their
  average and any gain over the pooled baseline is not from token structure.

This module contains no model or data loading; it consumes tensors so it can be
unit-tested end to end without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from project_adapters.token_late_interaction import (
    TokenLateInteractionAdapter,
    maxsim_matrix,
    mean_collapsed,
)

TASKS = ("NR", "TSR")
SOURCES = ("correct", "matched_wrong", "mean_collapse")


@dataclass
class TokenTrial:
    """One evaluation target and its frozen 24-way pool."""

    task: str                                  # "NR" or "TSR"
    positive_index: int                        # index of the positive in the pool
    eeg_tokens: torch.Tensor                    # [Te, D]
    candidate_text_tokens: torch.Tensor         # [C, Tt, D]
    eeg_mask: torch.Tensor | None = None        # [Te]
    candidate_text_mask: torch.Tensor | None = None  # [C, Tt]
    wrong_eeg_tokens: torch.Tensor | None = None     # [Te, D] matched-wrong donor
    wrong_eeg_mask: torch.Tensor | None = None       # [Te]

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.eeg_tokens.ndim != 2 or self.candidate_text_tokens.ndim != 3:
            raise ValueError("eeg_tokens must be [Te,D]; candidates [C,Tt,D]")
        if not (0 <= self.positive_index < self.candidate_text_tokens.shape[0]):
            raise ValueError("positive_index outside candidate pool")


def _query_tokens(trial: TokenTrial, source: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    if source == "correct":
        return trial.eeg_tokens, trial.eeg_mask
    if source == "matched_wrong":
        if trial.wrong_eeg_tokens is None:
            raise ValueError("trial has no matched-wrong donor")
        return trial.wrong_eeg_tokens, trial.wrong_eeg_mask
    if source == "mean_collapse":
        collapsed = mean_collapsed(
            trial.eeg_tokens.unsqueeze(0),
            None if trial.eeg_mask is None else trial.eeg_mask.unsqueeze(0),
        )[0]
        return collapsed, None
    raise ValueError(f"source must be one of {SOURCES}")


def trial_reciprocal_rank(
    model: TokenLateInteractionAdapter, trial: TokenTrial, source: str = "correct"
) -> float:
    """Reciprocal rank of the positive in the trial's pool (ties count against)."""
    tokens, mask = _query_tokens(trial, source)
    projected = model.project(tokens.unsqueeze(0))              # [1, Te, D]
    query_mask = None if mask is None else mask.unsqueeze(0)
    scores = maxsim_matrix(
        projected, trial.candidate_text_tokens,                # text tokens stay frozen
        eeg_mask=query_mask, text_mask=trial.candidate_text_mask,
    )[0]                                                          # [C]
    positive = scores[trial.positive_index]
    rank = int((scores > positive).sum().item()) + int((scores == positive).sum().item())
    return 1.0 / rank


def macro_mrr(reciprocal_ranks_by_task: dict[str, list[float]]) -> float:
    """Macro-average of per-task mean reciprocal rank over NR and TSR."""
    per_task = []
    for task in TASKS:
        values = reciprocal_ranks_by_task.get(task, [])
        if values:
            per_task.append(sum(values) / len(values))
    if not per_task:
        raise ValueError("no reciprocal ranks to aggregate")
    return sum(per_task) / len(per_task)


def evaluate(model: TokenLateInteractionAdapter, trials: list[TokenTrial]) -> dict[str, object]:
    """Compute correct/matched-wrong/mean-collapse macro MRR and the signal gap."""
    if not trials:
        raise ValueError("no trials to evaluate")
    have_wrong = all(t.wrong_eeg_tokens is not None for t in trials)
    sources = ["correct", "mean_collapse"] + (["matched_wrong"] if have_wrong else [])

    by_source: dict[str, dict[str, list[float]]] = {s: {task: [] for task in TASKS} for s in sources}
    with torch.inference_mode():
        for trial in trials:
            for source in sources:
                by_source[source][trial.task].append(
                    trial_reciprocal_rank(model, trial, source=source)
                )

    result: dict[str, object] = {
        "num_trials": len(trials),
        "macro_mrr": {s: macro_mrr(by_source[s]) for s in sources},
        "per_task_mrr": {
            s: {task: (sum(v) / len(v) if v else None) for task, v in by_source[s].items()}
            for s in sources
        },
    }
    if have_wrong:
        result["signal_gap"] = result["macro_mrr"]["correct"] - result["macro_mrr"]["matched_wrong"]
    result["token_vs_mean_collapse"] = (
        result["macro_mrr"]["correct"] - result["macro_mrr"]["mean_collapse"]
    )
    return result
