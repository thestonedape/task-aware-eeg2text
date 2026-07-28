"""Assemble the primary-pair campaign from the frozen P4b contracts (Gate 3, 4a).

The pooled-vs-token study reuses P4b's frozen evaluation contract verbatim, so its
candidate pools are byte-identical to the established results: the same 24-way,
within-task, sentence-disjoint pools (``candidate_pools.csv``), the same wrong-EEG
donor maps (``confirmation_donors.csv``), and the same normalized-text-grouped fold
roles (assignments). This module does the one error-prone step -- joining those
contracts to the extracted token/vector representations to build ``PairTrial``s --
as pure, unit-testable logic. Loading the CSVs and the .npz representations, the
training loop, and the aggregation live in the run harness that calls this.

A pool is a rank-ordered list of candidate texts; exactly one is the positive.
``positive_index`` is that candidate's position in the stacked pool. For the
confirmation partition each target also has a wrong-EEG donor trial whose EEG
(not text) supplies the matched-wrong control.
"""

from __future__ import annotations

import torch

from evaluation.primary_pair_eval import (
    MAXSIM_SOURCES,
    POOLED_SOURCES,
    PairTrial,
    arm_reciprocal_ranks,
    macro_mrr,
)
from evaluation.token_training import TrainConfig, deterministic_batches, train_arm

POOL_SIZE = 24


def assemble_pair_trials(
    targets: list[dict],
    pools: dict,
    donors: dict,
    eeg_lookup: dict,
    text_lookup: dict,
    *,
    require_donor: bool,
    pool_size: int = POOL_SIZE,
) -> list[PairTrial]:
    """Join frozen contracts + representations into ``PairTrial``s.

    ``targets``: dicts with ``trial_id``, ``reading_task``, ``subject_id``,
    ``text_target_id``.
    ``pools``: ``trial_id -> [candidate, ...]`` in candidate-rank order; each
    candidate has ``candidate_text_target_id`` and boolean ``is_positive``.
    ``donors``: ``trial_id -> donor_trial_id`` (required iff ``require_donor``).
    ``eeg_lookup``: ``trial_id -> {"tokens": [Te,D], "vector": [D]}``.
    ``text_lookup``: ``text_target_id -> {"tokens": [Tt,D], "mask": [Tt], "vector": [D]}``.
    """
    trials: list[PairTrial] = []
    for target in targets:
        tid = str(target["trial_id"])
        pool = pools.get(tid)
        if pool is None:
            raise KeyError(f"no candidate pool for target {tid}")
        if len(pool) != pool_size:
            raise ValueError(f"target {tid}: pool has {len(pool)} candidates, expected {pool_size}")

        positives = [i for i, c in enumerate(pool) if _as_bool(c["is_positive"])]
        if len(positives) != 1:
            raise ValueError(f"target {tid}: expected exactly one positive, got {len(positives)}")
        positive_index = positives[0]

        candidate_text_ids = [str(c["candidate_text_target_id"]) for c in pool]  # rank order
        for cid in candidate_text_ids:                      # validate; resolved at scoring time
            if cid not in text_lookup:
                raise KeyError(f"missing text representation for {cid}")

        eeg = eeg_lookup.get(tid)
        if eeg is None:
            raise KeyError(f"missing EEG representation for target {tid}")

        kwargs = {}
        if require_donor:
            donor_id = donors.get(tid)
            if donor_id is None:
                raise KeyError(f"no wrong-EEG donor for confirmation target {tid}")
            donor = eeg_lookup.get(str(donor_id))
            if donor is None:
                raise KeyError(f"missing EEG representation for donor {donor_id}")
            kwargs = {"wrong_eeg_tokens": donor["tokens"], "wrong_eeg_vector": donor["vector"]}

        trials.append(PairTrial(
            task=str(target["reading_task"]),
            subject_id=str(target["subject_id"]),
            text_id=str(target["text_target_id"]),
            trial_id=tid,
            positive_index=positive_index,
            eeg_vector=eeg["vector"],
            eeg_tokens=eeg["tokens"],
            candidate_text_ids=candidate_text_ids,
            **kwargs,
        ))
    return trials


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def train_and_confirm(
    arm: str,
    fit_features: dict,
    fit_text_ids: list,
    checkpoint_trials: list[PairTrial],
    confirmation_trials: list[PairTrial],
    text_lookup: dict,
    config: TrainConfig,
    seed: int,
    select_every: int,
    device: str = "cpu",
) -> dict:
    """One (arm, fold, seed) run: train on the fit trials with dev-MRR checkpoint
    selection, then score the confirmation pool. Returns per-source per-trial
    reciprocal ranks + labels.

    ``fit_features``: row-aligned training tensors for the fit trials (pooled arm
    needs ``eeg_vectors``/``text_vectors``; MaxSim arm ``eeg_tokens``/``text_tokens``
    /``text_masks``). ``fit_text_ids``: the text identity per fit trial, so batches
    hold distinct texts (no false negatives). The checkpoint-selection metric is
    macro-MRR on ``checkpoint_trials``; scoring differs by arm, everything else is
    shared, per the locked primary pair.
    """
    sources = MAXSIM_SOURCES if arm == "maxsim" else POOLED_SOURCES
    batches = deterministic_batches(
        len(fit_text_ids), config.batch_size, config.epochs, seed, text_ids=fit_text_ids
    )

    def select_hook(adapter):
        return macro_mrr(adapter, arm, checkpoint_trials, text_lookup)

    adapter, _trace = train_arm(
        arm, fit_features, batches, config, seed, device=device,
        select_hook=select_hook, select_every=select_every,
    )
    return {source: arm_reciprocal_ranks(adapter, arm, confirmation_trials, text_lookup, source)
            for source in sources}


__all__ = ["assemble_pair_trials", "train_and_confirm", "POOL_SIZE"]
