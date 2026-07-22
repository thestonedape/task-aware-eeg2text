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

from evaluation.primary_pair_eval import PairTrial

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

        text_tokens, text_masks, text_vectors = [], [], []
        for candidate in pool:                              # preserve candidate-rank order
            text = text_lookup.get(str(candidate["candidate_text_target_id"]))
            if text is None:
                raise KeyError(f"missing text representation for {candidate['candidate_text_target_id']}")
            text_tokens.append(text["tokens"])
            text_masks.append(text["mask"])
            text_vectors.append(text["vector"])

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
            positive_index=positive_index,
            eeg_vector=eeg["vector"],
            candidate_text_vectors=torch.stack(text_vectors),
            eeg_tokens=eeg["tokens"],
            candidate_text_tokens=torch.stack(text_tokens),
            candidate_text_mask=torch.stack(text_masks),
            **kwargs,
        ))
    return trials


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


__all__ = ["assemble_pair_trials", "POOL_SIZE"]
