"""Top-level pooled-vs-token campaign driver (Gate 3, 4d).

Runs the 5 folds x seeds x 2 arms loop over the frozen P4b 24-way contract and
aggregates the confirmation reciprocal ranks into the primary decision inputs
(lock 1/5/8/10/16): the seed-averaged paired macro-MRR and Top-1 deltas, the
per-seed consistency check, and the two controls computable from the 24-way pools
(mean-collapse and matched-wrong). Everything here calls the already-tested core
(assembly, orchestrator, bootstrap, operationalizations); this module only stitches.

The three conditions needing constructions beyond the 24-way contract -- the
subject-held-out split (16.3), the pool-size sweep (16.4), and the shuffled-label
control (9) -- are produced by deterministic, hash-bound extended pools (built
separately) and folded into ``apply_primary_decision`` by the run notebook.

Analytic 24-way chance macro-MRR is H_24/24 = 0.1573.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.clustered_bootstrap import task_macro_mean
from evaluation.token_campaign import assemble_pair_trials, train_and_confirm
from evaluation.token_campaign_io import select_partition
from evaluation.token_decision import (
    paired_delta,
    seed_averaged_effect,
    seed_consistent,
    top1_indicator,
)
from evaluation.token_training import TrainConfig

ARMS = ("pooled", "maxsim")
ANALYTIC_CHANCE_MRR = 0.15733159073974598
CHANCE_TOLERANCE = 0.02          # "matched-wrong at chance" band around analytic chance


def _fit_features(fit_assignments: list[dict], eeg_lookup: dict, text_lookup: dict):
    """Row-aligned training tensors (both arms) + the per-trial text id for batching."""
    if not fit_assignments:
        raise ValueError("fold has no fit trials")
    eeg = [eeg_lookup[str(a["trial_id"])] for a in fit_assignments]
    text = [text_lookup[str(a["text_target_id"])] for a in fit_assignments]
    pooled = {
        "eeg_vectors": torch.stack([e["vector"] for e in eeg]),
        "text_vectors": torch.stack([t["vector"] for t in text]),
    }
    maxsim = {
        "eeg_tokens": torch.stack([e["tokens"] for e in eeg]),
        "text_tokens": torch.stack([t["tokens"] for t in text]),
        "text_masks": torch.stack([t["mask"] for t in text]),
    }
    fit_text_ids = [str(a["text_target_id"]) for a in fit_assignments]
    return pooled, maxsim, fit_text_ids


def run_campaign(
    outer_folds: list,
    seeds: list[int],
    assignments: list[dict],
    candidate_pools: dict,
    donors: dict,
    eeg_lookup: dict,
    text_lookup: dict,
    config: TrainConfig,
    select_every: int,
    device: str = "cpu",
    pool_size: int = 24,
) -> tuple[dict, dict]:
    """Train + confirm every (fold, seed, arm). Returns
    ``records[arm][seed][source][trial_id] = rr`` and ``labels[trial_id] =
    (subject, text, task)``."""
    records = {arm: {seed: {} for seed in seeds} for arm in ARMS}
    labels: dict = {}
    for fold in outer_folds:
        c_targets, c_pools, c_donors = select_partition(
            assignments, candidate_pools, donors, fold, "confirmation")
        k_targets, k_pools, _ = select_partition(
            assignments, candidate_pools, donors, fold, "checkpoint")
        confirmation = assemble_pair_trials(
            c_targets, c_pools, c_donors, eeg_lookup, text_lookup,
            require_donor=True, pool_size=pool_size)
        checkpoint = assemble_pair_trials(
            k_targets, k_pools, {}, eeg_lookup, text_lookup,
            require_donor=False, pool_size=pool_size)
        fit = [a for a in assignments
               if a["outer_fold"] == str(fold) and a["role"] == "fit"]
        pooled_feats, maxsim_feats, fit_text_ids = _fit_features(fit, eeg_lookup, text_lookup)
        for trial in confirmation:
            labels[trial.trial_id] = (trial.subject_id, trial.text_id, trial.task)
        for seed in seeds:
            for arm, feats in (("pooled", pooled_feats), ("maxsim", maxsim_feats)):
                out = train_and_confirm(
                    arm, feats, fit_text_ids, checkpoint, confirmation,
                    text_lookup, config, seed, select_every, device)
                for source, res in out.items():
                    store = records[arm][seed].setdefault(source, {})
                    for trial_id, rr in zip(res["trial_ids"], res["rr"]):
                        store[trial_id] = rr
    return records, labels


def aggregate_primary(
    records: dict, labels: dict, seeds: list[int],
    *, replicates: int = 5000, seed: int = 20260722,
) -> dict:
    """Aggregate confirmation RR into the primary decision inputs (pool 24)."""
    trial_ids = sorted(labels)
    subjects = [labels[t][0] for t in trial_ids]
    texts = [labels[t][1] for t in trial_ids]
    tasks = [labels[t][2] for t in trial_ids]

    def col(arm: str, s: int, source: str) -> np.ndarray:
        store = records[arm][s][source]
        return np.array([store[t] for t in trial_ids], dtype=np.float64)

    seed_avg = {arm: seed_averaged_effect([col(arm, s, "correct") for s in seeds]) for arm in ARMS}
    mrr_delta = paired_delta(seed_avg["maxsim"], seed_avg["pooled"], subjects, texts, tasks,
                             replicates=replicates, seed=seed)
    top1_delta = paired_delta(top1_indicator(seed_avg["maxsim"]), top1_indicator(seed_avg["pooled"]),
                              subjects, texts, tasks, replicates=replicates, seed=seed)

    per_seed_delta = [task_macro_mean(col("maxsim", s, "correct") - col("pooled", s, "correct"), tasks)
                      for s in seeds]

    maxsim_mean_collapse = seed_averaged_effect([col("maxsim", s, "mean_collapse") for s in seeds])
    maxsim_wrong = seed_averaged_effect([col("maxsim", s, "matched_wrong") for s in seeds])
    correct_mrr = task_macro_mean(seed_avg["maxsim"], tasks)
    wrong_mrr = task_macro_mean(maxsim_wrong, tasks)
    mean_collapse_gap = task_macro_mean(seed_avg["maxsim"] - maxsim_mean_collapse, tasks)

    controls = {
        "mean_collapse_gap_positive": bool(mean_collapse_gap > 0.0),
        "matched_wrong_at_chance": bool(abs(wrong_mrr - ANALYTIC_CHANCE_MRR) <= CHANCE_TOLERANCE),
    }
    return {
        "n_trials": len(trial_ids),
        "mrr_delta": mrr_delta,
        "top1_delta": top1_delta,
        "seed_consistent": seed_consistent(per_seed_delta),
        "per_seed_delta": per_seed_delta,
        "controls": controls,
        "summary": {
            "maxsim_correct_macro_mrr": correct_mrr,
            "maxsim_matched_wrong_macro_mrr": wrong_mrr,
            "mean_collapse_gap": mean_collapse_gap,
            "analytic_chance_mrr": ANALYTIC_CHANCE_MRR,
        },
    }


__all__ = ["run_campaign", "aggregate_primary", "ARMS", "ANALYTIC_CHANCE_MRR"]
