"""MaxSim convergence + hyperparameter-sensitivity diagnostic (D-031 follow-up).

The Gate-3 primary result had MaxSim land 0.075 macro-MRR *below* the pooled arm
(not at parity) and fail the mean-collapse control. A clean "no advantage" null
looks like MaxSim ~= pooled; MaxSim well *under* pooled is a yellow flag that it may
be undertrained under the locked shared hyperparameters (temp 0.05 / lr 1e-3 were
natural for pooled cosine; MaxSim's mean-of-max-cosine scores live on a different
scale -- the F4 critic caveat).

This is a DIAGNOSTIC, not a protocol change. The locked primary pair *requires*
shared hyperparameters to isolate scoring, so nothing here revises the primary
number (lock 13). It only answers one interpretive question: does a fairly-tuned
MaxSim reach parity with pooled, or does it stay below regardless?

  * stays <= pooled at its own best temp/lr  -> the null is robust (report as-is,
    consistent with ABPR: multi-vector lost to CLS there too);
  * reaches parity only with a MaxSim-specific temp/lr -> report the primary as an
    equivalence null plus a shared-hyperparameter sensitivity note.

MaxSim-only, one fold + one seed, so it is fast. It reuses the tested core
(assembly, ``train_arm``, ``macro_mrr``); it only sweeps and records traces.
"""

from __future__ import annotations

from dataclasses import replace

from evaluation.primary_pair_eval import macro_mrr
from evaluation.token_campaign import assemble_pair_trials
from evaluation.token_campaign_driver import _fit_features
from evaluation.token_campaign_io import select_partition
from evaluation.token_training import TrainConfig, deterministic_batches, train_arm


def prepare_fold(
    fold, assignments, candidate_pools, donors, eeg_lookup, text_lookup, pool_size=24,
):
    """Build the confirmation + checkpoint trials and fit features for one fold
    (the same objects the campaign driver builds, so the diagnostic sees identical
    data to the primary run)."""
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
    _pooled_feats, maxsim_feats, fit_text_ids = _fit_features(fit, eeg_lookup, text_lookup)
    return confirmation, checkpoint, maxsim_feats, fit_text_ids, _pooled_feats


def _loss_summary(trace: list[float]) -> dict:
    """Compact convergence view of a per-step loss trace."""
    n = len(trace)
    tail = trace[max(0, n - 50):]
    mid = trace[max(0, n // 2 - 25):n // 2 + 25] or trace
    return {
        "steps": n,
        "first": round(trace[0], 5) if trace else None,
        "last": round(trace[-1], 5) if trace else None,
        "min": round(min(trace), 5) if trace else None,
        "tail_mean": round(sum(tail) / len(tail), 5) if tail else None,
        # plateau ~ tail_mean close to the mid-training mean: little late movement.
        "mid_mean": round(sum(mid) / len(mid), 5) if mid else None,
    }


def run_convergence_diagnostic(
    fold,
    seed: int,
    assignments: list,
    candidate_pools: dict,
    donors: dict,
    eeg_lookup: dict,
    text_lookup: dict,
    *,
    base_config: TrainConfig,
    temperatures,
    lrs=(None,),
    select_every: int,
    device: str = "cpu",
    pool_size: int = 24,
) -> dict:
    """Train MaxSim over a (temperature x lr) grid on one fold/seed; for each cell
    report confirmation macro-MRR, the mean-collapse gap, and loss/dev-MRR traces.

    ``lrs`` entries of ``None`` mean "use ``base_config.lr``". The pooled arm is NOT
    retrained here -- pass ``pooled_reference`` (its confirmation macro-MRR at the
    locked config, from the primary run) so every cell is read against the same bar.
    """
    from project_adapters.pooled_retrieval import PooledContrastiveAdapter  # local: keep import graph light

    confirmation, checkpoint, maxsim_feats, fit_text_ids, pooled_feats = prepare_fold(
        fold, assignments, candidate_pools, donors, eeg_lookup, text_lookup, pool_size)

    # Pooled reference on THIS fold at the locked config, so the comparison is
    # same-fold same-seed (not the whole-campaign 0.32 -- this isolates the gap).
    batches = deterministic_batches(
        len(fit_text_ids), base_config.batch_size, base_config.epochs, seed, text_ids=fit_text_ids)

    def pooled_hook(adapter):
        return macro_mrr(adapter, "pooled", checkpoint, text_lookup)

    pooled_adapter, _ = train_arm(
        "pooled", pooled_feats, batches, base_config, seed, device=device,
        select_hook=pooled_hook, select_every=select_every)
    pooled_mrr = macro_mrr(pooled_adapter, "pooled", confirmation, text_lookup)

    grid = []
    for temp in temperatures:
        for lr in lrs:
            cfg = replace(base_config, temperature=float(temp),
                          lr=base_config.lr if lr is None else float(lr))
            dev_trace: list[float] = []

            def select_hook(adapter, _dev=dev_trace):
                score = macro_mrr(adapter, "maxsim", checkpoint, text_lookup)
                _dev.append(round(score, 5))
                return score

            adapter, loss_trace = train_arm(
                "maxsim", maxsim_feats, batches, cfg, seed, device=device,
                select_hook=select_hook, select_every=select_every)
            corr = macro_mrr(adapter, "maxsim", confirmation, text_lookup, "correct")
            mc = macro_mrr(adapter, "maxsim", confirmation, text_lookup, "mean_collapse")
            grid.append({
                "temperature": float(temp),
                "lr": cfg.lr,
                "maxsim_confirmation_mrr": round(corr, 5),
                "mean_collapse_gap": round(corr - mc, 5),
                "delta_vs_pooled": round(corr - pooled_mrr, 5),
                "reaches_parity": bool(corr >= pooled_mrr),
                "loss": _loss_summary(loss_trace),
                "dev_mrr_trace": dev_trace,
            })

    best = max(grid, key=lambda c: c["maxsim_confirmation_mrr"]) if grid else None
    return {
        "fold": str(fold),
        "seed": seed,
        "pooled_reference_mrr": round(pooled_mrr, 5),
        "locked_temperature": base_config.temperature,
        "locked_lr": base_config.lr,
        "grid": grid,
        "best_cell": None if best is None else {
            "temperature": best["temperature"], "lr": best["lr"],
            "maxsim_confirmation_mrr": best["maxsim_confirmation_mrr"],
            "delta_vs_pooled": best["delta_vs_pooled"],
            "reaches_parity": best["reaches_parity"],
        },
        "verdict": None if best is None else (
            "PARITY_REACHABLE (fair MaxSim >= pooled: primary is an equivalence null "
            "under a hyperparameter-sensitivity note)"
            if best["reaches_parity"] else
            "NULL_ROBUST (MaxSim stays below pooled at its own best temp/lr)"),
    }


__all__ = ["run_convergence_diagnostic", "prepare_fold"]
