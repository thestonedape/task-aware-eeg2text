"""Gate-1 diagnostics: are GLIM's 96 emitted EEG tokens genuinely varying?

The make-or-break feasibility question for the token late-interaction study is
whether the Q-merger has already collapsed the 96 tokens into near-redundant
copies of one state. If it has, MaxSim cannot help and Route B falls back to A.

This module computes, from a batch of extracted ``eeg_tokens`` [B, T, D]:

* within-trial token redundancy   -- mean off-diagonal cosine among a trial's
  tokens (near 1.0 => collapsed to one direction);
* effective rank (participation ratio) of each trial's [T, D] token matrix
  (near 1 => rank-1 collapse; near T => rich, well-spread tokens);
* per-token norm sanity (no zeros / NaNs);
* position liveness -- for each token position, its variance across trials
  (near 0 => a fixed learned constant carrying no trial-specific information).

Pure tensor logic; unit-tested on synthetic collapsed vs rich tokens with no GPU.
This is a feasibility gate only -- it must not be used for model selection.
"""

from __future__ import annotations

import torch

from project_adapters.token_late_interaction import l2_normalize


def within_trial_redundancy(tokens: torch.Tensor, center: bool = False) -> torch.Tensor:
    """[B, T, D] -> [B] mean off-diagonal cosine among each trial's T tokens.

    ``center=True`` subtracts each trial's mean token first, so a large shared
    "DC" direction (transformer anisotropy) cannot masquerade as collapse. Raw
    and centered redundancy are both reported at Gate 1: raw high + centered low
    means the tokens are anisotropic but genuinely varied, not collapsed.
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B, T, D]")
    if center:
        tokens = tokens - tokens.mean(dim=1, keepdim=True)
    t = l2_normalize(tokens, dim=-1)
    gram = torch.matmul(t, t.transpose(1, 2))          # [B, T, T] cosine
    b, n, _ = gram.shape
    off = gram.sum(dim=(1, 2)) - gram.diagonal(dim1=1, dim2=2).sum(dim=1)
    return off / (n * (n - 1))


def effective_rank(tokens: torch.Tensor) -> torch.Tensor:
    """[B, T, D] -> [B] participation ratio (sum s)^2 / sum(s^2), range 1..T."""
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B, T, D]")
    s = torch.linalg.svdvals(tokens.float())           # [B, min(T,D)]
    s2 = s.pow(2)
    return s.sum(dim=1).pow(2) / s2.sum(dim=1).clamp_min(1e-12)


def position_liveness(tokens: torch.Tensor, dead_threshold: float = 1e-4) -> dict:
    """Across-trial variance per token position; fraction that are near-constant."""
    if tokens.ndim != 3 or tokens.shape[0] < 2:
        raise ValueError("need [B, T, D] with B >= 2")
    var_per_pos = tokens.float().var(dim=0).mean(dim=-1)   # [T]
    dead = (var_per_pos < dead_threshold).float().mean().item()
    return {
        "min_position_variance": float(var_per_pos.min()),
        "mean_position_variance": float(var_per_pos.mean()),
        "fraction_dead_positions": dead,
    }


def verdict_from_stats(
    mean_redundancy: float, mean_effective_rank: float, tokens_per_trial: int,
    finite: bool, min_norm: float,
) -> str:
    """Plain COLLAPSED / WEAK / RICH verdict from aggregate token statistics.

    Separated so it can be applied to streamed cohort-wide aggregates as well as
    a single in-memory batch.
    """
    eff_fraction = mean_effective_rank / tokens_per_trial
    if not finite or min_norm == 0.0:
        return "INVALID: non-finite or zero-norm tokens"
    if mean_redundancy > 0.97 and eff_fraction < 0.05:
        return "COLLAPSED: tokens near-redundant; MaxSim cannot help -> fall back to A"
    if mean_redundancy > 0.9 or eff_fraction < 0.15:
        return "WEAK: limited token diversity; MaxSim upside likely small"
    return "RICH: tokens are genuinely varying; MaxSim has headroom to test"


def token_collapse_report(tokens: torch.Tensor) -> dict:
    """Assemble the Gate-1 report + a plain verdict for one in-memory batch."""
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B, T, D]")
    b, n, d = tokens.shape
    norms = tokens.float().norm(dim=-1)                 # [B, T]
    finite = bool(torch.isfinite(tokens).all())

    # Short-circuit on invalid input before any SVD (which errors on non-finite).
    if not finite or float(norms.min()) == 0.0:
        return {
            "batch": b, "tokens_per_trial": n, "dim": d, "finite": finite,
            "token_norm_min": float("nan") if not finite else float(norms.min()),
            "verdict": verdict_from_stats(0.0, float(n), n, finite,
                                          0.0 if not finite else float(norms.min())),
        }

    redundancy = within_trial_redundancy(tokens)
    eff = effective_rank(tokens)
    live = position_liveness(tokens) if b >= 2 else {}

    mean_redundancy = float(redundancy.mean())
    mean_eff = float(eff.mean())

    return {
        "batch": b, "tokens_per_trial": n, "dim": d,
        "finite": finite,
        "token_norm_mean": float(norms.mean()),
        "token_norm_min": float(norms.min()),
        "token_norm_max": float(norms.max()),
        "within_trial_redundancy_mean": mean_redundancy,
        "within_trial_redundancy_max": float(redundancy.max()),
        "effective_rank_mean": mean_eff,
        "effective_rank_fraction_of_T": mean_eff / n,
        **live,
        "verdict": verdict_from_stats(mean_redundancy, mean_eff, n, finite, float(norms.min())),
    }
