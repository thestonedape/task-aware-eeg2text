"""Late-interaction (MaxSim) EEG->text retrieval over unpooled GLIM tokens.

Motivation. Every pooled-vector experiment (the P4 routing pilot, the P4b
task-segmented objective, and the factor-recoverability screen) consumed only
GLIM's pooled ``eeg_vector`` [B,1024] and discarded the ``eeg_tokens``
[B,96,1024] sequence that ``glim_representation`` already returns. This module
scores retrieval over the unpooled tokens with a ColBERT-style MaxSim.

Capacity control. The only trained module is one shared rank-96 residual adapter
applied per EEG token. Its trainable-parameter budget is therefore identical to
the P4b pooled adapter (196,608), so any retrieval gain over the pooled baseline
cannot be attributed to added capacity.

Order note. MaxSim (mean over query tokens of the max over key tokens) is
invariant to the order of both token sets. Permuting token order is therefore a
no-op and is NOT a valid "does token structure matter" control. The correct
ablation is ``mean_collapsed`` — replace the 96 tokens by their single mean — so
that the token model is compared against a one-token summary of the same budget.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from project_adapters.task_treatment_pilots import LowRankResidual

VECTOR_DIM = 1024
RANK = 96
P4B_TRAINABLE_PARAMETERS = 196_608  # LowRankResidual(1024, 96): 2 * 1024 * 96


class TokenLateInteractionAdapter(nn.Module):
    """One shared rank-96 residual applied per token; budget-matched to P4b."""

    def __init__(self, vector_dim: int = VECTOR_DIM, rank: int = RANK):
        super().__init__()
        if vector_dim != VECTOR_DIM or rank != RANK:
            raise ValueError("token model matches P4b: vector_dim=1024, rank=96")
        self.vector_dim = vector_dim
        self.rank = rank
        self.shared = LowRankResidual(vector_dim, rank)

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, D]; residual adapter applied to every token."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.vector_dim:
            raise ValueError(f"expected [batch, tokens, {self.vector_dim}]")
        b, t, d = tokens.shape
        flat = tokens.reshape(b * t, d)
        out = flat + self.shared.delta(flat)
        return out.reshape(b, t, d)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def maxsim_matrix(
    eeg_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    eeg_mask: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MaxSim score matrix.

    ``eeg_tokens`` [Bq, Te, D] (already projected), ``text_tokens`` [Bc, Tt, D].
    Returns [Bq, Bc] where score[i,j] is the mean over valid EEG tokens of the
    max cosine over valid text tokens. Cosine is a dot of L2-normalized tokens.
    """
    if eeg_tokens.ndim != 3 or text_tokens.ndim != 3:
        raise ValueError("token tensors must be [batch, tokens, dim]")
    if eeg_tokens.shape[-1] != text_tokens.shape[-1]:
        raise ValueError("EEG and text token dimensions must match")
    e = l2_normalize(eeg_tokens, dim=-1)
    t = l2_normalize(text_tokens, dim=-1)
    sim = torch.einsum("ipd,jqd->ijpq", e, t)  # [Bq, Bc, Te, Tt]
    if text_mask is not None:
        tm = text_mask.to(torch.bool)
        sim = sim.masked_fill(~tm[None, :, None, :], float("-inf"))
    maxed = sim.max(dim=-1).values  # [Bq, Bc, Te]
    if eeg_mask is not None:
        em = eeg_mask.to(torch.bool)
        maxed = maxed.masked_fill(~em[:, None, :], 0.0)
        denom = em.sum(dim=1).clamp_min(1).to(maxed.dtype)
    else:
        denom = torch.full((maxed.shape[0],), maxed.shape[-1],
                           dtype=maxed.dtype, device=maxed.device)
    return maxed.sum(dim=-1) / denom[:, None]


def symmetric_maxsim_loss(
    eeg_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    eeg_mask: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """In-batch symmetric contrastive loss; positives are the diagonal."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = maxsim_matrix(eeg_tokens, text_tokens, eeg_mask, text_mask) / temperature
    b = scores.shape[0]
    if scores.shape[1] != b:
        raise ValueError("symmetric loss requires a square score matrix")
    target = torch.arange(b, device=scores.device)
    return 0.5 * (F.cross_entropy(scores, target) + F.cross_entropy(scores.t(), target))


def mean_collapsed(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Structure ablation: replace each trial's tokens by their mean, as [B,1,D]."""
    if tokens.ndim != 3:
        raise ValueError("tokens must be [batch, tokens, dim]")
    if mask is not None:
        m = mask.to(tokens.dtype).unsqueeze(-1)
        mean = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
    else:
        mean = tokens.mean(dim=1)
    return mean.unsqueeze(1)


def reciprocal_rank_of_positive(
    query_eeg_tokens: torch.Tensor,
    candidate_text_tokens: torch.Tensor,
    positive_index: int,
    query_eeg_mask: torch.Tensor | None = None,
    candidate_text_mask: torch.Tensor | None = None,
) -> float:
    """Rank one query's positive among a candidate pool by MaxSim (ties: worst)."""
    scores = maxsim_matrix(
        query_eeg_tokens, candidate_text_tokens,
        eeg_mask=query_eeg_mask, text_mask=candidate_text_mask,
    )[0]  # [num_candidates]
    positive_score = scores[positive_index]
    rank = int((scores > positive_score).sum().item()) + int((scores == positive_score).sum().item())
    return 1.0 / rank
