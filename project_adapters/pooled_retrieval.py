"""Single-vector (pooled-cosine) EEG->text retrieval — the pooled arm of the
primary causal pair.

This is the structural twin of ``token_late_interaction.py``: it wraps the SAME
``LowRankResidual(1024, 96)`` (196,608 trainable parameters) and the SAME
symmetric contrastive objective, so the pooled and MaxSim arms differ in exactly
one thing — the scoring function. The pooled arm applies the residual to GLIM's
pooled ``eeg_vector`` [B,1024] and scores by cosine; the MaxSim arm applies the
same residual per token and scores by MaxSim. Any Δ between the arms is therefore
attributable to late interaction, not to capacity, loss, or optimization.

Gate-2B lock (2026-07-22): the pooled arm consumes the co-extracted pooled
vectors that are bit-identical to P4b's frozen vectors (Gate-2A), so its
operating point reproduces the established macro MRR ≈ 0.32 as a sanity anchor.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from project_adapters.task_treatment_pilots import LowRankResidual
from project_adapters.token_late_interaction import (
    P4B_TRAINABLE_PARAMETERS,
    RANK,
    VECTOR_DIM,
    l2_normalize,
)


class PooledContrastiveAdapter(nn.Module):
    """One shared rank-96 residual applied to the pooled vector; budget-matched.

    Trainable-parameter count is identical to ``TokenLateInteractionAdapter``
    (196,608), so the primary pair is capacity-matched by construction.
    """

    def __init__(self, vector_dim: int = VECTOR_DIM, rank: int = RANK):
        super().__init__()
        if vector_dim != VECTOR_DIM or rank != RANK:
            raise ValueError("pooled arm matches P4b/token arm: vector_dim=1024, rank=96")
        self.vector_dim = vector_dim
        self.rank = rank
        self.shared = LowRankResidual(vector_dim, rank)

    def project(self, vectors: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B, D]; residual adapter applied to each pooled vector."""
        if vectors.ndim != 2 or vectors.shape[-1] != self.vector_dim:
            raise ValueError(f"expected [batch, {self.vector_dim}]")
        return vectors + self.shared.delta(vectors)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pooled_similarity_matrix(
    eeg_vectors: torch.Tensor, text_vectors: torch.Tensor
) -> torch.Tensor:
    """Cosine score matrix.

    ``eeg_vectors`` [Bq, D] (already projected), ``text_vectors`` [Bc, D].
    Returns [Bq, Bc] cosine similarities (dot of L2-normalized vectors).
    """
    if eeg_vectors.ndim != 2 or text_vectors.ndim != 2:
        raise ValueError("vectors must be [batch, dim]")
    if eeg_vectors.shape[-1] != text_vectors.shape[-1]:
        raise ValueError("EEG and text vector dimensions must match")
    e = l2_normalize(eeg_vectors, dim=-1)
    t = l2_normalize(text_vectors, dim=-1)
    return e @ t.t()


def symmetric_pooled_loss(
    eeg_vectors: torch.Tensor, text_vectors: torch.Tensor, temperature: float = 0.05
) -> torch.Tensor:
    """In-batch symmetric contrastive loss; positives on the diagonal.

    Identical in form to ``symmetric_maxsim_loss`` (same temperature default), so
    the two arms share the objective and differ only in the similarity used.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = pooled_similarity_matrix(eeg_vectors, text_vectors) / temperature
    b = scores.shape[0]
    if scores.shape[1] != b:
        raise ValueError("symmetric loss requires a square score matrix")
    target = torch.arange(b, device=scores.device)
    return 0.5 * (F.cross_entropy(scores, target) + F.cross_entropy(scores.t(), target))


def pooled_reciprocal_rank_of_positive(
    query_eeg_vector: torch.Tensor,
    candidate_text_vectors: torch.Tensor,
    positive_index: int,
) -> float:
    """Rank one query's positive among a candidate pool by cosine (ties: worst)."""
    if query_eeg_vector.ndim == 1:
        query_eeg_vector = query_eeg_vector.unsqueeze(0)
    scores = pooled_similarity_matrix(query_eeg_vector, candidate_text_vectors)[0]
    positive_score = scores[positive_index]
    rank = int((scores > positive_score).sum().item()) + int((scores == positive_score).sum().item())
    return 1.0 / rank


__all__ = [
    "PooledContrastiveAdapter",
    "pooled_similarity_matrix",
    "symmetric_pooled_loss",
    "pooled_reciprocal_rank_of_positive",
    "P4B_TRAINABLE_PARAMETERS",
]
