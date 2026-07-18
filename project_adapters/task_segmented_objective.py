"""Task-neutral shared adapter and matched-degree objectives for P4b.

The three P4b arms use exactly the same model.  They differ only in which
candidate pairs are admitted to the symmetric in-batch contrastive loss:

``global_mixed``
    A deterministic, key-dependent mixed partition with no task semantics.
``true_task_segmented``
    Candidates sharing the same true reading-task group.
``pseudo_task_segmented``
    Candidates sharing the same prospectively frozen pseudo-task group.

This module deliberately has no task-conditioned model path.  True and pseudo
groups are consumed only while constructing training-loss masks.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from project_adapters.task_treatment_pilots import LowRankResidual


ARM_IDS = (
    "global_mixed",
    "true_task_segmented",
    "pseudo_task_segmented",
)

BATCH_ROWS = 64
GROUP_COUNT = 2
CELL_ROWS = 16
ALLOWED_CANDIDATES_PER_ANCHOR = 32


class SharedResidualAdapter(nn.Module):
    """The sole P4b model: one shared, bias-free rank-96 residual adapter."""

    def __init__(self, vector_dim: int = 1024, rank: int = 96):
        super().__init__()
        if vector_dim != 1024 or rank != 96:
            raise ValueError("P4b freezes vector_dim=1024 and rank=96")
        self.vector_dim = vector_dim
        self.rank = rank
        self.shared = LowRankResidual(vector_dim, rank)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 2 or vector.shape[1] != self.vector_dim:
            raise ValueError(f"expected [batch,{self.vector_dim}] vectors")
        return vector + self.shared.delta(vector)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def active_parameter_count_per_example(self) -> int:
        # Every parameter belongs to the one shared path used by every example.
        return self.trainable_parameter_count


def _canonical_groups(
    values: torch.Tensor | Sequence[object],
    *,
    name: str,
) -> torch.Tensor:
    """Map two arbitrary group labels to deterministic integer IDs on CPU."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        raw = values.detach().cpu().tolist()
    else:
        raw = list(values)
    if len(raw) != BATCH_ROWS:
        raise ValueError(f"{name} must contain exactly {BATCH_ROWS} rows")

    # repr plus type avoids treating, for example, integer 1 and string "1" as
    # the same label while still providing an ordering for arbitrary scalars.
    identities = [(type(value).__qualname__, repr(value)) for value in raw]
    unique = sorted(set(identities))
    if len(unique) != GROUP_COUNT:
        raise ValueError(f"{name} must contain exactly {GROUP_COUNT} groups")
    mapping = {identity: index for index, identity in enumerate(unique)}
    return torch.tensor([mapping[identity] for identity in identities], dtype=torch.long)


def _validated_groups(
    true_groups: torch.Tensor | Sequence[object],
    pseudo_groups: torch.Tensor | Sequence[object],
) -> tuple[torch.Tensor, torch.Tensor]:
    true_ids = _canonical_groups(true_groups, name="true_groups")
    pseudo_ids = _canonical_groups(pseudo_groups, name="pseudo_groups")
    cells = torch.zeros((GROUP_COUNT, GROUP_COUNT), dtype=torch.long)
    for true_id, pseudo_id in zip(true_ids.tolist(), pseudo_ids.tolist()):
        cells[true_id, pseudo_id] += 1
    expected = torch.full_like(cells, CELL_ROWS)
    if not torch.equal(cells, expected):
        raise ValueError(
            "true/pseudo groups must form a balanced 2x2 design with "
            f"{CELL_ROWS} rows per cell; got {cells.tolist()}"
        )
    return true_ids, pseudo_ids


def _hash_order(key: object, direction: str, anchor: int, candidate: int) -> str:
    payload = "\x1f".join((
        "p4b-global-mask-v1",
        str(key),
        direction,
        str(anchor),
        str(candidate),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _global_mask(key: object, direction: str) -> torch.Tensor:
    mask = torch.zeros((BATCH_ROWS, BATCH_ROWS), dtype=torch.bool)
    for anchor in range(BATCH_ROWS):
        candidates = [candidate for candidate in range(BATCH_ROWS) if candidate != anchor]
        candidates.sort(
            key=lambda candidate: (
                _hash_order(key, direction, anchor, candidate),
                candidate,
            )
        )
        selected = [anchor, *candidates[: ALLOWED_CANDIDATES_PER_ANCHOR - 1]]
        mask[anchor, selected] = True
    return mask


def build_partition_masks(
    arm_id: str,
    true_groups: torch.Tensor | Sequence[object],
    pseudo_groups: torch.Tensor | Sequence[object],
    *,
    key: object,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build EEG-to-text and text-to-EEG masks for one balanced P4b batch.

    Every returned row contains exactly 32 admitted candidates, including its
    aligned diagonal positive.  The two global directions are sampled from
    separate hash domains and are therefore deterministic but not forced to be
    transposes of one another.
    """

    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown P4b arm: {arm_id!r}")
    true_ids, pseudo_ids = _validated_groups(true_groups, pseudo_groups)

    if arm_id == "true_task_segmented":
        eeg_to_text = true_ids[:, None].eq(true_ids[None, :])
        text_to_eeg = eeg_to_text.clone()
    elif arm_id == "pseudo_task_segmented":
        eeg_to_text = pseudo_ids[:, None].eq(pseudo_ids[None, :])
        text_to_eeg = eeg_to_text.clone()
    else:
        eeg_to_text = _global_mask(key, "eeg-to-text")
        text_to_eeg = _global_mask(key, "text-to-eeg")

    expected_degrees = torch.full(
        (BATCH_ROWS,), ALLOWED_CANDIDATES_PER_ANCHOR, dtype=torch.long
    )
    if not torch.equal(eeg_to_text.sum(1), expected_degrees):
        raise AssertionError("EEG-to-text mask degree drifted from the frozen design")
    if not torch.equal(text_to_eeg.sum(1), expected_degrees):
        raise AssertionError("text-to-EEG mask degree drifted from the frozen design")
    if not bool(torch.diagonal(eeg_to_text).all()):
        raise AssertionError("EEG-to-text mask excluded a positive pair")
    if not bool(torch.diagonal(text_to_eeg).all()):
        raise AssertionError("text-to-EEG mask excluded a positive pair")

    if device is not None:
        target = torch.device(device)
        eeg_to_text = eeg_to_text.to(target)
        text_to_eeg = text_to_eeg.to(target)
    return eeg_to_text, text_to_eeg


def _validate_loss_mask(mask: torch.Tensor, rows: int, name: str) -> None:
    if mask.dtype != torch.bool or mask.shape != (rows, rows):
        raise ValueError(f"{name} must be a boolean [{rows},{rows}] tensor")
    if mask.device.type == "meta":
        raise ValueError(f"{name} cannot use the meta device")
    if not bool(torch.diagonal(mask).all()):
        raise ValueError(f"{name} must admit every aligned diagonal positive")
    if bool((mask.sum(1) < 2).any()):
        raise ValueError(f"{name} must admit at least one negative per anchor")


def masked_symmetric_alignment_loss(
    eeg_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    eeg_to_text_mask: torch.Tensor,
    text_to_eeg_mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric cosine cross-entropy over prospectively supplied masks."""

    if eeg_vectors.ndim != 2 or eeg_vectors.shape != text_vectors.shape:
        raise ValueError("EEG and text vectors must share one [batch,dimension] shape")
    if eeg_vectors.device != text_vectors.device:
        raise ValueError("EEG and text vectors must use the same device")
    rows = eeg_vectors.shape[0]
    _validate_loss_mask(eeg_to_text_mask, rows, "eeg_to_text_mask")
    _validate_loss_mask(text_to_eeg_mask, rows, "text_to_eeg_mask")
    if eeg_to_text_mask.device != eeg_vectors.device:
        raise ValueError("eeg_to_text_mask must use the vector device")
    if text_to_eeg_mask.device != eeg_vectors.device:
        raise ValueError("text_to_eeg_mask must use the vector device")

    eeg = F.normalize(eeg_vectors, dim=1)
    text = F.normalize(text_vectors, dim=1)
    logits = eeg @ text.T
    target = torch.arange(rows, device=logits.device)
    eeg_to_text_logits = logits.masked_fill(~eeg_to_text_mask, -torch.inf)
    text_to_eeg_logits = logits.T.masked_fill(~text_to_eeg_mask, -torch.inf)
    return (
        F.cross_entropy(eeg_to_text_logits, target)
        + F.cross_entropy(text_to_eeg_logits, target)
    ) / 2


def task_neutral_forward_signature() -> tuple[str, ...]:
    """Expose the auditable non-self inputs accepted by the shared model."""

    return tuple(
        name
        for name in inspect.signature(SharedResidualAdapter.forward).parameters
        if name != "self"
    )
