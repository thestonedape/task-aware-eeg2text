"""Fixed-budget task-treatment adapters for the bounded four-way pilot.

All configurations consume the same prompt-neutral frozen GLIM vector.  Task,
dataset, and subject prompts therefore cannot leak into the generic baseline.
Only the trainable treatment below differs between configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F


TASKS = ("SR", "NR", "TSR")
CONFIG_IDS = ("generic_pooled", "separate_per_task", "task_token", "masked_shared_private")
TaskCondition = Literal["correct", "masked", "shuffled"]


@dataclass(frozen=True)
class PilotSpec:
    config_id: str
    shared_rank: int
    private_rank: int
    task_token: bool
    description: str


PILOT_SPECS = {
    "generic_pooled": PilotSpec(
        "generic_pooled", 96, 0, False,
        "one shared adapter; task identity is never provided",
    ),
    "separate_per_task": PilotSpec(
        "separate_per_task", 0, 32, False,
        "three independent hard-routed adapters; no shared trainable adapter",
    ),
    "task_token": PilotSpec(
        "task_token", 96, 0, True,
        "one shared adapter conditioned by a canonical three-way task token",
    ),
    "masked_shared_private": PilotSpec(
        "masked_shared_private", 48, 16, False,
        "one shared adapter plus hard-masked task-private adapters",
    ),
}


class LowRankResidual(nn.Module):
    """Bias-free bottleneck residual with an explicit, auditable parameter count."""

    def __init__(self, vector_dim: int, rank: int, extra_dim: int = 0):
        super().__init__()
        if vector_dim <= 0 or rank <= 0 or extra_dim < 0:
            raise ValueError("vector_dim/rank must be positive and extra_dim non-negative")
        self.vector_dim = vector_dim
        self.extra_dim = extra_dim
        self.down = nn.Linear(vector_dim + extra_dim, rank, bias=False)
        self.up = nn.Linear(rank, vector_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def delta(self, vector: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        if self.extra_dim:
            if extra is None or extra.shape != (vector.shape[0], self.extra_dim):
                raise ValueError("task-token tensor has the wrong shape")
            vector = torch.cat((vector, extra.to(dtype=vector.dtype)), dim=-1)
        elif extra is not None:
            raise ValueError("this adapter does not accept an extra input")
        return self.up(F.gelu(self.down(vector)))


def canonical_task_ids(
    tasks: torch.Tensor | Sequence[str] | Sequence[int], device: torch.device
) -> torch.Tensor:
    if isinstance(tasks, torch.Tensor):
        ids = tasks.to(device=device, dtype=torch.long)
    else:
        values = list(tasks)
        ids = torch.tensor(
            [TASKS.index(str(value).strip("<>")) if isinstance(value, str) else int(value) for value in values],
            dtype=torch.long,
            device=device,
        )
    if ids.ndim != 1 or bool(((ids < 0) | (ids >= len(TASKS))).any()):
        raise ValueError("tasks must be a one-dimensional SR/NR/TSR ID vector")
    return ids


class TaskTreatmentPilot(nn.Module):
    """One of the four locked pilot treatments over a common frozen vector."""

    def __init__(self, config_id: str, vector_dim: int = 1024):
        super().__init__()
        if config_id not in PILOT_SPECS:
            raise ValueError(f"unknown pilot configuration: {config_id}")
        self.spec = PILOT_SPECS[config_id]
        self.config_id = config_id
        self.vector_dim = vector_dim
        self.shared = (
            LowRankResidual(vector_dim, self.spec.shared_rank, len(TASKS) if self.spec.task_token else 0)
            if self.spec.shared_rank
            else None
        )
        self.private = nn.ModuleList(
            [LowRankResidual(vector_dim, self.spec.private_rank) for _ in TASKS]
            if self.spec.private_rank
            else []
        )

    def _private_delta(self, vector: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(vector)
        for task_id, adapter in enumerate(self.private):
            mask = (task_ids == task_id).to(dtype=vector.dtype).unsqueeze(1)
            delta = delta + adapter.delta(vector) * mask
        return delta

    def forward(
        self,
        vector: torch.Tensor,
        tasks: torch.Tensor | Sequence[str] | Sequence[int],
        task_condition: TaskCondition = "correct",
    ) -> torch.Tensor:
        if vector.ndim != 2 or vector.shape[1] != self.vector_dim:
            raise ValueError(f"expected [batch,{self.vector_dim}] vectors")
        task_ids = canonical_task_ids(tasks, vector.device)
        if task_ids.shape[0] != vector.shape[0]:
            raise ValueError("task and vector batch sizes differ")
        if task_condition not in {"correct", "masked", "shuffled"}:
            raise ValueError(f"unknown task condition: {task_condition}")
        routed_ids = (task_ids + 1) % len(TASKS) if task_condition == "shuffled" else task_ids

        if self.config_id == "generic_pooled":
            assert self.shared is not None
            return vector + self.shared.delta(vector)

        if self.config_id == "task_token":
            assert self.shared is not None
            token = torch.zeros(vector.shape[0], len(TASKS), device=vector.device, dtype=vector.dtype)
            if task_condition != "masked":
                token.scatter_(1, routed_ids.unsqueeze(1), 1.0)
            return vector + self.shared.delta(vector, token)

        if self.config_id == "separate_per_task":
            if task_condition == "masked":
                mean_delta = torch.stack([adapter.delta(vector) for adapter in self.private]).mean(0)
                return vector + mean_delta
            return vector + self._private_delta(vector, routed_ids)

        assert self.config_id == "masked_shared_private" and self.shared is not None
        output = vector + self.shared.delta(vector)
        if task_condition != "masked":
            output = output + self._private_delta(vector, routed_ids)
        return output

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def symmetric_alignment_loss(eeg_vectors: torch.Tensor, text_vectors: torch.Tensor) -> torch.Tensor:
    """GLIM-compatible symmetric in-batch cosine contrastive loss."""
    if eeg_vectors.shape != text_vectors.shape or eeg_vectors.ndim != 2:
        raise ValueError("EEG and text vectors must have the same [batch,dimension] shape")
    eeg = F.normalize(eeg_vectors, dim=1)
    text = F.normalize(text_vectors, dim=1)
    logits = eeg @ text.T
    target = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def parameter_budget(vector_dim: int = 1024) -> dict[str, int | float]:
    counts = {
        config_id: TaskTreatmentPilot(config_id, vector_dim).trainable_parameter_count
        for config_id in CONFIG_IDS
    }
    reference = counts["generic_pooled"]
    maximum_relative_deviation = max(abs(count - reference) / reference for count in counts.values())
    return {**counts, "reference": reference, "maximum_relative_deviation": maximum_relative_deviation}

