"""Canonical three-task wrapper around a released GLIM representation model.

The released GLIM prompt vocabulary collapses SR and NR into ``<NR>``.  This
adapter preserves the released checkpoint path as a base representation and
adds separate, zero-initialized trainable deltas for canonical SR/NR/TSR task
identities.  Zero initialization makes the initial forward exactly compatible
with the released task embedding while allowing a controlled task-aware pilot.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from pathlib import Path
import sys
import types
from typing import Any, Dict, Literal

import torch
from torch import nn


CANONICAL_TASKS = ("<SR>", "<NR>", "<TSR>")
RELEASED_TASK = {"<SR>": "<NR>", "<NR>": "<NR>", "<TSR>": "<TSR>"}


def load_upstream_glim_class(glim_root: str | Path):
    """Load upstream ``model/glim.py`` without colliding with SemKey's ``model`` package."""
    model_root = Path(glim_root).resolve() / "model"
    if not (model_root / "glim.py").is_file() or not (model_root / "modules.py").is_file():
        raise FileNotFoundError(f"invalid GLIM source root: {glim_root}")
    package_name = "_task_aware_upstream_glim"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    elif list(package.__path__) != [str(model_root)]:
        raise RuntimeError("a different GLIM source root is already loaded in this process")
    # The public GLIM commit annotates one hook with torch.Dict/torch.Any,
    # which are not PyTorch attributes and make the released module fail at
    # import time. Supply the intended typing aliases only while importing;
    # this leaves the upstream checkout unchanged and does not alter weights.
    temporary_torch_aliases = {"Dict": Dict, "Any": Any}
    added_aliases = []
    for name, value in temporary_torch_aliases.items():
        if not hasattr(torch, name):
            setattr(torch, name, value)
            added_aliases.append(name)
    try:
        return importlib.import_module(f"{package_name}.glim").GLIM
    finally:
        for name in added_aliases:
            delattr(torch, name)


def _prompt_columns(prompts) -> tuple[list[str], list[str], list[str]]:
    """Normalize one prompt triple or a collated batch into three columns."""
    if len(prompts) != 3:
        raise ValueError("prompts must contain task, dataset, and subject")
    if all(isinstance(value, str) for value in prompts):
        return ([prompts[0]], [prompts[1]], [prompts[2]])
    columns = tuple(list(value) for value in prompts)
    if not columns[0] or len({len(value) for value in columns}) != 1:
        raise ValueError("collated prompt columns must have the same non-zero length")
    return columns


class CanonicalGLIMRepresentationAdapter(nn.Module):
    """Expose canonical task conditioning and identity-safe GLIM embeddings."""

    def __init__(self, glim_model: nn.Module):
        super().__init__()
        self.glim_model = glim_model
        prompt_dim = int(glim_model.p_embedder.dim)
        self.task_delta = nn.Embedding(len(CANONICAL_TASKS), prompt_dim)
        nn.init.zeros_(self.task_delta.weight)

    def prompt_embedding(
        self,
        prompts,
        mode: Literal["released", "canonical", "task_masked"] = "canonical",
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task, dataset, subject = _prompt_columns(prompts)
        unknown = sorted(set(task) - set(CANONICAL_TASKS))
        if unknown:
            raise ValueError(f"unknown canonical task prompts: {unknown}")
        canonical_ids = torch.tensor(
            [CANONICAL_TASKS.index(value) for value in task], dtype=torch.long, device=device
        )

        if mode == "task_masked":
            released_task = ["<UNK>"] * len(task)
        elif mode in {"released", "canonical"}:
            released_task = [RELEASED_TASK[value] for value in task]
        else:
            raise ValueError(f"unknown prompt mode: {mode}")

        released_prompts = (released_task, dataset, subject)
        prompt_ids = self.glim_model.p_embedder.encode(released_prompts, device=device)
        base = self.glim_model.p_embedder(prompt_ids, self.glim_model.eval_pembed)
        if mode == "canonical":
            base = base + self.task_delta(canonical_ids)
        return base, canonical_ids

    def forward(
        self,
        eeg: torch.Tensor,
        mask: torch.Tensor,
        prompts,
        sample_ids: Sequence[str] | None = None,
        source_dataframe_row_indices: Sequence[int] | None = None,
        mode: Literal["released", "canonical", "task_masked"] = "canonical",
    ) -> dict:
        if eeg.ndim == 2:
            eeg = eeg.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if eeg.ndim != 3 or mask.ndim != 2 or eeg.shape[:2] != mask.shape:
            raise ValueError(
                f"expected eeg [batch,time,channels] and mask [batch,time], got "
                f"{tuple(eeg.shape)} and {tuple(mask.shape)}"
            )
        device = eeg.device
        prompt_embed, canonical_task_ids = self.prompt_embedding(prompts, mode, device)
        if prompt_embed.shape[0] != eeg.shape[0]:
            raise ValueError("prompt batch size does not match EEG batch size")

        eeg_hiddens, _ = self.glim_model.eeg_encoder(eeg, mask, prompt_embed)
        eeg_tokens, eeg_vector = self.glim_model.aligner.embed_eeg(eeg_hiddens)
        if eeg_vector.ndim == 1:
            eeg_vector = eeg_vector.unsqueeze(0)
        if not torch.isfinite(eeg_tokens).all() or not torch.isfinite(eeg_vector).all():
            raise ValueError("GLIM representation contains non-finite values")

        batch_size = eeg.shape[0]
        ids = list(sample_ids) if sample_ids is not None else [None] * batch_size
        rows = (
            [int(value) for value in source_dataframe_row_indices]
            if source_dataframe_row_indices is not None
            else [None] * batch_size
        )
        if len(ids) != batch_size or len(rows) != batch_size:
            raise ValueError("identity metadata must match EEG batch size")
        return {
            "sample_id": ids,
            "source_dataframe_row_index": rows,
            "canonical_task_id": canonical_task_ids,
            "eeg_tokens": eeg_tokens,
            "eeg_vector": eeg_vector,
        }
