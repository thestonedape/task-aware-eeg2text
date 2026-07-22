"""Arm-symmetric contrastive trainer for the primary causal pair (Gate 3).

Trains the two arms of the Gate-2B primary pair under one code path so they are
identical in everything but the scoring function:

* the batch schedule is arm-independent (a seeded permutation shared by both arms);
* the adapter is initialized from the same seed (both wrap ``LowRankResidual(1024,
  96)``, so identical init weights);
* the optimizer, temperature, epochs, and gradient clipping are shared;
* only the forward/loss differs: pooled cosine (``symmetric_pooled_loss`` on the
  pooled EEG vector) vs token MaxSim (``symmetric_maxsim_loss`` on the 96 EEG
  tokens with the text mask).

Text is frozen in both arms. This module is the pure training core: it consumes
in-memory tensors and returns a trained adapter + loss trace. Checkpoint selection
by development macro-MRR and the control battery live in the evaluation harness
(next increment); the run harness handles hash-bound loading. CPU-testable.

Hyperparameters (lr/epochs/weight-decay/clip) are supplied by ``TrainConfig`` and
must be frozen in the Gate-3 run config (hashed at launch) exactly as P4b did.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch
from torch import nn

from project_adapters.pooled_retrieval import (
    PooledContrastiveAdapter,
    symmetric_pooled_loss,
)
from project_adapters.token_late_interaction import (
    TokenLateInteractionAdapter,
    symmetric_maxsim_loss,
)

Arm = Literal["pooled", "maxsim"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    temperature: float = 0.05
    grad_clip: float | None = 1.0


def deterministic_batches(
    num_trials: int, batch_size: int, epochs: int, seed: int,
    text_ids: "Sequence[str] | None" = None,
) -> list[list[int]]:
    """Seeded per-epoch batches, arm-independent by construction.

    Same (num_trials, batch_size, epochs, seed[, text_ids]) => identical schedule,
    so both arms see identical data ordering and identical in-batch negatives.
    Full batches only (partial final batch dropped) so every step has a square
    score matrix and both arms step in lockstep.

    ``text_ids`` (one normalized-text identity per trial) makes each batch contain
    DISTINCT texts: a trial whose text already sits in the current batch is
    deferred to the next. This removes false negatives (a repeated sentence acting
    as its own contrastive negative), matching the P4b batching rigor. Without
    ``text_ids`` the batcher is a plain permutation (used only in tests).
    """
    if num_trials < batch_size:
        raise ValueError("num_trials must be >= batch_size for a full batch")
    generator = torch.Generator().manual_seed(int(seed))
    batches: list[list[int]] = []
    for _ in range(int(epochs)):
        perm = torch.randperm(num_trials, generator=generator).tolist()
        if text_ids is None:
            for start in range(0, num_trials - batch_size + 1, batch_size):
                batches.append(perm[start:start + batch_size])
            continue
        # Greedy fill: place each trial in the earliest open batch whose texts are
        # all distinct from it, preserving determinism from the fixed permutation.
        open_batches: list[list[int]] = [[]]
        open_texts: list[set] = [set()]
        for idx in perm:
            tid = text_ids[idx]
            placed = False
            for b, seen in zip(open_batches, open_texts):
                if len(b) < batch_size and tid not in seen:
                    b.append(idx); seen.add(tid); placed = True
                    break
            if not placed:
                open_batches.append([idx]); open_texts.append({tid})
            full = [i for i, b in enumerate(open_batches) if len(b) == batch_size]
            for i in reversed(full):
                batches.append(open_batches.pop(i)); open_texts.pop(i)
    return batches


def _make_adapter(arm: Arm) -> nn.Module:
    if arm == "pooled":
        return PooledContrastiveAdapter()
    if arm == "maxsim":
        return TokenLateInteractionAdapter()
    raise ValueError(f"unknown arm: {arm!r}")


def _batch_loss(arm: Arm, adapter: nn.Module, features: dict, index: torch.Tensor,
                temperature: float) -> torch.Tensor:
    if arm == "pooled":
        projected = adapter.project(features["eeg_vectors"][index])          # [B, D]
        return symmetric_pooled_loss(projected, features["text_vectors"][index], temperature)
    projected = adapter.project(features["eeg_tokens"][index])                # [B, T, D]
    return symmetric_maxsim_loss(
        projected, features["text_tokens"][index],
        eeg_mask=None, text_mask=features["text_masks"][index], temperature=temperature,
    )


def _required_keys(arm: Arm) -> tuple[str, ...]:
    return (("eeg_vectors", "text_vectors") if arm == "pooled"
            else ("eeg_tokens", "text_tokens", "text_masks"))


def train_arm(
    arm: Arm,
    features: dict,
    batches: list[list[int]],
    config: TrainConfig,
    seed: int,
    device: str = "cpu",
    select_hook: "Callable[[nn.Module], float] | None" = None,
    select_every: int | None = None,
) -> tuple[nn.Module, list[float]]:
    """Train one arm over a fixed batch schedule; return (adapter, loss trace).

    ``features`` holds row-aligned tensors indexed by trial: pooled arm needs
    ``eeg_vectors``/``text_vectors`` [N,D]; MaxSim arm needs ``eeg_tokens``
    [N,T,D], ``text_tokens`` [N,T,D], ``text_masks`` [N,T]. Each trial's positive
    text occupies the same row index; in-batch negatives are the other rows.

    Checkpoint selection (locked protocol §11): if ``select_hook`` is given, it is
    called every ``select_every`` steps on the current adapter and must return a
    development score (macro-MRR on the selection fold); the adapter is restored to
    the highest-scoring snapshot before returning. Identical hook/schedule across
    arms keeps selection fair.
    """
    missing = [k for k in _required_keys(arm) if k not in features]
    if missing:
        raise ValueError(f"{arm} arm missing features: {missing}")
    if select_hook is not None and not select_every:
        raise ValueError("select_every must be a positive step interval when select_hook is set")
    torch.manual_seed(int(seed))                    # identical adapter init across arms
    adapter = _make_adapter(arm).to(device).train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    feats = {k: v.to(device) for k, v in features.items()}
    trace: list[float] = []
    best_score: float | None = None
    best_state: dict | None = None
    for step, batch in enumerate(batches, start=1):
        index = torch.as_tensor(batch, dtype=torch.long, device=device)
        loss = _batch_loss(arm, adapter, feats, index, config.temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip is not None:
            nn.utils.clip_grad_norm_(adapter.parameters(), config.grad_clip)
        optimizer.step()
        trace.append(float(loss.detach()))
        if select_hook is not None and step % select_every == 0:
            adapter.eval()
            with torch.inference_mode():
                score = float(select_hook(adapter))
            adapter.train()
            if best_score is None or score > best_score:
                best_score = score
                best_state = copy.deepcopy(adapter.state_dict())
    if best_state is not None:                       # restore the best-dev checkpoint
        adapter.load_state_dict(best_state)
    return adapter.eval(), trace


__all__ = ["TrainConfig", "deterministic_batches", "train_arm", "Arm"]
