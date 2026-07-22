"""Loaders that feed the tested campaign core from real artifacts (Gate 3, 4c).

Reads the extracted token/vector representations and the frozen P4b evaluation
contract into the plain dicts that ``token_campaign``/``primary_pair_eval`` consume:

* ``load_eeg_lookup``  : trial_id      -> {tokens [96,1024], vector [1024]}
* ``load_text_lookup`` : text_target_id-> {tokens [96,1024], mask [96], vector [1024]}
* ``load_assignments`` : per-fold trial roles (fit / checkpoint / confirmation)
* ``load_candidate_pools`` / ``load_donors`` : the frozen 24-way pools + wrong-EEG maps
* ``select_partition``  : slice all of the above for one (outer_fold, partition)

Parsing is pure and unit-tested against synthetic files mirroring the frozen
schemas. The training loop and hashing live in the driver that calls this.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.float32))


def load_eeg_lookup(root: Path) -> dict:
    """trial_id -> {tokens, vector} from the EEG token-run dataset."""
    index = json.loads((root / "token_index.json").read_text(encoding="utf-8"))
    lookup: dict = {}
    for entry in index["chunks"]:
        npz_path = root / entry["token_file"]
        meta = json.loads((npz_path.with_suffix(".json")).read_text(encoding="utf-8"))
        with np.load(npz_path) as arch:
            tokens, vectors = arch["tokens"], arch["vectors"]
        for i, trial_id in enumerate(meta["trial_ids"]):
            lookup[str(trial_id)] = {"tokens": _f32(tokens[i]), "vector": _f32(vectors[i])}
    return lookup


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_text_lookup(root: Path) -> dict:
    """text_target_id -> {tokens, mask, vector} from the text token-run dataset."""
    rows = _read_csv(root / "text_token_index.csv")
    lookup: dict = {}
    cache: dict = {}
    for row in rows:
        rel = row["token_file"]
        if rel not in cache:
            with np.load(root / rel) as arch:
                cache[rel] = (arch["tokens"], arch["masks"], arch["vectors"])
        tokens, masks, vectors = cache[rel]
        off = int(row["token_offset"])
        lookup[str(row["text_target_id"])] = {
            "tokens": _f32(tokens[off]),
            "mask": torch.from_numpy(np.ascontiguousarray(masks[off]).astype(np.int8)),
            "vector": _f32(vectors[off]),
        }
    return lookup


def load_assignments(path: Path) -> list[dict]:
    """Per-fold trial roles (outer_fold, role, trial_id, reading_task, subject_id,
    text_target_id, ...)."""
    return _read_csv(path)


def load_candidate_pools(path: Path) -> dict:
    """(outer_fold, partition, target_trial_id) -> candidates in candidate-rank order.

    Each candidate keeps ``candidate_text_target_id`` and ``is_positive`` as
    ``assemble_pair_trials`` expects.
    """
    grouped: dict = defaultdict(list)
    for row in _read_csv(path):
        grouped[(row["outer_fold"], row["partition"], row["target_trial_id"])].append(row)
    for key, candidates in grouped.items():
        candidates.sort(key=lambda r: int(r["candidate_rank"]))
    return dict(grouped)


def load_donors(path: Path, partition: str = "confirmation") -> dict:
    """(outer_fold, target_trial_id) -> donor_trial_id for the given partition."""
    donors: dict = {}
    for row in _read_csv(path):
        if row["partition"] == partition:
            donors[(row["outer_fold"], row["target_trial_id"])] = row["donor_trial_id"]
    return donors


def select_partition(
    assignments: list[dict],
    candidate_pools: dict,
    donors: dict,
    outer_fold: str,
    partition: str,
) -> tuple[list[dict], dict, dict]:
    """Slice contracts for one (outer_fold, partition): return (targets, pools, donors)
    keyed by trial_id, ready for ``assemble_pair_trials``.

    ``partition`` is the assignment role for the eval pool -- "checkpoint" or
    "confirmation".
    """
    targets = [
        {"trial_id": a["trial_id"], "reading_task": a["reading_task"],
         "subject_id": a["subject_id"], "text_target_id": a["text_target_id"]}
        for a in assignments
        if a["outer_fold"] == str(outer_fold) and a["role"] == partition
    ]
    pools = {
        t["trial_id"]: candidate_pools[(str(outer_fold), partition, t["trial_id"])]
        for t in targets
    }
    donor_map = {
        t["trial_id"]: donors[(str(outer_fold), t["trial_id"])]
        for t in targets
        if (str(outer_fold), t["trial_id"]) in donors
    }
    return targets, pools, donor_map


__all__ = [
    "load_eeg_lookup", "load_text_lookup", "load_assignments",
    "load_candidate_pools", "load_donors", "select_partition",
]
