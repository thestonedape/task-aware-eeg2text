"""Freeze the common deterministic 40-epoch P4b training schedule.

This child freezer consumes only the already frozen P4b protocol tables.  It
does not load an EEG/text vector, train a model, score an outcome, or inspect
validation/test data.  The resulting schedule is shared byte-for-byte by all
three P4b arms and authorizes only the separately bounded training smoke.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


CONTRACT_NAME = "task_segmented_training_schedule_contract.json"
REPORT_NAME = "task_segmented_training_schedule_report.json"
MANIFEST_NAME = "task_segmented_training_schedule_manifest.json"
INDEX_NAME = "schedule_indices.u32le"
PARENT_REPORT_NAME = "task_segmented_protocol_report.json"
PARENT_CONTRACT_NAME = "task_segmented_objective_contract.json"
PARENT_ARTIFACTS = (
    "batch_grid_feasibility.csv",
    "candidate_pools.csv",
    "confirmation_donors.csv",
    "outer_split_assignments.csv",
    "protocol_registry.json",
    "pseudo_groups.csv",
    "text_group_folds.csv",
)
SCHEDULE_ARTIFACTS = (
    "trial_catalog.csv",
    INDEX_NAME,
    "schedule_units.csv",
    "schedule_audit.csv",
    MANIFEST_NAME,
)
CELLS = (("NR", 0), ("NR", 1), ("TSR", 0), ("TSR", 1))
CATALOG_FIELDS = (
    "trial_index", "trial_id", "text_fold", "dataset_version", "reading_task",
    "subject_id", "normalized_text_sha256", "text_target_id", "pseudo_group",
    "length_words_whitespace_v1", "eeg_vector_file", "eeg_vector_offset",
    "eeg_vector_dim",
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_int(value: object, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in {field}: {value!r}") from exc


def load_schedule_contract(path: Path) -> dict[str, object]:
    contract = read_json(path)
    if contract.get("schema_version") != 1:
        raise ValueError("schedule contract schema must be 1")
    if contract.get("status") != "frozen_before_p4b_schedule_generation":
        raise ValueError("schedule contract is not prospectively frozen")
    schedule = contract.get("schedule")
    authorization = contract.get("authorization")
    if not isinstance(schedule, dict) or not isinstance(authorization, dict):
        raise ValueError("schedule contract lacks required sections")
    if schedule.get("batch_size") != 64 or schedule.get("examples_per_cell") != 16:
        raise ValueError("schedule contract must freeze 64 rows and 16 rows per cell")
    if schedule.get("epochs") != 40:
        raise ValueError("schedule contract must freeze 40 epochs")
    if authorization.get("full_training_authorized") is not False:
        raise ValueError("schedule contract cannot authorize full training")
    optimization = contract.get("optimization")
    if not isinstance(optimization, dict) or optimization.get("temperature") != 1.0:
        raise ValueError("schedule contract must freeze unit-temperature cosine loss")
    randomness = contract.get("randomness")
    if not isinstance(randomness, dict) or randomness.get("global_mask_seed") != 2026071806:
        raise ValueError("schedule contract lacks the frozen global-mask seed")
    return contract


def initialization_seed(outer_fold: int, training_seed: int) -> int:
    """Derive the exact paired model-initialization seed for one schedule unit."""

    return int(stable_hash("p4b-adapter-init-v1", outer_fold, training_seed)[:16], 16) & ((1 << 63) - 1)


def global_mask_key(
    schedule_contract_sha256: str,
    schedule_unit_sha256: str,
    outer_fold: int,
    training_seed: int,
    epoch: int,
    batch_index: int,
) -> str:
    """Derive the exact key consumed by ``build_partition_masks``."""

    if len(schedule_contract_sha256) != 64 or len(schedule_unit_sha256) != 64:
        raise ValueError("mask-key contract/unit hashes must be SHA-256 hex digests")
    if outer_fold not in range(5) or epoch not in range(1, 41) or batch_index < 0:
        raise ValueError("mask-key fold/epoch/batch coordinates are invalid")
    return stable_hash(
        "p4b-global-mask-key-v1",
        2026071806,
        schedule_contract_sha256,
        schedule_unit_sha256,
        outer_fold,
        training_seed,
        epoch,
        batch_index,
    )


def verify_parent(
    protocol_root: Path,
    contract: dict[str, object],
    artifact_verifier=None,
) -> dict[str, object]:
    parent = contract["parent"]
    assert isinstance(parent, dict)
    expected_artifacts = parent.get("artifact_sha256")
    if not isinstance(expected_artifacts, dict) or set(expected_artifacts) != set(PARENT_ARTIFACTS):
        raise ValueError("child contract has an invalid parent artifact inventory")
    parent_contract = protocol_root / PARENT_CONTRACT_NAME
    parent_report = protocol_root / PARENT_REPORT_NAME
    if sha256(parent_contract) != parent["task_segmented_objective_contract_sha256"]:
        raise ValueError("parent objective contract hash mismatch")
    if sha256(parent_report) != parent["task_segmented_protocol_report_sha256"]:
        raise ValueError("parent protocol report hash mismatch")
    report = read_json(parent_report)
    if report.get("status") != "pass" or report.get("contract_sha256") != parent["task_segmented_objective_contract_sha256"]:
        raise ValueError("parent protocol report is not the frozen PASS")
    if report.get("artifact_sha256") != expected_artifacts:
        raise ValueError("parent report artifact inventory differs from child contract")
    for name in PARENT_ARTIFACTS:
        if sha256(protocol_root / name) != expected_artifacts[name]:
            raise ValueError(f"parent protocol artifact hash mismatch: {name}")
    verification = report.get("input_verification")
    if not isinstance(verification, dict) or verification.get("preserved_source_id") != parent["upstream_input_source_id"]:
        raise ValueError("parent preserved source identity mismatch")
    readiness = report.get("execution_readiness")
    if not isinstance(readiness, dict) or readiness.get("training_authorized") is not False:
        raise ValueError("parent protocol unexpectedly authorized training")
    clean_remount_summary = None
    if artifact_verifier is not None:
        clean_remount_summary = artifact_verifier(
            protocol_root,
            str(parent["preserved_protocol_artifact_source_id"]),
        )
        if clean_remount_summary.get("status") != "pass":
            raise ValueError("parent clean-remount verifier did not pass")
        if clean_remount_summary.get("protocol_report_sha256") != parent["task_segmented_protocol_report_sha256"]:
            raise ValueError("parent clean-remount summary report hash mismatch")
        if clean_remount_summary.get("preserved_source_id") != parent["preserved_protocol_artifact_source_id"]:
            raise ValueError("parent clean-remount artifact source mismatch")
    return {"report": report, "clean_remount_summary": clean_remount_summary}


def canonical_trial_catalog(
    assignments: list[dict[str, str]], folds: Sequence[int]
) -> tuple[list[dict[str, object]], dict[tuple[int, str], str]]:
    """Collapse the five role rows per trial into one stable global catalog."""

    by_trial: dict[str, list[dict[str, str]]] = defaultdict(list)
    roles: dict[tuple[int, str], str] = {}
    required = {
        "outer_fold", "role", "text_fold", "trial_id", "dataset_version",
        "reading_task", "subject_id", "normalized_text_sha256", "text_target_id",
        "pseudo_group", "length_words_whitespace_v1", "eeg_vector_file",
        "eeg_vector_offset", "eeg_vector_dim",
    }
    for row in assignments:
        if not required.issubset(row):
            raise ValueError("outer assignment table lacks required fields")
        outer_fold = as_int(row["outer_fold"], "outer_fold")
        if outer_fold not in folds:
            raise ValueError(f"unexpected outer fold: {outer_fold}")
        trial_id = row["trial_id"]
        key = (outer_fold, trial_id)
        if key in roles:
            raise ValueError(f"duplicate outer-fold/trial assignment: {key}")
        roles[key] = row["role"]
        by_trial[trial_id].append(row)

    invariant_fields = (
        "text_fold", "dataset_version", "reading_task", "subject_id",
        "normalized_text_sha256", "text_target_id", "pseudo_group",
        "length_words_whitespace_v1", "eeg_vector_file", "eeg_vector_offset",
        "eeg_vector_dim",
    )
    catalog: list[dict[str, object]] = []
    for trial_index, trial_id in enumerate(sorted(by_trial)):
        members = by_trial[trial_id]
        if len(members) != len(folds):
            raise ValueError(f"trial does not have one row per outer fold: {trial_id}")
        first = members[0]
        for field in invariant_fields:
            if {member[field] for member in members} != {first[field]}:
                raise ValueError(f"trial invariant drift in {field}: {trial_id}")
        text_fold = as_int(first["text_fold"], "text_fold")
        if text_fold not in folds:
            raise ValueError(f"invalid text fold for {trial_id}")
        expected_roles = {
            fold: (
                "confirmation" if text_fold == fold
                else "checkpoint" if text_fold == (fold + 1) % len(folds)
                else "fit"
            )
            for fold in folds
        }
        if any(roles[(fold, trial_id)] != expected_roles[fold] for fold in folds):
            raise ValueError(f"cross-fitting role mismatch: {trial_id}")
        task = first["reading_task"]
        pseudo = as_int(first["pseudo_group"], "pseudo_group")
        if (task, pseudo) not in CELLS:
            raise ValueError(f"invalid task/pseudo cell: {task}::{pseudo}")
        if first["dataset_version"] != "ZuCo2" or as_int(first["eeg_vector_dim"], "eeg_vector_dim") != 1024:
            raise ValueError("schedule catalog is not eligible 1024-D ZuCo2")
        catalog.append({
            "trial_index": trial_index,
            "trial_id": trial_id,
            "text_fold": text_fold,
            "dataset_version": first["dataset_version"],
            "reading_task": task,
            "subject_id": first["subject_id"],
            "normalized_text_sha256": first["normalized_text_sha256"],
            "text_target_id": first["text_target_id"],
            "pseudo_group": pseudo,
            "length_words_whitespace_v1": as_int(first["length_words_whitespace_v1"], "length"),
            "eeg_vector_file": first["eeg_vector_file"],
            "eeg_vector_offset": as_int(first["eeg_vector_offset"], "eeg_vector_offset"),
            "eeg_vector_dim": 1024,
        })
    if not catalog:
        raise ValueError("empty schedule catalog")
    return catalog, roles


def _candidate_order(
    cell: tuple[str, int], identities: Iterable[str], uses: Counter[str], key: tuple[object, ...],
    remaining: Counter[str] | None = None,
) -> list[str]:
    return sorted(
        (identity for identity in identities if remaining is None or remaining[identity] > 0),
        key=lambda identity: (
            -(remaining[identity] if remaining is not None else 0),
            uses[identity],
            stable_hash(*key, cell[0], cell[1], identity, "identity-tie"),
            identity,
        ),
    )


def select_batch_identities(
    identities_by_cell: dict[tuple[str, int], set[str]],
    identity_uses: dict[tuple[str, int], Counter[str]],
    key: tuple[object, ...],
    per_cell: int = 16,
    remaining_quotas: dict[tuple[str, int], Counter[str]] | None = None,
    remaining_batches: int | None = None,
    identity_tie_ranks: dict[str, str] | None = None,
    cell_tie_ranks: dict[tuple[str, tuple[str, int]], str] | None = None,
) -> dict[tuple[str, int], list[str]]:
    """Find one deterministic perfect 4-cell matching.

    Each of the 64 cell slots can consume an identity from its cell, while an
    identity has capacity one across the complete batch.  Candidate ordering
    makes lower-use identities primary; SHA-256 resolves all remaining ties.
    """

    if remaining_quotas is not None:
        if remaining_batches is None or remaining_batches < 1:
            raise ValueError("remaining_batches is required with frozen identity quotas")
        for cell in CELLS:
            total = sum(remaining_quotas[cell].values())
            if total != per_cell * remaining_batches:
                raise ValueError(
                    f"future-feasibility cell sum failed for {cell}: "
                    f"expected {per_cell * remaining_batches}, got {total}"
                )
        global_remaining: Counter[str] = Counter()
        for cell in CELLS:
            for identity, quota in remaining_quotas[cell].items():
                if quota < 0:
                    raise ValueError("negative residual identity quota")
                global_remaining[identity] += quota
        overloaded = [
            identity for identity, quota in global_remaining.items()
            if quota > remaining_batches
        ]
        if overloaded:
            identity = min(overloaded)
            raise ValueError(
                f"future-feasibility global identity bound failed: {identity} has "
                f"{global_remaining[identity]} residual uses for {remaining_batches} batches"
            )
        identity_rank = identity_tie_ranks or {
            identity: stable_hash(*key[:3], identity, "unit-identity-tie")
            for identity in global_remaining
        }
        cell_rank = cell_tie_ranks or {
            (identity, cell): stable_hash(
                *key[:3], identity, cell[0], cell[1], "unit-cell-tie"
            )
            for identity in global_remaining
            for cell in CELLS
        }

        # The four cells are direct capacity-16 nodes.  Match every saturated
        # identity first, then augment optional identities until all 64 places
        # are occupied.  An alternating path can move an existing identity only
        # to its other eligible cell; with four capacity nodes this is the same
        # integral b-matching as a full flow graph without per-batch graph
        # allocation or 64 indistinguishable slot nodes.
        saturated = sorted(
            (
                identity for identity, quota in global_remaining.items()
                if quota == remaining_batches
            ),
            key=lambda identity: (
                -sum(int(remaining_quotas[cell][identity] > 0) for cell in CELLS),
                identity_rank[identity],
                identity,
            ),
        )
        optional = sorted(
            (identity for identity, quota in global_remaining.items() if 0 < quota < remaining_batches),
            key=lambda identity: (
                -global_remaining[identity],
                sum(identity_uses[cell][identity] for cell in CELLS),
                identity_rank[identity],
                identity,
            ),
        )
        identities = saturated + optional
        eligible_cells: dict[str, list[tuple[str, int]]] = {}
        occupant_rank = identity_rank
        for identity in identities:
            eligible = [cell for cell in CELLS if remaining_quotas[cell][identity] > 0]
            eligible.sort(
                key=lambda cell: (
                    -remaining_quotas[cell][identity],
                    cell_rank[(identity, cell)],
                    cell,
                )
            )
            eligible_cells[identity] = eligible
        assigned_by_cell: dict[tuple[str, int], list[str]] = {cell: [] for cell in CELLS}
        identity_to_cell: dict[str, tuple[str, int]] = {}

        def augment(
            identity: str,
            seen_cells: set[tuple[str, int]],
            seen_identities: set[str],
        ) -> bool:
            for cell in eligible_cells[identity]:
                if cell in seen_cells:
                    continue
                seen_cells.add(cell)
                if len(assigned_by_cell[cell]) < per_cell:
                    assigned_by_cell[cell].append(identity)
                    identity_to_cell[identity] = cell
                    return True
                occupants = sorted(
                    assigned_by_cell[cell],
                    key=lambda member: (
                        len(eligible_cells[member]),
                        occupant_rank[member],
                        member,
                    ),
                )
                for previous in occupants:
                    if previous in seen_identities:
                        continue
                    seen_identities.add(previous)
                    if augment(previous, seen_cells, seen_identities):
                        assigned_by_cell[cell].remove(previous)
                        assigned_by_cell[cell].append(identity)
                        identity_to_cell[identity] = cell
                        return True
            return False

        for identity in saturated:
            if not augment(identity, set(), {identity}):
                raise ValueError(
                    f"mandatory saturated identity matching failed with "
                    f"R={remaining_batches}: {identity}"
                )
        for identity in optional:
            if len(identity_to_cell) == 4 * per_cell:
                break
            augment(identity, set(), {identity})
        if len(identity_to_cell) != 4 * per_cell:
            raise ValueError(
                f"future-feasible b-matching filled {len(identity_to_cell)} of {4 * per_cell} slots "
                f"with R={remaining_batches}"
            )
        selected_identities = set(identity_to_cell)
        if not set(saturated).issubset(selected_identities):
            raise AssertionError("a mandatory saturated identity left the b-matching")
        if len(selected_identities) != 4 * per_cell:
            raise AssertionError("identity b-matching cardinality drift")
        output = assigned_by_cell
        for cell in CELLS:
            output[cell].sort(
                key=lambda identity: (
                    -remaining_quotas[cell][identity],
                    cell_rank[(identity, cell)],
                    identity,
                )
            )
            if len(output[cell]) != per_cell:
                raise AssertionError("future-feasible b-matching cell degree drift")
        return output

    candidates = {
        cell: _candidate_order(
            cell,
            identities_by_cell[cell],
            identity_uses[cell],
            key,
            remaining_quotas[cell] if remaining_quotas is not None else None,
        )
        for cell in CELLS
    }
    if any(len(candidates[cell]) < per_cell for cell in CELLS):
        raise ValueError("one task/pseudo cell has fewer than 16 fit text identities")
    slots = [(cell, slot) for cell in CELLS for slot in range(per_cell)]
    slots.sort(key=lambda item: stable_hash(*key, item[0][0], item[0][1], item[1], "slot-order"))
    identity_to_slot: dict[str, tuple[tuple[str, int], int]] = {}
    slot_to_identity: dict[tuple[tuple[str, int], int], str] = {}

    def augment(slot: tuple[tuple[str, int], int], seen: set[str]) -> bool:
        cell = slot[0]
        for identity in candidates[cell]:
            if identity in seen:
                continue
            seen.add(identity)
            previous = identity_to_slot.get(identity)
            if previous is None or augment(previous, seen):
                identity_to_slot[identity] = slot
                slot_to_identity[slot] = identity
                return True
        return False

    for slot in slots:
        if not augment(slot, set()):
            raise ValueError("cannot form a globally unique 64-identity task/pseudo batch")
    output: dict[tuple[str, int], list[str]] = {cell: [] for cell in CELLS}
    for slot, identity in slot_to_identity.items():
        output[slot[0]].append(identity)
    for cell in CELLS:
        output[cell].sort(key=lambda identity: candidates[cell].index(identity))
        if len(output[cell]) != per_cell or len(set(output[cell])) != per_cell:
            raise AssertionError("cell matching cardinality drift")
    if len({identity for values in output.values() for identity in values}) != 4 * per_cell:
        raise AssertionError("global normalized-text identity repeated in a batch")
    return output


def _little_endian_bytes(values: array) -> bytes:
    clone = array("I", values)
    if clone.itemsize != 4:
        raise RuntimeError("platform unsigned-int width is not 32 bits")
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        clone.byteswap()
    return clone.tobytes()


def build_unit_schedule(
    catalog: list[dict[str, object]], outer_fold: int, seed: int, epochs: int,
    batches_per_epoch: int, domain: str,
) -> tuple[array, list[dict[str, object]]]:
    fit = [row for row in catalog if row["text_fold"] not in {outer_fold, (outer_fold + 1) % 5}]
    rows_by_cell_identity: dict[tuple[str, int], dict[str, list[int]]] = {
        cell: defaultdict(list) for cell in CELLS
    }
    row_by_index = {as_int(row["trial_index"], "trial_index"): row for row in catalog}
    for row in fit:
        cell = (str(row["reading_task"]), as_int(row["pseudo_group"], "pseudo_group"))
        rows_by_cell_identity[cell][str(row["normalized_text_sha256"])].append(
            as_int(row["trial_index"], "trial_index")
        )
    for cell in CELLS:
        for identity in rows_by_cell_identity[cell]:
            rows_by_cell_identity[cell][identity].sort(
                key=lambda index: (stable_hash(domain, outer_fold, seed, cell, identity, row_by_index[index]["trial_id"], "row-base"), index)
            )

    identity_uses = {cell: Counter({identity: 0 for identity in rows_by_cell_identity[cell]}) for cell in CELLS}
    all_identities = sorted({
        identity
        for cell in CELLS
        for identity in rows_by_cell_identity[cell]
    })
    identity_tie_ranks = {
        identity: stable_hash(domain, outer_fold, seed, identity, "unit-identity-tie")
        for identity in all_identities
    }
    cell_tie_ranks = {
        (identity, cell): stable_hash(
            domain, outer_fold, seed, identity, cell[0], cell[1], "unit-cell-tie"
        )
        for identity in all_identities
        for cell in CELLS
    }
    slots_per_cell = epochs * batches_per_epoch * 16
    remaining_quotas: dict[tuple[str, int], Counter[str]] = {}
    for cell in CELLS:
        identities = sorted(rows_by_cell_identity[cell])
        base, extras = divmod(slots_per_cell, len(identities))
        extra_order = sorted(
            identities,
            key=lambda identity: (
                stable_hash(domain, outer_fold, seed, cell[0], cell[1], identity, "identity-quota"),
                identity,
            ),
        )
        extra_set = set(extra_order[:extras])
        remaining_quotas[cell] = Counter({
            identity: base + int(identity in extra_set) for identity in identities
        })
        for identity in identities:
            if remaining_quotas[cell][identity] < len(rows_by_cell_identity[cell][identity]):
                raise ValueError(
                    f"full schedule cannot cover every row for {cell}/{identity}: "
                    f"quota={remaining_quotas[cell][identity]} rows={len(rows_by_cell_identity[cell][identity])}"
                )
    row_uses: Counter[int] = Counter({as_int(row["trial_index"], "trial_index"): 0 for row in fit})
    values = array("I")
    for epoch in range(1, epochs + 1):
        for batch_index in range(batches_per_epoch):
            key = (domain, outer_fold, seed, epoch, batch_index)
            selected = select_batch_identities(
                {cell: set(rows_by_cell_identity[cell]) for cell in CELLS},
                identity_uses,
                key,
                remaining_quotas=remaining_quotas,
                remaining_batches=(
                    epochs * batches_per_epoch
                    - ((epoch - 1) * batches_per_epoch + batch_index)
                ),
                identity_tie_ranks=identity_tie_ranks,
                cell_tie_ranks=cell_tie_ranks,
            )
            batch: list[int] = []
            for cell in CELLS:
                for identity in selected[cell]:
                    candidates = rows_by_cell_identity[cell][identity]
                    row_index = min(
                        candidates,
                        key=lambda index: (
                            row_uses[index],
                            stable_hash(*key, cell[0], cell[1], identity, row_by_index[index]["trial_id"], "row-tie"),
                            index,
                        ),
                    )
                    identity_uses[cell][identity] += 1
                    remaining_quotas[cell][identity] -= 1
                    row_uses[row_index] += 1
                    batch.append(row_index)
            if len(batch) != 64:
                raise AssertionError("batch cardinality drift")
            batch.sort(key=lambda index: (stable_hash(*key, row_by_index[index]["trial_id"], "batch-order"), index))
            if len({row_by_index[index]["normalized_text_sha256"] for index in batch}) != 64:
                raise AssertionError("batch repeated a normalized-text identity")
            counts = Counter((row_by_index[index]["reading_task"], row_by_index[index]["pseudo_group"]) for index in batch)
            if counts != Counter({cell: 16 for cell in CELLS}):
                raise AssertionError(f"batch cell balance drift: {counts}")
            values.extend(batch)

    expected = epochs * batches_per_epoch * 64
    if len(values) != expected:
        raise AssertionError("unit schedule length drift")
    if any(value != 0 for quotas in remaining_quotas.values() for value in quotas.values()):
        raise AssertionError("identity quotas were not exhausted exactly")
    audit: list[dict[str, object]] = []
    for cell in CELLS:
        identities = rows_by_cell_identity[cell]
        cell_identity_counts = identity_uses[cell]
        identity_min = min(cell_identity_counts.values())
        identity_max = max(cell_identity_counts.values())
        row_gap_max = 0
        covered = 0
        cell_rows = 0
        for identity, indices in identities.items():
            counts = [row_uses[index] for index in indices]
            row_gap_max = max(row_gap_max, max(counts) - min(counts))
            covered += sum(count > 0 for count in counts)
            cell_rows += len(indices)
        if identity_max - identity_min > 1:
            raise ValueError(
                f"identity exposure imbalance exceeds one: fold={outer_fold} seed={seed} cell={cell} "
                f"min={identity_min} max={identity_max}"
            )
        if row_gap_max > 1:
            raise ValueError(f"conditional row exposure imbalance exceeds one: fold={outer_fold} seed={seed} cell={cell}")
        if covered != cell_rows:
            raise ValueError(f"fit row coverage incomplete: fold={outer_fold} seed={seed} cell={cell}")
        audit.append({
            "outer_fold": outer_fold,
            "training_seed": seed,
            "reading_task": cell[0],
            "pseudo_group": cell[1],
            "fit_trial_rows": cell_rows,
            "fit_unique_text_identities": len(identities),
            "scheduled_slots": epochs * batches_per_epoch * 16,
            "identity_use_min": identity_min,
            "identity_use_max": identity_max,
            "identity_use_gap": identity_max - identity_min,
            "conditional_row_use_gap_max": row_gap_max,
            "covered_fit_rows": covered,
            "status": "pass",
        })
    return values, audit


def _global_batches_per_epoch(catalog: list[dict[str, object]], folds: Sequence[int]) -> tuple[int, dict[tuple[int, str, int], int]]:
    counts: dict[tuple[int, str, int], int] = {}
    for outer_fold in folds:
        fit = [row for row in catalog if row["text_fold"] not in {outer_fold, (outer_fold + 1) % len(folds)}]
        for cell in CELLS:
            count = sum(
                row["reading_task"] == cell[0] and row["pseudo_group"] == cell[1]
                for row in fit
            )
            counts[(outer_fold, cell[0], cell[1])] = count
    batches = max(math.ceil(value / 16) for value in counts.values())
    if batches < 1:
        raise ValueError("no schedule batches")
    return batches, counts


def freeze_assignments(
    assignments: list[dict[str, str]], output_root: Path, contract: dict[str, object],
    contract_sha256: str, parent_report_sha256: str,
    parent_verification_summary: dict[str, object] | None = None,
    *, progress: bool = False,
) -> dict[str, object]:
    schedule = contract["schedule"]
    assert isinstance(schedule, dict)
    folds = [as_int(value, "outer_fold") for value in schedule["outer_folds"]]
    seeds = [as_int(value, "training_seed") for value in schedule["training_seeds"]]
    epochs = as_int(schedule["epochs"], "epochs")
    catalog, _ = canonical_trial_catalog(assignments, folds)
    batches_per_epoch, fit_cell_counts = _global_batches_per_epoch(catalog, folds)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("schedule output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "trial_catalog.csv", CATALOG_FIELDS, catalog)

    binary_path = output_root / INDEX_NAME
    unit_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    unit_hashes_by_fold: dict[int, list[str]] = defaultdict(list)
    byte_offset = 0
    with binary_path.open("wb") as binary:
        unit_index = 0
        for outer_fold in folds:
            for seed in seeds:
                values, audits = build_unit_schedule(
                    catalog, outer_fold, seed, epochs, batches_per_epoch, str(schedule["domain"])
                )
                payload = _little_endian_bytes(values)
                unit_sha = hashlib.sha256(payload).hexdigest()
                binary.write(payload)
                binary.flush()
                unit_hashes_by_fold[outer_fold].append(unit_sha)
                unit_rows.append({
                    "unit_index": unit_index,
                    "outer_fold": outer_fold,
                    "training_seed": seed,
                    "epochs": epochs,
                    "batches_per_epoch": batches_per_epoch,
                    "batch_size": 64,
                    "uint32_count": len(values),
                    "byte_offset": byte_offset,
                    "byte_length": len(payload),
                    "unit_sha256": unit_sha,
                    "initialization_seed": initialization_seed(outer_fold, seed),
                })
                audit_rows.extend(audits)
                byte_offset += len(payload)
                if progress:
                    print(json.dumps({
                        "schedule_unit_complete": unit_index,
                        "outer_fold": outer_fold,
                        "training_seed": seed,
                        "unit_sha256": unit_sha,
                        "byte_length": len(payload),
                    }, sort_keys=True), flush=True)
                unit_index += 1
    if any(len(set(hashes)) != len(seeds) for hashes in unit_hashes_by_fold.values()):
        raise ValueError("seed-specific schedules are unexpectedly byte-identical")
    write_csv(
        output_root / "schedule_units.csv",
        (
            "unit_index", "outer_fold", "training_seed", "epochs",
            "batches_per_epoch", "batch_size", "uint32_count", "byte_offset",
            "byte_length", "unit_sha256",
            "initialization_seed",
        ),
        unit_rows,
    )
    write_csv(
        output_root / "schedule_audit.csv",
        (
            "outer_fold", "training_seed", "reading_task", "pseudo_group",
            "fit_trial_rows", "fit_unique_text_identities", "scheduled_slots",
            "identity_use_min", "identity_use_max", "identity_use_gap",
            "conditional_row_use_gap_max", "covered_fit_rows", "status",
        ),
        audit_rows,
    )

    core_hashes = {
        name: sha256(output_root / name)
        for name in ("trial_catalog.csv", INDEX_NAME, "schedule_units.csv", "schedule_audit.csv")
    }
    manifest = {
        "status": "pass",
        "schema_version": 1,
        "schedule_contract_sha256": contract_sha256,
        "parent_protocol_report_sha256": parent_report_sha256,
        "parent_preserved_dataset_slug": contract["parent"]["preserved_dataset_slug"],
        "parent_preserved_dataset_version": contract["parent"]["preserved_dataset_version"],
        "parent_protocol_artifact_source_id": contract["parent"]["preserved_protocol_artifact_source_id"],
        "upstream_input_source_id": contract["parent"]["upstream_input_source_id"],
        "parent_clean_remount_verification": (
            parent_verification_summary
            if parent_verification_summary is not None
            else {"status": "synthetic_test_fixture"}
        ),
        "shape": [len(folds) * len(seeds), epochs, batches_per_epoch, 64],
        "dtype": "<u4",
        "order": "C",
        "unit_order": [
            {
                "unit_index": row["unit_index"],
                "outer_fold": row["outer_fold"],
                "training_seed": row["training_seed"],
                "unit_sha256": row["unit_sha256"],
                "initialization_seed": row["initialization_seed"],
            }
            for row in unit_rows
        ],
        "catalog_rows": len(catalog),
        "global_batches_per_epoch": batches_per_epoch,
        "fit_cell_trial_rows": {
            f"{fold}::{task}::{pseudo}": value
            for (fold, task, pseudo), value in sorted(fit_cell_counts.items())
        },
        "applicable_arms": [
            "global_mixed", "true_task_segmented", "pseudo_task_segmented"
        ],
        "arm_axis_absent_from_binary": True,
        "binary_encoding": "raw C-order little-endian unsigned 32-bit catalog indices",
        "randomness": {
            "initialization_seed_derivation": contract["randomness"]["initialization_seed"],
            "global_mask_seed": contract["randomness"]["global_mask_seed"],
            "global_mask_key_derivation": contract["randomness"]["global_mask_key"],
        },
        "same_schedule_across_arms": True,
        "bounded_smoke_authorized": True,
        "full_training_authorized": False,
        "held_out_test_accessed": False,
        "artifact_sha256": core_hashes,
    }
    write_json(output_root / MANIFEST_NAME, manifest)
    artifact_hashes = {name: sha256(output_root / name) for name in SCHEDULE_ARTIFACTS}
    report = {
        "status": "pass",
        "schema_version": 1,
        "schedule_contract_sha256": contract_sha256,
        "parent_protocol_report_sha256": parent_report_sha256,
        "counts": {
            "catalog_rows": len(catalog),
            "schedule_units": len(unit_rows),
            "epochs_per_unit": epochs,
            "global_batches_per_epoch": batches_per_epoch,
            "batch_size": 64,
            "scheduled_uint32_indices": byte_offset // 4,
        },
        "checks": {
            "parent_protocol_clean_remount_verified": parent_verification_summary is not None,
            "fit_rows_only": True,
            "exact_16_per_task_pseudo_cell": True,
            "globally_unique_normalized_text_per_batch": True,
            "identity_exposure_gap_at_most_one": True,
            "conditional_row_exposure_gap_at_most_one": True,
            "every_fit_row_covered": True,
            "same_schedule_across_arms": True,
            "arm_axis_absent_from_schedule": True,
            "schedule_differs_by_seed": True,
            "vector_array_or_model_score_loaded": False,
            "official_validation_used": False,
            "held_out_test_accessed": False,
        },
        "authorization": {
            "bounded_smoke_authorized": True,
            "full_training_authorized": False,
            "next_required_artifact": "bounded all-three-arm real-data smoke with exact schedule and paired initialization hashes",
        },
        "artifact_sha256": artifact_hashes,
    }
    write_json(output_root / REPORT_NAME, report)
    return report


def freeze(protocol_root: Path, output_root: Path, contract_path: Path) -> dict[str, object]:
    contract = load_schedule_contract(contract_path)
    try:
        from evaluation.verify_task_segmented_protocol_artifact import verify as artifact_verify
    except ModuleNotFoundError:  # direct script execution from evaluation/
        from verify_task_segmented_protocol_artifact import verify as artifact_verify
    parent_evidence = verify_parent(protocol_root, contract, artifact_verifier=artifact_verify)
    return freeze_assignments(
        read_csv(protocol_root / "outer_split_assignments.csv"),
        output_root,
        contract,
        sha256(contract_path),
        sha256(protocol_root / PARENT_REPORT_NAME),
        parent_evidence["clean_remount_summary"],
        progress=True,
    )


def verify_frozen_output(output_root: Path) -> dict[str, object]:
    report = read_json(output_root / REPORT_NAME)
    if report.get("status") != "pass":
        raise ValueError("schedule report is not PASS")
    expected = report.get("artifact_sha256")
    if not isinstance(expected, dict) or set(expected) != set(SCHEDULE_ARTIFACTS):
        raise ValueError("schedule report has an invalid artifact inventory")
    for name in SCHEDULE_ARTIFACTS:
        if sha256(output_root / name) != expected[name]:
            raise ValueError(f"schedule artifact hash mismatch: {name}")
    manifest = read_json(output_root / MANIFEST_NAME)
    shape = manifest.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError("schedule manifest shape is invalid")
    expected_bytes = math.prod(as_int(value, "shape") for value in shape) * 4
    if (output_root / INDEX_NAME).stat().st_size != expected_bytes:
        raise ValueError("schedule binary byte length differs from manifest shape")
    if manifest.get("full_training_authorized") is not False:
        raise ValueError("schedule output unexpectedly authorizes full training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name(CONTRACT_NAME))
    parser.add_argument("--verify-output-only", action="store_true")
    args = parser.parse_args()
    if args.verify_output_only:
        report = verify_frozen_output(args.output_root)
    else:
        if args.protocol_root is None:
            parser.error("--protocol-root is required unless --verify-output-only is used")
        report = freeze(args.protocol_root, args.output_root, args.contract)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TASK-SEGMENTED TRAINING SCHEDULE FREEZE: PASS")


if __name__ == "__main__":
    main()
