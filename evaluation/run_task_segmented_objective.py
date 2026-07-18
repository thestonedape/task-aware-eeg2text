"""Run the only authorized P4b task-segmented-objective training smoke.

This entry point is intentionally narrow.  It revalidates the three clean
Kaggle remounts, performs an exact metadata join, decodes only fold 0 / seed
20260717 / epoch 1 / batches 0 and 1, and runs those same two batches through
all three frozen arms.  It cannot run a full fit, evaluate a checkpoint, read
validation/test vectors, or make a scientific decision.

NumPy, PyTorch, and the task objective are imported only after authorization
and all standard-library provenance checks have passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXECUTION_CONTRACT_PATH = Path(__file__).with_name(
    "task_segmented_smoke_execution_contract.json"
)
RUNNER_PATH = Path(__file__)

ARMS = (
    "global_mixed",
    "true_task_segmented",
    "pseudo_task_segmented",
)
MODE = "bounded_smoke"
PROMPT_NEUTRAL_SOURCE_ID = (
    "kaggle-dataset-thestonedape-task-aware-eegtotext-version-2"
)
PROTOCOL_SOURCE_ID = (
    "kaggle-dataset-thestonedape-task-aware-eeg2text-"
    "task-segmented-protocol-version-1"
)
SCHEDULE_SOURCE_ID = (
    "kaggle-dataset-thestonedape-task-aware-eeg2text-"
    "task-segmented-schedule-version-1"
)

CATALOG_FIELDS = (
    "trial_index", "trial_id", "text_fold", "dataset_version", "reading_task",
    "subject_id", "normalized_text_sha256", "text_target_id", "pseudo_group",
    "length_words_whitespace_v1", "eeg_vector_file", "eeg_vector_offset",
    "eeg_vector_dim",
)
UNIT_FIELDS = (
    "unit_index", "outer_fold", "training_seed", "epochs",
    "batches_per_epoch", "batch_size", "uint32_count", "byte_offset",
    "byte_length", "unit_sha256", "initialization_seed",
)
ASSIGNMENT_FIELDS = (
    "outer_fold", "role", "text_fold", "trial_id", "dataset_version",
    "reading_task", "subject_id", "normalized_text_sha256", "text_target_id",
    "pseudo_group", "length_words_whitespace_v1", "eeg_vector_file",
    "eeg_vector_offset", "eeg_vector_dim",
)
EEG_INDEX_FIELDS = (
    "condition", "phase", "target_trial_id", "signal_trial_id",
    "target_source_dataframe_row_index", "signal_source_dataframe_row_index",
    "dataset_version", "reading_task", "subject_id", "text_uid", "vector_file",
    "vector_offset", "vector_dim", "prompt_mode", "checkpoint_sha256",
    "source_index_sha256",
)
TEXT_INDEX_FIELDS = (
    "text_target_id", "normalized_text_sha256", "representative_trial_id",
    "representative_text", "vector_file", "vector_offset", "vector_dim",
    "checkpoint_sha256", "source_index_sha256",
)
TEXT_MAPPING_FIELDS = (
    "trial_id", "split", "cohort", "dataset_version", "reading_task",
    "subject_id", "source_dataframe_row_index", "text_target_id",
    "normalized_text_sha256",
)
ARM_SUMMARY_FIELDS = (
    "arm_id", "status", "optimizer_steps", "initial_state_sha256",
    "final_state_sha256", "final_optimizer_state_sha256",
)
COMMON_BATCH_FIELDS = (
    "outer_fold", "training_seed", "epoch", "batch_index", "batch_position",
    "catalog_index", "trial_id", "subject_id", "reading_task",
    "pseudo_task_id", "normalized_text_sha256", "text_target_id",
    "eeg_vector_file", "eeg_vector_offset", "text_vector_file",
    "text_vector_offset",
)
STEP_TRACE_FIELDS = (
    "outer_fold", "training_seed", "arm_id", "epoch", "batch_index",
    "global_step", "schedule_unit_sha256", "batch_catalog_indices_sha256",
    "batch_trial_ids_sha256", "global_mask_key", "eeg_to_text_mask_sha256",
    "text_to_eeg_mask_sha256", "initial_state_sha256",
    "pre_step_state_sha256", "loss", "gradient_norm_preclip",
    "post_step_state_sha256", "optimizer_state_sha256",
)


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"unexpected CSV header: {path.name}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"extra CSV columns: {path.name}")
    return rows


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _atomic_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    _atomic_bytes(path, _csv_bytes(fields, rows))


def _as_int(value: object, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in {field}: {value!r}") from exc


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_execution_contract(path: Path = EXECUTION_CONTRACT_PATH) -> dict[str, Any]:
    contract = _json(path)
    validate_execution_contract(contract)
    return contract


def validate_execution_contract(contract: Mapping[str, Any]) -> None:
    """Reject any drift that could widen the bounded smoke."""

    if contract.get("schema_version") != 1:
        raise ValueError("execution contract schema must be 1")
    if contract.get("status") != "frozen_before_bounded_p4b_smoke":
        raise ValueError("execution contract is not prospectively frozen")
    bounded = contract.get("bounded_smoke")
    expected_bounded = {
        "outer_fold": 0,
        "training_seed": 20260717,
        "epoch": 1,
        "zero_based_batch_indices": [0, 1],
        "arms": list(ARMS),
        "optimizer_steps_per_arm": 2,
        "total_optimizer_steps": 6,
        "batch_size": 64,
        "examples_per_task_pseudo_cell": 16,
        "trainable_parameters": 196608,
        "same_catalog_indices_across_arms": True,
        "byte_identical_initial_state_across_arms": True,
    }
    if bounded != expected_bounded:
        raise ValueError("bounded-smoke definition drifted")
    authorization = contract.get("authorization")
    expected_authorization = {
        "bounded_smoke_authorized": True,
        "full_training_authorized": False,
        "checkpoint_or_confirmation_evaluation_authorized": False,
        "scientific_decision_permitted": False,
        "held_out_test_accessed": False,
        "next_gate_after_pass": (
            "independently verify and preserve the smoke artifact before any "
            "request to authorize the 45 cross-fitted fits"
        ),
    }
    if authorization != expected_authorization:
        raise ValueError("execution authorization drifted")
    execution = contract.get("execution")
    exact_execution = {
        "vector_dtype": "float32", "automatic_mixed_precision": False,
        "optimizer": "AdamW", "learning_rate": 0.001,
        "weight_decay": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8,
        "amsgrad": False, "maximize": False, "capturable": False,
        "differentiable": False, "foreach": False, "fused": False,
        "scheduler": "none", "gradient_clip_norm": 1.0,
        "gradient_clip_norm_type": 2.0,
        "gradient_clip_error_if_nonfinite": True,
        "resume_scope": "after either completed optimizer step within each arm",
        "completion_reuse": (
            "only when the binding and every declared artifact hash revalidate"
        ),
    }
    if execution != exact_execution:
        raise ValueError("execution hyperparameters drifted")
    inputs = contract.get("immutable_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("immutable input bindings are absent")
    expected_sources = {
        "prompt_neutral_vectors": PROMPT_NEUTRAL_SOURCE_ID,
        "task_segmented_protocol": PROTOCOL_SOURCE_ID,
        "training_schedule": SCHEDULE_SOURCE_ID,
    }
    for name, source_id in expected_sources.items():
        binding = inputs.get(name)
        if not isinstance(binding, dict) or binding.get("preserved_source_id") != source_id:
            raise ValueError(f"immutable source binding drifted: {name}")
        for key, value in binding.items():
            if key.endswith("sha256") and not _is_sha256(value):
                raise ValueError(f"invalid immutable digest: {name}.{key}")
    output = contract.get("output")
    if not isinstance(output, dict):
        raise ValueError("output contract is absent")
    if output.get("top_level_files") != [
        "arm_summary.csv", "common_batch_trace.csv", "smoke_run_metadata.json",
        "task_segmented_smoke_manifest.json",
    ]:
        raise ValueError("top-level output inventory drifted")
    if output.get("per_arm_files") != [
        "resume_checkpoint.pt", "run_summary.json", "step_trace.csv",
    ]:
        raise ValueError("per-arm output inventory drifted")
    if output.get("held_out_test_accessed") is not False:
        raise ValueError("output contract reports held-out-test access")


def authorize_mode(mode: str, contract: Mapping[str, Any]) -> None:
    """Hard-deny every operation except the prospectively bounded smoke."""

    validate_execution_contract(contract)
    if mode != MODE:
        raise PermissionError(
            "only bounded_smoke is authorized; full training, evaluation, and "
            "scientific decisions are hard-disabled"
        )
    authorization = contract["authorization"]
    if authorization["bounded_smoke_authorized"] is not True:
        raise PermissionError("bounded smoke is not authorized")
    if authorization["full_training_authorized"] is not False:
        raise PermissionError("invalid contract: full training became authorized")
    if authorization["scientific_decision_permitted"] is not False:
        raise PermissionError("invalid contract: scientific decision became permitted")


def validate_project_commit(project_commit: str) -> None:
    if (
        len(project_commit) != 40
        or project_commit != project_commit.lower()
        or any(character not in "0123456789abcdef" for character in project_commit)
    ):
        raise ValueError("project commit must be a full lowercase 40-character Git SHA")


def decode_u32le_slice(path: Path, byte_offset: int, count: int) -> list[int]:
    """Decode an exact bounded slice of a raw little-endian uint32 file."""

    if byte_offset < 0 or byte_offset % 4 or count < 0:
        raise ValueError("invalid uint32 byte offset/count")
    byte_count = count * 4
    size = path.stat().st_size
    if byte_offset + byte_count > size:
        raise ValueError("uint32 slice exceeds file bounds")
    with path.open("rb") as handle:
        handle.seek(byte_offset)
        payload = handle.read(byte_count)
    if len(payload) != byte_count:
        raise ValueError("short uint32 read")
    return [value[0] for value in struct.iter_unpack("<I", payload)]


def initialization_seed(outer_fold: int, training_seed: int) -> int:
    return int(
        stable_hash("p4b-adapter-init-v1", outer_fold, training_seed)[:16], 16
    ) & ((1 << 63) - 1)


def global_mask_key(
    schedule_contract_sha256: str,
    schedule_unit_sha256: str,
    outer_fold: int,
    training_seed: int,
    epoch: int,
    batch_index: int,
) -> str:
    if not _is_sha256(schedule_contract_sha256) or not _is_sha256(
        schedule_unit_sha256
    ):
        raise ValueError("mask-key inputs must be lowercase SHA-256 digests")
    return stable_hash(
        "p4b-global-mask-key-v1", 2026071806, schedule_contract_sha256,
        schedule_unit_sha256, outer_fold, training_seed, epoch, batch_index,
    )


def _verify_clean_remounts(
    vector_root: Path, protocol_root: Path, schedule_root: Path,
) -> dict[str, dict[str, Any]]:
    from evaluation.verify_prompt_neutral_pilot_inputs import verify as verify_vectors
    from evaluation.verify_task_segmented_protocol_artifact import (
        verify as verify_protocol,
    )
    from evaluation.verify_task_segmented_training_schedule_artifact import (
        verify as verify_schedule,
    )

    return {
        "prompt_neutral_vectors": verify_vectors(
            vector_root, PROMPT_NEUTRAL_SOURCE_ID
        ),
        "task_segmented_protocol": verify_protocol(
            protocol_root, PROTOCOL_SOURCE_ID
        ),
        "training_schedule": verify_schedule(
            schedule_root, SCHEDULE_SOURCE_ID
        ),
    }


def _validate_verification_bindings(
    reports: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]
) -> None:
    inputs = contract["immutable_inputs"]
    vectors = reports["prompt_neutral_vectors"]
    expected = inputs["prompt_neutral_vectors"]
    checks = {
        "preserved_source_id": vectors.get("preserved_source_id"),
        "combined_manifest_sha256": vectors.get("combined_manifest_sha256"),
        "eeg_vector_index_sha256": vectors.get("eeg", {}).get(
            "vector_index_sha256"
        ),
        "text_vector_index_sha256": vectors.get("text", {}).get(
            "text_vector_index_sha256"
        ),
        "trial_text_targets_sha256": vectors.get("text", {}).get(
            "trial_text_targets_sha256"
        ),
    }
    if checks != expected:
        raise ValueError("prompt-neutral verification binding differs from contract")
    protocol = reports["task_segmented_protocol"]
    expected = inputs["task_segmented_protocol"]
    checks = {
        "preserved_source_id": protocol.get("preserved_source_id"),
        "contract_sha256": protocol.get("contract_sha256"),
        "report_sha256": protocol.get("protocol_report_sha256"),
    }
    if checks != expected:
        raise ValueError("protocol verification binding differs from contract")
    schedule = reports["training_schedule"]
    expected = inputs["training_schedule"]
    checks = {
        "preserved_source_id": schedule.get("preserved_source_id"),
        "contract_sha256": schedule.get("schedule_contract_sha256"),
        "manifest_sha256": schedule.get("schedule_manifest_sha256"),
        "report_sha256": schedule.get("schedule_report_sha256"),
        "indices_sha256": schedule.get("verified_artifact_sha256", {}).get(
            "schedule_indices.u32le"
        ),
        "trial_catalog_sha256": schedule.get(
            "verified_artifact_sha256", {}
        ).get("trial_catalog.csv"),
        "shape": schedule.get("shape"),
    }
    if checks != expected:
        raise ValueError("schedule verification binding differs from contract")
    if vectors.get("checks", {}).get("held_out_test_accessed") is not False:
        raise ValueError("prompt-neutral verifier reported held-out-test access")
    if any(
        reports[name].get("held_out_test_accessed") is not False
        for name in ("task_segmented_protocol", "training_schedule")
    ):
        raise ValueError("an input verifier reported held-out-test access")


def _unique(rows: Sequence[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        identity = row[key]
        if identity in output:
            raise ValueError(f"duplicate {label}: {identity}")
        output[identity] = row
    return output


def _decode_smoke_indices(
    schedule_root: Path, contract: Mapping[str, Any]
) -> tuple[list[list[int]], dict[str, str]]:
    bounded = contract["bounded_smoke"]
    units = _csv(schedule_root / "schedule_units.csv", UNIT_FIELDS)
    matches = [
        row for row in units
        if _as_int(row["outer_fold"], "outer_fold") == bounded["outer_fold"]
        and _as_int(row["training_seed"], "training_seed")
        == bounded["training_seed"]
    ]
    if len(matches) != 1:
        raise ValueError("bounded schedule unit is not unique")
    unit = matches[0]
    if _as_int(unit["initialization_seed"], "initialization_seed") != initialization_seed(
        bounded["outer_fold"], bounded["training_seed"]
    ):
        raise ValueError("schedule initialization seed drifted")
    batch_size = bounded["batch_size"]
    batches_per_epoch = _as_int(unit["batches_per_epoch"], "batches_per_epoch")
    unit_offset = _as_int(unit["byte_offset"], "byte_offset")
    output: list[list[int]] = []
    for batch_index in bounded["zero_based_batch_indices"]:
        element_offset = (
            ((bounded["epoch"] - 1) * batches_per_epoch + batch_index)
            * batch_size
        )
        output.append(decode_u32le_slice(
            schedule_root / "schedule_indices.u32le",
            unit_offset + element_offset * 4,
            batch_size,
        ))
    return output, unit


def _strict_join(
    vector_root: Path,
    protocol_root: Path,
    schedule_root: Path,
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Join every 9011-row catalog entry before selecting the fixed 128 rows."""

    catalog = _csv(schedule_root / "trial_catalog.csv", CATALOG_FIELDS)
    if len(catalog) != 9011:
        raise ValueError("schedule catalog must contain exactly 9011 rows")
    for expected_index, row in enumerate(catalog):
        if _as_int(row["trial_index"], "trial_index") != expected_index:
            raise ValueError("catalog trial indices are not contiguous and ordered")

    assignments = _csv(
        protocol_root / "outer_split_assignments.csv", ASSIGNMENT_FIELDS
    )
    fold = contract["bounded_smoke"]["outer_fold"]
    fold_assignments = [
        row for row in assignments if _as_int(row["outer_fold"], "outer_fold") == fold
    ]
    assignment_by_trial = _unique(fold_assignments, "trial_id", "fold assignment")
    if len(assignment_by_trial) != len(catalog):
        raise ValueError("protocol/schedule catalog row cardinality mismatch")

    eeg_rows = _csv(vector_root / "eeg" / "vector_index.csv", EEG_INDEX_FIELDS)
    correct_train = [row for row in eeg_rows if row["condition"] == "correct_train"]
    eeg_by_trial = _unique(correct_train, "target_trial_id", "correct_train EEG trial")
    mapping_rows = _csv(
        vector_root / "text" / "trial_text_targets.csv", TEXT_MAPPING_FIELDS
    )
    mapping_by_trial = _unique(mapping_rows, "trial_id", "trial-text mapping")
    text_rows = _csv(
        vector_root / "text" / "text_vector_index.csv", TEXT_INDEX_FIELDS
    )
    text_by_target = _unique(text_rows, "text_target_id", "text target")

    invariant_fields = (
        "text_fold", "dataset_version", "reading_task", "subject_id",
        "normalized_text_sha256", "text_target_id", "pseudo_group",
        "length_words_whitespace_v1", "eeg_vector_file", "eeg_vector_offset",
        "eeg_vector_dim",
    )
    decoded, unit = _decode_smoke_indices(schedule_root, contract)
    scheduled_indices = {index for batch in decoded for index in batch}

    joined: list[dict[str, Any]] = []
    for row in catalog:
        trial_id = row["trial_id"]
        assignment = assignment_by_trial.get(trial_id)
        eeg = eeg_by_trial.get(trial_id)
        mapping = mapping_by_trial.get(trial_id)
        if assignment is None or eeg is None or mapping is None:
            raise ValueError(f"catalog trial does not resolve in every source: {trial_id}")
        if any(row[field] != assignment[field] for field in invariant_fields):
            raise ValueError(f"protocol/schedule metadata mismatch: {trial_id}")
        expected_role = "fit" if row["text_fold"] not in {"0", "1"} else (
            "confirmation" if row["text_fold"] == "0" else "checkpoint"
        )
        if assignment["role"] != expected_role:
            raise ValueError(f"outer-fold role mismatch: {trial_id}")
        if assignment["role"] != "fit" and _as_int(
            row["trial_index"], "trial_index"
        ) in scheduled_indices:
            raise ValueError("bounded schedule addressed checkpoint/confirmation row")
        if not (
            eeg["phase"] == "train"
            and eeg["prompt_mode"] == "all_masked"
            and eeg["signal_trial_id"] == trial_id
            and eeg["dataset_version"] == row["dataset_version"]
            and eeg["reading_task"] == row["reading_task"]
            and eeg["subject_id"] == row["subject_id"]
            and eeg["vector_file"] == row["eeg_vector_file"]
            and eeg["vector_offset"] == row["eeg_vector_offset"]
            and eeg["vector_dim"] == row["eeg_vector_dim"] == "1024"
        ):
            raise ValueError(f"EEG catalog binding mismatch: {trial_id}")
        if not (
            mapping["split"] == "train"
            and mapping["cohort"] == "primary_zuco2_nr_tsr"
            and mapping["dataset_version"] == row["dataset_version"] == "ZuCo2"
            and mapping["reading_task"] == row["reading_task"]
            and mapping["subject_id"] == row["subject_id"]
            and mapping["text_target_id"] == row["text_target_id"]
            and mapping["normalized_text_sha256"] == row["normalized_text_sha256"]
        ):
            raise ValueError(f"text mapping binding mismatch: {trial_id}")
        text = text_by_target.get(row["text_target_id"])
        if text is None or not (
            text["normalized_text_sha256"] == row["normalized_text_sha256"]
            and text["vector_dim"] == "1024"
        ):
            raise ValueError(f"text vector binding mismatch: {trial_id}")
        joined.append({"catalog": row, "eeg": eeg, "text": text})

    selected: list[dict[str, Any]] = []
    for step_index, indices in enumerate(decoded):
        if len(indices) != 64 or len(set(indices)) != 64:
            raise ValueError("bounded batch must contain 64 distinct catalog indices")
        if any(index < 0 or index >= len(joined) for index in indices):
            raise ValueError("bounded schedule contains an out-of-range catalog index")
        rows = [joined[index] for index in indices]
        if len({item["catalog"]["trial_id"] for item in rows}) != 64:
            raise ValueError("bounded batch repeats a trial identity")
        if len({item["catalog"]["text_target_id"] for item in rows}) != 64:
            raise ValueError("bounded batch repeats a normalized-text identity")
        cells = Counter(
            (item["catalog"]["reading_task"], item["catalog"]["pseudo_group"])
            for item in rows
        )
        if cells != Counter({("NR", "0"): 16, ("NR", "1"): 16,
                             ("TSR", "0"): 16, ("TSR", "1"): 16}):
            raise ValueError(f"bounded batch cell balance drifted: {cells}")
        if any(item["catalog"]["text_fold"] in {"0", "1"} for item in rows):
            raise ValueError("bounded batch contains non-fit text fold")
        selected.append({
            "step_index": step_index,
            "batch_index": contract["bounded_smoke"]["zero_based_batch_indices"][step_index],
            "indices": indices,
            "rows": rows,
        })
    return selected, unit


def _safe_vector_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe vector path: {relative!r}")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"vector path escapes its root: {relative!r}") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"vector path is not a regular file: {relative!r}")
    return resolved


def _load_vectors(root: Path, specs: Sequence[Mapping[str, str]]) -> Any:
    import numpy as np

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, spec in enumerate(specs):
        grouped[spec["vector_file"]].append(
            (position, _as_int(spec["vector_offset"], "vector_offset"))
        )
    output: list[Any | None] = [None] * len(specs)
    for relative, members in sorted(grouped.items()):
        path = _safe_vector_path(root, relative)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"vectors"}:
                raise ValueError(f"unexpected NPZ members: {relative}")
            vectors = archive["vectors"]
            for position, offset in members:
                if offset < 0 or offset >= len(vectors):
                    raise ValueError(f"vector offset out of range: {relative}:{offset}")
                vector = np.asarray(vectors[offset], dtype=np.float32)
                if vector.shape != (1024,) or not np.isfinite(vector).all():
                    raise ValueError(f"invalid vector: {relative}:{offset}")
                output[position] = vector.copy()
    if any(vector is None for vector in output):
        raise AssertionError("vector loader left an unresolved position")
    matrix = np.stack(output, axis=0)
    if matrix.dtype != np.float32 or matrix.shape != (len(specs), 1024):
        raise AssertionError("vector loader changed frozen shape/dtype")
    return matrix


def _materialize_batches(
    vector_root: Path,
    selected: Sequence[Mapping[str, Any]],
    unit: Mapping[str, str],
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    batches: list[dict[str, Any]] = []
    traces: list[dict[str, object]] = []
    schedule_contract_sha = contract["immutable_inputs"]["training_schedule"][
        "contract_sha256"
    ]
    for selected_batch in selected:
        catalog_rows = [item["catalog"] for item in selected_batch["rows"]]
        eeg_specs = [item["eeg"] for item in selected_batch["rows"]]
        text_specs = [item["text"] for item in selected_batch["rows"]]
        eeg = _load_vectors(vector_root / "eeg", eeg_specs)
        text = _load_vectors(vector_root / "text", text_specs)
        indices = selected_batch["indices"]
        true_groups = [row["reading_task"] for row in catalog_rows]
        pseudo_groups = [row["pseudo_group"] for row in catalog_rows]
        batch_index = selected_batch["batch_index"]
        mask_key = global_mask_key(
            schedule_contract_sha, unit["unit_sha256"], 0, 20260717, 1,
            batch_index,
        )
        indices_sha = hashlib.sha256(
            b"".join(struct.pack("<I", index) for index in indices)
        ).hexdigest()
        identity_hash = lambda values: hashlib.sha256(  # noqa: E731
            ("\n".join(values) + "\n").encode("utf-8")
        ).hexdigest()
        trial_ids_sha = identity_hash([row["trial_id"] for row in catalog_rows])
        for position, (catalog_index, item) in enumerate(
            zip(indices, selected_batch["rows"])
        ):
            row = item["catalog"]
            traces.append({
                "outer_fold": 0, "training_seed": 20260717, "epoch": 1,
                "batch_index": batch_index, "batch_position": position,
                "catalog_index": catalog_index, "trial_id": row["trial_id"],
                "subject_id": row["subject_id"],
                "reading_task": row["reading_task"],
                "pseudo_task_id": row["pseudo_group"],
                "normalized_text_sha256": row["normalized_text_sha256"],
                "text_target_id": row["text_target_id"],
                "eeg_vector_file": item["eeg"]["vector_file"],
                "eeg_vector_offset": item["eeg"]["vector_offset"],
                "text_vector_file": item["text"]["vector_file"],
                "text_vector_offset": item["text"]["vector_offset"],
            })
        batches.append({
            "step_index": selected_batch["step_index"], "epoch": 1,
            "batch_index": batch_index, "indices_sha256": indices_sha,
            "trial_ids_sha256": trial_ids_sha,
            "schedule_unit_sha256": unit["unit_sha256"],
            "global_mask_key": mask_key, "true_groups": true_groups,
            "pseudo_groups": pseudo_groups, "eeg": eeg, "text": text,
        })
    return batches, traces


def _set_determinism(seed: int) -> None:
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def canonical_model_state_sha256(model: Any) -> str:
    """Hash sorted name/dtype/shape/raw-CPU bytes for paired-state auditing."""

    state = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        state.update(name.encode("utf-8"))
        state.update(b"\x1f")
        state.update(str(value.dtype).encode("ascii"))
        state.update(b"\x1f")
        state.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        state.update(b"\x1f")
        state.update(value.numpy().tobytes(order="C"))
        state.update(b"\n")
    return state.hexdigest()


def canonical_optimizer_state_sha256(optimizer: Any) -> str:
    """Hash an optimizer state dict without relying on pickle byte stability."""

    state = hashlib.sha256()

    def update(value: Any) -> None:
        if hasattr(value, "detach") and hasattr(value, "dtype"):
            tensor = value.detach().cpu().contiguous()
            state.update(b"tensor\x1f")
            state.update(str(tensor.dtype).encode("ascii"))
            state.update(b"\x1f")
            state.update(
                json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            )
            state.update(b"\x1f")
            state.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(value, dict):
            state.update(b"dict{")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                update(key)
                update(value[key])
            state.update(b"}")
        elif isinstance(value, (list, tuple)):
            state.update(b"list[")
            for item in value:
                update(item)
            state.update(b"]")
        elif value is None:
            state.update(b"none")
        elif isinstance(value, bool):
            state.update(b"bool:1" if value else b"bool:0")
        elif isinstance(value, int):
            state.update(f"int:{value}".encode("ascii"))
        elif isinstance(value, float):
            state.update(b"float:")
            state.update(struct.pack("<d", value))
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
            state.update(f"str:{len(encoded)}:".encode("ascii"))
            state.update(encoded)
        else:
            raise TypeError(f"unsupported optimizer-state value: {type(value).__name__}")
        state.update(b"\x1e")

    update(optimizer.state_dict())
    return state.hexdigest()


def canonical_tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    state = hashlib.sha256()
    state.update(str(value.dtype).encode("ascii"))
    state.update(b"\x1f")
    state.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    state.update(b"\x1f")
    state.update(value.numpy().tobytes(order="C"))
    return state.hexdigest()


def _parameter_norm(model: Any) -> float:
    import torch

    squares = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        squares += parameter.detach().cpu().double().pow(2).sum()
    return float(squares.sqrt().item())


def _atomic_torch_save(path: Path, value: Mapping[str, object]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _optimizer(model: Any, contract: Mapping[str, Any]) -> Any:
    import torch

    execution = contract["execution"]
    return torch.optim.AdamW(
        model.parameters(), lr=execution["learning_rate"],
        betas=tuple(execution["betas"]), eps=execution["eps"],
        weight_decay=execution["weight_decay"], amsgrad=execution["amsgrad"],
        maximize=execution["maximize"], foreach=execution["foreach"],
        capturable=execution["capturable"],
        differentiable=execution["differentiable"], fused=execution["fused"],
    )


def _load_resume(
    arm_root: Path, arm_id: str, binding_sha256: str, model: Any,
    optimizer: Any, initial_state_sha256: str, device: Any,
) -> tuple[int, list[dict[str, str]], bool]:
    import torch

    checkpoint_path = arm_root / "resume_checkpoint.pt"
    trace_path = arm_root / "step_trace.csv"
    summary_path = arm_root / "run_summary.json"
    if summary_path.exists():
        raise ValueError("completed arm must be validated before resume loading")
    if checkpoint_path.exists() != trace_path.exists():
        raise ValueError(f"incomplete resume artifact inventory: {arm_id}")
    if not checkpoint_path.exists():
        return 0, [], False
    trace = _csv(trace_path, STEP_TRACE_FIELDS)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid resume checkpoint: {arm_id}")
    expected = {
        "schema_version": 1, "run_mode": MODE, "arm_id": arm_id,
        "binding_sha256": binding_sha256,
        "initial_state_sha256": initial_state_sha256,
        "completed_steps": len(trace), "step_trace_sha256": sha256(trace_path),
        "scientific_decision_permitted": False, "held_out_test_accessed": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint binding/tamper failure: {arm_id}.{key}")
    if len(trace) not in {1, 2}:
        raise ValueError(f"resume checkpoint has invalid step count: {arm_id}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if canonical_model_state_sha256(model) != checkpoint.get("model_state_sha256"):
        raise ValueError(f"resume model-state hash mismatch: {arm_id}")
    if canonical_optimizer_state_sha256(optimizer) != checkpoint.get(
        "optimizer_state_sha256"
    ):
        raise ValueError(f"resume optimizer-state hash mismatch: {arm_id}")
    return len(trace), trace, True


def _reuse_completed_arm(
    arm_root: Path, arm_id: str, binding_sha256: str
) -> dict[str, object] | None:
    summary_path = arm_root / "run_summary.json"
    if not summary_path.exists():
        return None
    expected_names = {"resume_checkpoint.pt", "run_summary.json", "step_trace.csv"}
    if {entry.name for entry in arm_root.iterdir()} != expected_names:
        raise ValueError(f"completed arm inventory mismatch: {arm_id}")
    summary = _json(summary_path)
    if not (
        summary.get("status") == "pass"
        and summary.get("schema_version") == 1
        and summary.get("run_mode") == MODE
        and summary.get("arm_id") == arm_id
        and summary.get("binding_sha256") == binding_sha256
        and summary.get("outer_fold") == 0
        and summary.get("training_seed") == 20260717
        and summary.get("epoch") == 1
        and summary.get("batch_indices") == [0, 1]
        and summary.get("optimizer_steps") == 2
        and summary.get("trainable_parameters") == 196608
        and summary.get("full_training_authorized") is False
        and summary.get("scientific_decision_permitted") is False
        and summary.get("held_out_test_accessed") is False
    ):
        raise ValueError(f"completed arm binding/tamper failure: {arm_id}")
    expected_hashes = {
        "resume_checkpoint.pt": sha256(arm_root / "resume_checkpoint.pt"),
        "step_trace.csv": sha256(arm_root / "step_trace.csv"),
    }
    if summary.get("artifact_sha256") != expected_hashes:
        raise ValueError(f"completed arm hash mismatch: {arm_id}")
    trace = _csv(arm_root / "step_trace.csv", STEP_TRACE_FIELDS)
    if len(trace) != 2:
        raise ValueError(f"completed arm trace length mismatch: {arm_id}")
    return {**summary, "resumed": True, "run_summary_sha256": sha256(summary_path)}


def _run_arm(
    arm_root: Path, arm_id: str, batches: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any], binding_sha256: str, init_seed: int,
    device: Any,
) -> dict[str, object]:
    import torch
    from project_adapters.task_segmented_objective import (
        SharedResidualAdapter, build_partition_masks,
        masked_symmetric_alignment_loss,
    )

    reused = _reuse_completed_arm(arm_root, arm_id, binding_sha256)
    if reused is not None:
        return reused
    arm_root.mkdir(parents=True, exist_ok=True)
    _set_determinism(init_seed)
    model = SharedResidualAdapter().to(device=device, dtype=torch.float32)
    if model.trainable_parameter_count != 196608:
        raise ValueError("shared adapter parameter count drifted")
    initial_hash = canonical_model_state_sha256(model)
    optimizer = _optimizer(model, contract)
    start, traces, resumed = _load_resume(
        arm_root, arm_id, binding_sha256, model, optimizer, initial_hash, device
    )
    execution = contract["execution"]
    model.train()
    for batch in batches[start:]:
        pre_step_state_sha256 = canonical_model_state_sha256(model)
        eeg = torch.as_tensor(batch["eeg"], dtype=torch.float32, device=device)
        text = torch.as_tensor(batch["text"], dtype=torch.float32, device=device)
        if eeg.shape != (64, 1024) or text.shape != (64, 1024):
            raise ValueError("training vectors must be float32 [64,1024]")
        if not bool(torch.isfinite(eeg).all() and torch.isfinite(text).all()):
            raise ValueError("training vectors contain a non-finite value")
        masks = build_partition_masks(
            arm_id, batch["true_groups"], batch["pseudo_groups"],
            key=batch["global_mask_key"], device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        adapted = model(eeg)
        loss = masked_symmetric_alignment_loss(adapted, text, *masks)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss: {arm_id}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=execution["gradient_clip_norm"],
            norm_type=execution["gradient_clip_norm_type"],
            error_if_nonfinite=execution["gradient_clip_error_if_nonfinite"],
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite gradient norm: {arm_id}")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite model parameter: {arm_id}")
        state_hash = canonical_model_state_sha256(model)
        optimizer_hash = canonical_optimizer_state_sha256(optimizer)
        trace = {
            "outer_fold": 0, "training_seed": 20260717, "arm_id": arm_id,
            "epoch": batch["epoch"], "batch_index": batch["batch_index"],
            "global_step": batch["step_index"] + 1,
            "schedule_unit_sha256": batch["schedule_unit_sha256"],
            "batch_catalog_indices_sha256": batch["indices_sha256"],
            "batch_trial_ids_sha256": batch["trial_ids_sha256"],
            "global_mask_key": batch["global_mask_key"],
            "eeg_to_text_mask_sha256": canonical_tensor_sha256(masks[0]),
            "text_to_eeg_mask_sha256": canonical_tensor_sha256(masks[1]),
            "initial_state_sha256": initial_hash,
            "pre_step_state_sha256": pre_step_state_sha256,
            "loss": format(float(loss.item()), ".17g"),
            "gradient_norm_preclip": format(float(gradient_norm.item()), ".17g"),
            "post_step_state_sha256": state_hash,
            "optimizer_state_sha256": optimizer_hash,
        }
        traces.append(trace)
        trace_path = arm_root / "step_trace.csv"
        _atomic_csv(trace_path, STEP_TRACE_FIELDS, traces)
        _atomic_torch_save(arm_root / "resume_checkpoint.pt", {
            "schema_version": 1, "run_mode": MODE, "arm_id": arm_id,
            "binding_sha256": binding_sha256, "initial_state_sha256": initial_hash,
            "model_state_sha256": state_hash, "completed_steps": len(traces),
            "optimizer_state_sha256": optimizer_hash,
            "step_trace_sha256": sha256(trace_path),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scientific_decision_permitted": False,
            "held_out_test_accessed": False,
        })
    if len(traces) != 2:
        raise ValueError(f"arm did not complete exactly two optimizer steps: {arm_id}")
    checkpoint_path = arm_root / "resume_checkpoint.pt"
    trace_path = arm_root / "step_trace.csv"
    summary = {
        "status": "pass", "schema_version": 1, "run_mode": MODE,
        "arm_id": arm_id, "binding_sha256": binding_sha256,
        "outer_fold": 0, "training_seed": 20260717, "epoch": 1,
        "batch_indices": [0, 1],
        "optimizer_steps": 2, "initial_state_sha256": initial_hash,
        "trainable_parameters": 196608,
        "final_state_sha256": canonical_model_state_sha256(model),
        "final_optimizer_state_sha256": canonical_optimizer_state_sha256(optimizer),
        "final_loss": float(traces[-1]["loss"]), "resumed": resumed,
        "artifact_sha256": {
            "resume_checkpoint.pt": sha256(checkpoint_path),
            "step_trace.csv": sha256(trace_path),
        },
        "full_training_authorized": False, "scientific_decision_permitted": False,
        "held_out_test_accessed": False,
    }
    _atomic_json(arm_root / "run_summary.json", summary)
    return {
        **summary,
        "run_summary_sha256": sha256(arm_root / "run_summary.json"),
    }


def _regular_file_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"output contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in exclude:
                output[relative] = sha256(path)
    return output


def _reuse_complete_output(
    output_root: Path, binding_sha256: str, execution_contract_sha256: str,
    project_commit: str,
) -> dict[str, Any] | None:
    manifest_path = output_root / "task_segmented_smoke_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _json(manifest_path)
    if not (
        manifest.get("status") == "pass"
        and manifest.get("schema_version") == 1
        and manifest.get("run_mode") == MODE
        and manifest.get("binding_sha256") == binding_sha256
        and manifest.get("execution_contract_sha256") == execution_contract_sha256
        and manifest.get("project_commit") == project_commit
        and manifest.get("runner_source_sha256") == sha256(RUNNER_PATH)
    ):
        raise ValueError("completed smoke manifest binding mismatch")
    contract = load_execution_contract()
    if (
        manifest.get("input_bindings") != contract["immutable_inputs"]
        or manifest.get("bounded_smoke") != contract["bounded_smoke"]
        or manifest.get("authorization") != contract["authorization"]
    ):
        raise ValueError("completed smoke frozen binding sections mismatch")
    from evaluation.verify_task_segmented_smoke_artifact import verify

    verification = verify(output_root, sha256(manifest_path))
    if verification.get("status") != "pass":
        raise ValueError("completed smoke independent verification failed")
    return manifest


def _binding_sha256(
    contract: Mapping[str, Any], execution_contract_sha256: str,
    project_commit: str, device_name: str,
    batch_trace: Sequence[Mapping[str, object]],
) -> str:
    value = {
        "schema_version": 1, "run_mode": MODE,
        "execution_contract_sha256": execution_contract_sha256,
        "project_commit": project_commit, "input_bindings": contract["immutable_inputs"],
        "bounded_smoke": contract["bounded_smoke"], "execution": contract["execution"],
        "device": device_name,
        "common_batch_trace_sha256": hashlib.sha256(
            _csv_bytes(COMMON_BATCH_FIELDS, batch_trace)
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run(
    vector_root: Path,
    protocol_root: Path,
    schedule_root: Path,
    output_root: Path,
    project_commit: str,
    device_name: str = "cuda",
    mode: str = MODE,
) -> dict[str, Any]:
    contract = load_execution_contract()
    # Authorization is deliberately first: denied modes cannot verify/load a
    # vector, create an output directory, or inspect the schedule binary.
    authorize_mode(mode, contract)
    validate_project_commit(project_commit)
    reports = _verify_clean_remounts(vector_root, protocol_root, schedule_root)
    _validate_verification_bindings(reports, contract)
    selected, unit = _strict_join(
        vector_root, protocol_root, schedule_root, contract
    )
    batches, batch_trace = _materialize_batches(
        vector_root, selected, unit, contract
    )

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("bounded smoke supports only cpu or cuda")
    contract_sha = sha256(EXECUTION_CONTRACT_PATH)
    binding_sha = _binding_sha256(
        contract, contract_sha, project_commit, str(device), batch_trace
    )
    if output_root.exists():
        reused = _reuse_complete_output(
            output_root, binding_sha, contract_sha, project_commit
        )
        if reused is not None:
            return {**reused, "reused_complete_output": True,
                    "output": str(output_root)}
    else:
        output_root.mkdir(parents=True)
    _atomic_csv(output_root / "common_batch_trace.csv", COMMON_BATCH_FIELDS, batch_trace)

    init_seed = initialization_seed(0, 20260717)
    arm_results: list[dict[str, object]] = []
    for arm_id in ARMS:
        arm_results.append(_run_arm(
            output_root / "runs" / arm_id, arm_id, batches, contract,
            binding_sha, init_seed, device,
        ))
    initial_hashes = {str(result["initial_state_sha256"]) for result in arm_results}
    if len(initial_hashes) != 1:
        raise ValueError("paired initial model states are not byte-identical")
    arm_summary = []
    for result in arm_results:
        arm_summary.append({
            "arm_id": result["arm_id"], "status": result["status"],
            "optimizer_steps": result["optimizer_steps"],
            "initial_state_sha256": result["initial_state_sha256"],
            "final_state_sha256": result["final_state_sha256"],
            "final_optimizer_state_sha256": result[
                "final_optimizer_state_sha256"
            ],
        })
    _atomic_csv(output_root / "arm_summary.csv", ARM_SUMMARY_FIELDS, arm_summary)
    metadata = {
        "status": "pass", "schema_version": 1, "run_mode": MODE,
        "project_commit": project_commit, "runner_source_sha256": sha256(RUNNER_PATH),
        "execution_contract_sha256": contract_sha, "binding_sha256": binding_sha,
        "input_bindings": contract["immutable_inputs"],
        "bounded_smoke": contract["bounded_smoke"],
        "authorization": contract["authorization"],
        "device": str(device), "python": sys.version.split()[0],
        "torch": torch.__version__, "completed_arms": list(ARMS),
        "optimizer_steps_completed": 6,
        "common_initial_state_sha256": next(iter(initial_hashes)),
        "full_training_authorized": False, "scientific_decision_permitted": False,
        "held_out_test_accessed": False,
    }
    _atomic_json(output_root / "smoke_run_metadata.json", metadata)
    artifacts = _regular_file_hashes(
        output_root, {"task_segmented_smoke_manifest.json"}
    )
    if len(artifacts) != 12:
        raise ValueError(f"smoke output must contain 12 bound files, got {len(artifacts)}")
    manifest = {
        "schema_version": 1, "status": "pass", "run_mode": MODE,
        "project_commit": project_commit, "runner_source_sha256": sha256(RUNNER_PATH),
        "execution_contract_sha256": contract_sha, "binding_sha256": binding_sha,
        "input_bindings": contract["immutable_inputs"],
        "bounded_smoke": contract["bounded_smoke"],
        "authorization": contract["authorization"],
        "artifact_sha256": artifacts,
    }
    _atomic_json(output_root / "task_segmented_smoke_manifest.json", manifest)
    from evaluation.verify_task_segmented_smoke_artifact import verify

    verification = verify(
        output_root, sha256(output_root / "task_segmented_smoke_manifest.json")
    )
    if verification.get("status") != "pass":
        raise ValueError("same-run independent smoke-artifact verification failed")
    return {
        **manifest,
        "independent_verification": verification,
        "reused_complete_output": False,
        "output": str(output_root),
    }


def synthetic_batches(seed: int = 20260718) -> list[dict[str, Any]]:
    """Create deterministic finite lower-level smoke inputs for unit tests."""

    import numpy as np

    rng = np.random.default_rng(seed)
    true_groups = ["NR"] * 32 + ["TSR"] * 32
    pseudo_groups = (["0"] * 16 + ["1"] * 16) * 2
    batches = []
    for step in range(2):
        indices_sha = hashlib.sha256(
            b"".join(struct.pack("<I", step * 64 + index) for index in range(64))
        ).hexdigest()
        trial_ids = [f"synthetic-trial-{step}-{index}" for index in range(64)]
        batches.append({
            "step_index": step, "epoch": 1, "batch_index": step,
            "indices_sha256": indices_sha,
            "trial_ids_sha256": hashlib.sha256(
                ("\n".join(trial_ids) + "\n").encode("utf-8")
            ).hexdigest(),
            "schedule_unit_sha256": stable_hash("synthetic-schedule-unit"),
            "global_mask_key": stable_hash("synthetic-p4b", step),
            "true_groups": true_groups, "pseudo_groups": pseudo_groups,
            "eeg": rng.normal(size=(64, 1024)).astype(np.float32),
            "text": rng.normal(size=(64, 1024)).astype(np.float32),
        })
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default=MODE)
    args = parser.parse_args()
    report = run(
        args.vector_root, args.protocol_root, args.schedule_root,
        args.output_root, args.project_commit, args.device, args.mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TASK-SEGMENTED OBJECTIVE BOUNDED SMOKE: PASS")


if __name__ == "__main__":
    main()
