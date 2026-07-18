"""Independently verify a bounded P4b task-segmented smoke artifact.

The verifier is deliberately standard-library only.  In particular it never
imports PyTorch/NumPy and never deserializes ``resume_checkpoint.pt``.  The
checkpoint is treated as opaque bytes whose digest is bound by the top-level
manifest.  This avoids turning verification into code execution.

The bounded smoke is an operational gate only: three arms, two optimizer
steps per arm, and the same two balanced batches for every arm.  A passing
artifact cannot authorize the 45-fit experiment or support a scientific
decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_NAME = "task_segmented_smoke_manifest.json"
METADATA_NAME = "smoke_run_metadata.json"
TOP_LEVEL_FILES = frozenset({
    "arm_summary.csv",
    "common_batch_trace.csv",
    METADATA_NAME,
    MANIFEST_NAME,
})
ARM_IDS = (
    "global_mixed",
    "true_task_segmented",
    "pseudo_task_segmented",
)
PER_ARM_FILES = frozenset({
    "resume_checkpoint.pt",
    "run_summary.json",
    "step_trace.csv",
})
NON_MANIFEST_FILES = frozenset(
    {
        "arm_summary.csv",
        "common_batch_trace.csv",
        METADATA_NAME,
    }
    | {
        f"runs/{arm}/{name}"
        for arm in ARM_IDS
        for name in PER_ARM_FILES
    }
)

EXECUTION_CONTRACT_SHA256 = (
    "9fd862b970ab95ded6f5efa0eb5290f8687bad7fd3a7f251f9ca66176de9f813"
)
IMMUTABLE_INPUTS: dict[str, dict[str, Any]] = {
    "prompt_neutral_vectors": {
        "preserved_source_id": (
            "kaggle-dataset-thestonedape-task-aware-eegtotext-version-2"
        ),
        "combined_manifest_sha256": (
            "6c1fff8d2e89e33a72d03c39651e8ecce678c3b93cdb66747dd6dcc00538cddb"
        ),
        "eeg_vector_index_sha256": (
            "373e49da7d3d6d00aaae414437886033e7f2c6a938a92662f76a0973744e0ae9"
        ),
        "text_vector_index_sha256": (
            "5b0a9579cbc586a7eaab353cee7ba65ab538a5e9cbd333e51ccab9855bde98ae"
        ),
        "trial_text_targets_sha256": (
            "1cde6365bdfe523d2497cba56375d1a27d917cfab02d7d6a407564ce32ca2b21"
        ),
    },
    "task_segmented_protocol": {
        "preserved_source_id": (
            "kaggle-dataset-thestonedape-task-aware-eeg2text-"
            "task-segmented-protocol-version-1"
        ),
        "contract_sha256": (
            "396670afc0244cb601364ff89df53944c4f63402191a9d120e6e2648e5baed3b"
        ),
        "report_sha256": (
            "f99ff4ad371e30b86dc9582bb4c35854f6bba863d7868a5f424f93776eab8116"
        ),
    },
    "training_schedule": {
        "preserved_source_id": (
            "kaggle-dataset-thestonedape-task-aware-eeg2text-"
            "task-segmented-schedule-version-1"
        ),
        "contract_sha256": (
            "a6ea34388cd98380654f413b1440d0d5cee0b8065555b0c04a53d0db6ea12287"
        ),
        "manifest_sha256": (
            "0cf2a752b5f0a67e7282bc0b4551b4792ceb3d346dd441fef6c359515885270a"
        ),
        "report_sha256": (
            "5e7d953a8ca31e48d6acbd6168b146d695af449ef0ad5bc573a37d306cb7250f"
        ),
        "indices_sha256": (
            "79543e72f496ee3f7a8140556b274c15ecc5992e900a07bed5e5c74a2ddd7cbc"
        ),
        "trial_catalog_sha256": (
            "3d93e0cea4290ac22e8111760241d04109392e6f024abae85a4d9504fc4f8fc9"
        ),
        "shape": [15, 40, 105, 64],
    },
}

BOUNDED_SMOKE: dict[str, Any] = {
    "outer_fold": 0,
    "training_seed": 20260717,
    "epoch": 1,
    "zero_based_batch_indices": [0, 1],
    "arms": list(ARM_IDS),
    "optimizer_steps_per_arm": 2,
    "total_optimizer_steps": 6,
    "batch_size": 64,
    "examples_per_task_pseudo_cell": 16,
    "trainable_parameters": 196608,
    "same_catalog_indices_across_arms": True,
    "byte_identical_initial_state_across_arms": True,
}

AUTHORIZATION: dict[str, Any] = {
    "bounded_smoke_authorized": True,
    "full_training_authorized": False,
    "checkpoint_or_confirmation_evaluation_authorized": False,
    "scientific_decision_permitted": False,
    "held_out_test_accessed": False,
    "next_gate_after_pass": (
        "independently verify and preserve the smoke artifact before any request "
        "to authorize the 45 cross-fitted fits"
    ),
}

COMMON_REQUIRED_FIELDS = frozenset({
    "outer_fold",
    "training_seed",
    "epoch",
    "batch_index",
    "batch_position",
    "catalog_index",
    "trial_id",
    "subject_id",
    "reading_task",
    "pseudo_task_id",
    "normalized_text_sha256",
    "text_target_id",
    "eeg_vector_file",
    "eeg_vector_offset",
    "text_vector_file",
    "text_vector_offset",
})
STEP_REQUIRED_FIELDS = frozenset({
    "outer_fold",
    "training_seed",
    "arm_id",
    "epoch",
    "batch_index",
    "global_step",
    "schedule_unit_sha256",
    "batch_catalog_indices_sha256",
    "batch_trial_ids_sha256",
    "global_mask_key",
    "eeg_to_text_mask_sha256",
    "text_to_eeg_mask_sha256",
    "initial_state_sha256",
    "pre_step_state_sha256",
    "loss",
    "gradient_norm_preclip",
    "post_step_state_sha256",
    "optimizer_state_sha256",
})
ARM_SUMMARY_REQUIRED_FIELDS = frozenset({
    "arm_id",
    "status",
    "optimizer_steps",
    "initial_state_sha256",
    "final_state_sha256",
    "final_optimizer_state_sha256",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path.name}") from exc
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            require(fields is not None, f"CSV lacks header: {path.name}")
            require(len(fields) == len(set(fields)), f"duplicate CSV header: {path.name}")
            require(all(field and field.strip() == field for field in fields),
                    f"invalid CSV header: {path.name}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot read CSV: {path.name}") from exc
    require(
        all(None not in row and all(value is not None for value in row.values()) for row in rows),
        f"malformed CSV row: {path.name}",
    )
    return fields, rows


def _require_fields(actual: Iterable[str], required: frozenset[str], label: str) -> None:
    missing = sorted(required - set(actual))
    require(not missing, f"{label} missing required fields: {missing}")


def _integer(value: str, label: str) -> int:
    require(value == value.strip() and value != "", f"invalid integer {label}")
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"invalid integer {label}: {value!r}") from exc
    require(str(parsed) == value, f"non-canonical integer {label}: {value!r}")
    return parsed


def _finite(value: str, label: str, *, nonnegative: bool = False) -> float:
    require(value == value.strip() and value != "", f"invalid float {label}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid float {label}: {value!r}") from exc
    require(math.isfinite(parsed), f"non-finite {label}: {value!r}")
    if nonnegative:
        require(parsed >= 0.0, f"negative {label}: {value!r}")
    return parsed


def _safe_manifest_path(value: object) -> str:
    require(isinstance(value, str) and value != "", "manifest path must be non-empty text")
    require("\\" not in value, f"manifest path must use POSIX separators: {value!r}")
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"absolute manifest path is forbidden: {value!r}")
    require(value == pure.as_posix(), f"non-canonical manifest path: {value!r}")
    require(all(part not in ("", ".", "..") for part in pure.parts),
            f"unsafe manifest path: {value!r}")
    return value


def _safe_npz_path(value: str, label: str) -> str:
    require(value != "" and value == value.strip(), f"invalid {label}")
    require("\\" not in value, f"{label} must use POSIX separators: {value!r}")
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"absolute {label} is forbidden: {value!r}")
    require(value == pure.as_posix(), f"non-canonical {label}: {value!r}")
    require(all(part not in ("", ".", "..") for part in pure.parts),
            f"unsafe {label}: {value!r}")
    require(pure.suffix == ".npz", f"{label} must identify an NPZ file: {value!r}")
    return value


def _exact_inventory(root: Path) -> None:
    require(not root.is_symlink(), "smoke artifact root must not be a symlink")
    names: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            require(not entry.is_symlink(), f"symlink is forbidden: {entry.name}")
            if entry.name == "runs":
                require(entry.is_dir(follow_symlinks=False), "runs must be a directory")
            else:
                require(entry.is_file(follow_symlinks=False),
                        f"non-file top-level entry: {entry.name}")
            names.add(entry.name)
    expected = set(TOP_LEVEL_FILES) | {"runs"}
    require_equal(names, expected, "smoke artifact top-level inventory")

    runs = root / "runs"
    arm_names: set[str] = set()
    with os.scandir(runs) as entries:
        for entry in entries:
            require(not entry.is_symlink(), f"symlink is forbidden: runs/{entry.name}")
            require(entry.is_dir(follow_symlinks=False),
                    f"non-directory arm entry: runs/{entry.name}")
            arm_names.add(entry.name)
    require_equal(arm_names, set(ARM_IDS), "smoke artifact arm inventory")

    for arm in ARM_IDS:
        names = set()
        arm_root = runs / arm
        with os.scandir(arm_root) as entries:
            for entry in entries:
                require(not entry.is_symlink(),
                        f"symlink is forbidden: runs/{arm}/{entry.name}")
                require(entry.is_file(follow_symlinks=False),
                        f"non-file arm entry: runs/{arm}/{entry.name}")
                names.add(entry.name)
        require_equal(names, set(PER_ARM_FILES), f"{arm} file inventory")
        require((arm_root / "resume_checkpoint.pt").stat().st_size > 0,
                f"empty opaque checkpoint for {arm}")


def _binding_section(document: Mapping[str, Any], label: str) -> None:
    require_equal(document.get("execution_contract_sha256"), EXECUTION_CONTRACT_SHA256,
                  f"{label} execution contract SHA256")
    require_equal(document.get("input_bindings"), IMMUTABLE_INPUTS,
                  f"{label} immutable input bindings")
    require_equal(document.get("bounded_smoke"), BOUNDED_SMOKE,
                  f"{label} bounded-smoke definition")
    require_equal(document.get("authorization"), AUTHORIZATION,
                  f"{label} authorization")


def _verify_common_trace(path: Path) -> dict[int, tuple[str, str]]:
    fields, rows = read_csv(path)
    _require_fields(fields, COMMON_REQUIRED_FIELDS, "common batch trace")
    require_equal(len(rows), 128, "common batch trace row count")
    batches: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    for row_number, row in enumerate(rows, start=2):
        require_equal(_integer(row["outer_fold"], f"common row {row_number} outer_fold"),
                      0, "common outer fold")
        require_equal(_integer(row["training_seed"], f"common row {row_number} seed"),
                      20260717, "common training seed")
        require_equal(_integer(row["epoch"], f"common row {row_number} epoch"),
                      1, "common epoch")
        batch = _integer(row["batch_index"], f"common row {row_number} batch")
        require(batch in batches, f"unexpected common batch index: {batch}")
        position = _integer(row["batch_position"], f"common row {row_number} position")
        require(0 <= position < 64, f"common batch position out of range: {position}")
        catalog_index = _integer(row["catalog_index"], f"common row {row_number} catalog")
        require(0 <= catalog_index <= 0xFFFFFFFF,
                f"catalog index outside uint32 range at common row {row_number}")
        require(row["trial_id"] != "" and row["trial_id"] == row["trial_id"].strip(),
                f"invalid trial id at common row {row_number}")
        require(row["subject_id"] != "" and row["subject_id"] == row["subject_id"].strip(),
                f"invalid subject id at common row {row_number}")
        require(row["reading_task"] in ("NR", "TSR"),
                f"invalid reading task at common row {row_number}")
        pseudo = _integer(row["pseudo_task_id"], f"common row {row_number} pseudo task")
        require(pseudo in (0, 1), f"invalid pseudo-task id at common row {row_number}")
        require(is_sha256(row["normalized_text_sha256"]),
                f"invalid normalized text SHA256 at common row {row_number}")
        require_equal(row["text_target_id"], row["normalized_text_sha256"],
                      f"common row {row_number} text target identity")
        _safe_npz_path(row["eeg_vector_file"], f"common row {row_number} EEG vector file")
        _safe_npz_path(row["text_vector_file"], f"common row {row_number} text vector file")
        require(_integer(row["eeg_vector_offset"],
                         f"common row {row_number} EEG vector offset") >= 0,
                f"negative EEG vector offset at common row {row_number}")
        require(_integer(row["text_vector_offset"],
                         f"common row {row_number} text vector offset") >= 0,
                f"negative text vector offset at common row {row_number}")
        batches[batch].append(row)

    batch_hashes: dict[int, tuple[str, str]] = {}
    for batch, batch_rows in batches.items():
        require_equal(len(batch_rows), 64, f"batch {batch} row count")
        positions = sorted(_integer(row["batch_position"], "batch position")
                           for row in batch_rows)
        require_equal(positions, list(range(64)), f"batch {batch} positions")
        cells: dict[tuple[str, int], int] = {
            (task, pseudo): 0 for task in ("NR", "TSR") for pseudo in (0, 1)
        }
        catalog_identities: set[int] = set()
        trial_identities: set[str] = set()
        text_identities: set[str] = set()
        for row in batch_rows:
            pseudo = _integer(row["pseudo_task_id"], "pseudo task")
            cells[(row["reading_task"], pseudo)] += 1
            catalog = _integer(row["catalog_index"], "catalog index")
            trial = row["trial_id"]
            text = row["text_target_id"]
            require(catalog not in catalog_identities,
                    f"duplicate catalog identity within batch {batch}")
            require(trial not in trial_identities,
                    f"duplicate trial identity within batch {batch}")
            require(text not in text_identities,
                    f"duplicate text identity within batch {batch}")
            catalog_identities.add(catalog)
            trial_identities.add(trial)
            text_identities.add(text)
        require_equal(cells, {key: 16 for key in cells}, f"batch {batch} task/pseudo balance")
        ordered = sorted(
            batch_rows,
            key=lambda row: _integer(row["batch_position"], "batch position"),
        )
        catalog_payload = b"".join(
            _integer(row["catalog_index"], "catalog index").to_bytes(
                4, "little", signed=False
            )
            for row in ordered
        )
        trial_payload = ("\n".join(row["trial_id"] for row in ordered) + "\n").encode(
            "utf-8"
        )
        batch_hashes[batch] = (
            hashlib.sha256(catalog_payload).hexdigest(),
            hashlib.sha256(trial_payload).hexdigest(),
        )
    return batch_hashes


def _verify_step_trace(path: Path, arm: str) -> list[dict[str, str]]:
    fields, rows = read_csv(path)
    _require_fields(fields, STEP_REQUIRED_FIELDS, f"{arm} step trace")
    require_equal(len(rows), 2, f"{arm} optimizer step count")
    for offset, row in enumerate(rows):
        label = f"{arm} step {offset + 1}"
        require_equal(row["arm_id"], arm, f"{label} arm")
        require_equal(_integer(row["outer_fold"], f"{label} fold"), 0, f"{label} fold")
        require_equal(_integer(row["training_seed"], f"{label} seed"), 20260717,
                      f"{label} seed")
        require_equal(_integer(row["epoch"], f"{label} epoch"), 1, f"{label} epoch")
        require_equal(_integer(row["batch_index"], f"{label} batch"), offset,
                      f"{label} batch")
        require_equal(_integer(row["global_step"], f"{label} global step"), offset + 1,
                      f"{label} global step")
        for field in (
            "schedule_unit_sha256",
            "batch_catalog_indices_sha256",
            "batch_trial_ids_sha256",
            "global_mask_key",
            "eeg_to_text_mask_sha256",
            "text_to_eeg_mask_sha256",
            "initial_state_sha256",
            "pre_step_state_sha256",
            "post_step_state_sha256",
            "optimizer_state_sha256",
        ):
            require(is_sha256(row[field]), f"invalid SHA256 in {label}.{field}")
        _finite(row["loss"], f"{label} loss", nonnegative=True)
        _finite(row["gradient_norm_preclip"], f"{label} gradient norm", nonnegative=True)
    require_equal(rows[0]["pre_step_state_sha256"], rows[0]["initial_state_sha256"],
                  f"{arm} first pre-step/initial state")
    require_equal(rows[1]["initial_state_sha256"], rows[0]["initial_state_sha256"],
                  f"{arm} repeated initial-state binding")
    require_equal(rows[1]["pre_step_state_sha256"], rows[0]["post_step_state_sha256"],
                  f"{arm} resume state chain")
    return rows


def _verify_run_summary(
    path: Path, arm: str, steps: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    summary = read_json(path)
    require_equal(summary.get("schema_version"), 1, f"{arm} summary schema")
    require_equal(summary.get("status"), "pass", f"{arm} summary status")
    require_equal(summary.get("run_mode"), "bounded_smoke", f"{arm} summary mode")
    require_equal(summary.get("arm_id"), arm, f"{arm} summary arm")
    require_equal(summary.get("outer_fold"), 0, f"{arm} summary fold")
    require_equal(summary.get("training_seed"), 20260717, f"{arm} summary seed")
    require_equal(summary.get("epoch"), 1, f"{arm} summary epoch")
    require_equal(summary.get("batch_indices"), [0, 1], f"{arm} summary batches")
    require_equal(summary.get("optimizer_steps"), 2, f"{arm} summary optimizer steps")
    require_equal(summary.get("trainable_parameters"), 196608,
                  f"{arm} summary trainable parameters")
    require_equal(summary.get("initial_state_sha256"), steps[0]["initial_state_sha256"],
                  f"{arm} summary initial state")
    require_equal(summary.get("final_state_sha256"), steps[-1]["post_step_state_sha256"],
                  f"{arm} summary final state")
    require_equal(summary.get("final_optimizer_state_sha256"),
                  steps[-1]["optimizer_state_sha256"],
                  f"{arm} summary final optimizer state")
    require_equal(summary.get("full_training_authorized"), False,
                  f"{arm} summary full-training authorization")
    require_equal(summary.get("scientific_decision_permitted"), False,
                  f"{arm} summary scientific-decision authorization")
    require_equal(summary.get("held_out_test_accessed"), False,
                  f"{arm} summary held-out test access")
    nested = summary.get("artifact_sha256")
    require(isinstance(nested, dict), f"{arm} summary lacks artifact hashes")
    require_equal(set(nested), {"resume_checkpoint.pt", "step_trace.csv"},
                  f"{arm} nested artifact inventory")
    require(all(is_sha256(value) for value in nested.values()),
            f"{arm} summary has invalid nested artifact SHA256")
    return summary


def _verify_arm_summary(
    path: Path,
    steps_by_arm: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    fields, rows = read_csv(path)
    _require_fields(fields, ARM_SUMMARY_REQUIRED_FIELDS, "arm summary")
    require_equal(len(rows), 3, "arm summary row count")
    by_arm = {row.get("arm_id", ""): row for row in rows}
    require_equal(set(by_arm), set(ARM_IDS), "arm summary arms")
    require_equal(len(by_arm), len(rows), "arm summary unique arms")
    for arm in ARM_IDS:
        row = by_arm[arm]
        steps = steps_by_arm[arm]
        require_equal(row["status"], "pass", f"{arm} aggregate status")
        require_equal(_integer(row["optimizer_steps"], f"{arm} aggregate steps"), 2,
                      f"{arm} aggregate steps")
        require_equal(row["initial_state_sha256"], steps[0]["initial_state_sha256"],
                      f"{arm} aggregate initial state")
        require_equal(row["final_state_sha256"], steps[-1]["post_step_state_sha256"],
                      f"{arm} aggregate final state")
        require_equal(row["final_optimizer_state_sha256"],
                      steps[-1]["optimizer_state_sha256"],
                      f"{arm} aggregate optimizer state")


def verify(
    artifact_root: Path,
    expected_manifest_sha256: str,
    *,
    preserved_source_id: str | None = None,
) -> dict[str, Any]:
    """Verify a sealed smoke artifact without loading its checkpoints."""

    require(is_sha256(expected_manifest_sha256),
            "expected manifest SHA256 must be lowercase hexadecimal")
    root = Path(os.path.abspath(os.fspath(artifact_root)))
    require(root.is_dir(), f"smoke artifact root is not a directory: {root}")
    _exact_inventory(root)

    manifest_path = root / MANIFEST_NAME
    require_equal(sha256(manifest_path), expected_manifest_sha256,
                  "externally expected smoke manifest SHA256")
    manifest = read_json(manifest_path)
    require_equal(manifest.get("schema_version"), 1, "manifest schema")
    require_equal(manifest.get("status"), "pass", "manifest status")
    require_equal(manifest.get("run_mode"), "bounded_smoke", "manifest run mode")
    _binding_section(manifest, "manifest")
    require(is_sha256(manifest.get("runner_source_sha256")),
            "manifest runner source SHA256 is invalid")
    project_commit = manifest.get("project_commit")
    require(isinstance(project_commit, str) and len(project_commit) == 40
            and all(character in "0123456789abcdef" for character in project_commit),
            "manifest project commit is invalid")
    if preserved_source_id is not None:
        require(isinstance(preserved_source_id, str) and preserved_source_id.strip() != "",
                "preserved smoke source id must be non-empty text")
        # The source id is assigned only after Kaggle preservation, whereas the
        # manifest is sealed before upload.  If a preservation wrapper adds the
        # field it must agree; its absence is expected for direct runner output.
        if "preserved_source_id" in manifest:
            require_equal(manifest.get("preserved_source_id"), preserved_source_id,
                          "preserved smoke source id")

    declared = manifest.get("artifact_sha256")
    require(isinstance(declared, dict), "manifest lacks artifact_sha256")
    safe_declared = {_safe_manifest_path(key): value for key, value in declared.items()}
    require_equal(set(safe_declared), set(NON_MANIFEST_FILES),
                  "manifest non-manifest artifact inventory")
    require(all(is_sha256(value) for value in safe_declared.values()),
            "manifest contains invalid artifact SHA256")
    actual = {relative: sha256(root / Path(*PurePosixPath(relative).parts))
              for relative in sorted(NON_MANIFEST_FILES)}
    require_equal(actual, dict(sorted(safe_declared.items())),
                  "manifest/non-manifest file hashes")

    metadata = read_json(root / METADATA_NAME)
    require_equal(metadata.get("schema_version"), 1, "metadata schema")
    require_equal(metadata.get("status"), "pass", "metadata status")
    require_equal(metadata.get("run_mode"), "bounded_smoke", "metadata run mode")
    _binding_section(metadata, "metadata")
    require_equal(metadata.get("project_commit"), project_commit, "metadata project commit")
    require_equal(metadata.get("runner_source_sha256"), manifest["runner_source_sha256"],
                  "metadata runner source SHA256")
    require_equal(metadata.get("completed_arms"), list(ARM_IDS), "metadata completed arms")
    require_equal(metadata.get("optimizer_steps_completed"), 6,
                  "metadata completed optimizer steps")
    require_equal(metadata.get("full_training_authorized"), False,
                  "metadata full-training authorization")
    require_equal(metadata.get("scientific_decision_permitted"), False,
                  "metadata scientific-decision authorization")
    require_equal(metadata.get("held_out_test_accessed"), False,
                  "metadata held-out test access")

    common_trace_hashes = _verify_common_trace(root / "common_batch_trace.csv")
    steps_by_arm: dict[str, list[dict[str, str]]] = {}
    initial_states: set[str] = set()
    common_step_bindings: dict[int, tuple[str, str, str, str]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARM_IDS:
        arm_root = root / "runs" / arm
        steps = _verify_step_trace(arm_root / "step_trace.csv", arm)
        steps_by_arm[arm] = steps
        initial_states.add(steps[0]["initial_state_sha256"])
        for batch, row in enumerate(steps):
            require_equal(
                (
                    row["batch_catalog_indices_sha256"],
                    row["batch_trial_ids_sha256"],
                ),
                common_trace_hashes[batch],
                f"step/common row binding for {arm} batch {batch}",
            )
            binding = (
                row["schedule_unit_sha256"],
                row["batch_catalog_indices_sha256"],
                row["batch_trial_ids_sha256"],
                row["global_mask_key"],
            )
            if batch in common_step_bindings:
                require_equal(binding, common_step_bindings[batch],
                              f"common batch binding for batch {batch}")
            else:
                common_step_bindings[batch] = binding
        summary = _verify_run_summary(arm_root / "run_summary.json", arm, steps)
        summaries[arm] = summary
        require_equal(summary["artifact_sha256"]["resume_checkpoint.pt"],
                      actual[f"runs/{arm}/resume_checkpoint.pt"],
                      f"{arm} nested checkpoint SHA256")
        require_equal(summary["artifact_sha256"]["step_trace.csv"],
                      actual[f"runs/{arm}/step_trace.csv"],
                      f"{arm} nested step trace SHA256")
    require_equal(len(initial_states), 1, "byte-identical initial state across arms")
    _verify_arm_summary(root / "arm_summary.csv", steps_by_arm)

    return {
        "status": "pass",
        "preserved_source_id": preserved_source_id,
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "smoke_manifest_sha256": expected_manifest_sha256,
        "project_commit": project_commit,
        "runner_source_sha256": manifest["runner_source_sha256"],
        "verified_artifact_sha256": actual,
        "arms": list(ARM_IDS),
        "optimizer_steps_per_arm": 2,
        "total_optimizer_steps": 6,
        "common_trace_rows": 128,
        "checkpoint_deserialized": False,
        "full_training_authorized": False,
        "scientific_decision_permitted": False,
        "held_out_test_accessed": False,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--preserved-source-id")
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    report = verify(
        args.artifact_root,
        args.expected_manifest_sha256,
        preserved_source_id=args.preserved_source_id,
    )
    if args.output_report is not None:
        write_report(args.output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TASK-SEGMENTED BOUNDED SMOKE ARTIFACT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
