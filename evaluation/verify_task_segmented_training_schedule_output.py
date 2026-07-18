"""Deep, standard-library verifier for a frozen P4b training schedule.

Unlike the freezers' lightweight same-run check, this verifier independently
decodes every scheduled index and recomputes the scientific schedule
invariants.  By default it accepts only the six core freezer files.  A sealed
artifact verifier may name a narrowly bounded set of additional provenance
files without weakening that default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


REPORT_NAME = "task_segmented_training_schedule_report.json"
MANIFEST_NAME = "task_segmented_training_schedule_manifest.json"
INDEX_NAME = "schedule_indices.u32le"
CATALOG_NAME = "trial_catalog.csv"
UNITS_NAME = "schedule_units.csv"
AUDIT_NAME = "schedule_audit.csv"

CORE_FILES = (
    CATALOG_NAME,
    INDEX_NAME,
    UNITS_NAME,
    AUDIT_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
)
REPORT_ARTIFACTS = CORE_FILES[:-1]
MANIFEST_ARTIFACTS = CORE_FILES[:4]

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
AUDIT_FIELDS = (
    "outer_fold", "training_seed", "reading_task", "pseudo_group",
    "fit_trial_rows", "fit_unique_text_identities", "scheduled_slots",
    "identity_use_min", "identity_use_max", "identity_use_gap",
    "conditional_row_use_gap_max", "covered_fit_rows", "status",
)

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (20260717, 20260718, 20260719)
CELLS = (("NR", 0), ("NR", 1), ("TSR", 0), ("TSR", 1))
ARMS = ("global_mixed", "true_task_segmented", "pseudo_task_segmented")
EPOCHS = 40
BATCH_SIZE = 64
EXAMPLES_PER_CELL = 16
PRODUCTION_CATALOG_ROWS = 9011
PRODUCTION_BATCHES_PER_EPOCH = 105

PARENT_DATASET_SLUG = "thestonedape/task-aware-eeg2text-task-segmented-protocol"
PARENT_DATASET_VERSION = 1
PARENT_SOURCE_ID = (
    "kaggle-dataset-thestonedape-task-aware-eeg2text-"
    "task-segmented-protocol-version-1"
)
UPSTREAM_SOURCE_ID = "kaggle-dataset-thestonedape-task-aware-eegtotext-version-2"


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _as_int(value: object, field: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in {field}: {value!r}") from exc


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path.name}") from exc
    _fail(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _fail(tuple(reader.fieldnames or ()) == tuple(expected_fields),
                  f"unexpected CSV header: {path.name}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot read CSV: {path.name}") from exc
    _fail(all(None not in row for row in rows), f"extra CSV columns: {path.name}")
    return rows


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def _initialization_seed(outer_fold: int, training_seed: int) -> int:
    return int(_stable_hash("p4b-adapter-init-v1", outer_fold, training_seed)[:16], 16) & (
        (1 << 63) - 1
    )


def _verify_inventory(root: Path, allowed_extra_files: Sequence[str] = ()) -> None:
    _fail(root.is_dir(), f"schedule output root is not a directory: {root}")
    extras = tuple(allowed_extra_files)
    _fail(len(set(extras)) == len(extras), "allowed extra-file names are duplicated")
    for name in extras:
        _fail(
            isinstance(name, str)
            and bool(name)
            and name not in {".", ".."}
            and Path(name).name == name,
            f"invalid allowed extra-file name: {name!r}",
        )
        _fail(name not in CORE_FILES, f"core file cannot be declared as an extra: {name}")
    expected_files = set(CORE_FILES) | set(extras)
    entries = {entry.name: entry for entry in root.iterdir()}
    _fail(set(entries) == expected_files,
          f"schedule output inventory mismatch: {sorted(entries)}")
    _fail(not any(entries[name].is_symlink() for name in expected_files),
          "schedule output inventory cannot contain symbolic links")
    _fail(all(entries[name].is_file() for name in expected_files),
          "all schedule entries must be regular files")


def _verify_hash_bindings(
    root: Path,
    report: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, str]:
    report_hashes = report.get("artifact_sha256")
    _fail(isinstance(report_hashes, dict), "report lacks artifact_sha256")
    _fail(set(report_hashes) == set(REPORT_ARTIFACTS),
          "report artifact inventory is not the exact five-file inventory")
    actual = {name: sha256(root / name) for name in REPORT_ARTIFACTS}
    for name in REPORT_ARTIFACTS:
        _fail(_is_sha256(report_hashes[name]), f"invalid report artifact digest: {name}")
        _fail(report_hashes[name] == actual[name], f"schedule artifact hash mismatch: {name}")

    manifest_hashes = manifest.get("artifact_sha256")
    _fail(isinstance(manifest_hashes, dict), "manifest lacks artifact_sha256")
    _fail(set(manifest_hashes) == set(MANIFEST_ARTIFACTS),
          "manifest artifact inventory is not the exact four-file inventory")
    for name in MANIFEST_ARTIFACTS:
        _fail(manifest_hashes[name] == actual[name],
              f"manifest/report artifact cross-binding mismatch: {name}")
    _fail(actual[MANIFEST_NAME] == report_hashes[MANIFEST_NAME],
          "report does not bind the mounted manifest")
    return actual


def _parse_catalog(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    _fail(bool(rows), "empty trial catalog")
    parsed: list[dict[str, object]] = []
    trial_ids: set[str] = set()
    previous_trial_id: str | None = None
    for position, row in enumerate(rows):
        index = _as_int(row["trial_index"], "trial_index")
        _fail(index == position, "catalog trial_index is not contiguous row order")
        trial_id = row["trial_id"]
        _fail(bool(trial_id), "catalog has an empty trial_id")
        _fail(trial_id not in trial_ids, f"duplicate catalog trial_id: {trial_id}")
        if previous_trial_id is not None:
            _fail(previous_trial_id < trial_id, "catalog is not in strict trial_id order")
        trial_ids.add(trial_id)
        previous_trial_id = trial_id
        text_fold = _as_int(row["text_fold"], "text_fold")
        pseudo_group = _as_int(row["pseudo_group"], "pseudo_group")
        task = row["reading_task"]
        identity = row["normalized_text_sha256"]
        _fail(text_fold in FOLDS, f"invalid catalog text_fold: {text_fold}")
        _fail((task, pseudo_group) in CELLS,
              f"invalid catalog task/pseudo cell: {task}::{pseudo_group}")
        _fail(row["dataset_version"] == "ZuCo2", "catalog includes a non-ZuCo2 row")
        _fail(bool(row["subject_id"]), "catalog has an empty subject_id")
        _fail(bool(identity), "catalog has an empty normalized-text identity")
        _fail(bool(row["text_target_id"]), "catalog has an empty text_target_id")
        _fail(_as_int(row["length_words_whitespace_v1"], "length") > 0,
              "catalog has a nonpositive text length")
        _fail(bool(row["eeg_vector_file"]), "catalog has an empty EEG vector file")
        _fail(_as_int(row["eeg_vector_offset"], "eeg_vector_offset") >= 0,
              "catalog has a negative EEG vector offset")
        _fail(_as_int(row["eeg_vector_dim"], "eeg_vector_dim") == 1024,
              "catalog includes a non-1024-D vector locator")
        parsed.append({
            "trial_index": index,
            "trial_id": trial_id,
            "text_fold": text_fold,
            "reading_task": task,
            "pseudo_group": pseudo_group,
            "normalized_text_sha256": identity,
        })
    return parsed


def _derived_fit_layout(
    catalog: list[dict[str, object]],
) -> tuple[
    int,
    dict[tuple[int, str, int], int],
    dict[tuple[int, str, int], list[int]],
]:
    fit_indices: dict[tuple[int, str, int], list[int]] = {}
    fit_counts: dict[tuple[int, str, int], int] = {}
    for fold in FOLDS:
        excluded = {fold, (fold + 1) % len(FOLDS)}
        for task, pseudo in CELLS:
            key = (fold, task, pseudo)
            values = [
                int(row["trial_index"])
                for row in catalog
                if row["text_fold"] not in excluded
                and row["reading_task"] == task
                and row["pseudo_group"] == pseudo
            ]
            _fail(bool(values), f"empty fit cell: {fold}::{task}::{pseudo}")
            fit_indices[key] = values
            fit_counts[key] = len(values)
    batches = max(math.ceil(value / EXAMPLES_PER_CELL) for value in fit_counts.values())
    return batches, fit_counts, fit_indices


def _parse_units(
    rows: list[dict[str, str]], batches: int, binary_bytes: int,
) -> list[dict[str, object]]:
    expected_order = [(fold, seed) for fold in FOLDS for seed in SEEDS]
    _fail(len(rows) == len(expected_order), "schedule_units.csv must contain 15 units")
    parsed: list[dict[str, object]] = []
    unit_uint32 = EPOCHS * batches * BATCH_SIZE
    unit_bytes = unit_uint32 * 4
    for unit_index, (row, (fold, seed)) in enumerate(zip(rows, expected_order)):
        values = {
            name: _as_int(row[name], name)
            for name in UNIT_FIELDS
            if name not in {"unit_sha256"}
        }
        _fail(values["unit_index"] == unit_index, "unit_index/order mismatch")
        _fail(values["outer_fold"] == fold and values["training_seed"] == seed,
              "unit fold/seed order mismatch")
        _fail(values["epochs"] == EPOCHS, "unit epoch count is not 40")
        _fail(values["batches_per_epoch"] == batches,
              "unit batches-per-epoch differs from derived B")
        _fail(values["batch_size"] == BATCH_SIZE, "unit batch size is not 64")
        _fail(values["uint32_count"] == unit_uint32, "unit uint32 count mismatch")
        _fail(values["byte_offset"] == unit_index * unit_bytes,
              "unit byte offsets are not contiguous canonical offsets")
        _fail(values["byte_length"] == unit_bytes, "unit byte length mismatch")
        _fail(_is_sha256(row["unit_sha256"]), "invalid unit SHA-256")
        _fail(values["initialization_seed"] == _initialization_seed(fold, seed),
              "unit initialization seed mismatch")
        parsed.append({**values, "unit_sha256": row["unit_sha256"]})
    _fail(binary_bytes == len(expected_order) * unit_bytes,
          "schedule binary length differs from the derived complete shape")
    return parsed


def _parse_audits(rows: list[dict[str, str]]) -> dict[tuple[int, int, str, int], dict[str, str]]:
    expected_order = [
        (fold, seed, task, pseudo)
        for fold in FOLDS
        for seed in SEEDS
        for task, pseudo in CELLS
    ]
    _fail(len(rows) == len(expected_order), "schedule_audit.csv must contain 60 rows")
    parsed: dict[tuple[int, int, str, int], dict[str, str]] = {}
    observed_order: list[tuple[int, int, str, int]] = []
    for row in rows:
        key = (
            _as_int(row["outer_fold"], "outer_fold"),
            _as_int(row["training_seed"], "training_seed"),
            row["reading_task"],
            _as_int(row["pseudo_group"], "pseudo_group"),
        )
        _fail(key not in parsed, f"duplicate schedule audit row: {key}")
        _fail(key[0] in FOLDS and key[1] in SEEDS and (key[2], key[3]) in CELLS,
              f"invalid schedule audit key: {key}")
        _fail(row["status"] == "pass", f"non-PASS schedule audit row: {key}")
        parsed[key] = row
        observed_order.append(key)
    _fail(observed_order == expected_order, "schedule audit row order is not canonical")
    return parsed


def _verify_manifest_and_report(
    report: dict[str, object],
    manifest: dict[str, object],
    expected_contract_sha256: str,
    expected_parent_report_sha256: str,
    catalog_rows: int,
    batches: int,
    fit_counts: Mapping[tuple[int, str, int], int],
    units: list[dict[str, object]],
    require_parent_clean_remount: bool,
) -> None:
    _fail(report.get("status") == "pass" and report.get("schema_version") == 1,
          "schedule report is not schema-1 PASS")
    _fail(manifest.get("status") == "pass" and manifest.get("schema_version") == 1,
          "schedule manifest is not schema-1 PASS")
    for payload, label in ((report, "report"), (manifest, "manifest")):
        _fail(payload.get("schedule_contract_sha256") == expected_contract_sha256,
              f"{label} child-contract SHA-256 mismatch")
        _fail(payload.get("parent_protocol_report_sha256") == expected_parent_report_sha256,
              f"{label} parent-report SHA-256 mismatch")

    expected_shape = [len(FOLDS) * len(SEEDS), EPOCHS, batches, BATCH_SIZE]
    _fail(manifest.get("shape") == expected_shape, "manifest schedule shape mismatch")
    _fail(manifest.get("dtype") == "<u4" and manifest.get("order") == "C",
          "manifest binary dtype/order mismatch")
    _fail(manifest.get("catalog_rows") == catalog_rows, "manifest catalog count mismatch")
    _fail(manifest.get("global_batches_per_epoch") == batches,
          "manifest global B mismatch")
    expected_fit_counts = {
        f"{fold}::{task}::{pseudo}": count
        for (fold, task, pseudo), count in sorted(fit_counts.items())
    }
    _fail(manifest.get("fit_cell_trial_rows") == expected_fit_counts,
          "manifest fit-cell counts differ from catalog-derived counts")

    expected_unit_order = [
        {
            "unit_index": unit["unit_index"],
            "outer_fold": unit["outer_fold"],
            "training_seed": unit["training_seed"],
            "unit_sha256": unit["unit_sha256"],
            "initialization_seed": unit["initialization_seed"],
        }
        for unit in units
    ]
    _fail(manifest.get("unit_order") == expected_unit_order,
          "manifest unit order/hash binding mismatch")
    _fail(manifest.get("applicable_arms") == list(ARMS),
          "manifest applicable-arm set/order mismatch")
    _fail(manifest.get("arm_axis_absent_from_binary") is True,
          "schedule binary unexpectedly declares an arm axis")
    _fail(manifest.get("same_schedule_across_arms") is True,
          "schedule is not declared byte-identical across arms")
    _fail(manifest.get("bounded_smoke_authorized") is True,
          "bounded three-arm smoke is not authorized")
    _fail(manifest.get("full_training_authorized") is False,
          "schedule artifact unexpectedly authorizes full training")
    _fail(manifest.get("held_out_test_accessed") is False,
          "schedule artifact accessed held-out test data")
    _fail(manifest.get("parent_preserved_dataset_slug") == PARENT_DATASET_SLUG,
          "manifest parent dataset slug mismatch")
    _fail(manifest.get("parent_preserved_dataset_version") == PARENT_DATASET_VERSION,
          "manifest parent dataset version mismatch")
    _fail(manifest.get("parent_protocol_artifact_source_id") == PARENT_SOURCE_ID,
          "manifest parent artifact source mismatch")
    _fail(manifest.get("upstream_input_source_id") == UPSTREAM_SOURCE_ID,
          "manifest upstream source mismatch")

    remount = manifest.get("parent_clean_remount_verification")
    _fail(isinstance(remount, dict), "manifest lacks parent clean-remount evidence")
    if require_parent_clean_remount:
        _fail(remount.get("status") == "pass", "parent clean-remount verification did not pass")
        _fail(remount.get("preserved_source_id") == PARENT_SOURCE_ID,
              "parent clean-remount source mismatch")
        _fail(remount.get("protocol_report_sha256") == expected_parent_report_sha256,
              "parent clean-remount report hash mismatch")
    else:
        _fail(remount.get("status") in {"pass", "synthetic_test_fixture"},
              "invalid synthetic parent verification marker")

    expected_counts = {
        "catalog_rows": catalog_rows,
        "schedule_units": len(FOLDS) * len(SEEDS),
        "epochs_per_unit": EPOCHS,
        "global_batches_per_epoch": batches,
        "batch_size": BATCH_SIZE,
        "scheduled_uint32_indices": len(FOLDS) * len(SEEDS) * EPOCHS * batches * BATCH_SIZE,
    }
    _fail(report.get("counts") == expected_counts, "report counts differ from decoded shape")
    expected_checks = {
        "parent_protocol_clean_remount_verified": require_parent_clean_remount,
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
    }
    _fail(report.get("checks") == expected_checks,
          "report check inventory/values are not the frozen safe state")
    authorization = report.get("authorization")
    _fail(isinstance(authorization, dict), "report lacks authorization section")
    _fail(authorization.get("bounded_smoke_authorized") is True,
          "report does not authorize the bounded smoke")
    _fail(authorization.get("full_training_authorized") is False,
          "report unexpectedly authorizes full training")


def _verify_binary_and_audits(
    root: Path,
    catalog: list[dict[str, object]],
    batches: int,
    fit_indices: Mapping[tuple[int, str, int], list[int]],
    units: list[dict[str, object]],
    audits: Mapping[tuple[int, int, str, int], dict[str, str]],
) -> None:
    binary_path = root / INDEX_NAME
    row_struct = struct.Struct("<64I")
    unit_hashes: dict[int, set[str]] = defaultdict(set)
    with binary_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            for unit in units:
                fold = int(unit["outer_fold"])
                seed = int(unit["training_seed"])
                unit_index = int(unit["unit_index"])
                offset = int(unit["byte_offset"])
                length = int(unit["byte_length"])
                digest = hashlib.sha256(payload[offset:offset + length]).hexdigest()
                _fail(digest == unit["unit_sha256"],
                      f"schedule unit hash mismatch: unit {unit_index}")
                unit_hashes[fold].add(digest)

                fit_sets = {
                    cell: set(fit_indices[(fold, cell[0], cell[1])]) for cell in CELLS
                }
                rows_by_identity: dict[tuple[str, int], dict[str, list[int]]] = {
                    cell: defaultdict(list) for cell in CELLS
                }
                for cell in CELLS:
                    for index in fit_sets[cell]:
                        rows_by_identity[cell][str(catalog[index]["normalized_text_sha256"])].append(index)
                identity_uses = {
                    cell: Counter({identity: 0 for identity in rows_by_identity[cell]})
                    for cell in CELLS
                }
                row_uses = Counter({index: 0 for cell in CELLS for index in fit_sets[cell]})

                for epoch_index in range(EPOCHS):
                    for batch_index in range(batches):
                        batch_offset = offset + (
                            (epoch_index * batches + batch_index) * BATCH_SIZE * 4
                        )
                        indices = row_struct.unpack_from(payload, batch_offset)
                        for index in indices:
                            _fail(index < len(catalog),
                                  f"out-of-range catalog index in unit {unit_index}")
                        cells = Counter(
                            (str(catalog[index]["reading_task"]), int(catalog[index]["pseudo_group"]))
                            for index in indices
                        )
                        _fail(cells == Counter({cell: EXAMPLES_PER_CELL for cell in CELLS}),
                              f"task/pseudo batch imbalance in unit {unit_index}")
                        identities = [
                            str(catalog[index]["normalized_text_sha256"]) for index in indices
                        ]
                        _fail(len(set(identities)) == BATCH_SIZE,
                              f"repeated normalized-text identity in unit {unit_index}")
                        for index in indices:
                            cell = (
                                str(catalog[index]["reading_task"]),
                                int(catalog[index]["pseudo_group"]),
                            )
                            _fail(index in fit_sets[cell],
                                  f"non-fit row scheduled in outer fold {fold}")
                            identity = str(catalog[index]["normalized_text_sha256"])
                            identity_uses[cell][identity] += 1
                            row_uses[index] += 1

                for cell in CELLS:
                    identity_values = identity_uses[cell]
                    _fail(bool(identity_values), f"empty fit identity set: {fold}/{cell}")
                    identity_min = min(identity_values.values())
                    identity_max = max(identity_values.values())
                    identity_gap = identity_max - identity_min
                    _fail(identity_gap <= 1,
                          f"identity exposure gap exceeds one: fold={fold} seed={seed} cell={cell}")
                    row_gap_max = 0
                    for row_group in rows_by_identity[cell].values():
                        uses = [row_uses[index] for index in row_group]
                        row_gap_max = max(row_gap_max, max(uses) - min(uses))
                    _fail(row_gap_max <= 1,
                          f"conditional row exposure gap exceeds one: fold={fold} seed={seed} cell={cell}")
                    covered = sum(row_uses[index] > 0 for index in fit_sets[cell])
                    _fail(covered == len(fit_sets[cell]),
                          f"fit-row coverage incomplete: fold={fold} seed={seed} cell={cell}")
                    audit = audits[(fold, seed, cell[0], cell[1])]
                    recomputed = {
                        "fit_trial_rows": len(fit_sets[cell]),
                        "fit_unique_text_identities": len(rows_by_identity[cell]),
                        "scheduled_slots": EPOCHS * batches * EXAMPLES_PER_CELL,
                        "identity_use_min": identity_min,
                        "identity_use_max": identity_max,
                        "identity_use_gap": identity_gap,
                        "conditional_row_use_gap_max": row_gap_max,
                        "covered_fit_rows": covered,
                    }
                    for field, expected in recomputed.items():
                        _fail(_as_int(audit[field], field) == expected,
                              f"schedule audit mismatch in {field}: fold={fold} seed={seed} cell={cell}")
    for fold in FOLDS:
        _fail(len(unit_hashes[fold]) == len(SEEDS),
              f"seed-specific schedules are not all distinct in fold {fold}")


def verify(
    output_root: Path,
    expected_schedule_contract_sha256: str,
    expected_parent_protocol_report_sha256: str,
    *,
    expected_catalog_rows: int | None = PRODUCTION_CATALOG_ROWS,
    expected_batches_per_epoch: int | None = PRODUCTION_BATCHES_PER_EPOCH,
    expected_shape: Sequence[int] | None = None,
    require_parent_clean_remount: bool = True,
    allowed_extra_files: Sequence[str] = (),
) -> dict[str, object]:
    """Verify a mounted schedule without trusting its self-declared checks.

    ``expected_catalog_rows``, ``expected_batches_per_epoch`` and
    ``expected_shape`` are injectable so regression tests can use small frozen
    fixtures.  Production verification defaults to the prospectively known
    9,011-row catalog and B=105; both are also independently derived from the
    mounted catalog rather than trusted from the report or manifest.

    ``allowed_extra_files`` is reserved for an enclosing sealed-artifact
    verifier.  It must contain exact root-level basenames; the default remains
    the original strict six-file inventory.
    """

    _fail(_is_sha256(expected_schedule_contract_sha256),
          "expected child-contract digest is not lowercase SHA-256")
    _fail(_is_sha256(expected_parent_protocol_report_sha256),
          "expected parent-report digest is not lowercase SHA-256")
    root = Path(output_root)
    _verify_inventory(root, allowed_extra_files)
    report = _read_json(root / REPORT_NAME)
    manifest = _read_json(root / MANIFEST_NAME)
    actual_hashes = _verify_hash_bindings(root, report, manifest)

    catalog_rows = _read_csv(root / CATALOG_NAME, CATALOG_FIELDS)
    catalog = _parse_catalog(catalog_rows)
    if expected_catalog_rows is not None:
        _fail(len(catalog) == expected_catalog_rows,
              f"catalog row count mismatch: expected {expected_catalog_rows}, got {len(catalog)}")
    batches, fit_counts, fit_indices = _derived_fit_layout(catalog)
    if expected_batches_per_epoch is not None:
        _fail(batches == expected_batches_per_epoch,
              f"derived B mismatch: expected {expected_batches_per_epoch}, got {batches}")
    derived_shape = [len(FOLDS) * len(SEEDS), EPOCHS, batches, BATCH_SIZE]
    if expected_shape is not None:
        _fail(list(expected_shape) == derived_shape,
              f"derived shape mismatch: expected {list(expected_shape)}, got {derived_shape}")

    binary_bytes = (root / INDEX_NAME).stat().st_size
    unit_rows = _read_csv(root / UNITS_NAME, UNIT_FIELDS)
    units = _parse_units(unit_rows, batches, binary_bytes)
    audit_rows = _read_csv(root / AUDIT_NAME, AUDIT_FIELDS)
    audits = _parse_audits(audit_rows)
    _verify_manifest_and_report(
        report,
        manifest,
        expected_schedule_contract_sha256,
        expected_parent_protocol_report_sha256,
        len(catalog),
        batches,
        fit_counts,
        units,
        require_parent_clean_remount,
    )
    _verify_binary_and_audits(root, catalog, batches, fit_indices, units, audits)
    return {
        "status": "pass",
        "schedule_contract_sha256": expected_schedule_contract_sha256,
        "parent_protocol_report_sha256": expected_parent_protocol_report_sha256,
        "schedule_report_sha256": sha256(root / REPORT_NAME),
        "schedule_manifest_sha256": actual_hashes[MANIFEST_NAME],
        "catalog_rows": len(catalog),
        "schedule_units": len(units),
        "epochs_per_unit": EPOCHS,
        "global_batches_per_epoch": batches,
        "batch_size": BATCH_SIZE,
        "shape": derived_shape,
        "audit_rows": len(audits),
        "applicable_arms": list(ARMS),
        "bounded_smoke_authorized": True,
        "full_training_authorized": False,
        "held_out_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-schedule-contract-sha256", required=True)
    parser.add_argument("--expected-parent-protocol-report-sha256", required=True)
    parser.add_argument("--expected-catalog-rows", type=int, default=PRODUCTION_CATALOG_ROWS)
    parser.add_argument(
        "--expected-batches-per-epoch", type=int,
        default=PRODUCTION_BATCHES_PER_EPOCH,
    )
    parser.add_argument("--expected-shape", type=int, nargs=4)
    args = parser.parse_args()
    result = verify(
        args.output_root,
        args.expected_schedule_contract_sha256,
        args.expected_parent_protocol_report_sha256,
        expected_catalog_rows=args.expected_catalog_rows,
        expected_batches_per_epoch=args.expected_batches_per_epoch,
        expected_shape=args.expected_shape,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("TASK-SEGMENTED TRAINING SCHEDULE OUTPUT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
