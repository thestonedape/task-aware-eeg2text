"""Build the metadata-only factor-recoverability and evaluation supplement.

The supplement freezes labels, eligibility, subject-held-out training exclusions,
cross-task duplicate identities, probe conditions, metrics, seeds, and admission
rules.  It reads canonical metadata only and never exposes the held-out test rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.build_protocol_manifests import (  # noqa: E402
    as_int,
    audit_split_identity,
    normalized_text,
    read_index,
    sha256,
    stable_hash,
    write_csv,
)


SR_LABELS = ("negative", "neutral", "positive")
RELATION_LABELS = (
    "AWARD",
    "EDUCATION",
    "EMPLOYER",
    "FOUNDER",
    "JOB_TITLE",
    "NATIONALITY",
    "POLITICAL_AFFILIATION",
    "VISITED",
    "WIFE",
)
TSR_LABELS = (*RELATION_LABELS, "CONTROL")

FACTORS = {
    "sr_sentiment_3": {
        "mask": "mask_sr_sentiment_3",
        "label": "sr_sentiment_3",
        "target_type": "multiclass",
        "ontology": list(SR_LABELS),
        "scope": "SR auxiliary supervision",
        "primary_metric": "macro_f1",
    },
    "nr_relation_content": {
        "mask": "mask_nr_relation_content",
        "label": "nr_relation_content",
        "target_type": "multilabel",
        "ontology": list(RELATION_LABELS),
        "negative_label": "NO-RELATION",
        "delimiter": ";",
        "scope": "ZuCo1 NR auxiliary diagnostic only",
        "primary_metric": "macro_f1",
    },
    "tsr_instruction_relation": {
        "mask": "mask_tsr_instruction_relation",
        "label": "tsr_instruction_relation",
        "target_type": "multiclass",
        "ontology": list(TSR_LABELS),
        "scope": "TSR instruction target; never an inference input",
        "primary_metric": "macro_f1",
    },
    "length_words_whitespace_v1": {
        "mask": None,
        "label": "length_words_whitespace_v1",
        "target_type": "regression",
        "scope": "shared SemKey-style baseline",
        "primary_metric": "negative_mae",
    },
}

REQUIRED_LABEL_FIELDS = {
    field
    for spec in FACTORS.values()
    for field in (spec.get("mask"), spec["label"])
    if field is not None
}

ARTIFACTS = (
    "recoverability_rows.csv",
    "subject_folds.csv",
    "factor_fold_coverage.csv",
    "duplicated_sentence_consistency.csv",
    "recoverability_registry.json",
)


def canonical_target(factor_id: str, row: dict[str, str]) -> tuple[str, str]:
    value = str(row[FACTORS[factor_id]["label"]]).strip()
    if factor_id == "sr_sentiment_3":
        if value not in SR_LABELS:
            raise ValueError(f"{row['trial_id']}: invalid SR label {value!r}")
        return value, value
    if factor_id == "nr_relation_content":
        atoms = [item.strip().upper() for item in value.split(";") if item.strip()]
        if not atoms or len(atoms) != len(set(atoms)):
            raise ValueError(f"{row['trial_id']}: invalid NR multi-label target {value!r}")
        if "NO-RELATION" in atoms:
            if atoms != ["NO-RELATION"]:
                raise ValueError(f"{row['trial_id']}: NO-RELATION cannot coexist with positives")
            return "NO-RELATION", "NO-RELATION"
        unknown = sorted(set(atoms) - set(RELATION_LABELS))
        if unknown:
            raise ValueError(f"{row['trial_id']}: unknown NR labels {unknown}")
        atoms = sorted(atoms)
        return ";".join(atoms), ";".join(atoms)
    if factor_id == "tsr_instruction_relation":
        value = value.upper()
        if value not in TSR_LABELS:
            raise ValueError(f"{row['trial_id']}: invalid TSR label {value!r}")
        return value, value
    if factor_id == "length_words_whitespace_v1":
        length = as_int(value, factor_id)
        if length <= 0:
            raise ValueError(f"{row['trial_id']}: non-positive sentence length")
        return str(length), ""
    raise KeyError(factor_id)


def build_factor_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (item["split"], item["trial_id"])):
        for factor_id, spec in FACTORS.items():
            mask_field = spec.get("mask")
            eligible = True if mask_field is None else as_int(row[str(mask_field)], str(mask_field)) == 1
            if not eligible:
                continue
            target, atoms = canonical_target(factor_id, row)
            if factor_id == "sr_sentiment_3" and row["reading_task"] != "SR":
                raise ValueError("SR target escaped its task mask")
            if factor_id == "nr_relation_content" and not (
                row["dataset_version"] == "ZuCo1" and row["reading_task"] == "NR"
            ):
                raise ValueError("NR relation target escaped ZuCo1 NR scope")
            if factor_id == "tsr_instruction_relation" and row["reading_task"] != "TSR":
                raise ValueError("TSR target escaped its task mask")
            output.append(
                {
                    "trial_id": row["trial_id"],
                    "split": row["split"],
                    "dataset_version": row["dataset_version"],
                    "reading_task": row["reading_task"],
                    "subject_id": row["subject_id"],
                    "text_uid": row["text_uid"],
                    "normalized_text_sha256": stable_hash(normalized_text(row["text"])),
                    "cohort": row["cohort"],
                    "factor_id": factor_id,
                    "target_type": spec["target_type"],
                    "target_value": target,
                    "target_atoms": atoms,
                    "subject_fold_id": f"loso::{row['subject_id']}" if row["split"] == "val" else "",
                    "oracle_policy": "supervision_only_never_probe_input",
                }
            )
    return output


def build_subject_folds(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    subjects = sorted({row["subject_id"] for row in validation})
    output: list[dict[str, object]] = []
    for subject in subjects:
        excluded = sum(row["subject_id"] == subject for row in train)
        eval_rows = sum(row["subject_id"] == subject for row in validation)
        if excluded == 0 or eval_rows == 0:
            raise ValueError(f"subject fold {subject!r} lacks train exclusion or validation rows")
        output.append(
            {
                "fold_id": f"loso::{subject}",
                "held_out_subject_id": subject,
                "training_split": "train",
                "evaluation_split": "val",
                "training_exclusion_rule": "subject_id_equals_held_out_subject_id",
                "training_rows_total": len(train),
                "training_rows_excluded": excluded,
                "training_rows_eligible": len(train) - excluded,
                "evaluation_rows_total": len(validation),
                "evaluation_rows_eligible": eval_rows,
            }
        )
    return output


def target_atoms(items: list[dict[str, object]]) -> set[str]:
    atoms: set[str] = set()
    for item in items:
        value = str(item["target_atoms"])
        atoms.update(part for part in value.split(";") if part and part != "NO-RELATION")
        if value == "NO-RELATION":
            atoms.add(value)
    return atoms


def coverage_row(
    fold_id: str,
    held_out_subject: str,
    factor_id: str,
    train_items: list[dict[str, object]],
    val_items: list[dict[str, object]],
) -> dict[str, object]:
    target_type = str(FACTORS[factor_id]["target_type"])
    if target_type == "regression":
        train_classes = val_classes = ""
        fit_viable = len(train_items) >= 2
    else:
        train_classes = len(target_atoms(train_items))
        val_classes = len(target_atoms(val_items))
        fit_viable = train_classes >= 2
    return {
        "fold_id": fold_id,
        "fold_mode": "ordinary" if fold_id == "ordinary" else "subject_held_out",
        "held_out_subject_id": held_out_subject,
        "factor_id": factor_id,
        "target_type": target_type,
        "train_eligible_rows": len(train_items),
        "validation_eligible_rows": len(val_items),
        "train_observed_target_atoms": train_classes,
        "validation_observed_target_atoms": val_classes,
        "fit_viable": int(fit_viable),
        "evaluation_available": int(bool(val_items)),
        "fold_included": int(fit_viable and bool(val_items)),
    }


def build_factor_fold_coverage(
    factor_rows: list[dict[str, object]], folds: list[dict[str, object]]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for factor_id in FACTORS:
        train_items = [
            row for row in factor_rows if row["split"] == "train" and row["factor_id"] == factor_id
        ]
        val_items = [
            row for row in factor_rows if row["split"] == "val" and row["factor_id"] == factor_id
        ]
        output.append(coverage_row("ordinary", "", factor_id, train_items, val_items))
        for fold in folds:
            subject = str(fold["held_out_subject_id"])
            fold_train = [row for row in train_items if row["subject_id"] != subject]
            fold_val = [row for row in val_items if row["subject_id"] == subject]
            output.append(
                coverage_row(str(fold["fold_id"]), subject, factor_id, fold_train, fold_val)
            )
    return output


def build_duplicate_catalog(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[normalized_text(row["text"])].append(row)
    output: list[dict[str, object]] = []
    for text, members in sorted(grouped.items(), key=lambda item: stable_hash(item[0])):
        tasks = sorted({row["reading_task"] for row in members})
        if len(tasks) < 2:
            continue
        splits = {row["split"] for row in members}
        if len(splits) != 1:
            raise ValueError("cross-task duplicate crosses canonical split")
        split = next(iter(splits))
        output.append(
            {
                "normalized_text_sha256": stable_hash(text),
                "split": split,
                "reading_tasks": "|".join(tasks),
                "dataset_versions": "|".join(sorted({row["dataset_version"] for row in members})),
                "text_uids": "|".join(sorted({row["text_uid"] for row in members})),
                "subject_ids": "|".join(sorted({row["subject_id"] for row in members})),
                "trial_ids": "|".join(sorted(row["trial_id"] for row in members)),
                "trial_count": len(members),
                "validation_consistency_eligible": int(split == "val"),
            }
        )
    return output


def registry(index_sha256: str, validation_duplicate_count: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_index_sha256": index_sha256,
        "development_splits": ["train", "val"],
        "held_out_test_accessed": False,
        "seeds": [20260716, 20260717, 20260718, 20260719, 20260720],
        "factors": FACTORS,
        "unavailable_factors": {
            "semkey_sentiment_2": "not_fabricated",
            "topic_label": "not_fabricated",
            "gpt2_mean_nll_v1": "not_fabricated",
            "keywords": "held_pending_versioned_provenance_and_coverage",
        },
        "conditions": {
            "task_only": {
                "inputs": ["reading_task"],
                "encoding": "one_hot",
            },
            "metadata_only": {
                "inputs": ["dataset_version", "reading_task", "subject_id", "cohort"],
                "optional_input": "length_words_whitespace_v1_except_when_length_is_target",
                "encoding": "one_hot_plus_train_standardized_numeric",
            },
            "text_only_diagnostic": {
                "classification": "train_fit_word_tfidf_ngram_1_2_min_df_2_max_features_50000",
                "length_target": "exact_whitespace_count_oracle_diagnostic",
                "deployable": False,
            },
            "frozen_glim_correct": {
                "input": "released_checkpoint_1024d_global_vector_from_correct_EEG",
                "warning": "contains_released_task_dataset_subject_conditioning_so_not_named_EEG_only",
            },
            "frozen_glim_plus_explicit_metadata": {
                "input": "frozen_glim_correct_concatenated_with_metadata_only_inputs",
            },
            "frozen_glim_matched_wrong_eeg": {
                "input": "task_dataset_subject_length_matched_donor_EEG_with_target_metadata_unchanged",
                "donor_source": "frozen_wrong_eeg_donors.csv",
            },
            "frozen_glim_zero_eeg": {"input": "zero_EEG_with_target_metadata_unchanged"},
            "frozen_glim_gaussian_eeg": {
                "input": "train_scale_matched_seeded_Gaussian_EEG_with_target_metadata_unchanged"
            },
        },
        "probe_models": {
            "multiclass": {
                "model": "L2_logistic_regression",
                "selection": "train_only_group_CV_by_text_uid",
                "grid_C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": "balanced",
            },
            "multilabel": {
                "model": "one_vs_rest_L2_logistic_regression",
                "selection": "train_only_group_CV_by_text_uid",
                "grid_C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": "balanced_per_label",
                "threshold": "0.5_primary_train_only_threshold_sensitivity_secondary",
            },
            "regression": {
                "model": "ridge_regression",
                "selection": "train_only_group_CV_by_text_uid",
                "grid_alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "normalization": "fit_on_training_rows_only",
            },
        },
        "metrics": {
            "multiclass": ["macro_f1", "balanced_accuracy", "accuracy", "ovr_auroc_when_defined"],
            "multilabel": ["macro_f1", "micro_f1", "macro_average_precision", "exact_match"],
            "regression": ["mae", "spearman_r", "r2"],
            "chance": {
                "multiclass": "training_majority_and_training_prior_stratified",
                "multilabel": "per_label_training_prevalence",
                "regression": "training_mean_and_training_median",
            },
        },
        "uncertainty": {
            "bootstrap_replicates": 5000,
            "paired_unit": "trial_id",
            "clusters": ["subject_id", "normalized_text_sha256", "two_way_subject_by_text"],
            "confidence_level": 0.95,
            "multiplicity": "Holm_alpha_0.05_across_four_factors_per_planned_contrast",
        },
        "subject_held_out": {
            "policy": "for_each_validation_subject_fit_on_train_rows_excluding_same_global_subject_id",
            "hyperparameters": "selected_inside_each_eligible_training_fold_only",
            "report_separately_from_ordinary_frozen_validation": True,
        },
        "duplicate_consistency": {
            "identity": "same_normalized_text_sha256_in_two_or_more_reading_tasks_within_one_split",
            "validation_group_count": validation_duplicate_count,
            "status": (
                "available" if validation_duplicate_count else "unavailable_under_frozen_validation_split"
            ),
            "split_is_not_changed_to_create_this_analysis": True,
        },
        "admission_rule": {
            "minimum_validation_rows": 50,
            "minimum_validation_subjects": 5,
            "planned_contrasts": [
                "frozen_glim_correct_minus_metadata_only",
                "frozen_glim_correct_minus_frozen_glim_matched_wrong_eeg",
            ],
            "criterion": (
                "both paired utility deltas positive with Holm-adjusted 95pct lower confidence bound above zero; "
                "subject-held-out utility delta must have the same direction"
            ),
            "utility": "macro_f1_for_classification_and_negative_mae_for_regression",
            "text_only_diagnostic_is_not_an_admission_comparator": True,
            "null_results_retained": True,
            "target_label_never_probe_input": True,
        },
    }


def build(dataset_root: Path, output_root: Path, expected_index_sha256: str | None = None) -> dict[str, object]:
    all_rows, source_manifest, index_path = read_index(dataset_root)
    if expected_index_sha256 and source_manifest["index_sha256"] != expected_index_sha256:
        raise ValueError("canonical index does not match the frozen evaluation source")
    missing = REQUIRED_LABEL_FIELDS - set(all_rows[0])
    if missing:
        raise ValueError(f"canonical index missing recoverability fields: {sorted(missing)}")
    split_audit = audit_split_identity(all_rows)
    development = [row for row in all_rows if row["split"] in {"train", "val"}]
    if {row["split"] for row in development} != {"train", "val"}:
        raise ValueError("both train and validation splits are required")
    if any(row["split"] == "test" for row in development):
        raise AssertionError("held-out test row entered recoverability protocol")

    factor_rows = build_factor_rows(development)
    folds = build_subject_folds(development)
    coverage = build_factor_fold_coverage(factor_rows, folds)
    duplicates = build_duplicate_catalog(development)
    validation_duplicates = sum(int(row["validation_consistency_eligible"]) for row in duplicates)
    registry_data = registry(str(source_manifest["index_sha256"]), validation_duplicates)

    ordinary = [row for row in coverage if row["fold_id"] == "ordinary"]
    if len(ordinary) != len(FACTORS) or not all(int(row["fit_viable"]) for row in ordinary):
        raise ValueError("one or more ordinary factor probes are not fit-viable")
    if not all(int(row["evaluation_available"]) for row in ordinary):
        raise ValueError("one or more factors have no frozen-validation labels")

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "recoverability_rows.csv",
        [
            "trial_id", "split", "dataset_version", "reading_task", "subject_id", "text_uid",
            "normalized_text_sha256", "cohort", "factor_id", "target_type", "target_value",
            "target_atoms", "subject_fold_id", "oracle_policy",
        ],
        factor_rows,
    )
    write_csv(
        output_root / "subject_folds.csv",
        [
            "fold_id", "held_out_subject_id", "training_split", "evaluation_split",
            "training_exclusion_rule", "training_rows_total", "training_rows_excluded",
            "training_rows_eligible", "evaluation_rows_total", "evaluation_rows_eligible",
        ],
        folds,
    )
    write_csv(
        output_root / "factor_fold_coverage.csv",
        [
            "fold_id", "fold_mode", "held_out_subject_id", "factor_id", "target_type",
            "train_eligible_rows", "validation_eligible_rows", "train_observed_target_atoms",
            "validation_observed_target_atoms", "fit_viable", "evaluation_available", "fold_included",
        ],
        coverage,
    )
    write_csv(
        output_root / "duplicated_sentence_consistency.csv",
        [
            "normalized_text_sha256", "split", "reading_tasks", "dataset_versions", "text_uids",
            "subject_ids", "trial_ids", "trial_count", "validation_consistency_eligible",
        ],
        duplicates,
    )
    (output_root / "recoverability_registry.json").write_text(
        json.dumps(registry_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    factor_counts: dict[str, dict[str, int]] = {}
    for factor_id in FACTORS:
        factor_counts[factor_id] = {
            split: sum(
                row["factor_id"] == factor_id and row["split"] == split for row in factor_rows
            )
            for split in ("train", "val")
        }
    report = {
        "status": "pass",
        "schema_version": 1,
        "source": {
            "index": str(index_path.relative_to(dataset_root)).replace("\\", "/"),
            "index_sha256": source_manifest["index_sha256"],
            "source_dataframe_sha256": source_manifest.get("source_dataframe_sha256", ""),
        },
        "counts": {
            "development_rows": len(development),
            "train_rows": sum(row["split"] == "train" for row in development),
            "validation_rows": sum(row["split"] == "val" for row in development),
            "recoverability_rows": len(factor_rows),
            "subject_folds": len(folds),
            "factor_fold_rows": len(coverage),
            "cross_task_duplicate_groups": len(duplicates),
            "validation_duplicate_groups": validation_duplicates,
            "factor_rows": factor_counts,
        },
        "checks": {
            **split_audit,
            "held_out_test_accessed": False,
            "all_factor_targets_valid": True,
            "ordinary_factor_probes_fit_viable": True,
            "ordinary_factor_validation_available": True,
            "subject_fold_training_exclusions_nonempty": all(
                int(row["training_rows_excluded"]) > 0 for row in folds
            ),
            "subject_fold_validation_rows_nonempty": all(
                int(row["evaluation_rows_eligible"]) > 0 for row in folds
            ),
            "validation_tuning_permitted": False,
            "target_label_as_probe_input_permitted": False,
            "missing_semkey_labels_fabricated": False,
            "frozen_glim_representation_called_eeg_only": False,
        },
        "duplicate_consistency_status": registry_data["duplicate_consistency"]["status"],
        "artifact_sha256": {name: sha256(output_root / name) for name in ARTIFACTS},
    }
    (output_root / "recoverability_contract_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-index-sha256")
    args = parser.parse_args()
    report = build(args.dataset_root, args.output_root, args.expected_index_sha256)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
