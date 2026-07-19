from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.aggregate_task_segmented_full_shards import (
    AggregationSpec,
    CONFIRMATION_SCORE_FIELDS,
    FULL_EXECUTION_CONTRACT_SHA256,
    HISTORY_FIELDS,
    PREDICTION_FIELDS,
    PRESERVED_EVIDENCE_SHA256,
    RUN_FIELDS,
    ShardInput,
    aggregate,
)


ARMS = ("global_mixed", "true_task_segmented", "pseudo_task_segmented")
LAUNCH_SHA = hashlib.sha256(b"synthetic launch authorization").hexdigest()
SPEC = AggregationSpec(
    arms=ARMS,
    folds=(0,),
    seeds=(7,),
    fold_counts=((0, 2),),
    task_counts=(("NR", 1), ("TSR", 1)),
    pool_size=3,
    epochs=1,
    batches_per_epoch=1,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _rewrite_csv(path: Path, mutate) -> None:
    fields, rows = _read_csv(path)
    mutate(rows)
    _write_csv(path, tuple(fields), rows)


def _report(source: str = "synthetic-preserved-shard") -> dict[str, object]:
    return {
        "status": "pass",
        "schema_version": 1,
        "preserved_source_id": source,
        "full_shard_manifest_sha256": _hash("manifest"),
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "launch_authorization_sha256": LAUNCH_SHA,
        "outer_fold": 0,
        "training_seed": 7,
        "arms": list(ARMS),
        "epochs_per_arm": 1,
        "optimizer_steps_per_arm": 1,
        "total_optimizer_steps": 3,
        "checkpoint_rows_per_arm": 4,
        "confirmation_prediction_rows": 12,
        "confirmation_candidate_score_rows": 36,
        "best_epochs": {arm: 1 for arm in ARMS},
        "schedule_unit_sha256": _hash("schedule-unit"),
        "verified_artifact_sha256": {"synthetic.txt": _hash("synthetic")},
        "paired_initial_state": True,
        "epoch_zero_eligible": True,
        "earliest_tie_selection_verified": True,
        "append_only_strict_incumbent_history_verified": True,
        "partial_scientific_decision_permitted": False,
        "correct_and_matched_wrong_provenance_verified": True,
        "preserved_protocol_evidence_verified": True,
        "preserved_schedule_evidence_verified": True,
        "recomputed_binding_sha256": _hash("binding"),
        "preserved_evidence_sha256": dict(PRESERVED_EVIDENCE_SHA256),
        "runtime_fingerprint_verified": True,
        "runtime_fingerprint_sha256": _hash("runtime-fingerprint"),
        "git_execution_boundary_verified": True,
        "checkpoint_deserialized": False,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
    }


def _refreeze_registry(base: Path, shard: ShardInput) -> tuple[Path, str]:
    report = json.loads(shard.verification_report.read_text(encoding="utf-8"))
    registry = {
        "schema_version": 1,
        "status": "frozen_after_all_p4b_shards_preserved",
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "launch_authorization_sha256": LAUNCH_SHA,
        "partial_scientific_decision_permitted": False,
        "held_out_test_accessed": False,
        "shards": [{
            "shard_id": "p4b-f0-s7",
            "outer_fold": 0,
            "training_seed": 7,
            "dataset_slug": "synthetic/p4b-f0-s7",
            "dataset_version": 1,
            "preserved_source_id": report["preserved_source_id"],
            "full_shard_manifest_sha256": report[
                "full_shard_manifest_sha256"],
            "verification_report_sha256": hashlib.sha256(
                shard.verification_report.read_bytes()).hexdigest(),
        }],
    }
    registry_path = base / "frozen_shard_registry.json"
    registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n",
                             encoding="utf-8")
    expected = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    (base / "expected_registry_sha256.txt").write_text(expected,
                                                        encoding="ascii")
    return registry_path, expected


def _build_shard(base: Path) -> ShardInput:
    root = base / "shard"
    root.mkdir()
    run_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    trials = (
        ("trial-nr", "NR", "subject-a"),
        ("trial-tsr", "TSR", "subject-b"),
    )
    for arm in ARMS:
        run_rows.append({
            "arm_id": arm,
            "outer_fold": 0,
            "training_seed": 7,
            "status": "complete",
            "run_mode": "full_scientific",
            "full_training_authorized": "true",
            "scientific_decision_permitted": "false",
            "official_validation_used_for_confirmation": "false",
            "held_out_test_accessed": "false",
        })
        history = []
        for epoch in range(2):
            history.append({
                "arm_id": arm,
                "outer_fold": 0,
                "training_seed": 7,
                "epoch": epoch,
                "optimizer_steps": epoch,
                "mean_train_loss": "" if epoch == 0 else "0.5",
                "checkpoint_macro_mrr": "0.5" if epoch == 0 else "0.6",
                "checkpoint_nr_mrr": "0.5" if epoch == 0 else "0.6",
                "checkpoint_tsr_mrr": "0.5" if epoch == 0 else "0.6",
                # The incumbent history is append-only: epoch zero and every
                # later strict improvement remain marked.
                "is_best": "true",
            })
        _write_csv(root / "runs" / arm / "training_history.csv", HISTORY_FIELDS, history)
        score_rows: list[dict[str, object]] = []
        for trial, task, subject in trials:
            target_text = _hash(f"target-{trial}")
            donor_text = _hash(f"donor-{trial}")
            negative_text = _hash(f"negative-{trial}")
            for condition in ("correct", "matched_wrong"):
                signal_trial = trial if condition == "correct" else f"donor-{trial}"
                signal_text = target_text if condition == "correct" else donor_text
                prediction_rows.append({
                    "arm_id": arm,
                    "training_seed": 7,
                    "outer_fold": 0,
                    "trial_id": trial,
                    "reading_task": task,
                    "subject_id": subject,
                    "normalized_text_sha256": target_text,
                    "signal_condition": condition,
                    # Positive rank is 2 because candidate 0 ties its score and
                    # wins the frozen candidate-rank tie break.
                    "positive_rank": 2,
                    "candidate_pool_size": 3,
                    "scientific_decision_permitted": "false",
                })
                candidates = (
                    (0, negative_text, False, False, "0.8"),
                    (1, target_text, True, False, "0.8"),
                    (2, donor_text, False, True, "0.1"),
                )
                for rank, text, positive, donor, score in candidates:
                    score_rows.append({
                        "arm_id": arm,
                        "training_seed": 7,
                        "outer_fold": 0,
                        "trial_id": trial,
                        "reading_task": task,
                        "subject_id": subject,
                        "normalized_text_sha256": target_text,
                        "signal_condition": condition,
                        "signal_trial_id": signal_trial,
                        "signal_subject_id": subject,
                        "signal_normalized_text_sha256": signal_text,
                        "candidate_rank": rank,
                        "candidate_normalized_text_sha256": text,
                        "candidate_text_target_id": f"target-{rank}",
                        "is_positive": str(positive).lower(),
                        "is_designated_donor_text": str(donor).lower(),
                        "score": score,
                    })
        _write_csv(
            root / "runs" / arm / "confirmation_candidate_scores.csv",
            CONFIRMATION_SCORE_FIELDS,
            score_rows,
        )
    _write_csv(root / "run_manifest.csv", RUN_FIELDS, run_rows)
    _write_csv(root / "confirmation_predictions.csv", PREDICTION_FIELDS,
               prediction_rows)
    report_path = base / "independent_verification_report.json"
    report_path.write_text(json.dumps(_report(), sort_keys=True) + "\n",
                           encoding="utf-8")
    shard = ShardInput(root, report_path)
    _refreeze_registry(base, shard)
    return shard


def _reverify(_root, report, _launch, _spec, _protocol_root, _schedule_root):
    return dict(report)


def _registry_kwargs(base: Path) -> dict[str, object]:
    protocol_root = base / "protocol"
    schedule_root = base / "schedule"
    protocol_root.mkdir(exist_ok=True)
    schedule_root.mkdir(exist_ok=True)
    return {
        "frozen_registry_path": base / "frozen_shard_registry.json",
        "expected_frozen_registry_sha256": (
            base / "expected_registry_sha256.txt").read_text(encoding="ascii"),
        "protocol_root": protocol_root,
        "schedule_root": schedule_root,
    }


def _decision(runs, predictions):
    if len(runs) != 3 or len(predictions) != 12:
        raise AssertionError("decision received an incomplete synthetic matrix")
    if any(row["scientific_decision_permitted"] != "true" for row in runs):
        raise AssertionError("decision received non-promoted run rows")
    if any(row["scientific_decision_permitted"] != "true"
           for row in predictions):
        raise AssertionError("decision received non-promoted prediction rows")
    return {
        "schema_version": 1,
        "status": "pass",
        "scientific_decision_permitted": True,
        "held_out_test_accessed": False,
        "continuation_decision": "continue_p4b",
    }


class FullMatrixAggregatorTests(unittest.TestCase):
    def _run(self, base: Path, shard: ShardInput, **kwargs):
        expected_registry_sha256 = (base / "expected_registry_sha256.txt").read_text(
            encoding="ascii")
        protocol_root = base / "protocol"
        schedule_root = base / "schedule"
        protocol_root.mkdir(exist_ok=True)
        schedule_root.mkdir(exist_ok=True)
        return aggregate(
            [shard], LAUNCH_SHA, base / "aggregate", spec=SPEC,
            frozen_registry_path=base / "frozen_shard_registry.json",
            expected_frozen_registry_sha256=expected_registry_sha256,
            protocol_root=protocol_root,
            schedule_root=schedule_root,
            reverify=_reverify, decision=_decision, **kwargs,
        )

    def test_success_recomputes_tied_rank_and_separates_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            result = self._run(base, shard)
            self.assertEqual(result["integrity_status"], "pass")
            self.assertEqual(result["scientific_decision_status"], "pass")
            self.assertEqual(result["run_manifest_rows"], 3)
            self.assertEqual(result["confirmation_prediction_rows"], 12)
            fields, predictions = _read_csv(
                base / "aggregate" / "confirmation_predictions.csv")
            self.assertEqual(fields, list(PREDICTION_FIELDS))
            self.assertEqual({row["positive_rank"] for row in predictions}, {"2"})
            self.assertEqual(
                {row["scientific_decision_permitted"] for row in predictions},
                {"true"},
            )
            _, runs = _read_csv(base / "aggregate" / "run_manifest.csv")
            self.assertEqual(
                {row["scientific_decision_permitted"] for row in runs}, {"true"})
            integrity = json.loads((base / "aggregate" / "integrity_report.json")
                                   .read_text(encoding="utf-8"))
            self.assertEqual(integrity["integrity_status"], "pass")
            self.assertEqual(integrity["scientific_decision_status"], "pass")
            self.assertFalse(integrity["held_out_test_accessed"])
            transition = integrity["scientific_decision_permission_transition"]
            self.assertFalse(
                transition["partial_shard_scientific_decision_permitted"])
            self.assertTrue(
                transition["complete_matrix_scientific_decision_permitted"])
            self.assertEqual(
                result["scientific_decision_permission_transition"], transition)
            expected_registry = (base / "expected_registry_sha256.txt").read_text(
                encoding="ascii")
            self.assertEqual(
                integrity["frozen_shard_registry_sha256"], expected_registry)
            self.assertEqual(
                result["frozen_shard_registry_sha256"], expected_registry)

    def test_production_shape_is_exactly_fifteen_shards_and_decision_rows(self) -> None:
        from evaluation.aggregate_task_segmented_full_shards import PRODUCTION_SPEC

        self.assertEqual(len(PRODUCTION_SPEC.expected_shards), 15)
        self.assertEqual(PRODUCTION_SPEC.expected_runs, 45)
        self.assertEqual(PRODUCTION_SPEC.expected_trials, 9011)
        self.assertEqual(PRODUCTION_SPEC.expected_predictions, 162198)

    def test_append_only_history_accepts_multiple_strict_incumbents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            _, rows = _read_csv(
                shard.artifact_root / "runs" / ARMS[0] / "training_history.csv")
            self.assertEqual([row["is_best"] for row in rows], ["true", "true"])
            result = self._run(base, shard)
            self.assertEqual(result["integrity_status"], "pass")

    def test_exact_tie_must_be_unmarked_and_retain_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            history = shard.artifact_root / "runs" / ARMS[0] / "training_history.csv"

            def make_tie_but_leave_mark(rows):
                for field in (
                    "checkpoint_macro_mrr", "checkpoint_nr_mrr",
                    "checkpoint_tsr_mrr",
                ):
                    rows[1][field] = rows[0][field]

            _rewrite_csv(history, make_tie_but_leave_mark)
            with self.assertRaisesRegex(
                    ValueError, "append-only strict-improvement history"):
                self._run(base, shard)

            _rewrite_csv(
                history,
                lambda rows: rows[1].__setitem__("is_best", "false"),
            )
            report = json.loads(shard.verification_report.read_text(encoding="utf-8"))
            report["best_epochs"][ARMS[0]] = 0
            shard.verification_report.write_text(json.dumps(report) + "\n",
                                                  encoding="utf-8")
            _refreeze_registry(base, shard)
            result = self._run(base, shard)
            self.assertEqual(result["integrity_status"], "pass")

    def test_negative_scientific_decision_keeps_integrity_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)

            def negative_decision(runs, predictions):
                result = _decision(runs, predictions)
                result["status"] = "fail"
                result["continuation_decision"] = "stop_p4b_permanently"
                return result

            result = aggregate(
                [shard], LAUNCH_SHA, base / "aggregate", spec=SPEC,
                reverify=_reverify, decision=negative_decision,
                **_registry_kwargs(base),
            )
            self.assertEqual(result["integrity_status"], "pass")
            self.assertEqual(result["scientific_decision_status"], "fail")
            integrity = json.loads((base / "aggregate" / "integrity_report.json")
                                   .read_text(encoding="utf-8"))
            self.assertEqual(integrity["integrity_status"], "pass")
            self.assertEqual(integrity["scientific_decision_status"], "fail")

    def test_rejects_missing_and_duplicate_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            with self.assertRaisesRegex(ValueError, "exact full-shard input count"):
                aggregate([], LAUNCH_SHA, base / "missing", spec=SPEC,
                          reverify=_reverify, decision=_decision,
                          **_registry_kwargs(base))
            two_seed_spec = AggregationSpec(
                arms=ARMS, folds=(0,), seeds=(7, 8), fold_counts=((0, 2),),
                task_counts=(("NR", 1), ("TSR", 1)), pool_size=3,
                epochs=1, batches_per_epoch=1,
            )
            registry_path = base / "frozen_shard_registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            second = dict(registry["shards"][0])
            second.update({
                "shard_id": "p4b-f0-s8",
                "training_seed": 8,
                "dataset_slug": "synthetic/p4b-f0-s8",
                "preserved_source_id": "synthetic-preserved-shard-8",
                "full_shard_manifest_sha256": _hash("manifest-8"),
                "verification_report_sha256": _hash("report-8"),
            })
            registry["shards"].append(second)
            registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n",
                                     encoding="utf-8")
            (base / "expected_registry_sha256.txt").write_text(
                hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "duplicate shard root"):
                aggregate([shard, shard], LAUNCH_SHA, base / "duplicate",
                          spec=two_seed_spec, reverify=_reverify,
                          decision=_decision, **_registry_kwargs(base))

    def test_rejects_unverified_wrong_unit_or_wrong_arms(self) -> None:
        mutations = (
            ("status", "fail", "verification status"),
            ("outer_fold", 1, "outside frozen matrix"),
            ("arms", [ARMS[0], ARMS[1]], "verified shard arms"),
            ("launch_authorization_sha256", _hash("wrong"),
             "verified launch authorization"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                shard = _build_shard(base)
                report = json.loads(shard.verification_report.read_text(encoding="utf-8"))
                report[field] = value
                shard.verification_report.write_text(json.dumps(report) + "\n",
                                                      encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self._run(base, shard)

    def test_rejects_incomplete_history_before_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            history = shard.artifact_root / "runs" / ARMS[0] / "training_history.csv"
            _rewrite_csv(history, lambda rows: rows.pop())
            called = [False]

            def forbidden_decision(_runs, _predictions):
                called[0] = True
                return _decision(_runs, _predictions)

            with self.assertRaisesRegex(ValueError, "training-history rows"):
                aggregate([shard], LAUNCH_SHA, base / "aggregate", spec=SPEC,
                          reverify=_reverify, decision=forbidden_decision,
                          **_registry_kwargs(base))
            self.assertFalse(called[0])
            self.assertFalse((base / "aggregate").exists())

    def test_rejects_any_partial_scientific_permission_tamper(self) -> None:
        mutations = (
            ("run", "prematurely permits a partial scientific decision"),
            ("prediction", "prematurely permits a partial scientific decision"),
            ("report", "partial-shard scientific-decision permission"),
        )
        for target, message in mutations:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                shard = _build_shard(base)
                if target == "run":
                    _rewrite_csv(
                        shard.artifact_root / "run_manifest.csv",
                        lambda rows: rows[0].__setitem__(
                            "scientific_decision_permitted", "true"),
                    )
                elif target == "prediction":
                    _rewrite_csv(
                        shard.artifact_root / "confirmation_predictions.csv",
                        lambda rows: rows[0].__setitem__(
                            "scientific_decision_permitted", "true"),
                    )
                else:
                    report = json.loads(
                        shard.verification_report.read_text(encoding="utf-8"))
                    report["partial_scientific_decision_permitted"] = True
                    shard.verification_report.write_text(
                        json.dumps(report) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self._run(base, shard)

    def test_rejects_out_of_band_registry_and_report_binding_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            registry_path = base / "frozen_shard_registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["status"] = "tampered"
            registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "out-of-band frozen-registry SHA256"):
                self._run(base, shard)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            registry_path = base / "frozen_shard_registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["shards"][0]["preserved_source_id"] = "wrong-preserved-source"
            registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n",
                                     encoding="utf-8")
            (base / "expected_registry_sha256.txt").write_text(
                hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError,
                                        "registry/report preserved source"):
                self._run(base, shard)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            # JSON whitespace preserves report semantics but changes its bytes;
            # the frozen registry must still reject it.
            shard.verification_report.write_text(
                shard.verification_report.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError,
                                        "registry verification-report file"):
                self._run(base, shard)

    def test_rejects_registry_extra_fields_and_nonpositive_dataset_version(self) -> None:
        mutations = (
            (lambda registry: registry.__setitem__("extra", True),
             "frozen-registry fields"),
            (lambda registry: registry["shards"][0].__setitem__(
                "dataset_version", 0), "invalid dataset version"),
        )
        for mutation, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                shard = _build_shard(base)
                registry_path = base / "frozen_shard_registry.json"
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                mutation(registry)
                registry_path.write_text(json.dumps(registry, sort_keys=True) + "\n",
                                         encoding="utf-8")
                (base / "expected_registry_sha256.txt").write_text(
                    hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                    encoding="ascii",
                )
                with self.assertRaisesRegex(ValueError, message):
                    self._run(base, shard)

    def test_rejects_incomplete_or_unrecomputed_predictions(self) -> None:
        for mutation, message in (
            (lambda rows: rows.pop(), "confirmation-prediction rows"),
            (lambda rows: rows[0].__setitem__("positive_rank", "1"),
             "raw-score rank recomputation"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                shard = _build_shard(base)
                _rewrite_csv(shard.artifact_root / "confirmation_predictions.csv",
                             mutation)
                with self.assertRaisesRegex(ValueError, message):
                    self._run(base, shard)

    def test_rejects_validation_test_and_pool_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            _rewrite_csv(
                shard.artifact_root / "run_manifest.csv",
                lambda rows: rows[0].__setitem__(
                    "official_validation_used_for_confirmation", "true"),
            )
            with self.assertRaisesRegex(ValueError, "used official validation"):
                self._run(base, shard)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            score_path = (shard.artifact_root / "runs" / ARMS[1]
                          / "confirmation_candidate_scores.csv")

            def drift(rows):
                rows[0]["candidate_normalized_text_sha256"] = _hash("pool drift")

            _rewrite_csv(score_path, drift)
            with self.assertRaisesRegex(ValueError, "candidate-pool drift"):
                self._run(base, shard)

    def test_rejects_report_test_drift_and_reverification_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)
            report = json.loads(shard.verification_report.read_text(encoding="utf-8"))
            report["held_out_test_accessed"] = True
            shard.verification_report.write_text(json.dumps(report) + "\n",
                                                  encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "held-out-test drift"):
                self._run(base, shard)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shard = _build_shard(base)

            def mismatched(
                _root, report, _launch, _spec, _protocol_root, _schedule_root,
            ):
                changed = dict(report)
                changed["status"] = "fail"
                return changed

            with self.assertRaisesRegex(ValueError, "re-verification report"):
                aggregate([shard], LAUNCH_SHA, base / "aggregate", spec=SPEC,
                          reverify=mismatched, decision=_decision,
                          **_registry_kwargs(base))


if __name__ == "__main__":
    unittest.main()
