"""Focused standard-library tests for the sealed full-shard verifier."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation import verify_task_segmented_full_shard_artifact as verifier


SHA = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
CONTRACT_SHA = "a" * 64
AUTH_SHA = "b" * 64
PROJECT_COMMIT = "c" * 40
SPEC = verifier.VerificationSpec(
    folds=(0, 1), seeds=(7,), fold_counts=(2, 2), epochs=2,
    batches_per_epoch=2, pool_size=3,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def arm_relative_hashes(root: Path, arm: str) -> dict[str, str]:
    output = {}
    for name in verifier.PER_ARM_FILES - {"run_summary.json"}:
        output[name] = verifier.sha256(root / "runs" / arm / name)
    return output


def reseal(root: Path, *, refresh_summaries: bool = True) -> str:
    if refresh_summaries:
        for arm in verifier.ARM_IDS:
            path = root / "runs" / arm / "run_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["artifact_sha256"] = arm_relative_hashes(root, arm)
            write_json(path, summary)
    artifacts = {
        relative: verifier.sha256(root / relative)
        for relative in sorted(verifier.NON_MANIFEST_FILES)
    }
    manifest_path = root / verifier.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = artifacts
    write_json(manifest_path, manifest)
    return verifier.sha256(manifest_path)


def build_preserved_inputs(
    base: Path,
    target_rows: list[tuple[str, str, str, str, str, str]],
) -> tuple[Path, Path, verifier.EvidenceHashes, str,
           dict[tuple[int, int], tuple[str, str]]]:
    protocol = base / "protocol"
    schedule = base / "schedule"
    protocol.mkdir()
    schedule.mkdir()
    special = {
        "target-0": ("NR", "S0", SHA("target-text-0")),
        "donor-0": ("NR", "S0", SHA("donor-text-0")),
        "target-1": ("TSR", "S1", SHA("target-text-1")),
        "donor-1": ("TSR", "S1", SHA("donor-text-1")),
        "checkpoint-0": ("NR", "Q0", SHA("checkpoint-text-0")),
        "checkpoint-1": ("TSR", "Q1", SHA("checkpoint-text-1")),
    }
    catalog_rows: list[dict[str, object]] = []
    ordered_trials = list(special) + [f"fit-{index}" for index in range(58)]
    for index, trial in enumerate(ordered_trials):
        task, subject, text_hash = special.get(
            trial, ("NR" if index % 2 == 0 else "TSR", f"F{index % 4}",
                    SHA(f"fit-text-{index}")),
        )
        catalog_rows.append({
            "trial_index": index, "trial_id": trial, "text_fold": index % 2,
            "dataset_version": "ZuCo2", "reading_task": task,
            "subject_id": subject, "normalized_text_sha256": text_hash,
            "text_target_id": f"catalog-text-target-{index}",
            "pseudo_group": index % 2, "length_words_whitespace_v1": 8 + index,
            "eeg_vector_file": "eeg.npz", "eeg_vector_offset": index,
            "eeg_vector_dim": 1024,
        })
    write_csv(schedule / "trial_catalog.csv", verifier.CATALOG_FIELDS, catalog_rows)

    indices = list(range(64)) * SPEC.optimizer_steps
    payload = b"".join(struct.pack("<I", index) for index in indices)
    schedule_sha = hashlib.sha256(payload).hexdigest()
    (schedule / "schedule_indices.u32le").write_bytes(payload)
    unit_rows = [{
        "unit_index": 0, "outer_fold": 0, "training_seed": 7,
        "epochs": SPEC.epochs, "batches_per_epoch": SPEC.batches_per_epoch,
        "batch_size": 64, "uint32_count": len(indices), "byte_offset": 0,
        "byte_length": len(payload), "unit_sha256": schedule_sha,
        "initialization_seed": 1,
    }]
    write_csv(schedule / "schedule_units.csv", verifier.UNIT_FIELDS, unit_rows)

    candidates: list[dict[str, object]] = []
    target_lookup = {row[0]: row for row in target_rows}
    checkpoint_rows = [
        ("checkpoint-0", "NR", "Q0", SHA("checkpoint-text-0")),
        ("checkpoint-1", "TSR", "Q1", SHA("checkpoint-text-1")),
    ]
    for partition, rows in (
        ("checkpoint", checkpoint_rows),
        ("confirmation", [(row[0], row[1], row[2], row[3]) for row in target_rows]),
    ):
        for trial, task, _subject, text_hash in rows:
            if partition == "confirmation":
                donor_hash = target_lookup[trial][5]
                hashes = (text_hash, donor_hash, SHA(f"third-{trial}"))
            else:
                hashes = (text_hash, SHA(f"cp-other-1-{trial}"),
                          SHA(f"cp-other-2-{trial}"))
            for rank, candidate_hash in enumerate(hashes):
                candidates.append({
                    "outer_fold": 0, "partition": partition,
                    "target_trial_id": trial, "candidate_rank": rank,
                    "candidate_normalized_text_sha256": candidate_hash,
                    "candidate_text_target_id": f"{trial}-text-target-{rank}",
                    "is_positive": int(rank == 0),
                    "is_designated_donor_text": int(
                        partition == "confirmation" and rank == 1
                    ),
                    "dataset_version": "ZuCo2", "reading_task": task,
                    "target_length": 10, "candidate_length": 10 + rank,
                    "absolute_length_difference": rank,
                    "selection_rule": "fixture",
                })
    write_csv(protocol / "candidate_pools.csv", verifier.CANDIDATE_FIELDS, candidates)

    donors = []
    for trial, task, subject, text_hash, donor_trial, donor_hash in target_rows:
        donors.append({
            "outer_fold": 0, "partition": "confirmation",
            "target_trial_id": trial, "donor_trial_id": donor_trial,
            "dataset_version": "ZuCo2", "reading_task": task,
            "subject_id": subject, "target_normalized_text_sha256": text_hash,
            "donor_normalized_text_sha256": donor_hash,
            "target_length": 10, "donor_length": 11,
            "absolute_length_difference": 1,
            "selection_rule": "fixture",
        })
    write_csv(protocol / "confirmation_donors.csv", verifier.DONOR_FIELDS, donors)

    hashes = verifier.EvidenceHashes(
        candidate_pools_sha256=verifier.sha256(protocol / "candidate_pools.csv"),
        confirmation_donors_sha256=verifier.sha256(
            protocol / "confirmation_donors.csv"
        ),
        schedule_indices_sha256=verifier.sha256(
            schedule / "schedule_indices.u32le"
        ),
        schedule_units_sha256=verifier.sha256(schedule / "schedule_units.csv"),
        trial_catalog_sha256=verifier.sha256(schedule / "trial_catalog.csv"),
    )
    write_json(base / "evidence_hashes.json", hashes.__dict__)
    index_sha = hashlib.sha256(
        b"".join(struct.pack("<I", index) for index in range(64))
    ).hexdigest()
    trial_sha = hashlib.sha256(
        ("\n".join(row["trial_id"] for row in catalog_rows) + "\n").encode("utf-8")
    ).hexdigest()
    batch_bindings = {
        (epoch, batch): (index_sha, trial_sha)
        for epoch in range(1, SPEC.epochs + 1)
        for batch in range(SPEC.batches_per_epoch)
    }
    return protocol, schedule, hashes, schedule_sha, batch_bindings


def build_fixture(root: Path) -> str:
    fold = 0
    seed = 7
    shard_id = "p4b-f0-s7"
    root.mkdir(parents=True)
    target_rows = [
        ("target-0", "NR", "S0", SHA("target-text-0"),
         "donor-0", SHA("donor-text-0")),
        ("target-1", "TSR", "S1", SHA("target-text-1"),
         "donor-1", SHA("donor-text-1")),
    ]
    _protocol, _schedule, _hashes, schedule_sha, batch_bindings = (
        build_preserved_inputs(root.parent, target_rows)
    )
    binding_source = {
        "project_commit": PROJECT_COMMIT,
        "runner_source_sha256": SHA("runner"),
        "adapter_source_sha256": SHA("adapter"),
        "task_treatment_pilots_source_sha256": SHA("task-treatment-pilots"),
        "full_execution_contract_sha256": CONTRACT_SHA,
        "launch_authorization_sha256": AUTH_SHA,
        "input_bindings": verifier.FULL_INPUT_BINDINGS,
        "outer_fold": fold, "training_seed": seed,
        "schedule_unit_sha256": schedule_sha,
        "git_worktree_clean": True,
        "git_submodules_clean": True,
    }
    binding_sha = verifier.recompute_binding_sha256(
        binding_source, verifier.EXACT_RUNTIME_ENVIRONMENT, SPEC
    )
    common = {
        "project_commit": PROJECT_COMMIT,
        "runner_source_sha256": SHA("runner"),
        "adapter_source_sha256": SHA("adapter"),
        "task_treatment_pilots_source_sha256": SHA("task-treatment-pilots"),
        "full_execution_contract_sha256": CONTRACT_SHA,
        "launch_authorization_sha256": AUTH_SHA,
        "binding_sha256": binding_sha,
        "input_bindings": verifier.FULL_INPUT_BINDINGS,
        "shard_id": shard_id,
        "schedule_unit_index": 0,
        "outer_fold": fold,
        "training_seed": seed,
        "schedule_unit_sha256": schedule_sha,
        "runtime_fingerprint": verifier.EXACT_RUNTIME_ENVIRONMENT,
        "runtime_fingerprint_sha256": verifier.runtime_fingerprint_sha256(
            verifier.EXACT_RUNTIME_ENVIRONMENT
        ),
        "observed_cuda_device_count": 1,
        "git_worktree_clean": True,
        "git_submodules_clean": True,
    }
    (root / "runs").mkdir(parents=True)
    for arm in verifier.ARM_IDS:
        (root / "runs" / arm).mkdir()

    run_rows = []
    for arm in verifier.ARM_IDS:
        run_rows.append({
            "arm_id": arm, "outer_fold": fold, "training_seed": seed,
            "status": "complete", "run_mode": "full_scientific",
            "full_training_authorized": "true",
            "scientific_decision_permitted": "false",
            "official_validation_used_for_confirmation": "false",
            "held_out_test_accessed": "false",
        })
    write_csv(root / "run_manifest.csv", verifier.RUN_FIELDS, run_rows)

    predictions: list[dict[str, object]] = []
    for arm in verifier.ARM_IDS:
        score_rows: list[dict[str, object]] = []
        for trial, task, subject, text_hash, donor_trial, donor_hash in target_rows:
            third_hash = SHA(f"third-{trial}")
            candidates = (text_hash, donor_hash, third_hash)
            for condition in ("correct", "matched_wrong"):
                signal_trial = trial if condition == "correct" else donor_trial
                signal_hash = text_hash if condition == "correct" else donor_hash
                scores = (0.9, 0.8, 0.1) if condition == "correct" else (0.5, 0.9, 0.1)
                predictions.append({
                    "arm_id": arm, "training_seed": seed, "outer_fold": fold,
                    "trial_id": trial, "reading_task": task, "subject_id": subject,
                    "normalized_text_sha256": text_hash,
                    "signal_condition": condition,
                    "positive_rank": 1 if condition == "correct" else 2,
                    "candidate_pool_size": SPEC.pool_size,
                    "scientific_decision_permitted": "false",
                })
                for rank, (candidate_hash, score) in enumerate(zip(candidates, scores)):
                    score_rows.append({
                        "arm_id": arm, "training_seed": seed, "outer_fold": fold,
                        "trial_id": trial, "reading_task": task, "subject_id": subject,
                        "normalized_text_sha256": text_hash,
                        "signal_condition": condition,
                        "signal_trial_id": signal_trial,
                        "signal_subject_id": subject,
                        "signal_normalized_text_sha256": signal_hash,
                        "candidate_rank": rank,
                        "candidate_normalized_text_sha256": candidate_hash,
                        "candidate_text_target_id": f"{trial}-text-target-{rank}",
                        "is_positive": int(rank == 0),
                        "is_designated_donor_text": int(rank == 1),
                        "score": score,
                    })
        write_csv(
            root / "runs" / arm / "confirmation_candidate_scores.csv",
            verifier.CONFIRMATION_SCORE_FIELDS, score_rows,
        )
    write_csv(root / "confirmation_predictions.csv", verifier.PREDICTION_FIELDS, predictions)

    shared_initial = SHA("paired-initial")
    for arm_index, arm in enumerate(verifier.ARM_IDS):
        arm_root = root / "runs" / arm
        (arm_root / "resume_checkpoint.pt").write_bytes(
            b"opaque-not-a-pytorch-checkpoint-resume-" + arm.encode("ascii")
        )
        (arm_root / "best_checkpoint.pt").write_bytes(
            b"opaque-not-a-pytorch-checkpoint-best-" + arm.encode("ascii")
        )
        steps: list[dict[str, object]] = []
        prior = shared_initial
        for offset in range(SPEC.optimizer_steps):
            epoch = offset // SPEC.batches_per_epoch + 1
            batch = offset % SPEC.batches_per_epoch
            post = SHA(f"{arm}-state-{offset}")
            steps.append({
                "outer_fold": fold, "training_seed": seed, "arm_id": arm,
                "epoch": epoch, "batch_index": batch, "global_step": offset + 1,
                "schedule_unit_sha256": schedule_sha,
                "batch_catalog_indices_sha256": batch_bindings[(epoch, batch)][0],
                "batch_trial_ids_sha256": batch_bindings[(epoch, batch)][1],
                "global_mask_key": verifier.stable_hash(
                    "p4b-global-mask-key-v1", 2026071806,
                    verifier.FULL_INPUT_BINDINGS["training_schedule"]["contract_sha256"],
                    schedule_sha, fold, seed, epoch, batch,
                ),
                "initial_state_sha256": shared_initial,
                "pre_step_state_sha256": prior, "loss": 1.0 / (offset + 1),
                "gradient_norm_preclip": 0.5,
                "post_step_state_sha256": post,
                "optimizer_state_sha256": SHA(f"{arm}-optimizer-{offset}"),
            })
            prior = post
        write_csv(arm_root / "step_trace.csv", verifier.STEP_FIELDS, steps)

        checkpoint_rows: list[dict[str, object]] = []
        rank_by_epoch = {0: (2, 2), 1: (1, 1), 2: (1, 1)}
        history: list[dict[str, object]] = []
        for epoch in range(SPEC.epochs + 1):
            ranks = rank_by_epoch[epoch]
            for trial_index, task in enumerate(("NR", "TSR")):
                checkpoint_rows.append({
                    "arm_id": arm, "outer_fold": fold, "training_seed": seed,
                    "epoch": epoch, "trial_id": f"checkpoint-{trial_index}",
                    "reading_task": task, "positive_rank": ranks[trial_index],
                    "candidate_pool_size": SPEC.pool_size,
                })
            nr, tsr = 1.0 / ranks[0], 1.0 / ranks[1]
            history.append({
                "arm_id": arm, "outer_fold": fold, "training_seed": seed,
                "epoch": epoch, "optimizer_steps": epoch * SPEC.batches_per_epoch,
                "mean_train_loss": "" if epoch == 0 else 0.25 / epoch,
                "checkpoint_macro_mrr": (nr + tsr) / 2.0,
                "checkpoint_nr_mrr": nr, "checkpoint_tsr_mrr": tsr,
                "is_best": str(epoch in {0, 1}).lower(),
            })
        write_csv(
            arm_root / "checkpoint_positive_ranks.csv",
            verifier.CHECKPOINT_RANK_FIELDS, checkpoint_rows,
        )
        write_csv(arm_root / "training_history.csv", verifier.HISTORY_FIELDS, history)

        summary = {
            "schema_version": 1, "status": "complete",
            "run_mode": "full_scientific", "shard_id": shard_id,
            "schedule_unit_index": 0, "outer_fold": fold, "training_seed": seed,
            "arm_id": arm, "binding_sha256": binding_sha,
            "schedule_unit_sha256": schedule_sha, "epochs": SPEC.epochs,
            "batches_per_epoch": SPEC.batches_per_epoch,
            "optimizer_steps": SPEC.optimizer_steps, "trainable_parameters": 196608,
            "initial_state_sha256": shared_initial,
            "final_state_sha256": steps[-1]["post_step_state_sha256"],
            "final_optimizer_state_sha256": steps[-1]["optimizer_state_sha256"],
            "best_epoch": 1, "best_checkpoint_macro_mrr": 1.0,
            "checkpoint_trial_count": SPEC.checkpoint_count(fold),
            "checkpoint_rank_rows": (SPEC.epochs + 1) * SPEC.checkpoint_count(fold),
            "confirmation_trial_count": SPEC.confirmation_count(fold),
            "confirmation_prediction_rows": 2 * SPEC.confirmation_count(fold),
            "confirmation_candidate_score_rows": (
                2 * SPEC.confirmation_count(fold) * SPEC.pool_size
            ),
            "resumed": False, "artifact_sha256": {},
            "full_training_authorized": True,
            "scientific_decision_permitted": False,
            "official_validation_used_for_confirmation": False,
            "held_out_test_accessed": False,
        }
        write_json(arm_root / "run_summary.json", summary)

    metadata = {
        "schema_version": 1, "status": "complete",
        "run_mode": "full_scientific_shard", **common,
        "arms": list(verifier.ARM_IDS), "epochs": SPEC.epochs,
        "batches_per_epoch": SPEC.batches_per_epoch, "batch_size": 64,
        "completed_arms": list(verifier.ARM_IDS),
        "optimizer_steps_completed": len(verifier.ARM_IDS) * SPEC.optimizer_steps,
        "full_training_authorized": True,
        "scientific_decision_permitted_after_complete_matrix_only": True,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
        "device": verifier.EXACT_RUNTIME_ENVIRONMENT["device"],
        "python": verifier.EXACT_RUNTIME_ENVIRONMENT["python"],
        "torch": verifier.EXACT_RUNTIME_ENVIRONMENT["torch"],
    }
    write_json(root / verifier.METADATA_NAME, metadata)
    manifest = {
        "schema_version": 1, "status": "complete",
        "run_mode": "full_scientific_shard", **common,
        "arms": list(verifier.ARM_IDS), "epochs": SPEC.epochs,
        "batches_per_epoch": SPEC.batches_per_epoch, "batch_size": 64,
        "optimizer_steps_per_arm": SPEC.optimizer_steps,
        "optimizer_steps_per_shard": len(verifier.ARM_IDS) * SPEC.optimizer_steps,
        "full_training_authorized": True,
        "scientific_decision_permitted_after_complete_matrix_only": True,
        "official_validation_used_for_confirmation": False,
        "held_out_test_accessed": False,
        "artifact_sha256": {},
    }
    write_json(root / verifier.MANIFEST_NAME, manifest)
    return reseal(root)


class FullShardVerifierTests(unittest.TestCase):
    def evidence(self, root: Path) -> verifier.EvidenceHashes:
        return verifier.EvidenceHashes(**json.loads(
            (root.parent / "evidence_hashes.json").read_text(encoding="utf-8")
        ))

    def run_verify(self, root: Path, manifest_sha: str) -> dict[str, object]:
        return verifier.verify(
            root, manifest_sha, CONTRACT_SHA, AUTH_SHA, spec=SPEC,
            protocol_root=root.parent / "protocol",
            schedule_root=root.parent / "schedule",
            evidence_hashes=self.evidence(root),
            preserved_source_id="fixture-v1",
        )

    def test_complete_fixture_passes_without_deserializing_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            manifest_sha = build_fixture(root)
            report = self.run_verify(root, manifest_sha)
            self.assertEqual(set(report), set(verifier.REPORT_FIELDS))
            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["checkpoint_deserialized"])
            self.assertEqual(report["total_optimizer_steps"], 12)
            self.assertEqual(report["best_epochs"], {arm: 1 for arm in verifier.ARM_IDS})
            self.assertTrue(report["append_only_strict_incumbent_history_verified"])
            self.assertFalse(report["partial_scientific_decision_permitted"])
            self.assertTrue(report["preserved_protocol_evidence_verified"])
            self.assertTrue(report["preserved_schedule_evidence_verified"])
            self.assertTrue(report["runtime_fingerprint_verified"])
            self.assertTrue(report["git_execution_boundary_verified"])
            self.assertEqual(
                report["runtime_fingerprint_sha256"],
                verifier.runtime_fingerprint_sha256(
                    verifier.EXACT_RUNTIME_ENVIRONMENT
                ),
            )
            self.assertEqual(report["recomputed_binding_sha256"],
                             json.loads((root / verifier.MANIFEST_NAME).read_text(
                                 encoding="utf-8"))["binding_sha256"])
            frozen_hashes = self.evidence(root)
            self.assertEqual(report["preserved_evidence_sha256"], {
                "candidate_pools.csv": frozen_hashes.candidate_pools_sha256,
                "confirmation_donors.csv": frozen_hashes.confirmation_donors_sha256,
                "schedule_indices.u32le": frozen_hashes.schedule_indices_sha256,
                "schedule_units.csv": frozen_hashes.schedule_units_sha256,
                "trial_catalog.csv": frozen_hashes.trial_catalog_sha256,
            })

    def test_checkpoint_byte_tamper_is_rejected_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            manifest_sha = build_fixture(root)
            with (root / "runs" / verifier.ARM_IDS[0] / "best_checkpoint.pt").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "manifest artifact hashes"):
                self.run_verify(root, manifest_sha)

    def test_extra_inventory_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            manifest_sha = build_fixture(root)
            (root / "undeclared.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level artifact inventory"):
                self.run_verify(root, manifest_sha)

    def test_symlink_is_rejected_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            manifest_sha = build_fixture(root)
            target = root / "runs" / verifier.ARM_IDS[0] / "best_checkpoint.pt"
            original = Path.is_symlink

            def pretend_one_file_is_a_symlink(path: Path) -> bool:
                return path == target or original(path)

            with mock.patch.object(Path, "is_symlink", pretend_one_file_is_a_symlink):
                with self.assertRaisesRegex(ValueError, "artifact contains symlink"):
                    self.run_verify(root, manifest_sha)

    def test_full_contract_or_launch_binding_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            manifest_sha = build_fixture(root)
            with self.assertRaisesRegex(ValueError, "manifest full contract"):
                verifier.verify(
                    root, manifest_sha, "d" * 64, AUTH_SHA, spec=SPEC,
                    protocol_root=root.parent / "protocol",
                    schedule_root=root.parent / "schedule",
                    evidence_hashes=self.evidence(root),
                )
            with self.assertRaisesRegex(ValueError, "manifest launch authorization"):
                verifier.verify(
                    root, manifest_sha, CONTRACT_SHA, "e" * 64, spec=SPEC,
                    protocol_root=root.parent / "protocol",
                    schedule_root=root.parent / "schedule",
                    evidence_hashes=self.evidence(root),
                )

    def test_official_validation_or_test_access_flags_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            manifest_path = root / verifier.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["held_out_test_accessed"] = True
            write_json(manifest_path, manifest)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "manifest held-out-test flag"):
                self.run_verify(root, manifest_sha)

    def test_partial_shard_decision_permission_is_rejected_everywhere(self) -> None:
        mutations = ("run_manifest", "summary", "prediction")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "artifact"
                build_fixture(root)
                if mutation == "run_manifest":
                    path = root / "run_manifest.csv"
                    fields, rows = read_csv(path)
                    rows[0]["scientific_decision_permitted"] = "true"
                    write_csv(path, tuple(fields), rows)
                    message = "partial shard improperly permits"
                elif mutation == "summary":
                    path = root / "runs" / verifier.ARM_IDS[0] / "run_summary.json"
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    summary["scientific_decision_permitted"] = True
                    write_json(path, summary)
                    message = "decision permission"
                else:
                    path = root / "confirmation_predictions.csv"
                    fields, rows = read_csv(path)
                    rows[0]["scientific_decision_permitted"] = "true"
                    write_csv(path, tuple(fields), rows)
                    message = "prediction improperly permits"
                manifest_sha = reseal(root)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_verify(root, manifest_sha)

    def test_paired_initialization_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            arm = verifier.ARM_IDS[1]
            path = root / "runs" / arm / "step_trace.csv"
            fields, rows = read_csv(path)
            changed = SHA("different-initial")
            rows[0]["initial_state_sha256"] = changed
            rows[0]["pre_step_state_sha256"] = changed
            for row in rows[1:]:
                row["initial_state_sha256"] = changed
            write_csv(path, tuple(fields), rows)
            summary_path = root / "runs" / arm / "run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["initial_state_sha256"] = changed
            write_json(summary_path, summary)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "paired initial model state"):
                self.run_verify(root, manifest_sha)

    def test_cross_arm_schedule_trace_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            arm = verifier.ARM_IDS[2]
            path = root / "runs" / arm / "step_trace.csv"
            fields, rows = read_csv(path)
            rows[0]["batch_catalog_indices_sha256"] = SHA("different-catalog-order")
            write_csv(path, tuple(fields), rows)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "preserved catalog-index batch"):
                self.run_verify(root, manifest_sha)

    def test_epoch_zero_earliest_tie_violation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            arm = verifier.ARM_IDS[0]
            path = root / "runs" / arm / "training_history.csv"
            fields, rows = read_csv(path)
            rows[2]["is_best"] = "true"
            write_csv(path, tuple(fields), rows)
            summary_path = root / "runs" / arm / "run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["best_epoch"] = 2
            write_json(summary_path, summary)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "append-only strict-improvement"):
                self.run_verify(root, manifest_sha)

    def test_matched_wrong_provenance_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            arm = verifier.ARM_IDS[0]
            path = root / "runs" / arm / "confirmation_candidate_scores.csv"
            fields, rows = read_csv(path)
            for row in rows:
                if row["signal_condition"] == "matched_wrong" and row["trial_id"] == "target-0":
                    row["signal_trial_id"] = "target-0"
                    row["signal_normalized_text_sha256"] = row["normalized_text_sha256"]
            write_csv(path, tuple(fields), rows)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "matched-wrong signal reuses"):
                self.run_verify(root, manifest_sha)

    def test_resealed_fabricated_frozen_donor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            fabricated_text = SHA("fabricated-donor-text")
            for arm in verifier.ARM_IDS:
                path = root / "runs" / arm / "confirmation_candidate_scores.csv"
                fields, rows = read_csv(path)
                for row in rows:
                    if row["trial_id"] != "target-0":
                        continue
                    rank = int(row["candidate_rank"])
                    row["is_designated_donor_text"] = str(int(rank == 2))
                    if rank == 2:
                        row["candidate_normalized_text_sha256"] = fabricated_text
                        row["candidate_text_target_id"] = "fabricated-donor-target"
                    if row["signal_condition"] == "matched_wrong":
                        row["signal_trial_id"] = "fabricated-donor-trial"
                        row["signal_normalized_text_sha256"] = fabricated_text
                write_csv(path, tuple(fields), rows)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "frozen matched_wrong signal provenance"):
                self.run_verify(root, manifest_sha)

    def test_resealed_fabricated_candidate_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            for arm in verifier.ARM_IDS:
                path = root / "runs" / arm / "confirmation_candidate_scores.csv"
                fields, rows = read_csv(path)
                for row in rows:
                    if row["trial_id"] != "target-1":
                        continue
                    rank = int(row["candidate_rank"])
                    if rank == 1:
                        row["candidate_normalized_text_sha256"] = SHA("third-target-1")
                        row["candidate_text_target_id"] = "target-1-text-target-2"
                        row["is_designated_donor_text"] = "0"
                    elif rank == 2:
                        row["candidate_normalized_text_sha256"] = SHA("donor-text-1")
                        row["candidate_text_target_id"] = "target-1-text-target-1"
                        row["is_designated_donor_text"] = "1"
                write_csv(path, tuple(fields), rows)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "frozen candidate_normalized_text_sha256"):
                self.run_verify(root, manifest_sha)

    def test_resealed_common_schedule_fabrication_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            fake_indices = SHA("fabricated-common-schedule-indices")
            fake_trials = SHA("fabricated-common-schedule-trials")
            for arm in verifier.ARM_IDS:
                path = root / "runs" / arm / "step_trace.csv"
                fields, rows = read_csv(path)
                rows[0]["batch_catalog_indices_sha256"] = fake_indices
                rows[0]["batch_trial_ids_sha256"] = fake_trials
                write_csv(path, tuple(fields), rows)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "preserved catalog-index batch"):
                self.run_verify(root, manifest_sha)

    def test_resealed_arbitrary_binding_sha256_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            fabricated = SHA("fabricated-but-internally-consistent-binding")
            manifest_path = root / verifier.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["binding_sha256"] = fabricated
            write_json(manifest_path, manifest)
            metadata_path = root / verifier.METADATA_NAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["binding_sha256"] = fabricated
            write_json(metadata_path, metadata)
            for arm in verifier.ARM_IDS:
                summary_path = root / "runs" / arm / "run_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["binding_sha256"] = fabricated
                write_json(summary_path, summary)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "recomputed shard binding"):
                self.run_verify(root, manifest_sha)

    def test_resealed_internally_consistent_runtime_fabrication_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            fabricated_runtime = dict(verifier.EXACT_RUNTIME_ENVIRONMENT)
            fabricated_runtime["selected_cuda_device_name"] = "Fabricated GPU"
            fabricated_runtime_sha = verifier.runtime_fingerprint_sha256(
                fabricated_runtime
            )
            manifest_path = root / verifier.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime_fingerprint"] = fabricated_runtime
            manifest["runtime_fingerprint_sha256"] = fabricated_runtime_sha
            manifest["binding_sha256"] = verifier.recompute_binding_sha256(
                manifest, fabricated_runtime, SPEC
            )
            write_json(manifest_path, manifest)
            metadata_path = root / verifier.METADATA_NAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["runtime_fingerprint"] = fabricated_runtime
            metadata["runtime_fingerprint_sha256"] = fabricated_runtime_sha
            metadata["binding_sha256"] = manifest["binding_sha256"]
            write_json(metadata_path, metadata)
            for arm in verifier.ARM_IDS:
                summary_path = root / "runs" / arm / "run_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["binding_sha256"] = manifest["binding_sha256"]
                write_json(summary_path, summary)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "exact runtime fingerprint"):
                self.run_verify(root, manifest_sha)

    def test_unused_second_cuda_device_is_nonbinding_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            manifest_path = root / verifier.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original_binding = manifest["binding_sha256"]
            manifest["observed_cuda_device_count"] = 2
            self.assertEqual(
                verifier.recompute_binding_sha256(
                    manifest, verifier.EXACT_RUNTIME_ENVIRONMENT, SPEC
                ),
                original_binding,
            )
            write_json(manifest_path, manifest)
            metadata_path = root / verifier.METADATA_NAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["observed_cuda_device_count"] = 2
            write_json(metadata_path, metadata)
            report = self.run_verify(root, reseal(root))
            self.assertTrue(report["runtime_fingerprint_verified"])

    def test_zero_observed_cuda_devices_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            manifest_path = root / verifier.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["observed_cuda_device_count"] = 0
            write_json(manifest_path, manifest)
            metadata_path = root / verifier.METADATA_NAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["observed_cuda_device_count"] = 0
            write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "below frozen minimum"):
                self.run_verify(root, reseal(root))

    def test_preserved_protocol_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            manifest_sha = build_fixture(root)
            with (root.parent / "protocol" / "candidate_pools.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "preserved input SHA256"):
                self.run_verify(root, manifest_sha)

    def test_resealed_nonfrozen_checkpoint_targets_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifact"
            build_fixture(root)
            for arm in verifier.ARM_IDS:
                path = root / "runs" / arm / "checkpoint_positive_ranks.csv"
                fields, rows = read_csv(path)
                for row in rows:
                    if row["trial_id"] == "checkpoint-0":
                        row["trial_id"] = "fabricated-checkpoint"
                write_csv(path, tuple(fields), rows)
            manifest_sha = reseal(root)
            with self.assertRaisesRegex(ValueError, "non-frozen checkpoint trial"):
                self.run_verify(root, manifest_sha)


if __name__ == "__main__":
    unittest.main()
