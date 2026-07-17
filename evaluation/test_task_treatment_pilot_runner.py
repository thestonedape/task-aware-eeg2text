import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evaluation.run_task_treatment_pilots import (
    CONTRACT_PATH,
    PROTOCOL_PATH,
    RUN_ARTIFACTS,
    PilotData,
    Trial,
    _binding,
    _valid_completed_run,
    execution_matrix,
    paired_bootstrap,
    read_csv,
    read_json,
    reconstruct_candidate_pools,
    run_configuration_seed,
    sha256,
    task_batches,
    validate_execution_contract,
)


class TaskTreatmentPilotRunnerTests(unittest.TestCase):
    def test_execution_protocol_is_bound_before_training(self):
        protocol = read_json(PROTOCOL_PATH)
        contract = read_json(CONTRACT_PATH)
        self.assertEqual(protocol["status"], "frozen_before_pilot_execution")
        self.assertEqual(protocol["parent_contract_sha256"], sha256(CONTRACT_PATH))
        self.assertEqual(protocol["training"]["auxiliary_factor_losses"], [])
        self.assertFalse(protocol["continuation"]["held_out_test_accessed"])
        validate_execution_contract(
            contract,
            protocol,
            protocol["input"]["preserved_source_id"],
            "1" * 40,
        )

    def test_smoke_matrix_exercises_all_configurations(self):
        contract = read_json(CONTRACT_PATH)
        configs, seeds = execution_matrix(contract, smoke=True)
        self.assertEqual(tuple(configs), (
            "generic_pooled", "separate_per_task", "task_token", "masked_shared_private"
        ))
        self.assertEqual(seeds, [20260717])

    def test_resume_binding_includes_code_and_source_identity(self):
        binding = _binding(
            "a" * 64, "b" * 64, "version-2", "c" * 40,
            "d" * 64, "generic_pooled", 7, True,
        )
        self.assertEqual(binding["preserved_source_id"], "version-2")
        self.assertEqual(binding["project_commit"], "c" * 40)
        self.assertEqual(binding["candidate_pool_sha256"], "d" * 64)
        self.assertEqual(len(binding["runner_source_sha256"]), 64)
        self.assertEqual(len(binding["adapter_source_sha256"]), 64)

    def test_completed_run_requires_exact_artifact_set_and_hashes(self):
        binding = _binding(
            "a" * 64, "b" * 64, "version-2", "c" * 40,
            "d" * 64, "generic_pooled", 7, True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_summary.json").write_text(json.dumps({
                "status": "pass",
                "binding": binding,
                "artifact_sha256": {},
            }), encoding="utf-8")
            self.assertFalse(_valid_completed_run(root, binding))
            hashes = {}
            for name in RUN_ARTIFACTS:
                path = root / name
                path.write_bytes(name.encode("utf-8"))
                hashes[name] = sha256(path)
            (root / "run_summary.json").write_text(json.dumps({
                "status": "pass",
                "binding": binding,
                "artifact_sha256": hashes,
            }), encoding="utf-8")
            self.assertTrue(_valid_completed_run(root, binding))
            (root / "metrics.csv").write_text("corrupted", encoding="utf-8")
            self.assertFalse(_valid_completed_run(root, binding))

    def test_candidate_pool_reconstruction_is_deterministic_and_model_free(self):
        trials = []
        records = {}
        for index in range(25):
            text = f"candidate sentence number {index}"
            identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
            records[identity] = {
                "text_target_id": identity,
                "representative_text": text,
            }
            trials.append(Trial(
                trial_id=f"trial-{index:02d}",
                split="val",
                cohort="primary_zuco2_nr_tsr",
                dataset_version="ZuCo2",
                reading_task="NR",
                subject_id=f"S{index % 3}",
                text_target_id=identity,
            ))
        first = reconstruct_candidate_pools(trials, records, 24, 20260716)
        second = reconstruct_candidate_pools(trials, records, 24, 20260716)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[3], second[3])
        self.assertEqual(len(first[0]), 25 * 24)
        self.assertTrue(all(len(pool) == 24 for pool in first[1].values()))
        self.assertTrue(all(0 <= offset < 24 for offset in first[2].values()))

    def test_candidate_pools_exclude_opposite_partition_targets(self):
        trials = []
        records = {}
        partitions = {}
        for scope, split in (
            ("training_catalog", "train"),
            ("checkpoint", "val"),
            ("decision", "val"),
        ):
            for index in range(25):
                text = f"{scope} candidate sentence number {index}"
                identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
                records[identity] = {
                    "text_target_id": identity,
                    "representative_text": text,
                }
                trial = Trial(
                    trial_id=f"{scope}-{index:02d}",
                    split=split,
                    cohort="primary_zuco2_nr_tsr",
                    dataset_version="ZuCo2",
                    reading_task="NR",
                    subject_id=f"S{index % 3}",
                    text_target_id=identity,
                )
                trials.append(trial)
                if split == "val":
                    partitions[trial.trial_id] = scope
        rows, pools, _, _ = reconstruct_candidate_pools(
            trials, records, 24, 20260716, partitions
        )
        self.assertEqual(len(pools), 50)
        for row in rows:
            allowed = {"training_catalog", row["target_evaluation_partition"]}
            self.assertIn(row["candidate_catalog_scope"], allowed)

    def test_task_schedule_is_complete_homogeneous_and_duplicate_text_safe(self):
        trials = []
        by_id = {}
        for task in ("SR", "NR", "TSR"):
            for index in range(11):
                trial = Trial(
                    trial_id=f"{task}-{index:02d}",
                    split="train",
                    cohort="auxiliary_sr" if task == "SR" else "primary_zuco2_nr_tsr",
                    dataset_version="ZuCo1" if task == "SR" else "ZuCo2",
                    reading_task=task,
                    subject_id=f"S{index % 4}",
                    text_target_id=f"{task}-text-{index // 2}",
                )
                trials.append(trial)
                by_id[trial.trial_id] = trial
        batches = task_batches(trials, batch_size=4, seed=20260717, epoch=1)
        flattened = [identity for batch in batches for identity in batch]
        self.assertEqual(sorted(flattened), sorted(by_id))
        for batch in batches:
            members = [by_id[identity] for identity in batch]
            self.assertEqual(len({member.reading_task for member in members}), 1)
            self.assertEqual(len({member.text_target_id for member in members}), len(members))
            self.assertGreaterEqual(len(batch), 2)

    def test_cluster_bootstrap_is_reproducible(self):
        effects = np.asarray([0.1, 0.2, -0.1, 0.3], dtype=np.float64)
        subjects = ["S1", "S1", "S2", "S2"]
        texts = ["T1", "T2", "T1", "T2"]
        tasks = ["NR", "NR", "TSR", "TSR"]
        first = paired_bootstrap(
            effects, subjects, texts, "two_way_subject_by_text", 50, 17, tasks
        )
        second = paired_bootstrap(
            effects, subjects, texts, "two_way_subject_by_text", 50, 17, tasks
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())

    def test_subject_bootstrap_shares_multiplicity_across_tasks(self):
        effects = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        subjects = ["S1", "S2", "S1", "S2"]
        texts = ["NR1", "NR2", "TSR1", "TSR2"]
        tasks = ["NR", "NR", "TSR", "TSR"]
        draws = paired_bootstrap(
            effects, subjects, texts, "subject_id", 100, 29, tasks
        )
        self.assertTrue(set(np.unique(draws)).issubset({-1.0, 0.0, 1.0}))

    def test_epoch_zero_identity_is_an_eligible_checkpoint(self):
        rng = np.random.default_rng(3)
        trials = {}
        text_vectors = {}
        eeg = {name: {} for name in (
            "correct_train", "correct_val", "zero_val",
            "gaussian_val", "matched_wrong_val",
        )}
        pools = {}
        positives = {}
        partitions = {}
        for task in ("SR", "NR", "TSR"):
            for index in range(2):
                trial_id = f"train-{task}-{index}"
                text_id = f"text-{task}-{index}"
                trials[trial_id] = Trial(
                    trial_id, "train",
                    "auxiliary_sr" if task == "SR" else "primary_zuco2_nr_tsr",
                    "ZuCo1" if task == "SR" else "ZuCo2",
                    task, f"S{index}", text_id,
                )
                text_vectors[text_id] = rng.normal(size=1024).astype(np.float32)
                eeg["correct_train"][trial_id] = rng.normal(size=1024).astype(np.float32)
        for task in ("NR", "TSR"):
            for index in range(2):
                trial_id = f"val-{task}-{index}"
                text_id = f"val-text-{task}-{index}"
                negative_id = f"negative-{task}-{index}"
                trials[trial_id] = Trial(
                    trial_id, "val", "primary_zuco2_nr_tsr",
                    "ZuCo2", task, f"V{index}", text_id,
                )
                text_vectors[text_id] = rng.normal(size=1024).astype(np.float32)
                text_vectors[negative_id] = rng.normal(size=1024).astype(np.float32)
                pools[trial_id] = [text_id, negative_id]
                positives[trial_id] = 0
                partitions[trial_id] = "checkpoint"
                for condition in ("correct_val", "zero_val", "gaussian_val", "matched_wrong_val"):
                    eeg[condition][trial_id] = rng.normal(size=1024).astype(np.float32)
        data = PilotData(
            artifact_root=Path("."),
            trials=trials,
            text_records={},
            text_vectors=text_vectors,
            eeg_vectors=eeg,
            partitions=partitions,
            pools=pools,
            positive_offsets=positives,
            pool_csv_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_configuration_seed(
                data=data,
                output_root=Path(temporary),
                config_id="generic_pooled",
                seed=20260717,
                contract=read_json(CONTRACT_PATH),
                protocol_sha="a" * 64,
                contract_sha=sha256(CONTRACT_PATH),
                preserved_source_id="version-2",
                project_commit="c" * 40,
                device=torch.device("cpu"),
                smoke=True,
            )
            history = read_csv(
                Path(temporary) / "runs" / "generic_pooled" / "20260717"
                / "training_history.csv"
            )
            self.assertEqual(history[0]["epoch"], "0")
            self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
