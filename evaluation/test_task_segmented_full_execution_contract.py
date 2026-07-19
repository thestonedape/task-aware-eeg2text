import hashlib
import json
import unittest
from pathlib import Path


CONTRACT_PATH = Path(__file__).with_name("task_segmented_full_execution_contract.json")
ARMS = ("global_mixed", "true_task_segmented", "pseudo_task_segmented")
FOLDS = tuple(range(5))
SEEDS = (20260717, 20260718, 20260719)


def load_contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class TaskSegmentedFullExecutionContractTests(unittest.TestCase):
    def test_exact_immutable_input_versions_and_hashes(self):
        contract = load_contract()
        inputs = contract["immutable_inputs"]

        vectors = inputs["prompt_neutral_vectors"]
        self.assertEqual(vectors["dataset_slug"], "thestonedape/task-aware-eegtotext")
        self.assertEqual(vectors["dataset_version"], 2)
        self.assertEqual(
            vectors["combined_manifest_sha256"],
            "6c1fff8d2e89e33a72d03c39651e8ecce678c3b93cdb66747dd6dcc00538cddb",
        )

        protocol = inputs["task_segmented_protocol"]
        self.assertEqual(protocol["dataset_version"], 1)
        self.assertEqual(
            protocol["contract_sha256"],
            "396670afc0244cb601364ff89df53944c4f63402191a9d120e6e2648e5baed3b",
        )
        self.assertEqual(len(protocol["artifact_sha256"]), 7)

        schedule = inputs["training_schedule"]
        self.assertEqual(schedule["dataset_version"], 1)
        self.assertEqual(schedule["shape"], [15, 40, 105, 64])
        self.assertEqual(
            schedule["manifest_sha256"],
            "0cf2a752b5f0a67e7282bc0b4551b4792ceb3d346dd441fef6c359515885270a",
        )
        self.assertEqual(len(schedule["artifact_sha256"]), 9)

        smoke = inputs["preserved_smoke"]
        self.assertEqual(smoke["dataset_version"], 1)
        self.assertEqual(
            smoke["manifest_sha256"],
            "2cf38c78bdc25815fbb16ad17832ac4062fe48dbc814bcf82c6999b2950ec2a3",
        )
        self.assertEqual(
            smoke["clean_remount_verification_report_sha256"],
            "a2537e4a10de659705c956ec74105239e2046b5d42ca01a3524e1480b1081187",
        )
        self.assertEqual(len(smoke["artifact_sha256"]), 12)

    def test_exact_fold_major_shard_matrix_and_roles(self):
        contract = load_contract()
        sharding = contract["sharding"]
        rows = sharding["shards"]
        expected_pairs = [(fold, seed) for fold in FOLDS for seed in SEEDS]
        expected_ids = [f"p4b-f{fold}-s{seed}" for fold, seed in expected_pairs]

        self.assertEqual(sharding["shard_count"], 15)
        self.assertEqual(sharding["fits_per_shard"], 3)
        self.assertEqual(sharding["total_fits"], 45)
        self.assertEqual(tuple(sharding["arms_per_shard"]), ARMS)
        self.assertEqual([row["shard_id"] for row in rows], expected_ids)
        self.assertEqual([row["schedule_unit_index"] for row in rows], list(range(15)))
        self.assertEqual(
            [(row["outer_fold"], row["training_seed"]) for row in rows],
            expected_pairs,
        )

        for row in rows:
            fold = row["outer_fold"]
            self.assertEqual(row["confirmation_fold"], fold)
            self.assertEqual(row["checkpoint_fold"], (fold + 1) % 5)
            self.assertEqual(
                row["fit_folds"],
                [candidate for candidate in FOLDS if candidate not in {fold, (fold + 1) % 5}],
            )

    def test_exact_training_and_checkpoint_selection_semantics(self):
        contract = load_contract()
        execution = contract["execution"]
        self.assertEqual(execution["vector_dtype"], "float32")
        self.assertFalse(execution["automatic_mixed_precision"])
        self.assertEqual(execution["optimizer"], "AdamW")
        self.assertEqual(execution["learning_rate"], 0.001)
        self.assertEqual(execution["weight_decay"], 0.0001)
        self.assertEqual(execution["betas"], [0.9, 0.999])
        self.assertEqual(execution["eps"], 1e-08)
        self.assertEqual(execution["gradient_clip_norm"], 1.0)
        self.assertEqual((execution["epochs"], execution["batches_per_epoch"]), (40, 105))
        self.assertEqual(execution["optimizer_steps_per_arm"], 40 * 105)
        self.assertEqual(execution["optimizer_steps_per_shard"], 3 * 40 * 105)
        self.assertEqual(execution["total_optimizer_steps"], 15 * 3 * 40 * 105)
        self.assertFalse(execution["early_stopping"])

        evaluation = contract["checkpoint_and_confirmation"]
        self.assertEqual(evaluation["checkpoint_evaluation_epochs"], list(range(41)))
        self.assertIn("strictly greater", evaluation["replacement_rule"])
        self.assertIn("earliest epoch", evaluation["tie_rule"])
        self.assertTrue(evaluation["epoch_zero_identity_eligible"])
        self.assertTrue(evaluation["training_always_completes_epoch_40_before_selection_is_final"])
        self.assertIn("only after all 40", evaluation["confirmation_timing"])
        self.assertFalse(evaluation["confirmation_may_select_checkpoint_or_change_training"])
        self.assertEqual(evaluation["confirmation_signal_conditions"], ["correct", "matched_wrong"])
        self.assertEqual(evaluation["candidate_pool_size"], 24)

    def test_exact_selected_device_t4_runtime_policy(self):
        runtime = load_contract()["runtime_environment"]
        self.assertEqual(runtime, {
            "python": "3.12.13",
            "numpy": "2.0.2",
            "torch": "2.10.0+cu128",
            "torch_cuda": "12.8",
            "device": "cuda:0",
            "minimum_cuda_device_count": 1,
            "selected_cuda_device_index": 0,
            "selected_cuda_device_name": "Tesla T4",
            "selected_cuda_compute_capability": [7, 5],
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms_required": True,
            "full_scientific_cpu_execution_permitted": False,
            "runtime_fingerprint_bound_to_shard_and_resume": True,
        })

    def test_non_adaptive_completion_and_split_seals(self):
        contract = load_contract()
        completion = contract["non_adaptive_completion"]
        self.assertTrue(completion["all_15_shards_required"])
        self.assertTrue(completion["all_45_fits_required"])
        self.assertFalse(completion["partial_result_scientific_inspection_permitted"])
        self.assertFalse(
            completion["result_conditioned_shard_launch_cancel_retry_or_reordering_permitted"]
        )
        self.assertFalse(completion["aggregation_before_all_shards_independently_verified"])
        self.assertFalse(completion["scientific_decision_before_complete_matrix"])
        self.assertEqual(completion["complete_matrix_run_cells"], 45)
        self.assertEqual(completion["expected_confirmation_prediction_rows_both_conditions"], 162198)

        access = contract["data_access"]
        self.assertEqual(access["training_and_evaluation_source_split"], "canonical train only")
        self.assertFalse(access["official_validation_rows_read"])
        self.assertFalse(access["official_validation_used_for_selection_confirmation_or_rescue"])
        self.assertFalse(access["held_out_test_rows_read"])
        self.assertFalse(access["held_out_test_accessed"])

    def test_launch_authorization_is_separate_absent_and_deny_by_default(self):
        contract = load_contract()
        authorization = contract["authorization"]
        required = authorization["required_launch_authorization"]
        self.assertTrue(authorization["contract_preparation_authorized"])
        self.assertFalse(authorization["full_training_authorized"])
        self.assertFalse(authorization["checkpoint_evaluation_authorized"])
        self.assertFalse(authorization["confirmation_evaluation_authorized"])
        self.assertFalse(authorization["scientific_decision_permitted"])
        self.assertFalse(authorization["held_out_test_accessed"])
        self.assertEqual(required["artifact_name"], "task_segmented_full_launch_authorization.json")
        self.assertIsNone(required["sha256"])
        self.assertEqual(required["status"], "not_supplied")
        self.assertTrue(required["must_bind_full_execution_contract_sha256"])
        self.assertTrue(required["must_bind_runner_verifier_aggregator_and_notebook_hashes"])
        self.assertTrue(
            required["must_bind_transitive_task_treatment_adapter_source_sha256"]
        )
        self.assertTrue(
            required["must_bind_scientific_decision_engine_source_sha256"]
        )
        self.assertTrue(required["must_require_exact_clean_project_commit_at_execution"])
        self.assertTrue(required["must_bind_exact_runtime_environment"])

    def test_resume_and_output_schema_are_frozen(self):
        contract = load_contract()
        resume = contract["resume"]
        self.assertEqual(
            resume["checkpoint_boundary"],
            "after every completed epoch and at arm completion",
        )
        self.assertIn("deterministically replay", resume["partial_epoch_policy"])
        self.assertEqual(
            resume["retained_training_checkpoints_per_arm"],
            ["latest_resume", "selected_best"],
        )
        self.assertFalse(resume["intermediate_epoch_checkpoint_retention"])
        self.assertIn("launch_authorization_sha256", resume["resume_must_bind"])
        self.assertIn("exact frozen runtime fingerprint", resume["resume_must_bind"])
        self.assertIn(
            "CPU and selected cuda:0 RNG states", resume["resume_must_bind"]
        )

        output = contract["output"]
        self.assertEqual(
            output["exact_top_level_files"],
            [
                "full_shard_manifest.json",
                "shard_run_metadata.json",
                "run_manifest.csv",
                "confirmation_predictions.csv",
            ],
        )
        self.assertEqual(output["exact_top_level_directories"], ["runs"])
        self.assertEqual(
            output["exact_per_arm_files"],
            [
                "resume_checkpoint.pt",
                "best_checkpoint.pt",
                "training_history.csv",
                "step_trace.csv",
                "checkpoint_positive_ranks.csv",
                "confirmation_candidate_scores.csv",
                "run_summary.json",
            ],
        )
        self.assertEqual(output["training_history_rows_per_arm"], 41)
        self.assertEqual(output["checkpoint_evaluations_per_arm"], 41)
        self.assertEqual(output["retained_checkpoints_per_arm"], 2)
        self.assertEqual(
            output["confirmation_candidate_scores_identity_fields"],
            [
                "signal_trial_id",
                "signal_subject_id",
                "signal_normalized_text_sha256",
            ],
        )

    def test_contract_raw_sha256_is_frozen(self):
        self.assertEqual(
            hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "99c1235d21ce0dd9eb80b1c1c0c3930b3b7347007ebc35be3385f0bc253a837c",
        )


if __name__ == "__main__":
    unittest.main()
