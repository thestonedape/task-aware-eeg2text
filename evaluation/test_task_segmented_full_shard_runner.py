from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from evaluation.run_task_segmented_full_shard import (
    ARMS,
    ADAPTER_PATH,
    AGGREGATOR_PATH,
    CHECKPOINT_RANK_FIELDS,
    CONFIRMATION_SCORE_FIELDS,
    DECISION_ENGINE_PATH,
    FULL_EXECUTION_CONTRACT_SHA256,
    FULL_MODE,
    EXACT_RUNTIME_ENVIRONMENT,
    HISTORY_FIELDS,
    NOTEBOOK_LAUNCH_PIN_PATHS,
    RUNNER_PATH,
    SHARD_VERIFIER_PATH,
    TASK_TREATMENT_PILOTS_PATH,
    STEP_TRACE_FIELDS,
    _atomic_csv,
    _atomic_torch_save,
    _binding_sha256,
    _configure_cuda_determinism_environment,
    _csv,
    _load_resume,
    _model_sha256,
    _optimizer_sha256,
    _restore_rng_state,
    _reported_resume_status,
    _require_exact_runtime_fingerprint,
    _runtime_fingerprint_sha256,
    _rng_state,
    _run_arm,
    _save_resume,
    _validate_correct_train_eeg_binding,
    authorize_launch,
    confirmation_predictions_from_scores,
    load_full_contract,
    positive_rank_from_scores,
    sha256,
    strict_best_epoch,
    verify_git_execution_boundary,
)


PROJECT_COMMIT = "1" * 40
LAUNCH_SHA256 = "a" * 64
TEST_RUNTIME_FINGERPRINT = dict(EXACT_RUNTIME_ENVIRONMENT)


def _observed_runtime(cuda_device_count: int = 1) -> dict[str, object]:
    return {
        "python": EXACT_RUNTIME_ENVIRONMENT["python"],
        "numpy": EXACT_RUNTIME_ENVIRONMENT["numpy"],
        "torch": EXACT_RUNTIME_ENVIRONMENT["torch"],
        "torch_cuda": EXACT_RUNTIME_ENVIRONMENT["torch_cuda"],
        "device": "cuda:0",
        "cuda_device_count": cuda_device_count,
        "selected_cuda_device_index": 0,
        "selected_cuda_device_name": "Tesla T4",
        "selected_cuda_compute_capability": [7, 5],
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms_enabled": True,
    }


def _launch(contract: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "authorized_for_full_p4b_launch",
        "full_execution_contract_sha256": FULL_EXECUTION_CONTRACT_SHA256,
        "project_commit": PROJECT_COMMIT,
        "runtime_environment": dict(EXACT_RUNTIME_ENVIRONMENT),
        "authorized_shard_ids": [
            row["shard_id"] for row in contract["sharding"]["shards"]
        ],
        "full_training_authorized": True,
        "checkpoint_evaluation_authorized": True,
        "confirmation_evaluation_authorized": True,
        "scientific_decision_permitted_after_complete_matrix_only": True,
        "partial_result_scientific_inspection_permitted": False,
        "official_validation_rows_read": False,
        "official_validation_used_for_confirmation": False,
        "held_out_test_rows_read": False,
        "held_out_test_accessed": False,
        "runner_source_sha256": sha256(RUNNER_PATH),
        "adapter_source_sha256": sha256(ADAPTER_PATH),
        "task_treatment_pilots_source_sha256": sha256(
            TASK_TREATMENT_PILOTS_PATH
        ),
        "shard_verifier_source_sha256": sha256(SHARD_VERIFIER_PATH),
        "aggregator_source_sha256": sha256(AGGREGATOR_PATH),
        "decision_engine_source_sha256": sha256(DECISION_ENGINE_PATH),
        **{
            key: sha256(path)
            for key, path in NOTEBOOK_LAUNCH_PIN_PATHS.items()
        },
    }


def _score_rows(arm: str = "global_mixed") -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for condition in ("correct", "matched_wrong"):
        signal_trial = "target" if condition == "correct" else "donor"
        signal_text = "target-text" if condition == "correct" else "donor-text"
        for candidate_rank in range(24):
            output.append({
                "arm_id": arm,
                "training_seed": 20260717,
                "outer_fold": 0,
                "trial_id": "target",
                "reading_task": "NR",
                "subject_id": "subject",
                "normalized_text_sha256": "target-text",
                "signal_condition": condition,
                "signal_trial_id": signal_trial,
                "signal_subject_id": "subject",
                "signal_normalized_text_sha256": signal_text,
                "candidate_rank": candidate_rank,
                "candidate_normalized_text_sha256": f"candidate-{candidate_rank}",
                "candidate_text_target_id": f"candidate-{candidate_rank}",
                "is_positive": int(candidate_rank == 1),
                "is_designated_donor_text": int(
                    condition == "matched_wrong" and candidate_rank == 2
                ),
                "score": "0.75" if candidate_rank in {0, 1} else "0.1",
            })
    return output


class AuthorizationAndRankingTests(unittest.TestCase):
    def test_launch_is_separate_hash_bound_and_deny_by_default(self) -> None:
        contract = load_full_contract()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "task_segmented_full_launch_authorization.json"
            with self.assertRaises(PermissionError):
                authorize_launch(
                    contract, missing, "0" * 64, PROJECT_COMMIT,
                    "p4b-f0-s20260717",
                )
            launch = _launch(contract)
            missing.write_text(
                json.dumps(launch, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            actual = hashlib.sha256(missing.read_bytes()).hexdigest()
            checked = authorize_launch(
                contract, missing, actual, PROJECT_COMMIT,
                "p4b-f0-s20260717",
            )
            self.assertTrue(checked["full_training_authorized"])
            drifted = dict(launch)
            drifted["held_out_test_accessed"] = True
            missing.write_text(
                json.dumps(drifted, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PermissionError, "held_out_test_accessed"):
                authorize_launch(
                    contract, missing, sha256(missing), PROJECT_COMMIT,
                    "p4b-f0-s20260717",
                )
            runtime_drift = _launch(contract)
            runtime_drift["runtime_environment"] = dict(
                runtime_drift["runtime_environment"]
            )
            runtime_drift["runtime_environment"]["numpy"] = "2.0.1"
            missing.write_text(
                json.dumps(runtime_drift, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PermissionError, "runtime_environment"):
                authorize_launch(
                    contract, missing, sha256(missing), PROJECT_COMMIT,
                    "p4b-f0-s20260717",
                )
            for field, value in (
                ("official_validation_rows_read", None),
                ("unexpected_field", False),
            ):
                with self.subTest(field_inventory=field):
                    inventory = _launch(contract)
                    if value is None:
                        inventory.pop(field)
                    else:
                        inventory[field] = value
                    missing.write_text(
                        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PermissionError, "field inventory"):
                        authorize_launch(
                            contract, missing, sha256(missing), PROJECT_COMMIT,
                            "p4b-f0-s20260717",
                        )
            for pin in (
                "task_treatment_pilots_source_sha256",
                "decision_engine_source_sha256",
                "execution_notebook_sha256",
            ):
                with self.subTest(source_drift=pin):
                    source_drift = _launch(contract)
                    source_drift[pin] = "0" * 64
                    missing.write_text(
                        json.dumps(source_drift, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(PermissionError, pin):
                        authorize_launch(
                            contract, missing, sha256(missing), PROJECT_COMMIT,
                            "p4b-f0-s20260717",
                        )

    def test_cublas_configuration_is_available_before_torch_setup(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            _configure_cuda_determinism_environment()
            import os

            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_exact_runtime_fingerprint_rejects_cpu_and_any_drift(self) -> None:
        observed = _observed_runtime()
        normalized_one = _require_exact_runtime_fingerprint(
            EXACT_RUNTIME_ENVIRONMENT, observed,
            requested_device="cuda:0", current_cuda_device=0,
        )
        self.assertEqual(normalized_one, EXACT_RUNTIME_ENVIRONMENT)
        with self.assertRaisesRegex(RuntimeError, "exactly cuda:0"):
            _require_exact_runtime_fingerprint(
                EXACT_RUNTIME_ENVIRONMENT, observed,
                requested_device="cpu", current_cuda_device=None,
            )
        drifted = dict(observed)
        drifted["numpy"] = "2.0.1"
        with self.assertRaisesRegex(RuntimeError, "numpy"):
            _require_exact_runtime_fingerprint(
                EXACT_RUNTIME_ENVIRONMENT, drifted,
                requested_device="cuda:0", current_cuda_device=0,
            )
        normalized_two = _require_exact_runtime_fingerprint(
            EXACT_RUNTIME_ENVIRONMENT, _observed_runtime(2),
            requested_device="cuda:0", current_cuda_device=0,
        )
        self.assertEqual(normalized_one, normalized_two)
        self.assertEqual(
            _runtime_fingerprint_sha256(normalized_one),
            _runtime_fingerprint_sha256(normalized_two),
        )
        with self.assertRaisesRegex(RuntimeError, "cuda_device_count"):
            _require_exact_runtime_fingerprint(
                EXACT_RUNTIME_ENVIRONMENT, _observed_runtime(0),
                requested_device="cuda:0", current_cuda_device=0,
            )
        wrong_gpu = _observed_runtime()
        wrong_gpu["selected_cuda_device_name"] = "NVIDIA A100-SXM4-40GB"
        with self.assertRaisesRegex(RuntimeError, "selected_cuda_device_name"):
            _require_exact_runtime_fingerprint(
                EXACT_RUNTIME_ENVIRONMENT, wrong_gpu,
                requested_device="cuda:0", current_cuda_device=0,
            )

    def test_runtime_and_clean_git_state_are_bound_into_shard_identity(self) -> None:
        contract = load_full_contract()
        shard = contract["sharding"]["shards"][0]
        first = _binding_sha256(
            contract, LAUNCH_SHA256, PROJECT_COMMIT, shard, "b" * 64,
            TEST_RUNTIME_FINGERPRINT, True, True,
        )
        self.assertEqual(first, _binding_sha256(
            contract, LAUNCH_SHA256, PROJECT_COMMIT, shard, "b" * 64,
            dict(TEST_RUNTIME_FINGERPRINT), True, True,
        ))
        drifted = dict(TEST_RUNTIME_FINGERPRINT)
        drifted["selected_cuda_device_name"] = "A100"
        self.assertNotEqual(first, _binding_sha256(
            contract, LAUNCH_SHA256, PROJECT_COMMIT, shard, "b" * 64,
            drifted, True, True,
        ))
        self.assertNotEqual(first, _binding_sha256(
            contract, LAUNCH_SHA256, PROJECT_COMMIT, shard, "b" * 64,
            TEST_RUNTIME_FINGERPRINT, False, True,
        ))
    def test_git_boundary_requires_head_clean_tree_and_clean_submodules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = "evaluation.run_task_segmented_full_shard._git_output"
            with mock.patch(
                target, side_effect=[str(root), PROJECT_COMMIT, "", ""]
            ) as git_output:
                result = verify_git_execution_boundary(PROJECT_COMMIT, root)
            self.assertTrue(result["git_worktree_clean"])
            self.assertTrue(result["git_submodules_clean"])
            git_output.assert_any_call(
                root, "status", "--porcelain=v1", "--untracked-files=all",
                "--ignored=matching",
            )

            with mock.patch(
                target, side_effect=[str(root), "2" * 40]
            ):
                with self.assertRaisesRegex(PermissionError, "Git HEAD"):
                    verify_git_execution_boundary(PROJECT_COMMIT, root)
            with mock.patch(
                target, side_effect=[str(root), PROJECT_COMMIT, "?? untracked.py"]
            ):
                with self.assertRaisesRegex(PermissionError, "worktree"):
                    verify_git_execution_boundary(PROJECT_COMMIT, root)
            with mock.patch(
                target,
                side_effect=[str(root), PROJECT_COMMIT, "!! evaluation/rogue.pyc"],
            ):
                with self.assertRaisesRegex(PermissionError, "worktree"):
                    verify_git_execution_boundary(PROJECT_COMMIT, root)
            with mock.patch(
                target, side_effect=[str(root), PROJECT_COMMIT, "", "+abc child"]
            ):
                with self.assertRaisesRegex(PermissionError, "submodule"):
                    verify_git_execution_boundary(PROJECT_COMMIT, root)

    def test_strict_greater_and_candidate_rank_tie_break(self) -> None:
        self.assertEqual(strict_best_epoch(0, 0.2, 1, 0.2), (0, 0.2, False))
        self.assertEqual(strict_best_epoch(0, 0.2, 1, 0.21), (1, 0.21, True))
        self.assertEqual(
            positive_rank_from_scores([0.75, 0.75, 0.1], [0, 1, 2], 1),
            2,
        )

    def test_resume_metadata_reports_only_a_preexisting_journal(self) -> None:
        self.assertFalse(_reported_resume_status(
            resume_preexisted=False, journal_loaded=True,
        ))
        self.assertTrue(_reported_resume_status(
            resume_preexisted=True, journal_loaded=True,
        ))
        with self.assertRaisesRegex(RuntimeError, "journal was not loaded"):
            _reported_resume_status(
                resume_preexisted=False, journal_loaded=False,
            )

    def test_correct_train_eeg_row_is_strictly_joined_to_catalog(self) -> None:
        catalog = {
            "dataset_version": "ZuCo2", "reading_task": "NR",
            "subject_id": "YAC", "eeg_vector_file": "eeg/chunk_000.npz",
            "eeg_vector_offset": "17", "eeg_vector_dim": "1024",
        }
        eeg = {
            "condition": "correct_train", "phase": "train",
            "target_trial_id": "trial-1", "signal_trial_id": "trial-1",
            "dataset_version": "ZuCo2", "reading_task": "NR",
            "subject_id": "YAC", "vector_file": "eeg/chunk_000.npz",
            "vector_offset": "17", "vector_dim": "1024",
            "prompt_mode": "all_masked",
        }
        _validate_correct_train_eeg_binding(catalog, eeg, "trial-1")
        mutations = {
            "dataset_version": "ZuCo1", "reading_task": "TSR",
            "subject_id": "YAG", "vector_file": "eeg/chunk_001.npz",
            "vector_offset": "18", "vector_dim": "512",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                drifted = dict(eeg)
                drifted[field] = value
                with self.assertRaisesRegex(ValueError, "EEG .*mismatch"):
                    _validate_correct_train_eeg_binding(
                        catalog, drifted, "trial-1"
                    )

    def test_confirmation_rank_reconstruction_and_signal_provenance(self) -> None:
        predictions = confirmation_predictions_from_scores(_score_rows())
        self.assertEqual(len(predictions), 2)
        self.assertEqual({row["positive_rank"] for row in predictions}, {2})
        self.assertEqual(
            {row["scientific_decision_permitted"] for row in predictions},
            {"false"},
        )
        bad = _score_rows()
        wrong = next(row for row in bad if row["signal_condition"] == "matched_wrong")
        wrong["signal_subject_id"] = "different-subject"
        with self.assertRaisesRegex(ValueError, "provenance"):
            confirmation_predictions_from_scores(bad)


class _TinyData:
    def __init__(self) -> None:
        self.shard = {
            "shard_id": "p4b-f0-s20260717", "schedule_unit_index": 0,
            "outer_fold": 0, "training_seed": 20260717,
        }
        self.unit = {"unit_sha256": "5" * 64}
        self.assignments = {
            "checkpoint-a": {"role": "checkpoint"},
            "checkpoint-b": {"role": "checkpoint"},
            "confirmation": {"role": "confirmation"},
        }
        self.schedule_indices = list(range(256))

    @property
    def batches_per_epoch(self) -> int:
        return 2


def _trace_row(step: int) -> dict[str, object]:
    return {
        "outer_fold": 0, "training_seed": 20260717,
        "arm_id": "global_mixed", "epoch": 1 if step < 2 else 2,
        "batch_index": step % 2, "global_step": step + 1,
        "schedule_unit_sha256": "5" * 64,
        "batch_catalog_indices_sha256": "6" * 64,
        "batch_trial_ids_sha256": "7" * 64,
        "global_mask_key": "8" * 64, "initial_state_sha256": "9" * 64,
        "pre_step_state_sha256": "a" * 64, "loss": "1",
        "gradient_norm_preclip": "1", "post_step_state_sha256": "b" * 64,
        "optimizer_state_sha256": "c" * 64,
    }


def _history_row(epoch: int) -> dict[str, object]:
    return {
        "arm_id": "global_mixed", "outer_fold": 0,
        "training_seed": 20260717, "epoch": epoch,
        "optimizer_steps": epoch * 2,
        "mean_train_loss": "" if epoch == 0 else "1",
        "checkpoint_macro_mrr": "0.2", "checkpoint_nr_mrr": "0.2",
        "checkpoint_tsr_mrr": "0.2", "is_best": int(epoch == 0),
    }


def _rank_rows(epoch: int) -> list[dict[str, object]]:
    return [{
        "arm_id": "global_mixed", "outer_fold": 0,
        "training_seed": 20260717, "epoch": epoch,
        "trial_id": trial, "reading_task": task,
        "positive_rank": 2, "candidate_pool_size": 24,
    } for trial, task in (("checkpoint-a", "NR"), ("checkpoint-b", "TSR"))]


@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None,
    "PyTorch is required for resume-state tests",
)
class ResumeAndReuseTests(unittest.TestCase):
    def test_epoch_checkpoint_discards_all_crash_window_csv_suffixes(self) -> None:
        import torch

        data = _TinyData()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "step_trace.csv"
            history_path = root / "training_history.csv"
            ranks_path = root / "checkpoint_positive_ranks.csv"
            checkpoint_path = root / "resume_checkpoint.pt"
            best_path = root / "best_checkpoint.pt"
            _atomic_csv(trace_path, STEP_TRACE_FIELDS, [_trace_row(0), _trace_row(1)])
            _atomic_csv(history_path, HISTORY_FIELDS, [_history_row(0), _history_row(1)])
            ranks = _rank_rows(0) + _rank_rows(1)
            _atomic_csv(ranks_path, CHECKPOINT_RANK_FIELDS, ranks)
            torch.manual_seed(11)
            model = torch.nn.Linear(2, 2, bias=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            # Materialize Adam state so its hash/restore path is exercised.
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            initial_hash = _model_sha256(model)
            best = {name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()}
            _save_resume(
                checkpoint_path, model=model, optimizer=optimizer,
                best_state=best, best_epoch=0, best_metric=0.2,
                completed_epoch=1, initial_state_sha256=initial_hash,
                binding_sha256="d" * 64, data=data, arm_id="global_mixed",
                trace_path=trace_path, history_path=history_path,
                ranks_path=ranks_path,
                launch_authorization_sha256=LAUNCH_SHA256,
                runtime_fingerprint=TEST_RUNTIME_FINGERPRINT,
            )
            drifted_runtime = dict(TEST_RUNTIME_FINGERPRINT)
            drifted_runtime["selected_cuda_device_name"] = "A100"
            drift_model = torch.nn.Linear(2, 2, bias=False)
            drift_optimizer = torch.optim.AdamW(
                drift_model.parameters(), lr=0.001
            )
            with self.assertRaisesRegex(ValueError, "runtime_fingerprint"):
                _load_resume(
                    checkpoint_path, model=drift_model,
                    optimizer=drift_optimizer,
                    initial_state_sha256=initial_hash,
                    binding_sha256="d" * 64, data=data,
                    arm_id="global_mixed", trace_path=trace_path,
                    history_path=history_path, ranks_path=ranks_path,
                    best_path=best_path, device=torch.device("cpu"),
                    launch_authorization_sha256=LAUNCH_SHA256,
                    runtime_fingerprint=drifted_runtime,
                )
            # Simulate a newly improved epoch-2 best file installed just before
            # the crash.  It is not committed because the epoch-2 resume journal
            # was never installed.
            future_best = {
                name: tensor.detach().cpu().clone() + 1
                for name, tensor in model.state_dict().items()
            }
            from evaluation.run_task_segmented_full_shard import _save_best_checkpoint
            _save_best_checkpoint(
                best_path, best_state=future_best, best_epoch=2,
                best_metric=0.9, binding_sha256="d" * 64,
                arm_id="global_mixed",
            )
            # Simulate interruption after all epoch-2 CSV replacements but
            # before installing the epoch-2 resume checkpoint.
            _atomic_csv(
                trace_path, STEP_TRACE_FIELDS,
                [_trace_row(0), _trace_row(1), _trace_row(2), _trace_row(3)],
            )
            _atomic_csv(
                history_path, HISTORY_FIELDS,
                [_history_row(0), _history_row(1), _history_row(2)],
            )
            _atomic_csv(ranks_path, CHECKPOINT_RANK_FIELDS, ranks + _rank_rows(2))
            clone = torch.nn.Linear(2, 2, bias=False)
            clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.001)
            original_torch_load = torch.load
            observed_map_locations: list[object] = []

            def checked_torch_load(*args: object, **kwargs: object) -> object:
                observed_map_locations.append(kwargs.get("map_location"))
                return original_torch_load(*args, **kwargs)

            with mock.patch("torch.load", side_effect=checked_torch_load):
                completed, _, _, _, resumed = _load_resume(
                    checkpoint_path, model=clone, optimizer=clone_optimizer,
                    initial_state_sha256=initial_hash, binding_sha256="d" * 64,
                    data=data, arm_id="global_mixed", trace_path=trace_path,
                    history_path=history_path, ranks_path=ranks_path,
                    best_path=best_path, device=torch.device("cpu"),
                    launch_authorization_sha256=LAUNCH_SHA256,
                    runtime_fingerprint=TEST_RUNTIME_FINGERPRINT,
                )
            self.assertTrue(resumed)
            self.assertEqual(observed_map_locations, ["cpu"])
            self.assertEqual(completed, 1)
            self.assertEqual(len(_csv(trace_path, STEP_TRACE_FIELDS)), 2)
            self.assertEqual(len(_csv(history_path, HISTORY_FIELDS)), 2)
            self.assertEqual(len(_csv(ranks_path, CHECKPOINT_RANK_FIELDS)), 4)
            self.assertEqual(_model_sha256(clone), _model_sha256(model))
            self.assertEqual(_optimizer_sha256(clone_optimizer), _optimizer_sha256(optimizer))
            restored_best = torch.load(
                best_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(restored_best["best_epoch"], 0)
            self.assertEqual(restored_best["best_checkpoint_macro_mrr"], 0.2)
            self.assertEqual(
                restored_best["model_state_sha256"], _model_sha256(model)
            )

    def test_cuda_rng_checkpoint_round_trip_remains_cpu_mapped(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        _configure_cuda_determinism_environment()
        torch.cuda.manual_seed_all(20260717)
        state = _rng_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            _atomic_torch_save(path, {"rng_state": state})
            restored = torch.load(path, map_location="cpu", weights_only=True)[
                "rng_state"
            ]
        self.assertEqual(restored["torch_cpu"].device.type, "cpu")
        self.assertEqual(restored["torch_cuda_device_0"].device.type, "cpu")
        _restore_rng_state(restored)

    def test_cuda_resume_restores_model_optimizer_and_rng_from_cpu_load(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        _configure_cuda_determinism_environment()
        device = torch.device("cuda")
        data = _TinyData()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "step_trace.csv"
            history_path = root / "training_history.csv"
            ranks_path = root / "checkpoint_positive_ranks.csv"
            checkpoint_path = root / "resume_checkpoint.pt"
            best_path = root / "best_checkpoint.pt"
            _atomic_csv(trace_path, STEP_TRACE_FIELDS, [_trace_row(0), _trace_row(1)])
            _atomic_csv(history_path, HISTORY_FIELDS, [_history_row(0), _history_row(1)])
            _atomic_csv(
                ranks_path, CHECKPOINT_RANK_FIELDS,
                _rank_rows(0) + _rank_rows(1),
            )
            torch.manual_seed(29)
            torch.cuda.manual_seed_all(29)
            model = torch.nn.Linear(2, 2, bias=False).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            model(torch.ones(1, 2, device=device)).sum().backward()
            optimizer.step()
            initial_hash = _model_sha256(model)
            best = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            _save_resume(
                checkpoint_path, model=model, optimizer=optimizer,
                best_state=best, best_epoch=0, best_metric=0.2,
                completed_epoch=1, initial_state_sha256=initial_hash,
                binding_sha256="e" * 64, data=data, arm_id="global_mixed",
                trace_path=trace_path, history_path=history_path,
                ranks_path=ranks_path,
                launch_authorization_sha256=LAUNCH_SHA256,
                runtime_fingerprint=TEST_RUNTIME_FINGERPRINT,
            )
            clone = torch.nn.Linear(2, 2, bias=False).to(device)
            clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=0.001)
            completed, _, _, _, resumed = _load_resume(
                checkpoint_path, model=clone, optimizer=clone_optimizer,
                initial_state_sha256=initial_hash, binding_sha256="e" * 64,
                data=data, arm_id="global_mixed", trace_path=trace_path,
                history_path=history_path, ranks_path=ranks_path,
                best_path=best_path, device=device,
                launch_authorization_sha256=LAUNCH_SHA256,
                runtime_fingerprint=TEST_RUNTIME_FINGERPRINT,
            )
            self.assertTrue(resumed)
            self.assertEqual(completed, 1)
            self.assertEqual(_model_sha256(clone), _model_sha256(model))
            self.assertEqual(
                _optimizer_sha256(clone_optimizer), _optimizer_sha256(optimizer)
            )

    def test_completed_arm_reuse_reconstructs_predictions_and_rejects_subset_hashes(self) -> None:
        data = _TinyData()
        contract = load_full_contract()
        with tempfile.TemporaryDirectory() as directory:
            arm_root = Path(directory)
            for name in (
                "resume_checkpoint.pt", "best_checkpoint.pt", "training_history.csv",
                "step_trace.csv", "checkpoint_positive_ranks.csv",
            ):
                (arm_root / name).write_bytes(name.encode("utf-8"))
            _atomic_csv(
                arm_root / "confirmation_candidate_scores.csv",
                CONFIRMATION_SCORE_FIELDS, _score_rows(),
            )
            artifacts = {
                name: sha256(arm_root / name)
                for name in (
                    "resume_checkpoint.pt", "best_checkpoint.pt",
                    "training_history.csv", "step_trace.csv",
                    "checkpoint_positive_ranks.csv",
                    "confirmation_candidate_scores.csv",
                )
            }
            summary = {
                "schema_version": 1, "status": "complete", "run_mode": FULL_MODE,
                "shard_id": data.shard["shard_id"], "schedule_unit_index": 0,
                "outer_fold": 0, "training_seed": 20260717,
                "arm_id": "global_mixed", "binding_sha256": "e" * 64,
                "schedule_unit_sha256": "5" * 64, "epochs": 40,
                "batches_per_epoch": 105, "optimizer_steps": 4200,
                "trainable_parameters": 196608,
                "initial_state_sha256": "1" * 64,
                "final_state_sha256": "2" * 64,
                "final_optimizer_state_sha256": "3" * 64,
                "best_epoch": 0, "best_checkpoint_macro_mrr": 0.2,
                "checkpoint_trial_count": 2, "checkpoint_rank_rows": 82,
                "confirmation_trial_count": 1, "confirmation_prediction_rows": 2,
                "confirmation_candidate_score_rows": 48, "resumed": False,
                "artifact_sha256": artifacts, "full_training_authorized": True,
                "scientific_decision_permitted": False,
                "official_validation_used_for_confirmation": False,
                "held_out_test_accessed": False,
            }
            (arm_root / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = _run_arm(
                arm_root, "global_mixed", data, contract, "e" * 64, "cpu",
                LAUNCH_SHA256, TEST_RUNTIME_FINGERPRINT,
            )
            self.assertEqual(len(result["confirmation_predictions"]), 2)
            tampered = copy.deepcopy(summary)
            tampered["artifact_sha256"] = {}
            (arm_root / "run_summary.json").write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "artifact declaration"):
                _run_arm(
                    arm_root, "global_mixed", data, contract, "e" * 64,
                    "cpu", LAUNCH_SHA256, TEST_RUNTIME_FINGERPRINT,
                )


if __name__ == "__main__":
    unittest.main()
