"""Authorization, decoding, determinism, and synthetic tests for the P4b runner."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.run_task_segmented_objective import (
    ARM_SUMMARY_FIELDS,
    ARMS,
    COMMON_BATCH_FIELDS,
    EXECUTION_CONTRACT_PATH,
    MODE,
    RUNNER_PATH,
    _atomic_csv,
    _atomic_json,
    _regular_file_hashes,
    _run_arm,
    authorize_mode,
    canonical_model_state_sha256,
    decode_u32le_slice,
    global_mask_key,
    initialization_seed,
    load_execution_contract,
    run,
    sha256,
    synthetic_batches,
    validate_execution_contract,
    validate_project_commit,
)


HAS_NUMPY_AND_TORCH = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("torch") is not None
)


class TaskSegmentedObjectiveRunnerStandardLibraryTests(unittest.TestCase):
    def test_frozen_contract_and_authorization_are_exact(self) -> None:
        contract = load_execution_contract()
        self.assertEqual(
            sha256(EXECUTION_CONTRACT_PATH),
            "9fd862b970ab95ded6f5efa0eb5290f8687bad7fd3a7f251f9ca66176de9f813",
        )
        authorize_mode(MODE, contract)
        for denied in ("full", "scientific_decision", "checkpoint_evaluation", ""):
            with self.subTest(denied=denied):
                with self.assertRaises(PermissionError):
                    authorize_mode(denied, contract)

        drifted = copy.deepcopy(contract)
        drifted["authorization"]["full_training_authorized"] = True
        with self.assertRaisesRegex(ValueError, "authorization drifted"):
            validate_execution_contract(drifted)

    def test_denied_mode_precedes_input_or_output_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "must-not-exist"
            with self.assertRaises(PermissionError):
                run(
                    root / "missing-vectors",
                    root / "missing-protocol",
                    root / "missing-schedule",
                    output,
                    "1" * 40,
                    "cpu",
                    "full",
                )
            self.assertFalse(output.exists())

    def test_project_commit_and_uint32_decoder(self) -> None:
        validate_project_commit("a" * 40)
        for invalid in ("a" * 39, "A" * 40, "z" * 40):
            with self.assertRaises(ValueError):
                validate_project_commit(invalid)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "indices.u32le"
            values = [0, 1, 2**32 - 1, 17, 99]
            path.write_bytes(b"".join(struct.pack("<I", value) for value in values))
            self.assertEqual(decode_u32le_slice(path, 4, 3), values[1:4])
            with self.assertRaisesRegex(ValueError, "bounds"):
                decode_u32le_slice(path, 16, 2)
            with self.assertRaises(ValueError):
                decode_u32le_slice(path, 1, 1)

    def test_seed_and_mask_key_match_schedule_freezer(self) -> None:
        from evaluation.freeze_task_segmented_training_schedule import (
            global_mask_key as freezer_global_mask_key,
            initialization_seed as freezer_initialization_seed,
        )

        contract_sha = "1" * 64
        unit_sha = "2" * 64
        self.assertEqual(
            initialization_seed(0, 20260717),
            freezer_initialization_seed(0, 20260717),
        )
        self.assertEqual(
            global_mask_key(contract_sha, unit_sha, 0, 20260717, 1, 0),
            freezer_global_mask_key(contract_sha, unit_sha, 0, 20260717, 1, 0),
        )


@unittest.skipUnless(HAS_NUMPY_AND_TORCH, "NumPy and PyTorch are required")
class TaskSegmentedObjectiveRunnerSyntheticTests(unittest.TestCase):
    def test_two_step_all_arm_smoke_is_paired_and_resume_safe(self) -> None:
        import torch

        contract = load_execution_contract()
        batches = synthetic_batches()
        seed = initialization_seed(0, 20260717)
        binding = "b" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for arm in ARMS:
                results.append(_run_arm(
                    root / "runs" / arm,
                    arm,
                    batches,
                    contract,
                    binding,
                    seed,
                    torch.device("cpu"),
                ))
            self.assertEqual(
                len({result["initial_state_sha256"] for result in results}), 1
            )
            self.assertTrue(all(result["optimizer_steps"] == 2 for result in results))
            self.assertTrue(all(result["full_training_authorized"] is False for result in results))
            self.assertTrue(all(result["scientific_decision_permitted"] is False for result in results))
            self.assertTrue(all(result["held_out_test_accessed"] is False for result in results))

            for result, arm in zip(results, ARMS):
                arm_root = root / "runs" / arm
                summary = json.loads(
                    (arm_root / "run_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    set(summary["artifact_sha256"]),
                    {"resume_checkpoint.pt", "step_trace.csv"},
                )
                reused = _run_arm(
                    arm_root, arm, batches, contract, binding, seed,
                    torch.device("cpu"),
                )
                self.assertTrue(reused["resumed"])
                self.assertEqual(
                    reused["final_state_sha256"], result["final_state_sha256"]
                )

            # Seal the lower-level real PyTorch runs in the exact production
            # layout and require the standard-library verifier to accept it.
            common_rows = []
            for batch in (0, 1):
                for position in range(64):
                    catalog = batch * 64 + position
                    task = "NR" if position < 32 else "TSR"
                    pseudo = 0 if position % 32 < 16 else 1
                    text_id = hashlib.sha256(
                        f"synthetic-text-{catalog}".encode("utf-8")
                    ).hexdigest()
                    common_rows.append({
                        "outer_fold": 0, "training_seed": 20260717,
                        "epoch": 1, "batch_index": batch,
                        "batch_position": position, "catalog_index": catalog,
                        "trial_id": f"synthetic-trial-{batch}-{position}",
                        "subject_id": "subject-0",
                        "reading_task": task, "pseudo_task_id": pseudo,
                        "normalized_text_sha256": text_id, "text_target_id": text_id,
                        "eeg_vector_file": f"vectors/eeg_{batch}.npz",
                        "eeg_vector_offset": position,
                        "text_vector_file": f"vectors/text_{batch}.npz",
                        "text_vector_offset": position,
                    })
            _atomic_csv(root / "common_batch_trace.csv", COMMON_BATCH_FIELDS, common_rows)
            _atomic_csv(root / "arm_summary.csv", ARM_SUMMARY_FIELDS, [
                {
                    "arm_id": result["arm_id"], "status": "pass",
                    "optimizer_steps": 2,
                    "initial_state_sha256": result["initial_state_sha256"],
                    "final_state_sha256": result["final_state_sha256"],
                    "final_optimizer_state_sha256": result[
                        "final_optimizer_state_sha256"
                    ],
                }
                for result in results
            ])
            binding_sections = {
                "execution_contract_sha256": sha256(EXECUTION_CONTRACT_PATH),
                "input_bindings": contract["immutable_inputs"],
                "bounded_smoke": contract["bounded_smoke"],
                "authorization": contract["authorization"],
            }
            metadata = {
                "schema_version": 1, "status": "pass", "run_mode": MODE,
                "project_commit": "1" * 40,
                "runner_source_sha256": sha256(RUNNER_PATH), **binding_sections,
                "completed_arms": list(ARMS), "optimizer_steps_completed": 6,
                "full_training_authorized": False,
                "scientific_decision_permitted": False,
                "held_out_test_accessed": False,
            }
            _atomic_json(root / "smoke_run_metadata.json", metadata)
            manifest = {
                "schema_version": 1, "status": "pass", "run_mode": MODE,
                "project_commit": "1" * 40,
                "runner_source_sha256": sha256(RUNNER_PATH), **binding_sections,
                "artifact_sha256": _regular_file_hashes(
                    root, {"task_segmented_smoke_manifest.json"}
                ),
            }
            _atomic_json(root / "task_segmented_smoke_manifest.json", manifest)
            from evaluation.verify_task_segmented_smoke_artifact import verify

            verified = verify(
                root, sha256(root / "task_segmented_smoke_manifest.json")
            )
            self.assertEqual(verified["status"], "pass")
            self.assertEqual(verified["total_optimizer_steps"], 6)

    def test_public_run_writes_full_artifact_accepted_by_independent_verifier(self) -> None:
        contract = load_execution_contract()
        batches = synthetic_batches()
        common_rows = []
        for batch in (0, 1):
            for position in range(64):
                catalog = batch * 64 + position
                text_id = hashlib.sha256(
                    f"synthetic-run-text-{catalog}".encode("utf-8")
                ).hexdigest()
                common_rows.append({
                    "outer_fold": 0, "training_seed": 20260717, "epoch": 1,
                    "batch_index": batch, "batch_position": position,
                    "catalog_index": catalog,
                    "trial_id": f"synthetic-trial-{batch}-{position}",
                    "subject_id": f"subject-{position % 8}",
                    "reading_task": "NR" if position < 32 else "TSR",
                    "pseudo_task_id": 0 if position % 32 < 16 else 1,
                    "normalized_text_sha256": text_id, "text_target_id": text_id,
                    "eeg_vector_file": f"vectors/eeg_{batch}.npz",
                    "eeg_vector_offset": position,
                    "text_vector_file": f"vectors/text_{batch}.npz",
                    "text_vector_offset": position,
                })
        inputs = contract["immutable_inputs"]
        reports = {
            "prompt_neutral_vectors": {
                "preserved_source_id": inputs["prompt_neutral_vectors"][
                    "preserved_source_id"
                ],
                "combined_manifest_sha256": inputs["prompt_neutral_vectors"][
                    "combined_manifest_sha256"
                ],
                "eeg": {"vector_index_sha256": inputs["prompt_neutral_vectors"][
                    "eeg_vector_index_sha256"
                ]},
                "text": {
                    "text_vector_index_sha256": inputs["prompt_neutral_vectors"][
                        "text_vector_index_sha256"
                    ],
                    "trial_text_targets_sha256": inputs["prompt_neutral_vectors"][
                        "trial_text_targets_sha256"
                    ],
                },
                "checks": {"held_out_test_accessed": False},
            },
            "task_segmented_protocol": {
                "preserved_source_id": inputs["task_segmented_protocol"][
                    "preserved_source_id"
                ],
                "contract_sha256": inputs["task_segmented_protocol"][
                    "contract_sha256"
                ],
                "protocol_report_sha256": inputs["task_segmented_protocol"][
                    "report_sha256"
                ],
                "held_out_test_accessed": False,
            },
            "training_schedule": {
                "preserved_source_id": inputs["training_schedule"][
                    "preserved_source_id"
                ],
                "schedule_contract_sha256": inputs["training_schedule"][
                    "contract_sha256"
                ],
                "schedule_manifest_sha256": inputs["training_schedule"][
                    "manifest_sha256"
                ],
                "schedule_report_sha256": inputs["training_schedule"][
                    "report_sha256"
                ],
                "verified_artifact_sha256": {
                    "schedule_indices.u32le": inputs["training_schedule"][
                        "indices_sha256"
                    ],
                    "trial_catalog.csv": inputs["training_schedule"][
                        "trial_catalog_sha256"
                    ],
                },
                "shape": inputs["training_schedule"]["shape"],
                "held_out_test_accessed": False,
            },
        }
        unit = {"unit_sha256": "3" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifact"
            with (
                mock.patch(
                    "evaluation.run_task_segmented_objective._verify_clean_remounts",
                    return_value=reports,
                ),
                mock.patch(
                    "evaluation.run_task_segmented_objective._strict_join",
                    return_value=([], unit),
                ),
                mock.patch(
                    "evaluation.run_task_segmented_objective._materialize_batches",
                    return_value=(batches, common_rows),
                ),
            ):
                result = run(
                    root / "vectors", root / "protocol", root / "schedule",
                    output, "1" * 40, "cpu",
                )
                self.assertFalse(result["reused_complete_output"])
                reused = run(
                    root / "vectors", root / "protocol", root / "schedule",
                    output, "1" * 40, "cpu",
                )
                self.assertTrue(reused["reused_complete_output"])

            from evaluation.verify_task_segmented_smoke_artifact import verify

            verified = verify(
                output, sha256(output / "task_segmented_smoke_manifest.json")
            )
            self.assertEqual(verified["status"], "pass")
            self.assertEqual(verified["total_optimizer_steps"], 6)

    def test_canonical_model_hash_detects_parameter_change(self) -> None:
        import torch
        from project_adapters.task_segmented_objective import SharedResidualAdapter

        torch.manual_seed(7)
        model = SharedResidualAdapter()
        before = canonical_model_state_sha256(model)
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] += 1
        self.assertNotEqual(before, canonical_model_state_sha256(model))


if __name__ == "__main__":
    unittest.main()
