"""Focused tests for the deny-by-default P4b launch freezer."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation import freeze_task_segmented_full_launch_authorization as freezer
from evaluation.run_task_segmented_full_shard import (
    authorize_launch,
    load_full_contract,
)


PROJECT_COMMIT = "1" * 40


class FullLaunchAuthorizationFreezerTests(unittest.TestCase):
    def _temporary_sources(
        self, root: Path
    ) -> tuple[dict[str, Path], dict[str, str]]:
        paths: dict[str, Path] = {}
        for index, key in enumerate(freezer.PIN_PATH_KEYS):
            path = root / f"reviewed-{index}.bin"
            path.write_bytes(f"{key}\n".encode("utf-8"))
            paths[key] = path
        return paths, {key: freezer.sha256(path) for key, path in paths.items()}

    def test_emits_runner_compatible_exact_matrix_and_hashes(self) -> None:
        contract = load_full_contract()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = self._temporary_sources(root)
            # The production runner locally validates every Python source pin;
            # only reviewed notebook pins remain opaque SHA-256 bindings.
            local_sources = {
                "runner_source_sha256": freezer.RUNNER_PATH,
                "adapter_source_sha256": freezer.ADAPTER_PATH,
                "task_treatment_pilots_source_sha256": (
                    freezer.TASK_TREATMENT_PILOTS_PATH
                ),
                "shard_verifier_source_sha256": freezer.SHARD_VERIFIER_PATH,
                "aggregator_source_sha256": freezer.AGGREGATOR_PATH,
                "decision_engine_source_sha256": freezer.DECISION_ENGINE_PATH,
                "execution_notebook_sha256": freezer.EXECUTION_NOTEBOOK_PATH,
                "shard_clean_remount_verification_notebook_sha256": (
                    freezer.SHARD_VERIFICATION_NOTEBOOK_PATH
                ),
                "complete_matrix_aggregation_notebook_sha256": (
                    freezer.COMPLETE_MATRIX_AGGREGATION_NOTEBOOK_PATH
                ),
            }
            for key, path in local_sources.items():
                paths[key] = path
                expected[key] = freezer.sha256(path)
            output = root / freezer.OUTPUT_NAME

            launch = freezer.freeze_launch_authorization(
                contract_path=freezer.FULL_CONTRACT_PATH,
                project_commit=PROJECT_COMMIT,
                source_paths=paths,
                expected_sha256=expected,
                output_path=output,
            )

            self.assertEqual(launch["authorized_shard_ids"], list(freezer.EXPECTED_SHARD_IDS))
            self.assertEqual(
                launch["runtime_environment"], freezer.EXACT_RUNTIME_ENVIRONMENT
            )
            self.assertEqual(len(launch["authorized_shard_ids"]), 15)
            self.assertTrue(launch["full_training_authorized"])
            self.assertTrue(launch["checkpoint_evaluation_authorized"])
            self.assertTrue(launch["confirmation_evaluation_authorized"])
            self.assertTrue(
                launch["scientific_decision_permitted_after_complete_matrix_only"]
            )
            self.assertFalse(launch["partial_result_scientific_inspection_permitted"])
            self.assertFalse(launch["official_validation_rows_read"])
            self.assertFalse(launch["official_validation_used_for_confirmation"])
            self.assertFalse(launch["held_out_test_rows_read"])
            self.assertFalse(launch["held_out_test_accessed"])
            self.assertEqual(len(freezer.PIN_PATH_KEYS), 9)
            for key in freezer.PIN_PATH_KEYS:
                self.assertEqual(launch[key], expected[key])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), launch)
            # This is the exact runtime guard used before any scientific input I/O.
            authorized = authorize_launch(
                contract,
                output,
                freezer.sha256(output),
                PROJECT_COMMIT,
                freezer.EXPECTED_SHARD_IDS[0],
            )
            self.assertEqual(authorized, launch)

    def test_refuses_reviewed_source_drift_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = self._temporary_sources(root)
            paths["aggregator_source_sha256"].write_bytes(b"changed after review\n")
            output = root / freezer.OUTPUT_NAME
            with self.assertRaisesRegex(ValueError, "reviewed source drifted"):
                freezer.freeze_launch_authorization(
                    contract_path=freezer.FULL_CONTRACT_PATH,
                    project_commit=PROJECT_COMMIT,
                    source_paths=paths,
                    expected_sha256=expected,
                    output_path=output,
                )
            self.assertFalse(output.exists())

    def test_refuses_missing_notebook_without_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = self._temporary_sources(root)
            notebook_key = "shard_clean_remount_verification_notebook_sha256"
            paths[notebook_key].unlink()
            output = root / freezer.OUTPUT_NAME
            with self.assertRaisesRegex(FileNotFoundError, "required reviewed source is missing"):
                freezer.freeze_launch_authorization(
                    contract_path=freezer.FULL_CONTRACT_PATH,
                    project_commit=PROJECT_COMMIT,
                    source_paths=paths,
                    expected_sha256=expected,
                    output_path=output,
                )
            self.assertFalse(output.exists())

    def test_base_contract_deny_flags_are_mandatory(self) -> None:
        contract = json.loads(
            freezer.FULL_CONTRACT_PATH.read_text(encoding="utf-8")
        )
        for key in (
            "full_training_authorized",
            "checkpoint_evaluation_authorized",
            "confirmation_evaluation_authorized",
            "scientific_decision_permitted",
            "held_out_test_accessed",
        ):
            with self.subTest(authorization_key=key):
                drifted = copy.deepcopy(contract)
                drifted["authorization"][key] = True
                with self.assertRaisesRegex(ValueError, "base full contract must deny"):
                    freezer.validate_contract_payload(drifted)

        for key in (
            "official_validation_rows_read",
            "official_validation_used_for_selection_confirmation_or_rescue",
            "held_out_test_rows_read",
            "held_out_test_accessed",
        ):
            with self.subTest(data_access_key=key):
                drifted = copy.deepcopy(contract)
                drifted["data_access"][key] = True
                with self.assertRaisesRegex(ValueError, "must deny data access"):
                    freezer.validate_contract_payload(drifted)

        drifted = copy.deepcopy(contract)
        drifted["non_adaptive_completion"][
            "partial_result_scientific_inspection_permitted"
        ] = True
        with self.assertRaisesRegex(ValueError, "completion flag drifted"):
            freezer.validate_contract_payload(drifted)

        drifted = copy.deepcopy(contract)
        drifted["runtime_environment"]["selected_cuda_device_name"] = "A100"
        with self.assertRaisesRegex(ValueError, "runtime environment drifted"):
            freezer.validate_contract_payload(drifted)

        drifted = copy.deepcopy(contract)
        drifted["authorization"]["required_launch_authorization"].pop(
            "must_bind_scientific_decision_engine_source_sha256"
        )
        with self.assertRaisesRegex(ValueError, "launch requirement drifted"):
            freezer.validate_contract_payload(drifted)

    def test_refuses_existing_output_and_preserves_prior_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = self._temporary_sources(root)
            output = root / freezer.OUTPUT_NAME
            output.write_bytes(b"prior evidence\n")
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                freezer.freeze_launch_authorization(
                    contract_path=freezer.FULL_CONTRACT_PATH,
                    project_commit=PROJECT_COMMIT,
                    source_paths=paths,
                    expected_sha256=expected,
                    output_path=output,
                )
            self.assertEqual(output.read_bytes(), b"prior evidence\n")


if __name__ == "__main__":
    unittest.main()
