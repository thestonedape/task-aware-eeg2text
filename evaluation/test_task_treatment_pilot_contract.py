import json
import unittest
from pathlib import Path

from project_adapters.task_treatment_pilots import CONFIG_IDS, PILOT_SPECS, parameter_budget


class TaskTreatmentPilotContractTests(unittest.TestCase):
    def test_machine_contract_matches_implementation(self):
        path = Path(__file__).with_name("task_treatment_pilot_contract.json")
        contract = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(contract["status"], "frozen_before_pilot_training")
        self.assertEqual(contract["inputs"]["representation_prompt_mode"], "all_masked")
        self.assertFalse(contract["inputs"]["backbone_trainable"])
        self.assertFalse(contract["evaluation"]["held_out_test_accessed"])
        self.assertEqual(contract["training"]["auxiliary_factor_losses"], [])

        rows = {row["id"]: row for row in contract["configurations"]}
        self.assertEqual(tuple(rows), CONFIG_IDS)
        counts = parameter_budget()
        for config_id in CONFIG_IDS:
            self.assertEqual(rows[config_id]["shared_rank"], PILOT_SPECS[config_id].shared_rank)
            self.assertEqual(
                rows[config_id]["private_rank_per_task"], PILOT_SPECS[config_id].private_rank
            )
            self.assertEqual(rows[config_id]["trainable_parameters_at_1024d"], counts[config_id])
        self.assertLessEqual(
            counts["maximum_relative_deviation"],
            contract["parameter_control"]["maximum_allowed_relative_deviation"],
        )


if __name__ == "__main__":
    unittest.main()
