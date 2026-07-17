import unittest

import torch

from project_adapters.task_treatment_pilots import (
    CONFIG_IDS,
    TaskTreatmentPilot,
    parameter_budget,
    symmetric_alignment_loss,
)


class TaskTreatmentPilotTests(unittest.TestCase):
    def test_parameter_budget_is_matched_within_frozen_tolerance(self):
        report = parameter_budget()
        self.assertEqual(report["generic_pooled"], 196608)
        self.assertEqual(report["separate_per_task"], 196608)
        self.assertEqual(report["masked_shared_private"], 196608)
        self.assertEqual(report["task_token"], 196896)
        self.assertLessEqual(report["maximum_relative_deviation"], 0.002)

    def test_all_configurations_start_as_identity(self):
        vectors = torch.randn(6, 16)
        tasks = ["SR", "NR", "TSR", "SR", "NR", "TSR"]
        for config_id in CONFIG_IDS:
            model = TaskTreatmentPilot(config_id, vector_dim=16)
            self.assertTrue(torch.equal(model(vectors, tasks), vectors), config_id)

    def test_generic_is_invariant_to_task_condition(self):
        model = TaskTreatmentPilot("generic_pooled", vector_dim=8)
        with torch.no_grad():
            model.shared.up.weight.fill_(0.1)
        vectors = torch.randn(3, 8)
        tasks = ["SR", "NR", "TSR"]
        expected = model(vectors, tasks, "correct")
        self.assertTrue(torch.equal(expected, model(vectors, tasks, "masked")))
        self.assertTrue(torch.equal(expected, model(vectors, tasks, "shuffled")))

    def test_masked_shared_private_disables_every_private_branch(self):
        model = TaskTreatmentPilot("masked_shared_private", vector_dim=8)
        with torch.no_grad():
            for index, adapter in enumerate(model.private):
                adapter.up.weight.fill_(float(index + 1))
        vectors = torch.randn(3, 8)
        tasks = ["SR", "NR", "TSR"]
        masked = model(vectors, tasks, "masked")
        self.assertTrue(torch.equal(masked, vectors))
        self.assertFalse(torch.equal(model(vectors, tasks, "correct"), masked))

    def test_alignment_loss_is_finite_and_backpropagates(self):
        eeg = torch.randn(4, 8, requires_grad=True)
        text = torch.randn(4, 8)
        loss = symmetric_alignment_loss(eeg, text)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(eeg.grad)


if __name__ == "__main__":
    unittest.main()

