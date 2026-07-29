"""Test the SR-decodability label/group assembly glue (no artifacts, no GPU).

Run: python -B -m unittest evaluation.test_run_sr_decodability
"""
import unittest

from evaluation.run_sr_decodability import labels_and_groups

ONTOLOGY = ["negative", "neutral", "positive"]


class LabelAssemblyTests(unittest.TestCase):
    def test_maps_labels_and_groups_in_order(self):
        rows = [
            {"target_value": "positive", "subject_id": "S1", "trial_id": "t0"},
            {"target_value": "negative", "subject_id": "S2", "trial_id": "t1"},
            {"target_value": "neutral", "subject_id": "S1", "trial_id": "t2"},
        ]
        y, groups = labels_and_groups(rows, ONTOLOGY)
        self.assertEqual(y.tolist(), [2, 0, 1])
        self.assertEqual(groups.tolist(), ["S1", "S2", "S1"])

    def test_label_outside_ontology_raises(self):
        rows = [{"target_value": "mixed", "subject_id": "S1", "trial_id": "t0"}]
        with self.assertRaises(ValueError):
            labels_and_groups(rows, ONTOLOGY)


if __name__ == "__main__":
    unittest.main()
