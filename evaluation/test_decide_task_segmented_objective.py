from __future__ import annotations

import importlib.util
import unittest

from evaluation.decide_task_segmented_objective import (
    ARM_IDS,
    DecisionSpec,
    build_parser,
    decide,
    reciprocal_rank_from_candidates,
    validate_inputs,
)


TEST_SPEC = DecisionSpec(
    folds=(0, 1),
    seeds=(11, 12),
    task_counts=(("NR", 2), ("TSR", 2)),
    fold_counts=((0, 2), (1, 2)),
    bootstrap_replicates=24,
    bootstrap_base_seed=97,
    chance_mrr=0.1,
)
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def make_runs() -> list[dict[str, object]]:
    return [
        {
            "arm_id": arm,
            "outer_fold": fold,
            "training_seed": seed,
            "status": "complete",
            "run_mode": "full_scientific",
            "full_training_authorized": "true",
            "scientific_decision_permitted": "true",
            "official_validation_used_for_confirmation": "false",
            "held_out_test_accessed": "false",
        }
        for arm in ARM_IDS for fold in TEST_SPEC.folds for seed in TEST_SPEC.seeds
    ]


TRIALS = (
    ("nr-0", 0, "NR", "shared-subject", "text-nr-0"),
    ("nr-1", 1, "NR", "shared-subject", "text-nr-1"),
    ("tsr-0", 0, "TSR", "shared-subject", "text-tsr-0"),
    ("tsr-1", 1, "TSR", "shared-subject", "text-tsr-1"),
)


def make_predictions(true_rank: int = 1) -> list[dict[str, object]]:
    correct_ranks = {
        "global_mixed": 4,
        "true_task_segmented": true_rank,
        "pseudo_task_segmented": 5,
    }
    rows: list[dict[str, object]] = []
    for arm in ARM_IDS:
        for seed in TEST_SPEC.seeds:
            for condition in ("correct", "matched_wrong"):
                rank = correct_ranks[arm] if condition == "correct" else 24
                for trial_id, fold, task, subject, text in TRIALS:
                    rows.append({
                        "arm_id": arm,
                        "training_seed": seed,
                        "outer_fold": fold,
                        "trial_id": trial_id,
                        "reading_task": task,
                        "subject_id": subject,
                        "normalized_text_sha256": text,
                        "signal_condition": condition,
                        "positive_rank": rank,
                        "candidate_pool_size": 24,
                        "scientific_decision_permitted": "true",
                    })
    return rows


class RankTests(unittest.TestCase):
    def test_score_ties_use_frozen_candidate_rank(self) -> None:
        candidates = [
            {"candidate_rank": 0, "is_positive": 0, "score": 0.75},
            {"candidate_rank": 1, "is_positive": 1, "score": 0.75},
            {"candidate_rank": 2, "is_positive": 0, "score": 0.10},
        ]
        self.assertEqual(reciprocal_rank_from_candidates(candidates), 0.5)


class InputValidationTests(unittest.TestCase):
    def test_accepts_complete_cross_fitted_matrix(self) -> None:
        data = validate_inputs(make_runs(), make_predictions(), TEST_SPEC)
        self.assertEqual(len(data.trials), 4)
        self.assertEqual(len(data.values), 12)

    def test_rejects_incomplete_45_run_analogue(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete run matrix"):
            validate_inputs(make_runs()[:-1], make_predictions(), TEST_SPEC)

    def test_rejects_smoke_or_decision_forbidden_run(self) -> None:
        runs = make_runs()
        runs[0]["run_mode"] = "bounded_smoke"
        runs[0]["scientific_decision_permitted"] = "false"
        with self.assertRaisesRegex(ValueError, "smoke/non-scientific"):
            validate_inputs(runs, make_predictions(), TEST_SPEC)

    def test_rejects_official_validation_as_confirmation(self) -> None:
        runs = make_runs()
        runs[0]["official_validation_used_for_confirmation"] = "true"
        with self.assertRaisesRegex(ValueError, "reused official validation"):
            validate_inputs(runs, make_predictions(), TEST_SPEC)

    def test_rejects_inconsistent_outer_fold_for_trial(self) -> None:
        rows = make_predictions()
        target = next(row for row in rows if row["arm_id"] == "global_mixed" and row["trial_id"] == "nr-0")
        target["outer_fold"] = 1
        with self.assertRaisesRegex(ValueError, "metadata/fold changed"):
            validate_inputs(make_runs(), rows, TEST_SPEC)

    def test_rejects_non_24_way_pool(self) -> None:
        rows = make_predictions()
        rows[0]["candidate_pool_size"] = 23
        with self.assertRaisesRegex(ValueError, "not frozen at 24"):
            validate_inputs(make_runs(), rows, TEST_SPEC)

    def test_production_cli_exposes_no_gate_or_bootstrap_override(self) -> None:
        options = {action.dest for action in build_parser()._actions}
        self.assertEqual(options, {"help", "run_manifest", "predictions", "output"})


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for the frozen PCG64 bootstrap")
class DecisionTests(unittest.TestCase):
    def test_all_frozen_gates_must_pass(self) -> None:
        result = decide(validate_inputs(make_runs(), make_predictions(), TEST_SPEC), TEST_SPEC)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["continuation_decision"], "continue_p4b")
        self.assertTrue(all(result["requirements"].values()))
        # 2 superiority + 4 task-NI + 2 signal-NI + chance + signal,
        # each under all three cluster schemes.
        self.assertEqual(len(result["bootstrap_intervals"]), 30)
        self.assertTrue(all(row["replicates"] == 24 for row in result["bootstrap_intervals"]))

    def test_any_failed_gate_forces_stop_and_never_claims_pass(self) -> None:
        rows = make_predictions(true_rank=6)
        result = decide(validate_inputs(make_runs(), rows, TEST_SPEC), TEST_SPEC)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["continuation_decision"], "stop_p4b_permanently")
        self.assertFalse(all(result["requirements"].values()))

    def test_bootstrap_is_deterministic(self) -> None:
        data = validate_inputs(make_runs(), make_predictions(), TEST_SPEC)
        first = decide(data, TEST_SPEC)
        second = decide(data, TEST_SPEC)
        self.assertEqual(first["bootstrap_intervals"], second["bootstrap_intervals"])

if __name__ == "__main__":
    unittest.main()
