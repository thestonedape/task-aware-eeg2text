import hashlib
import inspect
import unittest

import torch

from project_adapters.task_segmented_objective import (
    ALLOWED_CANDIDATES_PER_ANCHOR,
    ARM_IDS,
    BATCH_ROWS,
    SharedResidualAdapter,
    build_partition_masks,
    masked_symmetric_alignment_loss,
    task_neutral_forward_signature,
)


def balanced_groups():
    true_groups = []
    pseudo_groups = []
    for true_group in ("NR", "TSR"):
        for pseudo_group in ("P0", "P1"):
            true_groups.extend([true_group] * 16)
            pseudo_groups.extend([pseudo_group] * 16)
    return true_groups, pseudo_groups


def state_hash(model: torch.nn.Module) -> str:
    state = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state.update(name.encode("utf-8"))
        state.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return state.hexdigest()


class TaskSegmentedObjectiveTests(unittest.TestCase):
    def test_model_is_task_neutral_and_exactly_capacity_matched(self):
        self.assertEqual(task_neutral_forward_signature(), ("vector",))
        self.assertEqual(
            tuple(inspect.signature(SharedResidualAdapter.forward).parameters),
            ("self", "vector"),
        )
        model = SharedResidualAdapter()
        self.assertEqual(model.trainable_parameter_count, 196608)
        self.assertEqual(model.active_parameter_count_per_example, 196608)
        vectors = torch.randn(4, 1024)
        self.assertTrue(torch.equal(model(vectors), vectors))
        with self.assertRaises(TypeError):
            model(vectors, ["NR"] * 4)

    def test_every_arm_starts_from_the_identical_state_for_one_seed(self):
        hashes = []
        for _ in ARM_IDS:
            torch.manual_seed(20260718)
            hashes.append(state_hash(SharedResidualAdapter()))
        self.assertEqual(len(set(hashes)), 1)

    def test_true_and_pseudo_masks_have_frozen_semantics_and_degree(self):
        true_groups, pseudo_groups = balanced_groups()
        true_e2t, true_t2e = build_partition_masks(
            "true_task_segmented", true_groups, pseudo_groups, key="unused"
        )
        pseudo_e2t, pseudo_t2e = build_partition_masks(
            "pseudo_task_segmented", true_groups, pseudo_groups, key="unused"
        )
        for mask in (true_e2t, true_t2e, pseudo_e2t, pseudo_t2e):
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(tuple(mask.shape), (BATCH_ROWS, BATCH_ROWS))
            self.assertTrue(torch.equal(
                mask.sum(1),
                torch.full((BATCH_ROWS,), ALLOWED_CANDIDATES_PER_ANCHOR),
            ))
            self.assertTrue(bool(torch.diagonal(mask).all()))
        true_tensor = torch.tensor([0] * 32 + [1] * 32)
        pseudo_tensor = torch.tensor(([0] * 16 + [1] * 16) * 2)
        self.assertTrue(torch.equal(true_e2t, true_tensor[:, None].eq(true_tensor[None, :])))
        self.assertTrue(torch.equal(
            pseudo_e2t, pseudo_tensor[:, None].eq(pseudo_tensor[None, :])
        ))
        self.assertTrue(torch.equal(true_e2t, true_t2e))
        self.assertTrue(torch.equal(pseudo_e2t, pseudo_t2e))

    def test_global_masks_are_deterministic_directional_and_key_sensitive(self):
        true_groups, pseudo_groups = balanced_groups()
        first = build_partition_masks(
            "global_mixed", true_groups, pseudo_groups, key="seed/epoch/batch-A"
        )
        repeated = build_partition_masks(
            "global_mixed", true_groups, pseudo_groups, key="seed/epoch/batch-A"
        )
        changed = build_partition_masks(
            "global_mixed", true_groups, pseudo_groups, key="seed/epoch/batch-B"
        )
        self.assertTrue(torch.equal(first[0], repeated[0]))
        self.assertTrue(torch.equal(first[1], repeated[1]))
        self.assertFalse(torch.equal(first[0], first[1]))
        self.assertFalse(torch.equal(first[0], changed[0]))
        self.assertFalse(torch.equal(first[1], changed[1]))
        for mask in first:
            self.assertTrue(torch.equal(
                mask.sum(1),
                torch.full((BATCH_ROWS,), ALLOWED_CANDIDATES_PER_ANCHOR),
            ))
            self.assertTrue(bool(torch.diagonal(mask).all()))

    def test_invalid_balanced_batch_contract_is_rejected(self):
        true_groups, pseudo_groups = balanced_groups()
        with self.assertRaisesRegex(ValueError, "exactly 64"):
            build_partition_masks(
                "global_mixed", true_groups[:-1], pseudo_groups[:-1], key=1
            )
        with self.assertRaisesRegex(ValueError, "exactly 2 groups"):
            build_partition_masks(
                "global_mixed", ["NR"] * 64, pseudo_groups, key=1
            )
        broken = list(pseudo_groups)
        broken[0] = "P1"
        with self.assertRaisesRegex(ValueError, "balanced 2x2"):
            build_partition_masks(
                "true_task_segmented", true_groups, broken, key=1
            )
        with self.assertRaisesRegex(ValueError, "unknown P4b arm"):
            build_partition_masks("other", true_groups, pseudo_groups, key=1)

    def test_masked_symmetric_loss_is_finite_and_backpropagates(self):
        true_groups, pseudo_groups = balanced_groups()
        torch.manual_seed(9)
        model = SharedResidualAdapter()
        eeg = torch.randn(BATCH_ROWS, 1024)
        text = torch.randn(BATCH_ROWS, 1024)
        adapted = model(eeg)
        for arm_id in ARM_IDS:
            masks = build_partition_masks(
                arm_id, true_groups, pseudo_groups, key="gradient-test"
            )
            loss = masked_symmetric_alignment_loss(adapted, text, *masks)
            self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ))

    def test_loss_rejects_masks_without_aligned_positives(self):
        eeg = torch.randn(BATCH_ROWS, 8)
        text = torch.randn(BATCH_ROWS, 8)
        mask = torch.ones((BATCH_ROWS, BATCH_ROWS), dtype=torch.bool)
        broken = mask.clone()
        broken[0, 0] = False
        with self.assertRaisesRegex(ValueError, "diagonal"):
            masked_symmetric_alignment_loss(eeg, text, broken, mask)


if __name__ == "__main__":
    unittest.main()
