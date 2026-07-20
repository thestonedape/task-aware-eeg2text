"""Local regression tests for the token-level late-interaction retrieval module.

These prove the scorer, the controls, and the loss behave before any GPU run.
Run: python -B -m unittest evaluation.test_token_late_interaction
"""
import math
import unittest

import torch

from project_adapters.token_late_interaction import (
    P4B_TRAINABLE_PARAMETERS,
    TokenLateInteractionAdapter,
    l2_normalize,
    maxsim_matrix,
    mean_collapsed,
    reciprocal_rank_of_positive,
    symmetric_maxsim_loss,
)


class TokenLateInteractionTests(unittest.TestCase):
    def test_parameter_budget_matches_p4b(self):
        model = TokenLateInteractionAdapter()
        self.assertEqual(model.trainable_parameter_count, P4B_TRAINABLE_PARAMETERS)
        self.assertEqual(model.trainable_parameter_count, 196_608)

    def test_zero_init_is_identity(self):
        # up-projection is zero-initialised, so at init the residual is a no-op.
        torch.manual_seed(0)
        model = TokenLateInteractionAdapter()
        tokens = torch.randn(3, 96, 1024)
        projected = model.project(tokens)
        self.assertTrue(torch.allclose(projected, tokens, atol=1e-6))

    def test_project_shape_and_gradients(self):
        model = TokenLateInteractionAdapter()
        tokens = torch.randn(2, 96, 1024)
        out = model.project(tokens)
        self.assertEqual(out.shape, (2, 96, 1024))
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))

    def test_maxsim_matches_manual(self):
        # Two EEG tokens, two text tokens, 2-D. Build orthonormal-ish vectors.
        eeg = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])          # [1,2,2]
        txt = torch.tensor([[[1.0, 0.0], [0.7071, 0.7071]]])   # [1,2,2]
        # cos(e0,t0)=1, cos(e0,t1)=0.7071 -> max 1.0
        # cos(e1,t0)=0, cos(e1,t1)=0.7071 -> max 0.7071
        # mean = (1.0 + 0.7071)/2 = 0.85355
        score = maxsim_matrix(eeg, txt)[0, 0].item()
        self.assertAlmostEqual(score, 0.85355, places=4)

    def test_order_invariance_justifies_mean_collapse_control(self):
        # MaxSim is invariant to token order on BOTH sides. This is why permuting
        # token order is not a valid control; mean_collapsed is used instead.
        torch.manual_seed(1)
        eeg = torch.randn(2, 5, 8)
        txt = torch.randn(3, 6, 8)
        base = maxsim_matrix(eeg, txt)
        eeg_perm = eeg[:, torch.randperm(5), :]
        txt_perm = txt[:, torch.randperm(6), :]
        permuted = maxsim_matrix(eeg_perm, txt_perm)
        self.assertTrue(torch.allclose(base, permuted, atol=1e-6))

    def test_text_mask_excludes_padding(self):
        eeg = torch.tensor([[[1.0, 0.0]]])                       # one eeg token
        # real text token cos=0.5; a padded token that would be the max (cos=1).
        txt = torch.tensor([[[0.5, math.sqrt(0.75)], [1.0, 0.0]]])
        mask = torch.tensor([[True, False]])                     # second is padding
        masked = maxsim_matrix(eeg, txt, text_mask=mask)[0, 0].item()
        self.assertAlmostEqual(masked, 0.5, places=4)
        unmasked = maxsim_matrix(eeg, txt)[0, 0].item()
        self.assertAlmostEqual(unmasked, 1.0, places=4)

    def test_eeg_mask_averages_valid_only(self):
        eeg = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])           # two eeg tokens
        txt = torch.tensor([[[1.0, 0.0]]])                       # cos: e0->1, e1->0
        full = maxsim_matrix(eeg, txt)[0, 0].item()              # mean(1,0)=0.5
        self.assertAlmostEqual(full, 0.5, places=4)
        mask = torch.tensor([[True, False]])                     # drop e1
        masked = maxsim_matrix(eeg, txt, eeg_mask=mask)[0, 0].item()  # mean(1)=1.0
        self.assertAlmostEqual(masked, 1.0, places=4)

    def test_mean_collapsed(self):
        tokens = torch.tensor([[[0.0, 0.0], [2.0, 4.0]]])        # [1,2,2]
        mask = torch.tensor([[True, True]])
        collapsed = mean_collapsed(tokens, mask)
        self.assertEqual(collapsed.shape, (1, 1, 2))
        self.assertTrue(torch.allclose(collapsed[0, 0], torch.tensor([1.0, 2.0])))
        # masking out the second token changes the mean
        collapsed2 = mean_collapsed(tokens, torch.tensor([[True, False]]))
        self.assertTrue(torch.allclose(collapsed2[0, 0], torch.tensor([0.0, 0.0])))

    def test_symmetric_loss_decreases_and_separates(self):
        # Synthetic: each trial's EEG tokens and its positive text share a hidden
        # direction; a few Adam steps should reduce loss and lift the diagonal.
        torch.manual_seed(7)
        b, te, tt, d = 8, 6, 5, 16
        anchors = l2_normalize(torch.randn(b, d))
        eeg = l2_normalize(anchors[:, None, :] + 0.3 * torch.randn(b, te, d), dim=-1)
        txt = l2_normalize(anchors[:, None, :] + 0.3 * torch.randn(b, tt, d), dim=-1)
        model = TokenLateInteractionAdapter.__new__(TokenLateInteractionAdapter)
        torch.nn.Module.__init__(model)
        model.vector_dim, model.rank = d, 4
        from project_adapters.task_treatment_pilots import LowRankResidual
        model.shared = LowRankResidual(d, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        def loss_and_diag():
            pe = model.project(eeg)
            scores = maxsim_matrix(pe, txt)
            loss = symmetric_maxsim_loss(pe, txt, temperature=0.05)
            diag = scores.diag().mean().item()
            off = (scores.sum() - scores.diag().sum()).item() / (b * b - b)
            return loss, diag - off

        first_loss, first_gap = loss_and_diag()
        for _ in range(50):
            opt.zero_grad()
            symmetric_maxsim_loss(model.project(eeg), txt, temperature=0.05).backward()
            opt.step()
        last_loss, last_gap = loss_and_diag()
        self.assertLess(last_loss.item(), first_loss.item())
        self.assertGreater(last_gap, first_gap)

    def test_reciprocal_rank_of_positive(self):
        # Positive text matches the query exactly; distractors are orthogonal.
        query = torch.tensor([[[1.0, 0.0]]])                     # [1,1,2]
        candidates = torch.tensor([
            [[0.0, 1.0]],   # orthogonal
            [[1.0, 0.0]],   # positive (index 1)
            [[0.0, 1.0]],   # orthogonal
        ])
        rr = reciprocal_rank_of_positive(query, candidates, positive_index=1)
        self.assertAlmostEqual(rr, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
