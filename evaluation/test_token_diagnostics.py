"""Local tests for the Gate-1 token-collapse diagnostics (no GPU).

Verifies the metric cleanly separates collapsed tokens from rich ones, so the
verdict can be trusted when it runs on real extracted GLIM tokens.

Run: python -B -m unittest evaluation.test_token_diagnostics
"""
import unittest

import torch

from evaluation.token_diagnostics import (
    effective_rank,
    position_liveness,
    token_collapse_report,
    within_trial_redundancy,
)

B, T, D = 16, 96, 1024


def collapsed_tokens(seed=0):
    # every one of the 96 tokens is the same trial direction + tiny noise.
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, 1, D, generator=g)
    return base + 0.001 * torch.randn(B, T, D, generator=g)


def rich_tokens(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g)


class TokenDiagnosticsTests(unittest.TestCase):
    def test_redundancy_separates_collapsed_from_rich(self):
        self.assertGreater(within_trial_redundancy(collapsed_tokens()).mean().item(), 0.99)
        self.assertLess(abs(within_trial_redundancy(rich_tokens()).mean().item()), 0.1)

    def test_effective_rank_separates(self):
        # collapsed => participation ratio near 1; rich => large (many directions).
        self.assertLess(effective_rank(collapsed_tokens()).mean().item(), 1.5)
        self.assertGreater(effective_rank(rich_tokens()).mean().item(), 40.0)

    def test_verdict_collapsed(self):
        r = token_collapse_report(collapsed_tokens())
        self.assertTrue(r["verdict"].startswith("COLLAPSED"), r["verdict"])
        self.assertGreater(r["within_trial_redundancy_mean"], 0.97)
        self.assertLess(r["effective_rank_fraction_of_T"], 0.05)

    def test_verdict_rich(self):
        r = token_collapse_report(rich_tokens())
        self.assertTrue(r["verdict"].startswith("RICH"), r["verdict"])
        self.assertLess(r["within_trial_redundancy_mean"], 0.1)

    def test_invalid_tokens_flagged(self):
        bad = rich_tokens()
        bad[0, 0, 0] = float("nan")
        self.assertTrue(token_collapse_report(bad)["verdict"].startswith("INVALID"))
        zero = rich_tokens()
        zero[0, 0] = 0.0
        self.assertTrue(token_collapse_report(zero)["verdict"].startswith("INVALID"))

    def test_position_liveness_detects_dead_positions(self):
        toks = rich_tokens()
        toks[:, 5, :] = 3.14  # position 5 identical across all trials -> dead
        live = position_liveness(toks)
        self.assertGreater(live["fraction_dead_positions"], 0.0)
        self.assertLess(live["min_position_variance"], 1e-4)


if __name__ == "__main__":
    unittest.main()
