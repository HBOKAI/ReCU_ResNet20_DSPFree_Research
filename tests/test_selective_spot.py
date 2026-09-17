import unittest

import torch

from recu_hw.model import ReCUResNet20
from recu_hw.r4 import build_r4_from_r2
from recu_hw.selective_spot import (
    EXPECTED_BACKBONE_K,
    SelectiveSPoTResNet20,
    apply_ranked_selection,
    build_selective_spot_from_r2,
    coverage_to_count,
    flatten_target_channels,
    named_spot_affines,
    target_channel_count,
    verify_target_count,
)
from recu_hw.study_data import deterministic_train_split_indices


class TestSelectiveSPoT(unittest.TestCase):
    def _r2(self):
        return ReCUResNet20(
            num_classes=10,
            activation_mode="qrprelu",
            alpha_mode="float",
        )

    def test_target_population_is_672(self):
        m = SelectiveSPoTResNet20()
        self.assertEqual(target_channel_count(m), EXPECTED_BACKBONE_K)
        self.assertEqual(verify_target_count(m), EXPECTED_BACKBONE_K)
        self.assertEqual(len(list(named_spot_affines(m))), 18)

    def test_zero_coverage_matches_r4_forward(self):
        torch.manual_seed(7)
        r2 = self._r2()
        r4 = build_r4_from_r2(r2).eval()
        spot = build_selective_spot_from_r2(r2).eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            yr4 = r4(x)
            yspot = spot(x)
        self.assertTrue(torch.equal(yr4, yspot))

    def test_compatibility_alpha_is_still_frozen(self):
        spot = build_selective_spot_from_r2(self._r2())
        alphas = [m.alpha for m in spot.modules() if hasattr(m, "alpha")]
        self.assertEqual(sum(p.numel() for p in alphas), EXPECTED_BACKBONE_K)
        self.assertTrue(all(not p.requires_grad for p in alphas))

    def test_coverage_counts_are_explicit(self):
        self.assertEqual(coverage_to_count(672, 0), 0)
        self.assertEqual(coverage_to_count(672, 5), 34)
        self.assertEqual(coverage_to_count(672, 10), 67)
        self.assertEqual(coverage_to_count(672, 25), 168)
        self.assertEqual(coverage_to_count(672, 50), 336)
        self.assertEqual(coverage_to_count(672, 100), 672)

    def test_ranked_selection_is_deterministic_and_exact(self):
        m = SelectiveSPoTResNet20()
        ranking = flatten_target_channels(m)
        manifest1 = apply_ranked_selection(m, ranking, 10)
        selected1 = [
            (name, ch)
            for name, module in named_spot_affines(m)
            for ch, flag in enumerate(module.selected_2term.cpu().tolist())
            if flag
        ]
        manifest2 = apply_ranked_selection(m, ranking, 10)
        selected2 = [
            (name, ch)
            for name, module in named_spot_affines(m)
            for ch, flag in enumerate(module.selected_2term.cpu().tolist())
            if flag
        ]
        self.assertEqual(len(manifest1), 67)
        self.assertEqual(manifest1, manifest2)
        self.assertEqual(selected1, selected2)

    def test_invalid_or_duplicate_ranking_is_rejected(self):
        m = SelectiveSPoTResNet20()
        ranking = flatten_target_channels(m)
        bad = list(ranking)
        bad[-1] = dict(bad[0])
        with self.assertRaises(ValueError):
            apply_ranked_selection(m, bad, 25)

    def test_model_checkpoint_reload_keeps_effective_k(self):
        m = build_selective_spot_from_r2(self._r2())
        ranking = flatten_target_channels(m)
        apply_ranked_selection(m, ranking, 25)
        before = {
            name: module.effective_k().detach().clone()
            for name, module in named_spot_affines(m)
        }

        clone = SelectiveSPoTResNet20()
        clone.load_state_dict(m.state_dict(), strict=True)
        after = {
            name: module.effective_k().detach().clone()
            for name, module in named_spot_affines(clone)
        }
        self.assertEqual(set(before), set(after))
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]))

    def test_train_split_is_deterministic_disjoint_and_train_only(self):
        a = deterministic_train_split_indices(50000, 5000, 4096, 123)
        b = deterministic_train_split_indices(50000, 5000, 4096, 123)
        self.assertEqual(a, b)
        train = set(a["train_indices"])
        val = set(a["val_indices"])
        calibration = set(a["calibration_indices"])
        self.assertFalse(train & val)
        self.assertTrue(calibration <= train)
        self.assertEqual(len(val), 5000)
        self.assertEqual(len(calibration), 4096)


if __name__ == "__main__":
    unittest.main()
