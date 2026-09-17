import unittest

import torch

from recu_hw.fused_affine import FusedAffine2d
from recu_hw.spot_affine import (
    SelectiveSPoTAffine2d,
    best_two_term_or_one,
    quantize_one_term_like_r4,
)


class TestSPoTAffine(unittest.TestCase):
    def test_one_term_matches_r4_rule(self):
        k = torch.tensor([0.3, -0.7, 0.25, -2.0])
        q, exp, sign = quantize_one_term_like_r4(k)
        expected_exp = torch.round(torch.log2(k.abs())).to(torch.int64)
        expected = torch.sign(k) * torch.pow(2.0, expected_exp.to(k.dtype))
        self.assertTrue(torch.equal(exp, expected_exp))
        self.assertTrue(torch.equal(sign, torch.sign(k)))
        self.assertTrue(torch.equal(q, expected))

    def test_exact_two_term_known_value(self):
        k = torch.tensor([0.3125])  # 2^-2 + 2^-4
        rep = best_two_term_or_one(k)
        self.assertTrue(bool(rep["active2"][0]))
        self.assertEqual(float(rep["q"][0]), 0.3125)
        self.assertGreater(int(rep["k1"][0]), int(rep["k2"][0]))
        reconstructed = (
            float(rep["s1"][0]) * (2.0 ** int(rep["k1"][0]))
            + float(rep["s2"][0]) * (2.0 ** int(rep["k2"][0]))
        )
        self.assertEqual(reconstructed, 0.3125)

    def test_independent_signs_support_negative_target(self):
        k = torch.tensor([-0.375])
        rep = best_two_term_or_one(k)
        self.assertTrue(bool(rep["active2"][0]))
        self.assertEqual(float(rep["q"][0]), -0.375)
        self.assertIn(int(float(rep["s1"][0])), (-1, 1))
        self.assertIn(int(float(rep["s2"][0])), (-1, 1))
        self.assertGreater(int(rep["k1"][0]), int(rep["k2"][0]))

    def test_exact_one_term_falls_back_without_second_shift(self):
        for value in (0.25, -0.5, 2.0):
            rep = best_two_term_or_one(torch.tensor([value]))
            self.assertFalse(bool(rep["active2"][0]))
            self.assertEqual(float(rep["q"][0]), value)
            self.assertEqual(float(rep["s2"][0]), 0.0)

    def test_unselected_module_matches_fused_affine(self):
        k = torch.tensor([0.31, -0.73, 1.2, -2.7])
        b = torch.tensor([0.1, -0.2, 0.3, -0.4])
        r4 = FusedAffine2d(4, mode="pow2")
        spot = SelectiveSPoTAffine2d(4)
        r4.init_from_kb(k, b)
        spot.init_from_kb(k, b)

        x = torch.randn(2, 4, 3, 3)
        self.assertTrue(torch.equal(r4.effective_k(), spot.effective_k()))
        self.assertTrue(torch.equal(r4(x), spot(x)))

    def test_selected_forward_is_exact_discrete_value(self):
        k = torch.tensor([0.3125, 0.25, -0.375])
        b = torch.zeros(3)
        m = SelectiveSPoTAffine2d(3)
        m.init_from_kb(k, b)
        m.set_selected(torch.tensor([True, True, True]))
        terms = m.discrete_terms()
        effective = m.effective_k().detach()
        self.assertTrue(torch.equal(effective, terms["q"]))
        self.assertTrue(bool(terms["active2"][0]))
        self.assertFalse(bool(terms["active2"][1]))
        self.assertTrue(bool(terms["active2"][2]))

    def test_selected_ste_has_finite_gradient(self):
        m = SelectiveSPoTAffine2d(1)
        m.init_from_kb(torch.tensor([0.3125]), torch.tensor([0.0]))
        m.set_selected(torch.tensor([True]))
        x = torch.ones(1, 1, 1, 1)
        y = m(x).sum()
        y.backward()
        self.assertIsNotNone(m.log2_abs_k.grad)
        self.assertTrue(torch.isfinite(m.log2_abs_k.grad).all())

    def test_checkpoint_reload_preserves_effective_k_and_selection(self):
        src = SelectiveSPoTAffine2d(3)
        src.init_from_kb(torch.tensor([0.3125, -0.375, 0.25]), torch.zeros(3))
        src.set_selected(torch.tensor([True, False, True]))
        before = src.effective_k().detach().clone()

        dst = SelectiveSPoTAffine2d(3)
        dst.load_state_dict(src.state_dict(), strict=True)
        after = dst.effective_k().detach().clone()
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(src.selected_2term, dst.selected_2term))


if __name__ == "__main__":
    unittest.main()
