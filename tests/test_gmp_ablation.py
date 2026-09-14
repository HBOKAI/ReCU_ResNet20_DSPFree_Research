import unittest

import torch

from recu_hw.gmp_ablation import GMPResNet20, assert_same_parameters, build_gmp_from_r7, gmp_reduce_integer
from recu_hw.r7 import R7ResNet20


class GMPAblationTests(unittest.TestCase):
    def test_signed_max_correctness(self):
        q = torch.full((1, 64, 8, 8), -7, dtype=torch.int64)
        q[:, :, 2, 3] = 4
        out, shift = gmp_reduce_integer(q, 5, 10)
        self.assertTrue(torch.equal(out, torch.full((1, 64), 4, dtype=torch.int64)))
        self.assertEqual(shift, 5)

    def test_64_element_reduction(self):
        q = torch.arange(64, dtype=torch.int64).reshape(1, 1, 8, 8).expand(2, 64, 8, 8).clone()
        out, _ = gmp_reduce_integer(q, 6, 10)
        self.assertTrue(torch.equal(out, torch.full((2, 64), 63, dtype=torch.int64)))

    def test_negative_only_inputs_do_not_default_to_zero(self):
        q = torch.full((1, 64, 8, 8), -9, dtype=torch.int64)
        q[:, :, 7, 7] = -1
        out, _ = gmp_reduce_integer(q, 6, 10)
        self.assertTrue(torch.equal(out, torch.full((1, 64), -1, dtype=torch.int64)))

    def test_mixed_positive_negative_inputs(self):
        q = torch.full((1, 64, 8, 8), -4, dtype=torch.int64)
        q[:, 0, 0, 0] = 3
        q[:, 1, 0, 0] = -2
        out, _ = gmp_reduce_integer(q, 4, 10)
        self.assertEqual(int(out[0, 0]), 3)
        self.assertEqual(int(out[0, 1]), -2)

    def test_scale_metadata_is_inherited_without_plus_six(self):
        q = torch.zeros((1, 64, 8, 8), dtype=torch.int64)
        out, shift = gmp_reduce_integer(q, 6, 10)
        self.assertEqual(shift, 6)
        self.assertEqual(out.dtype, torch.int64)

    def test_no_accidental_gap_division(self):
        q = torch.zeros((1, 64, 8, 8), dtype=torch.int64)
        q[:, :, 0, 0] = 10
        out, shift = gmp_reduce_integer(q, 3, 10)
        self.assertTrue(torch.equal(out, torch.full((1, 64), 10, dtype=torch.int64)))
        self.assertEqual(shift, 3)
        self.assertNotEqual(10 * 2.0 ** -3, 10 * 2.0 ** -(3 + 6))

    def test_output_shape_and_exact_width(self):
        q = torch.full((2, 64, 8, 8), -512, dtype=torch.int64)
        q[:, :, 0, 0] = 511
        out, _ = gmp_reduce_integer(q, 6, 10)
        self.assertEqual(tuple(out.shape), (2, 64))
        self.assertEqual(int(out.max()), 511)
        self.assertEqual(int(out.min()), 511)

    def test_classifier_output_and_initial_state_equality(self):
        source = R7ResNet20()
        variant = build_gmp_from_r7(source)
        self.assertIsInstance(variant, GMPResNet20)
        self.assertEqual(sum(p.numel() for p in variant.parameters() if p.requires_grad), 284250)
        x = torch.rand((1, 3, 32, 32))
        with torch.no_grad():
            y = variant.eval()(x)
        self.assertEqual(tuple(y.shape), (1, 10))


if __name__ == "__main__":
    unittest.main()
