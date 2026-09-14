import unittest

import torch

from recu_hw.h2_workspace import verify_gap_deferred_scaling
from recu_hw.h2_integer_accum import (
    arithmetic_right_shift_round,
    binary_dot_exact_width,
    exact_add_width,
    gap_exact_width,
    signed_bits_for_range,
)


class TestH2IntegerAccum(unittest.TestCase):
    def test_binary_widths(self):
        self.assertEqual(binary_dot_exact_width(3, 96)["signed_acc_bits"], 11)
        self.assertEqual(binary_dot_exact_width(3, 16)["signed_acc_bits"], 9)
        self.assertEqual(binary_dot_exact_width(3, 32)["signed_acc_bits"], 10)
        self.assertEqual(binary_dot_exact_width(3, 64)["signed_acc_bits"], 11)

    def test_popcount_widths(self):
        self.assertEqual(binary_dot_exact_width(3, 96)["popcount_bits"], 10)
        self.assertEqual(binary_dot_exact_width(3, 16)["popcount_bits"], 8)
        self.assertEqual(binary_dot_exact_width(3, 32)["popcount_bits"], 9)
        self.assertEqual(binary_dot_exact_width(3, 64)["popcount_bits"], 10)

    def test_signed_range_width(self):
        self.assertEqual(signed_bits_for_range(-864, 864), 11)

    def test_gap_growth(self):
        self.assertEqual(gap_exact_width(6, 64), 12)
        self.assertEqual(gap_exact_width(8, 64), 14)

    def test_add_width(self):
        self.assertEqual(exact_add_width(10, 12), 13)

    def test_round_shift(self):
        x = torch.tensor([5, -5, 6, -6], dtype=torch.int64)
        y = arithmetic_right_shift_round(x, 2)
        self.assertTrue(torch.equal(y, torch.tensor([1, -1, 2, -2])))

    def test_gap_deferred_scale_is_exact(self):
        q_sum = torch.tensor([-32768, -65, -1, 0, 1, 65, 32704], dtype=torch.int64)
        result = verify_gap_deferred_scaling(q_sum, input_shift=6)
        self.assertTrue(result["mathematically_equal"])
        self.assertEqual(result["max_abs_error"], 0.0)
        self.assertFalse(result["integer_rounding_applied"])
        self.assertEqual(result["scale_shift_after_deferred_divide"], 12)


if __name__ == "__main__":
    unittest.main()
