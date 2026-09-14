import unittest
import torch

from recu_hw.integer_activation import (
    signed_int_range,
    choose_zero_saturation_shift,
    quantize_float_to_int,
    requantize_int,
    aligned_integer_add,
)


class TestIntegerActivation(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(signed_int_range(8), (-128, 127))
        self.assertEqual(signed_int_range(4), (-8, 7))

    def test_zero_sat_shift(self):
        s = choose_zero_saturation_shift(absmax=1.0, bits=8)
        r = quantize_float_to_int(torch.tensor([-1.0, 1.0]), 8, s)
        self.assertEqual(r["saturation_count"], 0)

    def test_requantize_left_shift(self):
        q = torch.tensor([1, -2], dtype=torch.int64)
        out, sat = requantize_int(q, src_shift=3, dst_shift=5)
        self.assertEqual(sat, 0)
        self.assertTrue(torch.equal(out, torch.tensor([4, -8])))

    def test_requantize_right_shift_rounding(self):
        q = torch.tensor([5, -5], dtype=torch.int64)
        out, sat = requantize_int(q, src_shift=5, dst_shift=3)
        self.assertEqual(sat, 0)
        self.assertTrue(torch.equal(out, torch.tensor([1, -1])))

    def test_aligned_add(self):
        a = torch.tensor([4], dtype=torch.int64)  # 4*2^-2 = 1
        b = torch.tensor([8], dtype=torch.int64)  # 8*2^-3 = 1
        r = aligned_integer_add(a, 2, b, 3, out_bits=8, out_shift=3)
        self.assertEqual(int(r["q"][0]), 16)       # 16*2^-3 = 2
        self.assertEqual(r["saturation_count"], 0)


if __name__ == "__main__":
    unittest.main()
