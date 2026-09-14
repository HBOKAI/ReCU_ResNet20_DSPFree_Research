import unittest
import torch

from recu_hw.integer_bias import (
    collect_integer_bias_targets,
    signed_int_range,
    quantize_bias_tensor,
    search_best_shift,
)
from recu_hw.r7 import R7ResNet20
from recu_hw.r8 import build_r8b_from_r7


class TestIntegerBias(unittest.TestCase):
    def test_signed_ranges(self):
        self.assertEqual(signed_int_range(8), (-128, 127))
        self.assertEqual(signed_int_range(4), (-8, 7))
        self.assertEqual(signed_int_range(3), (-4, 3))

    def test_integer_quantization_no_saturation(self):
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        r = quantize_bias_tensor(x, bits=8, shift=4)
        self.assertEqual(r["saturation_count"], 0)
        self.assertTrue(r["q"].dtype == torch.int64)
        self.assertTrue(torch.all(r["q"] >= -128))
        self.assertTrue(torch.all(r["q"] <= 127))

    def test_clipping(self):
        x = torch.tensor([-100.0, 100.0])
        r = quantize_bias_tensor(x, bits=4, shift=0)
        self.assertEqual(r["saturation_count"], 2)
        self.assertEqual(int(r["q"][0]), -8)
        self.assertEqual(int(r["q"][1]), 7)

    def test_shift_search_prefers_zero_saturation(self):
        x = torch.tensor([-1.3, 0.2, 1.1])
        shift, r = search_best_shift(
            x, bits=5, shift_min=0, shift_max=8, fallback_shift_min=-8
        )
        self.assertEqual(r["saturation_count"], 0)

    def test_export_equivalence(self):
        x = torch.tensor([-0.75, 0.125, 1.5])
        r = quantize_bias_tensor(x, bits=8, shift=5)
        reconstructed = r["q"].to(torch.float64) * (2.0 ** -5)
        self.assertTrue(torch.allclose(
            reconstructed,
            r["x_hat"].to(torch.float64),
            atol=0,
            rtol=0,
        ))

    def test_r8b_target_inventory_is_affine_only(self):
        source = R7ResNet20(num_classes=10, resolution=8)
        model = build_r8b_from_r7(source)
        targets = collect_integer_bias_targets(model)
        self.assertEqual(len(targets), 21)
        self.assertEqual(sum(module.bias.numel() for module in targets.values()), 762)
        self.assertIn("stem_affine", targets)
        self.assertIn("head_affine", targets)
        self.assertIn("linear", targets)
        self.assertEqual(
            len([name for name in targets if name.startswith(("layer1.", "layer2.", "layer3."))]),
            18,
        )
        self.assertFalse(any("post_act" in name for name in targets))

    def test_negative_shift_extension_is_recorded(self):
        x = torch.tensor([100.0, -100.0])
        shift, result = search_best_shift(
            x, bits=4, shift_min=0, shift_max=2, fallback_shift_min=-4
        )
        self.assertLess(shift, 0)
        self.assertTrue(result["used_negative_extension"])
        self.assertEqual(result["saturation_count"], 0)


if __name__ == "__main__":
    unittest.main()
