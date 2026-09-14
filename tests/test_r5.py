import unittest
import torch

from recu_hw.r4 import R4ResNet20
from recu_hw.r5 import (
    Pow2SignedInt8FakeQuant,
    R5ResNet20,
    build_r5_from_r4,
)


class TestR5(unittest.TestCase):
    def test_forward_shape(self):
        m = R5ResNet20(input_scale_exp=-5)
        y = m(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 10))

    def test_input_quantizer_is_int8_grid(self):
        q = Pow2SignedInt8FakeQuant(scale_exp=-5)
        x = torch.tensor([-4.5, -1.0, 0.0, 1.0, 4.5])
        qi = q.quantize_int(x)
        self.assertTrue(bool((qi >= -128).all()))
        self.assertTrue(bool((qi <= 127).all()))
        self.assertTrue(torch.allclose(qi, torch.round(qi)))

    def test_binary_stem_weight_is_pm1(self):
        m = R5ResNet20()
        bw = m.conv1.binary_weight()
        self.assertTrue(bool(((bw == 1) | (bw == -1)).all()))

    def test_r4_backbone_state_preserved(self):
        r4 = R4ResNet20()
        r5 = build_r5_from_r4(r4, input_scale_exp=-5)

        # Spot-check head and first R4 block state.
        self.assertTrue(torch.allclose(
            r4.linear.weight,
            r5.linear.weight,
        ))
        self.assertTrue(torch.allclose(
            r4.layer1[0].conv1.weight,
            r5.layer1[0].conv1.weight,
        ))

    def test_r4_compatibility_alpha_is_frozen_in_r5(self):
        r4 = R4ResNet20()
        r5 = build_r5_from_r4(r4, input_scale_exp=-5)
        alphas = [
            m.alpha for m in r5.modules()
            if hasattr(m, "alpha")
        ]
        self.assertEqual(sum(p.numel() for p in alphas), 672)
        self.assertTrue(all(not p.requires_grad for p in alphas))


if __name__ == "__main__":
    unittest.main()
