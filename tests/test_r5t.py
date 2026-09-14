import unittest
import torch

from recu_hw.r4 import R4ResNet20
from recu_hw.r5t import ThermometerEncoder, R5TResNet20, build_r5t_from_r4


class TestR5T(unittest.TestCase):
    def test_r8_length(self):
        self.assertEqual(ThermometerEncoder(8).length, 32)

    def test_r8_rgb_channels(self):
        y = ThermometerEncoder(8)(torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 96, 32, 32))

    def test_bipolar_only(self):
        y = ThermometerEncoder(8)(torch.rand(1, 3, 4, 4))
        self.assertTrue(bool(((y == 1) | (y == -1)).all()))

    def test_fig6_example_r32_109(self):
        bits = ThermometerEncoder(32).encode_uint8_reference(109)
        self.assertEqual(int(bits.sum()), 3)

    def test_r8_255_has_32_ones(self):
        bits = ThermometerEncoder(8).encode_uint8_reference(255)
        self.assertEqual(int(bits.sum()), 32)

    def test_forward_shape(self):
        y = R5TResNet20(resolution=8, binary_stem=True)(torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 10))

    def test_build_from_r4(self):
        r4 = R4ResNet20()
        m = build_r5t_from_r4(r4, resolution=8, binary_stem=False)
        self.assertEqual(tuple(m.conv1.weight.shape), (16, 96, 3, 3))
        self.assertTrue(torch.allclose(r4.linear.weight, m.linear.weight))

    def test_r4_backbone_state_is_copied(self):
        r4 = R4ResNet20()
        m = build_r5t_from_r4(r4, resolution=8, binary_stem=False)
        for src_layer, dst_layer in zip(
            [r4.layer1, r4.layer2, r4.layer3],
            [m.layer1, m.layer2, m.layer3],
        ):
            for src, dst in zip(src_layer, dst_layer):
                for key, value in src.state_dict().items():
                    self.assertTrue(torch.equal(value, dst.state_dict()[key]))

    def test_r4_compatibility_alpha_is_frozen(self):
        r4 = R4ResNet20()
        m = build_r5t_from_r4(r4, resolution=8, binary_stem=False)
        alphas = [module.alpha for module in m.modules() if hasattr(module, "alpha")]
        self.assertEqual(sum(p.numel() for p in alphas), 672)
        self.assertTrue(all(not p.requires_grad for p in alphas))

    def test_trainable_parameter_count(self):
        r4 = R4ResNet20()
        m = build_r5t_from_r4(r4, resolution=8, binary_stem=False)
        self.assertEqual(
            sum(p.numel() for p in m.parameters() if p.requires_grad),
            284250,
        )

    def test_binary_stem_pm1(self):
        m = R5TResNet20(resolution=8, binary_stem=True)
        w = m.conv1.effective_weight()
        self.assertTrue(bool(((w == 1) | (w == -1)).all()))


if __name__ == "__main__":
    unittest.main()
