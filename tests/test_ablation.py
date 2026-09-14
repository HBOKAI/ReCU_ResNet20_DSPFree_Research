import unittest
import torch
import torch.nn as nn

from recu_hw.ablation import (
    build_r1_from_official,
    build_r2_from_official,
    build_r3_from_official,
)
from recu_hw.fused_affine import FusedAffine2d, fold_recu_alpha_bn
from recu_hw.model import ReCUResNet20


class TestAblations(unittest.TestCase):
    def test_r1_alpha_is_inactive_and_frozen(self):
        official = ReCUResNet20(
            num_classes=10, activation_mode="prelu", alpha_mode="float"
        )
        r1 = build_r1_from_official(official)
        alphas = [m.alpha for m in r1.modules() if hasattr(m, "alpha")]
        self.assertEqual(sum(a.numel() for a in alphas), 672)
        self.assertTrue(all(not a.requires_grad for a in alphas))

    def test_r2_prelu_to_qrprelu_initialization(self):
        official = ReCUResNet20(
            num_classes=10, activation_mode="prelu", alpha_mode="float"
        )
        with torch.no_grad():
            for m in official.modules():
                if isinstance(m, nn.PReLU):
                    m.weight.fill_(0.125)

        r2, nbad = build_r2_from_official(official)
        self.assertEqual(nbad, 0)
        exps = []
        from recu_hw.qrprelu import QuantizedRPReLU
        for m in r2.modules():
            if isinstance(m, QuantizedRPReLU):
                exps.extend(m.integer_shift_exponents().tolist())
        self.assertTrue(all(v == -3 for v in exps))

    def test_fold_alpha_bn_module_equivalence(self):
        official = ReCUResNet20(
            num_classes=10, activation_mode="prelu", alpha_mode="float"
        )
        block = official.layer1[0]
        block.bn1.eval()

        x = torch.randn(2, 16, 8, 8)
        # Test the algebra on arbitrary raw S values, independent of BConv.
        with torch.no_grad():
            alpha = block.conv1.alpha.view(1, -1, 1, 1)
            y_ref = block.bn1(x * alpha)

            k, b = fold_recu_alpha_bn(block.conv1, block.bn1)
            y_fold = x * k.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)

        self.assertLess(float((y_ref - y_fold).abs().max()), 1e-5)

    def test_pow2_affine_effective_k_is_power_of_two(self):
        m = FusedAffine2d(4, mode="pow2")
        with torch.no_grad():
            m.sign_k.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
            m.log2_abs_k.copy_(torch.tensor([-2.2, -0.7, 0.3, 2.2]))
        k = m.effective_k().abs()
        expected = torch.tensor([0.25, 0.5, 1.0, 4.0])
        self.assertTrue(torch.allclose(k, expected))


if __name__ == "__main__":
    unittest.main()
