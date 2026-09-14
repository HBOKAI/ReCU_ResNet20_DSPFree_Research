import unittest
import torch

from recu_hw.model import ReCUResNet20
from recu_hw.qrprelu import QuantizedRPReLU
from recu_hw.r4 import R4ResNet20, build_r4_from_r2


class TestR4(unittest.TestCase):
    def test_forward_shape(self):
        m = R4ResNet20()
        y = m(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 10))

    def test_qrprelu_parameters_preserved(self):
        r2 = ReCUResNet20(
            num_classes=10,
            activation_mode="qrprelu",
            alpha_mode="float",
        )
        with torch.no_grad():
            for m in r2.modules():
                if isinstance(m, QuantizedRPReLU):
                    m.a.uniform_(-4, 0)
                    m.xi1.uniform_(-0.1, 0.1)
                    m.xi2.uniform_(-0.1, 0.1)

        r4 = build_r4_from_r2(r2)
        src = [m for m in r2.modules() if isinstance(m, QuantizedRPReLU)]
        dst = [m for m in r4.modules() if isinstance(m, QuantizedRPReLU)]

        self.assertEqual(len(src), len(dst))
        for s, d in zip(src, dst):
            self.assertTrue(torch.allclose(s.a, d.a))
            self.assertTrue(torch.allclose(s.xi1, d.xi1))
            self.assertTrue(torch.allclose(s.xi2, d.xi2))

    def test_alpha_is_frozen_in_r4(self):
        r2 = ReCUResNet20(
            num_classes=10,
            activation_mode="qrprelu",
            alpha_mode="float",
        )
        r4 = build_r4_from_r2(r2)
        alphas = [m.alpha for m in r4.modules() if hasattr(m, "alpha")]
        self.assertEqual(sum(p.numel() for p in alphas), 672)
        self.assertTrue(all(not p.requires_grad for p in alphas))


if __name__ == "__main__":
    unittest.main()
