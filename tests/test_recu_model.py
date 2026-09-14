import unittest
import torch

from recu_hw.model import ReCUResNet20, parameter_breakdown
from recu_hw.layers import ReCUBinaryConv2d, iter_recu_binary_convs
from recu_hw.schedule import recu_tau
from recu_hw.qrprelu import QuantizedRPReLU


class TestReCU(unittest.TestCase):
    def test_official_parameter_count(self):
        model = ReCUResNet20(
            activation_mode="prelu",
            alpha_mode="float",
        )
        p = parameter_breakdown(model)
        self.assertEqual(p["total"], 270858)
        self.assertEqual(p["alpha"], 672)
        self.assertEqual(p["prelu"], 336)
        self.assertEqual(p["qrprelu"], 0)

    def test_forward_shape(self):
        model = ReCUResNet20()
        y = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (2, 10))

    def test_tau_schedule(self):
        t0 = recu_tau(0, 600, 0.85, 0.99)
        t600 = recu_tau(600, 600, 0.85, 0.99)
        self.assertAlmostEqual(t0, 0.85, places=6)
        self.assertAlmostEqual(t600, 0.99, places=6)

    def test_binary_conv_count_and_alpha(self):
        model = ReCUResNet20()
        convs = list(iter_recu_binary_convs(model))
        self.assertEqual(len(convs), 18)
        self.assertEqual(sum(m.alpha.numel() for m in convs), 672)

    def test_qrprelu_parameter_count(self):
        model = ReCUResNet20(
            activation_mode="qrprelu",
            alpha_mode="float",
        )
        p = parameter_breakdown(model)
        # QRPReLU has a, xi1, xi2 per channel.
        self.assertEqual(p["qrprelu"], 1008)
        self.assertEqual(p["prelu"], 0)

    def test_qrprelu_slopes_are_power_of_two(self):
        m = QuantizedRPReLU(4)
        with torch.no_grad():
            m.a.copy_(torch.tensor([-2.4, -0.6, 0.2, 1.7]))
        exps = m.integer_shift_exponents()
        self.assertTrue(torch.equal(exps, torch.tensor([-2, -1, 0, 2], dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
