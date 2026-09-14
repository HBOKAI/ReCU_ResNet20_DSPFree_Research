import unittest

import torch

from recu_hw.h1_model_adapter import H1ModelAdapter
from recu_hw.integer_activation import IntegerActivationFakeQuant
from recu_hw.r7 import R7ResNet20


class TestH1Adapter(unittest.TestCase):
    def test_lists_true_boundaries_without_accumulators(self):
        adapter = H1ModelAdapter(R7ResNet20(num_classes=10, resolution=8))
        nodes = adapter.list_quant_nodes()
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(len(nodes), 58)
        self.assertIn("stem.post_hardtanh", nodes)
        self.assertIn("layer1.0.x1", nodes)
        self.assertIn("layer1.0.second_add", nodes)
        self.assertIn("layer1.0.qrprelu_output", nodes)
        self.assertIn("layer2.0.stage_shortcut", nodes)
        self.assertIn("layer3.0.stage_shortcut", nodes)
        self.assertIn("head.affine_output", nodes)
        self.assertFalse(any("accumulator" in node.lower() for node in nodes))
        self.assertFalse(any("gap" in node.lower() for node in nodes))

    def test_disabled_adapter_matches_actual_forward(self):
        torch.manual_seed(4)
        model = R7ResNet20(num_classes=10, resolution=8).eval()
        adapter = H1ModelAdapter(model)
        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x), adapter(x), atol=1e-6, rtol=0.0))

    def test_integer_residual_alignment_is_explicit(self):
        torch.manual_seed(5)
        model = R7ResNet20(num_classes=10, resolution=8).eval()
        adapter = H1ModelAdapter(model)
        shifts = {name: 4 for name in adapter.list_quant_nodes()}
        report = adapter.verify_integer_residual_alignment(shifts, bits=8)
        self.assertEqual(report["residual_add_count"], 18)
        self.assertTrue(report["integer_add_equivalence_all"])
        self.assertEqual(len(report["option_a"]), 2)

    def test_option_a_integer_transform(self):
        q = torch.ones(1, 16, 32, 32, dtype=torch.int64)
        out = H1ModelAdapter._option_a_integer(q, 16, 32, 2)
        self.assertEqual(tuple(out.shape), (1, 32, 16, 16))
        self.assertTrue(torch.equal(out[:, :8], torch.zeros_like(out[:, :8])))
        self.assertTrue(torch.equal(out[:, 8:24], torch.ones_like(out[:, 8:24])))
        self.assertTrue(torch.equal(out[:, 24:], torch.zeros_like(out[:, 24:])))

    def test_fake_quant_does_not_touch_h0_bias_parameters(self):
        torch.manual_seed(6)
        model = R7ResNet20(num_classes=10, resolution=8).eval()
        adapter = H1ModelAdapter(model)
        before = {
            name: module.bias.detach().clone()
            for name, module in model.named_modules()
            if getattr(module, "bias", None) is not None
        }
        fq = IntegerActivationFakeQuant(8, {name: 4 for name in adapter.list_quant_nodes()})
        adapter.set_fake_quant(fq)
        with torch.no_grad():
            adapter(torch.rand(1, 3, 32, 32))
        for name, old in before.items():
            self.assertTrue(torch.equal(dict(model.named_modules())[name].bias, old), name)


if __name__ == "__main__":
    unittest.main()
