import json
import unittest
from pathlib import Path

import torch

from recu_hw.h1_model_adapter import H1ModelAdapter
from recu_hw.h1_search_workspace import build_h1_search_loaders
from recu_hw.h1s_mp_quant import (
    CalibrationSampleCollector,
    SelectiveIntegerFakeQuant,
    choose_mse_shift,
    choose_percentile_shift,
    signed_int_range,
)
from recu_hw.r7 import R7ResNet20


class TestH1SMPQuant(unittest.TestCase):
    def test_ranges_and_policies(self):
        self.assertEqual(signed_int_range(8), (-128, 127))
        values = torch.linspace(-3.0, 3.0, 1000)
        self.assertIsInstance(choose_mse_shift(values, 8), int)
        self.assertIsInstance(choose_percentile_shift(values, 8, 99.9), int)

    def test_collector_policy_selection(self):
        collector = CalibrationSampleCollector(sample_cap_per_node=100, values_per_batch=100)
        collector.observe("node", torch.tensor([-1.0, 0.0, 1.0]))
        self.assertIn("node", collector.stats)
        self.assertIsInstance(collector.choose_shift("node", 8, "mse"), int)

    def test_selective_quant_only_targets_one_node(self):
        fq = SelectiveIntegerFakeQuant({"target": 8}, {"target": 4})
        self.assertTrue(fq.is_quantized("target"))
        self.assertFalse(fq.is_quantized("reference"))
        result = fq.quantize("target", torch.tensor([0.5]))
        self.assertEqual(result["q"].dtype, torch.int64)
        self.assertIsNone(fq.quantize("reference", torch.tensor([0.5])))

    def test_mixed_bit_map_residual_alignment(self):
        torch.manual_seed(7)
        model = R7ResNet20(num_classes=10, resolution=8).eval()
        adapter = H1ModelAdapter(model)
        nodes = adapter.list_quant_nodes()
        bits = {name: 6 + (index % 3) for index, name in enumerate(nodes)}
        shifts = {name: 3 for name in nodes}
        fq = SelectiveIntegerFakeQuant(bits, shifts)
        adapter.set_fake_quant(fq)
        with torch.no_grad():
            adapter(torch.rand(1, 3, 32, 32))
        report = adapter.verify_integer_residual_alignment(shifts, bits)
        self.assertEqual(report["residual_add_count"], 18)
        self.assertTrue(report["integer_add_equivalence_all"])
        self.assertTrue(all("branch_a_bits" in item for item in report["records"]))

    def test_train_split_is_deterministic_and_disjoint(self):
        c1, s1, _ = build_h1_search_loaders(20260913, 2, 2)
        c2, s2, _ = build_h1_search_loaders(20260913, 2, 2)
        self.assertEqual(c1.h1_indices, c2.h1_indices)
        self.assertEqual(s1.h1_indices, s2.h1_indices)
        self.assertTrue(set(c1.h1_indices).isdisjoint(s1.h1_indices))

    def test_h0_json_has_fixed_int6_record(self):
        path = Path(__file__).resolve().parents[1] / "H0_INTEGER_BIAS_SWEEP_RESULTS.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [row for row in data["results"] if int(row["bits"]) == 6]
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["layers"]), 21)
        self.assertTrue(all(int(item["bits"]) == 6 for item in rows[0]["layers"].values()))


if __name__ == "__main__":
    unittest.main()
