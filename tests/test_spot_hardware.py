import math
import unittest

import torch

from recu_hw.selective_spot import SelectiveSPoTResNet20, named_spot_affines
from recu_hw.spot_hardware import analyze_spot_hardware


class TestSPoTHardware(unittest.TestCase):
    def test_zero_coverage_has_no_extra_ops(self):
        model = SelectiveSPoTResNet20()
        report = analyze_spot_hardware(model)
        self.assertEqual(report["target_channels"], 672)
        self.assertEqual(report["selected_channels"], 0)
        self.assertEqual(report["active_second_term_channels"], 0)
        self.assertEqual(report["extra_shift_operations_per_image_direct_interpretation"], 0)
        self.assertEqual(report["extra_add_sub_operations_per_image_direct_interpretation"], 0)
        self.assertFalse(report["later_h2_width_reaudit_required_if_ported"])

    def test_one_active_layer1_channel_costs_1024_extra_ops(self):
        model = SelectiveSPoTResNet20()
        modules = dict(named_spot_affines(model))
        aff = modules["layer1.0.aff1"]
        with torch.no_grad():
            aff.sign_k[0] = 1.0
            aff.log2_abs_k[0] = math.log2(0.3125)
            mask = torch.zeros(aff.channels, dtype=torch.bool)
            mask[0] = True
            aff.set_selected(mask)

        report = analyze_spot_hardware(model)
        self.assertEqual(report["selected_channels"], 1)
        self.assertEqual(report["active_second_term_channels"], 1)
        self.assertEqual(report["extra_shift_operations_per_image_direct_interpretation"], 1024)
        self.assertEqual(report["extra_add_sub_operations_per_image_direct_interpretation"], 1024)
        self.assertEqual(report["dense_programmable_extra_metadata_bits"], 672 * 7)
        self.assertEqual(report["sparse_static_extra_metadata_bits"], 672 + 6)
        self.assertLessEqual(report["max_extra_carry_bits_at_two_term_add"], 1)
        self.assertTrue(report["later_h2_width_reaudit_required_if_ported"])
        self.assertFalse(report["physical_dsp48e1_zero_proven"])


if __name__ == "__main__":
    unittest.main()
