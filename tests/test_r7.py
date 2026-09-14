import unittest
import torch
import torch.nn as nn
from recu_hw.r4 import R4ResNet20
from recu_hw.r5t import build_r5t_from_r4
from recu_hw.r6 import R6ResNet20, build_r6_from_r5t, is_power_of_two_tensor, parameter_freeze_report
from recu_hw.r7 import HeadFusedPow2Affine1d, R7ResNet20, build_r7_from_r6

class TestR7(unittest.TestCase):
    def test_exact_head_bn_fold(self):
        torch.manual_seed(1)
        bn=nn.BatchNorm1d(64); bn.eval()
        with torch.no_grad():
            bn.weight.uniform_(0.2,1.5)
            bn.bias.uniform_(-0.5,0.5)
            bn.running_mean.uniform_(-1.0,1.0)
            bn.running_var.uniform_(0.2,2.0)
        aff=HeadFusedPow2Affine1d(64,quantize_k=False)
        aff.load_from_bn(bn)
        x=torch.randn(8,64)
        with torch.no_grad():
            err=(bn(x)-aff(x)).abs().max().item()
        self.assertLess(err,1e-5)

    def test_head_pow2(self):
        aff=HeadFusedPow2Affine1d(64,quantize_k=True)
        with torch.no_grad():
            aff.k_latent.uniform_(0.01,2.0)
        self.assertTrue(is_power_of_two_tensor(aff.effective_k()))

    def test_signed_head_k_supports_negative_channels(self):
        aff=HeadFusedPow2Affine1d(2,quantize_k=True)
        with torch.no_grad():
            aff.k_latent.copy_(torch.tensor([-0.20, 0.30]))
        self.assertTrue(torch.equal(aff.effective_k(), torch.tensor([-0.25, 0.25])))

    def test_shapes(self):
        aff=HeadFusedPow2Affine1d(64)
        self.assertEqual(tuple(aff.k_latent.shape),(64,))
        self.assertEqual(tuple(aff.bias.shape),(64,))

    def test_no_bn2(self):
        self.assertFalse(hasattr(R7ResNet20(),"bn2"))

    def test_constructor_freezes_compatibility_alpha(self):
        report=parameter_freeze_report(R7ResNet20())
        self.assertEqual(report["frozen_params"],672)
        self.assertEqual(report["frozen_parameter_categories"],{"r4_compatibility_alpha":18})

    def test_r6_source_compatibility(self):
        r5t=build_r5t_from_r4(R4ResNet20(),resolution=8,binary_stem=True)
        r6=build_r6_from_r5t(r5t,resolution=8)
        r7=build_r7_from_r6(r6,resolution=8)
        self.assertTrue(is_power_of_two_tensor(r7.stem_affine.effective_k()))
        self.assertTrue(is_power_of_two_tensor(r7.head_affine.effective_k()))
        self.assertEqual(parameter_freeze_report(r7)["frozen_params"],672)
        self.assertEqual(parameter_freeze_report(r7)["trainable_params"],284250)
        self.assertFalse(hasattr(r7,"bn2"))

    def test_reload_keeps_r7_invariants(self):
        r7=R7ResNet20()
        reloaded=R7ResNet20()
        reloaded.load_state_dict(r7.state_dict(),strict=True)
        self.assertTrue(is_power_of_two_tensor(reloaded.stem_affine.effective_k()))
        self.assertTrue(is_power_of_two_tensor(reloaded.head_affine.effective_k()))
        ew=reloaded.conv1.effective_weight()
        self.assertTrue(bool(((ew==1)|(ew==-1)).all()))
        self.assertIsNone(reloaded.conv1.bias)
        self.assertIsNotNone(reloaded.linear.bias)
        self.assertFalse(hasattr(reloaded,"bn2"))

    def test_fc_bias_preserved(self):
        self.assertIsNotNone(R7ResNet20().linear.bias)

    def test_forward_shape(self):
        y=R7ResNet20()(torch.rand(2,3,32,32))
        self.assertEqual(tuple(y.shape),(2,10))

if __name__=="__main__":
    unittest.main()
