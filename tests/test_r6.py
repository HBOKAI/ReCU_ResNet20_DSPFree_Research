import unittest
import torch
import torch.nn as nn
from recu_hw.r4 import R4ResNet20
from recu_hw.r5t import build_r5t_from_r4
from recu_hw.r6 import (
    StemFusedPow2Affine2d,
    R6ResNet20,
    build_r6_from_r5t,
    is_power_of_two_tensor,
    parameter_freeze_report,
)

class TestR6(unittest.TestCase):
    def test_exact_bn_fold(self):
        torch.manual_seed(1)
        bn=nn.BatchNorm2d(16); bn.eval()
        with torch.no_grad():
            bn.weight.uniform_(0.2,1.5)
            bn.bias.uniform_(-0.5,0.5)
            bn.running_mean.uniform_(-1.0,1.0)
            bn.running_var.uniform_(0.2,2.0)
        aff=StemFusedPow2Affine2d(16,quantize_k=False)
        aff.load_from_bn(bn); aff.eval()
        x=torch.randn(4,16,8,8)
        with torch.no_grad():
            err=(bn(x)-aff(x)).abs().max().item()
        self.assertLess(err,1e-5)

    def test_signed_pow2(self):
        aff=StemFusedPow2Affine2d(16,quantize_k=True)
        with torch.no_grad():
            aff.k_latent.copy_(torch.tensor([
                -0.031,-0.07,-0.13,-0.27,-0.55,-1.1,-2.1,-4.1,
                 0.031, 0.07, 0.13, 0.27, 0.55, 1.1, 2.1, 4.1
            ]))
        self.assertTrue(is_power_of_two_tensor(aff.effective_k()))

    def test_signed_k_supports_negative_channels(self):
        aff=StemFusedPow2Affine2d(2,quantize_k=True)
        with torch.no_grad():
            aff.k_latent.copy_(torch.tensor([-0.20, 0.30]))
        self.assertTrue(torch.equal(aff.effective_k(), torch.tensor([-0.25, 0.25])))

    def test_affine_parameter_shapes(self):
        m=R6ResNet20(resolution=8)
        self.assertEqual(tuple(m.stem_affine.k_latent.shape), (16,))
        self.assertEqual(tuple(m.stem_affine.bias.shape), (16,))

    def test_r6_source_compatibility_and_frozen_alpha(self):
        source=build_r5t_from_r4(R4ResNet20(), resolution=8, binary_stem=True)
        for module in source.modules():
            if hasattr(module, "alpha"):
                module.alpha.requires_grad_(False)
        r6=build_r6_from_r5t(source, resolution=8)
        report=parameter_freeze_report(r6)
        self.assertEqual(report["frozen_params"], 672)
        self.assertEqual(report["frozen_parameter_categories"], {"r4_compatibility_alpha": 18})
        self.assertEqual(report["trainable_params"], 284250)
        self.assertNotIn("bn1", dict(r6.named_modules()))

    def test_reload_keeps_pow2_inference(self):
        source=build_r5t_from_r4(R4ResNet20(), resolution=8, binary_stem=True)
        r6=build_r6_from_r5t(source, resolution=8)
        reloaded=R6ResNet20(resolution=8)
        reloaded.load_state_dict(r6.state_dict(), strict=True)
        self.assertTrue(is_power_of_two_tensor(reloaded.stem_affine.effective_k()))
        ew=reloaded.conv1.effective_weight()
        self.assertTrue(bool(((ew==1)|(ew==-1)).all()))
        self.assertIsNone(reloaded.conv1.bias)
        self.assertFalse(hasattr(reloaded, "bn1"))

    def test_no_bn1(self):
        self.assertFalse(hasattr(R6ResNet20(resolution=8),"bn1"))

    def test_no_conv_bias(self):
        self.assertIsNone(R6ResNet20(resolution=8).conv1.bias)

    def test_constructor_freezes_compatibility_alpha(self):
        report=parameter_freeze_report(R6ResNet20(resolution=8))
        self.assertEqual(report["frozen_params"], 672)
        self.assertEqual(report["frozen_parameter_categories"], {"r4_compatibility_alpha": 18})

    def test_binary_stem(self):
        m=R6ResNet20(resolution=8)
        w=m.conv1.effective_weight()
        self.assertTrue(bool(((w==1)|(w==-1)).all()))

    def test_forward_shape(self):
        y=R6ResNet20(resolution=8)(torch.rand(2,3,32,32))
        self.assertEqual(tuple(y.shape),(2,10))

if __name__=="__main__":
    unittest.main()
