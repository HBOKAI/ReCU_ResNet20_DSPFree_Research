import unittest, torch
from recu_hw.r4 import R4ResNet20
from recu_hw.r5ab import R5ABResNet20, build_r5ab_from_r4, progressive_lambda

class TestR5AB(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(tuple(R5ABResNet20()(torch.randn(2,3,32,32)).shape),(2,10))
    def test_schedule(self):
        self.assertAlmostEqual(progressive_lambda(0,30),1/30)
        self.assertAlmostEqual(progressive_lambda(29,30),1.0)
        self.assertAlmostEqual(progressive_lambda(80,30),1.0)
    def test_lambda_zero(self):
        m=R5ABResNet20(); m.conv1.set_progress_lambda(0)
        self.assertTrue(torch.allclose(m.conv1.effective_weight(),m.conv1.weight))
    def test_lambda_one(self):
        m=R5ABResNet20(); m.conv1.set_progress_lambda(1)
        ref=m.conv1.per_filter_alpha()*m.conv1.binary_sign_weight()
        self.assertTrue(torch.allclose(m.conv1.effective_weight(),ref))
    def test_alpha_shape(self):
        self.assertEqual(tuple(R5ABResNet20().conv1.per_filter_alpha().shape),(16,1,1,1))
    def test_r4_copy(self):
        r4=R4ResNet20(); r5=build_r5ab_from_r4(r4)
        self.assertTrue(torch.allclose(r4.linear.weight,r5.linear.weight))

    def test_r4_compatibility_alpha_is_frozen(self):
        r4=R4ResNet20(); r5=build_r5ab_from_r4(r4)
        alphas = [m.alpha for m in r5.modules() if hasattr(m, "alpha")]
        self.assertEqual(sum(p.numel() for p in alphas), 672)
        self.assertTrue(all(not p.requires_grad for p in alphas))

    def test_trainable_parameter_count(self):
        r4=R4ResNet20(); r5=build_r5ab_from_r4(r4)
        self.assertEqual(
            sum(p.numel() for p in r5.parameters() if p.requires_grad),
            270858,
        )

if __name__=="__main__":
    unittest.main()
