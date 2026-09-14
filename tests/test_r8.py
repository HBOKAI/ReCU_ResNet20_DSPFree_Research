import unittest
import torch
from recu_hw.r8 import BinaryLinearSTE, Pow2LinearSTE
from recu_hw.r6 import is_power_of_two_tensor

class TestR8(unittest.TestCase):
    def test_binary_fc(self):
        m=BinaryLinearSTE(64,10,bias=True)
        w=m.effective_weight()
        self.assertTrue(bool(((w==1)|(w==-1)).all()))
        self.assertIsNotNone(m.bias)

    def test_pow2_fc(self):
        m=Pow2LinearSTE(64,10,bias=True)
        with torch.no_grad():
            m.weight.uniform_(-0.2,0.2)
            m.weight[m.weight==0]=0.01
        self.assertTrue(is_power_of_two_tensor(m.effective_weight()))
        self.assertIsNotNone(m.bias)

    def test_output_shape_binary(self):
        m=BinaryLinearSTE(64,10,bias=True)
        self.assertEqual(tuple(m(torch.randn(4,64)).shape),(4,10))

    def test_output_shape_pow2(self):
        m=Pow2LinearSTE(64,10,bias=True)
        self.assertEqual(tuple(m(torch.randn(4,64)).shape),(4,10))

if __name__=="__main__":
    unittest.main()
