"""Isolated GMP variants of the frozen R8B and H2A-v2 datapaths.

No existing GAP model, numerical plan, or checkpoint is modified here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .h1_model_adapter import _IntegerState
from .h2_workspace import H2TraceAdapter
from .r7 import R7ResNet20
from .r8 import Pow2LinearSTE, build_r8b_from_r7


def gmp_reduce_integer(q: torch.Tensor, input_shift: int, input_bits: int):
    """Exact signed 64-to-1/channel max; preserve the input scale and width."""
    if q.ndim != 4 or q.shape[1:] != (64, 8, 8):
        raise ValueError(f"GMP expects [N,64,8,8], got {tuple(q.shape)}")
    if q.dtype != torch.int64:
        raise TypeError("GMP input must be int64 q values")
    qmin = -(1 << (int(input_bits) - 1))
    qmax = (1 << (int(input_bits) - 1)) - 1
    if bool(((q < qmin) | (q > qmax)).any()):
        raise ValueError("GMP input exceeds its frozen H1MP signed width")
    result = q.flatten(2).max(dim=2).values
    assert result.shape == (q.shape[0], 64)
    return result, int(input_shift)


class GMPResNet20(R7ResNet20):
    """R8B topology/parameters with only global average pooling changed to max."""

    def __init__(self, num_classes=10, resolution=8, min_exp=None, max_exp=None):
        super().__init__(num_classes=num_classes, resolution=resolution)
        self.linear = Pow2LinearSTE(64, num_classes, bias=True, min_exp=min_exp, max_exp=max_exp)

    def forward(self, x):
        x = self.thermo(x)
        s = self.conv1(x)
        out = F.hardtanh(self.stem_affine(s), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_max_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.head_affine(out)
        return self.linear(out)


@torch.no_grad()
def build_gmp_from_r7(source_r7, min_exp=None, max_exp=None):
    """Mirror R8B initialization exactly, including the R7 source FC weights."""
    gap_r8b = build_r8b_from_r7(source_r7, min_exp=min_exp, max_exp=max_exp)
    gmp = GMPResNet20(
        num_classes=gap_r8b.linear.out_features,
        resolution=8,
        min_exp=min_exp,
        max_exp=max_exp,
    )
    gmp.load_state_dict(gap_r8b.state_dict(), strict=True)
    assert_same_parameters(gap_r8b, gmp)
    return gmp


def assert_same_parameters(reference, variant):
    left = reference.state_dict()
    right = variant.state_dict()
    if left.keys() != right.keys():
        raise RuntimeError("GAP/GMP state_dict keys differ")
    for name in left:
        if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
            raise RuntimeError(f"GAP/GMP parameter/buffer differs: {name}")
    return True


class GMPTraceAdapter(H2TraceAdapter):
    """H2A-v2 integer replay with only the GAP reducer replaced by signed max."""

    def __init__(self, r8b_model, h1_fake_quant, h0_info):
        super().__init__(r8b_model, h1_fake_quant, h0_info)
        self.last_gmp_q = None
        self.last_gmp_shift = None
        self.last_stage3_q = None

    def reset_runtime(self):
        super().reset_runtime()
        self.last_gmp_q = None
        self.last_gmp_shift = None
        self.last_stage3_q = None

    def _gap_and_head(self, state: _IntegerState):
        if self.h2_plan is None:
            raise RuntimeError("GMP H2 adapter requires the frozen H2A-v2 plan")
        if state.q is None or state.shift is None:
            raise RuntimeError("GMP requires the frozen H1MP integer Stage3 state")
        input_bits = self._h1_bits(state.node_name)
        q, shift = gmp_reduce_integer(state.q.to(torch.int64), state.shift, input_bits)
        q = self._finite("gmp.max", q, input_bits, "assert")
        self.last_stage3_q = state.q.detach().clone()
        self.last_gmp_q = q.detach().clone()
        self.last_gmp_shift = shift
        self._record("gmp.max", q, bits=input_bits, kind="signed_64_element_channel_max")
        head_q, head_shift, _ = self._affine_integer(
            "head.affine_output", "head_affine", self.model.head_affine,
            q, conv_shift=shift, input_bits=input_bits,
        )
        return self._state(
            "head.affine_output", head_q.to(torch.float32) * (2.0 ** (-head_shift))
        )
