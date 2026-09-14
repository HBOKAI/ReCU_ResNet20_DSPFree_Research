import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import round_ste
from .r4 import R4BasicBlock


class SignSTE(torch.autograd.Function):
    """Binary weight sign.

    Forward:
        w >= 0 -> +1
        w <  0 -> -1

    Backward:
        clipped STE for |w| <= 1
    """
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        mask = (w.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask


class Pow2SignedInt8FakeQuant(nn.Module):
    """Per-tensor symmetric signed INT8 fake quantizer with power-of-two scale.

    q = clamp(round(x / 2^e), -128, 127)
    x_q = q * 2^e

    Deployment interpretation:
      - q is the actual signed INT8 activation consumed by the W1A8 stem
      - 2^e is a binary-point position / shift, not a general multiplier
      - the common scale can later be absorbed into stem BN/affine

    Default exponent -5 gives scale 1/32 and representable dequantized range
    [-4.0, 3.96875], which covers the existing normalized CIFAR-10 inputs.
    """
    def __init__(self, scale_exp=-5, qmin=-128, qmax=127):
        super().__init__()
        self.register_buffer("scale_exp", torch.tensor(int(scale_exp), dtype=torch.int32))
        self.qmin = int(qmin)
        self.qmax = int(qmax)

    @property
    def scale(self):
        return float(2.0 ** int(self.scale_exp.item()))

    def quantize_int(self, x):
        scale = 2.0 ** int(self.scale_exp.item())
        q = round_ste(x / scale)
        return torch.clamp(q, self.qmin, self.qmax)

    def forward(self, x):
        scale = 2.0 ** int(self.scale_exp.item())
        q = self.quantize_int(x)
        return q * scale

    @torch.no_grad()
    def stats(self, x):
        scale = 2.0 ** int(self.scale_exp.item())
        pre = torch.round(x / scale)
        sat_lo = int((pre < self.qmin).sum().item())
        sat_hi = int((pre > self.qmax).sum().item())
        total = int(pre.numel())
        q = torch.clamp(pre, self.qmin, self.qmax)
        return {
            "scale_exp": int(self.scale_exp.item()),
            "scale": scale,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "total": total,
            "sat_low": sat_lo,
            "sat_high": sat_hi,
            "saturation_fraction": (sat_lo + sat_hi) / max(total, 1),
            "q_observed_min": int(q.min().item()),
            "q_observed_max": int(q.max().item()),
            "input_min": float(x.min().item()),
            "input_max": float(x.max().item()),
        }


class W1A8StemConv2d(nn.Conv2d):
    """Training/reference model of a W1A8 first convolution.

    - latent weight is floating-point for optimization
    - forward weight is strictly +/-1
    - input activation is externally fake-quantized to signed INT8
    - convolution in PyTorch is numerical reference only

    Hardware mapping for every product:
        (+1) * q -> +q
        (-1) * q -> -q

    Therefore the stem convolution itself requires add/sub + accumulation,
    not a general multiplier/DSP.
    """
    def forward(self, x):
        bw = SignSTE.apply(self.weight)
        return F.conv2d(
            x, bw, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups,
        )

    @torch.no_grad()
    def binary_weight(self):
        return torch.where(
            self.weight >= 0,
            torch.ones_like(self.weight),
            -torch.ones_like(self.weight),
        )


class R5ResNet20(nn.Module):
    """R5 = R4 multiplier-free binary backbone + W1A8 stem.

    Still unchanged:
      - stem BN
      - R4 residual blocks
      - head BN
      - FP/multi-bit final FC
    """
    def __init__(self, num_classes=10, input_scale_exp=-5):
        super().__init__()
        self.in_planes = 16

        self.input_quant = Pow2SignedInt8FakeQuant(scale_exp=input_scale_exp)
        self.conv1 = W1A8StemConv2d(
            3, 16, 3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 3, 1)
        self.layer2 = self._make_layer(32, 3, 2)
        self.layer3 = self._make_layer(64, 3, 2)

        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(R4BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_quant(x)
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        return self.linear(out)


@torch.no_grad()
def build_r5_from_r4(r4_model, input_scale_exp=-5):
    """Warm-start R5 from a trained R4 checkpoint.

    The FP stem latent weights are copied directly, then binarized only in
    forward via SignSTE. All R4 backbone/head parameters are preserved.
    """
    target = R5ResNet20(
        num_classes=10,
        input_scale_exp=input_scale_exp,
    )

    target.conv1.weight.copy_(r4_model.conv1.weight)
    target.bn1.load_state_dict(r4_model.bn1.state_dict())
    target.bn2.load_state_dict(r4_model.bn2.state_dict())
    target.linear.load_state_dict(r4_model.linear.state_dict())

    for src_layer, dst_layer in zip(
        [r4_model.layer1, r4_model.layer2, r4_model.layer3],
        [target.layer1, target.layer2, target.layer3],
    ):
        for src, dst in zip(src_layer, dst_layer):
            dst.load_state_dict(src.state_dict(), strict=True)

            # R4's alpha tensors are retained only for checkpoint/interface
            # compatibility.  R4 uses alpha_mode="one", so they must remain
            # frozen in R5 as well; the new trainable stem is the only added
            # precision variable in this ablation.
            dst.conv1.alpha.requires_grad_(False)
            dst.conv2.alpha.requires_grad_(False)

    return target


@torch.no_grad()
def stem_weight_stats(model):
    w = model.conv1.weight.detach()
    bw = model.conv1.binary_weight()
    zero_count = int((w == 0).sum().item())
    pos = int((bw > 0).sum().item())
    neg = int((bw < 0).sum().item())

    # Binary projection error after optimal per-filter L1 scale is useful
    # diagnostically, but the scale is NOT used by R5 hardware.
    dims = (1, 2, 3)
    alpha_diag = w.abs().mean(dim=dims, keepdim=True)
    approx = alpha_diag * bw
    mse = ((w - approx) ** 2).mean()

    return {
        "weight_count": int(w.numel()),
        "latent_min": float(w.min().item()),
        "latent_max": float(w.max().item()),
        "latent_mean": float(w.mean().item()),
        "latent_std": float(w.std().item()),
        "binary_positive": pos,
        "binary_negative": neg,
        "latent_zero_count": zero_count,
        "diagnostic_scaled_binary_mse": float(mse.item()),
        "diagnostic_alpha_mean": float(alpha_diag.mean().item()),
    }
