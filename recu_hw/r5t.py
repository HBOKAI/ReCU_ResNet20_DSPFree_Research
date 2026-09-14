import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .r4 import R4BasicBlock
from .r5 import SignSTE


class ThermometerEncoder(nn.Module):
    """FracBNN-style thermometer encoding.

    Input: float RGB tensor in [0,1], shape [N,3,H,W].
    For resolution R, L=ceil(255/R), n=round(p/R), p=round(255*x).
    Output is bipolar {-1,+1} with 3*L channels.
    """
    def __init__(self, resolution=8):
        super().__init__()
        self.resolution = int(resolution)
        self.length = int(math.ceil(255.0 / self.resolution))

    def forward(self, x):
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError("ThermometerEncoder expects NCHW RGB input")
        p = torch.round(torch.clamp(x, 0.0, 1.0) * 255.0)
        n = torch.round(p / float(self.resolution))
        n = torch.clamp(n, 0, self.length).to(torch.long)
        idx = torch.arange(self.length, device=x.device).view(1, 1, self.length, 1, 1)
        threshold = (self.length - n).unsqueeze(2)
        bits = (idx >= threshold).to(x.dtype)
        bipolar = bits.mul(2.0).sub(1.0)
        N, C, L, H, W = bipolar.shape
        return bipolar.reshape(N, C * L, H, W)

    @torch.no_grad()
    def encode_uint8_reference(self, p):
        p = torch.as_tensor(p, dtype=torch.float32)
        n = torch.round(p / float(self.resolution))
        n = torch.clamp(n, 0, self.length).to(torch.long)
        idx = torch.arange(self.length)
        return (idx >= (self.length - n)).to(torch.int64)


class ThermometerStemConv2d(nn.Conv2d):
    """T1 uses floating latent weights; T2 uses sign(W) with STE."""
    def __init__(self, *args, binary_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_weight = bool(binary_weight)

    def set_binary_weight(self, enabled):
        self.binary_weight = bool(enabled)

    def effective_weight(self):
        return SignSTE.apply(self.weight) if self.binary_weight else self.weight

    def forward(self, x):
        return F.conv2d(
            x, self.effective_weight(), self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

    @torch.no_grad()
    def sign_stats(self):
        bw = torch.where(self.weight >= 0, torch.ones_like(self.weight), -torch.ones_like(self.weight))
        return {
            "positive": int((bw > 0).sum().item()),
            "negative": int((bw < 0).sum().item()),
            "zero_latent": int((self.weight == 0).sum().item()),
        }


class R5TResNet20(nn.Module):
    def __init__(self, num_classes=10, resolution=8, binary_stem=False):
        super().__init__()
        self.in_planes = 16
        self.thermo = ThermometerEncoder(resolution=resolution)
        cin = 3 * self.thermo.length
        self.conv1 = ThermometerStemConv2d(
            cin, 16, 3, stride=1, padding=1, bias=False,
            binary_weight=binary_stem,
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
        x = self.thermo(x)
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        return self.linear(out)


@torch.no_grad()
def project_r4_stem_to_thermometer(
    r4_weight,
    resolution=8,
    cifar_std=(0.2470, 0.2435, 0.2616),
):
    """Engineering initialization (not claimed as FracBNN method).

    Match the p-dependent slope of normalized-RGB stem:
      x_norm = p/(255*std) - mean/std
    with thermometer sum approx 2*p/R - L.

    Repeated channel weight:
      W_rep = W_orig * R / (2*255*std)

    Constant offset mismatch is left for stem BN to adapt.
    """
    if r4_weight.shape[1] != 3:
        raise ValueError("Expected RGB stem weight with Cin=3")
    L = int(math.ceil(255.0 / resolution))
    std = torch.tensor(cifar_std, dtype=r4_weight.dtype, device=r4_weight.device).view(1, 3, 1, 1)
    per_rgb = r4_weight * (float(resolution) / (2.0 * 255.0)) / std
    return per_rgb.repeat_interleave(L, dim=1)


@torch.no_grad()
def build_r5t_from_r4(
    r4_model,
    resolution=8,
    binary_stem=False,
    cifar_std=(0.2470, 0.2435, 0.2616),
):
    target = R5TResNet20(num_classes=10, resolution=resolution, binary_stem=binary_stem)
    target.conv1.weight.copy_(project_r4_stem_to_thermometer(
        r4_model.conv1.weight, resolution=resolution, cifar_std=cifar_std
    ))
    target.bn1.load_state_dict(r4_model.bn1.state_dict())
    target.bn2.load_state_dict(r4_model.bn2.state_dict())
    target.linear.load_state_dict(r4_model.linear.state_dict())
    for src_layer, dst_layer in zip(
        [r4_model.layer1, r4_model.layer2, r4_model.layer3],
        [target.layer1, target.layer2, target.layer3],
    ):
        for src, dst in zip(src_layer, dst_layer):
            dst.load_state_dict(src.state_dict(), strict=True)
            # R4 alpha tensors are retained only for checkpoint/interface
            # compatibility.  R4 uses alpha_mode="one", so freeze them to
            # keep the R4 backbone unchanged during T1/T2.
            dst.conv1.alpha.requires_grad_(False)
            dst.conv2.alpha.requires_grad_(False)
    return target
