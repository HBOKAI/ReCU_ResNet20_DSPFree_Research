import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_affine import FusedAffine2d, fold_recu_alpha_bn
from .layers import ReCUBinaryConv2d, iter_recu_binary_convs
from .model import ReCUResNet20, option_a_shortcut
from .qrprelu import QuantizedRPReLU


def freeze_inactive_alpha(model):
    for m in iter_recu_binary_convs(model):
        if getattr(m, "alpha_mode", None) == "one":
            m.alpha.requires_grad_(False)


def build_r1_from_official(official):
    """R1: remove alpha, preserve official PReLU/BN/topology."""
    target = ReCUResNet20(
        activation_mode="prelu",
        alpha_mode="one",
    )
    target.load_state_dict(official.state_dict(), strict=True)
    freeze_inactive_alpha(target)
    return target


@torch.no_grad()
def _copy_common_state_excluding_prelu(official, target):
    src = official.state_dict()
    dst = target.state_dict()

    copied = {}
    for k, v in src.items():
        # Official PReLU keys look like layerX.Y.post_act.weight.
        if ".post_act.weight" in k:
            continue
        if k in dst and dst[k].shape == v.shape:
            copied[k] = v

    missing, unexpected = target.load_state_dict(copied, strict=False)
    # QRPReLU parameters are expected to be missing.
    bad_unexpected = [k for k in unexpected]
    if bad_unexpected:
        raise RuntimeError(f"Unexpected keys during R2 conversion: {bad_unexpected}")
    return missing


@torch.no_grad()
def build_r2_from_official(official, slope_floor=2.0 ** -16):
    """R2: replace each official PReLU by QRPReLU and warm-start its slope.

    Mapping:
        QRPReLU a_i = log2(PReLU_slope_i)
        xi1_i = 0
        xi2_i = 0
    """
    target = ReCUResNet20(
        activation_mode="qrprelu",
        alpha_mode="float",
    )
    _copy_common_state_excluding_prelu(official, target)

    src_blocks = []
    dst_blocks = []
    for sm, dm in zip(official.modules(), target.modules()):
        if isinstance(sm, nn.PReLU):
            src_blocks.append(sm)
        if isinstance(dm, QuantizedRPReLU):
            dst_blocks.append(dm)

    if len(src_blocks) != len(dst_blocks):
        raise RuntimeError("PReLU/QRPReLU block count mismatch")

    negative_or_zero = 0
    for src_prelu, dst_q in zip(src_blocks, dst_blocks):
        p = src_prelu.weight.detach().clone()
        negative_or_zero += int((p <= 0).sum().item())
        # QRPReLU hardware slope is positive power-of-two.
        p = p.clamp_min(slope_floor)
        dst_q.a.copy_(torch.log2(p))
        dst_q.xi1.zero_()
        dst_q.xi2.zero_()

    return target, negative_or_zero


class ReCUFusedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, affine_mode="pow2"):
        super().__init__()
        self.conv1 = ReCUBinaryConv2d(
            in_planes, planes, 3,
            stride=stride, padding=1, bias=False,
            alpha_mode="one",
        )
        self.aff1 = FusedAffine2d(planes, mode=affine_mode)

        self.conv2 = ReCUBinaryConv2d(
            planes, planes, 3,
            stride=1, padding=1, bias=False,
            alpha_mode="one",
        )
        self.aff2 = FusedAffine2d(planes, mode=affine_mode)

        self.post_act = nn.PReLU(planes)
        self.shortcut = option_a_shortcut(in_planes, planes, stride)

    def forward(self, x):
        out = self.aff1(self.conv1(x))
        out = out + self.shortcut(x)
        x1 = out

        out = F.hardtanh(out)
        out = self.aff2(self.conv2(out))
        out = out + x1
        return self.post_act(out)


class ReCUFusedResNet20(nn.Module):
    """Deployment-oriented ReCU where alpha+BN after every BConv is folded."""

    def __init__(self, num_classes=10, affine_mode="pow2"):
        super().__init__()
        self.in_planes = 16
        self.affine_mode = affine_mode

        # Stem is intentionally unchanged in R3.
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 3, 1)
        self.layer2 = self._make_layer(32, 3, 2)
        self.layer3 = self._make_layer(64, 3, 2)

        # Head intentionally unchanged in R3.
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                ReCUFusedBasicBlock(
                    self.in_planes, planes, s,
                    affine_mode=self.affine_mode,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        return self.linear(out)


@torch.no_grad()
def build_r3_from_official(official, affine_mode="pow2"):
    """Fold each official BConv alpha + following BN into one affine.

    affine_mode="float" is used as an exactness audit.
    affine_mode="pow2" is R3 hardware-aware QAT.
    """
    target = ReCUFusedResNet20(affine_mode=affine_mode)

    # Stem and head.
    target.conv1.load_state_dict(official.conv1.state_dict())
    target.bn1.load_state_dict(official.bn1.state_dict())
    target.bn2.load_state_dict(official.bn2.state_dict())
    target.linear.load_state_dict(official.linear.state_dict())

    for src_layer, dst_layer in zip(
        [official.layer1, official.layer2, official.layer3],
        [target.layer1, target.layer2, target.layer3],
    ):
        for src_block, dst_block in zip(src_layer, dst_layer):
            # Binary weights.
            dst_block.conv1.weight.copy_(src_block.conv1.weight)
            dst_block.conv2.weight.copy_(src_block.conv2.weight)
            if dst_block.conv1.bias is not None and src_block.conv1.bias is not None:
                dst_block.conv1.bias.copy_(src_block.conv1.bias)
            if dst_block.conv2.bias is not None and src_block.conv2.bias is not None:
                dst_block.conv2.bias.copy_(src_block.conv2.bias)

            dst_block.conv1.alpha.copy_(src_block.conv1.alpha)
            dst_block.conv2.alpha.copy_(src_block.conv2.alpha)
            dst_block.conv1.alpha.requires_grad_(False)
            dst_block.conv2.alpha.requires_grad_(False)

            k1, b1 = fold_recu_alpha_bn(src_block.conv1, src_block.bn1)
            k2, b2 = fold_recu_alpha_bn(src_block.conv2, src_block.bn2)
            dst_block.aff1.init_from_kb(k1, b1)
            dst_block.aff2.init_from_kb(k2, b2)

            dst_block.post_act.load_state_dict(src_block.post_act.state_dict())

    return target


def iter_fused_affines(model):
    for m in model.modules():
        if isinstance(m, FusedAffine2d):
            yield m


@torch.no_grad()
def fused_exponent_stats(model):
    vals = []
    for m in iter_fused_affines(model):
        vals.extend(m.integer_exponents().cpu().tolist())
    hist = {}
    for v in vals:
        hist[int(v)] = hist.get(int(v), 0) + 1
    return {
        "count": len(vals),
        "histogram": dict(sorted(hist.items())),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


@torch.no_grad()
def verify_float_fold_equivalence(official, device="cpu", batches=3, batch_size=4):
    official = official.to(device).eval()
    folded = build_r3_from_official(official, affine_mode="float").to(device).eval()

    max_abs = 0.0
    max_rel = 0.0
    for _ in range(batches):
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        a = official(x)
        b = folded(x)
        diff = (a - b).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = a.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))

    return {
        "max_abs_logit_error": max_abs,
        "max_rel_logit_error": max_rel,
    }
