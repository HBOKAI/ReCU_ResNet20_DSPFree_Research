import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_affine import FusedAffine2d, fold_recu_alpha_bn
from .layers import ReCUBinaryConv2d
from .model import option_a_shortcut
from .qrprelu import QuantizedRPReLU


class R4BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = ReCUBinaryConv2d(
            in_planes, planes, 3,
            stride=stride, padding=1, bias=False,
            alpha_mode="one",
        )
        self.aff1 = FusedAffine2d(planes, mode="pow2")

        self.conv2 = ReCUBinaryConv2d(
            planes, planes, 3,
            stride=1, padding=1, bias=False,
            alpha_mode="one",
        )
        self.aff2 = FusedAffine2d(planes, mode="pow2")

        self.post_act = QuantizedRPReLU(planes)
        self.shortcut = option_a_shortcut(in_planes, planes, stride)

    def forward(self, x):
        out = self.aff1(self.conv1(x))
        out = out + self.shortcut(x)
        x1 = out

        out = F.hardtanh(out)
        out = self.aff2(self.conv2(out))
        out = out + x1
        return self.post_act(out)


class R4ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
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
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        return self.linear(out)


@torch.no_grad()
def build_r4_from_r2(r2_model):
    target = R4ResNet20()

    target.conv1.load_state_dict(r2_model.conv1.state_dict())
    target.bn1.load_state_dict(r2_model.bn1.state_dict())
    target.bn2.load_state_dict(r2_model.bn2.state_dict())
    target.linear.load_state_dict(r2_model.linear.state_dict())

    for src_layer, dst_layer in zip(
        [r2_model.layer1, r2_model.layer2, r2_model.layer3],
        [target.layer1, target.layer2, target.layer3],
    ):
        for src, dst in zip(src_layer, dst_layer):
            dst.conv1.weight.copy_(src.conv1.weight)
            dst.conv2.weight.copy_(src.conv2.weight)

            dst.conv1.alpha.copy_(src.conv1.alpha)
            dst.conv2.alpha.copy_(src.conv2.alpha)
            dst.conv1.alpha.requires_grad_(False)
            dst.conv2.alpha.requires_grad_(False)

            k1, b1 = fold_recu_alpha_bn(src.conv1, src.bn1)
            k2, b2 = fold_recu_alpha_bn(src.conv2, src.bn2)
            dst.aff1.init_from_kb(k1, b1)
            dst.aff2.init_from_kb(k2, b2)

            dst.post_act.a.copy_(src.post_act.a)
            dst.post_act.xi1.copy_(src.post_act.xi1)
            dst.post_act.xi2.copy_(src.post_act.xi2)

    return target


def iter_r4_affines(model):
    for m in model.modules():
        if isinstance(m, FusedAffine2d):
            yield m


@torch.no_grad()
def r4_exponent_stats(model):
    vals = []
    for m in iter_r4_affines(model):
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
def r4_k_quantization_stats(model):
    abs_errs, rel_errs, floats = [], [], []

    for m in iter_r4_affines(model):
        float_k = m.sign_k * torch.pow(
            2.0,
            torch.clamp(m.log2_abs_k, m.exp_min, m.exp_max),
        )
        quant_k = m.effective_k()
        ae = (quant_k - float_k).abs()
        re = ae / float_k.abs().clamp_min(1e-12)

        abs_errs.append(ae.flatten().cpu())
        rel_errs.append(re.flatten().cpu())
        floats.append(float_k.flatten().cpu())

    a = torch.cat(abs_errs)
    r = torch.cat(rel_errs)
    f = torch.cat(floats)

    return {
        "count": int(f.numel()),
        "float_k_min": float(f.min()),
        "float_k_max": float(f.max()),
        "float_abs_k_mean": float(f.abs().mean()),
        "float_abs_k_median": float(f.abs().median()),
        "negative_k_count": int((f < 0).sum()),
        "positive_k_count": int((f >= 0).sum()),
        "mean_abs_error": float(a.mean()),
        "max_abs_error": float(a.max()),
        "mean_relative_error": float(r.mean()),
        "max_relative_error": float(r.max()),
    }
