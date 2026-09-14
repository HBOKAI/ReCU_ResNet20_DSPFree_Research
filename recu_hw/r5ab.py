import torch
import torch.nn as nn
import torch.nn.functional as F

from .r4 import R4BasicBlock
from .r5 import Pow2SignedInt8FakeQuant, SignSTE


class ProgressiveScaledW1StemConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_lambda", torch.tensor(0.0, dtype=torch.float32))

    @property
    def progress_lambda(self):
        return float(self._lambda.item())

    @torch.no_grad()
    def set_progress_lambda(self, value):
        self._lambda.fill_(float(max(0.0, min(1.0, value))))

    def per_filter_alpha(self):
        return self.weight.abs().mean(dim=(1, 2, 3), keepdim=True)

    def binary_sign_weight(self):
        return SignSTE.apply(self.weight)

    def scaled_binary_weight(self):
        return self.per_filter_alpha() * self.binary_sign_weight()

    def effective_weight(self):
        lam = self._lambda.to(dtype=self.weight.dtype, device=self.weight.device)
        wb = self.scaled_binary_weight()
        return (1.0 - lam) * self.weight + lam * wb

    def forward(self, x):
        return F.conv2d(
            x, self.effective_weight(), self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


def progressive_lambda(epoch_index, ramp_epochs=30):
    if ramp_epochs <= 0:
        return 1.0
    return float(min(max((epoch_index + 1) / float(ramp_epochs), 0.0), 1.0))


class R5ABResNet20(nn.Module):
    def __init__(self, num_classes=10, input_scale_exp=-5):
        super().__init__()
        self.in_planes = 16
        self.input_quant = Pow2SignedInt8FakeQuant(scale_exp=input_scale_exp)
        self.conv1 = ProgressiveScaledW1StemConv2d(
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
def build_r5ab_from_r4(r4_model, input_scale_exp=-5):
    target = R5ABResNet20(num_classes=10, input_scale_exp=input_scale_exp)
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
            # R4 alpha tensors are retained only for checkpoint/interface
            # compatibility.  R4 uses alpha_mode="one", so they must stay
            # frozen; R5AB's only new trainable precision variable is the
            # progressive W1A8 stem latent weight.
            dst.conv1.alpha.requires_grad_(False)
            dst.conv2.alpha.requires_grad_(False)

    target.conv1.set_progress_lambda(0.0)
    return target


@torch.no_grad()
def stem_stats(model):
    w = model.conv1.weight.detach()
    bw = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    alpha = model.conv1.per_filter_alpha().detach().flatten()
    return {
        "lambda": model.conv1.progress_lambda,
        "latent_min": float(w.min()),
        "latent_max": float(w.max()),
        "latent_mean": float(w.mean()),
        "latent_std": float(w.std()),
        "binary_positive": int((bw > 0).sum()),
        "binary_negative": int((bw < 0).sum()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "alpha_mean": float(alpha.mean()),
        "alpha_median": float(alpha.median()),
    }


@torch.no_grad()
def folded_stem_bn_params(model):
    bn = model.bn1
    alpha = model.conv1.per_filter_alpha().flatten()
    denom = torch.sqrt(bn.running_var + bn.eps)
    k = bn.weight * alpha / denom
    b = bn.bias - bn.weight * bn.running_mean / denom
    return k, b
