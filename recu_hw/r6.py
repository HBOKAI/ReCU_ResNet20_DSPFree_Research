import torch
import torch.nn as nn
import torch.nn.functional as F

from .r4 import R4BasicBlock
from .r5t import ThermometerEncoder, ThermometerStemConv2d
from .layers import round_ste


def signed_pow2_ste(x, min_exp=None, max_exp=None):
    """Signed power-of-two quantization with the repository's RoundSTE.

    This intentionally matches the existing R4 ``FusedAffine2d`` semantics:
    round(log2(abs(x))) with optional post-round exponent bounds, while the
    sign remains explicit.  ``min_exp`` and ``max_exp`` are left optional for
    R6 because this experiment must measure the unconstrained source range.
    """
    tiny = torch.finfo(x.dtype).tiny
    exp = round_ste(torch.log2(x.abs().clamp_min(tiny)))
    if min_exp is not None:
        exp = torch.clamp_min(exp, float(min_exp))
    if max_exp is not None:
        exp = torch.clamp_max(exp, float(max_exp))
    sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return sign * torch.pow(torch.tensor(2.0, dtype=x.dtype, device=x.device), exp)


def is_power_of_two_tensor(x, atol=1e-6):
    ax = x.detach().abs()
    if torch.any(ax == 0):
        return False
    e = torch.round(torch.log2(ax))
    ref = torch.pow(torch.tensor(2.0, dtype=ax.dtype, device=ax.device), e)
    return bool(torch.allclose(ax, ref, atol=atol, rtol=0.0))


class StemFusedPow2Affine2d(nn.Module):
    def __init__(self, channels, quantize_k=True, min_exp=None, max_exp=None):
        super().__init__()
        self.channels = int(channels)
        self.k_latent = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.quantize_k = bool(quantize_k)
        self.min_exp = min_exp
        self.max_exp = max_exp

    @torch.no_grad()
    def load_from_bn(self, bn):
        denom = torch.sqrt(bn.running_var.detach() + bn.eps)
        gamma = bn.weight.detach()
        beta = bn.bias.detach()
        mu = bn.running_mean.detach()
        k = gamma / denom
        b = beta - gamma * mu / denom
        self.k_latent.copy_(k)
        self.bias.copy_(b)
        return k.clone(), b.clone()

    def effective_k(self):
        if self.quantize_k:
            return signed_pow2_ste(self.k_latent, self.min_exp, self.max_exp)
        return self.k_latent

    def forward(self, x):
        k = self.effective_k().view(1,-1,1,1)
        b = self.bias.view(1,-1,1,1)
        return x*k + b

    @torch.no_grad()
    def stats(self):
        kf = self.k_latent.detach()
        kq = signed_pow2_ste(kf, self.min_exp, self.max_exp).detach()
        exps = torch.round(torch.log2(kq.abs())).to(torch.int64)
        u,c = torch.unique(exps.cpu(), return_counts=True)
        hist = {int(a.item()): int(b.item()) for a,b in zip(u,c)}
        return {
            "channels": self.channels,
            "latent_k_min": float(kf.min()),
            "latent_k_max": float(kf.max()),
            "latent_k_abs_min": float(kf.abs().min()),
            "latent_k_abs_max": float(kf.abs().max()),
            "latent_k_abs_mean": float(kf.abs().mean()),
            "quant_k_min": float(kq.min()),
            "quant_k_max": float(kq.max()),
            "negative_k_count": int((kq<0).sum()),
            "positive_k_count": int((kq>0).sum()),
            "unique_exponents": sorted(hist),
            "exponent_histogram": hist,
            "bias_min": float(self.bias.detach().min()),
            "bias_max": float(self.bias.detach().max()),
            "bias_mean": float(self.bias.detach().mean()),
            "is_exact_signed_pow2": is_power_of_two_tensor(kq),
        }


class R6ResNet20(nn.Module):
    def __init__(self, num_classes=10, resolution=8, min_exp=None, max_exp=None):
        super().__init__()
        self.in_planes = 16
        self.thermo = ThermometerEncoder(resolution=resolution)
        cin = 3*self.thermo.length
        self.conv1 = ThermometerStemConv2d(
            cin,16,3,stride=1,padding=1,bias=False,binary_weight=True
        )
        self.stem_affine = StemFusedPow2Affine2d(
            16, quantize_k=True, min_exp=min_exp, max_exp=max_exp
        )
        self.layer1 = self._make_layer(16,3,1)
        self.layer2 = self._make_layer(32,3,2)
        self.layer3 = self._make_layer(64,3,2)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64,num_classes)
        # Compatibility alpha tensors belong to the inherited R4 backbone;
        # they are never part of the R6 optimization surface, including when
        # a fresh model is instantiated for checkpoint reload.
        for module in self.modules():
            if hasattr(module, "alpha"):
                module.alpha.requires_grad_(False)

    def _make_layer(self, planes, blocks, stride):
        layers=[]
        for s in [stride]+[1]*(blocks-1):
            layers.append(R4BasicBlock(self.in_planes,planes,s))
            self.in_planes=planes
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.thermo(x)
        s=self.conv1(x)
        out=F.hardtanh(self.stem_affine(s), inplace=False)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.avg_pool2d(out,out.size(3))
        out=out.view(out.size(0),-1)
        out=self.bn2(out)
        return self.linear(out)


@torch.no_grad()
def build_r6_from_r5t(source_model, resolution=8, min_exp=None, max_exp=None):
    source_model.eval()
    target=R6ResNet20(
        num_classes=source_model.linear.out_features,
        resolution=resolution,
        min_exp=min_exp,
        max_exp=max_exp,
    )
    target.conv1.weight.copy_(source_model.conv1.weight)
    target.conv1.set_binary_weight(True)
    target.stem_affine.load_from_bn(source_model.bn1)

    for src_layer,dst_layer in zip(
        [source_model.layer1,source_model.layer2,source_model.layer3],
        [target.layer1,target.layer2,target.layer3],
    ):
        for src,dst in zip(src_layer,dst_layer):
            dst.load_state_dict(src.state_dict(), strict=True)

    target.bn2.load_state_dict(source_model.bn2.state_dict())
    target.linear.load_state_dict(source_model.linear.state_dict())

    src_req={n:p.requires_grad for n,p in source_model.named_parameters()}
    for n,p in target.named_parameters():
        if n in src_req and not n.endswith(".alpha"):
            p.requires_grad_(src_req[n])
    for module in target.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    return target


@torch.no_grad()
def exact_fold_max_abs_error(source_model, r6_model, x):
    source_model.eval()
    r6_model.eval()
    tx=source_model.thermo(x)
    s0=source_model.conv1(tx)
    y0=source_model.bn1(s0)

    old=r6_model.stem_affine.quantize_k
    r6_model.stem_affine.quantize_k=False
    tx1=r6_model.thermo(x)
    s1=r6_model.conv1(tx1)
    y1=r6_model.stem_affine(s1)
    r6_model.stem_affine.quantize_k=old
    return float((y0-y1).abs().max().item())


@torch.no_grad()
def stem_binary_weight_stats(model):
    w=model.conv1.effective_weight().detach()
    unique=sorted(float(v) for v in torch.unique(w).cpu().tolist())
    return {
        "effective_unique_values": unique,
        "positive": int((w>0).sum()),
        "negative": int((w<0).sum()),
        "is_strict_binary": unique == [-1.0,1.0],
        "conv_bias_is_none": model.conv1.bias is None,
    }


@torch.no_grad()
def parameter_freeze_report(model):
    """Return reproducible trainable/frozen parameter accounting."""
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
    categories = {}
    for name in frozen_names:
        category = "r4_compatibility_alpha" if name.endswith(".alpha") else "other_frozen"
        categories[category] = categories.get(category, 0) + 1
    return {
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "frozen_params": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
        "frozen_parameter_names": frozen_names,
        "frozen_parameter_categories": categories,
    }
