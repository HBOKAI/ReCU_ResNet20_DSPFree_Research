import torch
import torch.nn as nn
import torch.nn.functional as F

from .r4 import R4BasicBlock
from .r5t import ThermometerEncoder, ThermometerStemConv2d
from .r6 import (
    StemFusedPow2Affine2d,
    signed_pow2_ste,
    is_power_of_two_tensor,
    parameter_freeze_report,
)


class HeadFusedPow2Affine1d(nn.Module):
    def __init__(self, channels=64, quantize_k=True, min_exp=None, max_exp=None):
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
        return x * self.effective_k().view(1,-1) + self.bias.view(1,-1)

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


class R7ResNet20(nn.Module):
    def __init__(self, num_classes=10, resolution=8,
                 stem_min_exp=None, stem_max_exp=None,
                 head_min_exp=None, head_max_exp=None):
        super().__init__()
        self.in_planes = 16
        self.thermo = ThermometerEncoder(resolution=resolution)
        cin = 3*self.thermo.length
        self.conv1 = ThermometerStemConv2d(
            cin,16,3,stride=1,padding=1,bias=False,binary_weight=True
        )
        self.stem_affine = StemFusedPow2Affine2d(
            16,quantize_k=True,min_exp=stem_min_exp,max_exp=stem_max_exp
        )
        self.layer1 = self._make_layer(16,3,1)
        self.layer2 = self._make_layer(32,3,2)
        self.layer3 = self._make_layer(64,3,2)
        self.head_affine = HeadFusedPow2Affine1d(
            64,quantize_k=True,min_exp=head_min_exp,max_exp=head_max_exp
        )
        self.linear = nn.Linear(64,num_classes)
        # R4 compatibility alpha tensors are not part of the R7 experiment.
        # Reapply the policy in the constructor because requires_grad is not
        # serialized in state_dict and fresh checkpoint reloads must be safe.
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
        out=self.head_affine(out)
        return self.linear(out)


@torch.no_grad()
def build_r7_from_r6(source_model, resolution=8, head_min_exp=None, head_max_exp=None):
    source_model.eval()
    target=R7ResNet20(
        num_classes=source_model.linear.out_features,
        resolution=resolution,
        stem_min_exp=source_model.stem_affine.min_exp,
        stem_max_exp=source_model.stem_affine.max_exp,
        head_min_exp=head_min_exp,
        head_max_exp=head_max_exp,
    )
    target.conv1.weight.copy_(source_model.conv1.weight)
    target.conv1.set_binary_weight(True)
    target.stem_affine.load_state_dict(source_model.stem_affine.state_dict(), strict=True)
    target.stem_affine.quantize_k=True

    for src_layer,dst_layer in zip(
        [source_model.layer1,source_model.layer2,source_model.layer3],
        [target.layer1,target.layer2,target.layer3],
    ):
        for src,dst in zip(src_layer,dst_layer):
            dst.load_state_dict(src.state_dict(), strict=True)

    target.head_affine.load_from_bn(source_model.bn2)
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
def exact_head_fold_max_abs_error(source_model, r7_model, x):
    source_model.eval()
    r7_model.eval()

    tx=source_model.thermo(x)
    s=source_model.conv1(tx)
    out=F.hardtanh(source_model.stem_affine(s), inplace=False)
    out=source_model.layer1(out)
    out=source_model.layer2(out)
    out=source_model.layer3(out)
    out=F.avg_pool2d(out,out.size(3))
    feat=out.view(out.size(0),-1)

    y0=source_model.bn2(feat)

    old=r7_model.head_affine.quantize_k
    r7_model.head_affine.quantize_k=False
    y1=r7_model.head_affine(feat)
    r7_model.head_affine.quantize_k=old

    return float((y0-y1).abs().max().item())
