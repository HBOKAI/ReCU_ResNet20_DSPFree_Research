import torch
import torch.nn as nn
import torch.nn.functional as F

from .r7 import R7ResNet20
from .r6 import signed_pow2_ste, is_power_of_two_tensor


class BinaryLinearSTE(nn.Linear):
    def effective_weight(self):
        return (self.weight.sign() - self.weight).detach() + self.weight

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class Pow2LinearSTE(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, min_exp=None, max_exp=None):
        super().__init__(in_features, out_features, bias=bias)
        self.min_exp = min_exp
        self.max_exp = max_exp

    def effective_weight(self):
        return signed_pow2_ste(self.weight, self.min_exp, self.max_exp)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def stats(self):
        wq = self.effective_weight().detach()
        exps = torch.round(torch.log2(wq.abs())).to(torch.int64)
        u, c = torch.unique(exps.cpu(), return_counts=True)
        hist = {int(a.item()): int(b.item()) for a, b in zip(u, c)}
        min_exp = int(exps.min().item())
        max_exp = int(exps.max().item())
        max_abs_exp = max(abs(min_exp), abs(max_exp))
        exponent_bits = 1 if max_abs_exp == 0 else max_abs_exp.bit_length() + 1
        bias_bits = int(self.bias.numel() * 32) if self.bias is not None else 0
        return {
            "weights": int(wq.numel()),
            "positive": int((wq > 0).sum()),
            "negative": int((wq < 0).sum()),
            "min_exponent": min_exp,
            "max_exponent": max_exp,
            "exponent_bit_width_signed": exponent_bits,
            "unique_exponents": sorted(hist),
            "exponent_histogram": hist,
            "is_exact_signed_pow2": is_power_of_two_tensor(wq),
            "bias_present": self.bias is not None,
            "weight_storage_bits": int(wq.numel() * (1 + exponent_bits)),
            "bias_storage_bits_fp32": bias_bits,
            "classifier_storage_bits_estimate": int(wq.numel() * (1 + exponent_bits) + bias_bits),
        }


@torch.no_grad()
def build_r8a_from_r7(source_model):
    source_model.eval()
    target = R7ResNet20(
        num_classes=source_model.linear.out_features,
        resolution=8,
        stem_min_exp=source_model.stem_affine.min_exp,
        stem_max_exp=source_model.stem_affine.max_exp,
        head_min_exp=source_model.head_affine.min_exp,
        head_max_exp=source_model.head_affine.max_exp,
    )
    target.load_state_dict(source_model.state_dict(), strict=True)

    fc = BinaryLinearSTE(64, source_model.linear.out_features, bias=True)
    fc.weight.copy_(source_model.linear.weight)
    fc.bias.copy_(source_model.linear.bias)
    target.linear = fc
    return target


@torch.no_grad()
def build_r8b_from_r7(source_model, min_exp=None, max_exp=None):
    source_model.eval()
    target = R7ResNet20(
        num_classes=source_model.linear.out_features,
        resolution=8,
        stem_min_exp=source_model.stem_affine.min_exp,
        stem_max_exp=source_model.stem_affine.max_exp,
        head_min_exp=source_model.head_affine.min_exp,
        head_max_exp=source_model.head_affine.max_exp,
    )
    target.load_state_dict(source_model.state_dict(), strict=True)

    fc = Pow2LinearSTE(
        64, source_model.linear.out_features, bias=True,
        min_exp=min_exp, max_exp=max_exp
    )
    fc.weight.copy_(source_model.linear.weight)
    fc.bias.copy_(source_model.linear.bias)
    target.linear = fc
    return target


@torch.no_grad()
def binary_fc_stats(model):
    w = model.linear.effective_weight().detach()
    u = sorted(float(v) for v in torch.unique(w).cpu().tolist())
    bias_bits = int(model.linear.bias.numel() * 32) if model.linear.bias is not None else 0
    return {
        "weights": int(w.numel()),
        "unique_values": u,
        "positive": int((w > 0).sum()),
        "negative": int((w < 0).sum()),
        "is_strict_binary": u == [-1.0, 1.0],
        "bias_present": model.linear.bias is not None,
        "weight_storage_bits": int(w.numel()),
        "bias_storage_bits_fp32": bias_bits,
        "classifier_storage_bits_estimate": int(w.numel() + bias_bits),
    }
