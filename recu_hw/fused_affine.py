import torch
import torch.nn as nn

from .layers import round_ste


class FusedAffine2d(nn.Module):
    """Per-channel y = K*x + B after folding ReCU alpha + BatchNorm.

    mode="float":
        deployment-equivalent floating affine, used to verify that folding
        itself is exact before quantization.

    mode="pow2":
        K is constrained to sign(K0) * 2^round(log2|K|).
        Deployment needs sign + shift + add, not a general multiplier.

    The sign of K is stored as a fixed buffer. log2_abs_k and bias are
    trainable during hardware-aware fine-tuning.
    """

    def __init__(self, channels, mode="pow2", exp_min=-16, exp_max=8):
        super().__init__()
        self.channels = int(channels)
        self.mode = mode
        self.exp_min = int(exp_min)
        self.exp_max = int(exp_max)

        self.log2_abs_k = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("sign_k", torch.ones(channels))

    @torch.no_grad()
    def init_from_kb(self, k, b):
        k = k.detach().flatten()
        b = b.detach().flatten()
        if k.numel() != self.channels or b.numel() != self.channels:
            raise ValueError("K/B channel count mismatch")

        sign = torch.where(k >= 0, torch.ones_like(k), -torch.ones_like(k))
        mag = k.abs().clamp_min(2.0 ** self.exp_min)
        self.sign_k.copy_(sign)
        self.log2_abs_k.copy_(torch.log2(mag))
        self.bias.copy_(b)

    def effective_k(self):
        if self.mode == "float":
            exp = self.log2_abs_k
        elif self.mode == "pow2":
            exp = round_ste(self.log2_abs_k)
        else:
            raise ValueError(f"Unknown affine mode: {self.mode}")

        exp = torch.clamp(exp, self.exp_min, self.exp_max)
        return self.sign_k * torch.pow(2.0, exp)

    def forward(self, x):
        k = self.effective_k().view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        return x * k + b

    @torch.no_grad()
    def integer_exponents(self):
        return torch.clamp(
            torch.round(self.log2_abs_k),
            self.exp_min,
            self.exp_max,
        ).to(torch.int32)


@torch.no_grad()
def fold_recu_alpha_bn(binary_conv, bn):
    """Fold y = BN(alpha * S) into y = K*S + B using BN running stats."""
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("Expected BatchNorm2d")

    alpha = binary_conv.alpha.detach().flatten()
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()

    inv_std = torch.rsqrt(var + bn.eps)
    k = gamma * alpha * inv_std
    b = beta - gamma * mean * inv_std
    return k, b
