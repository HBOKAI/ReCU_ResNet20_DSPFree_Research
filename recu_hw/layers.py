import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightBinaryQuantize(torch.autograd.Function):
    """Official ReCU-style weight sign with identity STE."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class ActivationBinaryQuantize(torch.autograd.Function):
    """Official ReCU activation sign with triangular surrogate gradient."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = (2.0 - torch.abs(2.0 * x)).clamp(min=0.0)
        return grad * grad_output


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def round_ste(x):
    return RoundSTE.apply(x)


class ReCUBinaryConv2d(nn.Conv2d):
    """Modern-PyTorch port of the official CIFAR ReCU BinarizeConv2d.

    Official behavior reproduced:
      1) per-filter mean removal
      2) per-filter variance normalization
      3) tau-dependent symmetric clipping
      4) sign-binarized weight
      5) training-only activation variance normalization
      6) sign-binarized activation
      7) binary convolution represented with float +/-1 tensors
      8) learnable per-output-channel alpha scaling

    Hardware note:
      - steps 1-5 are training/export concerns
      - final W/A are 1-bit
      - alpha is a multiplier candidate unless folded/quantized later
    """
    def __init__(self, *args, alpha_mode="float", **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1))
        self.register_buffer("tau", torch.tensor(1.0))
        self.alpha_mode = alpha_mode

    def set_tau(self, tau):
        if not torch.is_tensor(tau):
            tau = torch.tensor(float(tau), device=self.tau.device)
        self.tau.copy_(tau.detach().to(device=self.tau.device, dtype=self.tau.dtype))

    def _effective_alpha(self):
        if self.alpha_mode == "float":
            return self.alpha
        if self.alpha_mode == "one":
            # R1 / R3 mode: alpha is removed from the inference function.
            # Keep the parameter in state_dict only for checkpoint compatibility.
            return torch.ones_like(self.alpha)
        if self.alpha_mode == "pow2":
            # Legacy H2 mode: quantize alpha itself.
            sign = torch.where(self.alpha >= 0, torch.ones_like(self.alpha), -torch.ones_like(self.alpha))
            mag = self.alpha.abs().clamp_min(2.0 ** -16)
            log2mag = torch.log2(mag)
            qexp = round_ste(log2mag)
            return sign * torch.pow(2.0, qexp)
        raise ValueError(f"Unknown alpha_mode: {self.alpha_mode}")

    def forward(self, x):
        w = self.weight

        w0 = w - w.mean(dim=(1, 2, 3), keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var(dim=(1, 2, 3), keepdim=True) + 1e-5) / 2.0 / math.sqrt(2.0))

        ew = torch.mean(torch.abs(w1))
        tau = self.tau.clamp(max=0.999999)
        q_tau = -ew * torch.log(2.0 - 2.0 * tau)
        # Official code converts Q_tau to a Python scalar. Keep the same
        # scalar clipping semantics while avoiding accidental gradient flow.
        q_tau = q_tau.detach()
        w2 = torch.clamp(w1, -q_tau, q_tau)

        if self.training:
            a0 = x / torch.sqrt(x.var(dim=(1, 2, 3), keepdim=True) + 1e-5)
        else:
            a0 = x

        bw = WeightBinaryQuantize.apply(w2)
        ba = ActivationBinaryQuantize.apply(a0)

        out = F.conv2d(
            ba, bw, self.bias, self.stride, self.padding,
            self.dilation, self.groups
        )
        return out * self._effective_alpha()


def iter_recu_binary_convs(model):
    for module in model.modules():
        if isinstance(module, ReCUBinaryConv2d):
            yield module
