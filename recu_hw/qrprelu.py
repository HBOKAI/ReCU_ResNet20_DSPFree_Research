import torch
import torch.nn as nn

from .layers import round_ste


class QuantizedRPReLU(nn.Module):
    """A&B-BNN-style Quantized RPReLU.

    For channel i:
        y,                                  y >= 0
        2^round(a_i) * (y + xi1_i) + xi2_i, y < 0

    Training uses ordinary floating operations to learn the parameters.
    Deployment interpretation:
      - 2^round(a_i) is a bit shift
      - xi1 / xi2 are additions
      - therefore the negative branch does not require a general multiplier

    This module is NOT used in official ReCU reproduction mode.
    """
    def __init__(self, channels, init_log2_slope=-2.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((channels,), float(init_log2_slope)))
        self.xi1 = nn.Parameter(torch.zeros(channels))
        self.xi2 = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        if x.dim() == 4:
            shape = (1, -1, 1, 1)
        elif x.dim() == 2:
            shape = (1, -1)
        else:
            raise ValueError("QuantizedRPReLU expects NCHW or NC tensor")

        a = self.a.view(*shape)
        xi1 = self.xi1.view(*shape)
        xi2 = self.xi2.view(*shape)

        shift_exp = round_ste(a)
        slope = torch.pow(2.0, shift_exp)
        neg = slope * (x + xi1) + xi2
        return torch.where(x >= 0, x, neg)

    @torch.no_grad()
    def integer_shift_exponents(self):
        return torch.round(self.a).to(torch.int32)
