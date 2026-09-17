import itertools
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .layers import round_ste


def quantize_one_term_like_r4(
    k: torch.Tensor,
    exp_min: int = -16,
    exp_max: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the R4 one-term K quantizer exactly.

    R4 stores sign(K) separately and quantizes log2(|K|) with round-to-nearest,
    then clamps the integer exponent to [exp_min, exp_max].  Zero follows the
    historical R4 initialization convention: positive sign and minimum
    representable magnitude.
    """
    sign = torch.where(k >= 0, torch.ones_like(k), -torch.ones_like(k))
    mag = k.abs().clamp_min(2.0 ** int(exp_min))
    exp = torch.round(torch.log2(mag))
    exp = torch.clamp(exp, int(exp_min), int(exp_max)).to(torch.int64)
    q = sign * torch.pow(torch.tensor(2.0, device=k.device, dtype=k.dtype), exp.to(k.dtype))
    return q, exp, sign


def _two_term_candidate_table(
    *,
    exp_min: int,
    exp_max: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic canonical proper 2-term SPoT candidates.

    Canonical form requires k1 > k2.  Candidate enumeration order is stable,
    so torch.argmin also gives deterministic tie breaking.
    """
    vals = []
    k1s = []
    k2s = []
    s1s = []
    s2s = []

    # Larger first exponent first, then larger second exponent, then signs.
    # Positive sign precedes negative sign for deterministic exact ties.
    for k1 in range(int(exp_max), int(exp_min), -1):
        for k2 in range(k1 - 1, int(exp_min) - 1, -1):
            for s1, s2 in itertools.product((1.0, -1.0), repeat=2):
                vals.append(s1 * (2.0 ** k1) + s2 * (2.0 ** k2))
                k1s.append(k1)
                k2s.append(k2)
                s1s.append(s1)
                s2s.append(s2)

    if not vals:
        raise ValueError("2-term SPoT requires exp_max > exp_min")

    return (
        torch.tensor(vals, device=device, dtype=dtype),
        torch.tensor(k1s, device=device, dtype=torch.int64),
        torch.tensor(k2s, device=device, dtype=torch.int64),
        torch.tensor(s1s, device=device, dtype=dtype),
        torch.tensor(s2s, device=device, dtype=dtype),
    )


@torch.no_grad()
def best_two_term_or_one(
    k: torch.Tensor,
    exp_min: int = -16,
    exp_max: int = 8,
) -> Dict[str, torch.Tensor]:
    """Find the exact nearest legal representation using <=2 signed POT terms.

    Proper 2-term candidates are searched exhaustively over the bounded
    exponent/sign space.  If a one-term R4 value is equally good or better,
    one-term wins and the second term is marked inactive.  This avoids paying
    hardware for a redundant second term.
    """
    shape = k.shape
    flat = k.detach().reshape(-1)
    q1, e1, s1_one = quantize_one_term_like_r4(flat, exp_min, exp_max)

    cand, k1_table, k2_table, s1_table, s2_table = _two_term_candidate_table(
        exp_min=exp_min,
        exp_max=exp_max,
        device=flat.device,
        dtype=flat.dtype,
    )
    err = torch.abs(flat[:, None] - cand[None, :])
    idx = torch.argmin(err, dim=1)

    q2 = cand[idx]
    k1 = k1_table[idx]
    k2 = k2_table[idx]
    s1 = s1_table[idx]
    s2 = s2_table[idx]

    err1 = torch.abs(flat - q1)
    err2 = torch.abs(flat - q2)
    active2 = err2 < err1  # exact ties deliberately fall back to one-term

    q = torch.where(active2, q2, q1)
    out_k1 = torch.where(active2, k1, e1)
    out_s1 = torch.where(active2, s1, s1_one)
    out_k2 = torch.where(active2, k2, torch.zeros_like(k2))
    out_s2 = torch.where(active2, s2, torch.zeros_like(s2))

    def rs(x):
        return x.reshape(shape)

    return {
        "q": rs(q),
        "active2": rs(active2),
        "k1": rs(out_k1),
        "k2": rs(out_k2),
        "s1": rs(out_s1),
        "s2": rs(out_s2),
        "one_q": rs(q1),
        "one_exp": rs(e1),
        "one_sign": rs(s1_one),
        "one_abs_error": rs(err1),
        "best_proper_two_q": rs(q2),
        "best_proper_two_abs_error": rs(err2),
    }


class SelectiveSPoTAffine2d(nn.Module):
    """Per-channel y = K*x + B with mixed one-term / <=2-term SPoT K.

    This class is intentionally separate from FusedAffine2d so historical R4
    checkpoints and semantics are untouched.

    Unselected channels exactly use the R4 one-term forward:
        sign(K0) * 2^round(log2_abs_k)

    Selected channels use an exhaustive discrete <=2-term SPoT projection in
    the forward pass.  The projected value is attached to the continuous
    latent K with an identity STE for optimization.  The actual forward value
    remains exactly representable by shift/negate (+ optional second
    shift/negate and add); no arbitrary floating deployment multiplier leaks
    into inference.
    """

    def __init__(self, channels: int, exp_min: int = -16, exp_max: int = 8):
        super().__init__()
        self.channels = int(channels)
        self.exp_min = int(exp_min)
        self.exp_max = int(exp_max)

        self.log2_abs_k = nn.Parameter(torch.zeros(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.register_buffer("sign_k", torch.ones(self.channels))
        self.register_buffer("selected_2term", torch.zeros(self.channels, dtype=torch.bool))

    @torch.no_grad()
    def init_from_kb(self, k: torch.Tensor, b: torch.Tensor) -> None:
        k = k.detach().flatten()
        b = b.detach().flatten()
        if k.numel() != self.channels or b.numel() != self.channels:
            raise ValueError("K/B channel count mismatch")

        sign = torch.where(k >= 0, torch.ones_like(k), -torch.ones_like(k))
        mag = k.abs().clamp_min(2.0 ** self.exp_min)
        self.sign_k.copy_(sign)
        self.log2_abs_k.copy_(torch.log2(mag))
        self.bias.copy_(b)

    @torch.no_grad()
    def set_selected(self, mask: torch.Tensor) -> None:
        mask = torch.as_tensor(mask, device=self.selected_2term.device, dtype=torch.bool).flatten()
        if mask.numel() != self.channels:
            raise ValueError(f"selection mask has {mask.numel()} values, expected {self.channels}")
        self.selected_2term.copy_(mask)

    def _continuous_k(self) -> torch.Tensor:
        exp = torch.clamp(self.log2_abs_k, self.exp_min, self.exp_max)
        return self.sign_k * torch.pow(2.0, exp)

    def effective_k(self) -> torch.Tensor:
        exp_one = torch.clamp(
            round_ste(self.log2_abs_k), self.exp_min, self.exp_max
        )
        q_one = self.sign_k * torch.pow(2.0, exp_one)

        if not bool(self.selected_2term.any()):
            return q_one

        latent = self._continuous_k()
        with torch.no_grad():
            rep = best_two_term_or_one(latent.detach(), self.exp_min, self.exp_max)
            q_disc = rep["q"]
            active2 = rep["active2"] & self.selected_2term

        # Identity STE for the selected proper 2-term projection.  Channels
        # whose best <=2-term representation falls back to one-term retain the
        # exact historical one-term path and its round_ste gradient.
        q_two_ste = latent + (q_disc - latent).detach()
        return torch.where(active2, q_two_ste, q_one)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.effective_k().view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        return x * k + b

    @torch.no_grad()
    def discrete_terms(self) -> Dict[str, torch.Tensor]:
        latent = self._continuous_k().detach()
        rep = best_two_term_or_one(latent, self.exp_min, self.exp_max)

        # A channel only incurs a second term if it is selected AND a proper
        # second term strictly improves the one-term value.
        active2 = rep["active2"] & self.selected_2term
        q = torch.where(active2, rep["q"], rep["one_q"])
        k1 = torch.where(active2, rep["k1"], rep["one_exp"])
        s1 = torch.where(active2, rep["s1"], rep["one_sign"])
        k2 = torch.where(active2, rep["k2"], torch.zeros_like(rep["k2"]))
        s2 = torch.where(active2, rep["s2"], torch.zeros_like(rep["s2"]))

        return {
            "q": q,
            "active2": active2,
            "selected": self.selected_2term.clone(),
            "k1": k1,
            "k2": k2,
            "s1": s1,
            "s2": s2,
            "latent_k": latent,
        }
