"""Integer arithmetic helpers used by the H2 finite-width experiment."""

from typing import Dict, Iterable, List, Tuple
import math

import torch


def signed_range(bits: int) -> Tuple[int, int]:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def signed_bits_for_range(lo: int, hi: int) -> int:
    if lo > hi:
        raise ValueError("invalid range")
    bits = 2
    while True:
        qlo, qhi = signed_range(bits)
        if lo >= qlo and hi <= qhi:
            return bits
        bits += 1


def binary_dot_exact_width(kernel: int, cin: int) -> Dict[str, int]:
    terms = int(kernel) * int(kernel) * int(cin)
    return {
        "terms": terms,
        "popcount_bits": math.ceil(math.log2(terms + 1)),
        "signed_min": -terms,
        "signed_max": terms,
        "signed_acc_bits": signed_bits_for_range(-terms, terms),
    }


def exact_add_width(width_a: int, width_b: int) -> int:
    return max(int(width_a), int(width_b)) + 1


def gap_exact_width(input_bits: int, elements: int = 64) -> int:
    if elements < 1:
        raise ValueError("elements must be >= 1")
    return int(input_bits) + math.ceil(math.log2(elements))


def arithmetic_right_shift_round(x: torch.Tensor, amount: int) -> torch.Tensor:
    x = x.to(torch.int64)
    if amount <= 0:
        return x
    offset = 1 << (amount - 1)
    pos = (x + offset) >> amount
    neg = -(((-x) + offset) >> amount)
    return torch.where(x >= 0, pos, neg)


def requantize_integer(q: torch.Tensor, src_shift: int, dst_shift: int) -> torch.Tensor:
    q = q.to(torch.int64)
    delta = int(dst_shift) - int(src_shift)
    if delta >= 0:
        return q << delta
    return arithmetic_right_shift_round(q, -delta)


def clamp_signed(q: torch.Tensor, bits: int):
    qmin, qmax = signed_range(bits)
    sat = (q < qmin) | (q > qmax)
    return torch.clamp(q, qmin, qmax), int(sat.sum().item())


def simulate_finite_width(q: torch.Tensor, bits: int, mode: str = "saturate"):
    q = q.to(torch.int64)
    if mode == "saturate":
        return clamp_signed(q, bits)
    if mode == "assert":
        qmin, qmax = signed_range(bits)
        bad = (q < qmin) | (q > qmax)
        if bool(bad.any()):
            raise OverflowError(f"{int(bad.sum().item())} values exceed INT{bits}")
        return q, 0
    raise ValueError("mode must be 'saturate' or 'assert'")


def worst_case_shifted_sum_range(
    qmin: int,
    qmax: int,
    left_shifts: Iterable[int],
    signs: Iterable[int],
) -> Tuple[int, int]:
    lo = 0
    hi = 0
    for shift, sign in zip(left_shifts, signs):
        if shift < 0:
            raise ValueError("left_shifts must be >= 0 after scale alignment")
        if sign not in (-1, 1):
            raise ValueError("sign must be +/-1")
        a = qmin << int(shift)
        b = qmax << int(shift)
        if sign < 0:
            a, b = -b, -a
        lo += min(a, b)
        hi += max(a, b)
    return lo, hi


def exact_fc_acc_width_per_class(
    input_bits: int,
    left_shifts_by_class: List[List[int]],
    signs_by_class: List[List[int]],
):
    qmin, qmax = signed_range(input_bits)
    result = []
    for cls, (shifts, signs) in enumerate(zip(left_shifts_by_class, signs_by_class)):
        lo, hi = worst_case_shifted_sum_range(qmin, qmax, shifts, signs)
        result.append({
            "class": cls,
            "min": int(lo),
            "max": int(hi),
            "bits": signed_bits_for_range(int(lo), int(hi)),
        })
    return result
