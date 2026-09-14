from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import math
import torch

def signed_int_range(bits: int) -> Tuple[int, int]:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

def _round_shift_right_signed(x: torch.Tensor, amount: int) -> torch.Tensor:
    if amount <= 0:
        return x
    offset = 1 << (amount - 1)
    pos = (x + offset) >> amount
    neg = -(((-x) + offset) >> amount)
    return torch.where(x >= 0, pos, neg)

@torch.no_grad()
def quantize_float_to_int(x: torch.Tensor, bits: int, shift: int):
    qmin, qmax = signed_int_range(bits)
    scaled = x.detach().to(torch.float64) * (2.0 ** shift)
    rounded = torch.round(scaled)
    sat = (rounded < qmin) | (rounded > qmax)
    q = torch.clamp(rounded, qmin, qmax).to(torch.int64)
    x_hat = q.to(torch.float64) * (2.0 ** (-shift))
    ref = x.detach().to(torch.float64)
    err = x_hat - ref
    return {
        "q": q,
        "x_hat": x_hat.to(dtype=x.dtype, device=x.device),
        "saturation_count": int(sat.sum().item()),
        "mse": float(torch.mean(err * err).item()) if err.numel() else 0.0,
        "max_abs_error": float(torch.max(torch.abs(err)).item()) if err.numel() else 0.0,
        "qmin": qmin,
        "qmax": qmax,
    }

def choose_zero_saturation_shift(absmax: float, bits: int, min_shift=-16, max_shift=24) -> int:
    _, qmax = signed_int_range(bits)
    if absmax <= 0.0:
        return max_shift
    raw = math.floor(math.log2(qmax / absmax))
    return max(min(raw, max_shift), min_shift)

@torch.no_grad()
def requantize_int(q: torch.Tensor, src_shift: int, dst_shift: int, out_bits: Optional[int] = None):
    q = q.to(torch.int64)
    delta = int(dst_shift) - int(src_shift)
    if delta >= 0:
        out = q << delta
    else:
        out = _round_shift_right_signed(q, -delta)

    saturation_count = 0
    if out_bits is not None:
        qmin, qmax = signed_int_range(out_bits)
        sat = (out < qmin) | (out > qmax)
        saturation_count = int(sat.sum().item())
        out = torch.clamp(out, qmin, qmax)
    return out, saturation_count

@torch.no_grad()
def aligned_integer_add(qa, shift_a, qb, shift_b, out_bits, out_shift):
    a, _ = requantize_int(qa, shift_a, out_shift, out_bits=None)
    b, _ = requantize_int(qb, shift_b, out_shift, out_bits=None)
    summed = a + b
    qmin, qmax = signed_int_range(out_bits)
    sat = (summed < qmin) | (summed > qmax)
    out = torch.clamp(summed, qmin, qmax)
    return {
        "q": out,
        "saturation_count": int(sat.sum().item()),
    }

@dataclass
class RangeStats:
    name: str
    count: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    absmax: float = 0.0

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        d = x.detach()
        mn = float(d.min().item())
        mx = float(d.max().item())
        self.count += int(d.numel())
        self.min_value = min(self.min_value, mn)
        self.max_value = max(self.max_value, mx)
        self.absmax = max(self.absmax, abs(mn), abs(mx))

    def to_dict(self):
        return asdict(self)

class RangeObserverRegistry:
    def __init__(self):
        self.stats: Dict[str, RangeStats] = {}

    def observe(self, name: str, x: torch.Tensor):
        if name not in self.stats:
            self.stats[name] = RangeStats(name=name)
        self.stats[name].update(x)

    def shifts_for_bits(self, bits: int, min_shift=-16, max_shift=24):
        return {
            name: choose_zero_saturation_shift(s.absmax, bits, min_shift, max_shift)
            for name, s in self.stats.items()
        }

    def to_dict(self):
        return {name: s.to_dict() for name, s in self.stats.items()}

class IntegerActivationFakeQuant:
    def __init__(self, bits: int, shift_map: Dict[str, int]):
        self.bits = int(bits)
        self.shift_map = dict(shift_map)
        self.saturation_by_node: Dict[str, int] = {}
        self.error_by_node: Dict[str, dict] = {}

    @torch.no_grad()
    def quantize(self, name: str, x: torch.Tensor):
        if name not in self.shift_map:
            raise KeyError(f"No calibrated shift for node: {name}")
        r = quantize_float_to_int(x, self.bits, self.shift_map[name])
        self.saturation_by_node[name] = self.saturation_by_node.get(name, 0) + r["saturation_count"]
        error = r["x_hat"].to(torch.float64) - x.detach().to(torch.float64)
        stats = self.error_by_node.setdefault(
            name,
            {"element_count": 0, "squared_error_sum": 0.0, "max_abs_error": 0.0},
        )
        stats["element_count"] += int(error.numel())
        stats["squared_error_sum"] += float(torch.sum(error * error).item())
        if error.numel():
            stats["max_abs_error"] = max(
                stats["max_abs_error"], float(torch.max(torch.abs(error)).item())
            )
        r["shift"] = int(self.shift_map[name])
        return r

    @torch.no_grad()
    def record_saturation(self, name: str, count: int):
        self.saturation_by_node[name] = self.saturation_by_node.get(name, 0) + int(count)

    @torch.no_grad()
    def __call__(self, name: str, x: torch.Tensor):
        return self.quantize(name, x)["x_hat"]

    def total_saturation(self):
        return sum(self.saturation_by_node.values())

    def error_statistics(self):
        result = {}
        for name, stats in self.error_by_node.items():
            count = max(int(stats["element_count"]), 1)
            result[name] = {
                "element_count": int(stats["element_count"]),
                "mse": float(stats["squared_error_sum"] / count),
                "max_abs_error": float(stats["max_abs_error"]),
            }
        return result
