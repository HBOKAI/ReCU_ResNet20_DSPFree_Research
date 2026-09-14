"""Arithmetic and calibration utilities for H1-S/H1-MP."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import math

import torch


def signed_int_range(bits: int) -> Tuple[int, int]:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


@torch.no_grad()
def quantize_float_to_int(x: torch.Tensor, bits: int, shift: int):
    qmin, qmax = signed_int_range(bits)
    src = x.detach().to(torch.float64)
    rounded = torch.round(src * (2.0 ** int(shift)))
    sat = (rounded < qmin) | (rounded > qmax)
    q = torch.clamp(rounded, qmin, qmax).to(torch.int64)
    x_hat = q.to(torch.float64) * (2.0 ** (-int(shift)))
    err = x_hat - src
    return {
        "q": q,
        "x_hat": x_hat.to(dtype=x.dtype, device=x.device),
        "saturation_count": int(sat.sum().item()),
        "mse": float(torch.mean(err * err).item()) if err.numel() else 0.0,
        "max_abs_error": float(torch.max(torch.abs(err)).item()) if err.numel() else 0.0,
        "qmin": qmin,
        "qmax": qmax,
    }


def choose_absmax_shift(absmax: float, bits: int, min_shift=-16, max_shift=24) -> int:
    _, qmax = signed_int_range(bits)
    if absmax <= 0.0:
        return int(max_shift)
    shift = math.floor(math.log2(qmax / absmax))
    return max(min(int(shift), int(max_shift)), int(min_shift))


def choose_percentile_shift(samples: torch.Tensor, bits: int, percentile: float,
                            min_shift=-16, max_shift=24) -> int:
    if not (0.0 < percentile <= 100.0):
        raise ValueError("percentile must be in (0,100]")
    if samples.numel() == 0:
        return int(max_shift)
    magnitudes = torch.abs(samples.detach().to(torch.float64).cpu())
    threshold = float(torch.quantile(magnitudes, percentile / 100.0).item())
    return choose_absmax_shift(threshold, bits, min_shift, max_shift)


def choose_mse_shift(samples: torch.Tensor, bits: int, min_shift=-16, max_shift=24) -> int:
    if samples.numel() == 0:
        return int(max_shift)
    values = samples.detach().to(torch.float64).cpu()
    best = None
    for shift in range(int(min_shift), int(max_shift) + 1):
        result = quantize_float_to_int(values, bits, shift)
        key = (result["mse"], result["saturation_count"], result["max_abs_error"], abs(shift))
        if best is None or key < best[0]:
            best = (key, shift)
    return int(best[1])


@dataclass
class NodeCalibrationStats:
    name: str
    count_seen: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    absmax: float = 0.0

    def update_range(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        values = x.detach()
        mn = float(values.min().item())
        mx = float(values.max().item())
        self.count_seen += int(values.numel())
        self.min_value = min(self.min_value, mn)
        self.max_value = max(self.max_value, mx)
        self.absmax = max(self.absmax, abs(mn), abs(mx))

    def to_dict(self):
        return asdict(self)


class CalibrationSampleCollector:
    def __init__(self, sample_cap_per_node=65536, values_per_batch=2048):
        self.sample_cap_per_node = int(sample_cap_per_node)
        self.values_per_batch = int(values_per_batch)
        self.stats: Dict[str, NodeCalibrationStats] = {}
        self.samples: Dict[str, List[torch.Tensor]] = {}
        self.sample_counts: Dict[str, int] = {}

    @torch.no_grad()
    def observe(self, name: str, x: torch.Tensor):
        if name not in self.stats:
            self.stats[name] = NodeCalibrationStats(name=name)
            self.samples[name] = []
            self.sample_counts[name] = 0
        self.stats[name].update_range(x)
        remaining = self.sample_cap_per_node - self.sample_counts[name]
        if remaining <= 0 or x.numel() == 0:
            return
        flat = x.detach().reshape(-1)
        take = min(self.values_per_batch, remaining, flat.numel())
        if take <= 0:
            return
        if take == flat.numel():
            sampled = flat
        else:
            indices = torch.linspace(0, flat.numel() - 1, steps=take, device=flat.device).round().long()
            sampled = flat.index_select(0, indices)
        self.samples[name].append(sampled.to(torch.float32).cpu())
        self.sample_counts[name] += int(sampled.numel())

    def tensor(self, name: str) -> torch.Tensor:
        chunks = self.samples.get(name, [])
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0)

    def choose_shift(self, name: str, bits: int, policy: str, min_shift=-16, max_shift=24) -> int:
        policy = policy.lower()
        if policy == "absmax":
            return choose_absmax_shift(self.stats[name].absmax, bits, min_shift, max_shift)
        if policy == "p99.9":
            return choose_percentile_shift(self.tensor(name), bits, 99.9, min_shift, max_shift)
        if policy == "p99.99":
            return choose_percentile_shift(self.tensor(name), bits, 99.99, min_shift, max_shift)
        if policy == "mse":
            return choose_mse_shift(self.tensor(name), bits, min_shift, max_shift)
        raise ValueError(f"Unknown policy {policy}")

    def export(self):
        result = {}
        for name, stats in self.stats.items():
            result[name] = stats.to_dict()
            result[name]["sample_count"] = int(self.sample_counts.get(name, 0))
        return result


class SelectiveIntegerFakeQuant:
    """Quantize only selected nodes; all other nodes remain float reference."""

    def __init__(self, node_bits: Dict[str, int], node_shifts: Dict[str, int]):
        self.node_bits = {str(name): int(bits) for name, bits in node_bits.items()}
        self.node_shifts = {str(name): int(shift) for name, shift in node_shifts.items()}
        self.shift_map = self.node_shifts
        self.saturation_by_node: Dict[str, int] = {}
        missing = set(self.node_bits) - set(self.node_shifts)
        if missing:
            raise ValueError("Missing shifts for: " + ", ".join(sorted(missing)))

    def is_quantized(self, name: str) -> bool:
        return name in self.node_bits

    def bits_for(self, name: str) -> int:
        return int(self.node_bits[name])

    @torch.no_grad()
    def quantize(self, name: str, x: torch.Tensor):
        if name not in self.node_bits:
            return None
        result = quantize_float_to_int(x, self.node_bits[name], self.node_shifts[name])
        self.saturation_by_node[name] = self.saturation_by_node.get(name, 0) + result["saturation_count"]
        result["shift"] = int(self.node_shifts[name])
        return result

    @torch.no_grad()
    def record_saturation(self, name: str, count: int):
        self.saturation_by_node[name] = self.saturation_by_node.get(name, 0) + int(count)

    @torch.no_grad()
    def __call__(self, name: str, x: torch.Tensor):
        result = self.quantize(name, x)
        return x if result is None else result["x_hat"]

    def total_saturation(self):
        return sum(self.saturation_by_node.values())


def classify_sensitivity(drop_pp: float) -> str:
    if drop_pp <= 0.02:
        return "robust"
    if drop_pp <= 0.10:
        return "mild"
    if drop_pp <= 0.30:
        return "sensitive"
    return "very_sensitive"


def candidate_bits_for_class(cls: str):
    return {
        "robust": [6, 7, 8],
        "mild": [7, 8, 9],
        "sensitive": [8, 9, 10, 12],
        "very_sensitive": [10, 12, 14],
    }[cls]


def initial_bits_for_class(cls: str):
    return {"robust": 6, "mild": 7, "sensitive": 10, "very_sensitive": 12}[cls]
