from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .fused_affine import FusedAffine2d
from .r6 import StemFusedPow2Affine2d
from .r7 import HeadFusedPow2Affine1d


@dataclass
class IntegerBiasResult:
    module_name: str
    class_name: str
    param_count: int
    bits: int
    shift: int
    qmin: int
    qmax: int
    q_values: List[int]
    original_min: float
    original_max: float
    dequant_min: float
    dequant_max: float
    saturation_count: int
    mse: float
    max_abs_error: float

    def to_dict(self):
        result = asdict(self)
        result["name"] = result["module_name"]
        result["q_dtype"] = "int64"
        result["hardware_representation"] = "signed_integer_q_plus_power_of_two_shift"
        return result


def signed_int_range(bits: int) -> Tuple[int, int]:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


@torch.no_grad()
def quantize_bias_tensor(x: torch.Tensor, bits: int, shift: int):
    qmin, qmax = signed_int_range(bits)
    scaled = x.detach().to(torch.float64) * (2.0 ** shift)
    rounded = torch.round(scaled)
    sat_mask = (rounded < qmin) | (rounded > qmax)
    q = torch.clamp(rounded, qmin, qmax).to(torch.int64)
    x_hat = q.to(torch.float64) * (2.0 ** (-shift))
    ref = x.detach().to(torch.float64)
    err = x_hat - ref
    return {
        "q": q,
        "x_hat": x_hat.to(dtype=x.dtype, device=x.device),
        "saturation_count": int(sat_mask.sum().item()),
        "mse": float(torch.mean(err * err).item()),
        "max_abs_error": float(torch.max(torch.abs(err)).item()),
        "qmin": qmin,
        "qmax": qmax,
    }


@torch.no_grad()
def search_best_shift(
    x: torch.Tensor,
    bits: int,
    shift_min: int = 0,
    shift_max: int = 12,
    fallback_shift_min: int = -12,
):
    if shift_min > shift_max:
        raise ValueError("shift_min must be <= shift_max")

    candidates = []
    for s in range(shift_min, shift_max + 1):
        candidates.append((s, quantize_bias_tensor(x, bits, s)))

    extended = False
    if all(q["saturation_count"] > 0 for _, q in candidates):
        extended = True
        for s in range(shift_min - 1, fallback_shift_min - 1, -1):
            candidates.append((s, quantize_bias_tensor(x, bits, s)))

    def key(item):
        s, q = item
        sat = q["saturation_count"]
        return (
            0 if sat == 0 else 1,
            sat,
            q["mse"],
            q["max_abs_error"],
            abs(s),
        )

    shift, result = min(candidates, key=key)
    result["used_negative_extension"] = extended and shift < shift_min
    result["candidate_shift_min"] = min(s for s, _ in candidates)
    result["candidate_shift_max"] = max(s for s, _ in candidates)
    return shift, result


def _candidate_bias_module(name: str, module: nn.Module) -> bool:
    bias = getattr(module, "bias", None)
    if not isinstance(bias, nn.Parameter) or bias.ndim != 1:
        return False

    if name == "stem_affine":
        return isinstance(module, StemFusedPow2Affine2d)
    if name == "head_affine":
        return isinstance(module, HeadFusedPow2Affine1d)
    if name == "linear":
        return isinstance(module, nn.Linear)
    if name.startswith(("layer1.", "layer2.", "layer3.")):
        return isinstance(module, FusedAffine2d)

    return False


def collect_integer_bias_targets(model: nn.Module) -> Dict[str, nn.Module]:
    targets = {}
    for name, module in model.named_modules():
        if _candidate_bias_module(name, module):
            targets[name] = module

    required = {"stem_affine", "head_affine", "linear"}
    missing = required.difference(targets)
    if missing:
        raise RuntimeError(
            "Missing required bias targets: " + ", ".join(sorted(missing))
        )
    backbone = [name for name in targets if name.startswith(("layer1.", "layer2.", "layer3."))]
    if len(backbone) != 18:
        raise RuntimeError(
            f"Expected 18 R4 fused affine biases, found {len(backbone)}: {backbone}"
        )
    if len(targets) != 21:
        raise RuntimeError(f"Expected 21 integer bias targets, found {len(targets)}")
    return targets


@torch.no_grad()
def quantize_model_biases_inplace(
    model: nn.Module,
    bits: int,
    shift_min: int = 0,
    shift_max: int = 12,
    fallback_shift_min: int = -12,
):
    targets = collect_integer_bias_targets(model)
    exports = {}

    for name, module in targets.items():
        original = module.bias.detach().clone()
        shift, qres = search_best_shift(
            original,
            bits=bits,
            shift_min=shift_min,
            shift_max=shift_max,
            fallback_shift_min=fallback_shift_min,
        )
        module.bias.copy_(qres["x_hat"])

        exports[name] = IntegerBiasResult(
            module_name=name,
            class_name=module.__class__.__name__,
            param_count=int(original.numel()),
            bits=bits,
            shift=int(shift),
            qmin=int(qres["qmin"]),
            qmax=int(qres["qmax"]),
            q_values=[int(v) for v in qres["q"].cpu().tolist()],
            original_min=float(original.min().item()),
            original_max=float(original.max().item()),
            dequant_min=float(qres["x_hat"].min().item()),
            dequant_max=float(qres["x_hat"].max().item()),
            saturation_count=int(qres["saturation_count"]),
            mse=float(qres["mse"]),
            max_abs_error=float(qres["max_abs_error"]),
        ).to_dict()
        exports[name]["used_negative_extension"] = bool(qres["used_negative_extension"])
        exports[name]["candidate_shift_min"] = int(qres["candidate_shift_min"])
        exports[name]["candidate_shift_max"] = int(qres["candidate_shift_max"])

    return exports
