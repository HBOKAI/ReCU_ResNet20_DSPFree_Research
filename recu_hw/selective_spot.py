import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_affine import fold_recu_alpha_bn
from .layers import ReCUBinaryConv2d
from .model import option_a_shortcut
from .qrprelu import QuantizedRPReLU
from .spot_affine import SelectiveSPoTAffine2d, best_two_term_or_one


EXPECTED_BACKBONE_K = 672


class SelectiveSPoTBasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = ReCUBinaryConv2d(
            in_planes, planes, 3,
            stride=stride, padding=1, bias=False,
            alpha_mode="one",
        )
        self.aff1 = SelectiveSPoTAffine2d(planes)

        self.conv2 = ReCUBinaryConv2d(
            planes, planes, 3,
            stride=1, padding=1, bias=False,
            alpha_mode="one",
        )
        self.aff2 = SelectiveSPoTAffine2d(planes)

        self.post_act = QuantizedRPReLU(planes)
        self.shortcut = option_a_shortcut(in_planes, planes, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.aff1(self.conv1(x))
        out = out + self.shortcut(x)
        x1 = out

        out = F.hardtanh(out)
        out = self.aff2(self.conv2(out))
        out = out + x1
        return self.post_act(out)


class SelectiveSPoTResNet20(nn.Module):
    """R4-level ResNet20 variant with isolated selectable SPoT affines.

    Stem/head remain exactly at the R4 level (floating stem BN, head BN, FC).
    Only the 18 backbone post-binary-conv affine K vectors are eligible for
    <=2-term SPoT.  This is deliberately not the E1/H2A-v2 deployment graph.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 3, 1)
        self.layer2 = self._make_layer(32, 3, 2)
        self.layer3 = self._make_layer(64, 3, 2)

        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(SelectiveSPoTBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        return self.linear(out)


@torch.no_grad()
def build_selective_spot_from_r2(r2_model: nn.Module) -> SelectiveSPoTResNet20:
    """Build the isolated SPoT study graph from the same formal R2 source as R4."""
    target = SelectiveSPoTResNet20()

    target.conv1.load_state_dict(r2_model.conv1.state_dict())
    target.bn1.load_state_dict(r2_model.bn1.state_dict())
    target.bn2.load_state_dict(r2_model.bn2.state_dict())
    target.linear.load_state_dict(r2_model.linear.state_dict())

    for src_layer, dst_layer in zip(
        [r2_model.layer1, r2_model.layer2, r2_model.layer3],
        [target.layer1, target.layer2, target.layer3],
    ):
        for src, dst in zip(src_layer, dst_layer):
            dst.conv1.weight.copy_(src.conv1.weight)
            dst.conv2.weight.copy_(src.conv2.weight)

            # Keep the historical compatibility alpha values in state_dict,
            # but alpha_mode='one' removes them from inference exactly as R4.
            dst.conv1.alpha.copy_(src.conv1.alpha)
            dst.conv2.alpha.copy_(src.conv2.alpha)
            dst.conv1.alpha.requires_grad_(False)
            dst.conv2.alpha.requires_grad_(False)

            k1, b1 = fold_recu_alpha_bn(src.conv1, src.bn1)
            k2, b2 = fold_recu_alpha_bn(src.conv2, src.bn2)
            dst.aff1.init_from_kb(k1, b1)
            dst.aff2.init_from_kb(k2, b2)

            dst.post_act.a.copy_(src.post_act.a)
            dst.post_act.xi1.copy_(src.post_act.xi1)
            dst.post_act.xi2.copy_(src.post_act.xi2)

    verify_target_count(target)
    return target


def named_spot_affines(model: nn.Module) -> Iterable[Tuple[str, SelectiveSPoTAffine2d]]:
    for name, module in model.named_modules():
        if isinstance(module, SelectiveSPoTAffine2d):
            yield name, module


def target_channel_count(model: nn.Module) -> int:
    return sum(m.channels for _, m in named_spot_affines(model))


def verify_target_count(model: nn.Module, expected: int = EXPECTED_BACKBONE_K) -> int:
    count = target_channel_count(model)
    if count != int(expected):
        raise RuntimeError(
            f"Selective SPoT target count mismatch: got {count}, expected {expected}. "
            "Stop rather than forcing historical 672 assumptions."
        )
    return count


def flatten_target_channels(model: nn.Module) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    global_index = 0
    for module_name, module in named_spot_affines(model):
        for channel in range(module.channels):
            rows.append({
                "global_index": global_index,
                "module": module_name,
                "channel": channel,
            })
            global_index += 1
    return rows


def coverage_to_count(total: int, coverage_pct: float) -> int:
    pct = float(coverage_pct)
    if not 0.0 <= pct <= 100.0:
        raise ValueError("coverage_pct must be in [0, 100]")
    # Explicit round-half-up avoids Python banker's rounding ambiguity.
    return min(int(total), max(0, int(math.floor(total * pct / 100.0 + 0.5))))


@torch.no_grad()
def clear_selection(model: nn.Module) -> None:
    for _, module in named_spot_affines(model):
        module.set_selected(torch.zeros(module.channels, dtype=torch.bool, device=module.selected_2term.device))


@torch.no_grad()
def apply_ranked_selection(
    model: nn.Module,
    ranking: Sequence[Dict],
    coverage_pct: float,
) -> List[Dict]:
    """Apply top-ranked deterministic channel selection and return manifest rows."""
    total = verify_target_count(model)
    if len(ranking) != total:
        raise ValueError(f"ranking has {len(ranking)} entries, expected {total}")

    expected_keys = {
        (row["module"], int(row["channel"]))
        for row in flatten_target_channels(model)
    }
    ranking_keys = [(str(r["module"]), int(r["channel"])) for r in ranking]
    if len(set(ranking_keys)) != total or set(ranking_keys) != expected_keys:
        raise ValueError("ranking must contain every target module/channel exactly once")

    clear_selection(model)
    n = coverage_to_count(total, coverage_pct)
    chosen = list(ranking[:n])

    module_map = dict(named_spot_affines(model))
    masks = {
        name: torch.zeros(m.channels, dtype=torch.bool, device=m.selected_2term.device)
        for name, m in module_map.items()
    }
    for row in chosen:
        name = str(row["module"])
        ch = int(row["channel"])
        masks[name][ch] = True

    for name, module in module_map.items():
        module.set_selected(masks[name])

    manifest = []
    for rank, row in enumerate(chosen, start=1):
        manifest.append({
            "rank": rank,
            "module": str(row["module"]),
            "channel": int(row["channel"]),
            "global_index": int(row.get("global_index", -1)),
        })
    return manifest


@torch.no_grad()
def coefficient_audit_rows(model: nn.Module) -> List[Dict]:
    """Return per-channel one-term vs exact <=2-term coefficient audit."""
    verify_target_count(model)
    rows: List[Dict] = []
    global_index = 0

    for module_name, module in named_spot_affines(model):
        latent = module._continuous_k().detach()
        rep = best_two_term_or_one(latent, module.exp_min, module.exp_max)

        for ch in range(module.channels):
            kf = float(latent[ch].cpu())
            q1 = float(rep["one_q"][ch].cpu())
            qbest = float(rep["q"][ch].cpu())
            ae1 = abs(kf - q1)
            aeb = abs(kf - qbest)
            denom = max(abs(kf), 1e-12)
            active2 = bool(rep["active2"][ch].cpu())

            rows.append({
                "global_index": global_index,
                "module": module_name,
                "channel": ch,
                "k_float": kf,
                "k_1term": q1,
                "one_exp": int(rep["one_exp"][ch].cpu()),
                "one_sign": int(float(rep["one_sign"][ch].cpu())),
                "one_abs_error": ae1,
                "one_relative_error": ae1 / denom,
                "k_2term_or_one": qbest,
                "second_term_needed": active2,
                "spot_k1": int(rep["k1"][ch].cpu()),
                "spot_k2": int(rep["k2"][ch].cpu()),
                "spot_s1": int(float(rep["s1"][ch].cpu())),
                "spot_s2": int(float(rep["s2"][ch].cpu())),
                "spot_abs_error": aeb,
                "spot_relative_error": aeb / denom,
                "abs_error_reduction": ae1 - aeb,
                "relative_error_reduction": (ae1 - aeb) / denom,
                "best_proper_two_q": float(rep["best_proper_two_q"][ch].cpu()),
                "best_proper_two_abs_error": float(rep["best_proper_two_abs_error"][ch].cpu()),
            })
            global_index += 1

    return rows
