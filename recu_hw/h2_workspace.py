"""Workspace-specific H2 finite-width integration for the frozen R8B-H1MP model.

The H1MP state contract is intentionally kept intact.  This module adds an
integer datapath around that contract and records every remaining arithmetic
boundary needed by the H2 finite-width experiment.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .h1_model_adapter import H1ModelAdapter, _IntegerState
from .h1s_mp_quant import SelectiveIntegerFakeQuant
from .h2_integer_accum import (
    arithmetic_right_shift_round,
    binary_dot_exact_width,
    clamp_signed,
    exact_add_width,
    exact_fc_acc_width_per_class,
    gap_exact_width,
    requantize_integer,
    signed_bits_for_range,
    signed_range,
    worst_case_shifted_sum_range,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _range_from_bits(bits: int) -> Tuple[int, int]:
    return signed_range(int(bits))


def _round_int_scalar(value: float, shift: int) -> int:
    return int(torch.round(torch.tensor(float(value) * (2.0 ** int(shift)))).item())


def verify_gap_deferred_scaling(q_sum: torch.Tensor, input_shift: int) -> dict:
    """Verify GAP average and deferred scale metadata are bit-accurate.

    The two expressions differ only in where the power-of-two division is
    represented:

        (q_sum / 64) * 2**(-s) == q_sum * 2**(-(s + 6))

    No integer rounding is part of this check.  The helper intentionally uses
    float64 only for the mathematical-value comparison; the datapath keeps
    ``q_sum`` as the integer representation.
    """
    q = q_sum.detach().to(torch.float64)
    reference = (q / 64.0) * (2.0 ** (-int(input_shift)))
    implicit_scale = q * (2.0 ** (-(int(input_shift) + 6)))
    error = (reference - implicit_scale).abs()
    return {
        "input_shift": int(input_shift),
        "scale_shift_after_deferred_divide": int(input_shift) + 6,
        "elements": int(q.numel()),
        "max_abs_error": float(error.max().item()) if error.numel() else 0.0,
        "mean_abs_error": float(error.mean().item()) if error.numel() else 0.0,
        "mathematically_equal": bool(torch.equal(reference, implicit_scale)),
        "integer_rounding_applied": False,
    }


class TraceBook:
    """Compact min/max/overflow bookkeeping without retaining feature maps."""

    def __init__(self):
        self.stats: Dict[str, dict] = {}

    def reset(self):
        self.stats = {}

    def record(
        self,
        name: str,
        value: torch.Tensor,
        bits: Optional[int] = None,
        saturation: int = 0,
        overflow: int = 0,
        kind: Optional[str] = None,
    ):
        if value is None or value.numel() == 0:
            return
        x = value.detach()
        if x.dtype.is_floating_point:
            x_min = float(x.min().item())
            x_max = float(x.max().item())
        else:
            x_min = int(x.min().item())
            x_max = int(x.max().item())
        item = self.stats.setdefault(
            str(name),
            {
                "name": str(name),
                "count": 0,
                "min": float("inf"),
                "max": float("-inf"),
                "absmax": 0.0,
                "saturation_count": 0,
                "overflow_count": 0,
            },
        )
        item["count"] += int(x.numel())
        item["min"] = min(item["min"], x_min)
        item["max"] = max(item["max"], x_max)
        item["absmax"] = max(item["absmax"], abs(x_min), abs(x_max))
        item["saturation_count"] += int(saturation)
        item["overflow_count"] += int(overflow)
        if bits is not None:
            item["bits"] = int(bits)
        if kind is not None:
            item["kind"] = str(kind)

    def export(self) -> Dict[str, dict]:
        return {name: dict(value) for name, value in self.stats.items()}


def _signed_align_range(bits: int, src_shift: int, dst_shift: int) -> Tuple[int, int]:
    qmin, qmax = signed_range(bits)
    endpoints = torch.tensor([qmin, qmax], dtype=torch.int64)
    aligned = requantize_integer(endpoints, src_shift, dst_shift)
    return int(aligned.min().item()), int(aligned.max().item())


def _shifted_range(lo: int, hi: int, delta: int) -> Tuple[int, int]:
    if delta >= 0:
        return lo << delta, hi << delta
    values = arithmetic_right_shift_round(
        torch.tensor([lo, hi], dtype=torch.int64), -delta
    )
    return int(values.min().item()), int(values.max().item())


def _module_exponents(module) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(module, "integer_exponents"):
        exponents = module.integer_exponents().detach().to(torch.int64)
        effective = module.effective_k().detach()
    else:
        effective = module.effective_k().detach()
        exponents = torch.round(torch.log2(effective.abs())).to(torch.int64)
    signs = torch.where(effective >= 0, torch.ones_like(exponents), -torch.ones_like(exponents))
    return exponents.cpu(), signs.cpu()


def _bias_record(h0_info: Dict[str, dict], name: str) -> dict:
    if name not in h0_info:
        raise RuntimeError(f"H0 INT6 bias record missing for {name}")
    record = h0_info[name]
    if int(record.get("bits", -1)) != 6:
        raise RuntimeError(f"H0 bias {name} is not INT6")
    return record


class H2TraceAdapter(H1ModelAdapter):
    """H1MP adapter extended with finite-width integer internal arithmetic."""

    QRP_PARAM_SHIFT = 24

    def __init__(self, r8b_model, h1_fake_quant, h0_info):
        super().__init__(r8b_model)
        self._fake_quant = h1_fake_quant
        self.h0_info = h0_info
        self.h2_plan: Optional[dict] = None
        self.trace = TraceBook()
        self.trace_enabled = False
        self.runtime_overflow: Dict[str, int] = {}
        self.runtime_saturation: Dict[str, int] = {}
        self.runtime_observed: Dict[str, dict] = {}
        self.last_gap_sum: Optional[torch.Tensor] = None
        self.last_gap_input_shift: Optional[int] = None

    def set_h2_plan(self, plan: Optional[dict]):
        self.h2_plan = plan
        self.reset_runtime()

    def reset_runtime(self):
        self.trace.reset()
        self.runtime_overflow = {}
        self.runtime_saturation = {}
        self.runtime_observed = {}
        self.last_gap_sum = None
        self.last_gap_input_shift = None
        if self._fake_quant is not None:
            self._fake_quant.saturation_by_node = {}

    def set_trace_enabled(self, enabled: bool):
        self.trace_enabled = bool(enabled)

    def _record(self, name, value, bits=None, saturation=0, overflow=0, kind=None):
        if self.trace_enabled:
            self.trace.record(name, value, bits, saturation, overflow, kind)

    def _finite(self, name: str, q: torch.Tensor, bits: int, mode: str = "assert"):
        q = q.to(torch.int64)
        raw_min = int(q.min().item()) if q.numel() else 0
        raw_max = int(q.max().item()) if q.numel() else 0
        previous = self.runtime_observed.get(name)
        if previous is None:
            self.runtime_observed[name] = {
                "min": raw_min,
                "max": raw_max,
                "count": int(q.numel()),
            }
        else:
            previous["min"] = min(int(previous["min"]), raw_min)
            previous["max"] = max(int(previous["max"]), raw_max)
            previous["count"] += int(q.numel())
        qmin, qmax = signed_range(int(bits))
        mask = (q < qmin) | (q > qmax)
        overflow = int(mask.sum().item())
        if mode == "assert" and overflow:
            # Keep the run alive to produce a complete diagnostic report.  H2A
            # reports this as a failure through the recorded overflow count.
            pass
        bounded = torch.clamp(q, qmin, qmax)
        self.runtime_overflow[name] = self.runtime_overflow.get(name, 0) + overflow
        self.runtime_saturation[name] = self.runtime_saturation.get(name, 0) + overflow
        self._record(name, bounded, bits=bits, saturation=overflow, overflow=overflow, kind="finite_integer")
        return bounded

    def _h1_bits(self, name: str) -> int:
        return int(self._fake_quant.bits_for(name))

    def _h1_shift(self, name: str) -> int:
        return int(self._fake_quant.shift_map[name])

    def _binary_conv(self, name: str, conv, x: torch.Tensor, stem: bool = False):
        if stem:
            binary_x = x
            binary_w = torch.sign(conv.weight)
        else:
            # This is the exact eval-time ReCU activation/weight sign path.
            w = conv.weight
            w0 = w - w.mean(dim=(1, 2, 3), keepdim=True)
            w1 = w0 / (torch.sqrt(w0.var(dim=(1, 2, 3), keepdim=True) + 1e-5) / 2.0 / math.sqrt(2.0))
            ew = torch.mean(torch.abs(w1))
            q_tau = (-ew * torch.log(2.0 - 2.0 * conv.tau.clamp(max=0.999999))).detach()
            w2 = torch.clamp(w1, -q_tau, q_tau)
            binary_x = torch.sign(x)
            binary_w = torch.sign(w2)
        raw = F.conv2d(
            binary_x.to(torch.float32),
            binary_w.to(torch.float32),
            None,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )
        q = torch.round(raw).to(torch.int64)
        exact = binary_dot_exact_width(conv.kernel_size[0], conv.in_channels)
        self._record(
            name,
            q,
            bits=int(exact["signed_acc_bits"]),
            kind="binary_conv_signed_accumulator",
        )
        return q

    def _affine_integer(
        self,
        node_name: str,
        module_name: str,
        module,
        conv_q: torch.Tensor,
        conv_shift: int = 0,
        input_bits: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, dict]:
        exponents, signs = _module_exponents(module)
        bias = _bias_record(self.h0_info, module_name)
        bias_shift = int(bias["shift"])
        exponent_min = int(exponents.min().item())
        common_shift = max(bias_shift, conv_shift - exponent_min)
        term_delta = common_shift + exponents - conv_shift
        if bool((term_delta < 0).any()):
            raise RuntimeError(f"Negative affine integer shift for {node_name}")

        shape = [1, int(exponents.numel())] + [1] * (conv_q.ndim - 2)
        signs_device = signs.to(device=conv_q.device).view(*shape)
        powers = torch.tensor(
            [1 << int(value) for value in term_delta.tolist()],
            dtype=torch.int64,
            device=conv_q.device,
        ).view(*shape)
        q_term = conv_q * signs_device * powers
        q_bias = torch.tensor(bias["q_values"], dtype=torch.int64, device=conv_q.device)
        q_bias = q_bias * (1 << (common_shift - bias_shift))
        q_bias = q_bias.view(*shape)
        q_sum = q_term + q_bias

        if self.h2_plan is not None:
            record = self.h2_plan["affine_intermediates"][node_name]
            width = int(record["bits"])
            q_sum = self._finite(f"affine.{node_name}.add_output", q_sum, width, record.get("mode", "assert"))
        else:
            width = signed_bits_for_range(int(q_sum.detach().min().item()), int(q_sum.detach().max().item()))
        self._record(
            f"affine.{node_name}.shifted_term",
            q_term,
            bits=width,
            kind="pow2_affine_shifted_term",
        )
        self._record(
            f"affine.{node_name}.bias_aligned",
            q_bias,
            bits=width,
            kind="int6_bias_alignment",
        )
        self._record(
            f"affine.{node_name}.add_output",
            q_sum,
            bits=width,
            kind="pow2_affine_integer_add",
        )
        details = {
            "node": node_name,
            "module": module_name,
            "bits": int(width),
            "shift": int(common_shift),
            "exponents": [int(value) for value in exponents.tolist()],
            "signs": [int(value) for value in signs.tolist()],
            "bias_shift": bias_shift,
        }
        return q_sum, common_shift, details

    def _state(self, name: str, value: torch.Tensor) -> _IntegerState:
        state = super()._state(name, value)
        if state.q is not None:
            self._record(f"h1.{name}", state.q, bits=self._h1_bits(name), kind="h1mp_boundary_q")
        return state

    def _stage_shortcut(self, record, input_state: _IntegerState) -> _IntegerState:
        if self.h2_plan is None:
            return super()._stage_shortcut(record, input_state)
        if not record["transition"]:
            return input_state
        if input_state.q is None or input_state.shift is None:
            raise RuntimeError("H2 requires a fully quantized H1MP input state")
        block = record["block"]
        prefix = record["name"]
        q_short = self._option_a_integer(
            input_state.q,
            block.conv1.in_channels,
            block.conv1.out_channels,
            block.conv1.stride[0],
        )
        self._record(
            f"residual.{prefix}.stage_shortcut_option_a",
            q_short,
            bits=self._h1_bits(f"{prefix}.stage_shortcut"),
            kind="option_a_integer_shortcut",
        )
        inherited = q_short.to(torch.float32) * (2.0 ** (-int(input_state.shift)))
        return self._state(f"{prefix}.stage_shortcut", inherited)

    def _residual_add(self, add_name: str, branch_a: _IntegerState, branch_b: _IntegerState) -> _IntegerState:
        if self.h2_plan is None:
            return super()._residual_add(add_name, branch_a, branch_b)
        if any(state.q is None or state.shift is None for state in (branch_a, branch_b)):
            raise RuntimeError(f"H2 residual {add_name} received a non-integer branch")
        out_shift = self._h1_shift(add_name)
        a = requantize_integer(branch_a.q, branch_a.shift, out_shift)
        b = requantize_integer(branch_b.q, branch_b.shift, out_shift)
        record = self.h2_plan["residuals"][add_name]
        self._record(
            f"residual.{add_name}.align_a",
            a,
            bits=int(record["bits"]),
            kind="residual_scale_alignment",
        )
        self._record(
            f"residual.{add_name}.align_b",
            b,
            bits=int(record["bits"]),
            kind="residual_scale_alignment",
        )
        summed = a + b
        summed = self._finite(
            f"residual.{add_name}.add_output",
            summed,
            int(record["bits"]),
            record.get("mode", "assert"),
        )
        self._record(
            f"residual.{add_name}.add_output",
            summed,
            bits=int(record["bits"]),
            kind="residual_integer_add",
        )
        self._alignment_runtime[add_name] = {
            "saturation_count": self.runtime_saturation.get(f"residual.{add_name}.add_output", 0),
            "tensor_elements": int(summed.numel()),
        }
        real = summed.to(torch.float32) * (2.0 ** (-out_shift))
        return self._state(add_name, real)

    def _qrprelu_integer(self, record, second_add: _IntegerState) -> torch.Tensor:
        prefix = record["name"]
        block = record["block"]
        if second_add.q is None or second_add.shift is None:
            raise RuntimeError(f"H2 QRPReLU {prefix} received a non-integer state")
        q_in = second_add.q
        in_shift = int(second_add.shift)
        a_exp = block.post_act.integer_shift_exponents().detach().to(torch.int64).cpu()
        min_exp = int(a_exp.min().item())
        common_inner_shift = max(in_shift, self.QRP_PARAM_SHIFT)
        q_x = q_in << (common_inner_shift - in_shift)
        shape = [1, int(a_exp.numel())] + [1] * (q_in.ndim - 2)
        q_xi1 = torch.round(
            block.post_act.xi1.detach().to(torch.float64) * (2.0 ** common_inner_shift)
        ).to(torch.int64).to(q_in.device).view(*shape)
        q_inner = q_x + q_xi1
        q_xi2 = torch.round(
            block.post_act.xi2.detach().to(torch.float64) * (2.0 ** common_inner_shift)
        ).to(torch.int64).to(q_in.device).view(*shape)
        qrp_plan = self.h2_plan["qrprelu"][f"{prefix}.qrprelu_output"] if self.h2_plan is not None else {}
        self._record(
            f"qrprelu.{prefix}.shifted_input",
            q_x,
            bits=qrp_plan.get("inner_bits"),
            kind="qrprelu_shifted_input_branch_intermediate",
        )
        self._record(
            f"qrprelu.{prefix}.offset_xi1",
            q_xi1,
            bits=qrp_plan.get("inner_bits"),
            kind="qrprelu_xi1_offset",
        )
        self._record(
            f"qrprelu.{prefix}.offset_xi2",
            q_xi2,
            bits=qrp_plan.get("bits"),
            kind="qrprelu_xi2_offset",
        )
        out_shift = max(common_inner_shift, common_inner_shift - min_exp)
        deltas = out_shift + a_exp - common_inner_shift
        powers = torch.tensor(
            [1 << int(value) for value in deltas.tolist()],
            dtype=torch.int64,
            device=q_in.device,
        ).view(*shape)
        q_neg = q_inner * powers + (q_xi2 << (out_shift - common_inner_shift))
        q_pos = q_in << (out_shift - in_shift)
        q_out = torch.where(q_in >= 0, q_pos, q_neg)
        if self.h2_plan is not None:
            qrp = qrp_plan
            q_inner = self._finite(
                f"qrprelu.{prefix}.inner_add", q_inner, int(qrp["inner_bits"]), qrp.get("mode", "assert")
            )
            # Recompute the negative branch from the finite inner value so a
            # narrowed inner width has the same semantics as hardware.
            q_neg = q_inner * powers + (q_xi2 << (out_shift - common_inner_shift))
            q_out = torch.where(q_in >= 0, q_pos, q_neg)
            self._record(
                f"qrprelu.{prefix}.output_before_h1mp",
                q_out,
                bits=int(qrp["bits"]),
                kind="qrprelu_output_before_h1mp_requantization",
            )
            q_neg = self._finite(
                f"qrprelu.{prefix}.negative_branch", q_neg, int(qrp["bits"]), qrp.get("mode", "assert")
            )
            q_out = self._finite(
                f"qrprelu.{prefix}.output", q_out, int(qrp["bits"]), qrp.get("mode", "assert")
            )
        self._record(
            f"qrprelu.{prefix}.inner_add",
            q_inner,
            kind="qrprelu_x_plus_xi1",
        )
        self._record(
            f"qrprelu.{prefix}.negative_branch",
            q_neg,
            kind="qrprelu_shift_add_branch",
        )
        self._record(
            f"qrprelu.{prefix}.output",
            q_out,
            kind="qrprelu_integer_output",
        )
        self.trace.stats.setdefault(f"qrprelu.{prefix}.meta", {
            "name": f"qrprelu.{prefix}.meta",
            "shift": int(out_shift),
            "slope_exponents": [int(value) for value in a_exp.tolist()],
            "param_shift": int(common_inner_shift),
            "count": 0,
            "min": 0,
            "max": 0,
            "absmax": 0,
            "saturation_count": 0,
            "overflow_count": 0,
        })
        return q_out.to(torch.float32) * (2.0 ** (-out_shift))

    def _run_block(self, record, input_state: _IntegerState) -> _IntegerState:
        block = record["block"]
        prefix = record["name"]
        if self.h2_plan is None:
            x = input_state.value
            first = block.aff1(block.conv1(x))
            first_state = self._state(f"{prefix}.first_affine", first)
            shortcut_state = self._stage_shortcut(record, input_state)
            x1 = self._residual_add(f"{prefix}.x1", first_state, shortcut_state)
            pre_second = self._state(
                f"{prefix}.pre_bconv2_hardtanh", F.hardtanh(x1.value, inplace=False)
            )
            second = block.aff2(block.conv2(pre_second.value))
            second_state = self._state(f"{prefix}.second_affine", second)
            second_add = self._residual_add(f"{prefix}.second_add", second_state, x1)
            output = block.post_act(second_add.value)
            return self._state(f"{prefix}.qrprelu_output", output)

        first_q = self._binary_conv(f"binary.{prefix}.conv1_signed_acc", block.conv1, input_state.value)
        first_q, first_shift, _ = self._affine_integer(
            f"{prefix}.first_affine", f"{prefix}.aff1", block.aff1, first_q
        )
        first_state = self._state(
            f"{prefix}.first_affine", first_q.to(torch.float32) * (2.0 ** (-first_shift))
        )
        shortcut_state = self._stage_shortcut(record, input_state)
        x1 = self._residual_add(f"{prefix}.x1", first_state, shortcut_state)
        pre_value = F.hardtanh(x1.value, inplace=False)
        pre_second = self._state(f"{prefix}.pre_bconv2_hardtanh", pre_value)
        second_q = self._binary_conv(f"binary.{prefix}.conv2_signed_acc", block.conv2, pre_second.value)
        second_q, second_shift, _ = self._affine_integer(
            f"{prefix}.second_affine", f"{prefix}.aff2", block.aff2, second_q
        )
        second_state = self._state(
            f"{prefix}.second_affine", second_q.to(torch.float32) * (2.0 ** (-second_shift))
        )
        second_add = self._residual_add(f"{prefix}.second_add", second_state, x1)
        output = self._qrprelu_integer(record, second_add)
        return self._state(f"{prefix}.qrprelu_output", output)

    def _gap_and_head(self, state: _IntegerState):
        if self.h2_plan is None:
            gap = F.avg_pool2d(state.value, state.value.size(3))
            features = gap.view(gap.size(0), -1)
            self._record("gap.reference_output", features, kind="reference_gap")
            head = self._state("head.affine_output", self.model.head_affine(features))
            return head

        if state.q is None or state.shift is None:
            raise RuntimeError("H2 GAP requires an integer final H1MP state")
        q = state.q.to(torch.int64)
        if q.size(2) != 8 or q.size(3) != 8:
            raise RuntimeError(f"Expected 8x8 GAP map, got {tuple(q.shape)}")
        q_sum = q.reshape(q.size(0), q.size(1), -1).sum(dim=2)
        gap_plan = self.h2_plan["gap"]
        q_sum = self._finite("gap.sum", q_sum, int(gap_plan["sum_bits"]), gap_plan.get("mode", "assert"))
        self.last_gap_sum = q_sum.detach().clone()
        self.last_gap_input_shift = int(state.shift)
        self._record("gap.sum", q_sum, bits=int(gap_plan["sum_bits"]), kind="gap_64_sum_accumulator")
        deferred = gap_plan.get("division_mode") == "deferred_scale_metadata"
        if deferred:
            # Keep the exact sum.  GAP /64 is represented only by the output
            # scale, so no integer right-shift rounding is introduced.
            q_gap = q_sum
            gap_output_bits = int(gap_plan["output_bits"])
            gap_output_shift = int(state.shift) + int(gap_plan.get("scale_shift_increment", 6))
            gap_kind = "gap_div64_deferred_scale_metadata"
        else:
            q_gap = arithmetic_right_shift_round(q_sum, 6)
            gap_output_bits = int(gap_plan["output_bits"])
            gap_output_shift = int(state.shift)
            gap_kind = "gap_div64_arithmetic_shift"
        q_gap = self._finite("gap.output", q_gap, int(gap_plan["output_bits"]), gap_plan.get("output_mode", "assert"))
        self._record("gap.output", q_gap, bits=gap_output_bits, kind=gap_kind)
        gap_state = _IntegerState(
            value=q_gap.to(torch.float32) * (2.0 ** (-gap_output_shift)),
            q=q_gap,
            shift=gap_output_shift,
            node_name="gap.output",
        )
        head_q, head_shift, _ = self._affine_integer(
            "head.affine_output",
            "head_affine",
            self.model.head_affine,
            gap_state.q,
            conv_shift=gap_state.shift,
            input_bits=int(gap_plan["output_bits"]),
        )
        head = self._state(
            "head.affine_output", head_q.to(torch.float32) * (2.0 ** (-head_shift))
        )
        return head

    def _fc_integer(self, head: _IntegerState):
        if self.h2_plan is None:
            return self.model.linear(head.value)
        if head.q is None or head.shift is None:
            raise RuntimeError("H2 FC requires integer head state")
        fc_plan = self.h2_plan["fc"]
        weights = self.model.linear.effective_weight().detach()
        exponents = torch.round(torch.log2(weights.abs())).to(torch.int64)
        signs = torch.where(weights >= 0, torch.ones_like(weights), -torch.ones_like(weights)).to(torch.int64)
        common_shift = int(fc_plan["common_shift"])
        deltas = common_shift + exponents - int(head.shift)
        if bool((deltas < 0).any()):
            raise RuntimeError("FC common scale is too small for a shifted product")
        q_in = head.q.to(torch.int64).unsqueeze(1)
        powers = torch.tensor(
            [1 << int(value) for value in deltas.flatten().tolist()],
            dtype=torch.int64,
            device=head.q.device,
        ).view(1, weights.size(0), weights.size(1))
        term = q_in * signs.to(head.q.device).view(1, weights.size(0), weights.size(1)) * powers
        self._record("fc.shifted_products", term, kind="signed_pow2_fc_shifted_products")
        # term is [batch, classes, 64 input products].
        acc = term.sum(dim=2)
        acc = self._finite("fc.accumulator", acc, int(fc_plan["accumulator_bits"]), fc_plan.get("accumulator_mode", "assert"))
        self._record("fc.accumulator", acc, bits=int(fc_plan["accumulator_bits"]), kind="fc_per_class_accumulator")

        bias = _bias_record(self.h0_info, "linear")
        bias_shift = int(bias["shift"])
        out_shift = int(fc_plan["bias_add_shift"])
        acc_aligned = requantize_integer(acc, common_shift, out_shift)
        q_bias = torch.tensor(bias["q_values"], dtype=torch.int64, device=head.q.device)
        q_bias = q_bias << (out_shift - bias_shift)
        logits = acc_aligned + q_bias.view(1, -1)
        logits = self._finite("fc.bias_add", logits, int(fc_plan["bias_add_bits"]), fc_plan.get("bias_add_mode", "assert"))
        self._record("fc.bias_add", logits, bits=int(fc_plan["bias_add_bits"]), kind="fc_int6_bias_add")
        final = self._finite("final_logits", logits, int(fc_plan["final_logits_bits"]), fc_plan.get("logit_mode", "assert"))
        self._record("final_logits", final, bits=int(fc_plan["final_logits_bits"]), kind="final_integer_logits")
        return final.to(torch.float32) * (2.0 ** (-out_shift))

    def forward(self, x):
        encoded = self.model.thermo(x)
        if self.h2_plan is None:
            stem_acc = self.model.conv1(encoded)
            self._record("binary.stem.signed_acc", stem_acc, kind="reference_binary_accumulator")
            stem = F.hardtanh(self.model.stem_affine(stem_acc), inplace=False)
        else:
            stem_q = self._binary_conv("binary.stem.signed_acc", self.model.conv1, encoded, stem=True)
            stem_q, stem_shift, _ = self._affine_integer(
                "stem.post_hardtanh", "stem_affine", self.model.stem_affine, stem_q
            )
            stem = F.hardtanh(stem_q.to(torch.float32) * (2.0 ** (-stem_shift)), inplace=False)
        state = self._state("stem.post_hardtanh", stem)
        for record in self._blocks:
            state = self._run_block(record, state)
        head = self._gap_and_head(state)
        return self._fc_integer(head)

    # H1ModelAdapter defines __call__ = forward at class creation time.  Bind
    # it again here so adapter(x) dispatches to this subclass's finite-width
    # stem/GAP/FC path rather than the parent implementation.
    __call__ = forward


def _load_h1mp_plan(h1mp_results_json: str) -> dict:
    data = json.loads(_resolve(h1mp_results_json).read_text(encoding="utf-8"))
    if data.get("status") != "COMPLETE":
        raise RuntimeError("H1MP results are not COMPLETE")
    plan = data.get("final_plan", {})
    required = ("bits_by_node", "shift_by_node", "policy_by_node")
    if any(key not in plan for key in required):
        raise RuntimeError("H1MP final plan is missing frozen bits/shifts/policies")
    if not (len(plan["bits_by_node"]) == len(plan["shift_by_node"]) == len(plan["policy_by_node"]) == 58):
        raise RuntimeError("H1MP frozen plan does not contain exactly 58 nodes")
    if len(data.get("quant_nodes", [])) != 58:
        raise RuntimeError("H1MP quant node contract is not 58 nodes")
    return data


def _load_h0_info(h0_results_json: str) -> dict:
    data = json.loads(_resolve(h0_results_json).read_text(encoding="utf-8"))
    matches = [item for item in data.get("results", []) if int(item.get("bits", -1)) == 6]
    if len(matches) != 1 or len(matches[0].get("layers", {})) != 21:
        raise RuntimeError("H0 INT6 export must contain exactly 21 bias targets")
    return matches[0]["layers"]


def load_r8b_h1mp_model(
    source_r8b_checkpoint: str,
    h0_results_json: str,
    h1mp_results_json: str,
    device,
):
    from tools.run_h1_integer_sweep import apply_h0_int6_biases, load_r8b

    h1mp = _load_h1mp_plan(h1mp_results_json)
    h0_info = _load_h0_info(h0_results_json)
    model, _ = load_r8b(_resolve(source_r8b_checkpoint), device)
    apply_h0_int6_biases(model, _resolve(h0_results_json))
    fake = SelectiveIntegerFakeQuant(
        node_bits={name: int(value) for name, value in h1mp["final_plan"]["bits_by_node"].items()},
        node_shifts={name: int(value) for name, value in h1mp["final_plan"]["shift_by_node"].items()},
    )
    adapter = H1ModelAdapter(model)
    adapter.set_fake_quant(fake)
    model._h1_adapter = adapter
    model._h1_fake_quant = fake
    model._h1mp_results = h1mp
    model._h0_info = h0_info
    model.eval()
    return model


def build_h2_loaders(seed=20260914, calibration_samples=10000, search_validation_samples=10000):
    from .h1_search_workspace import build_h1_search_loaders

    return build_h1_search_loaders(
        seed=int(seed),
        calibration_samples=int(calibration_samples),
        search_validation_samples=int(search_validation_samples),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    predictor = getattr(model, "_h2_adapter", None)
    if predictor is None:
        predictor = getattr(model, "_h1_adapter", model)
    if isinstance(predictor, (H1ModelAdapter, H2TraceAdapter)):
        predictor.model.eval()
    else:
        predictor.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += int((predictor(x).argmax(1) == y).sum().item())
        total += int(y.size(0))
    return 100.0 * correct / max(total, 1)


def build_h2_trace_adapter(model):
    if not hasattr(model, "_h1_fake_quant") or not hasattr(model, "_h0_info"):
        raise RuntimeError("load_r8b_h1mp_model must be called before build_h2_trace_adapter")
    adapter = H2TraceAdapter(model, model._h1_fake_quant, model._h0_info)
    model._h2_adapter = adapter
    return adapter


def _binary_plan(binary_exact: dict) -> dict:
    result = {}
    for key, item in binary_exact.items():
        result[key] = dict(item)
        result[key]["mode"] = "assert"
        result[key]["overflow_policy"] = "assert_zero_overflow"
    return result


def _affine_width_record(adapter: H2TraceAdapter, node_name: str, module_name: str, module, conv_terms: int, conv_shift: int = 0, input_bits: Optional[int] = None, input_shift: int = 0):
    exponents, signs = _module_exponents(module)
    bias = _bias_record(adapter.h0_info, module_name)
    bias_shift = int(bias["shift"])
    exponent_min = int(exponents.min().item())
    common_shift = max(bias_shift, int(input_shift) - exponent_min)
    ranges = []
    qmin, qmax = signed_range(int(input_bits)) if input_bits is not None else (-int(conv_terms), int(conv_terms))
    for index, (exp, sign) in enumerate(zip(exponents.tolist(), signs.tolist())):
        term_delta = common_shift + int(exp) - int(input_shift)
        term_lo, term_hi = _shifted_range(qmin, qmax, term_delta)
        if int(sign) < 0:
            term_lo, term_hi = -term_hi, -term_lo
        bq = int(bias["q_values"][index]) << (common_shift - bias_shift)
        ranges.append((term_lo + bq, term_hi + bq))
    lo = min(item[0] for item in ranges)
    hi = max(item[1] for item in ranges)
    bits = signed_bits_for_range(lo, hi)
    return {
        "node": node_name,
        "module": module_name,
        "bits": int(bits),
        "min": int(lo),
        "max": int(hi),
        "shift": int(common_shift),
        "bias_shift": int(bias_shift),
        "exponents": [int(value) for value in exponents.tolist()],
        "signs": [int(value) for value in signs.tolist()],
        "mathematical_basis": "worst-case signed input range plus exact INT6 bias range",
        "mode": "assert",
    }


def _residual_width_record(adapter: H2TraceAdapter, add_name: str, branch_a_name: str, branch_b_name: str, out_bits: int, out_shift: int, transition=False):
    a_bits = adapter._h1_bits(branch_a_name)
    a_shift = adapter._h1_shift(branch_a_name)
    b_bits = adapter._h1_bits(branch_b_name)
    b_shift = adapter._h1_shift(branch_b_name)
    a_lo, a_hi = _signed_align_range(a_bits, a_shift, out_shift)
    b_lo, b_hi = _signed_align_range(b_bits, b_shift, out_shift)
    lo, hi = a_lo + b_lo, a_hi + b_hi
    bits = signed_bits_for_range(lo, hi)
    return {
        "add_node": add_name,
        "branch_a_node": branch_a_name,
        "branch_b_node": branch_b_name,
        "branch_a_bits": int(a_bits),
        "branch_b_bits": int(b_bits),
        "branch_a_shift": int(a_shift),
        "branch_b_shift": int(b_shift),
        "output_h1_bits": int(out_bits),
        "output_shift": int(out_shift),
        "aligned_min": int(lo),
        "aligned_max": int(hi),
        "bits": int(bits),
        "mode": "assert",
        "mathematical_basis": "exact aligned branch endpoint sum",
    }


def _qrprelu_width_record(adapter: H2TraceAdapter, record) -> dict:
    prefix = record["name"]
    node = f"{prefix}.qrprelu_output"
    block = record["block"]
    input_node = f"{prefix}.second_add"
    in_bits = adapter._h1_bits(input_node)
    in_shift = adapter._h1_shift(input_node)
    a_exp = block.post_act.integer_shift_exponents().detach().cpu().tolist()
    qmin, qmax = signed_range(in_bits)
    inner_shift = max(in_shift, H2TraceAdapter.QRP_PARAM_SHIFT)
    xi1_lo = min(_round_int_scalar(float(v), inner_shift) for v in block.post_act.xi1.detach().cpu().tolist())
    xi1_hi = max(_round_int_scalar(float(v), inner_shift) for v in block.post_act.xi1.detach().cpu().tolist())
    in_lo, in_hi = _shifted_range(qmin, qmax, inner_shift - in_shift)
    inner_lo, inner_hi = in_lo + xi1_lo, in_hi + xi1_hi
    xi2_lo = min(_round_int_scalar(float(v), inner_shift) for v in block.post_act.xi2.detach().cpu().tolist())
    xi2_hi = max(_round_int_scalar(float(v), inner_shift) for v in block.post_act.xi2.detach().cpu().tolist())
    out_shift = max(inner_shift, inner_shift - min(a_exp))
    neg_ranges = []
    for exp in a_exp:
        delta = out_shift + int(exp) - inner_shift
        nlo, nhi = _shifted_range(inner_lo, inner_hi, delta)
        xlo, xhi = _shifted_range(xi2_lo, xi2_hi, out_shift - inner_shift)
        neg_ranges.append((nlo + xlo, nhi + xhi))
    pos_lo, pos_hi = _shifted_range(qmin, qmax, out_shift - in_shift)
    out_lo = min(pos_lo, *(x[0] for x in neg_ranges))
    out_hi = max(pos_hi, *(x[1] for x in neg_ranges))
    return {
        "node": node,
        "input_bits": int(in_bits),
        "input_shift": int(in_shift),
        "param_shift": int(inner_shift),
        "output_shift": int(out_shift),
        "slope_exponents": [int(value) for value in a_exp],
        "inner_min": int(inner_lo),
        "inner_max": int(inner_hi),
        "inner_bits": int(signed_bits_for_range(inner_lo, inner_hi)),
        "min": int(out_lo),
        "max": int(out_hi),
        "bits": int(signed_bits_for_range(out_lo, out_hi)),
        "mode": "assert",
        "mathematical_basis": "signed input range plus quantized xi1/xi2 at 24 fractional bits",
    }


def _fc_plan(adapter: H2TraceAdapter, input_bits: int, input_shift: int) -> dict:
    weights = adapter.model.linear.effective_weight().detach().cpu()
    exponents = torch.round(torch.log2(weights.abs())).to(torch.int64)
    signs = torch.where(weights >= 0, torch.ones_like(weights), -torch.ones_like(weights)).to(torch.int64)
    common_shift = max(int(input_shift - int(value)) for value in exponents.flatten().tolist())
    left_shifts = []
    signs_by_class = []
    shifts_by_class = []
    for cls in range(weights.size(0)):
        shifts = [common_shift + int(value) - int(input_shift) for value in exponents[cls].tolist()]
        signs_row = [int(value) for value in signs[cls].tolist()]
        shifts_by_class.append(shifts)
        signs_by_class.append(signs_row)
        left_shifts.extend(shifts)
        signs_by_class.append(signs_row)
    class_ranges = exact_fc_acc_width_per_class(input_bits, shifts_by_class, signs_by_class[::2])
    acc_bits = max(item["bits"] for item in class_ranges)
    bias = _bias_record(adapter.h0_info, "linear")
    bias_shift = int(bias["shift"])
    bias_add_shift = max(common_shift, bias_shift)
    acc_ranges = [(item["min"], item["max"]) for item in class_ranges]
    aligned_ranges = [_shifted_range(lo, hi, bias_add_shift - common_shift) for lo, hi in acc_ranges]
    bias_values = [int(value) << (bias_add_shift - bias_shift) for value in bias["q_values"]]
    final_ranges = [(lo + bias_values[i], hi + bias_values[i]) for i, (lo, hi) in enumerate(aligned_ranges)]
    final_lo = min(item[0] for item in final_ranges)
    final_hi = max(item[1] for item in final_ranges)
    bias_add_bits = signed_bits_for_range(final_lo, final_hi)
    return {
        "input_bits": int(input_bits),
        "input_shift": int(input_shift),
        "common_shift": int(common_shift),
        "weight_exponents": [[int(value) for value in row] for row in exponents.tolist()],
        "weight_signs": [[int(value) for value in row] for row in signs.tolist()],
        "left_shifts_by_class": shifts_by_class,
        "per_class_worst_case": class_ranges,
        # H2 fixes the FC accumulator and final logits at INT24.  The exact
        # worst-case range remains recorded for audit, but these widths are
        # not optimization variables in H2A-v2.
        "accumulator_bits": 24,
        "accumulator_min": int(min(item["min"] for item in class_ranges)),
        "accumulator_max": int(max(item["max"] for item in class_ranges)),
        "accumulator_mode": "assert",
        "bias_shift": int(bias_shift),
        "bias_add_shift": int(bias_add_shift),
        "bias_add_min": int(final_lo),
        "bias_add_max": int(final_hi),
        "bias_add_bits": 24,
        "bias_add_mode": "assert",
        "final_logits_bits": 24,
        "logit_mode": "assert",
        "mathematical_basis": "per-class worst-case signed-pow2 product sum plus fixed INT6 FC bias",
    }


@torch.no_grad()
def derive_exact_width_plan(
    model,
    adapter,
    calibration_loader,
    device,
    binary_exact,
    gap_deferred_scaling: bool = False,
):
    adapter.set_h2_plan(None)
    adapter.set_trace_enabled(True)
    seen = 0
    for x, _ in calibration_loader:
        adapter(x.to(device, non_blocking=True))
        seen += int(x.size(0))
    calibration_ranges = adapter.trace.export()
    adapter.set_trace_enabled(False)

    affine = {}
    affine["stem.post_hardtanh"] = _affine_width_record(
        adapter, "stem.post_hardtanh", "stem_affine", model.stem_affine, 3 * 3 * 96
    )
    for item in adapter._blocks:
        prefix = item["name"]
        block = item["block"]
        affine[f"{prefix}.first_affine"] = _affine_width_record(
            adapter, f"{prefix}.first_affine", f"{prefix}.aff1", block.aff1, 3 * 3 * block.conv1.in_channels
        )
        affine[f"{prefix}.second_affine"] = _affine_width_record(
            adapter, f"{prefix}.second_affine", f"{prefix}.aff2", block.aff2, 3 * 3 * block.conv2.in_channels
        )
    last_node = adapter._blocks[-1]["name"] + ".qrprelu_output"
    last_bits = adapter._h1_bits(last_node)
    last_shift = adapter._h1_shift(last_node)
    gap_input_bits = gap_exact_width(last_bits, 64) if gap_deferred_scaling else last_bits
    gap_input_shift = last_shift + 6 if gap_deferred_scaling else last_shift
    affine["head.affine_output"] = _affine_width_record(
        adapter,
        "head.affine_output",
        "head_affine",
        model.head_affine,
        0,
        input_bits=gap_input_bits,
        input_shift=gap_input_shift,
    )

    residuals = {}
    input_node = "stem.post_hardtanh"
    for item in adapter._blocks:
        prefix = item["name"]
        first = f"{prefix}.first_affine"
        x1 = f"{prefix}.x1"
        shortcut = f"{prefix}.stage_shortcut" if item["transition"] else input_node
        residuals[x1] = _residual_width_record(
            adapter, x1, first, shortcut, adapter._h1_bits(x1), adapter._h1_shift(x1), item["transition"]
        )
        second = f"{prefix}.second_affine"
        second_add = f"{prefix}.second_add"
        residuals[second_add] = _residual_width_record(
            adapter, second_add, second, x1, adapter._h1_bits(second_add), adapter._h1_shift(second_add)
        )
        input_node = f"{prefix}.qrprelu_output"

    qrprelu = {
        item["name"] + ".qrprelu_output": _qrprelu_width_record(adapter, item)
        for item in adapter._blocks
    }
    gap_sum_bits = gap_exact_width(last_bits, 64)
    gap = {
        "elements": 64,
        "input_bits": int(last_bits),
        "input_shift": int(last_shift),
        "sum_bits": int(gap_sum_bits),
        "sum_min": int(64 * signed_range(last_bits)[0]),
        "sum_max": int(64 * signed_range(last_bits)[1]),
        "sum_mode": "assert",
        "divide_by": 64,
        "division_mode": "deferred_scale_metadata" if gap_deferred_scaling else "arithmetic_right_shift",
        "arithmetic_right_shift_applied": not gap_deferred_scaling,
        "scale_shift_increment": 6,
        "arithmetic_shift": None if gap_deferred_scaling else 6,
        "rounding": "none; /64 is represented by scale metadata" if gap_deferred_scaling else "symmetric round-to-nearest, ties away from zero",
        "output_bits": int(gap_sum_bits if gap_deferred_scaling else last_bits),
        "output_shift": int(last_shift + 6 if gap_deferred_scaling else last_shift),
        "output_mode": "assert",
    }
    head_bits = adapter._h1_bits("head.affine_output")
    head_shift = adapter._h1_shift("head.affine_output")
    fc = _fc_plan(adapter, head_bits, head_shift)
    return {
        "name": "H2A-v2 GAP Deferred Scaling" if gap_deferred_scaling else "H2A exact/conservative finite-width anchor",
        "binary_convolution": _binary_plan(binary_exact),
        "affine_intermediates": affine,
        "residuals": residuals,
        "qrprelu": qrprelu,
        "gap": gap,
        "fc": fc,
        "h1mp_frozen_node_count": 58,
        "h1mp_frozen_residual_count": 18,
        "calibration_samples_seen": int(seen),
        "calibration_ranges": calibration_ranges,
        "all_h2a_modes": "mathematical exact/conservative widths; no deliberate narrowing",
        "gap_deferred_scaling": bool(gap_deferred_scaling),
    }


@torch.no_grad()
def evaluate_exact_width_plan(model, adapter, plan, loader, device):
    adapter.set_h2_plan(plan)
    adapter.set_trace_enabled(True)
    accuracy = evaluate(model, loader, device)
    adapter.set_trace_enabled(False)
    trace = adapter.trace.export()
    overflow = dict(adapter.runtime_overflow)
    saturation = dict(adapter.runtime_saturation)
    if adapter._fake_quant is not None:
        saturation["h1mp_boundaries"] = int(adapter._fake_quant.total_saturation())
    return {
        "accuracy": float(accuracy),
        "trace": trace,
        "overflow_by_node": overflow,
        "saturation_by_node": saturation,
        "observed_by_node": copy.deepcopy(adapter.runtime_observed),
        "overflow_total": int(sum(overflow.values())),
        "saturation_total": int(sum(saturation.values(), 0) if saturation else 0),
        "residual_alignment": adapter.verify_integer_residual_alignment(
            shift_map=adapter._fake_quant.shift_map,
            bits_or_map=adapter._fake_quant.node_bits,
        ),
    }


def _try_width(
    model,
    adapter,
    plan,
    field: str,
    candidate: int,
    search_loader,
    device,
    baseline_accuracy: float,
    target_drop_pp: float,
    max_drop_pp: float,
):
    trial = copy.deepcopy(plan)
    if field == "gap_sum_bits":
        trial["gap"]["sum_bits"] = int(candidate)
        trial["gap"]["sum_mode"] = "saturate"
    elif field == "fc_accumulator_bits":
        trial["fc"]["accumulator_bits"] = int(candidate)
        trial["fc"]["accumulator_mode"] = "saturate"
    elif field == "final_logits_bits":
        trial["fc"]["final_logits_bits"] = int(candidate)
        trial["fc"]["logit_mode"] = "saturate"
    else:
        raise ValueError(field)
    result = evaluate_exact_width_plan(model, adapter, trial, search_loader, device)
    drop = baseline_accuracy - result["accuracy"]
    accepted = drop <= float(target_drop_pp)
    return trial, result, float(drop), accepted, max_drop_pp


@torch.no_grad()
def search_remaining_widths(
    model,
    adapter,
    base_plan,
    calibration_loader,
    search_loader,
    device,
    target_drop_pp=0.10,
    max_drop_pp=0.30,
):
    # H2A has already fixed every shift and all exact nodes.  This validation
    # baseline is TRAIN-derived and is the only baseline used by H2B.
    base_validation = evaluate_exact_width_plan(model, adapter, base_plan, search_loader, device)
    baseline_accuracy = float(base_validation["accuracy"])
    current_plan = copy.deepcopy(base_plan)
    current_result = base_validation
    history = [{
        "action": "initial_h2a",
        "accuracy": baseline_accuracy,
        "drop_pp": 0.0,
        "widths": {
            "gap_sum_bits": int(current_plan["gap"]["sum_bits"]),
            "fc_accumulator_bits": int(current_plan["fc"]["accumulator_bits"]),
            "final_logits_bits": int(current_plan["fc"]["final_logits_bits"]),
        },
    }]
    search_specs = [
        ("gap_sum_bits", int(current_plan["gap"]["sum_bits"])),
        ("fc_accumulator_bits", int(current_plan["fc"]["accumulator_bits"])),
        ("final_logits_bits", int(current_plan["fc"]["final_logits_bits"])),
    ]
    for field, safe_width in search_specs:
        for candidate in range(safe_width - 1, 1, -1):
            trial, result, drop, accepted, _ = _try_width(
                model, adapter, current_plan, field, candidate, search_loader, device,
                baseline_accuracy, target_drop_pp, max_drop_pp,
            )
            history.append({
                "action": "accept" if accepted else "reject",
                "field": field,
                "from_bits": int(safe_width if not accepted else current_plan["gap"]["sum_bits"] if field == "gap_sum_bits" else current_plan["fc"]["accumulator_bits"] if field == "fc_accumulator_bits" else current_plan["fc"]["final_logits_bits"]),
                "to_bits": int(candidate),
                "accuracy": float(result["accuracy"]),
                "drop_pp": float(drop),
                "overflow_total": int(result["overflow_total"]),
            })
            if accepted:
                current_plan = trial
                current_result = result
                if field == "gap_sum_bits":
                    safe_width = candidate
                elif field == "fc_accumulator_bits":
                    safe_width = candidate
                else:
                    safe_width = candidate
            else:
                # A one-bit rollback is the stopping point for this class.
                break
    final_validation = evaluate_exact_width_plan(model, adapter, current_plan, search_loader, device)
    return {
        "status": "COMPLETE",
        "validation_baseline_accuracy": baseline_accuracy,
        "target_validation_drop_pp": float(target_drop_pp),
        "absolute_max_validation_drop_pp": float(max_drop_pp),
        "history": history,
        "final_validation": final_validation,
        "final_validation_drop_pp": baseline_accuracy - final_validation["accuracy"],
        "final_plan": current_plan,
        "calibration_source": "TRAIN calibration subset only; no new TEST-derived decisions",
    }


def _h2b_trace_snapshot(result: dict, prefix: str, suffixes: Iterable[str]) -> dict:
    snapshot = {}
    for suffix in suffixes:
        name = f"{prefix}.{suffix}"
        item = result.get("trace", {}).get(name)
        if item is not None:
            snapshot[suffix] = {
                "min": item.get("min"),
                "max": item.get("max"),
                "absmax": item.get("absmax"),
                "count": item.get("count"),
            }
    return snapshot


def _h2b_finite_node_snapshot(
    result: dict,
    calibration_result: dict,
    node_names: Iterable[str],
    safe_range: Tuple[int, int],
    width_by_node: Dict[str, int],
) -> dict:
    nodes = {}
    for node in node_names:
        observed = result.get("observed_by_node", {}).get(node, {})
        cal_observed = calibration_result.get("observed_by_node", {}).get(node, {})
        trace = result.get("trace", {}).get(node, {})
        cal_trace = calibration_result.get("trace", {}).get(node, {})
        nodes[node] = {
            "width": int(width_by_node[node]),
            "mathematical_safe_range": [int(safe_range[0]), int(safe_range[1])],
            "calibration_observed_range": [
                int(cal_observed.get("min", cal_trace.get("min", 0))),
                int(cal_observed.get("max", cal_trace.get("max", 0))),
            ],
            "validation_observed_range": [
                int(observed.get("min", trace.get("min", 0))),
                int(observed.get("max", trace.get("max", 0))),
            ],
            "validation_saturation": int(result.get("saturation_by_node", {}).get(node, 0)),
            "validation_overflow": int(result.get("overflow_by_node", {}).get(node, 0)),
        }
    return nodes


def _h2b_candidate_record(
    family: str,
    candidate: int,
    previous_accepted_width: int,
    result: dict,
    current_plan: dict,
    base_plan: dict,
    validation_baseline: float,
    calibration_result: dict,
    target_drop_pp: float,
    max_drop_pp: float,
) -> dict:
    drop = float(validation_baseline - result["accuracy"])
    accepted = drop <= float(max_drop_pp)
    preferred = drop <= float(target_drop_pp)
    record = {
        "family": family,
        "candidate_width": int(candidate),
        "previous_accepted_width": int(previous_accepted_width),
        "action": "accept" if accepted else "reject_rollback",
        "preferred": bool(preferred),
        "accuracy": float(result["accuracy"]),
        "delta_vs_h2a_validation_pp": drop,
        "target_drop_pp": float(target_drop_pp),
        "maximum_drop_pp": float(max_drop_pp),
        "overflow_total": int(result["overflow_total"]),
        "saturation_total_including_h1mp": int(result["saturation_total"]),
        "binary_overflow_total": 0,
    }
    if family == "inner":
        nodes = []
        width_by_node = {}
        safe_by_node = {}
        internal = {}
        for name, item in base_plan["qrprelu"].items():
            prefix = name.rsplit(".qrprelu_output", 1)[0]
            node = f"qrprelu.{prefix}.inner_add"
            nodes.append(node)
            width_by_node[node] = min(int(item["inner_bits"]), int(candidate))
            safe_by_node[node] = (int(item["inner_min"]), int(item["inner_max"]))
            internal[node] = {
                "calibration": _h2b_trace_snapshot(
                    calibration_result,
                    f"qrprelu.{prefix}",
                    ("shifted_input", "offset_xi1", "offset_xi2", "inner_add"),
                ),
                "validation": _h2b_trace_snapshot(
                    result,
                    f"qrprelu.{prefix}",
                    ("shifted_input", "offset_xi1", "offset_xi2", "inner_add"),
                ),
            }
        record["nodes"] = {
            node: _h2b_finite_node_snapshot(
                result,
                calibration_result,
                (node,),
                safe_by_node[node],
                width_by_node,
            )[node]
            for node in nodes
        }
        record["internal_trace"] = internal
    elif family == "output":
        nodes = []
        width_by_node = {}
        safe_by_node = {}
        internal = {}
        for name, item in base_plan["qrprelu"].items():
            prefix = name.rsplit(".qrprelu_output", 1)[0]
            for suffix in ("negative_branch", "output"):
                node = f"qrprelu.{prefix}.{suffix}"
                nodes.append(node)
                width_by_node[node] = min(int(item["bits"]), int(candidate))
                safe_by_node[node] = (int(item["min"]), int(item["max"]))
            internal[f"qrprelu.{prefix}"] = {
                "calibration": _h2b_trace_snapshot(
                    calibration_result,
                    f"qrprelu.{prefix}",
                    ("shifted_input", "offset_xi1", "offset_xi2", "inner_add", "negative_branch", "output_before_h1mp", "output"),
                ),
                "validation": _h2b_trace_snapshot(
                    result,
                    f"qrprelu.{prefix}",
                    ("shifted_input", "offset_xi1", "offset_xi2", "inner_add", "negative_branch", "output_before_h1mp", "output"),
                ),
            }
        record["nodes"] = {
            node: _h2b_finite_node_snapshot(
                result,
                calibration_result,
                (node,),
                safe_by_node[node],
                width_by_node,
            )[node]
            for node in nodes
        }
        record["internal_trace"] = internal
    else:
        finite_name, safe_range, width = {
            "fc_accumulator": (
                "fc.accumulator",
                (int(base_plan["fc"]["accumulator_min"]), int(base_plan["fc"]["accumulator_max"])),
                int(current_plan["fc"]["accumulator_bits"]),
            ),
            "final_logits": (
                "final_logits",
                (int(base_plan["fc"]["bias_add_min"]), int(base_plan["fc"]["bias_add_max"])),
                int(current_plan["fc"]["final_logits_bits"]),
            ),
            "gap": (
                "gap.sum",
                (int(base_plan["gap"]["sum_min"]), int(base_plan["gap"]["sum_max"])),
                int(current_plan["gap"]["sum_bits"]),
            ),
        }[family]
        record["nodes"] = _h2b_finite_node_snapshot(
            result,
            calibration_result,
            (finite_name,),
            safe_range,
            {finite_name: int(width)},
        )
    return record


def _apply_h2b_family(plan: dict, family: str, candidate: int) -> dict:
    trial = copy.deepcopy(plan)
    if family == "inner":
        for item in trial["qrprelu"].values():
            item["inner_bits"] = min(int(item["inner_bits"]), int(candidate))
            item["mode"] = "saturate"
    elif family == "output":
        for item in trial["qrprelu"].values():
            item["bits"] = min(int(item["bits"]), int(candidate))
            item["mode"] = "saturate"
    elif family == "fc_accumulator":
        trial["fc"]["accumulator_bits"] = int(candidate)
        trial["fc"]["accumulator_mode"] = "saturate"
    elif family == "final_logits":
        trial["fc"]["final_logits_bits"] = int(candidate)
        trial["fc"]["logit_mode"] = "saturate"
    elif family == "gap":
        trial["gap"]["sum_bits"] = int(candidate)
        trial["gap"]["sum_mode"] = "saturate"
    else:
        raise ValueError(family)
    return trial


@torch.no_grad()
def search_h2b_widths(
    model,
    adapter,
    base_plan,
    calibration_loader,
    search_loader,
    device,
    target_drop_pp=0.10,
    max_drop_pp=0.30,
):
    """Optimize H2A-v2 widths using only disjoint TRAIN-derived data."""
    if base_plan.get("gap", {}).get("division_mode") != "deferred_scale_metadata":
        raise RuntimeError("H2B requires the PASS H2A-v2 deferred GAP plan")
    base_validation = evaluate_exact_width_plan(model, adapter, base_plan, search_loader, device)
    calibration_result = evaluate_exact_width_plan(model, adapter, base_plan, calibration_loader, device)
    baseline_accuracy = float(base_validation["accuracy"])
    current_plan = copy.deepcopy(base_plan)
    history = []
    accepted_widths = {}
    family_specs = [
        ("inner", max(int(item["inner_bits"]) for item in base_plan["qrprelu"].values())),
        ("output", max(int(item["bits"]) for item in base_plan["qrprelu"].values())),
        ("fc_accumulator", int(base_plan["fc"]["accumulator_bits"])),
        ("final_logits", int(base_plan["fc"]["final_logits_bits"])),
        ("gap", int(base_plan["gap"]["sum_bits"])),
    ]
    for family, safe_width in family_specs:
        accepted_width = int(safe_width)
        family_records = []
        for candidate in range(safe_width - 1, 1, -1):
            trial = _apply_h2b_family(current_plan, family, candidate)
            result = evaluate_exact_width_plan(model, adapter, trial, search_loader, device)
            record = _h2b_candidate_record(
                family,
                candidate,
                accepted_width,
                result,
                trial,
                base_plan,
                baseline_accuracy,
                calibration_result,
                target_drop_pp,
                max_drop_pp,
            )
            family_records.append(record)
            history.append(record)
            if record["action"] == "accept":
                current_plan = trial
                accepted_width = int(candidate)
            else:
                break
        accepted_widths[family] = int(accepted_width)
        history.append({
            "family": family,
            "action": "freeze",
            "accepted_width": int(accepted_width),
            "candidate_count": len(family_records),
            "last_candidate_action": family_records[-1]["action"] if family_records else "none",
        })
    final_validation = evaluate_exact_width_plan(model, adapter, current_plan, search_loader, device)
    return {
        "status": "COMPLETE",
        "validation_baseline_accuracy": baseline_accuracy,
        "target_validation_drop_pp": float(target_drop_pp),
        "absolute_max_validation_drop_pp": float(max_drop_pp),
        "history": history,
        "final_validation": final_validation,
        "final_validation_drop_pp": float(baseline_accuracy - final_validation["accuracy"]),
        "final_plan": current_plan,
        "accepted_widths": accepted_widths,
        "calibration_result": calibration_result,
        "calibration_source": "CIFAR-10 TRAIN calibration subset only",
        "search_validation_source": "non-overlapping CIFAR-10 TRAIN search-validation subset only",
        "official_test_used_for_search": False,
        "family_order": [item[0] for item in family_specs],
    }
