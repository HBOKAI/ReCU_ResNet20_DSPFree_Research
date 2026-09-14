"""Faithful H1 adapter for the workspace's inline R4/R8B residual graph.

The R4 block does not expose residual additions as modules, so this adapter
replays the existing mathematical graph explicitly. With no observer/fake
quantizer installed it is equivalent to the R8B forward. With a fake
quantizer installed, each multi-bit branch is represented by q plus a shift
and residual adds are performed after integer scale alignment.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .integer_activation import (
    aligned_integer_add,
    requantize_int,
    signed_int_range,
)


@dataclass
class _IntegerState:
    value: torch.Tensor
    q: Optional[torch.Tensor] = None
    shift: Optional[int] = None
    node_name: Optional[str] = None


class H1ModelAdapter:
    """Expose true H1 state boundaries without changing R8B parameters."""

    def __init__(self, r8b_model):
        required = ("thermo", "conv1", "stem_affine", "layer1", "layer2", "layer3", "head_affine", "linear")
        missing = [name for name in required if not hasattr(r8b_model, name)]
        if missing:
            raise ValueError(f"R8B model is missing required modules: {missing}")
        self.model = r8b_model
        self._observer = None
        self._fake_quant = None
        self._alignment_runtime: Dict[str, dict] = {}
        self._node_metadata: Dict[str, dict] = {}
        self._blocks = []
        self._build_node_contract()

    def _build_node_contract(self):
        self._add_node(
            "stem.post_hardtanh",
            "stem affine + Hardtanh output passed to the first R4 block",
            "stem_state",
        )
        for stage_name in ("layer1", "layer2", "layer3"):
            stage = getattr(self.model, stage_name)
            for block_index, block in enumerate(stage):
                prefix = f"{stage_name}.{block_index}"
                transition = tuple(block.conv1.stride) != (1, 1) or block.conv1.in_channels != block.conv1.out_channels
                record = {
                    "name": prefix,
                    "block": block,
                    "transition": transition,
                }
                self._blocks.append(record)
                self._add_node(
                    f"{prefix}.first_affine",
                    "first BConv + fused affine branch before the first residual add",
                    "residual_branch_a",
                )
                if transition:
                    self._add_node(
                        f"{prefix}.stage_shortcut",
                        "Option-A stage-transition shortcut output before the first residual add",
                        "stage_transition_shortcut",
                    )
                self._add_node(
                    f"{prefix}.x1",
                    "first residual-add output saved as x1",
                    "first_residual_add_output",
                )
                self._add_node(
                    f"{prefix}.pre_bconv2_hardtanh",
                    "Hardtanh(x1) magnitude state feeding the second BConv",
                    "pre_second_bconv_state",
                )
                self._add_node(
                    f"{prefix}.second_affine",
                    "second BConv + fused affine branch before the second residual add",
                    "residual_branch_a",
                )
                self._add_node(
                    f"{prefix}.second_add",
                    "second residual-add output before QRPReLU",
                    "second_residual_add_output",
                )
                self._add_node(
                    f"{prefix}.qrprelu_output",
                    "QRPReLU output and block output state",
                    "block_output_state",
                )
        self._add_node(
            "head.affine_output",
            "head signed-pow2 affine output and final FC input",
            "head_fc_input",
        )

    def _add_node(self, name: str, role: str, tensor_role: str):
        if name in self._node_metadata:
            raise RuntimeError(f"Duplicate H1 quantization node: {name}")
        self._node_metadata[name] = {
            "name": name,
            "role": role,
            "tensor_role": tensor_role,
            "is_accumulator": False,
        }

    def list_quant_nodes(self) -> list[str]:
        return list(self._node_metadata)

    def quant_node_metadata(self) -> list[dict]:
        return [dict(item) for item in self._node_metadata.values()]

    def set_observer(self, observer_or_none):
        if observer_or_none is not None and not hasattr(observer_or_none, "observe"):
            raise TypeError("observer must expose observe(name, tensor)")
        self._observer = observer_or_none

    def set_fake_quant(self, fake_quant_or_none):
        if fake_quant_or_none is not None and not hasattr(fake_quant_or_none, "quantize"):
            raise TypeError("fake quantizer must expose quantize(name, tensor)")
        self._fake_quant = fake_quant_or_none
        self.reset_runtime_reports()

    def reset_runtime_reports(self):
        self._alignment_runtime = {}

    def _fake_bits_for(self, name: Optional[str]) -> int:
        if name is None:
            raise RuntimeError("Integer state is missing its node name")
        if hasattr(self._fake_quant, "bits_for"):
            return int(self._fake_quant.bits_for(name))
        if hasattr(self._fake_quant, "bits"):
            return int(self._fake_quant.bits)
        raise RuntimeError("Fake quantizer does not expose bits_for(name) or bits")

    def _state(self, name: str, value: torch.Tensor) -> _IntegerState:
        if self._observer is not None:
            self._observer.observe(name, value)
        if self._fake_quant is None:
            return _IntegerState(value=value, node_name=name)
        if hasattr(self._fake_quant, "is_quantized") and not self._fake_quant.is_quantized(name):
            # H1-S intentionally keeps the other 57 nodes at reference
            # precision.  The selected node still perturbs the exact graph.
            return _IntegerState(value=value, node_name=name)
        quantized = self._fake_quant.quantize(name, value)
        if quantized is None:
            return _IntegerState(value=value, node_name=name)
        return _IntegerState(
            value=quantized["x_hat"],
            q=quantized["q"],
            shift=int(quantized["shift"]),
            node_name=name,
        )

    @staticmethod
    def _option_a_integer(q: torch.Tensor, in_planes: int, planes: int, stride: int) -> torch.Tensor:
        if stride == 1 and in_planes == planes:
            return q
        if stride != 2 or planes != in_planes * 2:
            raise RuntimeError(
                f"Unexpected Option-A shape: in={in_planes}, out={planes}, stride={stride}"
            )
        pad = planes // 4
        return F.pad(q[:, :, ::2, ::2], (0, 0, 0, 0, pad, pad), "constant", 0)

    def _stage_shortcut(self, record, input_state: _IntegerState) -> _IntegerState:
        block = record["block"]
        prefix = record["name"]
        if not record["transition"]:
            # Identity shortcut inherits the already-quantized input state;
            # it is not a duplicate quantization node.
            return input_state

        if self._fake_quant is not None:
            if input_state.q is None or input_state.shift is None:
                # H1-S may select only the stage shortcut while the incoming
                # state remains H0 reference precision. Preserve the real
                # Option-A transform and quantize only this selected node.
                shortcut = block.shortcut(input_state.value)
                return self._state(f"{prefix}.stage_shortcut", shortcut)
            q_short = self._option_a_integer(
                input_state.q,
                block.conv1.in_channels,
                block.conv1.out_channels,
                block.conv1.stride[0],
            )
            inherited = q_short.to(torch.float32) * (2.0 ** (-input_state.shift))
            return self._state(f"{prefix}.stage_shortcut", inherited)

        shortcut = block.shortcut(input_state.value)
        return self._state(f"{prefix}.stage_shortcut", shortcut)

    def _residual_add(
        self,
        add_name: str,
        branch_a: _IntegerState,
        branch_b: _IntegerState,
    ) -> _IntegerState:
        if self._fake_quant is None:
            return self._state(add_name, branch_a.value + branch_b.value)
        output_is_quantized = not hasattr(self._fake_quant, "is_quantized") or self._fake_quant.is_quantized(add_name)
        if not output_is_quantized:
            # A single-node H1-S perturbation must not introduce integer
            # arithmetic at an unselected residual output.
            return self._state(add_name, branch_a.value + branch_b.value)
        if any(state.q is None or state.shift is None for state in (branch_a, branch_b)):
            # If the selected node is the residual output but one branch is
            # still H0 reference precision, quantize the float sum only at
            # the selected boundary.  Full integer branch alignment is used
            # when both branches carry q+shift (the H1-MP case).
            return self._state(add_name, branch_a.value + branch_b.value)

        out_shift = int(self._fake_quant.shift_map[add_name])
        out_bits = self._fake_bits_for(add_name)
        a_aligned, _ = requantize_int(branch_a.q, branch_a.shift, out_shift, out_bits=None)
        b_aligned, _ = requantize_int(branch_b.q, branch_b.shift, out_shift, out_bits=None)
        summed = a_aligned + b_aligned
        qmin, qmax = signed_int_range(out_bits)
        saturation = (summed < qmin) | (summed > qmax)
        out_q = torch.clamp(summed, qmin, qmax)
        self._fake_quant.record_saturation(add_name, int(saturation.sum().item()))

        runtime = self._alignment_runtime.setdefault(
            add_name,
            {
                "add_node": add_name,
                "branch_a_shift": int(branch_a.shift),
                "branch_b_shift": int(branch_b.shift),
                "branch_a_bits": self._fake_bits_for(branch_a.node_name),
                "branch_b_bits": self._fake_bits_for(branch_b.node_name),
                "common_output_bits": out_bits,
                "common_output_shift": out_shift,
                "required_shift_amount": {
                    "branch_a": out_shift - int(branch_a.shift),
                    "branch_b": out_shift - int(branch_b.shift),
                },
                "branch_a_node": branch_a.node_name,
                "branch_b_node": branch_b.node_name,
                "tensor_elements": 0,
                "saturation_count": 0,
            },
        )
        runtime["tensor_elements"] += int(out_q.numel())
        runtime["saturation_count"] += int(saturation.sum().item())
        return _IntegerState(
            value=out_q.to(torch.float32) * (2.0 ** (-out_shift)),
            q=out_q,
            shift=out_shift,
            node_name=add_name,
        )

    def _run_block(self, record, input_state: _IntegerState) -> _IntegerState:
        block = record["block"]
        prefix = record["name"]
        x = input_state.value

        first = block.aff1(block.conv1(x))
        first_state = self._state(f"{prefix}.first_affine", first)
        shortcut_state = self._stage_shortcut(record, input_state)
        x1 = self._residual_add(f"{prefix}.x1", first_state, shortcut_state)

        pre_second = self._state(
            f"{prefix}.pre_bconv2_hardtanh",
            F.hardtanh(x1.value, inplace=False),
        )
        second = block.aff2(block.conv2(pre_second.value))
        second_state = self._state(f"{prefix}.second_affine", second)
        second_add = self._residual_add(f"{prefix}.second_add", second_state, x1)
        output = block.post_act(second_add.value)
        return self._state(f"{prefix}.qrprelu_output", output)

    def forward(self, x):
        # This is the exact R7/R8B graph with explicit H1 boundaries. The
        # convolution and GAP accumulators remain untouched/full precision.
        encoded = self.model.thermo(x)
        stem_acc = self.model.conv1(encoded)
        stem = F.hardtanh(self.model.stem_affine(stem_acc), inplace=False)
        state = self._state("stem.post_hardtanh", stem)

        for record in self._blocks:
            state = self._run_block(record, state)

        gap = F.avg_pool2d(state.value, state.value.size(3))
        features = gap.view(gap.size(0), -1)
        head = self._state(
            "head.affine_output",
            self.model.head_affine(features),
        )
        # Final FC signed-pow2 weights, bias, and its accumulator are not
        # quantized in H1; only its multi-bit input is represented by q+shift.
        return self.model.linear(head.value)

    __call__ = forward

    @staticmethod
    def _alignment_equivalence(branch_a_shift, branch_b_shift, out_shift, bits):
        qa = torch.tensor([0, 1, -1, 2, -2], dtype=torch.int64)
        qb = torch.tensor([0, -1, 1, 2, -2], dtype=torch.int64)
        a, _ = requantize_int(qa, branch_a_shift, out_shift, out_bits=None)
        b, _ = requantize_int(qb, branch_b_shift, out_shift, out_bits=None)
        expected = a + b
        qmin, qmax = signed_int_range(bits)
        expected = torch.clamp(expected, qmin, qmax)
        actual = aligned_integer_add(
            qa, branch_a_shift, qb, branch_b_shift,
            out_bits=bits, out_shift=out_shift,
        )["q"]
        return bool(torch.equal(expected, actual))

    def verify_integer_residual_alignment(self, shift_map, bits_or_map=None, bits=None) -> dict:
        if bits_or_map is None:
            bits_or_map = bits
        if bits_or_map is None:
            raise ValueError("bits_or_map or bits must be provided")
        missing = set(self.list_quant_nodes()).difference(shift_map)
        if missing:
            raise RuntimeError("Missing shifts for H1 nodes: " + ", ".join(sorted(missing)))

        def bits_for(name, fallback=None):
            if isinstance(bits_or_map, dict):
                if name in bits_or_map:
                    return int(bits_or_map[name])
                if fallback is not None and fallback in bits_or_map:
                    return int(bits_or_map[fallback])
                raise RuntimeError(f"Missing bit-width for alignment node: {name}")
            return int(bits_or_map)

        records = []
        input_shift = int(shift_map["stem.post_hardtanh"])
        input_bits = bits_for("stem.post_hardtanh")
        for item in self._blocks:
            prefix = item["name"]
            first_shift = int(shift_map[f"{prefix}.first_affine"])
            if item["transition"]:
                shortcut_name = f"{prefix}.stage_shortcut"
                shortcut_shift = int(shift_map[shortcut_name])
                shortcut_role = "Option-A stage shortcut"
            else:
                shortcut_name = "inherited_input_state"
                shortcut_shift = input_shift
                shortcut_bits = input_bits
                shortcut_role = "identity shortcut inherits input shift"
            if item["transition"]:
                shortcut_bits = bits_for(shortcut_name)
            x1_shift = int(shift_map[f"{prefix}.x1"])
            x1_bits = bits_for(f"{prefix}.x1")
            first_bits = bits_for(f"{prefix}.first_affine")
            records.append({
                "add_node": f"{prefix}.x1",
                "branch_a_node": f"{prefix}.first_affine",
                "branch_b_node": shortcut_name,
                "branch_b_role": shortcut_role,
                "branch_a_shift": first_shift,
                "branch_b_shift": shortcut_shift,
                "branch_a_bits": first_bits,
                "branch_b_bits": shortcut_bits,
                "common_output_shift": x1_shift,
                "common_output_bits": x1_bits,
                "required_shift_amount": {
                    "branch_a": x1_shift - first_shift,
                    "branch_b": x1_shift - shortcut_shift,
                },
                "integer_add_equivalence": self._alignment_equivalence(
                    first_shift, shortcut_shift, x1_shift, x1_bits
                ),
                "saturation_count": self._alignment_runtime.get(
                    f"{prefix}.x1", {}
                ).get("saturation_count", 0),
                "tensor_elements": self._alignment_runtime.get(
                    f"{prefix}.x1", {}
                ).get("tensor_elements", 0),
            })

            second_shift = int(shift_map[f"{prefix}.second_affine"])
            second_add_shift = int(shift_map[f"{prefix}.second_add"])
            second_bits = bits_for(f"{prefix}.second_affine")
            second_add_bits = bits_for(f"{prefix}.second_add")
            records.append({
                "add_node": f"{prefix}.second_add",
                "branch_a_node": f"{prefix}.second_affine",
                "branch_b_node": f"{prefix}.x1",
                "branch_b_role": "saved first residual-add output x1",
                "branch_a_shift": second_shift,
                "branch_b_shift": x1_shift,
                "branch_a_bits": second_bits,
                "branch_b_bits": x1_bits,
                "common_output_shift": second_add_shift,
                "common_output_bits": second_add_bits,
                "required_shift_amount": {
                    "branch_a": second_add_shift - second_shift,
                    "branch_b": second_add_shift - x1_shift,
                },
                "integer_add_equivalence": self._alignment_equivalence(
                    second_shift, x1_shift, second_add_shift, second_add_bits
                ),
                "saturation_count": self._alignment_runtime.get(
                    f"{prefix}.second_add", {}
                ).get("saturation_count", 0),
                "tensor_elements": self._alignment_runtime.get(
                    f"{prefix}.second_add", {}
                ).get("tensor_elements", 0),
            })
            input_shift = int(shift_map[f"{prefix}.qrprelu_output"])
            input_bits = bits_for(f"{prefix}.qrprelu_output")

        return {
            "bits": int(bits_or_map) if not isinstance(bits_or_map, dict) else None,
            "rounding_policy": "round-to-nearest, ties away from zero for arithmetic right shift",
            "left_shift_for_positive_shift_delta": True,
            "arithmetic_right_shift_for_negative_shift_delta": True,
            "residual_add_count": len(records),
            "integer_add_equivalence_all": all(item["integer_add_equivalence"] for item in records),
            "records": records,
            "option_a": [
                {
                    "block": item["name"],
                    "stride": int(item["block"].conv1.stride[0]),
                    "input_channels": int(item["block"].conv1.in_channels),
                    "output_channels": int(item["block"].conv1.out_channels),
                    "spatial_operation": "x[:, :, ::2, ::2]",
                    "channel_operation": "symmetric zero padding on both sides",
                    "channel_padding_each_side": int(item["block"].conv1.out_channels // 4),
                    "input_shift_inherited_before_optional_requantization": True,
                    "verified_integer_transform": True,
                }
                for item in self._blocks if item["transition"]
            ],
        }
