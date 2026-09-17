import math
from typing import Dict, List

import torch
import torch.nn as nn

from .selective_spot import EXPECTED_BACKBONE_K, named_spot_affines, verify_target_count


def _spatial_area(module_name: str) -> int:
    if module_name.startswith("layer1."):
        return 32 * 32
    if module_name.startswith("layer2."):
        return 16 * 16
    if module_name.startswith("layer3."):
        return 8 * 8
    raise ValueError(f"unexpected backbone affine module name: {module_name}")


def _signed_bits_for_magnitude(max_abs_integer: int) -> int:
    max_abs_integer = int(max_abs_integer)
    if max_abs_integer < 0:
        raise ValueError("max_abs_integer must be non-negative")
    if max_abs_integer == 0:
        return 1
    # Signed two's-complement range needs positive max <= 2^(bits-1)-1.
    return int(math.ceil(math.log2(max_abs_integer + 1))) + 1


def _conv_for_affine(model: nn.Module, affine_name: str) -> nn.Module:
    if affine_name.endswith(".aff1"):
        conv_name = affine_name[:-4] + "conv1"
    elif affine_name.endswith(".aff2"):
        conv_name = affine_name[:-4] + "conv2"
    else:
        raise ValueError(f"cannot map affine to binary conv: {affine_name}")

    modules = dict(model.named_modules())
    if conv_name not in modules:
        raise KeyError(f"missing paired convolution {conv_name} for {affine_name}")
    conv = modules[conv_name]
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"paired module {conv_name} is not Conv2d")
    return conv


def _kernel_terms(conv: nn.Conv2d) -> int:
    kh, kw = conv.kernel_size
    return int((conv.in_channels // conv.groups) * kh * kw)


@torch.no_grad()
def analyze_spot_hardware(model: nn.Module) -> Dict:
    """Analytical shift/add cost and pre-bias width audit for selective SPoT.

    This is deliberately not an RTL synthesis result.  It audits the K*S path
    only.  R4-level bias B is still floating in this sensitivity study, so a
    complete fixed-point affine-output width is intentionally not claimed.
    """
    total = verify_target_count(model, expected=EXPECTED_BACKBONE_K)

    exp_min = None
    exp_max = None
    rows: List[Dict] = []
    selected_count = 0
    active_count = 0
    extra_shift_ops = 0
    extra_add_ops = 0
    max_extra_product_bits = 0
    max_two_term_product_bits = 0
    max_leading_product_bits = 0
    max_triangle_product_bound = 0.0

    for module_name, module in named_spot_affines(model):
        exp_min = module.exp_min if exp_min is None else min(exp_min, module.exp_min)
        exp_max = module.exp_max if exp_max is None else max(exp_max, module.exp_max)
        terms = module.discrete_terms()
        conv = _conv_for_affine(model, module_name)
        n_terms = _kernel_terms(conv)
        area = _spatial_area(module_name)

        for ch in range(module.channels):
            selected = bool(terms["selected"][ch].cpu())
            active2 = bool(terms["active2"][ch].cpu())
            k1 = int(terms["k1"][ch].cpu())
            k2 = int(terms["k2"][ch].cpu())
            s1 = int(float(terms["s1"][ch].cpu()))
            s2 = int(float(terms["s2"][ch].cpu()))
            q = float(terms["q"][ch].cpu())

            selected_count += int(selected)
            active_count += int(active2)

            row = {
                "module": module_name,
                "channel": ch,
                "selected": selected,
                "active2": active2,
                "kernel_terms_max_abs_S": n_terms,
                "spatial_outputs_per_image": area,
                "effective_k": q,
                "k1": k1,
                "k2": k2,
                "s1": s1,
                "s2": s2,
            }

            if active2:
                if not k1 > k2:
                    raise RuntimeError(
                        f"non-canonical active SPoT term at {module_name}[{ch}]: {k1}, {k2}"
                    )
                delta = k1 - k2
                # Express both terms in units of 2^k2.  The leading term is
                # S*2^delta and the second term is +/-S.
                leading_bound = n_terms * (1 << delta)
                sum_bound = n_terms * ((1 << delta) + 1)
                leading_bits = _signed_bits_for_magnitude(leading_bound)
                sum_bits = _signed_bits_for_magnitude(sum_bound)
                extra_bits = sum_bits - leading_bits
                if extra_bits not in (0, 1):
                    raise RuntimeError("2-term add should require at most one extra carry bit")

                triangle_bound = n_terms * ((2.0 ** k1) + (2.0 ** k2))
                actual_bound = n_terms * abs(q)
                row.update({
                    "common_lsb_exp": k2,
                    "term_exp_gap": delta,
                    "leading_term_integer_bound_at_common_lsb": leading_bound,
                    "two_term_integer_bound_at_common_lsb": sum_bound,
                    "leading_term_signed_bits_at_common_lsb": leading_bits,
                    "two_term_signed_bits_at_common_lsb": sum_bits,
                    "extra_carry_bits_vs_leading_term": extra_bits,
                    "triangle_product_bound_before_bias": triangle_bound,
                    "actual_abs_k_times_max_abs_S": actual_bound,
                })

                extra_shift_ops += area
                extra_add_ops += area
                max_extra_product_bits = max(max_extra_product_bits, extra_bits)
                max_two_term_product_bits = max(max_two_term_product_bits, sum_bits)
                max_leading_product_bits = max(max_leading_product_bits, leading_bits)
                max_triangle_product_bound = max(max_triangle_product_bound, triangle_bound)
            else:
                row.update({
                    "common_lsb_exp": k1,
                    "term_exp_gap": 0,
                    "leading_term_integer_bound_at_common_lsb": n_terms,
                    "two_term_integer_bound_at_common_lsb": n_terms,
                    "leading_term_signed_bits_at_common_lsb": _signed_bits_for_magnitude(n_terms),
                    "two_term_signed_bits_at_common_lsb": _signed_bits_for_magnitude(n_terms),
                    "extra_carry_bits_vs_leading_term": 0,
                    "triangle_product_bound_before_bias": n_terms * abs(q),
                    "actual_abs_k_times_max_abs_S": n_terms * abs(q),
                })
            rows.append(row)

    if exp_min is None or exp_max is None:
        raise RuntimeError("no selective SPoT affines found")

    exponent_levels = int(exp_max - exp_min + 1)
    exponent_bits = int(math.ceil(math.log2(exponent_levels)))
    baseline_bits_per_channel = 1 + exponent_bits
    second_term_bits_per_active = 1 + exponent_bits
    active_flag_bits_per_channel = 1

    # Dense programmable encoding provisions second-term metadata for every
    # channel.  A fixed/static implementation can instead keep only the active
    # second-term metadata plus one activity bitmap.
    dense_extra_metadata_bits = total * (active_flag_bits_per_channel + second_term_bits_per_active)
    sparse_static_extra_metadata_bits = total * active_flag_bits_per_channel + active_count * second_term_bits_per_active

    return {
        "scope": "R4-level backbone K*S path only; floating B excluded from complete fixed-point width claim",
        "target_channels": total,
        "selected_channels": selected_count,
        "active_second_term_channels": active_count,
        "selected_pct": 100.0 * selected_count / total,
        "active_second_term_pct": 100.0 * active_count / total,
        "exponent_bounds": [int(exp_min), int(exp_max)],
        "exponent_levels": exponent_levels,
        "exponent_bits": exponent_bits,
        "baseline_one_term_metadata_bits_per_channel": baseline_bits_per_channel,
        "extra_second_term_bits_per_active_channel": second_term_bits_per_active,
        "activity_flag_bits_per_channel": active_flag_bits_per_channel,
        "dense_programmable_extra_metadata_bits": dense_extra_metadata_bits,
        "dense_programmable_extra_metadata_bytes": dense_extra_metadata_bits / 8.0,
        "sparse_static_extra_metadata_bits": sparse_static_extra_metadata_bits,
        "sparse_static_extra_metadata_bytes": sparse_static_extra_metadata_bits / 8.0,
        "extra_shift_operations_per_image_direct_interpretation": int(extra_shift_ops),
        "extra_add_sub_operations_per_image_direct_interpretation": int(extra_add_ops),
        "serialized_single_lane_extra_affine_cycles_proxy": int(extra_shift_ops),
        "serialized_p_lane_cycle_formula": "ceil(extra_second_term_channel-elements / P)",
        "parallel_engine_note": "Second term can use a parallel shift/sign path plus add/sub; expect extra LUT/routing and possible critical-path impact.",
        "serialized_engine_note": "A reused shift path can serialize the second term; cycle overhead scales with active second-term channel-elements and engine parallelism P.",
        "max_extra_carry_bits_at_two_term_add": int(max_extra_product_bits),
        "max_leading_term_signed_bits_at_common_lsb": int(max_leading_product_bits),
        "max_two_term_signed_bits_at_common_lsb": int(max_two_term_product_bits),
        "max_triangle_product_bound_before_bias": float(max_triangle_product_bound),
        "later_h2_width_reaudit_required_if_ported": bool(active_count > 0),
        "mathematically_general_multiplier_free": True,
        "physical_dsp48e1_zero_proven": False,
        "physical_dsp_note": "DSP48E1=0 requires RTL/Vivado synthesis; this Python audit is not physical proof.",
        "rows": rows,
    }
