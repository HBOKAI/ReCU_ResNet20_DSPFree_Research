"""Read-only feasibility audit for the requested QRPReLU threshold fold.

This intentionally does not edit the model, H0/H1MP/H2 plans, or checkpoints.
The diagnostic counterfactual tries to shift xi1 into the incoming H1MP grid;
it is *not* a valid deployed folded path, because it changes the branch/identity
semantics and may require new rounding. It provides a concrete integer witness
for why the requested exact-equivalence gate cannot be passed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.qrprelu import QuantizedRPReLU

CHECKPOINT = ROOT / "experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt"
H1MP = ROOT / "H1MP_SEARCH_RESULTS.json"
H2A = ROOT / "H2_FINITE_WIDTH_RESULTS.json"
H2B = ROOT / "H2B_WIDTH_OPTIMIZATION_RESULTS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clamp(q: torch.Tensor, bits: int) -> torch.Tensor:
    return q.clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def zero_input_counterfactual(
    xi1: torch.Tensor, xi2: torch.Tensor, exponents: torch.Tensor, plan: dict,
    h1_output_bits: int, h1_output_shift: int,
) -> dict:
    """Compare original q_in=0 with the would-be upstream-shifted xi1 path.

    Original returns the identity branch 0. The counterfactual removes xi1
    from the negative branch, rounds it onto q_in's grid, then compares that
    altered incoming q with zero and retains xi2 and all frozen widths.
    """
    in_shift = int(plan["input_shift"])
    param_shift = int(plan["param_shift"])
    out_shift = int(plan["output_shift"])
    q_shift = torch.round(xi1.double() * (2.0 ** in_shift)).to(torch.int64)
    q_inner = clamp(q_shift << (param_shift - in_shift), int(plan["inner_bits"]))
    q_xi2 = torch.round(xi2.double() * (2.0 ** param_shift)).to(torch.int64)
    q_xi1 = torch.round(xi1.double() * (2.0 ** param_shift)).to(torch.int64)
    powers = torch.tensor(
        [1 << int(out_shift + k - param_shift) for k in exponents.tolist()],
        dtype=torch.int64,
    )
    original_negative_shifted_term = clamp(q_xi1, int(plan["inner_bits"])) * powers
    folded_negative_shifted_term = q_inner * powers
    q_neg = folded_negative_shifted_term + (q_xi2 << (out_shift - param_shift))
    q_pos = q_shift << (out_shift - in_shift)
    selected_before_offset = torch.where(q_shift >= 0, q_pos, folded_negative_shifted_term)
    selected_after_offset = torch.where(q_shift >= 0, q_pos, q_neg)
    q_out = clamp(torch.where(q_shift >= 0, q_pos, q_neg), int(plan["bits"]))
    h1_q = torch.round(
        (q_out.to(torch.float32) * (2.0 ** (-out_shift))).to(torch.float64)
        * (2.0 ** int(h1_output_shift))
    ).to(torch.int64)
    h1_q = clamp(h1_q, int(h1_output_bits))
    return {
        "diagnostic_input_q_zero": 0,
        "diagnostic_original_output_q": 0,
        "counterfactual_shifted_input_q_nonzero_channels": int(torch.count_nonzero(q_shift).item()),
        "counterfactual_branch_mismatch_channels": int((q_shift < 0).sum().item()),
        "counterfactual_negative_shifted_term_mismatch_channels": int((original_negative_shifted_term != folded_negative_shifted_term).sum().item()),
        "counterfactual_selected_before_output_offset_mismatch_channels": int(torch.count_nonzero(selected_before_offset).item()),
        "counterfactual_selected_after_output_offset_mismatch_channels": int(torch.count_nonzero(selected_after_offset).item()),
        "counterfactual_output_mismatch_channels": int(torch.count_nonzero(q_out).item()),
        "counterfactual_h1mp_requantized_output_mismatch_channels": int(torch.count_nonzero(h1_q).item()),
        "counterfactual_max_abs_output_integer_error": int(q_out.abs().max().item()),
    }


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    h1mp = json.loads(H1MP.read_text(encoding="utf-8"))
    h2a = json.loads(H2A.read_text(encoding="utf-8"))["h2a_plan"]["qrprelu"]
    h2b = json.loads(H2B.read_text(encoding="utf-8"))["h2b_search"]["final_plan"]["qrprelu"]
    rows = []
    for name, safe_plan in h2a.items():
        prefix = name.removesuffix(".qrprelu_output")
        xi1 = state[f"{prefix}.post_act.xi1"].detach().cpu()
        xi2 = state[f"{prefix}.post_act.xi2"].detach().cpu()
        a = state[f"{prefix}.post_act.a"].detach().cpu()
        assert xi1.numel() == xi2.numel() == a.numel()
        module = QuantizedRPReLU(int(xi1.numel())).eval()
        module.load_state_dict({"a": a, "xi1": xi1, "xi2": xi2}, strict=True)
        synthetic_x = torch.tensor([-1.0, 0.0, 1.0]).view(3, 1, 1, 1).expand(3, int(xi1.numel()), 1, 1)
        manual_negative = torch.pow(2.0, torch.round(a)).view(1, -1, 1, 1) * (
            synthetic_x + xi1.view(1, -1, 1, 1)
        ) + xi2.view(1, -1, 1, 1)
        manual_output = torch.where(synthetic_x >= 0, synthetic_x, manual_negative)
        assert torch.equal(module(synthetic_x), manual_output)
        assert torch.count_nonzero(module(torch.zeros_like(synthetic_x))) == 0
        exponents = torch.round(a).to(torch.int64)
        input_node = f"{prefix}.second_add"
        input_shift = int(h1mp["final_plan"]["shift_by_node"][input_node])
        assert input_shift == int(safe_plan["input_shift"])
        assert int(h2b[name]["input_shift"]) == input_shift
        h1_output_bits = int(h1mp["final_plan"]["bits_by_node"][name])
        h1_output_shift = int(h1mp["final_plan"]["shift_by_node"][name])
        param_shift = int(safe_plan["param_shift"])
        q_xi1 = torch.round(xi1.double() * (2.0 ** param_shift)).to(torch.int64)
        incoming_grid_unit = 1 << (param_shift - input_shift)
        row = {
            "node": name,
            "parameter_names": [f"{prefix}.post_act.a", f"{prefix}.post_act.xi1", f"{prefix}.post_act.xi2"],
            "channels": int(xi1.numel()),
            "xi1_nonzero_float_channels": int(torch.count_nonzero(xi1).item()),
            "xi1_nonzero_h2_integer_channels": int(torch.count_nonzero(q_xi1).item()),
            "xi1_exactly_representable_on_incoming_grid_channels": int((q_xi1.remainder(incoming_grid_unit) == 0).sum().item()),
            "xi1_min": float(xi1.min().item()),
            "xi1_max": float(xi1.max().item()),
            "input_h1mp_bits": int(safe_plan["input_bits"]),
            "input_h1mp_shift": input_shift,
            "xi1_xi2_parameter_shift": param_shift,
            "safe_inner_bits": int(safe_plan["inner_bits"]),
            "safe_output_bits": int(safe_plan["bits"]),
            "safe_output_shift": int(safe_plan["output_shift"]),
            "h2b_inner_bits": int(h2b[name]["inner_bits"]),
            "h2b_output_bits": int(h2b[name]["bits"]),
            "h1mp_output_bits": h1_output_bits,
            "h1mp_output_shift": h1_output_shift,
            "zero_input_counterfactual_safe": zero_input_counterfactual(xi1, xi2, exponents, safe_plan, h1_output_bits, h1_output_shift),
            "zero_input_counterfactual_h2b": zero_input_counterfactual(xi1, xi2, exponents, h2b[name], h1_output_bits, h1_output_shift),
        }
        rows.append(row)
    assert len(rows) == 9
    assert sum(row["channels"] for row in rows) == 336
    print(json.dumps({
        "source_sha256": {
            "r8b_checkpoint": sha256(CHECKPOINT),
            "h1mp_results": sha256(H1MP),
            "h2a_v2_results": sha256(H2A),
            "h2b_results": sha256(H2B),
        },
        "nodes": rows,
        "totals": {
            "qrprelu_nodes": len(rows),
            "per_channel_xi1_values": sum(row["channels"] for row in rows),
            "nonzero_h2_integer_xi1_values": sum(row["xi1_nonzero_h2_integer_channels"] for row in rows),
            "exactly_representable_xi1_values_on_h1mp_input_grid": sum(row["xi1_exactly_representable_on_incoming_grid_channels"] for row in rows),
            "zero_input_safe_counterfactual_output_mismatches": sum(row["zero_input_counterfactual_safe"]["counterfactual_output_mismatch_channels"] for row in rows),
            "zero_input_h2b_counterfactual_output_mismatches": sum(row["zero_input_counterfactual_h2b"]["counterfactual_output_mismatch_channels"] for row in rows),
            "zero_input_safe_counterfactual_h1mp_requantized_mismatches": sum(row["zero_input_counterfactual_safe"]["counterfactual_h1mp_requantized_output_mismatch_channels"] for row in rows),
            "zero_input_h2b_counterfactual_h1mp_requantized_mismatches": sum(row["zero_input_counterfactual_h2b"]["counterfactual_h1mp_requantized_output_mismatch_channels"] for row in rows),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
