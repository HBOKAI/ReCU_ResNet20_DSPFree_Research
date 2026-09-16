import torch
import json
from pathlib import Path

from tools.run_b_precision_ablation import int1_signed_range, quantize_int1_tensor, search_int1_shift


def test_int1_is_direct_h0_twos_complement_formula_extension():
    assert int1_signed_range() == (-1, 0)


def test_int1_uses_torch_round_ties_to_even_and_range():
    x = torch.tensor([-0.75, -0.5, -0.25, 0.25, 0.5, 0.75])
    result = quantize_int1_tensor(x, shift=1)
    assert result["q"].tolist() == [-1, -1, 0, 0, 0, 0]
    assert result["qmin"] == -1 and result["qmax"] == 0


def test_int1_shift_search_follows_h0_zero_saturation_priority():
    x = torch.tensor([-7.0, -2.0, 1.0])
    shift, result = search_int1_shift(x)
    assert result["saturation_count"] == 0
    assert shift < 0


def test_completed_ablation_reproduces_every_h0_scale_and_q_value():
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/B_PRECISION_ABLATION.json"
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    historical = json.loads((root / "H0_INTEGER_BIAS_SWEEP_RESULTS.json").read_text(encoding="utf-8"))
    rows = {item["format"]: item for item in result["results"]}
    for old in historical["results"]:
        current = rows[f"INT{old['bits']}"]
        assert current["accuracy"] == old["accuracy"]
        for name, layer in old["layers"].items():
            assert current["layers"][name]["shift"] == layer["shift"]
            assert current["layers"][name]["q_values"] == layer["q_values"]
