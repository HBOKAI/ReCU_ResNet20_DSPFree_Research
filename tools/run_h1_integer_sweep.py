"""Run the H1 train-calibrated, zero-shot activation/residual sweep."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.fused_affine import FusedAffine2d
from recu_hw.h1_model_adapter import H1ModelAdapter
from recu_hw.integer_activation import (
    IntegerActivationFakeQuant,
    RangeObserverRegistry,
    choose_zero_saturation_shift,
    quantize_float_to_int,
    signed_int_range,
)
from recu_hw.integer_bias import collect_integer_bias_targets
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.r6 import is_power_of_two_tensor, parameter_freeze_report, stem_binary_weight_stats
from recu_hw.r7 import R7ResNet20
from recu_hw.r8 import Pow2LinearSTE

from tools.eval_integer_bias_sweep import discover_formal_r8b_checkpoint


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_r8b(path, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("config", {})
    cls_cfg = cfg.get("classifier", {}) if isinstance(cfg, dict) else {}
    model = R7ResNet20(num_classes=10, resolution=8)
    model.linear = Pow2LinearSTE(
        64,
        10,
        bias=True,
        min_exp=cls_cfg.get("min_exp"),
        max_exp=cls_cfg.get("max_exp"),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.conv1.set_binary_weight(True)
    model.stem_affine.quantize_k = True
    model.head_affine.quantize_k = True
    for module in model.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def verify_r8b_invariants(model):
    errors = []
    stem = stem_binary_weight_stats(model)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        errors.append(f"stem invariant failed: {stem}")
    if not is_power_of_two_tensor(model.stem_affine.effective_k()):
        errors.append("stem K is not signed-pow2")
    if not is_power_of_two_tensor(model.head_affine.effective_k()):
        errors.append("head K is not signed-pow2")
    for name, module in model.named_modules():
        if isinstance(module, FusedAffine2d) and not is_power_of_two_tensor(module.effective_k()):
            errors.append(f"{name} K is not signed-pow2")
    if not model.linear.stats()["is_exact_signed_pow2"]:
        errors.append("FC weight is not signed-pow2")
    if model.linear.bias is None:
        errors.append("FC bias is missing")
    if hasattr(model, "bn2"):
        errors.append("BN2 is present")
    freeze = parameter_freeze_report(model)
    if freeze["frozen_params"] != 672:
        errors.append(f"frozen alpha count changed: {freeze['frozen_params']}")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        errors.append("non-finite model parameter")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "stem": stem,
        "all_backbone_affine_k_signed_pow2": True,
        "head_k_signed_pow2": True,
        "fc": model.linear.stats(),
        "fc_bias_present": True,
        "bn2_removed": True,
        "freeze_report": freeze,
        "finite": True,
    }


def load_h0_int6_layers(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in data.get("results", []) if int(item.get("bits", -1)) == 6]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one H0 INT6 result, found {len(matches)}")
    item = matches[0]
    if len(item.get("layers", {})) != 21:
        raise RuntimeError("H0 INT6 export does not contain all 21 bias targets")
    return data, item


@torch.no_grad()
def apply_h0_int6_biases(model, h0_path):
    h0_data, h0_item = load_h0_int6_layers(h0_path)
    modules = dict(model.named_modules())
    expected = collect_integer_bias_targets(model)
    if set(h0_item["layers"]) != set(expected):
        missing = sorted(set(expected) - set(h0_item["layers"]))
        extra = sorted(set(h0_item["layers"]) - set(expected))
        raise RuntimeError(f"H0 target mapping mismatch; missing={missing}, extra={extra}")

    records = []
    for name, info in h0_item["layers"].items():
        if int(info.get("bits", -1)) != 6:
            raise RuntimeError(f"H0 layer {name} is not INT6")
        qmin, qmax = signed_int_range(6)
        q_values = info.get("q_values", [])
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in q_values):
            raise RuntimeError(f"H0 q values for {name} are not integers")
        if any(value < qmin or value > qmax for value in q_values):
            raise RuntimeError(f"H0 q values for {name} are outside INT6 range")
        module = modules[name]
        q = torch.tensor(q_values, dtype=torch.float64, device=module.bias.device)
        shift = int(info["shift"])
        reconstructed = q * (2.0 ** (-shift))
        if reconstructed.numel() != module.bias.numel():
            raise RuntimeError(f"H0 bias size mismatch for {name}")
        module.bias.copy_(reconstructed.to(module.bias.dtype).reshape_as(module.bias))
        records.append({
            "module_name": name,
            "class_name": module.__class__.__name__,
            "param_count": int(module.bias.numel()),
            "bits": 6,
            "shift": shift,
            "qmin": qmin,
            "qmax": qmax,
            "q_values": [int(value) for value in q_values],
            "reconstructed_min": float(reconstructed.min().item()),
            "reconstructed_max": float(reconstructed.max().item()),
        })
    return {
        "source_results_json": str(h0_path),
        "source_recorded_r8b_reload_acc": h0_data.get("source_reload_acc"),
        "recorded_h0_int6_accuracy": h0_item.get("accuracy"),
        "bits": 6,
        "layer_count": len(records),
        "scalar_count": sum(record["param_count"] for record in records),
        "representation": "signed INT6 q plus layer-level integer shift; Bhat=q*2^-shift",
        "bias_search_repeated": False,
        "float_bias_storage_exported": False,
        "layers": records,
    }


def snapshot_biases(model):
    return {
        name: module.bias.detach().clone()
        for name, module in collect_integer_bias_targets(model).items()
    }


def verify_h0_biases_unchanged(model, snapshot):
    modules = dict(model.named_modules())
    errors = []
    for name, before in snapshot.items():
        if not torch.equal(modules[name].bias.detach(), before):
            errors.append(name)
    if errors:
        raise RuntimeError("H0 INT6 bias changed during H1: " + ", ".join(errors))
    return True


@torch.no_grad()
def evaluate(predictor, loader, device):
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += int((predictor(x).argmax(1) == y).sum().item())
        total += int(y.size(0))
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def adapter_matches_model(adapter, loader, device):
    x, _ = next(iter(loader))
    x = x[: min(8, x.size(0))].to(device, non_blocking=True)
    direct = adapter.model(x)
    adapted = adapter(x)
    max_abs = float((direct - adapted).abs().max().item())
    return {"max_abs_logit_difference": max_abs, "matches": max_abs <= 1e-5}


@torch.no_grad()
def calibrate(model, adapter, train_loader, device, max_samples):
    observer = RangeObserverRegistry()
    adapter.set_fake_quant(None)
    adapter.set_observer(observer)
    seen = 0
    model.eval()
    for x, _ in train_loader:
        if seen >= max_samples:
            break
        remaining = max_samples - seen
        if x.size(0) > remaining:
            x = x[:remaining]
        adapter(x.to(device, non_blocking=True))
        seen += int(x.size(0))
    adapter.set_observer(None)
    if seen != max_samples and max_samples <= len(train_loader.dataset):
        raise RuntimeError(f"Calibration saw {seen} samples, expected {max_samples}")
    missing = set(adapter.list_quant_nodes()) - set(observer.stats)
    if missing:
        raise RuntimeError("Calibration missed H1 nodes: " + ", ".join(sorted(missing)))
    return observer, seen


def calibration_shift_table(observer, nodes, widths, min_shift, max_shift):
    ranges = observer.to_dict()
    shifts_by_width = {}
    for bits in widths:
        shift_map = observer.shifts_for_bits(bits, min_shift=min_shift, max_shift=max_shift)
        shifts_by_width[str(bits)] = shift_map
        _, qmax = signed_int_range(bits)
        for name in nodes:
            stats = ranges[name]
            shift = int(shift_map[name])
            qres = quantize_float_to_int(torch.tensor([stats["min_value"], stats["max_value"]]), bits, shift)
            stats.setdefault("selected_shift_by_bits", {})[str(bits)] = shift
            stats.setdefault("calibration_zero_saturation_by_bits", {})[str(bits)] = qres["saturation_count"] == 0
            stats.setdefault("shift_search_boundary_by_bits", {})[str(bits)] = (
                "min" if shift == min_shift else "max" if shift == max_shift else "interior"
            )
            stats.setdefault("range_check_absmax_times_scale_by_bits", {})[str(bits)] = float(
                stats["absmax"] * (2.0 ** shift)
            )
            if qres["saturation_count"] != 0:
                raise RuntimeError(
                    f"Calibration range requires shift search expansion for {name}, INT{bits}; "
                    f"selected s={shift}, absmax={stats['absmax']} and qmax={qmax}"
                )
    return ranges, shifts_by_width


def test_saturation_map(fake_quant, nodes):
    return {name: int(fake_quant.saturation_by_node.get(name, 0)) for name in nodes}


def test_error_map(fake_quant, nodes):
    empty = {"element_count": 0, "mse": 0.0, "max_abs_error": 0.0}
    return {name: dict(fake_quant.error_statistics().get(name, empty)) for name in nodes}


def analyze_sensitivity(results, nodes):
    int4 = next(item for item in results if item["bits"] == 4)
    int5 = next(item for item in results if item["bits"] == 5)
    ranked = sorted(
        nodes,
        key=lambda name: (
            int4["test_error_by_node"][name]["mse"],
            int4["test_error_by_node"][name]["max_abs_error"],
        ),
        reverse=True,
    )
    residual_nodes = [
        name for name in nodes
        if name.endswith(".x1") or name.endswith(".second_add")
    ]
    residual_ranked = sorted(
        residual_nodes,
        key=lambda name: (
            int4["test_saturation_by_node"][name],
            int4["test_error_by_node"][name]["mse"],
        ),
        reverse=True,
    )
    top_residual = residual_ranked[0]
    residual_sat_total = sum(int4["test_saturation_by_node"][name] for name in residual_nodes)
    return {
        "most_sensitive_node_by_int4_test_mse": ranked[0],
        "most_sensitive_node_int4_mse": int4["test_error_by_node"][ranked[0]],
        "top_residual_state_by_int4_saturation_then_mse": top_residual,
        "top_residual_state_int4_stats": {
            "saturation": int4["test_saturation_by_node"][top_residual],
            "error": int4["test_error_by_node"][top_residual],
        },
        "int5_accuracy_drop_pp": float(int5["accuracy_drop_vs_h0_pp"]),
        "int4_accuracy_drop_pp": float(int4["accuracy_drop_vs_h0_pp"]),
        "residual_saturation_total_int4": int(residual_sat_total),
        "residual_collapse_evidence": (
            "saturation_observed_at_residual_outputs"
            if residual_sat_total > 0
            else "no_residual_output_saturation; accuracy loss is not attributable to one saturation event"
        ),
        "note": "Sensitivity ranking is a quantization-error/saturation proxy, not a one-node ablation attribution.",
    }


def render_report(summary):
    source = summary["source_verification"]
    h0 = summary["h0_int6_reconstruction"]
    lines = [
        "# H1 Integer Activation / Residual Report",
        "",
        "> R8B-H0 zero-shot multi-bit activation/residual state sweep. No fine-tuning was performed.",
        "",
        "## A. Source R8B / H0 verification",
        "",
        f"- Source checkpoint: `{summary['source_r8b_checkpoint']}`",
        f"- Stored R8B best: **{summary['source_stored_best_acc']:.2f}%** @ epoch **{summary['source_stored_best_epoch']}**",
        f"- R8B actual reload before H0: **{summary['source_r8b_reload_accuracy']:.2f}%**",
        f"- H0 INT6 baseline after fixed q+shift reconstruction: **{summary['h0_int6_baseline_accuracy']:.2f}%** (target 85.13% +/- 0.08 pp)",
        f"- Adapter disabled-quantization equivalence max logit error: `{summary['adapter_equivalence']['max_abs_logit_difference']:.3g}`; PASS.",
        "- R8B invariants: PASS; binary stem, signed-pow2 affine/FC weights, FC bias present, no BN2, 672 alpha parameters frozen.",
        "",
        "## B. H0 INT6 bias reconstruction",
        "",
        f"- Source JSON: `{h0['source_results_json']}`",
        f"- Fixed targets: {h0['layer_count']} tensors / {h0['scalar_count']} scalar biases.",
        "- For every layer, the stored H0 `q_values` and `shift` were applied as `Bhat = q * 2^(-shift)`.",
        "- Bias bit-width was not searched again; H0 bias remained fixed during every H1 width evaluation.",
        "",
        "| Module | Class | Shift | q range | Parameters |",
        "|---|---|---:|---:|---:|",
    ]
    for item in h0["layers"]:
        lines.append(
            f"| `{item['module_name']}` | `{item['class_name']}` | {item['shift']} | "
            f"[{item['qmin']}, {item['qmax']}] | {item['param_count']} |"
        )

    lines += [
        "",
        "## C. H1 quantization nodes",
        "",
        f"- Total quant nodes: **{len(summary['quant_nodes'])}**.",
        "- The adapter uses controlled forward execution; it does not rely on generic module hooks.",
        "- Binary-convolution accumulators and GAP accumulation are explicitly not quantized.",
        "",
        "| Node | Tensor role | Description |",
        "|---|---|---|",
    ]
    for node in summary["quant_nodes"]:
        lines.append(f"| `{node['name']}` | `{node['tensor_role']}` | {node['role']} |")

    lines += [
        "",
        "## D. Calibration methodology",
        "",
        f"- Dataset: CIFAR-10 TRAIN only; samples used: **{summary['calibration']['samples']}**.",
        "- Test data was not used to choose any shift. Test data was used only for final accuracy and saturation statistics.",
        f"- Shift search: `{summary['calibration']['min_shift']}..{summary['calibration']['max_shift']}`.",
        "- Selection: largest integer shift with zero calibration saturation based on observed absmax.",
        "- Right-shift alignment rounding: nearest integer, ties away from zero.",
        "",
        "## E. Train calibration ranges",
        "",
        "| Node | Count | Min | Max | Absmax |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in summary["quant_nodes_order"]:
        item = summary["calibration_ranges"][name]
        lines.append(
            f"| `{name}` | {item['count']} | {item['min_value']:.8g} | "
            f"{item['max_value']:.8g} | {item['absmax']:.8g} |"
        )

    lines += ["", "## F. Per-width shifts", ""]
    for bits in summary["tested_widths"]:
        lines += [
            f"### INT{bits}",
            "",
            "| Node | Shift | Range | Calibration zero-sat |",
            "|---|---:|---:|---:|",
        ]
        for name in summary["quant_nodes_order"]:
            item = summary["calibration_ranges"][name]
            qmin, qmax = signed_int_range(bits)
            lines.append(
                f"| `{name}` | {item['selected_shift_by_bits'][str(bits)]} | [{qmin}, {qmax}] | "
                f"{item['calibration_zero_saturation_by_bits'][str(bits)]} |"
            )
        lines.append("")

    lines += [
        "## G. Residual integer scale alignment",
        "",
        "Every residual add aligns both integer branches to the common output shift. Positive deltas use integer left shifts; negative deltas use signed arithmetic right shifts with defined rounding. Output saturation is applied only at the designated W-bit output node.",
        "",
    ]
    for result in summary["results"]:
        lines += [
            f"### INT{result['bits']}",
            "",
            "| Add node | Branch A s | Branch B s | Output s | Delta A/B | Test saturation | Equivalence |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
        for item in result["residual_alignment"]["records"]:
            delta = item["required_shift_amount"]
            lines.append(
                f"| `{item['add_node']}` | {item['branch_a_shift']} | {item['branch_b_shift']} | "
                f"{item['common_output_shift']} | {delta['branch_a']}/{delta['branch_b']} | "
                f"{item['saturation_count']} | {item['integer_add_equivalence']} |"
            )
        lines.append("")

    lines += [
        "## H. Option-A handling",
        "",
        "Stage2 and Stage3 shortcuts use the actual workspace Option-A implementation: `x[:, :, ::2, ::2]`, followed by symmetric zero channel padding. The integer adapter performs the same transform directly on q, inherits the input shift before any designated shortcut requantization, and verifies the resulting shape/alignment metadata.",
        "",
        "| Block | Stride | Input channels | Output channels | Padding per side |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["results"][0]["residual_alignment"]["option_a"]:
        lines.append(
            f"| `{item['block']}` | {item['stride']} | {item['input_channels']} | "
            f"{item['output_channels']} | {item['channel_padding_each_side']} |"
        )

    lines += [
        "",
        "## I. Accuracy table",
        "",
        "| Width | Accuracy | Delta vs H0 85.13% | Test saturation |",
        "|---|---:|---:|---:|",
        f"| H0 reference | {summary['h0_int6_baseline_accuracy']:.2f}% | 0.00 pp | - |",
    ]
    for result in summary["results"]:
        lines.append(
            f"| INT{result['bits']} | {result['accuracy']:.2f}% | "
            f"{result['delta_vs_h0_pp']:+.2f} pp | {result['test_total_saturation']} |"
        )

    lines += [
        "",
        "## J. Test saturation statistics",
        "",
        "The JSON contains per-node saturation and error statistics for every width. The following table shows total saturation and the largest per-node count.",
        "",
        "| Width | Total saturation | Node with largest saturation | Count |",
        "|---|---:|---|---:|",
    ]
    for result in summary["results"]:
        node = max(summary["quant_nodes_order"], key=lambda name: result["test_saturation_by_node"][name])
        lines.append(
            f"| INT{result['bits']} | {result['test_total_saturation']} | `{node}` | "
            f"{result['test_saturation_by_node'][node]} |"
        )

    sensitivity = summary["sensitivity"]
    lines += [
        "",
        "## K. Sensitive-node analysis",
        "",
        f"- Most sensitive node by INT4 test quantization MSE proxy: **`{sensitivity['most_sensitive_node_by_int4_test_mse']}`**.",
        f"- Top residual state by INT4 saturation/MSE proxy: **`{sensitivity['top_residual_state_by_int4_saturation_then_mse']}`**.",
        f"- INT5 accuracy drop: **{sensitivity['int5_accuracy_drop_pp']:.2f} pp**; INT4 accuracy drop: **{sensitivity['int4_accuracy_drop_pp']:.2f} pp**.",
        f"- Residual-output saturation total at INT4: **{sensitivity['residual_saturation_total_int4']}**.",
        f"- Interpretation: {sensitivity['residual_collapse_evidence']}.",
        f"- This is a quantization-error/saturation proxy, not a one-node ablation attribution: {sensitivity['note']}",
        "",
        "## L. Recommended production width",
        "",
        f"- Lowest near-lossless (drop <= 0.10 pp): **{summary['lowest_near_lossless_bits'] or 'none'}**.",
        f"- Recommended production (drop <= 0.30 pp): **{summary['recommended_bits'] or 'none'}**.",
        f"- Absolute minimum acceptable (drop <= 0.50 pp): **{summary['absolute_minimum_bits'] or 'none'}**.",
        f"- INT4 drop was {summary['int4_drop_pp']:.2f} pp, so INT3 was {'tested' if 3 in summary['tested_widths'] else 'not tested'} under the optional rule.",
        "",
        "## M. Hardware interpretation",
        "",
        "Affine offsets/classifier bias use fixed INT6 integer storage from H0, and the selected multi-bit activation/residual states have a verified W-bit integer representation with power-of-two scale metadata.",
        "",
        "This does not mean the entire network is fully finite-width integer-only. Binary-convolution accumulators, GAP accumulation, final FC accumulation, and final logit width remain full precision for this H1 experiment and are reserved for later experiments.",
        "",
        "- Binary weights, signed-pow2 K, FC signed-pow2 weights, Thermometer, QRPReLU function/parameters, and architecture are unchanged.",
        "- No extra inference BMAC was introduced; H1 only changes representation boundaries of existing multi-bit states.",
        "- No training, fine-tuning, KD, accumulator truncation, GAP quantization, FC accumulator quantization, RTL, or FPGA work was performed.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "h1_integer_activation_sweep.json"))
    parser.add_argument("--source-r8b-checkpoint", default=None)
    parser.add_argument("--h0-results-json", default=str(ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--calibration-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibration_cfg = config.get("calibration", {})
    shift_cfg = config.get("shift_policy", {})
    widths = [int(bits) for bits in config.get("widths", [8, 7, 6, 5, 4])]
    calibration_samples = int(args.calibration_samples or calibration_cfg.get("max_samples", 10000))
    min_shift = int(shift_cfg.get("min_shift", -16))
    max_shift = int(shift_cfg.get("max_shift", 24))
    int3_threshold = float(config.get("optional_int3_if_int4_drop_le_pp", 0.50))

    if args.source_r8b_checkpoint:
        source_checkpoint = resolve_path(args.source_r8b_checkpoint)
        source_summary_path = source_checkpoint.parent / "summary.json"
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
        source_candidates = [str(source_summary_path)]
    else:
        source_meta, source_candidates_raw = discover_formal_r8b_checkpoint(ROOT)
        source_checkpoint = source_meta["checkpoint_path"]
        source_summary_path = source_meta["summary_path"]
        source_summary = source_meta["summary"]
        source_candidates = [str(path) for path in source_candidates_raw]

    h0_path = resolve_path(args.h0_results_json)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model, checkpoint = load_r8b(source_checkpoint, device)
    source_invariants = verify_r8b_invariants(model)

    data_cfg = copy.deepcopy(config.get("dataset", {"name": "cifar10", "root": "./data", "download": True}))
    data_cfg["root"] = str(resolve_path(data_cfg.get("root", "./data")))
    training_cfg = copy.deepcopy(config.get("training", {}))
    training_cfg.setdefault("batch_size", 128)
    training_cfg.setdefault("test_batch_size", 128)
    training_cfg["num_workers"] = 0
    train_loader, test_loader = build_r5t_loaders(
        {"dataset": data_cfg, "training": training_cfg}, smoke=False
    )

    source_r8b_acc = evaluate(model, test_loader, device)
    h0_reconstruction = apply_h0_int6_biases(model, h0_path)
    h0_bias_snapshot = snapshot_biases(model)
    h0_invariants = verify_r8b_invariants(model)

    adapter = H1ModelAdapter(model)
    if len(adapter.list_quant_nodes()) != len(set(adapter.list_quant_nodes())):
        raise RuntimeError("H1 adapter returned duplicate quantization nodes")
    print("H1 quantization nodes:", flush=True)
    for node in adapter.list_quant_nodes():
        print(f"  {node}", flush=True)

    adapter_equivalence = adapter_matches_model(adapter, test_loader, device)
    if not adapter_equivalence["matches"]:
        raise RuntimeError(f"H1 adapter changes the unquantized R8B graph: {adapter_equivalence}")

    adapter.set_observer(None)
    adapter.set_fake_quant(None)
    h0_acc = evaluate(adapter, test_loader, device)
    expected_h0 = float(config["source"]["expected_accuracy"])
    tolerance = float(config["source"].get("baseline_tolerance_pp", 0.08))
    if abs(h0_acc - expected_h0) > tolerance:
        raise RuntimeError(
            f"H0 baseline mismatch: got {h0_acc:.2f}% vs expected "
            f"{expected_h0:.2f}% +/- {tolerance:.2f} pp"
        )
    if abs(h0_acc - float(h0_reconstruction["recorded_h0_int6_accuracy"])) > 0.08:
        raise RuntimeError("H0 INT6 baseline does not reproduce H0 result within 0.08 pp")

    observer, calibrated_samples = calibrate(
        model, adapter, train_loader, device, calibration_samples
    )
    nodes = adapter.list_quant_nodes()
    calibration_ranges, shifts_by_width = calibration_shift_table(
        observer, nodes, widths, min_shift, max_shift
    )
    for name in nodes:
        calibration_ranges[name]["calibration_sample_count"] = calibrated_samples

    results = []
    for bits in widths:
        shift_map = shifts_by_width[str(bits)]
        fake_quant = IntegerActivationFakeQuant(bits, shift_map)
        adapter.set_observer(None)
        adapter.set_fake_quant(fake_quant)
        accuracy = evaluate(adapter, test_loader, device)
        alignment = adapter.verify_integer_residual_alignment(shift_map, bits)
        saturation = test_saturation_map(fake_quant, nodes)
        errors = test_error_map(fake_quant, nodes)
        verify_h0_biases_unchanged(model, h0_bias_snapshot)
        result = {
            "bits": bits,
            "accuracy": float(accuracy),
            "delta_vs_h0_pp": float(accuracy - h0_acc),
            "accuracy_drop_vs_h0_pp": float(h0_acc - accuracy),
            "shift_map": shift_map,
            "test_saturation_by_node": saturation,
            "test_total_saturation": int(sum(saturation.values())),
            "test_error_by_node": errors,
            "residual_alignment": alignment,
            "h0_int6_bias_fixed": True,
            "r8b_invariants_unchanged": True,
            "binary_conv_accumulators_quantized": False,
            "gap_accumulator_quantized": False,
            "fc_accumulator_quantized": False,
        }
        results.append(result)
        print(
            f"INT{bits}: accuracy={accuracy:.2f}% "
            f"delta_vs_H0={accuracy - h0_acc:+.2f} pp "
            f"saturation={result['test_total_saturation']}",
            flush=True,
        )

    int4 = next(item for item in results if item["bits"] == 4)
    int4_drop = h0_acc - int4["accuracy"]
    if int4_drop <= int3_threshold and 3 not in widths:
        bits = 3
        shift_map = observer.shifts_for_bits(bits, min_shift=min_shift, max_shift=max_shift)
        fake_quant = IntegerActivationFakeQuant(bits, shift_map)
        adapter.set_fake_quant(fake_quant)
        accuracy = evaluate(adapter, test_loader, device)
        alignment = adapter.verify_integer_residual_alignment(shift_map, bits)
        saturation = test_saturation_map(fake_quant, nodes)
        errors = test_error_map(fake_quant, nodes)
        verify_h0_biases_unchanged(model, h0_bias_snapshot)
        results.append({
            "bits": bits,
            "accuracy": float(accuracy),
            "delta_vs_h0_pp": float(accuracy - h0_acc),
            "accuracy_drop_vs_h0_pp": float(h0_acc - accuracy),
            "shift_map": shift_map,
            "test_saturation_by_node": saturation,
            "test_total_saturation": int(sum(saturation.values())),
            "test_error_by_node": errors,
            "residual_alignment": alignment,
            "h0_int6_bias_fixed": True,
            "r8b_invariants_unchanged": True,
            "binary_conv_accumulators_quantized": False,
            "gap_accumulator_quantized": False,
            "fc_accumulator_quantized": False,
        })
        widths.append(3)
        print(
            f"INT3: accuracy={accuracy:.2f}% "
            f"delta_vs_H0={accuracy - h0_acc:+.2f} pp "
            f"saturation={results[-1]['test_total_saturation']}",
            flush=True,
        )

    adapter.set_fake_quant(None)

    def lowest_with_drop(limit):
        feasible = [item for item in results if item["accuracy_drop_vs_h0_pp"] <= limit + 1e-9]
        return min(feasible, key=lambda item: item["bits"])["bits"] if feasible else None

    sensitivity = analyze_sensitivity(results, nodes)
    output_cfg = config.get("output", {})
    output_dir = resolve_path(args.output_dir or output_cfg.get("root", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / output_cfg.get("results", "H1_INTEGER_ACTIVATION_SWEEP_RESULTS.json")
    report_path = output_dir / output_cfg.get("report", "H1_INTEGER_ACTIVATION_RESIDUAL_REPORT.md")
    summary = {
        "experiment": "H1_integer_activation_residual_sweep",
        "status": "complete",
        "device": str(device),
        "config_path": str(config_path),
        "source_r8b_checkpoint": str(source_checkpoint),
        "source_summary": str(source_summary_path),
        "candidate_formal_summaries": source_candidates,
        "source_stored_best_acc": float(checkpoint.get("best_acc", source_summary.get("best_acc", float("nan")))),
        "source_stored_best_epoch": int(checkpoint.get("best_epoch", source_summary.get("best_epoch", -1))),
        "source_r8b_reload_accuracy": float(source_r8b_acc),
        "source_verification": source_invariants,
        "h0_int6_reconstruction": h0_reconstruction,
        "h0_int6_baseline_accuracy": float(h0_acc),
        "h0_invariants_after_reconstruction": h0_invariants,
        "adapter_equivalence": adapter_equivalence,
        "quant_nodes": adapter.quant_node_metadata(),
        "quant_nodes_order": nodes,
        "calibration": {
            "source": "CIFAR-10 train only",
            "test_used_for_calibration": False,
            "samples": calibrated_samples,
            "min_shift": min_shift,
            "max_shift": max_shift,
            "policy": "largest integer shift with zero calibration saturation from observed absmax",
            "right_shift_rounding": "nearest, ties away from zero",
        },
        "calibration_ranges": calibration_ranges,
        "tested_widths": sorted(set(widths), reverse=True),
        "results": results,
        "int4_drop_pp": float(int4_drop),
        "lowest_near_lossless_bits": lowest_with_drop(0.10),
        "recommended_bits": lowest_with_drop(0.30),
        "absolute_minimum_bits": lowest_with_drop(0.50),
        "sensitivity": sensitivity,
        "hardware_scope": {
            "h0_bias_bits": 6,
            "activation_residual_state_is_w_bit_integer": True,
            "binary_conv_accumulators_quantized": False,
            "gap_accumulator_quantized": False,
            "fc_accumulator_quantized": False,
            "final_logit_width_quantized": False,
            "architecture_changed": False,
        },
    }
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print(json.dumps({
        "h0_baseline_acc": h0_acc,
        "tested_widths": summary["tested_widths"],
        "lowest_near_lossless_bits": summary["lowest_near_lossless_bits"],
        "recommended_bits": summary["recommended_bits"],
        "absolute_minimum_bits": summary["absolute_minimum_bits"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
