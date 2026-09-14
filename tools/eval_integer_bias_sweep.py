"""H0 zero-shot integer bias sweep for the formal R8B checkpoint.

Only affine bias tensors are replaced in the evaluation copy. The source
checkpoint, all K/weight tensors, activation/residual/GAP behavior, and the
R8B model files are never modified.
"""

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
from recu_hw.integer_bias import (
    collect_integer_bias_targets,
    quantize_model_biases_inplace,
)
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.r6 import (
    is_power_of_two_tensor,
    parameter_freeze_report,
    stem_binary_weight_stats,
)
from recu_hw.r7 import R7ResNet20
from recu_hw.r8 import build_r8b_from_r7


def _resolve_path(value, base=ROOT):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def discover_formal_r8b_checkpoint(project_root=ROOT):
    """Select the non-smoke R8B run matching the formal 85.40% record."""
    candidates = []
    search_root = project_root / "experiments" / "recu_r8b"
    for summary_path in sorted(search_root.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = summary_path.parent / "best.pt"
        if (
            summary.get("variant") == "r8b"
            and abs(float(summary.get("best_acc", -999.0)) - 85.40) < 1e-6
            and int(summary.get("best_epoch", -1)) == 100
            and abs(float(summary.get("reload_acc", -999.0)) - 85.40) < 1e-6
            and checkpoint.exists()
            and "smoke" not in summary_path.parent.name.lower()
        ):
            candidates.append({
                "summary_path": summary_path,
                "checkpoint_path": checkpoint,
                "summary": summary,
            })

    if not candidates:
        raise FileNotFoundError(
            "No formal R8B checkpoint matching best=85.40%, epoch=100, "
            "reload=85.40% was found under experiments/recu_r8b."
        )
    candidates.sort(key=lambda item: item["checkpoint_path"].stat().st_mtime, reverse=True)
    return candidates[0], [item["summary_path"] for item in candidates]


def load_r8b_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("config", {})
    classifier = cfg.get("classifier", {}) if isinstance(cfg, dict) else {}
    source = R7ResNet20(num_classes=10, resolution=8)
    source.load_state_dict(checkpoint["model"], strict=True)
    source.conv1.set_binary_weight(True)
    source.stem_affine.quantize_k = True
    source.head_affine.quantize_k = True
    for module in source.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)

    model = build_r8b_from_r7(
        source,
        min_exp=classifier.get("min_exp"),
        max_exp=classifier.get("max_exp"),
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


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += int((model(x).argmax(1) == y).sum().item())
        total += int(y.size(0))
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def verify_r8b_invariants(model):
    errors = []
    stem = stem_binary_weight_stats(model)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        errors.append(f"stem invariant failed: {stem}")
    if not is_power_of_two_tensor(model.stem_affine.effective_k()):
        errors.append("stem K is not exact signed-pow2")
    if not is_power_of_two_tensor(model.head_affine.effective_k()):
        errors.append("head K is not exact signed-pow2")
    for name, module in model.named_modules():
        if isinstance(module, FusedAffine2d) and not is_power_of_two_tensor(module.effective_k()):
            errors.append(f"{name} K is not exact signed-pow2")
    fc_stats = model.linear.stats()
    if not fc_stats["is_exact_signed_pow2"]:
        errors.append("FC effective weight is not exact signed-pow2")
    if model.linear.bias is None:
        errors.append("FC bias is missing")
    if hasattr(model, "bn2"):
        errors.append("unexpected bn2 is present")
    freeze = parameter_freeze_report(model)
    if freeze["frozen_params"] != 672:
        errors.append(f"frozen parameter count changed: {freeze}")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        errors.append("model contains non-finite parameters")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "stem": stem,
        "stem_k_exact_signed_pow2": True,
        "backbone_fused_affine_count": 18,
        "backbone_k_exact_signed_pow2": True,
        "head_k_exact_signed_pow2": True,
        "fc": fc_stats,
        "fc_bias_present": True,
        "bn2_removed": True,
        "freeze_report": freeze,
        "finite": True,
    }


def build_bias_inventory(model):
    targets = collect_integer_bias_targets(model)
    return [
        {
            "module_name": name,
            "class_name": module.__class__.__name__,
            "param_count": int(module.bias.numel()),
            "original_min": float(module.bias.detach().min().item()),
            "original_max": float(module.bias.detach().max().item()),
        }
        for name, module in targets.items()
    ]


def verify_integer_export(model, export):
    """Check q/shift export and that the evaluation copy uses q*2^-shift."""
    targets = collect_integer_bias_targets(model)
    errors = []
    total_params = 0
    for name, record in export.items():
        if not isinstance(record["q_values"], list) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in record["q_values"]
        ):
            errors.append(f"{name}: q_values are not integer values")
        if any(value < record["qmin"] or value > record["qmax"] for value in record["q_values"]):
            errors.append(f"{name}: q value outside signed range")
        if record.get("q_dtype") != "int64":
            errors.append(f"{name}: exported q dtype is not int64")
        q = torch.tensor(record["q_values"], dtype=torch.float64, device=model.linear.bias.device)
        reconstructed = q * (2.0 ** (-int(record["shift"])))
        actual = targets[name].bias.detach().to(torch.float64)
        if not torch.allclose(reconstructed, actual, atol=2e-6, rtol=0.0):
            errors.append(f"{name}: simulation bias does not match q+shift reconstruction")
        total_params += len(record["q_values"])
    if total_params != 762:
        errors.append(f"integer export parameter count is {total_params}, expected 762")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "q_dtype": "int64",
        "q_values_are_signed_integers": True,
        "q_values_within_declared_ranges": True,
        "simulation_reconstructs_from_q_and_shift": True,
        "float_bias_storage_exported": False,
        "general_multiplier_for_bias_quantization": False,
        "floating_adder_for_bias_quantization": False,
        "exported_bias_scalar_count": total_params,
    }


def summarize_width(bits, accuracy, source_acc, export, export_verification):
    saturation = sum(item["saturation_count"] for item in export.values())
    params = sum(item["param_count"] for item in export.values())
    worst_name = max(export, key=lambda name: export[name]["mse"])
    return {
        "bits": int(bits),
        "accuracy": float(accuracy),
        "delta_vs_r8b_reload_pp": float(accuracy - source_acc),
        "accuracy_drop_vs_r8b_reload_pp": float(source_acc - accuracy),
        "total_bias_params": int(params),
        "total_saturation_count": int(saturation),
        "worst_layer_by_mse": worst_name,
        "worst_layer_mse": float(export[worst_name]["mse"]),
        "export_verification": export_verification,
        "layers": export,
    }


def _fmt(value, digits=6):
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(summary):
    cfg = summary["configuration"]
    lines = [
        "# H0 Integer Bias Sweep Report",
        "",
        "> Zero-shot pure-integer bias quantization on the formal R8B checkpoint. No fine-tuning was performed.",
        "",
        "## A. Configuration",
        "",
        "- Architecture: R8B unchanged; Thermometer `R=8`, raw RGB `[0,1]`, bipolar `{-1,+1}`.",
        f"- Bias widths tested: {', '.join(f'INT{b}' for b in summary['tested_widths'])}.",
        f"- Shift search: `s={cfg['shift_min']}..{cfg['shift_max']}`; negative extension to `{cfg['fallback_shift_min']}` only when the initial range has no zero-saturation candidate.",
        "- Per-layer encoding: `q = round(B * 2^s)`, signed integer `q`, hardware reconstruction `Bhat = q * 2^-s`.",
        "- Only `stem_affine.bias`, the 18 R4 fused-affine biases, `head_affine.bias`, and `linear.bias` were changed in evaluation copies.",
        "- K/weight tensors, activation, residual, GAP, accumulators, alpha, and architecture were not changed.",
        "",
        "## B. Formal R8B reference",
        "",
        f"- Checkpoint: `{summary['source_checkpoint']}`",
        f"- Summary: `{summary['source_summary']}`",
        f"- Stored best: **{summary['stored_best_acc']:.2f}%** @ epoch **{summary['stored_best_epoch']}**",
        f"- Actual full-test reload: **{summary['source_reload_acc']:.2f}%**",
        "- R8B invariant check: PASS; stem binary, all affine/head/FC K exact signed-pow2, FC bias present, 672 alpha parameters frozen, no BN2.",
        "",
        "### Bias target inventory",
        "",
        "| Module | Class | Parameters | Original min | Original max |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["bias_inventory"]:
        lines.append(
            f"| `{item['module_name']}` | `{item['class_name']}` | {item['param_count']} | "
            f"{item['original_min']:.8g} | {item['original_max']:.8g} |"
        )

    lines += [
        "",
        "## C. Accuracy sweep",
        "",
        "| Width | Accuracy | Delta vs R8B reload | Total saturation |",
        "|---:|---:|---:|---:|",
    ]
    for result in sorted(summary["results"], key=lambda item: -item["bits"]):
        lines.append(
            f"| INT{result['bits']} | {result['accuracy']:.2f}% | "
            f"{result['delta_vs_r8b_reload_pp']:+.2f} pp | {result['total_saturation_count']} |"
        )

    lines += [
        "",
        "## D. Per-layer integer export",
        "",
        "The complete integer parameter tables (`q_values`) are in `H0_INTEGER_BIAS_SWEEP_RESULTS.json`. The table below records every target, selected layer shift, range, saturation, MSE, and maximum absolute reconstruction error.",
        "",
    ]
    for result in sorted(summary["results"], key=lambda item: -item["bits"]):
        lines += [
            f"### INT{result['bits']}",
            "",
            "| Module | Shift | q range | Saturation | MSE | Max abs error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, item in result["layers"].items():
            lines.append(
                f"| `{name}` | {item['shift']} | [{item['qmin']}, {item['qmax']}] | "
                f"{item['saturation_count']} | {_fmt(item['mse'])} | {_fmt(item['max_abs_error'])} |"
            )
        lines.append("")

    lines += [
        "## E. Interpretation and decision",
        "",
        f"- Lowest near-lossless width (drop <= 0.10 pp): **{summary['lowest_near_lossless_bits'] or 'none'}**.",
        f"- Recommended production width (drop <= 0.30 pp): **{summary['recommended_bits'] or 'none'}**.",
        f"- Absolute minimum acceptable width (drop <= 0.50 pp): **{summary['absolute_minimum_bits'] or 'none'}**.",
        f"- INT4 drop was {summary['int4_drop_pp']:.2f} pp; optional INT3 rule threshold was {cfg['int3_threshold_pp']:.2f} pp, so INT3 was {'tested' if 3 in summary['tested_widths'] else 'not tested'}.",
        "",
        "This experiment answers the H0 question by measuring the accuracy cost of storing only affine offsets and the FC bias as signed integers with a per-layer power-of-two shift. It does not retrain or alter the R8B effective weights.",
        "",
        "## F. Hardware",
        "",
        "Inference architecture is unchanged from formal R8B.",
        "",
        "- No extra inference cost.",
        "- No extra BMAC and no extra parameters caused by this zero-shot representation change.",
        "- Stem remains pure binary convolution: XNOR + popcount, general multiplier = 0, convolution DSP = 0.",
        "- Bias hardware representation is integer `q` plus layer-shared power-of-two shift metadata; no floating-point bias storage, general multiplier, or floating-point adder is part of the exported H0 representation.",
        "",
        "## G. Reproducibility",
        "",
        "- Full CIFAR-10 test set was used for the R8B reference and every reported width.",
        "- No fine-tuning, KD, activation/residual/GAP/accumulator quantization, RTL, FPGA, or architecture change was performed.",
        "- Export verification: PASS for integer dtype, signed range, q+shift reconstruction, and unchanged R8B invariants.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "h0_integer_bias_sweep.json"))
    parser.add_argument("--source-r8b-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sweep_cfg = config.get("shift_search", {})
    shift_min = int(sweep_cfg.get("shift_min", 0))
    shift_max = int(sweep_cfg.get("shift_max", 12))
    fallback_shift_min = int(sweep_cfg.get("fallback_shift_min", -12))
    int3_threshold = float(config.get("optional_int3_if_int4_drop_le_pp", 0.50))
    widths = [int(bits) for bits in config.get("widths", [8, 7, 6, 5, 4])]

    if args.source_r8b_checkpoint:
        source_checkpoint = _resolve_path(args.source_r8b_checkpoint)
        source_summary = source_checkpoint.parent / "summary.json"
        source_meta = {
            "summary_path": source_summary,
            "checkpoint_path": source_checkpoint,
            "summary": json.loads(source_summary.read_text(encoding="utf-8")) if source_summary.exists() else {},
        }
        candidates = [source_summary]
    else:
        source_meta, candidates = discover_formal_r8b_checkpoint(ROOT)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, checkpoint = load_r8b_checkpoint(source_meta["checkpoint_path"], device)
    invariant_report = verify_r8b_invariants(model)

    dataset_cfg = copy.deepcopy(config.get("dataset", {"name": "cifar10", "root": "./data", "download": True}))
    dataset_cfg["root"] = str(_resolve_path(dataset_cfg.get("root", "./data")))
    training_cfg = copy.deepcopy(config.get("training", {}))
    training_cfg.setdefault("batch_size", 128)
    training_cfg.setdefault("test_batch_size", 128)
    training_cfg["num_workers"] = 0
    _, test_loader = build_r5t_loaders(
        {"dataset": dataset_cfg, "training": training_cfg}, smoke=False
    )

    source_acc = evaluate(model, test_loader, device)
    source_summary_data = source_meta["summary"]
    stored_best_acc = float(checkpoint.get("best_acc", source_summary_data.get("best_acc", float("nan"))))
    stored_best_epoch = int(checkpoint.get("best_epoch", source_summary_data.get("best_epoch", -1)))
    bias_inventory = build_bias_inventory(model)
    if sum(item["param_count"] for item in bias_inventory) != 762:
        raise RuntimeError(f"Unexpected integer bias inventory: {bias_inventory}")

    results = []
    for bits in widths:
        if bits < 3:
            raise ValueError(f"unsupported bias width: {bits}")
        quantized = copy.deepcopy(model)
        export = quantize_model_biases_inplace(
            quantized,
            bits=bits,
            shift_min=shift_min,
            shift_max=shift_max,
            fallback_shift_min=fallback_shift_min,
        )
        quantized_invariants = verify_r8b_invariants(quantized)
        export_verification = verify_integer_export(quantized, export)
        accuracy = evaluate(quantized, test_loader, device)
        result = summarize_width(bits, accuracy, source_acc, export, export_verification)
        result["invariants_unchanged"] = {
            "stem_binary": quantized_invariants["stem"]["is_strict_binary"],
            "all_k_signed_pow2": True,
            "fc_bias_present": True,
            "frozen_alpha_params": quantized_invariants["freeze_report"]["frozen_params"],
        }
        results.append(result)
        print(
            f"INT{bits}: accuracy={accuracy:.2f}% "
            f"delta_vs_R8B={accuracy - source_acc:+.2f} pp "
            f"saturation={result['total_saturation_count']}",
            flush=True,
        )
        del quantized
        if device.type == "cuda":
            torch.cuda.empty_cache()

    int4 = next(item for item in results if item["bits"] == 4)
    int4_drop = source_acc - int4["accuracy"]
    if int4_drop <= int3_threshold and 3 not in widths:
        bits = 3
        quantized = copy.deepcopy(model)
        export = quantize_model_biases_inplace(
            quantized,
            bits=bits,
            shift_min=shift_min,
            shift_max=shift_max,
            fallback_shift_min=fallback_shift_min,
        )
        quantized_invariants = verify_r8b_invariants(quantized)
        export_verification = verify_integer_export(quantized, export)
        accuracy = evaluate(quantized, test_loader, device)
        result = summarize_width(bits, accuracy, source_acc, export, export_verification)
        result["invariants_unchanged"] = {
            "stem_binary": quantized_invariants["stem"]["is_strict_binary"],
            "all_k_signed_pow2": True,
            "fc_bias_present": True,
            "frozen_alpha_params": quantized_invariants["freeze_report"]["frozen_params"],
        }
        results.append(result)
        widths.append(3)
        print(
            f"INT3: accuracy={accuracy:.2f}% "
            f"delta_vs_R8B={accuracy - source_acc:+.2f} pp "
            f"saturation={result['total_saturation_count']}",
            flush=True,
        )
        del quantized
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def lowest_with_drop(limit):
        feasible = [item for item in results if item["accuracy_drop_vs_r8b_reload_pp"] <= limit + 1e-9]
        return min(feasible, key=lambda item: item["bits"])["bits"] if feasible else None

    output_cfg = config.get("output", {})
    output_dir = _resolve_path(args.output_dir or output_cfg.get("root", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / output_cfg.get("results", "H0_INTEGER_BIAS_SWEEP_RESULTS.json")
    report_path = output_dir / output_cfg.get("report", "H0_INTEGER_BIAS_SWEEP_REPORT.md")
    source_summary_path = source_meta["summary_path"]

    summary = {
        "experiment": "H0_integer_bias_sweep",
        "status": "complete",
        "device": str(device),
        "config_path": str(config_path),
        "configuration": {
            "thermometer_R": 8,
            "input_representation": "CIFAR-10 raw RGB -> ToTensor [0,1] -> thermometer -> bipolar {-1,+1}",
            "shift_min": shift_min,
            "shift_max": shift_max,
            "fallback_shift_min": fallback_shift_min,
            "int3_threshold_pp": int3_threshold,
            "no_finetuning": True,
            "only_biases_quantized": True,
            "target_bias_parameter_count": 762,
            "target_tensor_count": 21,
        },
        "source_checkpoint": str(source_meta["checkpoint_path"]),
        "source_summary": str(source_summary_path),
        "candidate_formal_summaries": [str(path) for path in candidates],
        "stored_best_acc": stored_best_acc,
        "stored_best_epoch": stored_best_epoch,
        "source_reload_acc": source_acc,
        "source_invariants": invariant_report,
        "bias_inventory": bias_inventory,
        "tested_widths": sorted(set(widths), reverse=True),
        "results": results,
        "int4_drop_pp": float(int4_drop),
        "lowest_near_lossless_bits": lowest_with_drop(0.10),
        "recommended_bits": lowest_with_drop(0.30),
        "absolute_minimum_bits": lowest_with_drop(0.50),
        "hardware": {
            "architecture_unchanged": True,
            "bmac_unchanged": True,
            "stem_bmac": 14155776,
            "backbone_bmac": 40108032,
            "total_bmac_per_image": 54263808,
            "stem_general_multiplier": 0,
            "stem_convolution_dsp": 0,
            "bias_export_has_float_storage": False,
        },
    }
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote {results_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print(json.dumps({
        "source_reload_acc": source_acc,
        "tested_widths": summary["tested_widths"],
        "lowest_near_lossless_bits": summary["lowest_near_lossless_bits"],
        "recommended_bits": summary["recommended_bits"],
        "absolute_minimum_bits": summary["absolute_minimum_bits"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
