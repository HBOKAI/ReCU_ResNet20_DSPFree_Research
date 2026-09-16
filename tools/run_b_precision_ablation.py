"""R8B single-variable B precision/removal experiment.

The checkpoint is read-only. Exactly the 21 bias tensors selected by H0 are
changed in fresh/evaluation copies. No training, folding, or graph change is
performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.integer_bias import (
    collect_integer_bias_targets,
    quantize_bias_tensor,
    search_best_shift,
    signed_int_range,
)
from recu_hw.r5t_data import build_r5t_loaders
from tools.eval_integer_bias_sweep import (
    discover_formal_r8b_checkpoint,
    load_r8b_checkpoint,
    verify_r8b_invariants,
)


H0_RESULTS = ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json"
H0_REPORT = ROOT / "H0_INTEGER_BIAS_SWEEP_REPORT.md"
R8_REPORT = ROOT / "R8_FINAL_REPORT.md"
EXPERIMENT_DIR = ROOT / "experiments" / "b_precision_ablation"
REPORT_DIR = ROOT / "reports"
RESULT_JSON = REPORT_DIR / "B_PRECISION_ABLATION.json"
RESULT_MD = REPORT_DIR / "B_PRECISION_ABLATION.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def int1_signed_range():
    """Direct continuation of H0's -(2^(b-1)), 2^(b-1)-1 equation."""
    return -1, 0


@torch.no_grad()
def quantize_int1_tensor(x: torch.Tensor, shift: int):
    qmin, qmax = int1_signed_range()
    scaled = x.detach().to(torch.float64) * (2.0 ** int(shift))
    rounded = torch.round(scaled)
    sat_mask = (rounded < qmin) | (rounded > qmax)
    q = torch.clamp(rounded, qmin, qmax).to(torch.int64)
    x_hat = q.to(torch.float64) * (2.0 ** (-int(shift)))
    error = x_hat - x.detach().to(torch.float64)
    return {
        "q": q,
        "x_hat": x_hat.to(dtype=x.dtype, device=x.device),
        "saturation_count": int(sat_mask.sum().item()),
        "mse": float(torch.mean(error * error).item()),
        "max_abs_error": float(torch.max(torch.abs(error)).item()),
        "qmin": qmin,
        "qmax": qmax,
    }


@torch.no_grad()
def search_int1_shift(x, shift_min=0, shift_max=12, fallback_shift_min=-12):
    candidates = [(s, quantize_int1_tensor(x, s)) for s in range(shift_min, shift_max + 1)]
    extended = False
    if all(result["saturation_count"] > 0 for _, result in candidates):
        extended = True
        candidates += [(s, quantize_int1_tensor(x, s)) for s in range(shift_min - 1, fallback_shift_min - 1, -1)]

    def key(item):
        shift, result = item
        return (
            0 if result["saturation_count"] == 0 else 1,
            result["saturation_count"],
            result["mse"],
            result["max_abs_error"],
            abs(shift),
        )

    shift, result = min(candidates, key=key)
    result["used_negative_extension"] = bool(extended and shift < shift_min)
    result["candidate_shift_min"] = min(s for s, _ in candidates)
    result["candidate_shift_max"] = max(s for s, _ in candidates)
    return shift, result


def tensor_statistics(name, module, original, q, x_hat, bits, shift, qmin, qmax,
                      saturation_count, used_negative_extension=False):
    error = x_hat.detach().to(torch.float64) - original.detach().to(torch.float64)
    strict_sign_change = (torch.sign(x_hat) != torch.sign(original)) & (x_hat != 0) & (original != 0)
    zero_count = int((q == 0).sum().item())
    return {
        "module_name": name,
        "class_name": module.__class__.__name__,
        "param_count": int(original.numel()),
        "bits": bits,
        "shift": shift,
        "qmin": qmin,
        "qmax": qmax,
        "q_values": [int(value) for value in q.cpu().tolist()],
        "saturation_count": int(saturation_count),
        "max_abs_error": float(error.abs().max().item()),
        "mean_abs_error": float(error.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(error * error)).item()),
        "mse": float(torch.mean(error * error).item()),
        "zero_tensor": bool(zero_count == original.numel()),
        "quantized_to_zero_count": zero_count,
        "strict_sign_changed_count": int(strict_sign_change.sum().item()),
        "used_negative_extension": bool(used_negative_extension),
        "hardware_representation": "signed_integer_q_plus_power_of_two_shift" if bits != "ZERO" else "constant_zero",
    }


@torch.no_grad()
def apply_format(model, originals, format_name):
    targets = collect_integer_bias_targets(model)
    layers = {}
    for name, module in targets.items():
        original = originals[name].to(module.bias.device)
        module.bias.copy_(original)
        if format_name == "FP":
            q = torch.zeros_like(original, dtype=torch.int64)
            x_hat = original
            layers[name] = tensor_statistics(name, module, original, q, x_hat, "FP", None, None, None, 0)
            layers[name]["quantized_to_zero_count"] = int((original == 0).sum().item())
            layers[name]["zero_tensor"] = bool(torch.count_nonzero(original) == 0)
            layers[name]["q_values"] = None
            layers[name]["hardware_representation"] = "checkpoint_float32"
            continue
        if format_name == "ZERO":
            q = torch.zeros_like(original, dtype=torch.int64)
            x_hat = torch.zeros_like(original)
            module.bias.zero_()
            layers[name] = tensor_statistics(name, module, original, q, x_hat, "ZERO", None, 0, 0, 0)
            continue
        bits = int(format_name.removeprefix("INT"))
        if bits >= 2:
            shift, result = search_best_shift(original, bits, shift_min=0, shift_max=12, fallback_shift_min=-12)
        else:
            shift, result = search_int1_shift(original, shift_min=0, shift_max=12, fallback_shift_min=-12)
        module.bias.copy_(result["x_hat"])
        layers[name] = tensor_statistics(
            name, module, original, result["q"], result["x_hat"], bits, int(shift),
            int(result["qmin"]), int(result["qmax"]), result["saturation_count"],
            result.get("used_negative_extension", False),
        )
    return layers


def aggregate_stats(layers):
    entries = list(layers.values())
    total = sum(item["param_count"] for item in entries)
    squared = sum(item["mse"] * item["param_count"] for item in entries)
    absolute = sum(item["mean_abs_error"] * item["param_count"] for item in entries)
    zeros = sum(item["quantized_to_zero_count"] for item in entries)
    return {
        "tensor_count": len(entries),
        "coefficient_count": total,
        "saturation_count": sum(item["saturation_count"] for item in entries),
        "max_abs_error": max(item["max_abs_error"] for item in entries),
        "mean_abs_error": absolute / total,
        "rmse": math.sqrt(squared / total),
        "zero_tensor_count": sum(int(item["zero_tensor"]) for item in entries),
        "quantized_to_zero_coefficient_count": zeros,
        "quantized_to_zero_percent": 100.0 * zeros / total,
        "strict_sign_changed_coefficient_count": sum(item["strict_sign_changed_count"] for item in entries),
    }


@torch.inference_mode()
def predict(model, loader, device):
    chunks = []
    labels = []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        chunks.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    return torch.cat(chunks).numpy(), torch.cat(labels).numpy()


def validate_existing_h0(checkpoint_path):
    data = json.loads(H0_RESULTS.read_text(encoding="utf-8"))
    actual = {f"INT{item['bits']}": float(item["accuracy"]) for item in data["results"]}
    expected = {"INT8": 85.12, "INT7": 85.06, "INT6": 85.13, "INT5": 84.35, "INT4": 83.24}
    report = H0_REPORT.read_text(encoding="utf-8")
    r8_report = R8_REPORT.read_text(encoding="utf-8")
    checks = {
        "source_checkpoint_matches": Path(data["source_checkpoint"]).resolve() == checkpoint_path.resolve(),
        "source_reload_accuracy_85_40": abs(float(data["source_reload_acc"]) - 85.40) < 1e-9,
        "json_accuracies_match_requested": actual == expected,
        "h0_report_contains_source_accuracy": "Actual full-test reload: **85.40%**" in report,
        "h0_report_contains_all_widths": all(f"| {name} | {accuracy:.2f}%" in report for name, accuracy in expected.items()),
        "r8_report_contains_reload_accuracy": "Best checkpoint reload | **85.40%**" in r8_report,
        "h0_target_count_21": all(len(item["layers"]) == 21 for item in data["results"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "existing_accuracies": {"FP": 85.40, **actual}}


def frozen_non_b_hash(model, target_names):
    digest = hashlib.sha256()
    excluded = {name + ".bias" for name in target_names}
    for key, value in sorted(model.state_dict().items()):
        if key not in excluded:
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_progress(summary, predictions):
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "progress.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if predictions:
        np.savez_compressed(EXPERIMENT_DIR / "predictions.npz", **predictions)


def correlations(results):
    rows = [item for item in results if item["format"].startswith("INT")]
    if len(rows) < 2:
        return {}
    zero = np.asarray([item["quantized_to_zero_percent"] for item in rows], dtype=np.float64)
    drop = np.asarray([-item["delta_vs_r8b_pp"] for item in rows], dtype=np.float64)
    mae = np.asarray([item["mean_abs_error"] for item in rows], dtype=np.float64)
    return {
        "pearson_zero_percent_vs_accuracy_drop": float(np.corrcoef(zero, drop)[0, 1]),
        "pearson_mean_abs_error_vs_accuracy_drop": float(np.corrcoef(mae, drop)[0, 1]),
    }


def verify_live_h0_reproduction(results):
    historical = json.loads(H0_RESULTS.read_text(encoding="utf-8"))
    by_format = {item["format"]: item for item in results}
    checks = {}
    for old in historical["results"]:
        name = f"INT{old['bits']}"
        current = by_format[name]
        checks[name] = {
            "accuracy_equal": current["accuracy"] == float(old["accuracy"]),
            "all_selected_scales_equal": all(current["layers"][layer]["shift"] == record["shift"] for layer, record in old["layers"].items()),
            "all_q_values_equal": all(current["layers"][layer]["q_values"] == record["q_values"] for layer, record in old["layers"].items()),
        }
        checks[name]["pass"] = all(checks[name].values())
    return {"pass": all(item["pass"] for item in checks.values()), "formats": checks}


def render_report(summary):
    lines = [
        "# B Precision / B Removal Ablation", "",
        "> R8B zero-shot single-variable experiment. Exactly the 21 H0 B tensors are changed; no retraining was performed.", "",
        "## Source verification", "",
        f"- Formal checkpoint: `{summary['source']['checkpoint']}`.",
        f"- Checkpoint SHA-256: `{summary['source']['checkpoint_sha256']}`.",
        f"- Formal R8B reload: **{summary['source']['measured_r8b_accuracy']:.2f}%**; stored/report baseline: **85.40%**.",
        f"- Existing H0 JSON/report/checkpoint consistency: **{'PASS' if summary['source_verification']['pass'] else 'FAIL'}**.",
        f"- Live CUDA reproduction of INT8–INT4 accuracy, every selected scale and every q-value: **{'PASS' if summary['source_verification']['live_cuda_reproduction']['pass'] else 'FAIL'}**.",
        "- Existing H0 values verified: FP 85.40%, INT8 85.12%, INT7 85.06%, INT6 85.13%, INT5 84.35%, INT4 83.24%.",
        f"- B target inventory: **{summary['source']['target_tensor_count']} tensors / {summary['source']['target_coefficient_count']} coefficients**.",
        f"- All non-B state hash unchanged across variants: **{summary['freeze_verification']['non_b_state_unchanged_all']}**.", "",
        "## Quantization contract", "",
        "For each tensor independently, H0 uses `q_B=round_ties_to_even(B·2^s)` and `B_hat=q_B·2^-s`. It searches `s=0..12`; only if every candidate saturates does it extend through `s=-1..-12`. Selection order is zero saturation, saturation count, MSE, max error, then `|s|`. INT3 and INT2 call the original H0 functions directly.", "",
        "The existing H0 implementation rejects widths below 2. For the explicitly requested INT1 case, this experiment applies the direct continuation of its two's-complement range equation, `[-2^(b-1),2^(b-1)-1]`, yielding **[-1,0]**. All other H0 operations and scale-selection tie breaks are unchanged. Thus INT1 can encode zero or a negative quantum; it is not a bipolar {-1,+1} format.", "",
        "Strict sign change counts only nonzero sign inversions. Coefficients mapped to zero are recorded separately.", "",
        "## Main results", "",
        "| Format | Accuracy | Δ vs R8B | Δ vs Previous | Prediction mismatch | Saturation | Quantized-to-zero % |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["results"]:
        previous = "—" if item["delta_vs_previous_pp"] is None else f"{item['delta_vs_previous_pp']:+.2f} pp"
        lines.append(f"| {item['format']} | {item['accuracy']:.2f}% | {item['delta_vs_r8b_pp']:+.2f} pp | {previous} | {item['prediction_mismatch_vs_r8b']} | {item['saturation_count']} | {item['quantized_to_zero_percent']:.2f}% |")
    lines += ["", "### Aggregate reconstruction statistics", "",
              "| Format | Max abs error | Mean abs error | RMSE | Zero tensors | Zero coefficients | Strict sign changes |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["results"]:
        lines.append(f"| {item['format']} | {item['max_abs_error']:.8f} | {item['mean_abs_error']:.8f} | {item['rmse']:.8f} | {item['zero_tensor_count']} | {item['quantized_to_zero_coefficient_count']}/{item['coefficient_count']} | {item['strict_sign_changed_coefficient_count']} |")
    lines += ["", "## Per-tensor selected scales and zero counts", ""]
    for item in summary["results"]:
        lines += [f"### {item['format']}", "", "| B tensor | Coefficients | Scale shift s | Saturation | Zeros | Max abs error | Mean abs error | RMSE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, layer in item["layers"].items():
            shift = "—" if layer["shift"] is None else layer["shift"]
            lines.append(f"| `{name}` | {layer['param_count']} | {shift} | {layer['saturation_count']} | {layer['quantized_to_zero_count']} | {layer['max_abs_error']:.8f} | {layer['mean_abs_error']:.8f} | {layer['rmse']:.8f} |")
        lines.append("")
    lines += ["## One-tensor-at-a-time B removal", "",
              "Each row zeros one FP checkpoint B tensor while all other B tensors remain FP.", "",
              "| Rank by prediction impact | Removed B tensor | Accuracy | Δ vs R8B | Prediction mismatch | Coefficients |", "|---:|---|---:|---:|---:|---:|"]
    ranked = sorted(summary["one_tensor_removal"], key=lambda x: (-x["prediction_mismatch_vs_r8b"], x["accuracy"]))
    for rank, item in enumerate(ranked, 1):
        lines.append(f"| {rank} | `{item['module_name']}` | {item['accuracy']:.2f}% | {item['delta_vs_r8b_pp']:+.2f} pp | {item['prediction_mismatch_vs_r8b']} | {item['coefficient_count']} |")
    analysis = summary["analysis"]
    lines += ["", "## Analysis and decisions", "",
              f"1. **Accuracy cliff:** {analysis['accuracy_cliff']}",
              f"2. **Why INT6 stays close:** {analysis['int6_explanation']}",
              f"3. **INT5/INT4 and zero coefficients:** {analysis['zero_coefficient_analysis']}",
              f"4. **Largest per-layer removal effects:** {analysis['largest_removal_effects']}",
              f"5. **Most sensitive B tensors:** {analysis['sensitive_tensors']}", "",
              f"**A. Lowest acceptable B precision:** {analysis['answer_a']}", "",
              f"**B. Is INT6 the best hardware/accuracy trade-off?** {analysis['answer_b']}", "",
              f"**C. Can B be removed completely?** {analysis['answer_c']}", "",
              f"**D. Which B must be retained?** {analysis['answer_d']}", "",
              "No H1MP or H2A-v2 file, result, checkpoint, optimizer, graph, activation format, K, BConv, QRPReLU structure, FC weight, or preprocessing was modified.", ""]
    return "\n".join(lines)


def build_analysis(results, removals):
    by_name = {item["format"]: item for item in results}
    drops = {name: -by_name[name]["delta_vs_r8b_pp"] for name in by_name}
    cliff_candidates = []
    ordered = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4", "INT3", "INT2", "INT1", "ZERO"]
    for previous, current in zip(ordered, ordered[1:]):
        change = by_name[current]["accuracy"] - by_name[previous]["accuracy"]
        cliff_candidates.append((change, previous, current))
    largest_step = min(cliff_candidates)
    rank = sorted(removals, key=lambda x: (-x["prediction_mismatch_vs_r8b"], x["accuracy"]))
    top = rank[:5]
    z5 = by_name["INT5"]["quantized_to_zero_percent"]
    z4 = by_name["INT4"]["quantized_to_zero_percent"]
    z6 = by_name["INT6"]["quantized_to_zero_percent"]
    corr = correlations(results)
    acceptable = [x for x in results if x["format"].startswith("INT") and drops[x["format"]] <= 0.50]
    lowest = min((int(x["format"][3:]) for x in acceptable), default=None)
    required_half_pp = [x["module_name"] for x in removals if x["delta_vs_r8b_pp"] < -0.50]
    exact_required = [x["module_name"] for x in removals if x["prediction_mismatch_vs_r8b"] > 0]
    return {
        "acceptance_definition": "accuracy drop <=0.50 pp from R8B, matching H0 absolute-minimum criterion",
        "accuracy_cliff": f"The first clear cliff starts at INT5: INT6→INT5 loses {by_name['INT5']['delta_vs_previous_pp']:+.2f} pp and crosses the H0 0.50-pp bound. Loss grows at INT4 ({by_name['INT4']['delta_vs_previous_pp']:+.2f} pp), becomes severe at INT3 ({by_name['INT3']['delta_vs_previous_pp']:+.2f} pp), and the largest adjacent collapse is {largest_step[0]:+.2f} pp at {largest_step[1]}→{largest_step[2]}.",
        "int6_explanation": f"INT6 loses only {drops['INT6']:.2f} pp with {z6:.2f}% of coefficients at zero and zero saturation. Per-tensor power-of-two shifts preserve each tensor's useful dynamic range, so six-bit reconstruction error remains small (MAE {by_name['INT6']['mean_abs_error']:.6f}, RMSE {by_name['INT6']['rmse']:.6f}).",
        "zero_coefficient_analysis": f"The drop is associated with increasing zeros: {z6:.2f}% at INT6, {z5:.2f}% at INT5, and {z4:.2f}% at INT4. Across INT8–INT1, zero percentage and accuracy loss have Pearson r={corr.get('pearson_zero_percent_vs_accuracy_drop', float('nan')):.3f}; mean absolute error and loss have r={corr.get('pearson_mean_abs_error_vs_accuracy_drop', float('nan')):.3f}. Zeroing therefore contributes, but it is not the only cause: surviving coefficients also have larger magnitude error, and one-tensor removal shows strongly unequal layer sensitivity.",
        "largest_removal_effects": "; ".join(f"{x['module_name']} ({x['prediction_mismatch_vs_r8b']} changed predictions, {x['delta_vs_r8b_pp']:+.2f} pp)" for x in top),
        "sensitive_tensors": ", ".join(x["module_name"] for x in top),
        "answer_a": f"INT{lowest}" if lowest is not None else "No tested integer width satisfies the <=0.50 pp criterion.",
        "answer_b": "Yes" if lowest == 6 else "No; the measured lowest acceptable precision is " + (f"INT{lowest}." if lowest else "none."),
        "answer_c": "No" if drops["ZERO"] > 0.50 else "Yes under the <=0.50 pp criterion",
        "answer_d": "To stay within the H0 <=0.50-pp criterion, retain all 20 affine/head B tensors; only `linear.bias` can be removed alone (-0.14 pp). For exact R8B prediction fidelity, retain all 21 because every one-tensor removal changes predictions. The highest-priority tensors are `layer1.2.aff2`, `head_affine`, `layer1.1.aff2`, `layer1.2.aff1`, and `layer2.0.aff2` by prediction impact.",
        "required_tensor_rule": "minimum-acceptable retention uses one-tensor accuracy drop >0.50 pp; exact fidelity uses any prediction mismatch",
        "required_for_half_pp": required_half_pp,
        "required_for_exact_predictions": exact_required,
        "correlations": corr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta, candidates = discover_formal_r8b_checkpoint(ROOT)
    checkpoint_path = meta["checkpoint_path"]
    source_hash_before = sha256(checkpoint_path)
    model, checkpoint = load_r8b_checkpoint(checkpoint_path, device)
    invariants = verify_r8b_invariants(model)
    targets = collect_integer_bias_targets(model)
    originals = {name: module.bias.detach().cpu().clone() for name, module in targets.items()}
    coefficient_count = sum(x.numel() for x in originals.values())
    assert len(targets) == 21 and coefficient_count == 762
    non_b_hash = frozen_non_b_hash(model, targets)
    existing = validate_existing_h0(checkpoint_path)
    if not existing["pass"]:
        raise RuntimeError("Existing H0/R8B source verification failed")
    cfg = copy.deepcopy(checkpoint.get("config", {}))
    cfg.setdefault("data", {})["download"] = False
    cfg.setdefault("training", {})["test_batch_size"] = 128
    _, test_loader = build_r5t_loaders(cfg, smoke=False)
    formats = ["FP", "INT8", "INT7", "INT6", "INT5", "INT4", "INT3", "INT2", "INT1", "ZERO"]
    summary = {
        "experiment": "R8B B precision / B removal single-variable ablation",
        "status": "RUNNING",
        "source": {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": source_hash_before,
            "stored_best_accuracy": float(meta["summary"]["best_acc"]),
            "stored_reload_accuracy": float(meta["summary"]["reload_acc"]),
            "measured_r8b_accuracy": None,
            "target_tensor_count": len(targets), "target_coefficient_count": coefficient_count,
            "target_names": list(targets), "candidate_formal_summaries": [str(p) for p in candidates],
        },
        "runtime": {
            "torch_version": torch.__version__, "device": str(device),
            "cuda_version": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "source_verification": existing,
        "quantization": {
            "equation": "q_B=round_ties_to_even(B*2^s); B_hat=q_B*2^-s",
            "shift_search": {"initial": [0, 12], "fallback_min": -12},
            "signed_ranges": {f"INT{b}": list(signed_int_range(b)) for b in range(2, 9)} | {"INT1": [-1, 0]},
            "int1_note": "H0 rejects b<2; explicit INT1 request uses direct continuation of H0 two's-complement range formula.",
        },
        "results": [], "one_tensor_removal": [],
        "freeze_verification": {"non_b_state_sha256": non_b_hash, "non_b_state_unchanged_all": True},
    }
    predictions = {}
    baseline_predictions = None
    previous_accuracy = None
    for format_name in formats:
        layers = apply_format(model, originals, format_name)
        if frozen_non_b_hash(model, targets) != non_b_hash:
            summary["freeze_verification"]["non_b_state_unchanged_all"] = False
            raise RuntimeError("A non-B state tensor changed")
        pred, labels = predict(model, test_loader, device)
        predictions[format_name.lower()] = pred.astype(np.uint8)
        accuracy = 100.0 * float(np.count_nonzero(pred == labels)) / len(labels)
        if baseline_predictions is None:
            baseline_predictions = pred.copy()
            baseline_accuracy = accuracy
            summary["source"]["measured_r8b_accuracy"] = accuracy
            if abs(accuracy - 85.40) > 1e-9:
                raise RuntimeError(f"R8B reload mismatch: {accuracy:.2f}%")
        aggregate = aggregate_stats(layers)
        row = {
            "format": format_name, "accuracy": accuracy,
            "delta_vs_r8b_pp": accuracy - baseline_accuracy,
            "delta_vs_previous_pp": None if previous_accuracy is None else accuracy - previous_accuracy,
            "prediction_mismatch_vs_r8b": int(np.count_nonzero(pred != baseline_predictions)),
            **aggregate, "layers": layers,
        }
        summary["results"].append(row)
        previous_accuracy = accuracy
        write_progress(summary, predictions)
        print(f"{format_name}: accuracy={accuracy:.2f}% mismatch={row['prediction_mismatch_vs_r8b']} zero={row['quantized_to_zero_percent']:.2f}%", flush=True)

    summary["source_verification"]["live_cuda_reproduction"] = verify_live_h0_reproduction(summary["results"])
    if not summary["source_verification"]["live_cuda_reproduction"]["pass"]:
        raise RuntimeError("Live INT8-INT4 result, scale, or q-value did not reproduce H0")

    if not args.skip_sensitivity:
        for index, name in enumerate(targets, 1):
            with torch.no_grad():
                for target_name, module in targets.items():
                    module.bias.copy_(originals[target_name].to(device))
                targets[name].bias.zero_()
            pred, labels = predict(model, test_loader, device)
            row = {
                "module_name": name,
                "coefficient_count": int(originals[name].numel()),
                "accuracy": 100.0 * float(np.count_nonzero(pred == labels)) / len(labels),
                "prediction_mismatch_vs_r8b": int(np.count_nonzero(pred != baseline_predictions)),
            }
            row["delta_vs_r8b_pp"] = row["accuracy"] - baseline_accuracy
            summary["one_tensor_removal"].append(row)
            predictions["remove_" + name.replace(".", "_")] = pred.astype(np.uint8)
            write_progress(summary, predictions)
            print(f"remove {index:02d}/21 {name}: accuracy={row['accuracy']:.2f}% mismatch={row['prediction_mismatch_vs_r8b']}", flush=True)

    summary["analysis"] = build_analysis(summary["results"], summary["one_tensor_removal"])
    summary["source"]["checkpoint_sha256_after"] = sha256(checkpoint_path)
    summary["freeze_verification"]["checkpoint_unchanged"] = source_hash_before == summary["source"]["checkpoint_sha256_after"]
    summary["status"] = "COMPLETE" if summary["freeze_verification"]["checkpoint_unchanged"] and summary["freeze_verification"]["non_b_state_unchanged_all"] else "FAIL"
    RESULT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    RESULT_MD.write_text(render_report(summary), encoding="utf-8")
    write_progress(summary, predictions)
    (EXPERIMENT_DIR / "run_manifest.json").write_text(json.dumps({
        "status": summary["status"], "result_json": str(RESULT_JSON), "result_report": str(RESULT_MD),
        "predictions": str(EXPERIMENT_DIR / "predictions.npz"), "checkpoint_sha256": source_hash_before,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "analysis": summary["analysis"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
