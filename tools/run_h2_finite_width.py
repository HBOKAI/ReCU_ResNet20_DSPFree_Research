"""Run the H2 finite-width anchor and TRAIN-only accumulator-width search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.h2_integer_accum import binary_dot_exact_width
from recu_hw.h2_workspace import (
    build_h2_loaders,
    build_h2_trace_adapter,
    derive_exact_width_plan,
    evaluate,
    evaluate_exact_width_plan,
    load_r8b_h1mp_model,
    search_remaining_widths,
    verify_gap_deferred_scaling,
)
from tools.eval_integer_bias_sweep import discover_formal_r8b_checkpoint


def _path(value, default):
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _fmt_range(item):
    return f"[{item.get('min', item.get('aligned_min'))}, {item.get('max', item.get('aligned_max'))}]"


def render_report(result):
    source = result["source"]
    binary = result["binary_exact"]
    h2a = result["h2a_exact_width_anchor"]
    plan = result["h2a_plan"]
    h2b = result.get("h2b_search")
    final = result.get("final_official_test")
    deferred = bool(plan.get("gap", {}).get("division_mode") == "deferred_scale_metadata")
    title = "# H2A-v2 GAP Deferred Scaling Report" if deferred else "# H2 Finite-Width Accumulator Report"
    lines = [
        title,
        "",
        "> H2 closes finite-width accumulator, GAP, FC, bias-add, and logit arithmetic around the frozen R8B-H1MP plan. No retraining was performed.",
        "",
        "## 1. Source verification and frozen scope",
        "",
        f"- R8B checkpoint: `{source['r8b_checkpoint']}`",
        f"- H0 INT6 bias export: `{source['h0_results_json']}`",
        f"- Frozen H1MP plan: `{source['h1mp_results_json']}`",
        f"- H1MP source official-test accuracy reproduced: **{source['baseline_official_test_accuracy']:.2f}%**; expected 85.03% ±0.08 pp.",
        "- H0 INT6 bias, H1MP 58-node bits/shifts/policies, binary weights, signed-pow2 K, FC weights, Thermometer R=8, and architecture were not modified.",
        "- Official TEST was evaluation-only after the H2A plan was frozen; it was not used for any width decision.",
        "- H2B was not run in this H2A-v2 experiment by design.",
        "",
        "## 2. Binary-convolution exact widths",
        "",
        "| Path | Terms | Popcount width | Signed accumulator | Mathematical range | Overflow |",
        "|---|---:|---:|---:|---|---:|",
    ]
    binary_order = [("stem_cin96", "Stem Cin=96"), ("cin16", "Cin=16"), ("cin32", "Cin=32"), ("cin64", "Cin=64")]
    for key, label in binary_order:
        item = binary[key]
        lines.append(
            f"| {label} | {item['terms']} | INT{item['popcount_bits']} | INT{item['signed_acc_bits']} | "
            f"[{item['signed_min']}, {item['signed_max']}] | 0 |"
        )
    lines += [
        "",
        "## 3. H2A-v2 exact/conservative plan" if deferred else "## 3. H2A exact/conservative plan",
        "",
        "### Pow2-affine intermediate widths",
        "",
        "| H1 boundary | Width | Integer shift | Range | Basis |",
        "|---|---:|---:|---|---|",
    ]
    for name, item in plan["affine_intermediates"].items():
        lines.append(f"| `{name}` | INT{item['bits']} | {item['shift']} | {_fmt_range(item)} | {item['mathematical_basis']} |")
    lines += [
        "",
        "### Residual alignment/add intermediates",
        "",
        "| Add | Branch A | Branch B | Output shift | Width | Range |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in plan["residuals"].values():
        lines.append(
            f"| `{item['add_node']}` | INT{item['branch_a_bits']}/{item['branch_a_shift']} | "
            f"INT{item['branch_b_bits']}/{item['branch_b_shift']} | {item['output_shift']} | INT{item['bits']} | "
            f"[{item['aligned_min']}, {item['aligned_max']}] |"
        )
    lines += [
        "",
        "### QRPReLU integer intermediates",
        "",
        "| Block | Input | Parameter shift | Inner add | Negative/output width | Output shift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in plan["qrprelu"].items():
        lines.append(
            f"| `{name}` | INT{item['input_bits']} | {item['param_shift']} | INT{item['inner_bits']} | "
            f"INT{item['bits']} | {item['output_shift']} |"
        )
    gap = plan["gap"]
    lines += [
        "",
        "### GAP",
        "",
        f"- Input: INT{gap['input_bits']} at shift {gap['input_shift']}, 64 values (8x8).",
        f"- Sum accumulator: INT{gap['sum_bits']} with mathematical range [{gap['sum_min']}, {gap['sum_max']}].",
        (f"- Divide-by-64: **deferred scale metadata**; q_gap = q_sum (no arithmetic right shift and no integer rounding), with s_gap = s_input + 6."
         if deferred else f"- Divide-by-64: arithmetic right shift 6 with **{gap['rounding']}**."),
        (f"- GAP output: q_gap remains the INT{gap['output_bits']} sum representation at scale shift {gap['output_shift']}; the /64 factor is metadata-only."
         if deferred else f"- GAP output: INT{gap['output_bits']} at the inherited H1MP shift."),
        "",
        "### FC / bias-add / final logits",
        "",
        f"- FC input: INT{plan['fc']['input_bits']} at shift {plan['fc']['input_shift']}; common product scale shift {plan['fc']['common_shift']}.",
        "",
        "| Class | Worst-case min | Worst-case max | FC accumulator width |",
        "|---:|---:|---:|---:|",
    ]
    for item in plan["fc"]["per_class_worst_case"]:
        lines.append(f"| {item['class']} | {item['min']} | {item['max']} | INT{item['bits']} |")
    lines += [
        "",
        f"- H2A FC accumulator: **INT{plan['fc']['accumulator_bits']}**.",
        f"- FC INT6 bias-add: **INT{plan['fc']['bias_add_bits']}** at shift {plan['fc']['bias_add_shift']}.",
        f"- H2A final logits: **INT{plan['fc']['final_logits_bits']}**.",
        "",
        "## 4. H2A-v2 anchor" if deferred else "## 4. H2A anchor",
        "",
        f"- Accuracy: **{h2a['accuracy']:.2f}%**.",
        f"- Delta vs reproduced H1MP source: **{h2a['accuracy'] - source['baseline_official_test_accuracy']:+.2f} pp**.",
        f"- H2A-v2 gate |delta| <=0.10 pp: **{'PASS' if abs(h2a['accuracy'] - source['baseline_official_test_accuracy']) <= 0.10 else 'FAIL'}**." if deferred else f"- H2A gate |delta| <=0.10 pp: **{'PASS' if abs(h2a['accuracy'] - source['baseline_official_test_accuracy']) <= 0.10 else 'FAIL'}**.",
        f"- H2A overflow total: **{h2a['overflow_total']}**; saturation total including frozen H1MP boundaries: **{h2a['saturation_total']}**.",
        "",
    ]
    if deferred:
        verification = result.get("gap_scale_verification", {})
        lines += [
            "### Bit-accurate GAP verification",
            "",
            "- Reference: `(q_sum / 64) * 2^-s_input`.",
            "- Integer datapath interpretation: `q_sum * 2^-(s_input+6)`.",
            f"- Mathematical equality: **{verification.get('mathematically_equal', False)}**; max absolute error: **{verification.get('max_abs_error', float('nan')):.3e}**.",
            f"- Integer rounding applied by GAP division: **{verification.get('integer_rounding_applied', True)}**.",
            "- Explicit implementation statement: GAP division by 64 is implemented as a scale metadata adjustment, not arithmetic right-shift rounding.",
            "",
        ]
    if h2b is None:
        lines += [
            "## 5. H2B",
            "",
            "H2B was not run by design. H2A-v2 is a single-variable verification pass only.",
            "",
        ]
    else:
        lines += [
            "## 5. H2B TRAIN-only width optimization",
            "",
            f"- Search-validation baseline (H2A plan): **{h2b['validation_baseline_accuracy']:.2f}%**.",
            f"- Target drop: **{h2b['target_validation_drop_pp']:.2f} pp**; absolute maximum: **{h2b['absolute_max_validation_drop_pp']:.2f} pp**.",
            f"- Final search-validation accuracy: **{h2b['final_validation']['accuracy']:.2f}%**.",
            f"- Final search-validation drop: **{h2b['final_validation_drop_pp']:+.2f} pp**.",
            "- Search data: calibration and non-overlapping TRAIN-derived validation only; official TEST was excluded from all width decisions.",
            "",
            "| Step | Action | Field | Bits | Accuracy | Drop | Overflow |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
        for index, item in enumerate(h2b["history"]):
            field = item.get("field", "-")
            transition = "-"
            if "from_bits" in item:
                transition = f"{item['from_bits']}->{item['to_bits']}"
            lines.append(
                f"| {index} | `{item['action']}` | `{field}` | {transition} | {item.get('accuracy', 0.0):.2f}% | "
                f"{item.get('drop_pp', 0.0):+.2f} pp | {item.get('overflow_total', 0)} |"
            )
        lines += ["", "## 6. Final frozen H2 result", ""]
        if final is not None:
            lines += [
                f"- Final official TEST accuracy: **{final['accuracy']:.2f}%**.",
                f"- Delta vs H1MP 85.03%: **{final['accuracy'] - 85.03:+.2f} pp**.",
                f"- Delta vs reproduced H1MP source: **{final['accuracy'] - source['baseline_official_test_accuracy']:+.2f} pp**.",
                f"- Final overflow total: **{final['overflow_total']}**; saturation total including H1MP boundaries: **{final['saturation_total']}**.",
                f"- Final residual alignment records: **{len(final['residual_alignment']['records'])}/18**, equivalence all: **{final['residual_alignment']['integer_add_equivalence_all']}**.",
                "",
            ]
        lines += [
            "### Final optimized widths",
            "",
            f"- GAP sum: INT{h2b['final_plan']['gap']['sum_bits']}.",
            f"- FC accumulator: INT{h2b['final_plan']['fc']['accumulator_bits']}.",
            f"- FC bias-add: INT{h2b['final_plan']['fc']['bias_add_bits']}.",
            f"- Final logits: INT{h2b['final_plan']['fc']['final_logits_bits']}.",
            f"- QRPReLU remains conservative finite-width: maximum INT{max(item['bits'] for item in plan['qrprelu'].values())} output / INT{max(item['inner_bits'] for item in plan['qrprelu'].values())} inner.",
            "",
            "## 7. Claim discipline",
            "",
            "- All H2-listed inference accumulators/intermediates have explicit finite integer widths in the software datapath contract.",
            "- No unbounded Python integer is used as the production result; every finite node is checked and narrowed nodes use explicit saturation accounting.",
            "- The PyTorch implementation uses q*2^-shift float bridges at existing H1 boundaries for sign/clamp dispatch; these are representation bridges, not unspecified accumulator widths.",
            "- No RTL, FPGA synthesis, DSP48=0, or ASIC multiplier-cell claim is made by H2.",
            "",
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-r8b-checkpoint", default=None)
    ap.add_argument("--h0-results-json", default="H0_INTEGER_BIAS_SWEEP_RESULTS.json")
    ap.add_argument("--h1mp-results-json", default="H1MP_SEARCH_RESULTS.json")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--calibration-samples", type=int, default=10000)
    ap.add_argument("--search-validation-samples", type=int, default=10000)
    ap.add_argument(
        "--gap-deferred-v2",
        action="store_true",
        help="Run H2A-v2 only: represent GAP /64 in scale metadata and never run H2B.",
    )
    args = ap.parse_args()

    if args.source_r8b_checkpoint:
        source_checkpoint = _path(args.source_r8b_checkpoint, None)
    else:
        source_meta, _ = discover_formal_r8b_checkpoint(ROOT)
        source_checkpoint = source_meta["checkpoint_path"]
    h0_path = _path(args.h0_results_json, ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json")
    h1mp_path = _path(args.h1mp_results_json, ROOT / "H1MP_SEARCH_RESULTS.json")
    output_dir = _path(args.output_dir, ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "H2_FINITE_WIDTH_RESULTS.json"
    report_path = output_dir / "H2_FINITE_WIDTH_REPORT.md"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = load_r8b_h1mp_model(source_checkpoint, h0_path, h1mp_path, device)
    calib_loader, search_loader, test_loader = build_h2_loaders(
        seed=args.seed,
        calibration_samples=args.calibration_samples,
        search_validation_samples=args.search_validation_samples,
    )
    baseline_test = evaluate(model, test_loader, device)
    if abs(baseline_test - 85.03) > 0.08:
        raise RuntimeError(f"H1MP source mismatch: got {baseline_test:.2f}% vs expected 85.03%")
    adapter = build_h2_trace_adapter(model)
    binary_exact = {
        "stem_cin96": binary_dot_exact_width(3, 96),
        "cin16": binary_dot_exact_width(3, 16),
        "cin32": binary_dot_exact_width(3, 32),
        "cin64": binary_dot_exact_width(3, 64),
    }
    exact_plan = derive_exact_width_plan(
        model,
        adapter,
        calib_loader,
        device,
        binary_exact,
        gap_deferred_scaling=args.gap_deferred_v2,
    )

    gap_scale_verification = None
    if args.gap_deferred_v2:
        # Verification is TRAIN-derived and happens before the official TEST.
        # It exercises the actual H2 GAP path, not only an algebraic fixture.
        adapter.set_h2_plan(exact_plan)
        adapter.set_trace_enabled(True)
        verification_batch = next(iter(calib_loader))[0].to(device, non_blocking=True)
        adapter(verification_batch)
        gap_scale_verification = verify_gap_deferred_scaling(
            adapter.last_gap_sum,
            adapter.last_gap_input_shift,
        )
        adapter.set_trace_enabled(False)
        if not gap_scale_verification["mathematically_equal"]:
            raise RuntimeError("H2A-v2 GAP deferred-scale bit-accurate verification failed")

    h2a = evaluate_exact_width_plan(model, adapter, exact_plan, test_loader, device)

    common_source = {
        "r8b_checkpoint": str(source_checkpoint),
        "h0_results_json": str(h0_path),
        "h1mp_results_json": str(h1mp_path),
        "baseline_official_test_accuracy": float(baseline_test),
        "expected_h1mp_accuracy": 85.03,
        "source_tolerance_pp": 0.08,
    }
    result = {
        "status": (
            "H2A_V2_PASS"
            if args.gap_deferred_v2 and abs(h2a["accuracy"] - baseline_test) <= 0.10
            else "H2A_V2_FAIL"
            if args.gap_deferred_v2
            else "H2A_PASS"
            if abs(h2a["accuracy"] - baseline_test) <= 0.10
            else "H2A_FAIL"
        ),
        "source": common_source,
        "methodology": {
            "seed": int(args.seed),
            "calibration_samples": int(args.calibration_samples),
            "search_validation_samples": int(args.search_validation_samples),
            "official_test_used_for_search": False,
            "h1mp_plan_modified": False,
            "h2b_executed": False if args.gap_deferred_v2 else None,
            "gap_division_mode": exact_plan["gap"]["division_mode"],
            "retrained": False,
        },
        "binary_exact": binary_exact,
        "h2a_plan": exact_plan,
        "h2a_exact_width_anchor": h2a,
    }
    if args.gap_deferred_v2:
        result["gap_scale_verification"] = gap_scale_verification
        result["h2a_v2_anchor"] = h2a
        result["summary"] = {
            "h2a_v2_pass": result["status"] == "H2A_V2_PASS",
            "h2a_v2_accuracy": h2a["accuracy"],
            "delta_vs_h1mp_pp": h2a["accuracy"] - baseline_test,
            "gap_division_by_64": "scale_metadata_adjustment",
            "gap_arithmetic_right_shift_applied": False,
            "gap_integer_rounding_applied": False,
            "gap_sum_bits": exact_plan["gap"]["sum_bits"],
            "gap_output_bits": exact_plan["gap"]["output_bits"],
            "gap_output_shift": exact_plan["gap"]["output_shift"],
            "fc_accumulator_bits": 24,
            "final_logits_bits": 24,
            "qrprelu_max_output_bits": max(item["bits"] for item in exact_plan["qrprelu"].values()),
            "qrprelu_max_inner_bits": max(item["inner_bits"] for item in exact_plan["qrprelu"].values()),
            "h2b_executed": False,
            "retrained": False,
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        report_path.write_text(render_report(result), encoding="utf-8")
        print(f"Wrote {result_path}", flush=True)
        print(f"Wrote {report_path}", flush=True)
        print(json.dumps(result["summary"], indent=2), flush=True)
        return
    if result["status"] == "H2A_FAIL":
        report_path.write_text(render_report(result), encoding="utf-8")
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"H2A changed accuracy by {h2a['accuracy'] - baseline_test:+.2f} pp; H2B was not run."
        )

    h2b = search_remaining_widths(
        model=model,
        adapter=adapter,
        base_plan=exact_plan,
        calibration_loader=calib_loader,
        search_loader=search_loader,
        device=device,
        target_drop_pp=0.10,
        max_drop_pp=0.30,
    )
    final = evaluate_exact_width_plan(model, adapter, h2b["final_plan"], test_loader, device)
    result["status"] = "COMPLETE"
    result["h2b_search"] = h2b
    result["final_official_test"] = final
    result["summary"] = {
        "h2a_pass": True,
        "h2a_accuracy": h2a["accuracy"],
        "h2a_delta_vs_h1mp_pp": h2a["accuracy"] - baseline_test,
        "final_h2_accuracy": final["accuracy"],
        "final_delta_vs_h1mp_pp": final["accuracy"] - baseline_test,
        "binary_accumulator_bits": {
            key: item["signed_acc_bits"] for key, item in binary_exact.items()
        },
        "gap_sum_bits": h2b["final_plan"]["gap"]["sum_bits"],
        "fc_accumulator_bits": h2b["final_plan"]["fc"]["accumulator_bits"],
        "final_logits_bits": h2b["final_plan"]["fc"]["final_logits_bits"],
        "qrprelu_max_output_bits": max(item["bits"] for item in exact_plan["qrprelu"].values()),
        "qrprelu_max_inner_bits": max(item["inner_bits"] for item in exact_plan["qrprelu"].values()),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
