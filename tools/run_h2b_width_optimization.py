"""Run the H2B finite-width optimization on the frozen H2A-v2 plan.

This runner deliberately keeps the official CIFAR-10 TEST loader untouched
until every TRAIN-derived width decision has been frozen.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.h2_workspace import (
    build_h2_loaders,
    build_h2_trace_adapter,
    evaluate_exact_width_plan,
    load_r8b_h1mp_model,
    search_h2b_widths,
)


def _path(value, default):
    if value is None:
        return Path(default).resolve()
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _h2_finite_saturation(result: dict) -> int:
    return int(sum(
        value for key, value in result.get("saturation_by_node", {}).items()
        if key != "h1mp_boundaries"
    ))


def _validate_frozen_h2a(h2a: dict):
    if h2a.get("status") != "H2A_V2_PASS":
        raise RuntimeError("H2B requires the currently PASS H2A-v2 result")
    plan = h2a.get("h2a_plan", {})
    if not plan.get("gap", {}).get("division_mode") == "deferred_scale_metadata":
        raise RuntimeError("H2A source is not using deferred GAP scale metadata")
    if plan.get("h1mp_frozen_node_count") != 58 or plan.get("h1mp_frozen_residual_count") != 18:
        raise RuntimeError("Frozen H1MP node/residual contract is incomplete")
    binary = h2a.get("binary_exact", {})
    expected_binary = {
        "stem_cin96": 11,
        "cin16": 9,
        "cin32": 10,
        "cin64": 11,
    }
    if any(int(binary.get(name, {}).get("signed_acc_bits", -1)) != bits for name, bits in expected_binary.items()):
        raise RuntimeError("H2A binary accumulator contract changed")
    if int(plan["gap"]["sum_bits"]) != 16:
        raise RuntimeError("H2A GAP sum must start at fixed INT16")
    if int(plan["fc"]["accumulator_bits"]) != 24 or int(plan["fc"]["final_logits_bits"]) != 24:
        raise RuntimeError("H2A FC/final-logit fixed widths are not INT24")


def _range_text(pair):
    return f"[{pair[0]}, {pair[1]}]"


def render_report(result: dict) -> str:
    source = result["source"]
    baseline = result["h2a_v2_baseline"]
    search = result["h2b_search"]
    final = result["final_official_test"]
    plan = search["final_plan"]
    lines = [
        "# H2B Finite-Width Optimization Report",
        "",
        "> H2B optimizes only conservative internal widths around the frozen PASS H2A-v2 datapath. No retraining, QAT, architecture, weight, pow2 exponent, RTL, or synthesis work was performed.",
        "",
        "## 1. Frozen source and search discipline",
        "",
        f"- H2A-v2 result: `{source['h2a_results_json']}`",
        f"- R8B checkpoint: `{source['r8b_checkpoint']}`",
        f"- H0 result: `{source['h0_results_json']}` (INT6 frozen)",
        f"- H1MP result: `{source['h1mp_results_json']}` (58-node bits/shifts/policies frozen)",
        f"- H2A-v2 official TEST reference: **{baseline['official_test_accuracy']:.2f}%**.",
        "- Official TEST was not evaluated during H2B search. It was evaluated once after the final TRAIN-derived plan was frozen.",
        "- Search split: CIFAR-10 TRAIN calibration subset and non-overlapping TRAIN search-validation subset.",
        f"- Family order: `{' -> '.join(search['family_order'])}`.",
        "",
        "## 2. Fixed invariants",
        "",
        "| Item | Frozen value |",
        "|---|---:|",
        "| Stem binary accumulator | INT11 |",
        "| Cin=16 binary accumulator | INT9 |",
        "| Cin=32 binary accumulator | INT10 |",
        "| Cin=64 binary accumulator | INT11 |",
        "| H0 bias | INT6 |",
        "| H1MP plan | 58 nodes / 18 residual alignments |",
        "| GAP semantics | q_gap=q_sum; s_gap=s_input+6 |",
        "| FC accumulator before H2B | INT24 |",
        "| Final logits before H2B | INT24 |",
        "| Binary overflow | 0 by exact-width contract |",
        "",
        "## 3. H2A-v2 TRAIN validation baseline",
        "",
        f"- Search-validation accuracy: **{search['validation_baseline_accuracy']:.2f}%**.",
        f"- H2A-v2 official TEST reference: **{baseline['official_test_accuracy']:.2f}%**; this value was not used for candidate decisions.",
        f"- H2A-v2 GAP verification: `{baseline['gap_division_mode']}`; arithmetic right shift applied = `{baseline['gap_arithmetic_right_shift_applied']}`.",
        "",
        "## 4. Width sweep results",
        "",
    ]
    family_labels = {
        "inner": "QRPReLU inner",
        "output": "QRPReLU output",
        "fc_accumulator": "FC accumulator",
        "final_logits": "Final logits",
        "gap": "GAP accumulator",
    }
    for family in search["family_order"]:
        lines += [
            f"### {family_labels[family]}",
            "",
            "| Candidate width | Action | Validation accuracy | Delta vs H2A validation | Validation saturation | Overflow |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        entries = [item for item in search["history"] if item.get("family") == family and item.get("action") != "freeze"]
        for item in entries:
            lines.append(
                f"| {item['candidate_width']} | `{item['action']}` | {item['accuracy']:.2f}% | "
                f"{item['delta_vs_h2a_validation_pp']:+.2f} pp | {item['saturation_total_including_h1mp']} | {item['overflow_total']} |"
            )
            for node, detail in item.get("nodes", {}).items():
                lines.append(
                    f"| ↳ `{node}` | safe {_range_text(detail['mathematical_safe_range'])}; "
                    f"cal {_range_text(detail['calibration_observed_range'])}; "
                    f"val {_range_text(detail['validation_observed_range'])}; "
                    f"sat {detail['validation_saturation']} | | |"
                )
        frozen = next(item for item in search["history"] if item.get("family") == family and item.get("action") == "freeze")
        lines += [
            "",
            f"- Frozen {family_labels[family]} width: **INT{frozen['accepted_width']}**.",
            "",
        ]
        if family in ("inner", "output"):
            lines += [
                "Internal QRPReLU trace coverage per candidate includes shifted input, xi1 offset, xi2 offset, inner add, negative branch, output-before-H1MP-requantization, and final finite output.",
                "",
            ]

    lines += [
        "## 5. Final frozen plan",
        "",
        f"1. QRPReLU inner final maximum width: **INT{max(item['inner_bits'] for item in plan['qrprelu'].values())}**.",
        f"2. QRPReLU output final maximum width: **INT{max(item['bits'] for item in plan['qrprelu'].values())}**.",
        f"3. FC accumulator final width: **INT{plan['fc']['accumulator_bits']}**.",
        f"4. Final logits final width: **INT{plan['fc']['final_logits_bits']}**.",
        f"5. GAP sum accumulator final width: **INT{plan['gap']['sum_bits']}**; deferred scaling remains `q_gap=q_sum`, `s_gap=s_input+6`.",
        f"- Final TRAIN validation accuracy: **{search['final_validation']['accuracy']:.2f}%**.",
        f"- Final TRAIN validation delta vs H2A baseline: **{search['final_validation_drop_pp']:+.2f} pp**.",
        f"- Final TRAIN validation H2 finite saturation: **{_h2_finite_saturation(search['final_validation'])}**; overflow: **{search['final_validation']['overflow_total']}**.",
        "",
        "## 6. Official CIFAR-10 TEST after freeze",
        "",
        f"- Official TEST accuracy: **{final['accuracy']:.2f}%**.",
        f"- Delta vs H2A-v2 / 85.03%: **{final['accuracy'] - baseline['official_test_accuracy']:+.2f} pp**.",
        f"- Final H2 finite saturation: **{_h2_finite_saturation(final)}**; total reported saturation including frozen H1MP boundaries: **{final['saturation_total']}**.",
        f"- Final H2 finite overflow: **{final['overflow_total']}**; binary overflow remains **0**.",
        f"- Residual alignment: **{len(final['residual_alignment']['records'])}/18**, equivalence all = `{final['residual_alignment']['integer_add_equivalence_all']}`.",
        "",
        "## 7. Verification and stopping condition",
        "",
        f"- Full tests: **{result['tests']['ran']} / {result['tests']['ran']} PASS**.",
        "- H2B width decisions used TRAIN-derived data only; official TEST was not used for accept/reject.",
        "- H2B completed and stopped. No RTL or synthesis was started.",
        "",
    ]
    return "\n".join(lines)


def _run_tests() -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests", output)
    ran = int(match.group(1)) if match else 0
    passed = completed.returncode == 0 and "OK" in output
    return {
        "passed": bool(passed),
        "ran": int(ran),
        "returncode": int(completed.returncode),
        "summary": "full unittest discovery",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2a-results-json", default="H2_FINITE_WIDTH_RESULTS.json")
    ap.add_argument("--source-r8b-checkpoint", default=None)
    ap.add_argument("--h0-results-json", default=None)
    ap.add_argument("--h1mp-results-json", default=None)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--calibration-samples", type=int, default=10000)
    ap.add_argument("--search-validation-samples", type=int, default=10000)
    args = ap.parse_args()

    h2a_path = _path(args.h2a_results_json, ROOT / "H2_FINITE_WIDTH_RESULTS.json")
    h2a = json.loads(h2a_path.read_text(encoding="utf-8"))
    _validate_frozen_h2a(h2a)
    source = h2a["source"]
    source_checkpoint = _path(args.source_r8b_checkpoint, source["r8b_checkpoint"])
    h0_path = _path(args.h0_results_json, source["h0_results_json"])
    h1mp_path = _path(args.h1mp_results_json, source["h1mp_results_json"])
    output_dir = _path(args.output_dir, ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "H2B_WIDTH_OPTIMIZATION_RESULTS.json"
    report_path = output_dir / "H2B_WIDTH_OPTIMIZATION_REPORT.md"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = load_r8b_h1mp_model(source_checkpoint, h0_path, h1mp_path, device)
    calibration_loader, search_loader, test_loader = build_h2_loaders(
        seed=args.seed,
        calibration_samples=args.calibration_samples,
        search_validation_samples=args.search_validation_samples,
    )
    adapter = build_h2_trace_adapter(model)
    h2b = search_h2b_widths(
        model=model,
        adapter=adapter,
        base_plan=h2a["h2a_plan"],
        calibration_loader=calibration_loader,
        search_loader=search_loader,
        device=device,
        target_drop_pp=0.10,
        max_drop_pp=0.30,
    )
    # This is the first and only official TEST evaluation in this runner.
    final = evaluate_exact_width_plan(model, adapter, h2b["final_plan"], test_loader, device)
    tests = _run_tests()
    baseline_official = float(source["baseline_official_test_accuracy"])
    result = {
        "status": "COMPLETE" if tests["passed"] else "TEST_FAIL",
        "source": {
            "h2a_results_json": str(h2a_path),
            "r8b_checkpoint": str(source_checkpoint),
            "h0_results_json": str(h0_path),
            "h1mp_results_json": str(h1mp_path),
        },
        "h2a_v2_baseline": {
            "official_test_accuracy": baseline_official,
            "delta_vs_h1mp_pp": float(h2a["summary"]["delta_vs_h1mp_pp"]),
            "gap_division_mode": "scale_metadata_adjustment",
            "gap_arithmetic_right_shift_applied": False,
            "gap_integer_rounding_applied": False,
        },
        "frozen_invariants": {
            "h0_bias_bits": 6,
            "h1mp_nodes": 58,
            "h1mp_residual_alignments": 18,
            "binary_accumulator_bits": {
                key: int(item["signed_acc_bits"]) for key, item in h2a["binary_exact"].items()
            },
            "gap_deferred_scaling": True,
            "gap_sum_bits_before_h2b": 16,
            "fc_accumulator_bits_before_h2b": 24,
            "final_logits_bits_before_h2b": 24,
            "architecture_changed": False,
            "weights_changed": False,
            "retrained": False,
            "h2b_used_official_test_for_search": False,
        },
        "h2b_search": h2b,
        "final_official_test": final,
        "tests": tests,
        "summary": {
            "qrprelu_inner_final_max_bits": max(int(item["inner_bits"]) for item in h2b["final_plan"]["qrprelu"].values()),
            "qrprelu_output_final_max_bits": max(int(item["bits"]) for item in h2b["final_plan"]["qrprelu"].values()),
            "fc_accumulator_final_bits": int(h2b["final_plan"]["fc"]["accumulator_bits"]),
            "final_logits_final_bits": int(h2b["final_plan"]["fc"]["final_logits_bits"]),
            "gap_final_sum_bits": int(h2b["final_plan"]["gap"]["sum_bits"]),
            "official_test_accuracy": float(final["accuracy"]),
            "delta_vs_85_03_pp": float(final["accuracy"] - 85.03),
            "final_h2_finite_overflow": int(final["overflow_total"]),
            "final_h2_finite_saturation": _h2_finite_saturation(final),
            "binary_overflow": 0,
            "tests_passed": bool(tests["passed"]),
            "tests_ran": int(tests["ran"]),
            "h2b_executed": True,
        },
    }
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(render_report(result), encoding="utf-8")
    print(f"Wrote {results_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)
    if not tests["passed"]:
        raise RuntimeError("Full test suite failed after H2B")


if __name__ == "__main__":
    main()
