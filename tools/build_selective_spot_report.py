import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from recu_hw.selective_spot import SelectiveSPoTResNet20
from recu_hw.spot_hardware import analyze_spot_hardware
from recu_hw.utils import load_json


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_formal_run(run_dir: Path, device: torch.device) -> Dict:
    summary_path = run_dir / "summary.json"
    official_path = run_dir / "official_test.json"
    best_path = run_dir / "best.pt"
    for p in (summary_path, official_path, best_path):
        if not p.exists():
            raise FileNotFoundError(f"required formal artifact missing: {p}")

    summary = read_json(summary_path)
    official = read_json(official_path)
    if summary.get("stage") != "formal":
        raise ValueError(f"{run_dir} is not a formal run")
    if summary.get("official_test_used_for_selection"):
        raise RuntimeError(f"TEST leakage flag is true in {run_dir}")
    if not official.get("selection_was_train_derived_only"):
        raise RuntimeError(f"official TEST record does not confirm TRAIN-only selection: {run_dir}")

    state = torch.load(best_path, map_location=device)
    model = SelectiveSPoTResNet20().to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    hardware = analyze_spot_hardware(model)

    return {
        "run_dir": str(run_dir),
        "coverage_pct": float(summary["coverage_requested_pct"]),
        "initial_train_val_acc": float(summary["initial_train_derived_val_acc"]),
        "best_train_val_acc": float(summary["best_train_derived_val_acc"]),
        "reload_train_val_acc": float(summary["reload_train_derived_val_acc"]),
        "best_epoch": int(summary["best_epoch"]),
        "official_test_acc": float(official["official_test_acc"]),
        "official_test_loss": float(official["official_test_loss"]),
        "hardware": hardware,
    }


def classify_gain(delta_pp: float) -> str:
    if delta_pp <= 0.0:
        return "regression"
    if delta_pp < 0.10:
        return "negligible"
    if delta_pp >= 0.50:
        return "near-full recovery"
    if delta_pp >= 0.35:
        return "strong selective gain"
    if delta_pp >= 0.20:
        return "meaningful selective gain"
    return "small positive gain"


def pearson(xs: List[float], ys: List[float]):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if den == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / den


def stage_counts(hardware: Dict) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in hardware["rows"]:
        stage = r["module"].split(".")[0]
        rec = out.setdefault(stage, {"selected": 0, "active2": 0})
        rec["selected"] += int(r["selected"])
        rec["active2"] += int(r["active2"])
    return out


def module_active_counts(hardware: Dict) -> List[Dict]:
    counts: Dict[str, Dict[str, int]] = {}
    for r in hardware["rows"]:
        rec = counts.setdefault(r["module"], {"selected": 0, "active2": 0})
        rec["selected"] += int(r["selected"])
        rec["active2"] += int(r["active2"])
    rows = [
        {"module": k, "selected": v["selected"], "active2": v["active2"]}
        for k, v in counts.items()
    ]
    rows.sort(key=lambda r: (-r["active2"], -r["selected"], r["module"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selective_spot.json")
    ap.add_argument("--control-run", required=True)
    ap.add_argument("--selective-run", required=True)
    ap.add_argument("--full-run", required=True)
    ap.add_argument("--s0-json", default="reports/selective_spot/S0_COEFFICIENT_AUDIT.json")
    ap.add_argument("--ranking-json", default="reports/selective_spot/S0B_ACTIVATION_AWARE_RANKING.json")
    ap.add_argument("--s1-json", default="reports/selective_spot/S1_ZERO_SHOT_COVERAGE_SWEEP.json")
    ap.add_argument("--s2-json", default="reports/selective_spot/S2_DIAGNOSTIC_SELECTION.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_json(args.config)
    device = torch.device(args.device)
    control = load_formal_run(Path(args.control_run), device)
    selective = load_formal_run(Path(args.selective_run), device)
    full = load_formal_run(Path(args.full_run), device)

    if control["coverage_pct"] != 0.0:
        raise ValueError("control run must be 0% coverage")
    if not 0.0 < selective["coverage_pct"] < 100.0:
        raise ValueError("selective run coverage must be strictly between 0 and 100%")
    if full["coverage_pct"] != 100.0:
        raise ValueError("full run must be 100% coverage")

    s0 = read_json(Path(args.s0_json))
    ranking_obj = read_json(Path(args.ranking_json))
    s1 = read_json(Path(args.s1_json))
    s2 = read_json(Path(args.s2_json))

    selected_from_s2 = float(s2["best_selective"]["coverage_pct"])
    if selected_from_s2 != selective["coverage_pct"]:
        raise RuntimeError(
            f"formal selective coverage {selective['coverage_pct']}% does not match "
            f"TRAIN-val S2 selection {selected_from_s2}%"
        )

    gap = float(cfg["experiment"]["historical_r2_test_acc"]) - float(
        cfg["experiment"]["historical_r4_test_acc"]
    )
    selective_gain = selective["official_test_acc"] - control["official_test_acc"]
    full_gain = full["official_test_acc"] - control["official_test_acc"]
    selective_recovered = selective_gain / gap if gap != 0 else None
    full_recovered = full_gain / gap if gap != 0 else None
    full_capture = selective_gain / full_gain if full_gain > 0 else None

    s0_summary = s0["summary"]
    one_abs = float(s0_summary["one_term"]["abs"]["mean"])
    spot_abs = float(s0_summary["spot"]["abs"]["mean"])
    one_rel = float(s0_summary["one_term"]["relative"]["mean"])
    spot_rel = float(s0_summary["spot"]["relative"]["mean"])
    abs_error_reduction_fraction = (one_abs - spot_abs) / one_abs if one_abs > 0 else None
    rel_error_reduction_fraction = (one_rel - spot_rel) / one_rel if one_rel > 0 else None

    # Coverage-level analytic-vs-zero-shot correlation.  The analytic x-axis
    # is cumulative TRAIN-calibration recoverable distortion from the fixed
    # activation-aware ranking.  This is descriptive, not causal evidence.
    ranking = ranking_obj["primary_ranking"]
    total_recoverable = sum(max(0.0, float(r["recoverable_distortion"])) for r in ranking)
    cumulative = []
    running = 0.0
    for r in ranking:
        running += max(0.0, float(r["recoverable_distortion"]))
        cumulative.append(running)

    corr_x = []
    corr_y = []
    s1_control = next(r for r in s1["results"] if float(r["coverage_requested_pct"]) == 0.0)
    s1_control_acc = float(s1_control["train_derived_val_acc"])
    for row in s1["results"]:
        cov = float(row["coverage_requested_pct"])
        selected_n = int(row["selected_channels"])
        recovered_distortion_fraction = (
            cumulative[selected_n - 1] / total_recoverable
            if selected_n > 0 and total_recoverable > 0
            else 0.0
        )
        val_gain = float(row["train_derived_val_acc"]) - s1_control_acc
        corr_x.append(recovered_distortion_fraction)
        corr_y.append(val_gain)
    analytic_zero_shot_corr = pearson(corr_x, corr_y)

    interesting = (
        selective["coverage_pct"] <= 25.0
        and full_gain > 0
        and selective_gain >= 0.8 * full_gain
    )
    consider_e1_port = selective_gain >= 0.20 and selective["hardware"]["active_second_term_channels"] > 0

    selected_modules = module_active_counts(selective["hardware"])
    results = {
        "scope": "isolated R4-level selective 2-term SPoT sensitivity study",
        "mainline_unchanged": True,
        "mainline_reference_test_acc": float(cfg["experiment"]["mainline_reference_test_acc"]),
        "historical_r2_test_acc": float(cfg["experiment"]["historical_r2_test_acc"]),
        "historical_r4_test_acc": float(cfg["experiment"]["historical_r4_test_acc"]),
        "historical_r2_to_r4_gap_pp": gap,
        "formal": {
            "control": control,
            "selective": selective,
            "full_2term": full,
        },
        "accuracy": {
            "selective_gain_vs_matched_control_pp": selective_gain,
            "full_gain_vs_matched_control_pp": full_gain,
            "selective_recovered_fraction_of_historical_gap": selective_recovered,
            "full_recovered_fraction_of_historical_gap": full_recovered,
            "selective_fraction_of_full_2term_gain": full_capture,
            "selective_label": classify_gain(selective_gain),
            "full_label": classify_gain(full_gain),
            "interesting_le25pct_captures_ge80pct_full_gain": interesting,
        },
        "coefficient_error": {
            "mean_abs_one_term": one_abs,
            "mean_abs_best_le2term": spot_abs,
            "mean_abs_error_reduction_fraction": abs_error_reduction_fraction,
            "mean_relative_one_term": one_rel,
            "mean_relative_best_le2term": spot_rel,
            "mean_relative_error_reduction_fraction": rel_error_reduction_fraction,
            "proper_second_term_improves_count": int(s0_summary["proper_second_term_improves_count"]),
            "spot_exact_match_count": int(s0_summary["spot_exact_match_count"]),
        },
        "activation_metric_vs_zero_shot_val": {
            "pearson_cumulative_recoverable_distortion_fraction_vs_val_gain": analytic_zero_shot_corr,
            "note": "Coverage-level TRAIN-only descriptive correlation; not causal proof.",
        },
        "selective_stage_counts": stage_counts(selective["hardware"]),
        "selective_top_modules_by_active_second_term": selected_modules[:10],
        "port_to_e1_screen": {
            "evidence_threshold_met": consider_e1_port,
            "rule": "selective matched-control TEST gain >= +0.20 pp and at least one active second term",
            "automatic_promotion": False,
            "required_before_port": [
                "repeat H0/H1MP/H2A-v2 integer/scale audit with two-term K",
                "recompute affine intermediate widths and overflow headroom",
                "decide serialized versus parallel second-shift microarchitecture",
                "re-run exact-deploy QAT after the SPoT arithmetic contract is frozen",
                "RTL/Vivado synthesis to prove DSP48E1=0 and measure LUT/Fmax/cycles",
            ],
        },
    }

    report_root = Path(cfg["output"].get("report_root", "reports/selective_spot"))
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(results, report_root / "SELECTIVE_SPOT_RESULTS.json")

    def pct_or_na(x):
        return "N/A" if x is None else f"{100.0 * x:.1f}%"

    def corr_or_na(x):
        return "N/A" if x is None else f"{x:.4f}"

    rows = [control, selective, full]
    md = [
        "# Selective 2-term SPoT Final Report",
        "",
        "This remains an isolated R4-level sensitivity study. The selected E1/H2A-v2-QAT mainline is unchanged.",
        "",
        "## Formal comparison",
        "",
        "| Coverage | Active 2-term K | Best TRAIN-val | Official TEST | Delta vs 0% | Recovered fraction of historical 0.61 pp | Extra shifts/image | Extra adds/image |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        delta = r["official_test_acc"] - control["official_test_acc"]
        recovered = delta / gap if gap != 0 else None
        hw = r["hardware"]
        md.append(
            f"| {r['coverage_pct']:.1f}% | {hw['active_second_term_channels']} | "
            f"{r['best_train_val_acc']:.2f}% | {r['official_test_acc']:.2f}% | "
            f"{delta:+.2f} pp | {pct_or_na(recovered)} | "
            f"{hw['extra_shift_operations_per_image_direct_interpretation']} | "
            f"{hw['extra_add_sub_operations_per_image_direct_interpretation']} |"
        )

    md += [
        "",
        "## Coefficient approximation",
        "",
        f"- Mean absolute K error: {one_abs:.9g} -> {spot_abs:.9g} "
        f"({pct_or_na(abs_error_reduction_fraction)} reduction).",
        f"- Mean relative K error: {one_rel:.4%} -> {spot_rel:.4%} "
        f"({pct_or_na(rel_error_reduction_fraction)} reduction).",
        f"- Proper second term improves {s0_summary['proper_second_term_improves_count']} / {s0_summary['count']} channels.",
        f"- TRAIN-only coverage-level correlation between cumulative recoverable-distortion fraction and zero-shot validation gain: **{corr_or_na(analytic_zero_shot_corr)}**. This is descriptive, not causal.",
        "",
        "## Hardware interpretation",
        "",
        f"Selective candidate active second-term channels: **{selective['hardware']['active_second_term_channels']}**.",
        f"Dense programmable extra coefficient metadata: **{selective['hardware']['dense_programmable_extra_metadata_bits']} bits** "
        f"({selective['hardware']['dense_programmable_extra_metadata_bytes']:.1f} B).",
        f"Sparse/static proxy extra metadata: **{selective['hardware']['sparse_static_extra_metadata_bits']} bits** "
        f"({selective['hardware']['sparse_static_extra_metadata_bytes']:.1f} B).",
        f"Maximum extra carry width at the 2-term K*S add: **{selective['hardware']['max_extra_carry_bits_at_two_term_add']} bit**.",
        "These widths cover the multiplier-free K*S path only. R4 floating B is outside the fixed-point output-width claim.",
        "Physical DSP48E1=0 is not proven here; that requires RTL/Vivado synthesis.",
        "",
        "## Study answers",
        "",
        f"1. Exact <=2-term SPoT reduced mean absolute coefficient error by **{pct_or_na(abs_error_reduction_fraction)}** versus one-term.",
        f"2. The TRAIN-only coverage-level analytic/zero-shot validation correlation is **{corr_or_na(analytic_zero_shot_corr)}**; interpret it as descriptive evidence only.",
        f"3. Highest active-second-term concentration in the selective candidate starts with: "
        + ", ".join(f"{r['module']} ({r['active2']})" for r in selected_modules[:5]) + ".",
        f"4. <=25% selective coverage captures >=80% of the full-2term TEST gain: **{interesting}**.",
        f"5. Full 2-term gain vs matched control is **{full_gain:+.2f} pp**; selective gain is **{selective_gain:+.2f} pp**.",
        f"6. Screen for a later isolated E1 port (>=+0.20 pp matched-control gain): **{consider_e1_port}**. This does not promote or merge anything automatically.",
        "7. Before any E1 port, re-audit H0/H1MP/H2A-v2 scales, affine intermediate widths/overflow, serialized-vs-parallel shift scheduling, exact-deploy QAT, and then RTL/Vivado LUT/Fmax/DSP/cycle results.",
        "",
        "## Stop condition",
        "",
        "Stop here. Do not merge into mainline, change H2A-v2, or begin RTL solely from this report.",
    ]
    (report_root / "SELECTIVE_SPOT_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "selective_gain_pp": selective_gain,
        "full_gain_pp": full_gain,
        "selective_label": classify_gain(selective_gain),
        "selective_fraction_of_full_gain": full_capture,
        "consider_e1_port_screen": consider_e1_port,
        "report": str(report_root / "SELECTIVE_SPOT_REPORT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
