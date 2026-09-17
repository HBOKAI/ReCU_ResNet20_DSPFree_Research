import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.layers import iter_recu_binary_convs
from recu_hw.model import ReCUResNet20
from recu_hw.r4 import build_r4_from_r2, iter_r4_affines
from recu_hw.selective_spot import (
    EXPECTED_BACKBONE_K,
    build_selective_spot_from_r2,
    coefficient_audit_rows,
    named_spot_affines,
    verify_target_count,
)
from recu_hw.study_data import build_train_val_calibration_loaders
from recu_hw.utils import load_json, seed_everything


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_tau_all(model, tau: float) -> None:
    for m in iter_recu_binary_convs(model):
        m.set_tau(float(tau))


def stats(values: List[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "max": max(vals),
    }


@torch.no_grad()
def verify_one_term_equivalence(spot_model, r4_model) -> Dict[str, float]:
    spot_vals = []
    for _, m in named_spot_affines(spot_model):
        if bool(m.selected_2term.any()):
            raise RuntimeError("S0 equivalence check requires all channels unselected")
        spot_vals.append(m.effective_k().detach().cpu().flatten())

    r4_vals = [m.effective_k().detach().cpu().flatten() for m in iter_r4_affines(r4_model)]
    a = torch.cat(spot_vals)
    b = torch.cat(r4_vals)
    if a.numel() != b.numel():
        raise RuntimeError(f"R4/SPoT K count mismatch: {a.numel()} vs {b.numel()}")
    diff = (a - b).abs()
    return {
        "count": int(a.numel()),
        "max_abs_error": float(diff.max()) if diff.numel() else 0.0,
        "mismatch_count": int((diff != 0).sum()),
    }


@torch.no_grad()
def collect_pre_affine_mean_square(model, loader, device) -> Dict[str, torch.Tensor]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs):
            x = inputs[0].detach()
            sq = x.to(torch.float64).square().sum(dim=(0, 2, 3)).cpu()
            n = int(x.shape[0] * x.shape[2] * x.shape[3])
            if name not in sums:
                sums[name] = sq
                counts[name] = n
            else:
                sums[name] += sq
                counts[name] += n
        return hook

    for name, module in named_spot_affines(model):
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        model.eval()
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            model(x)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name, module in named_spot_affines(model):
        if name not in sums or counts[name] <= 0:
            raise RuntimeError(f"missing calibration statistics for {name}")
        mean_sq = sums[name] / float(counts[name])
        if mean_sq.numel() != module.channels:
            raise RuntimeError(f"channel statistics mismatch for {name}")
        out[name] = mean_sq
    return out


def build_markdown(summary: Dict) -> str:
    one = summary["one_term"]
    spot = summary["spot"]
    eq = summary["r4_one_term_equivalence"]
    return f"""# S0 Selective 2-term SPoT Coefficient Audit

This report is an isolated R4-level sensitivity study. It does **not** modify or replace the selected E1/H2A-v2-QAT mainline (85.51% official TEST).

## Scope

- Backbone fused-affine K coefficients: **{summary['count']}**
- Historical expected count: **{summary['expected_count']}**
- Proper 2-term is used only when it strictly improves the R4 one-term value.
- Bias B, stem, head, FC, QRPReLU and binary-conv topology are unchanged.

## R4 one-term equivalence

- Count: {eq['count']}
- Max absolute error vs current `FusedAffine2d`: **{eq['max_abs_error']:.12g}**
- Exact mismatches: **{eq['mismatch_count']}**

## Approximation error

| Metric | One-term R4 | Best <=2-term SPoT |
|---|---:|---:|
| Mean abs error | {one['abs']['mean']:.9g} | {spot['abs']['mean']:.9g} |
| Median abs error | {one['abs']['median']:.9g} | {spot['abs']['median']:.9g} |
| Max abs error | {one['abs']['max']:.9g} | {spot['abs']['max']:.9g} |
| Mean relative error | {one['relative']['mean']:.6%} | {spot['relative']['mean']:.6%} |
| Median relative error | {one['relative']['median']:.6%} | {spot['relative']['median']:.6%} |
| Max relative error | {one['relative']['max']:.6%} | {spot['relative']['max']:.6%} |

- One-term exact-match channels: **{summary['one_term_exact_match_count']}**
- Channels for which a proper second term strictly improves error: **{summary['proper_second_term_improves_count']}**
- Best <=2-term exact-match channels: **{summary['spot_exact_match_count']}**

Coverage and checkpoint selection must use TRAIN-derived validation only. No official TEST value is consumed by this audit.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selective_spot.json")
    ap.add_argument("--source-r2-checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed = int(cfg["experiment"].get("seed", 123))
    seed_everything(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    source = torch.load(args.source_r2_checkpoint, map_location=device)
    r2 = ReCUResNet20(
        num_classes=10,
        activation_mode="qrprelu",
        alpha_mode="float",
    ).to(device)
    r2.load_state_dict(source["model"], strict=True)
    r2.eval()

    model = build_selective_spot_from_r2(r2).to(device)
    r4 = build_r4_from_r2(r2).to(device)
    tau = float(cfg["formal"].get("tau", 0.99))
    set_tau_all(model, tau)
    set_tau_all(r4, tau)
    model.eval()
    r4.eval()

    expected = int(cfg.get("study", {}).get("expected_backbone_k", EXPECTED_BACKBONE_K))
    verify_target_count(model, expected=expected)
    rows = coefficient_audit_rows(model)
    equivalence = verify_one_term_equivalence(model, r4)
    if equivalence["mismatch_count"] != 0:
        raise RuntimeError(
            "new SPoT path does not exactly reproduce R4 at 0% coverage; stop study"
        )

    summary = {
        "source_r2_checkpoint": args.source_r2_checkpoint,
        "count": len(rows),
        "expected_count": expected,
        "r4_one_term_equivalence": equivalence,
        "one_term": {
            "abs": stats([r["one_abs_error"] for r in rows]),
            "relative": stats([r["one_relative_error"] for r in rows]),
        },
        "spot": {
            "abs": stats([r["spot_abs_error"] for r in rows]),
            "relative": stats([r["spot_relative_error"] for r in rows]),
        },
        "one_term_exact_match_count": sum(r["one_abs_error"] == 0.0 for r in rows),
        "proper_second_term_improves_count": sum(bool(r["second_term_needed"]) for r in rows),
        "spot_exact_match_count": sum(r["spot_abs_error"] == 0.0 for r in rows),
    }

    report_root = Path(cfg["output"].get("report_root", "reports/selective_spot"))
    report_root.mkdir(parents=True, exist_ok=True)
    write_csv(rows, report_root / "S0_COEFFICIENT_AUDIT.csv")
    write_json({"summary": summary, "rows": rows}, report_root / "S0_COEFFICIENT_AUDIT.json")
    (report_root / "S0_COEFFICIENT_AUDIT.md").write_text(
        build_markdown(summary), encoding="utf-8"
    )

    _, _, calibration_loader, split_meta = build_train_val_calibration_loaders(
        cfg, smoke=args.smoke
    )
    mean_s2 = collect_pre_affine_mean_square(model, calibration_loader, device)

    ranked = []
    for r in rows:
        name = r["module"]
        ch = int(r["channel"])
        es2 = float(mean_s2[name][ch])
        d1 = (r["k_float"] - r["k_1term"]) ** 2 * es2
        d2 = (r["k_float"] - r["k_2term_or_one"]) ** 2 * es2
        rr = dict(r)
        rr.update({
            "mean_s_squared": es2,
            "distortion_1term": d1,
            "distortion_2term": d2,
            "recoverable_distortion": d1 - d2,
        })
        ranked.append(rr)

    primary = sorted(
        ranked,
        key=lambda r: (-r["recoverable_distortion"], r["global_index"]),
    )
    secondary = sorted(
        ranked,
        key=lambda r: (-r["one_relative_error"], r["global_index"]),
    )
    for idx, row in enumerate(primary, start=1):
        row["activation_rank"] = idx
    secondary_keys = {
        (r["module"], int(r["channel"])): idx
        for idx, r in enumerate(secondary, start=1)
    }
    for row in primary:
        row["coefficient_error_rank"] = secondary_keys[(row["module"], int(row["channel"]))]

    write_csv(primary, report_root / "S0B_ACTIVATION_AWARE_RANKING.csv")
    write_json(
        {
            "source_r2_checkpoint": args.source_r2_checkpoint,
            "ranking_metric": "recoverable_distortion",
            "calibration_is_train_only": True,
            "primary_ranking": primary,
            "secondary_ranking_keys": [
                {
                    "rank": i,
                    "global_index": int(r["global_index"]),
                    "module": r["module"],
                    "channel": int(r["channel"]),
                    "one_relative_error": r["one_relative_error"],
                }
                for i, r in enumerate(secondary, start=1)
            ],
        },
        report_root / "S0B_ACTIVATION_AWARE_RANKING.json",
    )
    write_json(split_meta, report_root / "S0B_SPLIT_METADATA.json")

    top10 = primary[:10]
    ranking_md = [
        "# S0B TRAIN-only Activation-aware Ranking",
        "",
        "Primary score: `recoverable_distortion = E[S^2] * (error_1term^2 - error_2term^2)`.",
        "Official CIFAR-10 TEST is not loaded or used in this stage.",
        "",
        "| Rank | Module | Ch | Recoverable distortion | One-term rel err | <=2-term rel err |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(top10, start=1):
        ranking_md.append(
            f"| {i} | `{r['module']}` | {r['channel']} | {r['recoverable_distortion']:.9g} | "
            f"{r['one_relative_error']:.4%} | {r['spot_relative_error']:.4%} |"
        )
    (report_root / "S0B_ACTIVATION_AWARE_RANKING.md").write_text(
        "\n".join(ranking_md) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "coefficient_summary": summary,
        "top10_activation_aware": [
            {
                "module": r["module"],
                "channel": r["channel"],
                "recoverable_distortion": r["recoverable_distortion"],
            }
            for r in top10
        ],
        "report_root": str(report_root),
    }, indent=2))


if __name__ == "__main__":
    main()
