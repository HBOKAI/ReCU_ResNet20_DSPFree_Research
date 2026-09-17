import argparse
import json
from pathlib import Path

from recu_hw.utils import load_json


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def latest_complete_run_for_coverage(root: Path, coverage: float):
    tag = str(float(coverage)).replace(".", "p")
    prefix = f"diagnostic_cov{tag}_"
    candidates = []
    for d in root.glob(prefix + "*"):
        if not d.is_dir() or d.name.endswith("_smoke"):
            continue
        summary_path = d / "summary.json"
        best_path = d / "best.pt"
        if summary_path.exists() and best_path.exists():
            summary = read_json(summary_path)
            if summary.get("stage") != "diagnostic":
                continue
            if float(summary.get("coverage_requested_pct")) != float(coverage):
                continue
            if summary.get("official_test_acc") is not None:
                raise RuntimeError(
                    f"diagnostic run {d} unexpectedly contains official TEST accuracy"
                )
            candidates.append((d.name, d, summary))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, d, summary = candidates[-1]
    return d, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selective_spot.json")
    ap.add_argument("--experiment-root", default=None)
    ap.add_argument("--report-root", default=None)
    ap.add_argument("--source-r2-checkpoint-placeholder", default="<R2_BEST_PT>")
    args = ap.parse_args()

    cfg = load_json(args.config)
    root = Path(args.experiment_root or cfg["output"]["root"])
    report_root = Path(args.report_root or cfg["output"].get("report_root", "reports/selective_spot"))
    planned = [float(x) for x in cfg["study"].get("diagnostic_coverage_pct", [10, 25, 50, 100])]

    rows = []
    missing = []
    for cov in planned:
        found = latest_complete_run_for_coverage(root, cov)
        if found is None:
            missing.append(cov)
            continue
        d, summary = found
        rows.append({
            "coverage_pct": cov,
            "run_dir": str(d),
            "initial_train_derived_val_acc": float(summary["initial_train_derived_val_acc"]),
            "best_train_derived_val_acc": float(summary["best_train_derived_val_acc"]),
            "best_epoch": int(summary["best_epoch"]),
            "reload_train_derived_val_acc": float(summary["reload_train_derived_val_acc"]),
            "official_test_used_for_selection": bool(summary["official_test_used_for_selection"]),
        })

    if missing:
        raise RuntimeError(
            "missing completed non-smoke diagnostic runs for coverages: "
            + ", ".join(f"{x:g}%" for x in missing)
        )
    if any(r["official_test_used_for_selection"] for r in rows):
        raise RuntimeError("official TEST leakage detected in diagnostic selection")

    selective_rows = [r for r in rows if r["coverage_pct"] < 100.0]
    if not selective_rows:
        raise RuntimeError("no selective (<100%) diagnostic candidates available")

    # Primary criterion is TRAIN-derived validation accuracy.  Deterministic
    # tie break prefers lower hardware coverage, then earlier best epoch.
    best = sorted(
        selective_rows,
        key=lambda r: (
            -r["best_train_derived_val_acc"],
            r["coverage_pct"],
            r["best_epoch"],
        ),
    )[0]
    full = next((r for r in rows if r["coverage_pct"] == 100.0), None)
    if full is None:
        raise RuntimeError("100% full-2term diagnostic control is required")

    out = {
        "selection_metric": "best_train_derived_val_acc",
        "official_test_used": False,
        "planned_diagnostic_coverages": planned,
        "diagnostic_runs": rows,
        "best_selective": best,
        "full_2term_control": full,
        "formal_coverages_required": [0.0, best["coverage_pct"], 100.0],
    }
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(out, report_root / "S2_DIAGNOSTIC_SELECTION.json")

    md = [
        "# S2 Diagnostic Selection",
        "",
        "Selection uses TRAIN-derived validation only. Official CIFAR-10 TEST is not used.",
        "",
        "| Coverage | Initial TRAIN-val | Best TRAIN-val | Reload TRAIN-val | Best epoch |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: x["coverage_pct"]):
        md.append(
            f"| {r['coverage_pct']:.1f}% | {r['initial_train_derived_val_acc']:.2f}% | "
            f"{r['best_train_derived_val_acc']:.2f}% | {r['reload_train_derived_val_acc']:.2f}% | "
            f"{r['best_epoch']} |"
        )
    md += [
        "",
        f"Best selective coverage: **{best['coverage_pct']:.1f}%** "
        f"(best TRAIN-val {best['best_train_derived_val_acc']:.2f}%).",
        "",
        "Formal S3 must run exactly: 0% matched control, this selected coverage, and 100% full-2term control.",
    ]
    (report_root / "S2_DIAGNOSTIC_SELECTION.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    r2 = args.source_r2_checkpoint_placeholder
    print(json.dumps(out, indent=2))
    print("\nFormal commands:")
    for cov in (0.0, best["coverage_pct"], 100.0):
        print(
            "python tools/train_selective_spot.py --stage formal "
            f"--coverage {cov:g} --config {args.config} --source-r2-checkpoint {r2}"
        )


if __name__ == "__main__":
    main()
