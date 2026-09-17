import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from recu_hw.layers import iter_recu_binary_convs
from recu_hw.model import ReCUResNet20
from recu_hw.selective_spot import (
    apply_ranked_selection,
    build_selective_spot_from_r2,
    named_spot_affines,
    verify_target_count,
)
from recu_hw.study_data import build_train_val_calibration_loaders
from recu_hw.utils import load_json, seed_everything


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def set_tau_all(model, tau):
    for m in iter_recu_binary_convs(model):
        m.set_tau(float(tau))


@torch.no_grad()
def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def load_ranking(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = obj.get("primary_ranking", [])
    if not rows:
        raise ValueError("ranking JSON contains no primary_ranking")
    return rows


@torch.no_grad()
def representation_summary(model):
    selected = active = 0
    by_layer = {}
    for name, module in named_spot_affines(model):
        terms = module.discrete_terms()
        s = int(terms["selected"].sum().item())
        a = int(terms["active2"].sum().item())
        selected += s
        active += a
        stage = name.split(".")[0]
        rec = by_layer.setdefault(stage, {"selected": 0, "active2": 0})
        rec["selected"] += s
        rec["active2"] += a
    return {
        "selected_channels": selected,
        "active_second_term_channels": active,
        "by_stage": by_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selective_spot.json")
    ap.add_argument("--source-r2-checkpoint", required=True)
    ap.add_argument("--ranking-json", default="reports/selective_spot/S0B_ACTIVATION_AWARE_RANKING.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed", 123)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    state = torch.load(args.source_r2_checkpoint, map_location=device)
    r2 = ReCUResNet20(
        num_classes=10,
        activation_mode="qrprelu",
        alpha_mode="float",
    ).to(device)
    r2.load_state_dict(state["model"], strict=True)
    r2.eval()

    model = build_selective_spot_from_r2(r2).to(device)
    verify_target_count(model, int(cfg["study"].get("expected_backbone_k", 672)))
    set_tau_all(model, float(cfg["formal"].get("tau", 0.99)))

    _, val_loader, _, split_meta = build_train_val_calibration_loaders(cfg, smoke=args.smoke)
    ranking = load_ranking(args.ranking_json)
    coverages = [float(x) for x in cfg["study"].get("coverage_pct", [0, 5, 10, 25, 50, 100])]

    report_root = Path(cfg["output"].get("report_root", "reports/selective_spot"))
    manifest_root = report_root / "coverage_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    results = []
    for coverage in coverages:
        manifest = apply_ranked_selection(model, ranking, coverage)
        loss, acc = evaluate(model, val_loader, device)
        rep = representation_summary(model)
        row = {
            "coverage_requested_pct": coverage,
            "selected_channels": len(manifest),
            "active_second_term_channels": rep["active_second_term_channels"],
            "train_derived_val_loss": loss,
            "train_derived_val_acc": acc,
            "official_test_used": False,
            "by_stage": rep["by_stage"],
        }
        results.append(row)
        tag = str(coverage).replace(".", "p")
        write_json({
            "coverage_requested_pct": coverage,
            "manifest": manifest,
            "representation": rep,
        }, manifest_root / f"coverage_{tag}.json")
        print(
            f"coverage={coverage:g}% selected={len(manifest)} "
            f"active2={rep['active_second_term_channels']} TRAIN-val={acc:.2f}%"
        )

    out = {
        "source_r2_checkpoint": args.source_r2_checkpoint,
        "ranking_json": args.ranking_json,
        "selection_is_train_only": True,
        "official_test_used": False,
        "split_seed": split_meta["seed"],
        "results": results,
    }
    write_json(out, report_root / "S1_ZERO_SHOT_COVERAGE_SWEEP.json")

    md = [
        "# S1 Zero-shot Selective SPoT Coverage Sweep",
        "",
        "All values below use the fixed TRAIN-derived validation split; official CIFAR-10 TEST is not loaded.",
        "",
        "| Requested coverage | Selected K | Active 2-term K | TRAIN-derived val |",
        "|---:|---:|---:|---:|",
    ]
    for r in results:
        md.append(
            f"| {r['coverage_requested_pct']:.1f}% | {r['selected_channels']} | "
            f"{r['active_second_term_channels']} | {r['train_derived_val_acc']:.2f}% |"
        )
    (report_root / "S1_ZERO_SHOT_COVERAGE_SWEEP.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
