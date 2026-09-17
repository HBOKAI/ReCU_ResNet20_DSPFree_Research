import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from recu_hw.layers import iter_recu_binary_convs
from recu_hw.model import ReCUResNet20
from recu_hw.selective_spot import (
    SelectiveSPoTResNet20,
    apply_ranked_selection,
    build_selective_spot_from_r2,
    named_spot_affines,
    verify_target_count,
)
from recu_hw.study_data import (
    build_official_test_loader,
    build_train_val_calibration_loaders,
)
from recu_hw.utils import git_commit, load_json, seed_everything


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def set_tau_all(model, tau: float) -> None:
    for m in iter_recu_binary_convs(model):
        m.set_tau(float(tau))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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


def load_ranking(path: str) -> List[Dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = obj.get("primary_ranking") if isinstance(obj, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("ranking JSON must contain non-empty primary_ranking")
    return rows


@torch.no_grad()
def term_manifest(model) -> List[Dict]:
    out = []
    for module_name, module in named_spot_affines(model):
        terms = module.discrete_terms()
        for ch in range(module.channels):
            out.append({
                "module": module_name,
                "channel": ch,
                "selected": bool(terms["selected"][ch].cpu()),
                "active2": bool(terms["active2"][ch].cpu()),
                "effective_k": float(terms["q"][ch].cpu()),
                "k1": int(terms["k1"][ch].cpu()),
                "k2": int(terms["k2"][ch].cpu()),
                "s1": int(float(terms["s1"][ch].cpu())),
                "s2": int(float(terms["s2"][ch].cpu())),
            })
    return out


def spatial_area_for_module(name: str) -> int:
    if name.startswith("layer1."):
        return 32 * 32
    if name.startswith("layer2."):
        return 16 * 16
    if name.startswith("layer3."):
        return 8 * 8
    raise ValueError(f"unexpected backbone affine module name: {name}")


def hardware_proxy(manifest: List[Dict]) -> Dict:
    active = [r for r in manifest if r["active2"]]
    selected = [r for r in manifest if r["selected"]]
    extra_ops = sum(spatial_area_for_module(r["module"]) for r in active)
    return {
        "target_channels": len(manifest),
        "selected_channels": len(selected),
        "active_second_term_channels": len(active),
        "selected_pct": 100.0 * len(selected) / max(len(manifest), 1),
        "active_second_term_pct": 100.0 * len(active) / max(len(manifest), 1),
        "extra_shift_operations_per_image_direct_interpretation": int(extra_ops),
        "extra_add_sub_operations_per_image_direct_interpretation": int(extra_ops),
        "parallel_engine_note": "A parallel implementation needs a second shift/sign path and an adder on active channels; LUT/timing must be synthesized later.",
        "serialized_engine_note": "A reused single shift path can serialize the second term; added affine cycles depend on PE schedule and are not claimed here.",
        "physical_dsp_zero_proven": False,
    }


def checkpoint_payload(model, *, epoch, best_val_acc, best_epoch, cfg, source_path, coverage, manifest):
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "best_train_derived_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "config": cfg,
        "source_r2_checkpoint": source_path,
        "coverage_pct": float(coverage),
        "selection_manifest": manifest,
    }


def train_stage(args, cfg, device):
    seed = int(cfg["experiment"].get("seed", 123))
    seed_everything(seed)

    source = torch.load(args.source_r2_checkpoint, map_location=device)
    r2 = ReCUResNet20(
        num_classes=10,
        activation_mode="qrprelu",
        alpha_mode="float",
    ).to(device)
    r2.load_state_dict(source["model"], strict=True)
    r2.eval()

    model = build_selective_spot_from_r2(r2).to(device)
    verify_target_count(model, int(cfg["study"].get("expected_backbone_k", 672)))
    ranking = load_ranking(args.ranking_json)
    selection_manifest = apply_ranked_selection(model, ranking, args.coverage)

    stage_cfg = cfg[args.stage]
    tau = float(stage_cfg.get("tau", 0.99))
    set_tau_all(model, tau)

    train_loader, val_loader, _, split_meta = build_train_val_calibration_loaders(
        cfg, smoke=args.smoke
    )
    criterion = nn.CrossEntropyLoss()
    initial_val_loss, initial_val_acc = evaluate(model, val_loader, criterion, device)

    epochs = 1 if args.smoke else int(stage_cfg["epochs"])
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(stage_cfg.get("lr", 0.005)),
        momentum=float(stage_cfg.get("momentum", 0.9)),
        weight_decay=float(stage_cfg.get("weight_decay", 5e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=0.0
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pct_tag = str(float(args.coverage)).replace(".", "p")
    run_dir = Path(cfg["output"]["root"]) / f"{args.stage}_cov{pct_tag}_{stamp}"
    if args.smoke:
        run_dir = Path(str(run_dir) + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(cfg, run_dir / "config.json")
    # Keep exact split metadata in the experiment for reproducibility.
    write_json(split_meta, run_dir / "train_derived_split.json")
    write_json(selection_manifest, run_dir / "selection_manifest.json")

    initial_terms = term_manifest(model)
    write_json({
        "coverage_requested_pct": float(args.coverage),
        "initial_train_derived_val_loss": initial_val_loss,
        "initial_train_derived_val_acc": initial_val_acc,
        "hardware_proxy": hardware_proxy(initial_terms),
        "terms": initial_terms,
        "official_test_used_for_selection": False,
    }, run_dir / "initial_state.json")

    best_val_acc = initial_val_acc
    best_epoch = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history = []

    torch.save(
        checkpoint_payload(
            model,
            epoch=0,
            best_val_acc=best_val_acc,
            best_epoch=0,
            cfg=cfg,
            source_path=args.source_r2_checkpoint,
            coverage=args.coverage,
            manifest=selection_manifest,
        ),
        best_path,
    )

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        set_tau_all(model, tau)
        total = correct = 0
        loss_sum = 0.0
        epoch_t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)

        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": loss_sum / max(total, 1),
            "train_acc": 100.0 * correct / max(total, 1),
            "train_derived_val_loss": val_loss,
            "train_derived_val_acc": val_acc,
            "seconds": time.time() - epoch_t0,
            "official_test": None,
        }
        history.append(row)
        write_json(history, run_dir / "history.json")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                checkpoint_payload(
                    model,
                    epoch=epoch + 1,
                    best_val_acc=best_val_acc,
                    best_epoch=best_epoch,
                    cfg=cfg,
                    source_path=args.source_r2_checkpoint,
                    coverage=args.coverage,
                    manifest=selection_manifest,
                ),
                best_path,
            )

        torch.save(
            checkpoint_payload(
                model,
                epoch=epoch + 1,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                cfg=cfg,
                source_path=args.source_r2_checkpoint,
                coverage=args.coverage,
                manifest=selection_manifest,
            ),
            last_path,
        )

        print(
            f"stage={args.stage} coverage={args.coverage:g}% epoch={epoch + 1}/{epochs} "
            f"train={row['train_acc']:.2f}% val={val_acc:.2f}% "
            f"best_val={best_val_acc:.2f}% lr={row['lr']:.6g}"
        )

    elapsed = time.time() - t0
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    reload_val_loss, reload_val_acc = evaluate(model, val_loader, criterion, device)
    final_terms = term_manifest(model)

    summary = {
        "stage": args.stage,
        "coverage_requested_pct": float(args.coverage),
        "source_r2_checkpoint": args.source_r2_checkpoint,
        "initial_train_derived_val_acc": initial_val_acc,
        "best_train_derived_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "reload_train_derived_val_acc": reload_val_acc,
        "reload_train_derived_val_loss": reload_val_loss,
        "final_epoch_train_derived_val_acc": history[-1]["train_derived_val_acc"] if history else initial_val_acc,
        "elapsed_s": elapsed,
        "official_test_used_for_selection": False,
        "official_test_acc": None,
        "hardware_proxy_at_best": hardware_proxy(final_terms),
        "git_commit": git_commit(),
    }
    write_json(final_terms, run_dir / "best_terms.json")
    write_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))


def official_test_stage(args, cfg, device):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --stage official-test")
    checkpoint_path = Path(args.checkpoint)
    output_path = checkpoint_path.parent / "official_test.json"
    if output_path.exists() and not args.force_official_retest:
        raise RuntimeError(
            f"{output_path} already exists; refusing repeated TEST inspection. "
            "Use --force-official-retest only for a documented reproducibility check."
        )

    state = torch.load(checkpoint_path, map_location=device)
    model = SelectiveSPoTResNet20().to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loader = build_official_test_loader(cfg, smoke=args.smoke)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "checkpoint_best_train_derived_val_acc": state.get("best_train_derived_val_acc"),
        "coverage_pct": state.get("coverage_pct"),
        "official_test_loss": test_loss,
        "official_test_acc": test_acc,
        "selection_was_train_derived_only": True,
        "git_commit": git_commit(),
    }
    write_json(result, output_path)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selective_spot.json")
    ap.add_argument("--stage", choices=["diagnostic", "formal", "official-test"], required=True)
    ap.add_argument("--source-r2-checkpoint", default=None)
    ap.add_argument("--ranking-json", default="reports/selective_spot/S0B_ACTIVATION_AWARE_RANKING.json")
    ap.add_argument("--coverage", type=float, default=0.0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--force-official-retest", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_json(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.stage == "official-test":
        official_test_stage(args, cfg, device)
        return

    if not args.source_r2_checkpoint:
        raise ValueError("--source-r2-checkpoint is required for diagnostic/formal training")
    train_stage(args, cfg, device)


if __name__ == "__main__":
    main()
