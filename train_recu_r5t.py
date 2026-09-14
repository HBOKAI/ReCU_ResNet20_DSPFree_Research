import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from recu_hw.layers import iter_recu_binary_convs
from recu_hw.r4 import R4ResNet20
from recu_hw.r5t import build_r5t_from_r4, R5TResNet20
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.utils import git_commit, load_json, save_json, seed_everything


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
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def set_tau_all(model, tau):
    for m in iter_recu_binary_convs(model):
        m.set_tau(tau)


def linear_lambda(epoch_idx, total_epochs):
    if total_epochs <= 1:
        return 0.0
    return max(0.0, 1.0 - (epoch_idx + 1) / float(total_epochs))


def run_stage(model, stage_name, stage_cfg, train_loader, test_loader, criterion, device, run_dir, tau):
    epochs = int(stage_cfg["epochs"])
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(stage_cfg["lr"]),
        momentum=float(stage_cfg.get("momentum", 0.9)),
        weight_decay=float(stage_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: linear_lambda(e, epochs),
    )

    history = []
    best_acc = -1.0
    best_epoch = -1
    best_path = run_dir / f"{stage_name}_best.pt"
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        set_tau_all(model, tau)
        total = correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {stage_name} epoch {epoch+1}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(total, 1),
            "train_acc": 100.0 * correct / max(total, 1),
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        save_json(history, run_dir / f"{stage_name}_history.json")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save({
                "stage": stage_name,
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
            }, best_path)

        print(f"{stage_name} epoch={epoch+1}/{epochs} test={test_acc:.2f}% best={best_acc:.2f}%")

    elapsed = time.time() - t0
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    _, reload_acc = evaluate(model, test_loader, criterion, device)
    return {
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "reload_acc": reload_acc,
        "elapsed_s": elapsed,
        "best_path": str(best_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-r4-checkpoint", required=True)
    ap.add_argument("--stage", choices=["t1", "t2", "both"], default="both")
    ap.add_argument("--source-t1-checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diagnostic-epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed", 123)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    src = torch.load(args.source_r4_checkpoint, map_location=device)
    r4 = R4ResNet20(num_classes=10).to(device)
    r4.load_state_dict(src["model"], strict=True)
    r4.eval()

    resolution = int(cfg["thermometer"].get("resolution", 8))
    cifar_std = tuple(cfg["thermometer"].get(
        "cifar_normalize_std_reference", [0.2470, 0.2435, 0.2616]
    ))

    model = build_r5t_from_r4(
        r4,
        resolution=resolution,
        binary_stem=False,
        cifar_std=cifar_std,
    ).to(device)

    train_loader, test_loader = build_r5t_loaders(cfg, smoke=args.smoke)
    criterion = nn.CrossEntropyLoss()
    tau = float(cfg.get("recu_tau", 0.99))
    set_tau_all(model, tau)

    _, projected_fp_acc = evaluate(model, test_loader, criterion, device)
    model.conv1.set_binary_weight(True)
    _, zero_binary_acc = evaluate(model, test_loader, criterion, device)
    model.conv1.set_binary_weight(False)

    out_root = Path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{cfg['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.smoke:
        run_dir = Path(str(run_dir) + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json({
        "source_r4_checkpoint": args.source_r4_checkpoint,
        "source_r4_best_acc": src.get("best_acc"),
        "source_r4_best_epoch": src.get("best_epoch"),
        "resolution": resolution,
        "thermometer_length_per_rgb": model.thermo.length,
        "input_channels": 3 * model.thermo.length,
        "projected_fp_stem_zero_epoch_acc": projected_fp_acc,
        "binary_stem_zero_epoch_acc": zero_binary_acc,
        "binary_shock_pp": zero_binary_acc - projected_fp_acc,
    }, run_dir / "conversion.json")

    results = {}

    if args.stage in ("t1", "both"):
        model.conv1.set_binary_weight(False)
        t1_cfg = dict(cfg["stage_t1"])
        if args.smoke:
            t1_cfg["epochs"] = 1
        elif args.diagnostic_epochs is not None:
            t1_cfg["epochs"] = int(args.diagnostic_epochs)
        results["t1"] = run_stage(
            model, "t1", t1_cfg, train_loader, test_loader, criterion, device, run_dir, tau
        )

    if args.stage in ("t2", "both"):
        if args.stage == "t2":
            if not args.source_t1_checkpoint:
                raise ValueError("--source-t1-checkpoint is required for --stage t2")
            t1state = torch.load(args.source_t1_checkpoint, map_location=device)
            model.load_state_dict(t1state["model"], strict=True)

        model.conv1.set_binary_weight(True)
        _, t2_zero = evaluate(model, test_loader, criterion, device)
        save_json({"t2_zero_epoch_binary_acc": t2_zero}, run_dir / "t2_conversion.json")

        t2_cfg = dict(cfg["stage_t2"])
        if args.smoke:
            t2_cfg["epochs"] = 1
        elif args.diagnostic_epochs is not None:
            t2_cfg["epochs"] = int(args.diagnostic_epochs)
        results["t2"] = run_stage(
            model, "t2", t2_cfg, train_loader, test_loader, criterion, device, run_dir, tau
        )
        model.conv1.set_binary_weight(True)

    final_summary = {
        "source_r4_best_acc": src.get("best_acc"),
        "source_r4_best_epoch": src.get("best_epoch"),
        "resolution": resolution,
        "thermometer_input_channels": 3 * model.thermo.length,
        "projected_fp_stem_zero_epoch_acc": projected_fp_acc,
        "binary_stem_zero_epoch_acc": zero_binary_acc,
        "stages": results,
        "stem_binary_weight": model.conv1.binary_weight,
        "stem_sign_stats": model.conv1.sign_stats(),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "git_commit": git_commit(),
    }
    save_json(final_summary, run_dir / "summary.json")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
