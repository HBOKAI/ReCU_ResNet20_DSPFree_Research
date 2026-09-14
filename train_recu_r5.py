import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from recu_hw.data import build_recu_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.r4 import R4ResNet20
from recu_hw.r5 import build_r5_from_r4, stem_weight_stats
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


@torch.no_grad()
def activation_quant_stats(model, loader, device, max_batches=16):
    totals = {
        "total": 0,
        "sat_low": 0,
        "sat_high": 0,
        "input_min": float("inf"),
        "input_max": float("-inf"),
        "q_min": 10**9,
        "q_max": -10**9,
    }

    for idx, (x, _) in enumerate(loader):
        if idx >= max_batches:
            break
        x = x.to(device)
        s = model.input_quant.stats(x)
        totals["total"] += s["total"]
        totals["sat_low"] += s["sat_low"]
        totals["sat_high"] += s["sat_high"]
        totals["input_min"] = min(totals["input_min"], s["input_min"])
        totals["input_max"] = max(totals["input_max"], s["input_max"])
        totals["q_min"] = min(totals["q_min"], s["q_observed_min"])
        totals["q_max"] = max(totals["q_max"], s["q_observed_max"])

    totals["saturation_fraction"] = (
        totals["sat_low"] + totals["sat_high"]
    ) / max(totals["total"], 1)
    totals["scale_exp"] = int(model.input_quant.scale_exp.item())
    totals["scale"] = model.input_quant.scale
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-r4-checkpoint", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diagnostic-epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed", 123)))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    source = torch.load(args.source_r4_checkpoint, map_location=device)

    r4 = R4ResNet20(num_classes=10).to(device)
    r4.load_state_dict(source["model"], strict=True)
    r4.eval()

    input_scale_exp = int(cfg["model"].get("input_scale_exp", -5))
    model = build_r5_from_r4(
        r4,
        input_scale_exp=input_scale_exp,
    ).to(device)

    tau = float(cfg["finetune"].get("tau", 0.99))
    set_tau_all(model, tau)

    train_loader, test_loader = build_recu_loaders(
        {
            "dataset": cfg["dataset"],
            "training": cfg["training"],
        },
        smoke=args.smoke,
    )

    criterion = nn.CrossEntropyLoss()

    _, initial_acc = evaluate(
        model, test_loader, criterion, device
    )
    qstats = activation_quant_stats(
        model, train_loader, device,
        max_batches=int(cfg["model"].get("calibration_batches", 16)),
    )
    wstats = stem_weight_stats(model)

    ft = cfg["finetune"]
    full_epochs = int(ft.get("epochs", 100))
    if args.smoke:
        epochs = 1
    elif args.diagnostic_epochs is not None:
        epochs = int(args.diagnostic_epochs)
    else:
        epochs = full_epochs

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(ft.get("lr", 0.005)),
        momentum=float(ft.get("momentum", 0.9)),
        weight_decay=float(ft.get("weight_decay", 5e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, full_epochs),
        eta_min=0.0,
    )

    out_root = Path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{cfg['experiment']['name']}_{stamp}"
    if args.smoke:
        run_dir = Path(str(run_dir) + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(cfg, run_dir / "config.json")
    save_json({
        "source_r4_checkpoint": args.source_r4_checkpoint,
        "source_r4_best_acc": source.get("best_acc"),
        "source_r4_best_epoch": source.get("best_epoch"),
        "initial_converted_test_acc": initial_acc,
        "input_quantization": qstats,
        "stem_weight_projection": wstats,
    }, run_dir / "conversion.json")

    best_acc = initial_acc
    best_epoch = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history = []

    torch.save({
        "epoch": 0,
        "model": model.state_dict(),
        "best_acc": best_acc,
        "best_epoch": 0,
        "config": cfg,
        "source_r4_checkpoint": args.source_r4_checkpoint,
    }, best_path)

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
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch+1}"
                )

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(total, 1),
            "train_acc": 100.0 * correct / max(total, 1),
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        save_json(history, run_dir / "history.json")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "config": cfg,
                "source_r4_checkpoint": args.source_r4_checkpoint,
            }, best_path)

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "config": cfg,
            "source_r4_checkpoint": args.source_r4_checkpoint,
        }, last_path)

        print(
            f"epoch={epoch+1}/{epochs} "
            f"train={row['train_acc']:.2f}% "
            f"test={test_acc:.2f}% "
            f"best={best_acc:.2f}%"
        )

    elapsed = time.time() - t0

    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state["model"], strict=True)
    _, reload_acc = evaluate(
        model, test_loader, criterion, device
    )

    summary = {
        "source_r4_checkpoint": args.source_r4_checkpoint,
        "source_r4_best_acc": source.get("best_acc"),
        "source_r4_best_epoch": source.get("best_epoch"),
        "initial_converted_test_acc": initial_acc,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "reload_test_acc": reload_acc,
        "final_test_acc": (
            history[-1]["test_acc"] if history else initial_acc
        ),
        "elapsed_s": elapsed,
        "trainable_params": sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        ),
        "input_quantization": activation_quant_stats(
            model, train_loader, device, max_batches=16
        ),
        "stem_weight_projection": stem_weight_stats(model),
        "git_commit": git_commit(),
    }

    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
