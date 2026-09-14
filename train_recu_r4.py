import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from recu_hw.data import build_recu_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.model import ReCUResNet20
from recu_hw.r4 import build_r4_from_r2, r4_exponent_stats, r4_k_quantization_stats
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-r2-checkpoint", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diagnostic-epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed", 123)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    source = torch.load(args.source_r2_checkpoint, map_location=device)

    r2 = ReCUResNet20(
        num_classes=10,
        activation_mode="qrprelu",
        alpha_mode="float",
    ).to(device)
    r2.load_state_dict(source["model"], strict=True)
    r2.eval()

    model = build_r4_from_r2(r2).to(device)
    tau = float(cfg["finetune"].get("tau", 0.99))
    set_tau_all(model, tau)

    train_loader, test_loader = build_recu_loaders(
        {"dataset": cfg["dataset"], "training": cfg["training"]},
        smoke=args.smoke,
    )

    criterion = nn.CrossEntropyLoss()
    _, initial_acc = evaluate(model, test_loader, criterion, device)

    ft = cfg["finetune"]
    full_epochs = int(ft.get("epochs", 100))
    epochs = 1 if args.smoke else (
        int(args.diagnostic_epochs)
        if args.diagnostic_epochs is not None
        else full_epochs
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(ft.get("lr", 0.005)),
        momentum=float(ft.get("momentum", 0.9)),
        weight_decay=float(ft.get("weight_decay", 5e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, full_epochs), eta_min=0.0
    )

    run_root = Path(cfg["output"]["root"])
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{cfg['experiment']['name']}_{stamp}"
    if args.smoke:
        run_dir = Path(str(run_dir) + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(cfg, run_dir / "config.json")
    save_json({
        "source_r2_checkpoint": args.source_r2_checkpoint,
        "source_r2_best_acc": source.get("best_acc"),
        "source_r2_best_epoch": source.get("best_epoch"),
        "initial_converted_test_acc": initial_acc,
        "initial_exponents": r4_exponent_stats(model),
        "initial_k_quantization": r4_k_quantization_stats(model),
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
        "source_r2_checkpoint": args.source_r2_checkpoint,
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
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}")

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
                "source_r2_checkpoint": args.source_r2_checkpoint,
            }, best_path)

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "config": cfg,
            "source_r2_checkpoint": args.source_r2_checkpoint,
        }, last_path)

        print(
            f"epoch={epoch+1}/{epochs} "
            f"train={row['train_acc']:.2f}% "
            f"test={test_acc:.2f}% best={best_acc:.2f}%"
        )

    elapsed = time.time() - t0

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    _, reload_acc = evaluate(model, test_loader, criterion, device)

    summary = {
        "source_r2_checkpoint": args.source_r2_checkpoint,
        "source_r2_best_acc": source.get("best_acc"),
        "source_r2_best_epoch": source.get("best_epoch"),
        "initial_converted_test_acc": initial_acc,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "reload_test_acc": reload_acc,
        "final_test_acc": history[-1]["test_acc"] if history else initial_acc,
        "elapsed_s": elapsed,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "final_exponents": r4_exponent_stats(model),
        "final_k_quantization": r4_k_quantization_stats(model),
        "git_commit": git_commit(),
    }

    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
