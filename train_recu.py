import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn

from recu_hw.data import build_recu_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.model import build_recu_resnet20, parameter_breakdown
from recu_hw.schedule import official_warmup_lr, recu_tau
from recu_hw.utils import (
    append_registry,
    git_commit,
    load_json,
    make_run_dir,
    save_json,
    seed_everything,
)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    n = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return loss_sum / max(n, 1), 100.0 * correct / max(n, 1)


def set_tau_all(model, tau):
    for m in iter_recu_binary_convs(model):
        m.set_tau(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diagnostic-epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed = int(cfg["experiment"].get("seed", 123))
    seed_everything(seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_recu_resnet20(cfg).to(device)
    breakdown = parameter_breakdown(model)
    print("Parameter breakdown:", breakdown)

    train_loader, test_loader = build_recu_loaders(cfg, smoke=args.smoke)
    tcfg = cfg["training"]

    official_epochs = int(tcfg.get("epochs", 600))
    if args.smoke:
        epochs = 1
    elif args.diagnostic_epochs is not None:
        epochs = int(args.diagnostic_epochs)
    else:
        epochs = official_epochs

    base_lr = float(tcfg.get("lr", 0.1))
    wd = float(tcfg.get("weight_decay", 5e-4))
    momentum = float(tcfg.get("momentum", 0.9))
    tau_min = float(tcfg.get("tau_min", 0.85))
    tau_max = float(tcfg.get("tau_max", 0.99))
    warm_up = bool(tcfg.get("warm_up", True))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=wd,
    )

    # Reproduce official scheduler semantics:
    # T_max = 600 - 4 when warm_up=True.
    scheduler_tmax = max(1, official_epochs - (4 if warm_up else 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_tmax, eta_min=0
    )
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc = -1.0
    best_epoch = -1
    run_dir = make_run_dir(cfg, smoke=args.smoke)
    save_json(cfg, run_dir / "config.json")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"])
        best_acc = float(ckpt.get("best_acc", -1.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        print(f"Resumed from epoch {start_epoch}")

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history = []
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        # Official warm-up modifies the optimizer LR directly for first 5 epochs.
        if warm_up and epoch < 5:
            lr = official_warmup_lr(base_lr, epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        tau = recu_tau(epoch, official_epochs, tau_min, tau_max)
        set_tau_all(model, tau)

        model.train()
        n = 0
        correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)

        # Official scheduler starts stepping at epoch >= 4 when warm_up=True.
        if (not warm_up) or epoch >= 4:
            scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        train_loss = loss_sum / max(n, 1)
        train_acc = 100.0 * correct / max(n, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch + 1,
            "tau": tau,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)

        print(
            f"epoch={epoch+1}/{epochs} "
            f"tau={tau:.6f} lr={lr_now:.6g} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%"
        )

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch + 1

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "config": cfg,
            "parameter_breakdown": breakdown,
        }
        torch.save(state, last_path)
        if is_best:
            torch.save(state, best_path)

        save_json(history, run_dir / "history.json")

    elapsed = time.time() - t0

    # Best checkpoint reload verification.
    if best_path.exists():
        reload_state = torch.load(best_path, map_location=device)
        model.load_state_dict(reload_state["model"])
        _, reload_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"best_epoch={best_epoch} best_test_acc={best_acc:.2f}% "
            f"reload_test_acc={reload_acc:.2f}% elapsed_s={elapsed:.1f}"
        )
    else:
        reload_acc = float("nan")

    append_registry(
        {
            "experiment_id": run_dir.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "seed": seed,
            "activation_mode": cfg["model"].get("activation_mode", "prelu"),
            "alpha_mode": cfg["model"].get("alpha_mode", "float"),
            "params": breakdown["total"],
            "epochs": epochs,
            "batch_size": tcfg.get("batch_size", 256),
            "lr": base_lr,
            "weight_decay": wd,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "checkpoint_path": str(best_path),
            "config_path": str(run_dir / "config.json"),
            "status": "completed",
        },
        cfg["output"].get("registry", "experiments/recu_registry.csv"),
    )


if __name__ == "__main__":
    main()
