import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from recu_hw.ablation import (
    build_r1_from_official,
    build_r2_from_official,
    build_r3_from_official,
    fused_exponent_stats,
)
from recu_hw.data import build_recu_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.model import ReCUResNet20
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


def load_official(checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    official = ReCUResNet20(
        activation_mode="prelu",
        alpha_mode="float",
    ).to(device)
    official.load_state_dict(state["model"], strict=True)
    official.eval()
    return official, state


def build_target(mode, official):
    if mode == "r1":
        return build_r1_from_official(official), {}
    if mode == "r2":
        model, nbad = build_r2_from_official(official)
        return model, {"nonpositive_prelu_slopes_clamped": nbad}
    if mode == "r3":
        model = build_r3_from_official(official, affine_mode="pow2")
        return model, {"initial_fused_exponents": fused_exponent_stats(model)}
    raise ValueError(mode)


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-checkpoint", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diagnostic-epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    mode = cfg["experiment"]["mode"]
    seed = int(cfg["experiment"].get("seed", 123))
    seed_everything(seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    official, source_state = load_official(args.source_checkpoint, device)
    model, conversion_meta = build_target(mode, official)
    model = model.to(device)

    # Dataset loader reuses the known official ReCU data protocol.
    loader_cfg = {
        "dataset": cfg["dataset"],
        "training": cfg["training"],
    }
    train_loader, test_loader = build_recu_loaders(loader_cfg, smoke=args.smoke)

    ft = cfg["finetune"]
    full_epochs = int(ft.get("epochs", 100))
    if args.smoke:
        epochs = 1
    elif args.diagnostic_epochs is not None:
        epochs = int(args.diagnostic_epochs)
    else:
        epochs = full_epochs

    lr = float(ft.get("lr", 0.01))
    momentum = float(ft.get("momentum", 0.9))
    wd = float(ft.get("weight_decay", 5e-4))
    tau = float(ft.get("tau", 0.99))

    # Warm-start fine-tuning: keep ReCU tau at the converged official value.
    set_tau_all(model, tau)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, full_epochs), eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss()

    output_root = Path(cfg["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{cfg['experiment']['name']}_{stamp}"
    if args.smoke:
        run_dir = Path(str(run_dir) + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(cfg, run_dir / "config.json")
    save_json(
        {
            "source_checkpoint": str(args.source_checkpoint),
            "source_best_acc": float(source_state.get("best_acc", float("nan"))),
            "source_best_epoch": int(source_state.get("best_epoch", -1)),
            "conversion": conversion_meta,
            "trainable_params": trainable_count(model),
        },
        run_dir / "conversion.json",
    )

    # Accuracy immediately after conversion, before any fine-tuning.
    _, initial_acc = evaluate(model, test_loader, criterion, device)
    print(f"mode={mode} initial_converted_test_acc={initial_acc:.2f}%")

    best_acc = initial_acc
    best_epoch = 0
    history = []
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    # Save epoch-0 converted model as candidate best.
    torch.save(
        {
            "epoch": 0,
            "model": model.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "config": cfg,
            "source_checkpoint": str(args.source_checkpoint),
        },
        best_path,
    )

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
        train_loss = loss_sum / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "tau": tau,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        save_json(history, run_dir / "history.json")

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
            "source_checkpoint": str(args.source_checkpoint),
        }
        torch.save(state, last_path)
        if is_best:
            torch.save(state, best_path)

        print(
            f"epoch={epoch+1}/{epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"train={train_acc:.2f}% test={test_acc:.2f}% "
            f"best={best_acc:.2f}%"
        )

    elapsed = time.time() - t0

    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state["model"])
    _, reload_acc = evaluate(model, test_loader, criterion, device)

    summary = {
        "mode": mode,
        "source_checkpoint": str(args.source_checkpoint),
        "initial_converted_test_acc": initial_acc,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "reload_test_acc": reload_acc,
        "final_test_acc": history[-1]["test_acc"] if history else initial_acc,
        "elapsed_s": elapsed,
        "trainable_params": trainable_count(model),
        "conversion": conversion_meta,
        "git_commit": git_commit(),
    }

    if mode == "r3":
        summary["final_fused_exponents"] = fused_exponent_stats(model)

    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
