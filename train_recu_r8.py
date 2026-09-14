import argparse, json, time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from recu_hw.r7 import R7ResNet20
from recu_hw.r8 import build_r8a_from_r7, build_r8b_from_r7, binary_fc_stats
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.layers import iter_recu_binary_convs
from recu_hw.r6 import is_power_of_two_tensor, parameter_freeze_report, stem_binary_weight_stats
from recu_hw.utils import load_json, save_json, seed_everything, git_commit


def set_tau_all(model, tau):
    for m in iter_recu_binary_convs(model):
        m.set_tau(tau)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def load_r7(path, device):
    st = torch.load(path, map_location=device)
    stored_acc = float(st.get("best_acc", float("nan")))
    stored_epoch = int(st.get("best_epoch", -1))
    if abs(stored_acc - 85.22) > 0.20 or stored_epoch != 99:
        raise RuntimeError(
            "R8 source is not the required R7 best checkpoint: "
            f"best_acc={stored_acc}, best_epoch={stored_epoch}"
        )
    m = R7ResNet20(num_classes=10, resolution=8).to(device)
    m.load_state_dict(st["model"], strict=True)
    m.conv1.set_binary_weight(True)
    m.stem_affine.quantize_k = True
    m.head_affine.quantize_k = True
    for module in m.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    if hasattr(m, "bn2") or m.linear.bias is None:
        raise RuntimeError("R7 source head invariant failed")
    if not is_power_of_two_tensor(m.stem_affine.effective_k()):
        raise RuntimeError("R7 source stem K is not signed-pow2")
    if not is_power_of_two_tensor(m.head_affine.effective_k()):
        raise RuntimeError("R7 source head K is not signed-pow2")
    stem = stem_binary_weight_stats(m)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        raise RuntimeError(f"R7 source stem invariant failed: {stem}")
    m.eval()
    return m, st


def classifier_stats(model, variant):
    return binary_fc_stats(model) if variant == "r8a" else model.linear.stats()


def assert_invariants(model, variant, frozen_snapshot):
    if hasattr(model, "bn2"):
        raise RuntimeError("R8 must not contain standalone bn2")
    if model.linear.bias is None:
        raise RuntimeError("R8 FC bias is missing")
    if not all(bool(torch.isfinite(p).all().item()) for p in model.parameters()):
        raise RuntimeError("R8 model contains NaN/Inf parameters")
    stem = stem_binary_weight_stats(model)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        raise RuntimeError(f"R8 stem invariant failed: {stem}")
    if not is_power_of_two_tensor(model.stem_affine.effective_k()):
        raise RuntimeError("R8 stem K is not signed-pow2")
    if not is_power_of_two_tensor(model.head_affine.effective_k()):
        raise RuntimeError("R8 head K is not signed-pow2")
    cls = classifier_stats(model, variant)
    if variant == "r8a":
        if cls["weights"] != 640 or not cls["is_strict_binary"]:
            raise RuntimeError(f"R8A FC invariant failed: {cls}")
    else:
        if cls["weights"] != 640 or not cls["is_exact_signed_pow2"]:
            raise RuntimeError(f"R8B FC invariant failed: {cls}")
    current = dict(model.named_parameters())
    for name, value in frozen_snapshot.items():
        if current[name].requires_grad:
            raise RuntimeError(f"Frozen alpha was unfrozen: {name}")
        if not torch.equal(current[name].detach(), value):
            raise RuntimeError(f"Frozen alpha changed: {name}")
    return {"stem": stem, "classifier": cls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-r7-checkpoint", required=True)
    ap.add_argument("--variant", choices=["r8a", "r8b"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    seed_everything(int(cfg["experiment"].get("seed", 123)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    src, src_state = load_r7(args.source_r7_checkpoint, device)

    if args.variant == "r8a":
        model = build_r8a_from_r7(src).to(device)
    else:
        model = build_r8b_from_r7(
            src,
            min_exp=cfg["classifier"].get("min_exp"),
            max_exp=cfg["classifier"].get("max_exp"),
        ).to(device)

    freeze = parameter_freeze_report(model)
    if freeze["frozen_params"] != 672:
        raise RuntimeError(f"Expected 672 frozen alpha parameters, got {freeze['frozen_params']}")
    if set(freeze["frozen_parameter_categories"]) != {"r4_compatibility_alpha"}:
        raise RuntimeError(f"Unexpected frozen categories: {freeze['frozen_parameter_categories']}")
    frozen_snapshot = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }

    train_loader, test_loader = build_r5t_loaders(
        {"dataset": cfg["dataset"], "training": cfg["training"]},
        smoke=args.smoke,
    )
    criterion = nn.CrossEntropyLoss()

    _, source_acc = evaluate(src, test_loader, criterion, device)
    _, zero_acc = evaluate(model, test_loader, criterion, device)
    # Smoke evaluates a 512-image subset, so its source accuracy is not
    # expected to equal the stored full CIFAR-10 test accuracy. The source
    # metadata check above remains strict for both modes; the numerical
    # reload comparison is strict only for the formal full-test run.
    if not args.smoke and abs(source_acc - float(src_state.get("best_acc", source_acc))) > 0.20:
        raise RuntimeError(f"R7 source reload mismatch: {source_acc:.4f}%")
    initial_invariants = assert_invariants(model, args.variant, frozen_snapshot)

    ft = cfg["finetune"]
    epochs = 1 if args.smoke else int(ft.get("epochs", 100))
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(ft.get("lr", 1e-4)),
        momentum=float(ft.get("momentum", 0.9)),
        weight_decay=float(ft.get("weight_decay", 0.0)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    tau = float(ft.get("tau", 0.99))
    set_tau_all(model, tau)

    out_root = Path(cfg["output"]["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{cfg['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.smoke:
        run_dir = Path(str(run_dir) + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=False)

    best_acc = zero_acc
    best_epoch = 0
    best_path = run_dir / "best.pt"
    history = []

    save_json({
        "variant": args.variant,
        "source_checkpoint": args.source_r7_checkpoint,
        "source_stored_best_acc": src_state.get("best_acc"),
        "source_stored_best_epoch": src_state.get("best_epoch"),
        "source_reload_acc": source_acc,
        "zero_epoch_projection_acc": zero_acc,
        "delta_zero_vs_source_pp": zero_acc - source_acc,
        "initial_invariants": initial_invariants,
        "freeze_report": freeze,
        "linear_bias_present": model.linear.bias is not None,
    }, run_dir / "conversion.json")

    torch.save({
        "epoch": 0,
        "model": model.state_dict(),
        "best_acc": best_acc,
        "best_epoch": 0,
        "variant": args.variant,
        "config": cfg,
        "source_r7_checkpoint": args.source_r7_checkpoint,
    }, best_path)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        set_tau_all(model, tau)

        total = correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += int((logits.argmax(1) == y).sum())
            total += y.size(0)

        sched.step()
        _, acc = evaluate(model, test_loader, criterion, device)
        inv = assert_invariants(model, args.variant, frozen_snapshot)
        cls_stats = inv["classifier"]
        row = {
            "epoch": epoch + 1,
            "lr": opt.param_groups[0]["lr"],
            "train_loss": loss_sum / max(total, 1),
            "train_acc": 100.0 * correct / max(total, 1),
            "test_acc": acc,
            "stem": inv["stem"],
            "classifier": cls_stats,
        }
        history.append(row)
        save_json(history, run_dir / "history.json")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "variant": args.variant,
                "config": cfg,
                "source_r7_checkpoint": args.source_r7_checkpoint,
            }, best_path)

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "variant": args.variant,
            "config": cfg,
            "source_r7_checkpoint": args.source_r7_checkpoint,
        }, run_dir / "last.pt")

        print(f"{args.variant} epoch={epoch+1}/{epochs} acc={acc:.2f}% best={best_acc:.2f}%")

    elapsed = time.time() - t0
    bst = torch.load(best_path, map_location=device)
    model.load_state_dict(bst["model"], strict=True)
    _, reload_acc = evaluate(model, test_loader, criterion, device)
    final_invariants = assert_invariants(model, args.variant, frozen_snapshot)

    summary = {
        "variant": args.variant,
        "source_stored_best_acc": src_state.get("best_acc"),
        "source_stored_best_epoch": src_state.get("best_epoch"),
        "source_reload_acc": source_acc,
        "zero_epoch_projection_acc": zero_acc,
        "delta_zero_vs_source_pp": zero_acc - source_acc,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_acc": history[-1]["test_acc"] if history else zero_acc,
        "reload_acc": reload_acc,
        "delta_best_vs_source_pp": best_acc - source_acc,
        "elapsed_s": elapsed,
        "classifier": final_invariants["classifier"],
        "stem": final_invariants["stem"],
        "freeze_report": freeze,
        "fc_bias_present": model.linear.bias is not None,
        "git_commit": git_commit(),
    }
    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
