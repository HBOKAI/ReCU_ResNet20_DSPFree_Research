"""R8B-recipe GMP fine-tune from the identical R7 source initialization.

The official CIFAR-10 test set is loaded only after the best validation
checkpoint has been frozen. Existing GAP checkpoints are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.gmp_ablation import assert_same_parameters, build_gmp_from_r7
from recu_hw.r6 import parameter_freeze_report
from recu_hw.r8 import build_r8b_from_r7
from recu_hw.utils import git_commit, load_json, save_json, seed_everything
from train_recu_r8 import assert_invariants, evaluate, load_r7, set_tau_all


CONFIG = ROOT / "configs/recu_r8b.json"
SOURCE_R7 = ROOT / "experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/best.pt"
OUTPUT_ROOT = ROOT / "experiments/gmp_ablation"


def frozen_alpha_snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}


def binary_terms(model):
    from recu_hw.layers import iter_recu_binary_convs

    counts = {}
    modules = [("stem", model.conv1)] + [
        (f"backbone.{i}", module) for i, module in enumerate(iter_recu_binary_convs(model))
    ]
    handles = []
    for name, module in modules:
        def hook(mod, _inputs, output, label=name):
            counts[label] = int(output.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1])

        handles.append(module.register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, 3, 32, 32), device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    if len(counts) != 19:
        raise RuntimeError(f"Expected 19 binary convolutions, got {len(counts)}")
    return {
        "stem": counts["stem"],
        "backbone": sum(v for k, v in counts.items() if k != "stem"),
        "total": sum(counts.values()),
    }


def make_train_validation(cfg, smoke):
    root = str(ROOT / cfg["dataset"].get("root", "./data")) if not Path(cfg["dataset"].get("root", "./data")).is_absolute() else cfg["dataset"]["root"]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    plain_tf = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root, train=True, transform=train_tf, download=False)
    val_dataset = datasets.CIFAR10(root, train=True, transform=plain_tf, download=False)
    if len(train_dataset) != 50000 or len(val_dataset) != 50000:
        raise RuntimeError("Expected complete CIFAR-10 TRAIN = 50,000 images")
    seed = int(cfg["experiment"].get("seed", 123))
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices, val_indices = indices[:45000], indices[45000:]
    if not set(train_indices).isdisjoint(val_indices):
        raise RuntimeError("TRAIN/validation split overlaps")
    split = {
        "source": "CIFAR-10 TRAIN only",
        "seed": seed,
        "train_count_full": len(train_indices),
        "validation_count_full": len(val_indices),
        "train_indices_sha256": hashlib.sha256(np.asarray(train_indices, dtype=np.int32).tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(np.asarray(val_indices, dtype=np.int32).tobytes()).hexdigest(),
        "validation_indices": val_indices,
    }
    if smoke:
        train_indices, val_indices = train_indices[:1024], val_indices[:512]
    batch = int(cfg["training"].get("batch_size", 128))
    val_batch = int(cfg["training"].get("test_batch_size", 128))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=val_batch, shuffle=False,
        num_workers=0, pin_memory=pin,
    )
    split["train_count_used"] = len(train_indices)
    split["validation_count_used"] = len(val_indices)
    return train_loader, val_loader, split


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, best_epoch, cfg):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_validation_acc": best_acc,
        "best_epoch": best_epoch,
        "config": cfg,
        "source_r7_checkpoint": str(SOURCE_R7),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", type=Path, help="Existing GMP run directory; resume from last.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.smoke and args.resume:
        raise ValueError("Smoke and resume are mutually exclusive")

    cfg = load_json(CONFIG)
    seed_everything(int(cfg["experiment"].get("seed", 123)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source, source_state = load_r7(str(SOURCE_R7), device)
    gap_control = build_r8b_from_r7(
        source, min_exp=cfg["classifier"].get("min_exp"), max_exp=cfg["classifier"].get("max_exp")
    )
    model = build_gmp_from_r7(
        source, min_exp=cfg["classifier"].get("min_exp"), max_exp=cfg["classifier"].get("max_exp")
    )
    assert_same_parameters(gap_control, model)
    model.to(device)
    freeze = parameter_freeze_report(model)
    if freeze["frozen_params"] != 672 or set(freeze["frozen_parameter_categories"]) != {"r4_compatibility_alpha"}:
        raise RuntimeError(f"Frozen compatibility-alpha mismatch: {freeze}")
    frozen = frozen_alpha_snapshot(model)
    initial_invariants = assert_invariants(model, "r8b", frozen)
    complexity = {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "binary_conv_terms_per_image": binary_terms(model),
    }
    if complexity["trainable_parameters"] != 284250 or complexity["binary_conv_terms_per_image"]["total"] != 54263808:
        raise RuntimeError(f"GMP complexity drift: {complexity}")

    train_loader, val_loader, split = make_train_validation(cfg, args.smoke)
    criterion = nn.CrossEntropyLoss()
    set_tau_all(model, float(cfg["finetune"]["tau"]))
    _, zero_val = evaluate(model, val_loader, criterion, device)
    epochs = 1 if args.smoke else int(cfg["finetune"]["epochs"])
    ft = cfg["finetune"]
    if not args.smoke and epochs != 100:
        raise RuntimeError("Formal GMP run must use the frozen R8B 100-epoch recipe")
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(ft["lr"]), momentum=float(ft["momentum"]), weight_decay=float(ft["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if args.resume:
        run_dir = args.resume.resolve()
        if not run_dir.is_relative_to(OUTPUT_ROOT.resolve()):
            raise ValueError("Resume directory must be inside experiments/gmp_ablation")
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if ckpt["config"] != cfg or ckpt["source_r7_checkpoint"] != str(SOURCE_R7):
            raise RuntimeError("Resume checkpoint provenance/config mismatch")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if ckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        start_epoch = int(ckpt["epoch"]) + 1
        best_val, best_epoch = float(ckpt["best_validation_acc"]), int(ckpt["best_epoch"])
        if len(history) != start_epoch - 1:
            raise RuntimeError("Resume history/epoch mismatch")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_ROOT / f"gmp_r8b_from_r7_{stamp}{'_smoke' if args.smoke else ''}"
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(split, run_dir / "train_validation_split.json")
        save_json({
            "method": "GMP fine-tuned from GAP R7 source; signed-pow2 FC initialized per R8B",
            "source_r7_checkpoint": str(SOURCE_R7),
            "source_r7_best_acc": source_state.get("best_acc"),
            "source_r7_best_epoch": source_state.get("best_epoch"),
            "initial_gap_gmp_state_equal": True,
            "zero_epoch_gmp_validation_accuracy": zero_val,
            "split": {k: v for k, v in split.items() if k != "validation_indices"},
            "complexity": complexity,
            "frozen": freeze,
            "initial_invariants": initial_invariants,
            "recipe": {
                "epochs": epochs,
                "batch_size": cfg["training"]["batch_size"],
                "validation_batch_size": cfg["training"]["test_batch_size"],
                "optimizer": "SGD",
                "learning_rate": ft["lr"],
                "momentum": ft["momentum"],
                "weight_decay": ft["weight_decay"],
                "scheduler": "CosineAnnealingLR",
                "T_max": epochs,
                "tau": ft["tau"],
                "augmentation": "RandomCrop(32,padding=4), RandomHorizontalFlip, ToTensor; no Normalize",
                "validation_source": "disjoint CIFAR-10 TRAIN-derived 5,000 images",
            },
            "comparable_gap_r8b_official_test_accuracy": 85.40,
            "safe_gap_h2a_v2_official_test_accuracy": 85.03,
        }, run_dir / "setup.json")
        history = []
        start_epoch = 1
        best_val, best_epoch = zero_val, 0
        save_checkpoint(run_dir / "best.pt", 0, model, optimizer, scheduler, best_val, best_epoch, cfg)

    print(f"GMP run={run_dir} device={device} train={split['train_count_used']} validation={split['validation_count_used']} start={start_epoch}/{epochs}", flush=True)
    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        set_tau_all(model, float(ft["tau"]))
        total = correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite GMP train loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        invariants = assert_invariants(model, "r8b", frozen)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            save_checkpoint(run_dir / "best.pt", epoch, model, optimizer, scheduler, best_val, best_epoch, cfg)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": loss_sum / total,
            "train_accuracy": 100.0 * correct / total,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "best_validation_accuracy_so_far": best_val,
            "stem_binary": invariants["stem"]["is_strict_binary"],
            "classifier_signed_pow2": invariants["classifier"]["is_exact_signed_pow2"],
            "elapsed_s": time.time() - epoch_t0,
        }
        history.append(row)
        save_json(history, run_dir / "history.json")
        save_checkpoint(run_dir / "last.pt", epoch, model, optimizer, scheduler, best_val, best_epoch, cfg)
        print(
            f"GMP epoch={epoch}/{epochs} loss={row['train_loss']:.4f} train={row['train_accuracy']:.2f}% "
            f"val={val_acc:.2f}% best={best_val:.2f}%@{best_epoch} lr={row['lr']:.8f} "
            f"time={row['elapsed_s']:.1f}s", flush=True,
        )

    best_state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    if int(best_state["best_epoch"]) != best_epoch:
        raise RuntimeError("Best checkpoint/selection mismatch")
    model.load_state_dict(best_state["model"], strict=True)
    _, reload_val = evaluate(model, val_loader, criterion, device)
    if abs(reload_val - best_val) > 1e-9:
        raise RuntimeError(f"Best validation checkpoint reload mismatch: {reload_val} vs {best_val}")
    final_invariants = assert_invariants(model, "r8b", frozen)
    official_test_acc = None
    official_test_count = 0
    if not args.smoke:
        test_dataset = datasets.CIFAR10(
            str(ROOT / "data"), train=False, transform=transforms.ToTensor(), download=False,
        )
        if len(test_dataset) != 10000:
            raise RuntimeError("Expected complete CIFAR-10 official TEST")
        test_loader = DataLoader(
            test_dataset, batch_size=int(cfg["training"]["test_batch_size"]),
            shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        _, official_test_acc = evaluate(model, test_loader, criterion, device)
        official_test_count = len(test_dataset)
    summary = {
        "phase": "B_GMP_R8B_level_fine_tune_from_GAP_R7_source",
        "run_dir": str(run_dir),
        "method": "GMP fine-tuned from GAP R7 source; not scratch training",
        "source_checkpoint": str(SOURCE_R7),
        "seed": int(cfg["experiment"]["seed"]),
        "epochs": epochs,
        "train_count": split["train_count_used"],
        "validation_count": split["validation_count_used"],
        "zero_epoch_validation_accuracy": zero_val,
        "best_validation_accuracy": best_val,
        "best_epoch": best_epoch,
        "final_epoch_validation_accuracy": history[-1]["validation_accuracy"] if history else zero_val,
        "best_validation_reload_accuracy": reload_val,
        "official_test_accuracy": official_test_acc,
        "official_test_count": official_test_count,
        "delta_vs_comparable_gap_r8b_85_40_pp": official_test_acc - 85.40 if official_test_acc is not None else None,
        "delta_vs_safe_gap_h2a_v2_85_03_pp": official_test_acc - 85.03 if official_test_acc is not None else None,
        "complexity": complexity,
        "classifier": final_invariants["classifier"],
        "stem": final_invariants["stem"],
        "initial_gap_gmp_state_equal": True,
        "elapsed_this_process_s": time.time() - t0,
        "git_commit": git_commit(),
    }
    save_json(summary, run_dir / "summary.json")
    print(json.dumps({
        "run_dir": str(run_dir),
        "best_validation_accuracy": best_val,
        "best_epoch": best_epoch,
        "best_validation_reload_accuracy": reload_val,
        "official_test_accuracy": official_test_acc,
        "delta_vs_comparable_gap_r8b_85_40_pp": summary["delta_vs_comparable_gap_r8b_85_40_pp"],
        "delta_vs_safe_gap_h2a_v2_85_03_pp": summary["delta_vs_safe_gap_h2a_v2_85_03_pp"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
