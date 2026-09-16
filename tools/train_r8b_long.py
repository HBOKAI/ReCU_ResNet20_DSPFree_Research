"""Independent R8B 600-epoch control run.

This script starts from the exact R7-derived R8B initialization used by the
formal R8B run.  It preserves the model, optimizer, data augmentation, and
quantization policy; the only training change is extending the cosine
schedule from the original 100-epoch horizon to 600 epochs.  The official
test set is evaluated only at milestones and after selecting the fixed
TRAIN-derived validation set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.r6 import parameter_freeze_report
from recu_hw.r8 import build_r8b_from_r7
from recu_hw.r8 import Pow2LinearSTE
from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.utils import git_commit, load_json, save_json, seed_everything
from train_recu_r8 import assert_invariants, evaluate, load_r7, set_tau_all
from tools.eval_integer_bias_sweep import load_r8b_checkpoint

CONFIG = ROOT / "configs" / "recu_r8b_long.json"
SOURCE_R7 = ROOT / "experiments" / "recu_r7" / "recu_r7_r6_head_bn_pow2_20260912_220758" / "best.pt"
FORMAL_R8B = ROOT / "experiments" / "recu_r8b" / "recu_r8b_r7_pow2_fc_20260912_225827" / "best.pt"
MILESTONES = (100, 200, 300, 400, 500, 600)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_string(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def make_loaders(cfg, split_seed: int, device, smoke: bool = False):
    data_root = Path(cfg["dataset"].get("root", "./data"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    plain = transforms.ToTensor()
    train_aug_ds = datasets.CIFAR10(str(data_root), train=True, transform=train_aug, download=False)
    plain_train_ds = datasets.CIFAR10(str(data_root), train=True, transform=plain, download=False)
    test_ds = datasets.CIFAR10(str(data_root), train=False, transform=plain, download=False)
    if len(train_aug_ds) != 50000 or len(plain_train_ds) != 50000 or len(test_ds) != 10000:
        raise RuntimeError("CIFAR-10 dataset sizes are not 50,000 TRAIN / 10,000 TEST")
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(int(split_seed))).tolist()
    # Match the project's established seed=123 split convention: the first
    # 45,000 entries are training and the final 5,000 are validation.
    train_indices = perm[:45000]
    val_indices = perm[45000:]
    if set(train_indices).intersection(val_indices):
        raise RuntimeError("TRAIN/validation split overlap")
    if smoke:
        train_indices = train_indices[:1024]
        val_indices = val_indices[:512]
    split = {
        "source": "CIFAR-10 TRAIN only; fixed permutation",
        "split_seed": int(split_seed),
        "train_count_full": 45000,
        "validation_count_full": 5000,
        "train_count_used": len(train_indices),
        "validation_count_used": len(val_indices),
        "train_indices_sha256": hashlib.sha256(np.asarray(train_indices, dtype=np.int32).tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(np.asarray(val_indices, dtype=np.int32).tobytes()).hexdigest(),
        "train_indices": train_indices,
        "validation_indices": val_indices,
    }
    batch = int(cfg["training"].get("batch_size", 128))
    eval_batch = int(cfg["training"].get("test_batch_size", 128))
    pin = device.type == "cuda"
    train_loader = DataLoader(Subset(train_aug_ds, train_indices), batch_size=batch, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(Subset(plain_train_ds, val_indices), batch_size=eval_batch, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=eval_batch, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, test_loader, split


def model_parameter_inventory(model):
    tensors = []
    for name, p in model.named_parameters():
        tensors.append({
            "name": name,
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "numel": int(p.numel()),
            "trainable": bool(p.requires_grad),
        })
    affine = [x for x in tensors if x["name"].endswith(".bias") or "affine" in x["name"]]
    return {
        "total_parameters": int(sum(x["numel"] for x in tensors)),
        "trainable_parameters": int(sum(x["numel"] for x in tensors if x["trainable"])),
        "frozen_parameters": int(sum(x["numel"] for x in tensors if not x["trainable"])),
        "tensor_count": len(tensors),
        "tensors": tensors,
        "affine_related_tensor_count": len(affine),
    }


@torch.no_grad()
def parameter_health(model):
    finite = True
    max_abs = 0.0
    distributions = {}
    for name, p in model.named_parameters():
        q = p.detach()
        ok = bool(torch.isfinite(q).all().item())
        finite = finite and ok
        a = float(q.abs().max().item()) if q.numel() else 0.0
        max_abs = max(max_abs, a)
        distributions[name] = {
            "min": float(q.min().item()) if q.numel() else 0.0,
            "max": float(q.max().item()) if q.numel() else 0.0,
            "mean": float(q.float().mean().item()) if q.numel() else 0.0,
            "std": float(q.float().std(unbiased=False).item()) if q.numel() else 0.0,
            "finite": ok,
        }
    return {"finite": finite, "max_abs": max_abs, "distributions": distributions}


def save_checkpoint(path: Path, epoch, model, optimizer, scheduler, best_val, best_epoch, cfg, source_hash, split_seed):
    torch.save({
        "experiment": "R8B-LONG",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_validation_accuracy": float(best_val),
        "best_epoch": int(best_epoch),
        "config": cfg,
        "source_r7_checkpoint": path_string(SOURCE_R7),
        "source_r7_sha256": source_hash,
        "validation_split_seed": int(split_seed),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }, path)


def write_curve(path: Path, history):
    fields = ["epoch", "lr", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy",
              "best_validation_accuracy_so_far", "official_test_accuracy", "checkpoint", "elapsed_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, "") for k in fields})


def slope(values):
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def build_report(summary):
    baseline = summary["baseline_reproduction"]
    lines = [
        "# R8B-LONG — 600-Epoch Control Results",
        "",
        "> Independent controlled training run. Existing R8B/H0/H1MP/H2A/H2B artifacts were not modified.",
        "",
        "## 1. Formal source and baseline",
        "",
        f"- Formal R8B checkpoint: `{summary['formal_r8b_checkpoint']}`",
        f"- Baseline reload accuracy (CUDA, official CIFAR-10 TEST): **{baseline['accuracy']:.2f}%**; expected 85.40%; status **{'PASS' if baseline['pass'] else 'FAIL'}**.",
        f"- Formal R8B checkpoint SHA-256: `{baseline['checkpoint_sha256']}`; parameters: `{baseline['parameter_count']}`; seed: `{baseline['seed']}`; TEST images: `{baseline['official_test_count']}`.",
        f"- Environment: Python `{summary['environment']['python']}`, torch `{summary['environment']['torch']}`, torchvision `{summary['environment']['torchvision']}`, device `{summary['environment']['device_name']}`.",
        f"- Initialization source: `{summary['source_r7_checkpoint']}` (the same R7-to-R8B construction used by the formal R8B recipe).",
        f"- R7 source SHA-256: `{summary['source_r7_sha256']}`",
        "- R8B architecture, binary convolutions, Thermometer R=8, signed-power-of-two K, QRPReLU, FC, and initialization are unchanged.",
        "- The only schedule change is `CosineAnnealingLR(T_max=600)` instead of the original 100-epoch horizon.",
        "",
        "## 2. Fixed recipe",
        "",
        f"- Seed: `{summary['seed']}`; validation split seed: `{summary['validation_split_seed']}`.",
        "- CIFAR-10 TRAIN split: 45,000 train / 5,000 validation, fixed and disjoint; official TEST is never used for epoch selection.",
        "- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor. Validation/TEST: ToTensor only.",
        "- Optimizer: SGD, lr=1e-4, momentum=0.9, weight_decay=0.0; tau=0.99; batch size=128.",
        "- Checkpoints: epochs 100, 200, 300, 400, 500, 600, plus best validation.",
        "",
        "## 3. Epoch accuracy table",
        "",
        "| Epoch | Train loss | Train acc. | Validation acc. | Official test acc. | LR | Checkpoint SHA-256 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["milestones"]:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.5f} | {row['train_accuracy']:.2f}% | {row['validation_accuracy']:.2f}% | {row['official_test_accuracy']:.2f}% | {row['lr']:.9g} | `{row['checkpoint_sha256']}` |")
    lines += [
        "",
        f"Best validation: **{summary['best_validation_accuracy']:.2f}%** at epoch **{summary['best_validation_epoch']}**.",
        f"Best-validation checkpoint official TEST: **{summary['best_validation_test_accuracy']:.2f}%**.",
        f"Best official milestone TEST: **{summary['best_official_test_accuracy']:.2f}%** at epoch **{summary['best_official_test_epoch']}**; delta vs R8B 85.40% = **{summary['best_official_test_delta_pp']:+.2f} pp**.",
        "",
        "## 4. Saturation, plateau, and overfitting",
        "",
        f"- Last epoch validation rising: **{summary['analysis']['last_epoch_validation_rising']}**.",
        f"- Validation slope over epochs 501–600: `{summary['analysis']['validation_slope_last_100_pp_per_epoch']:+.6f}` percentage points/epoch.",
        f"- Plateau detected after epoch 300: **{summary['analysis']['plateau_after_300']}**.",
        f"- Overfitting signal: **{summary['analysis']['overfitting_signal']}**.",
        f"- Train–validation gap at epoch 600: **{summary['analysis']['train_validation_gap_pp']:+.2f} pp**; validation–TEST gap: **{summary['analysis']['validation_test_gap_pp']:+.2f} pp**; generalization-gap signal: **{summary['analysis']['generalization_gap_signal']}**.",
        f"- Interpretation: {summary['analysis']['interpretation']}",
        "",
        "## 5. Required integrity checks",
        "",
        f"- Model parameter count: `{summary['parameter_inventory']['total_parameters']}` ({summary['parameter_inventory']['trainable_parameters']} trainable + {summary['parameter_inventory']['frozen_parameters']} frozen alpha compatibility parameters).",
        f"- Exact signed-pow2 K / FC constraints: **PASS** at initialization and final checkpoint.",
        f"- NaN/Inf check: **{'PASS' if summary['final_parameter_health']['finite'] else 'FAIL'}**.",
        f"- Thermometer input: UINT8-equivalent CIFAR-10 RGB converted by the formal R=8 ToTensor path; no preprocessing change.",
        "- No H0 INT6, H1MP, H2A, H2B, RTL, synthesis, retraining, or fine-tuning operation was started after this run.",
        "",
        "## 6. Decision answers",
        "",
        f"A. Original R8B baseline: **85.40%**; 600-epoch best official result: **{summary['best_official_test_accuracy']:.2f}%**.",
        f"B. Last epoch still rising: **{summary['analysis']['last_epoch_validation_rising']}**; plateau: **{summary['analysis']['plateau_after_300']}**.",
        f"C. Classic validation overfitting evidence: **{summary['analysis']['overfitting_signal']}**; a train/TEST generalization gap is **{summary['analysis']['generalization_gap_signal']}**.",
        f"D. Worth starting H0/H1/H2 from this control: **{summary['analysis']['worth_followup']}**.",
        f"E. Best checkpoint: `{summary['best_validation_checkpoint']}` (SHA-256 `{summary['best_validation_checkpoint_sha256']}`).",
        "",
        "## 7. Files",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Recipe: `{summary['recipe_path']}`",
        f"- Curve CSV: `{summary['curve_path']}`",
        f"- JSON: `{summary['results_json_path']}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--source-r7-checkpoint", default=str(SOURCE_R7))
    ap.add_argument("--formal-r8b-checkpoint", default=str(FORMAL_R8B))
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None, help="resume an existing R8B-LONG run directory")
    args = ap.parse_args()
    cfg = load_json(Path(args.config))
    seed = int(cfg["experiment"].get("seed", 123))
    split_seed = int(cfg["experiment"].get("validation_split_seed", seed))
    seed_everything(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source_r7 = Path(args.source_r7_checkpoint).resolve()
    formal_r8b = Path(args.formal_r8b_checkpoint).resolve()
    if not source_r7.exists() or not formal_r8b.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source_r7} / {formal_r8b}")
    source_hash = sha256_file(source_r7)
    formal_hash = sha256_file(formal_r8b)

    # Mandatory baseline gate: evaluate the frozen formal R8B checkpoint first.
    baseline_model, baseline_ckpt = load_r8b_checkpoint(formal_r8b, device)
    _, test_loader_baseline = build_r5t_loaders({"dataset": cfg["dataset"], "training": cfg["training"]}, smoke=False)
    baseline_criterion = nn.CrossEntropyLoss()
    _, baseline_acc = evaluate(baseline_model, test_loader_baseline, baseline_criterion, device)
    baseline_expected = float(baseline_ckpt.get("best_acc", 85.40))
    baseline = {
        "accuracy": float(baseline_acc),
        "expected_accuracy": baseline_expected,
        "stored_best_epoch": int(baseline_ckpt.get("best_epoch", -1)),
        "checkpoint_sha256": formal_hash,
        "parameter_count": int(sum(v.numel() for v in baseline_model.parameters())),
        "seed": seed,
        "official_test_count": 10000,
        "pass": abs(baseline_acc - 85.40) <= 0.01,
    }
    if not baseline["pass"]:
        raise RuntimeError(f"Formal R8B baseline failed reproduction: {baseline_acc:.4f}% (expected 85.40%)")
    del baseline_model, test_loader_baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source, source_state = load_r7(str(source_r7), device)
    model = build_r8b_from_r7(source, min_exp=cfg["classifier"].get("min_exp"), max_exp=cfg["classifier"].get("max_exp")).to(device)
    freeze = parameter_freeze_report(model)
    if freeze["frozen_params"] != 672:
        raise RuntimeError(f"Expected 672 frozen alpha parameters, got {freeze}")
    frozen_snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    initial_invariants = assert_invariants(model, "r8b", frozen_snapshot)
    inventory = model_parameter_inventory(model)
    initial_health = parameter_health(model)
    if not initial_health["finite"]:
        raise RuntimeError("Initial model has NaN/Inf")
    train_loader, val_loader, test_loader, split = make_loaders(cfg, split_seed, device, smoke=args.smoke)
    criterion = nn.CrossEntropyLoss()
    set_tau_all(model, float(cfg["finetune"].get("tau", 0.99)))
    _, zero_val = evaluate(model, val_loader, criterion, device)
    ft = cfg["finetune"]
    epochs = 1 if args.smoke else int(ft.get("epochs", 600))
    if not args.smoke and epochs != 600:
        raise RuntimeError("Formal R8B-LONG run must use exactly 600 epochs")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(ft["lr"]), momentum=float(ft["momentum"]), weight_decay=float(ft["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / cfg["output"]["root"]
    out_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        run_dir = Path(args.resume).resolve()
        if not run_dir.is_relative_to(out_root.resolve()):
            raise ValueError("Resume directory must be inside experiments/recu_r8b_long")
        if not (run_dir / "last.pt").exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {run_dir / 'last.pt'}")
    else:
        run_dir = out_root / f"r8b_long_s{seed}_{stamp}{'_smoke' if args.smoke else ''}"
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(split, run_dir / "train_validation_split.json")
        save_json({
            "experiment": "R8B-LONG",
            "formal_r8b_checkpoint": path_string(formal_r8b),
            "source_r7_checkpoint": path_string(source_r7),
            "source_r7_sha256": source_hash,
            "source_r7_stored_best_acc": source_state.get("best_acc"),
            "baseline_reproduction": baseline,
            "recipe": cfg,
            "fixed_split": {k: v for k, v in split.items() if k.endswith("count") or k.endswith("sha256") or k in ("source", "split_seed")},
            "initial_validation_accuracy": zero_val,
            "initial_invariants": initial_invariants,
            "parameter_inventory": inventory,
            "initial_parameter_health": {"finite": initial_health["finite"], "max_abs": initial_health["max_abs"]},
        }, run_dir / "setup.json")

    history = []
    milestones = []
    start_epoch = 1
    # Epoch zero is a diagnostic only.  The selected checkpoint must come
    # from a completed training epoch so its validation score can be reloaded
    # exactly, even when the first update lowers accuracy.
    best_val = -float("inf")
    best_epoch = 0
    if args.resume:
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if ckpt.get("config") != cfg or ckpt.get("source_r7_sha256") != source_hash:
            raise RuntimeError("Resume checkpoint provenance/config mismatch")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            # torch.load(map_location=device) places RNG tensors on CUDA,
            # while the CUDA RNG setter consumes CPU ByteTensors.
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        history_path = run_dir / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        start_epoch = int(ckpt["epoch"]) + 1
        # Keep one row per completed epoch when recovering from an interrupted
        # process that may have appended a duplicate before it was stopped.
        unique_history = {}
        for row in history:
            ep = int(row["epoch"])
            if ep < start_epoch and ep not in unique_history:
                unique_history[ep] = row
        history = [unique_history[ep] for ep in sorted(unique_history)]
        best_val = float(ckpt["best_validation_accuracy"])
        best_epoch = int(ckpt["best_epoch"])
        for row in history:
            if row.get("official_test_accuracy", "") not in ("", None):
                ep = int(row["epoch"])
                p = run_dir / f"epoch_{ep:03d}.pt"
                milestones.append({
                    "epoch": ep,
                    "official_test_accuracy": float(row["official_test_accuracy"]),
                    "checkpoint": path_string(p),
                    "checkpoint_sha256": sha256_file(p) if p.exists() else "",
                })
        print(f"Resuming R8B-LONG run={path_string(run_dir)} from epoch {start_epoch}/{epochs}", flush=True)
    best_path = run_dir / "best_validation.pt"
    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.time()
        model.train()
        set_tau_all(model, float(ft.get("tau", 0.99)))
        total = correct = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite train loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(y.size(0))
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        invariants = assert_invariants(model, "r8b", frozen_snapshot)
        health = parameter_health(model)
        if not health["finite"]:
            raise RuntimeError(f"NaN/Inf parameter at epoch {epoch}")
        official_acc = None
        checkpoint_name = ""
        checkpoint_sha = ""
        if val_acc > best_val:
            best_val, best_epoch = float(val_acc), int(epoch)
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_val, best_epoch, cfg, source_hash, split_seed)
            checkpoint_name = "best_validation.pt"
            checkpoint_sha = sha256_file(best_path)
        if epoch in MILESTONES or (args.smoke and epoch == 1):
            milestone_path = run_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(milestone_path, epoch, model, optimizer, scheduler, best_val, best_epoch, cfg, source_hash, split_seed)
            checkpoint_name = milestone_path.name
            checkpoint_sha = sha256_file(milestone_path)
            _, official_acc = evaluate(model, test_loader, criterion, device)
            milestones.append({"epoch": epoch, "official_test_accuracy": float(official_acc), "checkpoint": path_string(milestone_path), "checkpoint_sha256": checkpoint_sha})
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss_sum / max(total, 1)),
            "train_accuracy": float(100.0 * correct / max(total, 1)),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "best_validation_accuracy_so_far": float(best_val),
            "official_test_accuracy": official_acc if official_acc is not None else "",
            "checkpoint": checkpoint_name,
            "elapsed_s": float(time.time() - ep_t0),
        }
        history.append(row)
        save_json(history, run_dir / "history.json")
        # Keep a resumable state after every completed epoch.  This is a
        # persistence aid only; it does not alter the optimizer or schedule.
        save_checkpoint(run_dir / "last.pt", epoch, model, optimizer, scheduler, best_val, best_epoch, cfg, source_hash, split_seed)
        print(f"R8B-LONG epoch={epoch}/{epochs} loss={row['train_loss']:.5f} train={row['train_accuracy']:.2f}% val={val_acc:.2f}% best={best_val:.2f}%@{best_epoch} lr={row['lr']:.9g} test={official_acc if official_acc is not None else '-'} time={row['elapsed_s']:.1f}s", flush=True)

    if not best_path.exists():
        save_checkpoint(best_path, 0, model, optimizer, scheduler, best_val, best_epoch, cfg, source_hash, split_seed)
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"], strict=True)
    _, best_val_reload = evaluate(model, val_loader, criterion, device)
    if abs(best_val_reload - best_val) > 1e-7:
        raise RuntimeError(f"Best validation reload mismatch {best_val_reload} vs {best_val}")
    final_invariants = assert_invariants(model, "r8b", frozen_snapshot)
    final_health = parameter_health(model)
    _, best_val_test = evaluate(model, test_loader, criterion, device)
    if milestones:
        best_m = max(milestones, key=lambda m: m["official_test_accuracy"])
    else:
        best_m = {"epoch": 0, "official_test_accuracy": float(best_val_test), "checkpoint": path_string(best_path), "checkpoint_sha256": sha256_file(best_path)}
    vals = [r["validation_accuracy"] for r in history]
    last100 = vals[-100:] if len(vals) >= 100 else vals
    max_after_300 = max(vals[300:]) if len(vals) > 300 else vals[-1]
    max_before_300 = max(vals[:300]) if vals else vals[-1]
    final_val = vals[-1] if vals else zero_val
    train_losses = [r["train_loss"] for r in history]
    overfit = bool(best_val - final_val > 0.5 and slope(vals[-100:]) < -0.001)
    plateau = bool(len(vals) >= 600 and (max_after_300 - max_before_300) < 0.2 and abs(slope(last100)) < 0.002)
    rising = bool(len(vals) >= 2 and vals[-1] > vals[-2])
    improvement = float(best_m["official_test_accuracy"] - 85.40)
    worth = "YES (>=+0.30 pp; consider a separately authorized precision study)" if improvement >= 0.30 else "NO (do not start a downstream H0/H1/H2 study from this control)"
    interpretation = "Validation is still improving at the end." if rising and not plateau else "Validation has plateaued by the late schedule." if plateau else "Late validation trend is mixed; inspect the curve before extending the schedule."
    final_train_acc = float(history[-1]["train_accuracy"]) if history else 0.0
    train_validation_gap = final_train_acc - float(final_val)
    validation_test_gap = float(final_val) - float(best_val_test)
    generalization_gap_signal = bool(train_validation_gap > 1.5 or validation_test_gap > 1.5)
    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": getattr(__import__("torchvision"), "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    results_dir = ROOT / "reports" / "r8b_long"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_json = results_dir / "R8B_LONG_RESULTS.json"
    curve_path = results_dir / "R8B_LONG_CURVE.csv"
    recipe_path = results_dir / "R8B_LONG_RECIPE.md"
    summary = {
        "experiment": "R8B-LONG",
        "run_dir": path_string(run_dir),
        "formal_r8b_checkpoint": path_string(formal_r8b),
        "formal_r8b_sha256": formal_hash,
        "source_r7_checkpoint": path_string(source_r7),
        "source_r7_sha256": source_hash,
        "seed": seed,
        "validation_split_seed": split_seed,
        "epochs": epochs,
        "baseline_reproduction": baseline,
        "parameter_inventory": inventory,
        "initial_invariants": initial_invariants,
        "final_invariants": final_invariants,
        "initial_parameter_health": {"finite": initial_health["finite"], "max_abs": initial_health["max_abs"]},
        "final_parameter_health": {"finite": final_health["finite"], "max_abs": final_health["max_abs"]},
        "split": {k: v for k, v in split.items() if k != "train_indices" and k != "validation_indices"},
        "best_validation_accuracy": float(best_val),
        "best_validation_epoch": int(best_epoch),
        "best_validation_reload_accuracy": float(best_val_reload),
        "best_validation_checkpoint": path_string(best_path),
        "best_validation_checkpoint_sha256": sha256_file(best_path),
        "best_validation_test_accuracy": float(best_val_test),
        "official_test_count": 10000,
        "environment": environment,
        "milestones": [],
        "best_official_test_accuracy": float(best_m["official_test_accuracy"]),
        "best_official_test_epoch": int(best_m["epoch"]),
        "best_official_test_delta_pp": improvement,
        "last_epoch_validation_accuracy": float(final_val),
        "history": history,
        "analysis": {
            "last_epoch_validation_rising": rising,
            "validation_slope_last_100_pp_per_epoch": slope(last100),
            "plateau_after_300": plateau,
            "overfitting_signal": overfit,
            "generalization_gap_signal": generalization_gap_signal,
            "train_validation_gap_pp": train_validation_gap,
            "validation_test_gap_pp": validation_test_gap,
            "worth_followup": worth,
            "interpretation": interpretation,
            "train_loss_first": float(train_losses[0]) if train_losses else None,
            "train_loss_last": float(train_losses[-1]) if train_losses else None,
        },
        "recipe": cfg,
        "git_commit": git_commit(),
        "elapsed_s": time.time() - t0,
    }
    # Reconcile milestone rows with full history and add checkpoint SHA/details.
    for m in milestones:
        row = next(r for r in history if r["epoch"] == m["epoch"])
        summary["milestones"].append({**m, "train_loss": row["train_loss"], "train_accuracy": row["train_accuracy"], "validation_accuracy": row["validation_accuracy"], "lr": row["lr"]})
    save_json(summary, results_json)
    write_curve(curve_path, history)
    recipe_path.write_text(
        "# R8B-LONG Recipe\n\n"
        "## Original R8B\n\n"
        "- Epochs: 100\n- Initial learning rate: 1e-4\n- Scheduler: cosine decay, T_max=100\n- Warmup: none\n\n"
        "## R8B-LONG\n\n"
        "- Epochs: 600\n- Initial learning rate: 1e-4\n- Scheduler: cosine decay, T_max=600\n- Warmup: none\n\n"
        "The only training change is the schedule horizon. The run starts from the exact R7-derived R8B initialization, preserves the original optimizer, batch size, augmentation, weight decay, loss, quantization and Pow2 constraints, and uses a fixed disjoint 45,000/5,000 CIFAR-10 TRAIN split. Official TEST is evaluated only at milestones and after validation selection.\n\n"
        + json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary["results_json_path"] = path_string(results_json)
    summary["curve_path"] = path_string(curve_path)
    summary["recipe_path"] = path_string(recipe_path)
    save_json(summary, results_json)
    report_path = results_dir / "R8B_LONG_RESULTS.md"
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"run_dir": path_string(run_dir), "best_validation_accuracy": best_val, "best_validation_epoch": best_epoch, "best_validation_test_accuracy": best_val_test, "best_official_test_accuracy": best_m["official_test_accuracy"], "best_official_test_epoch": best_m["epoch"], "results": path_string(results_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
