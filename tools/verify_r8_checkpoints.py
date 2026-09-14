import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.r6 import is_power_of_two_tensor, parameter_freeze_report, stem_binary_weight_stats
from recu_hw.r7 import R7ResNet20
from recu_hw.r8 import binary_fc_stats, build_r8a_from_r7, build_r8b_from_r7


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += int((logits.argmax(1) == y).sum())
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["r8a", "r8b"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source-r7-checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    source_state = torch.load(args.source_r7_checkpoint, map_location=device)
    source = R7ResNet20(num_classes=10, resolution=8).to(device)
    source.load_state_dict(source_state["model"], strict=True)
    source.conv1.set_binary_weight(True)
    source.stem_affine.quantize_k = True
    source.head_affine.quantize_k = True
    for module in source.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    if args.variant == "r8a":
        model = build_r8a_from_r7(source).to(device)
    else:
        model = build_r8b_from_r7(source, min_exp=None, max_exp=None).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.conv1.set_binary_weight(True)
    model.stem_affine.quantize_k = True
    model.head_affine.quantize_k = True

    freeze = parameter_freeze_report(model)
    if freeze["frozen_params"] != 672:
        raise RuntimeError(freeze)
    if model.linear.bias is None or hasattr(model, "bn2"):
        raise RuntimeError("FC bias/BN2 invariant failed")
    stem = stem_binary_weight_stats(model)
    if not stem["is_strict_binary"] or not stem["conv_bias_is_none"]:
        raise RuntimeError(stem)
    if not is_power_of_two_tensor(model.stem_affine.effective_k()):
        raise RuntimeError("stem K is not signed-pow2")
    if not is_power_of_two_tensor(model.head_affine.effective_k()):
        raise RuntimeError("head K is not signed-pow2")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("non-finite model parameter")

    if args.variant == "r8a":
        classifier = binary_fc_stats(model)
        if not classifier["is_strict_binary"] or classifier["weights"] != 640:
            raise RuntimeError(classifier)
    else:
        classifier = model.linear.stats()
        if not classifier["is_exact_signed_pow2"] or classifier["weights"] != 640:
            raise RuntimeError(classifier)

    _, test_loader = build_r5t_loaders(
        {"dataset": {"root": "./data", "download": True},
         "training": {"batch_size": 128, "test_batch_size": 128, "num_workers": 8}},
        smoke=False,
    )
    criterion = nn.CrossEntropyLoss()
    source_loss, source_acc = evaluate(source, test_loader, criterion, device)
    reload_loss, reload_acc = evaluate(model, test_loader, criterion, device)
    stored = float(checkpoint["best_acc"])
    if abs(reload_acc - stored) > 1e-6:
        raise RuntimeError(f"reload accuracy mismatch: {reload_acc} vs {stored}")
    result = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "checkpoint_best_acc": stored,
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "source_reload_acc": source_acc,
        "source_stored_best_acc": float(source_state["best_acc"]),
        "reload_acc": reload_acc,
        "reload_loss": reload_loss,
        "delta_reload_vs_source_pp": reload_acc - source_acc,
        "classifier": classifier,
        "stem": stem,
        "head_pow2": True,
        "bn2_removed": not hasattr(model, "bn2"),
        "fc_bias_present": model.linear.bias is not None,
        "freeze_report": freeze,
        "finite": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
