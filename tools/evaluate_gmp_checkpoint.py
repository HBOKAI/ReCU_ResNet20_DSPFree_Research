"""Read-only official-test reload of a preselected GMP checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.gmp_ablation import GMPResNet20
from train_recu_r8 import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    args = ap.parse_args()
    path = args.checkpoint.resolve()
    if not path.is_relative_to((ROOT / "experiments/gmp_ablation").resolve()):
        raise ValueError("Only isolated GMP checkpoints may be evaluated")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = GMPResNet20(
        num_classes=10, resolution=8,
        min_exp=cfg["classifier"].get("min_exp"),
        max_exp=cfg["classifier"].get("max_exp"),
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    test_dataset = datasets.CIFAR10(
        str(ROOT / "data"), train=False, transform=transforms.ToTensor(), download=False,
    )
    if len(test_dataset) != 10000:
        raise RuntimeError("Incomplete official CIFAR-10 TEST")
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    _, accuracy = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    print(json.dumps({
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": state["epoch"],
        "validation_best_in_training": state["best_validation_acc"],
        "official_test_count": len(test_dataset),
        "official_test_accuracy": accuracy,
    }, indent=2))


if __name__ == "__main__":
    main()
