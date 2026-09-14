import argparse
import torch
import torch.nn as nn

from recu_hw.data import build_recu_loaders
from recu_hw.model import build_recu_resnet20
from recu_hw.utils import load_json


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_recu_resnet20(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    _, test_loader = build_recu_loaders(cfg, smoke=False)
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    print(
        f"test_loss={loss_sum/max(total,1):.4f} "
        f"test_acc={100.0*correct/max(total,1):.2f}%"
    )


if __name__ == "__main__":
    main()
