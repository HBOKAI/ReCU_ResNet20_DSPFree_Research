import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from recu_hw.r5t_data import build_r5t_loaders
from recu_hw.r7 import build_r7_from_r6
from recu_hw.r6 import R6ResNet20
from recu_hw.utils import seed_everything


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    good = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        good += int((model(x).argmax(1) == y).sum())
        total += y.numel()
    return 100.0 * good / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    seed_everything(123)
    device = torch.device(args.device)
    state = torch.load(args.source, map_location=device)
    source = R6ResNet20(num_classes=10, resolution=8).to(device)
    source.load_state_dict(state["model"], strict=True)
    source.conv1.set_binary_weight(True)
    source.stem_affine.quantize_k = True
    for module in source.modules():
        if hasattr(module, "alpha"):
            module.alpha.requires_grad_(False)
    r7 = build_r7_from_r6(source, resolution=8).to(device)
    loaders = build_r5t_loaders(
        {"dataset": {"root": "./data", "download": True},
         "training": {"batch_size": 128, "test_batch_size": 128, "num_workers": 8}},
        smoke=False,
    )
    train_loader, test_loader = loaders
    print(f"zero_eval={accuracy(r7, test_loader, device):.4f}")
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    criterion = nn.CrossEntropyLoss()
    r7.train()
    logits = r7(x)
    loss = criterion(logits, y)
    loss.backward()
    watch = {
        "layer1.0.conv1.weight",
        "layer1.0.aff1.log2_abs_k",
        "layer3.2.conv2.weight",
        "head_affine.k_latent",
        "head_affine.bias",
        "linear.weight",
        "linear.bias",
    }
    print("first_batch_loss=%.6f first_batch_acc=%.4f" %
          (float(loss), 100.0 * float((logits.argmax(1) == y).float().mean())))
    for name, p in r7.named_parameters():
        if name in watch:
            gn = 0.0 if p.grad is None else float(p.grad.detach().norm())
            print("grad", name, "param_norm=%.6g grad_norm=%.6g" %
                  (float(p.detach().norm()), gn))

    base = copy.deepcopy(r7.state_dict())
    cases = [("all_lr1e-3", 1e-3, "all"), ("all_lr1e-4", 1e-4, "all"),
             ("all_lr1e-5", 1e-5, "all"), ("head_fc_lr1e-3", 1e-3, "head_fc")]
    for label, lr, scope in cases:
        m = build_r7_from_r6(source, resolution=8).to(device)
        m.load_state_dict(base, strict=True)
        if scope == "head_fc":
            for name, p in m.named_parameters():
                p.requires_grad_(name.startswith("head_affine.") or name.startswith("linear."))
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                              lr=lr, momentum=0.9, weight_decay=0.0)
        m.train()
        opt.zero_grad(set_to_none=True)
        out = m(x)
        l = criterion(out, y)
        l.backward()
        opt.step()
        print("after_one_step", label, "loss=%.6f eval=%.4f" %
              (float(l), accuracy(m, test_loader, device)))


if __name__ == "__main__":
    main()
