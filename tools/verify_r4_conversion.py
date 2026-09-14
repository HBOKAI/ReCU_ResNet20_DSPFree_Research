import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.model import ReCUResNet20
from recu_hw.r4 import build_r4_from_r2, r4_exponent_stats, r4_k_quantization_stats
from recu_hw.fused_affine import fold_recu_alpha_bn


@torch.no_grad()
def branch_float_fold_error(r2, device):
    max_abs = 0.0
    pairs = 0

    for layer in [r2.layer1, r2.layer2, r2.layer3]:
        for block in layer:
            for conv, bn in [(block.conv1, block.bn1), (block.conv2, block.bn2)]:
                c = conv.out_channels
                s = torch.randn(4, c, 8, 8, device=device)

                alpha = conv.alpha.view(1, -1, 1, 1)
                ref = bn(s * alpha)

                k, b = fold_recu_alpha_bn(conv, bn)
                folded = s * k.view(1, -1, 1, 1) + b.view(1, -1, 1, 1)

                max_abs = max(max_abs, float((ref - folded).abs().max()))
                pairs += 1

    return {"pairs": pairs, "max_abs_error": max_abs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-r2-checkpoint", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(args.source_r2_checkpoint, map_location=device)

    r2 = ReCUResNet20(
        num_classes=10,
        activation_mode="qrprelu",
        alpha_mode="float",
    ).to(device)
    r2.load_state_dict(state["model"], strict=True)
    r2.eval()

    fold = branch_float_fold_error(r2, device)
    r4 = build_r4_from_r2(r2).to(device).eval()

    report = {
        "source_best_acc": state.get("best_acc"),
        "source_best_epoch": state.get("best_epoch"),
        "float_fold_equivalence": fold,
        "initial_exponents": r4_exponent_stats(r4),
        "initial_k_quantization": r4_k_quantization_stats(r4),
        "r4_trainable_params": sum(p.numel() for p in r4.parameters() if p.requires_grad),
    }
    print(json.dumps(report, indent=2))

    if fold["max_abs_error"] > 1e-4:
        raise SystemExit("FAIL: float alpha+BN folding error > 1e-4")


if __name__ == "__main__":
    main()
