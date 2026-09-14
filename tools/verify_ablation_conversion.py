import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.ablation import (
    build_r1_from_official,
    build_r2_from_official,
    build_r3_from_official,
    fused_exponent_stats,
    verify_float_fold_equivalence,
)
from recu_hw.model import ReCUResNet20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    state = torch.load(args.checkpoint, map_location=device)
    official = ReCUResNet20(
        activation_mode="prelu",
        alpha_mode="float",
    ).to(device)
    official.load_state_dict(state["model"], strict=True)
    official.eval()

    r1 = build_r1_from_official(official)
    r2, nbad = build_r2_from_official(official)
    r3 = build_r3_from_official(official, affine_mode="pow2")

    fold = verify_float_fold_equivalence(
        official, device=device, batches=3, batch_size=4
    )

    report = {
        "source_best_acc": state.get("best_acc"),
        "source_best_epoch": state.get("best_epoch"),
        "r1_trainable_params": sum(p.numel() for p in r1.parameters() if p.requires_grad),
        "r2_trainable_params": sum(p.numel() for p in r2.parameters() if p.requires_grad),
        "r2_nonpositive_prelu_slopes_clamped": nbad,
        "r3_trainable_params": sum(p.numel() for p in r3.parameters() if p.requires_grad),
        "r3_initial_exponents": fused_exponent_stats(r3),
        "float_fold_equivalence": fold,
    }

    print(json.dumps(report, indent=2))

    # Folding must be numerically equivalent before power-of-two quantization.
    if fold["max_abs_logit_error"] > 1e-4:
        raise SystemExit(
            "FAIL: alpha+BN float folding is not sufficiently equivalent."
        )


if __name__ == "__main__":
    main()
