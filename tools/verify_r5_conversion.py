import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.data import build_recu_loaders
from recu_hw.r4 import R4ResNet20
from recu_hw.r5 import build_r5_from_r4, stem_weight_stats


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-r4-checkpoint", required=True)
    ap.add_argument("--input-scale-exp", type=int, default=-5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    state = torch.load(
        args.source_r4_checkpoint,
        map_location=device,
    )

    r4 = R4ResNet20(num_classes=10).to(device)
    r4.load_state_dict(state["model"], strict=True)
    r4.eval()

    r5 = build_r5_from_r4(
        r4,
        input_scale_exp=args.input_scale_exp,
    ).to(device)
    r5.eval()

    # Verify actual forward binary weights are +/-1.
    bw = r5.conv1.binary_weight()
    unique = sorted(set(float(x) for x in torch.unique(bw).cpu().tolist()))
    if unique != [-1.0, 1.0] and unique != [1.0] and unique != [-1.0]:
        raise SystemExit(f"FAIL: unexpected binary stem weights: {unique}")

    cfg = {
        "dataset": {
            "name": "cifar10",
            "root": "./data",
            "download": True,
            "protocol": "official_recu_train_test",
        },
        "training": {
            "batch_size": 256,
            "test_batch_size": 128,
            "num_workers": 0,
            "windows_safe_smoke": True,
        },
    }
    train_loader, _ = build_recu_loaders(cfg, smoke=False)

    totals = {
        "total": 0,
        "sat_low": 0,
        "sat_high": 0,
        "input_min": float("inf"),
        "input_max": float("-inf"),
        "q_min": 999999,
        "q_max": -999999,
    }

    for idx, (x, _) in enumerate(train_loader):
        if idx >= 16:
            break
        x = x.to(device)
        s = r5.input_quant.stats(x)
        totals["total"] += s["total"]
        totals["sat_low"] += s["sat_low"]
        totals["sat_high"] += s["sat_high"]
        totals["input_min"] = min(totals["input_min"], s["input_min"])
        totals["input_max"] = max(totals["input_max"], s["input_max"])
        totals["q_min"] = min(totals["q_min"], s["q_observed_min"])
        totals["q_max"] = max(totals["q_max"], s["q_observed_max"])

    totals["saturation_fraction"] = (
        totals["sat_low"] + totals["sat_high"]
    ) / max(totals["total"], 1)
    totals["scale_exp"] = args.input_scale_exp
    totals["scale"] = 2.0 ** args.input_scale_exp

    report = {
        "source_r4_best_acc": state.get("best_acc"),
        "source_r4_best_epoch": state.get("best_epoch"),
        "binary_weight_unique_values": unique,
        "stem_weight_stats": stem_weight_stats(r5),
        "input_int8_quantization": totals,
        "hardware_product": "w in {-1,+1}, q in signed INT8 -> select +q/-q then accumulate",
    }

    print(json.dumps(report, indent=2))

    if totals["saturation_fraction"] > 1e-4:
        raise SystemExit(
            "FAIL: INT8 input saturation fraction > 1e-4. "
            "Reconsider power-of-two input scale exponent."
        )


if __name__ == "__main__":
    main()
