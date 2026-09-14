import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.r4 import R4ResNet20
from recu_hw.r5t import ThermometerEncoder, build_r5t_from_r4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-r4-checkpoint", required=True)
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(args.source_r4_checkpoint, map_location=device)
    r4 = R4ResNet20(num_classes=10).to(device)
    r4.load_state_dict(state["model"], strict=True)

    model = build_r5t_from_r4(r4, resolution=args.resolution, binary_stem=False).to(device)
    enc = ThermometerEncoder(args.resolution)
    p109 = enc.encode_uint8_reference(109)

    model.conv1.set_binary_weight(True)
    ew = model.conv1.effective_weight()
    unique = sorted(set(float(v) for v in torch.unique(ew).cpu().tolist()))

    report = {
        "source_r4_best_acc": state.get("best_acc"),
        "source_r4_best_epoch": state.get("best_epoch"),
        "resolution": args.resolution,
        "length_per_rgb": enc.length,
        "input_channels": 3 * enc.length,
        "r32_pixel109_ones_reference": int(ThermometerEncoder(32).encode_uint8_reference(109).sum()),
        "binary_weight_unique_values": unique,
        "binary_stem_weight_count": int(model.conv1.weight.numel()),
        "binary_stem_weight_bits": int(model.conv1.weight.numel()),
        "original_fp32_stem_storage_bits": int(r4.conv1.weight.numel() * 32),
    }
    print(json.dumps(report, indent=2))

    if args.resolution == 8:
        assert enc.length == 32
        assert 3 * enc.length == 96
    if any(v not in (-1.0, 1.0) for v in unique):
        raise SystemExit("FAIL: binary stem effective weight is not strictly +/-1")


if __name__ == "__main__":
    main()
