import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.model import build_recu_resnet20, parameter_breakdown
from recu_hw.utils import load_json


def theoretical_ops():
    # CIFAR ResNet-20:
    first_conv = 32 * 32 * 16 * (3 * 3 * 3)
    binary_terms = 40_108_032
    final_fc = 64 * 10

    # Each binary conv output element has one alpha scaling in official ReCU.
    alpha_scale_elems = (
        6 * 32 * 32 * 16
        + 6 * 16 * 16 * 32
        + 6 * 8 * 8 * 64
    )

    # One PReLU per residual block in official ReCU.
    prelu_elems = (
        3 * 32 * 32 * 16
        + 3 * 16 * 16 * 32
        + 3 * 8 * 8 * 64
    )

    return {
        "first_conv_multibit_MACs": first_conv,
        "binary_conv_terms": binary_terms,
        "final_fc_multibit_MACs": final_fc,
        "conventional_MAC_equivalent_total": first_conv + binary_terms + final_fc,
        "official_alpha_scale_elements_per_image": alpha_scale_elems,
        "official_prelu_elements_per_image": prelu_elems,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_json(args.config)
    model = build_recu_resnet20(cfg)

    print("PARAMETERS")
    print(json.dumps(parameter_breakdown(model), indent=2))
    print("OPS")
    print(json.dumps(theoretical_ops(), indent=2))


if __name__ == "__main__":
    main()
