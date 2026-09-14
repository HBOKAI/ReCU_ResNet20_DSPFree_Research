"""Paired official-test GAP vs GMP replay using the frozen H2A-v2 datapath."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recu_hw.gmp_ablation import GMPTraceAdapter
from recu_hw.h2_workspace import build_h2_trace_adapter, load_r8b_h1mp_model
from recu_hw.layers import iter_recu_binary_convs


SOURCE = ROOT / "experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt"
H0 = ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json"
H1MP = ROOT / "H1MP_SEARCH_RESULTS.json"
H2A = ROOT / "H2_FINITE_WIDTH_RESULTS.json"
OUTPUT = ROOT / "experiments/gmp_ablation/zero_shot.json"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def binary_terms(model):
    counts = {}
    handles = []
    modules = [("stem", model.conv1)] + [
        (f"backbone.{index}", module) for index, module in enumerate(iter_recu_binary_convs(model))
    ]
    for name, module in modules:
        def hook(mod, _input, output, label=name):
            counts[label] = int(output.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1])
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model.eval()(torch.zeros((1, 3, 32, 32), device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    assert len(counts) == 19
    return {"stem": counts["stem"], "backbone": sum(v for k, v in counts.items() if k != "stem"), "total": sum(counts.values())}


class Running:
    def __init__(self):
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.sum = 0.0
        self.sum_sq = 0.0

    def add(self, tensor):
        x = tensor.detach().to(torch.float64)
        self.count += int(x.numel())
        self.min = min(self.min, float(x.min().item()))
        self.max = max(self.max, float(x.max().item()))
        self.sum += float(x.sum().item())
        self.sum_sq += float((x * x).sum().item())

    def as_dict(self):
        mean = self.sum / self.count
        return {"count": self.count, "min": self.min, "max": self.max,
                "mean": mean, "std": max(self.sum_sq / self.count - mean * mean, 0.0) ** 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h2_result = json.loads(H2A.read_text(encoding="utf-8"))
    if h2_result.get("status") != "H2A_V2_PASS":
        raise RuntimeError("Expected frozen H2A-v2 PASS source")
    plan = h2_result["h2a_plan"]
    if plan["gap"].get("division_mode") != "deferred_scale_metadata":
        raise RuntimeError("Frozen GAP source lost deferred scaling")
    model = load_r8b_h1mp_model(str(SOURCE), str(H0), str(H1MP), device)
    before = state_sha256(model)
    complexity = {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "binary_conv_terms_per_image": binary_terms(model),
    }
    if complexity["trainable_parameters"] != 284250 or complexity["binary_conv_terms_per_image"]["total"] != 54263808:
        raise RuntimeError(f"R8B complexity drift: {complexity}")
    gap = build_h2_trace_adapter(model)
    gap.set_h2_plan(copy.deepcopy(plan))
    gmp = GMPTraceAdapter(model, model._h1_fake_quant, model._h0_info)
    gmp.set_h2_plan(copy.deepcopy(plan))
    input_bits = model._h1_fake_quant.bits_for("layer3.2.qrprelu_output")
    input_shift = model._h1_fake_quant.shift_map["layer3.2.qrprelu_output"]

    dataset = datasets.CIFAR10(str(ROOT / "data"), train=False, transform=transforms.ToTensor(), download=False)
    if args.smoke:
        dataset = torch.utils.data.Subset(dataset, range(128))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    correct_gap = correct_gmp = total = mismatch = 0
    logit_abs = Running()
    gap_head_input = Running()
    gmp_head_input = Running()
    stage3_q_stats = Running()
    gmp_q_stats = Running()
    hist = torch.zeros(1 << input_bits, dtype=torch.int64)
    with torch.no_grad():
        model.eval()
        for batch_no, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits_gap = gap(x)
            logits_gmp = gmp(x)
            pred_gap, pred_gmp = logits_gap.argmax(1), logits_gmp.argmax(1)
            correct_gap += int((pred_gap == y).sum().item())
            correct_gmp += int((pred_gmp == y).sum().item())
            mismatch += int((pred_gap != pred_gmp).sum().item())
            total += int(y.numel())
            logit_abs.add((logits_gmp - logits_gap).abs())
            gap_head_input.add(gap.last_gap_sum.to(torch.float64) * (2.0 ** -(gap.last_gap_input_shift + 6)))
            gmp_head_input.add(gmp.last_gmp_q.to(torch.float64) * (2.0 ** -gmp.last_gmp_shift))
            stage3_q_stats.add(gmp.last_stage3_q)
            gmp_q_stats.add(gmp.last_gmp_q)
            hist += torch.bincount((gmp.last_gmp_q.cpu().flatten() + (1 << (input_bits - 1))), minlength=1 << input_bits)
            if batch_no % 10 == 0 or batch_no == len(loader):
                print(f"phaseA batch={batch_no}/{len(loader)} GAP={100*correct_gap/total:.2f}% GMP={100*correct_gmp/total:.2f}%", flush=True)

    if not args.smoke and (total != 10000 or 100.0 * correct_gap / total != 85.03):
        raise RuntimeError(f"Official frozen H2A-v2 GAP reload mismatch: {100.0 * correct_gap / total:.4f}%")
    after = state_sha256(model)
    if before != after:
        raise RuntimeError("Source model state changed during zero-shot evaluation")
    gmp_qmin = -(1 << (input_bits - 1))
    nonzero_bins = [{"q": q + gmp_qmin, "count": int(value)} for q, value in enumerate(hist.tolist()) if value]
    result = {
        "phase": "A_zero_shot_direct_replacement",
        "official_test": not args.smoke,
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": sha256(SOURCE),
        "h0_results_sha256": sha256(H0),
        "h1mp_results_sha256": sha256(H1MP),
        "h2a_v2_results_sha256": sha256(H2A),
        "source_model_state_sha256_before": before,
        "source_model_state_sha256_after": after,
        "same_weights": before == after,
        "device": str(device),
        "samples": total,
        "gap_accuracy": 100.0 * correct_gap / total,
        "gmp_zero_shot_accuracy": 100.0 * correct_gmp / total,
        "delta_vs_gap_pp": 100.0 * (correct_gmp - correct_gap) / total,
        "prediction_mismatch_count": mismatch,
        "absolute_logit_difference": logit_abs.as_dict(),
        "stage3_input_q": {**stage3_q_stats.as_dict(), "bits": input_bits, "shift": input_shift},
        "gap_head_input_real": gap_head_input.as_dict(),
        "gmp_head_input_real": gmp_head_input.as_dict(),
        "gmp_output_q": {**gmp_q_stats.as_dict(), "bits": input_bits, "shift": input_shift,
                         "histogram_nonzero_bins": nonzero_bins},
        "gmp_max_register_bits": input_bits,
        "gmp_saturation_count": int(gmp.runtime_saturation.get("gmp.max", 0)),
        "gap_h2_overflow_total": sum(gap.runtime_overflow.values()),
        "gmp_h2_overflow_total": sum(gmp.runtime_overflow.values()),
        "complexity": complexity,
        "note": "GMP zero-shot is direct replacement in the GAP-trained H2A-v2 model, not a trained GMP architecture result.",
    }
    if not args.smoke:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "phase", "samples", "gap_accuracy", "gmp_zero_shot_accuracy", "delta_vs_gap_pp",
        "prediction_mismatch_count", "gmp_max_register_bits", "gmp_saturation_count", "gmp_h2_overflow_total"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
