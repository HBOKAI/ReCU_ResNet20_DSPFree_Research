"""Export the frozen checkpoint and H0/H1MP/H2A-v2 plans to NPZ + JSON.

PyTorch is used here only to load and transform checkpoint tensors. The
runtime reference_inference.py imports neither torch nor model modules.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt"
H0 = ROOT / "H0_INTEGER_BIAS_SWEEP_RESULTS.json"
H1 = ROOT / "H1MP_SEARCH_RESULTS.json"
H2 = ROOT / "H2_FINITE_WIDTH_RESULTS.json"


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_conv(weight, tau):
    # recu_hw/layers.py and H2TraceAdapter._binary_conv, eval path.
    w0 = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
    w1 = w0 / (torch.sqrt(w0.var(dim=(1, 2, 3), keepdim=True) + 1e-5) / 2.0 / math.sqrt(2.0))
    ew = torch.mean(torch.abs(w1))
    q_tau = (-ew * torch.log(2.0 - 2.0 * tau.clamp(max=0.999999))).detach()
    return torch.sign(torch.clamp(w1, -q_tau, q_tau)).to(torch.int8)


def main():
    outdir = HERE / "params_export"
    outdir.mkdir(exist_ok=True)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    h0 = json.loads(H0.read_text(encoding="utf-8"))
    h1 = json.loads(H1.read_text(encoding="utf-8"))
    h2 = json.loads(H2.read_text(encoding="utf-8"))
    assert h2["status"] == "H2A_V2_PASS" and h2["h2a_plan"]["gap_deferred_scaling"]
    assert h1["status"] == "COMPLETE"
    h0_layers = next(item["layers"] for item in h0["results"] if item["bits"] == 6)
    assert len(h0_layers) == 21 and len(h1["final_plan"]["bits_by_node"]) == 58
    arrays = {}
    manifest = []

    def put(name, value, source, meaning):
        data = np.asarray(value)
        arrays[name] = data
        manifest.append({"name": name, "original_state_dict_key": source,
                         "shape": list(data.shape), "dtype": str(data.dtype),
                         "export_file": "params_export/inference_params.npz", "hardware_meaning": meaning})

    put("conv1.weight_sign", torch.sign(state["conv1.weight"]).to(torch.int8).numpy(),
        "conv1.weight", "Stem binary kernel: -1/0/+1; 0 bit means -1, 1 bit means +1 when active")
    block_names = [f"layer{stage}.{index}" for stage in (1, 2, 3) for index in range(3)]
    for prefix in block_names:
        for sub in ("conv1", "conv2"):
            name = f"{prefix}.{sub}"
            put(name + ".weight_sign", _effective_conv(state[name + ".weight"], state[name + ".tau"]).numpy(),
                name + ".weight, " + name + ".tau", "Eval-time ReCU clipped centered weight sign; alpha_mode=one")
    affine_names = ["stem_affine"] + [f"{p}.aff{i}" for p in block_names for i in (1, 2)] + ["head_affine"]
    for name in affine_names:
        if name in ("stem_affine", "head_affine"):
            latent = state[name + ".k_latent"]
            exponent = torch.round(torch.log2(latent.abs().clamp_min(torch.finfo(latent.dtype).tiny))).to(torch.int16)
            sign = torch.where(latent >= 0, 1, -1).to(torch.int8)
            k_source = name + ".k_latent"
        else:
            exponent = torch.clamp(torch.round(state[name + ".log2_abs_k"]), -16, 8).to(torch.int16)
            sign = state[name + ".sign_k"].to(torch.int8)
            k_source = name + ".log2_abs_k, " + name + ".sign_k"
        put(name + ".sign", sign.numpy(), k_source, "Signed-pow2 K sign")
        put(name + ".exponent", exponent.numpy(), k_source, "Signed-pow2 K power-of-two exponent")
        b = h0_layers[name]
        assert b["bits"] == 6 and len(b["q_values"]) == len(sign)
        put(name + ".bias_q", np.asarray(b["q_values"], dtype=np.int8),
            name + ".bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json", "INT6 bias at per-module scale in model_config.json")
    for prefix in block_names:
        node = prefix + ".qrprelu_output"
        record = h2["h2a_plan"]["qrprelu"][node]
        for key in ("a", "xi1", "xi2"):
            put(prefix + ".post_act." + key + "_raw", state[prefix + ".post_act." + key].numpy(),
                prefix + ".post_act." + key, "Checkpoint QRP parameter before H2 finite-width integer interpretation")
        exponent = torch.round(state[prefix + ".post_act.a"]).to(torch.int16)
        assert exponent.tolist() == record["slope_exponents"]
        put(node + ".slope_exponent", exponent.numpy(), prefix + ".post_act.a", "Negative-branch pow2 exponent")
        for key in ("xi1", "xi2"):
            quant = torch.round(state[prefix + ".post_act." + key].to(torch.float64) * (2.0 ** record["param_shift"]))
            put(node + "." + key + "_q", quant.to(torch.int64).numpy(),
                prefix + ".post_act." + key, "QRP offset integer at H2 param_shift")
    fc_weight = state["linear.weight"]
    fc_exp = torch.round(torch.log2(fc_weight.abs().clamp_min(torch.finfo(fc_weight.dtype).tiny)))
    fc_sign = torch.where(fc_weight >= 0, 1, -1)
    put("linear.weight_sign", fc_sign.to(torch.int8).numpy(), "linear.weight", "64-to-10 signed-pow2 FC term sign")
    put("linear.weight_exponent", fc_exp.to(torch.int16).numpy(), "linear.weight", "64-to-10 signed-pow2 FC term exponent")
    put("linear.bias_q", np.asarray(h0_layers["linear"]["q_values"], dtype=np.int8),
        "linear.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json", "INT6 FC bias")
    assert fc_exp.to(torch.int64).tolist() == h2["h2a_plan"]["fc"]["weight_exponents"]
    assert fc_sign.to(torch.int64).tolist() == h2["h2a_plan"]["fc"]["weight_signs"]

    np.savez_compressed(outdir / "inference_params.npz", **arrays)
    plan = h2["h2a_plan"]
    config = {
        "profile": "R8B-H2A-v2 Safe Profile", "official_test_accuracy_percent": 85.03,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): _hash(p) for p in (CHECKPOINT, H0, H1, H2)},
        "thermometer": {"resolution": 8, "length": 32, "input": "uint8 RGB CHW", "binary_order": "RGB, then increasing threshold index"},
        "binary_encoding": {"bit_0": -1, "bit_1": 1, "zero_activation": "masked", "padding": "zero"},
        "h0_bias": {name: {"bits": 6, "shift": int(record["shift"])} for name, record in h0_layers.items()},
        "h1mp": {key: h1["final_plan"][key] for key in ("bits_by_node", "shift_by_node", "policy_by_node")},
        "h2a": {key: plan[key] for key in ("binary_convolution", "affine_intermediates", "residuals", "qrprelu", "gap", "fc")},
        "block_names": block_names,
        "qrprelu_branch_threshold": 0,
        "qrprelu_branch_rule": "q_in >= 0 uses positive branch",
        "rounding": {"h1_boundary": "nearest ties to even", "residual_alignment": "nearest ties away from zero", "qrp_offset": "nearest ties to even"},
    }
    (HERE / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {
        "exported_tensors": len(manifest),
        "exported_scalar_values": int(sum(np.asarray(a).size for a in arrays.values())),
        "binary_weight_values": int(sum(arrays[name + ".weight_sign"].size for name in ["conv1"] + [f"{p}.conv{i}" for p in block_names for i in (1, 2)])),
        "affine_effective_values": int(sum(arrays[name + ".sign"].size + arrays[name + ".exponent"].size + arrays[name + ".bias_q"].size for name in affine_names)),
        "qrp_effective_values": int(sum(arrays[p + ".qrprelu_output.slope_exponent"].size + arrays[p + ".qrprelu_output.xi1_q"].size + arrays[p + ".qrprelu_output.xi2_q"].size for p in block_names)),
        "fc_effective_values": int(arrays["linear.weight_sign"].size + arrays["linear.weight_exponent"].size + arrays["linear.bias_q"].size),
    }
    lines = ["# Parameter export manifest", "", "Frozen R8B-H2A-v2 Safe Profile. All inference arrays are in `params_export/inference_params.npz`; scale and width metadata are in `model_config.json`.", "", f"Checkpoint: `{config['checkpoint']}`", "", "```json", json.dumps(counts, indent=2), "```", "", "| Export tensor | Original state_dict key / source | Shape | dtype | Export file | Hardware meaning |", "|---|---|---|---|---|---|"]
    for item in manifest:
        lines.append("| `{name}` | `{original_state_dict_key}` | `{shape}` | `{dtype}` | `{export_file}` | {hardware_meaning} |".format(**item))
    lines += ["", "The checkpoint's `conv*.alpha` tensors are retained for compatibility but unused because the frozen R4 blocks set `alpha_mode=one`. `conv*.tau` is consumed only during effective weight-sign export. Bias source is H0 INT6 JSON, which the formal loader applies after loading the checkpoint.", ""]
    (HERE / "PARAM_EXPORT_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
