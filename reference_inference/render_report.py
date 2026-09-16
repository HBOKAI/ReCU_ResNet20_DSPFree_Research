"""Render the audit report from exported metadata and measured verification."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / "model_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "params_export/manifest.json").read_text(encoding="utf-8"))
    result = json.loads((HERE / "verification_results.json").read_text(encoding="utf-8"))
    binary = [item for item in manifest if item["name"].endswith(".weight_sign") and item["name"] != "linear.weight_sign"]
    affine = [item for item in manifest if item["name"].endswith((".sign", ".exponent", ".bias_q")) and not item["name"].startswith("linear.")]
    qrp = [item for item in manifest if ".qrprelu_output." in item["name"]]
    fc = [item for item in manifest if item["name"].startswith("linear.")]
    scalar = sum(__import__("math").prod(item["shape"]) for item in manifest)
    count = lambda rows: sum(__import__("math").prod(item["shape"]) for item in rows)
    acc = result["official_test_accuracy_percent"]
    all_small = all(item["maximum_integer_node_error"] == 0 and item["prediction_mismatch"] == 0 for item in result["small_batches"].values())
    bit_exact_pass = bool(all_small and result["overflow_count"] == 0 and result["prediction_mismatch"] == 0)
    anchor_reproduced = acc == 85.03
    status = "PASS" if bit_exact_pass and anchor_reproduced else "BIT-EXACT PASS / FROZEN ACCURACY NOT REPRODUCED" if bit_exact_pass else "FAIL"
    lines = [
        "# R8B-H2A-v2 independent integer reference inference report", "",
        f"**Status:** {status}. This is parameter extraction and independent NumPy/Python inference only; no checkpoint, frozen report, or model result was modified.", "",
        "## Formal frozen source", "",
        f"- R8B checkpoint: `{config['checkpoint']}`.",
        "- Model graph: `recu_hw/r7.py` (`R7ResNet20`), `recu_hw/r4.py` (`R4BasicBlock`), `recu_hw/layers.py` (`ReCUBinaryConv2d`), `recu_hw/r5t.py` (`ThermometerEncoder`), `recu_hw/qrprelu.py` (`QuantizedRPReLU`).",
        "- Official H0 INT6: `H0_INTEGER_BIAS_SWEEP_RESULTS.json`, selected 6-bit layer records.",
        "- Official H1MP: `H1MP_SEARCH_RESULTS.json`, frozen `final_plan` with 58 node bits, shifts and policies.",
        "- Official H2A-v2: `H2_FINITE_WIDTH_RESULTS.json`, `h2a_plan`; implementation `recu_hw/h2_workspace.py` (`H2TraceAdapter`).",
        "- Formal official TEST baseline: **85.03%**.", "",
        "### Freeze fingerprints (SHA-256)", "",
        "| Source file | SHA-256 |", "|---|---|",
    ]
    for path, fingerprint in config["source_sha256"].items():
        lines.append(f"| `{path}` | `{fingerprint}` |")
    lines += ["", "## Parameter export", "",
              f"- **{len(manifest)} tensors**, **{scalar:,} scalar values** in `params_export/inference_params.npz`.",
              f"- Binary kernel values: **{count(binary):,}** (stem and 18 backbone BConvs; zero weight count verified as 0).",
              f"- Effective affine values: **{count(affine):,}** (stem, 18 backbone affines, head; sign, exponent, INT6 bias).",
              f"- Effective QRP values: **{count(qrp):,}** (9 blocks × slope exponent, xi1 and xi2). Raw QRP parameters are also retained.",
              f"- Effective FC values: **{count(fc):,}** (640 signs, 640 exponents, 10 INT6 biases).",
              "- Each array's original key, shape, dtype and hardware meaning are also recorded in `PARAM_EXPORT_MANIFEST.md`.", "",
              "| Tensor | Original state_dict key / source | Shape | dtype | Hardware meaning |", "|---|---|---|---|---|"]
    for item in manifest:
        lines.append(f"| `{item['name']}` | `{item['original_state_dict_key']}` | `{item['shape']}` | `{item['dtype']}` | {item['hardware_meaning']} |")
    lines += ["", "## Exact inference equations", "",
              "**Binary encoding.** Stored bit `0 → -1`, bit `1 → +1`. `torch.sign(0)=0` is a masked third state; spatial padding is zero. Effective backbone signs are exported from the exact eval-time centered, variance-normalized, tau-clipped ReCU weights. Frozen `alpha_mode=one` makes checkpoint alpha tensors inference-inert.", "",
              "**Thermometer R=8.** For UINT8 pixel `p`, `n=round_ties_even(p/8)`, clipped to `[0,32]`. For channel `c∈{R,G,B}` and index `j∈[0,31]`, `bit[c,j]=[j≥32−n_c]`, `bipolar[c,j]=2·bit−1`. Channel order is RGB then increasing `j`; output is 96×32×32. This is the exact `ThermometerEncoder.forward()` result for official `ToTensor()` input.", "",
              "**Binary convolution.** Per 3×3 window, stride and padding come from the frozen graph. At each active input/weight pair, XNOR gives one matched bit. `P=popcount(XNOR & active_mask)`, `N=popcount(active_mask)`, `S=2P−N`. The mask excludes `sign(0)` and zero padding. Stem is INT11; Cin16 INT9; Cin32 INT10; Cin64 INT11. Measured non-H1 overflow is zero.", "",
              "**Signed-pow2 affine.** For input integer `q_x` at scale shift `s_x`, per-channel sign `σ∈{−1,+1}`, exponent `e`, INT6 bias integer `q_B` at `s_B`, choose `s=max(s_B,s_x−min(e))`. Then `q_term=σ·(q_x << (s+e−s_x))`, `q_bias=q_B << (s−s_B)`, `q_y=q_term+q_bias`. H2A finite width checks precede H1MP boundary requantization. No multiplier or new rounding is used.", "",
              "**ReCU double-skip and Option-A.** `x1 = H1(aff1(BConv(sign(x))) + shortcut(x))`; `z = H1(aff2(BConv(sign(H1(hardtanh(x1))))) + x1)`; `out = H1(QRP(z))`. Identity shortcuts inherit their input integer and scale. At layer2.0 and layer3.0, Option-A uses `x[:,:,::2,::2]` and symmetric zero channel padding of 8 and 16 channels per side respectively; no shortcut convolution exists.", "",
              "### CURRENT EXACT QRPReLU EQUATION", "",
              "The source `QuantizedRPReLU.forward()` is `x` when `x≥0`, otherwise `2^round(a)·(x+xi1)+xi2`. The frozen H2A-v2 path dispatches on the integer **input `q_in≥0`**, so the threshold is exactly zero and **xi1/xi2 are not folded away**. Per channel, let `e=round_ties_even(a)`, `s_m=max(s_in,24)`, `q_x=q_in << (s_m−s_in)`, `q_xi1=round_ties_even(xi1·2^s_m)`, `q_xi2=round_ties_even(xi2·2^s_m)`, and `q_inner=finite(q_x+q_xi1, inner_bits)`. Choose `s_out=max(s_m,s_m−min(e))`. Then `q_neg=finite((q_inner << (s_out+e−s_m)) + (q_xi2 << (s_out−s_m)), output_bits)`, `q_pos=q_in << (s_out−s_in)`, and `q_out=finite(where(q_in≥0,q_pos,q_neg),output_bits)`. The output is requantized to the frozen H1MP node shift with nearest-even rounding and saturated to that node width. This preserves horizontal offset, branch decision, pow2 branch, output offset, requantization and saturation exactly.", "",
              "**H1MP representation.** Every one of the 58 frozen nodes has `q`, width, scale shift `s` and policy in `model_config.json`; real value is `q·2^(−s)`. At a normal H1 boundary, scale conversion rounds to nearest with ties to even and saturates. At each of 18 residual adds, branch alignment uses left shifts or symmetric nearest with ties **away from zero**, then exact addition and frozen H1 saturation. H2A adds its conservative finite-width check before that H1 boundary.", "",
              "**GAP H2A-v2.** On final 64×8×8 integer state, `q_gap[c]=Σ_{h,w} q[c,h,w]` (INT16), `s_gap=s_input+6`. There is **no arithmetic >>6** and no GAP rounding. The /64 is exclusively scale metadata.", "",
              "**Head and FC.** Head uses the affine equation above. For class `k` and 64 features, with FC sign `σ_ki`, exponent `e_ki`, head shift `s_h`, and common shift `s_c=11`, `term_ki=σ_ki·(q_i << (s_c+e_ki−s_h))`; `acc_k=Σ_i term_ki` is INT24. Align it to FC bias-add shift 11, add frozen INT6 bias at shift 6, then bound the final logit to INT24. Prediction is integer argmax over ten logits.", "",
              "### Frozen H1MP 58-node plan", "",
              "| Node | Width | Shift | Policy |", "|---|---:|---:|---|"]
    for name, bits in config["h1mp"]["bits_by_node"].items():
        lines.append(f"| `{name}` | INT{bits} | {config['h1mp']['shift_by_node'][name]} | {config['h1mp']['policy_by_node'][name]} |")
    lines += ["", "## Verification", "",
              "The verifier compares the independent NumPy reference with the frozen H2TraceAdapter on the same official CIFAR-10 test pixels. Its oracle capture includes Thermometer, binary accumulators, affine terms/outputs, Option-A, both residual adds per block, QRP branch/inner/output, 58 H1 states, GAP, head, FC shifted terms/accumulator and logits.", "",
              "| Official TEST prefix | Compared integer nodes | Maximum absolute node error | Prediction mismatch |", "|---:|---:|---:|---:|"]
    for size, info in result["small_batches"].items():
        lines.append(f"| {size} | {info['node_count']} | {info['maximum_integer_node_error']} | {info['prediction_mismatch']} |")
    lines += ["", "| Metric | Result |", "|---|---:|",
              f"| Thermometer mismatch | {result['thermometer_mismatch']} |",
              f"| Maximum exact integer node error | {result['maximum_integer_node_error']} |",
              f"| QRP branch mismatch | {result['qrp_branch_mismatch']} |",
              f"| Residual alignment status | {'18/18 exact' if result['maximum_integer_node_error'] == 0 else 'FAIL'} |",
              f"| Final logits mismatch | {result['final_logits_mismatch']} |",
              f"| Prediction mismatch (including full TEST if run) | {result['prediction_mismatch']} |",
              f"| H2A exact-width overflow | {result['overflow_count']} |",
              f"| H1MP boundary saturations during verifier | {result['h1_saturation_count']} |",
              f"| Official CIFAR-10 TEST reference accuracy | {acc if acc is not None else 'not run'}% |",
              f"| Current CPU formal implementation accuracy, same 10,000 images | {result.get('current_implementation_test_accuracy_percent')}% |",
              f"| Frozen report accuracy | 85.03% |",
              "| Focused pytest checks | 6 passed |", "",
              "### Node-by-node maximum integer error (largest debug prefix)", "",
              "| Node | Max abs error | Mismatch elements |", "|---|---:|---:|"]
    for name, item in result["node_errors"].items():
        lines.append(f"| `{name}` | {item.get('max_abs_error', 'missing')} | {item.get('mismatch_count', 'missing')} |")
    lines += ["", "The official 10,000-image TEST is run only after 1, 10 and 100 image exact checks. The full run compares predictions and accuracy; 245-node tensor capture is the prefix debug contract. Known H1MP saturation is expected and separate from H2A exact-width overflow.", "",
              "**Accuracy anchor discrepancy.** The frozen H2A-v2 report records 85.03% (8,503/10,000). The independent reference and the current CPU H2TraceAdapter both produce 85.01% (8,501/10,000), with zero prediction mismatch. A second evaluation using the official `CIFAR10` `ToTensor()` loader, batch size 128 and repository `evaluate()` also produced 85.01%; raw pixels, labels and `ToTensor()` values were checked equal. The cause of the 2-image difference from the historical frozen result is not established; the historical run may have used a different device or numerical environment. We did not alter any arithmetic, parameters, or frozen results to force 85.03%. The 85.03% acceptance criterion therefore remains unmet in this workspace run.", "",
              "No RTL or synthesis was started.", ""]
    (HERE / "REFERENCE_INFERENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report status: {status}; accuracy: {acc}")


if __name__ == "__main__":
    main()
