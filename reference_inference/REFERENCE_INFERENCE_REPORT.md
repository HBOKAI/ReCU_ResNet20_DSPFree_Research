# R8B-H2A-v2 independent integer reference inference report

**Status:** BIT-EXACT PASS / FROZEN ACCURACY NOT REPRODUCED. This is parameter extraction and independent NumPy/Python inference only; no checkpoint, frozen report, or model result was modified.

## Formal frozen source

- R8B checkpoint: `experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt`.
- Model graph: `recu_hw/r7.py` (`R7ResNet20`), `recu_hw/r4.py` (`R4BasicBlock`), `recu_hw/layers.py` (`ReCUBinaryConv2d`), `recu_hw/r5t.py` (`ThermometerEncoder`), `recu_hw/qrprelu.py` (`QuantizedRPReLU`).
- Official H0 INT6: `H0_INTEGER_BIAS_SWEEP_RESULTS.json`, selected 6-bit layer records.
- Official H1MP: `H1MP_SEARCH_RESULTS.json`, frozen `final_plan` with 58 node bits, shifts and policies.
- Official H2A-v2: `H2_FINITE_WIDTH_RESULTS.json`, `h2a_plan`; implementation `recu_hw/h2_workspace.py` (`H2TraceAdapter`).
- Formal official TEST baseline: **85.03%**.

### Freeze fingerprints (SHA-256)

| Source file | SHA-256 |
|---|---|
| `experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt` | `c07ec3b29b88bbf70d3300f0bb19c7113abdadd305eb3d73cc12c7f40c2b0ba8` |
| `H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `b608bc9d5325ee9cfa79802c3d8bf4904365c8bf49ac4ad82235357e57465ac3` |
| `H1MP_SEARCH_RESULTS.json` | `83da34d5c4f77643c2e9b2f69d6a8f478f70b2eb8673bbb37457d5e59efda82d` |
| `H2_FINITE_WIDTH_RESULTS.json` | `13e7889628432bcdf2b3419503492e72c7604ba858993b4b9008746aec857f12` |

## Parameter export

- **136 tensors**, **286,650 scalar values** in `params_export/inference_params.npz`.
- Binary kernel values: **281,088** (stem and 18 backbone BConvs; zero weight count verified as 0).
- Effective affine values: **2,256** (stem, 18 backbone affines, head; sign, exponent, INT6 bias).
- Effective QRP values: **1,008** (9 blocks × slope exponent, xi1 and xi2). Raw QRP parameters are also retained.
- Effective FC values: **1,290** (640 signs, 640 exponents, 10 INT6 biases).
- Each array's original key, shape, dtype and hardware meaning are also recorded in `PARAM_EXPORT_MANIFEST.md`.

| Tensor | Original state_dict key / source | Shape | dtype | Hardware meaning |
|---|---|---|---|---|
| `conv1.weight_sign` | `conv1.weight` | `[16, 96, 3, 3]` | `int8` | Stem binary kernel: -1/0/+1; 0 bit means -1, 1 bit means +1 when active |
| `layer1.0.conv1.weight_sign` | `layer1.0.conv1.weight, layer1.0.conv1.tau` | `[16, 16, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.0.conv2.weight_sign` | `layer1.0.conv2.weight, layer1.0.conv2.tau` | `[16, 16, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.1.conv1.weight_sign` | `layer1.1.conv1.weight, layer1.1.conv1.tau` | `[16, 16, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.1.conv2.weight_sign` | `layer1.1.conv2.weight, layer1.1.conv2.tau` | `[16, 16, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.2.conv1.weight_sign` | `layer1.2.conv1.weight, layer1.2.conv1.tau` | `[16, 16, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.2.conv2.weight_sign` | `layer1.2.conv2.weight, layer1.2.conv2.tau` | `[16, 16, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.0.conv1.weight_sign` | `layer2.0.conv1.weight, layer2.0.conv1.tau` | `[32, 16, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.0.conv2.weight_sign` | `layer2.0.conv2.weight, layer2.0.conv2.tau` | `[32, 32, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.1.conv1.weight_sign` | `layer2.1.conv1.weight, layer2.1.conv1.tau` | `[32, 32, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.1.conv2.weight_sign` | `layer2.1.conv2.weight, layer2.1.conv2.tau` | `[32, 32, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.2.conv1.weight_sign` | `layer2.2.conv1.weight, layer2.2.conv1.tau` | `[32, 32, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.2.conv2.weight_sign` | `layer2.2.conv2.weight, layer2.2.conv2.tau` | `[32, 32, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.0.conv1.weight_sign` | `layer3.0.conv1.weight, layer3.0.conv1.tau` | `[64, 32, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.0.conv2.weight_sign` | `layer3.0.conv2.weight, layer3.0.conv2.tau` | `[64, 64, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.1.conv1.weight_sign` | `layer3.1.conv1.weight, layer3.1.conv1.tau` | `[64, 64, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.1.conv2.weight_sign` | `layer3.1.conv2.weight, layer3.1.conv2.tau` | `[64, 64, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.2.conv1.weight_sign` | `layer3.2.conv1.weight, layer3.2.conv1.tau` | `[64, 64, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.2.conv2.weight_sign` | `layer3.2.conv2.weight, layer3.2.conv2.tau` | `[64, 64, 3, 3]` | `int8` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `stem_affine.sign` | `stem_affine.k_latent` | `[16]` | `int8` | Signed-pow2 K sign |
| `stem_affine.exponent` | `stem_affine.k_latent` | `[16]` | `int16` | Signed-pow2 K power-of-two exponent |
| `stem_affine.bias_q` | `stem_affine.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer1.0.aff1.sign` | `layer1.0.aff1.log2_abs_k, layer1.0.aff1.sign_k` | `[16]` | `int8` | Signed-pow2 K sign |
| `layer1.0.aff1.exponent` | `layer1.0.aff1.log2_abs_k, layer1.0.aff1.sign_k` | `[16]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer1.0.aff1.bias_q` | `layer1.0.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer1.0.aff2.sign` | `layer1.0.aff2.log2_abs_k, layer1.0.aff2.sign_k` | `[16]` | `int8` | Signed-pow2 K sign |
| `layer1.0.aff2.exponent` | `layer1.0.aff2.log2_abs_k, layer1.0.aff2.sign_k` | `[16]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer1.0.aff2.bias_q` | `layer1.0.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer1.1.aff1.sign` | `layer1.1.aff1.log2_abs_k, layer1.1.aff1.sign_k` | `[16]` | `int8` | Signed-pow2 K sign |
| `layer1.1.aff1.exponent` | `layer1.1.aff1.log2_abs_k, layer1.1.aff1.sign_k` | `[16]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer1.1.aff1.bias_q` | `layer1.1.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer1.1.aff2.sign` | `layer1.1.aff2.log2_abs_k, layer1.1.aff2.sign_k` | `[16]` | `int8` | Signed-pow2 K sign |
| `layer1.1.aff2.exponent` | `layer1.1.aff2.log2_abs_k, layer1.1.aff2.sign_k` | `[16]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer1.1.aff2.bias_q` | `layer1.1.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer1.2.aff1.sign` | `layer1.2.aff1.log2_abs_k, layer1.2.aff1.sign_k` | `[16]` | `int8` | Signed-pow2 K sign |
| `layer1.2.aff1.exponent` | `layer1.2.aff1.log2_abs_k, layer1.2.aff1.sign_k` | `[16]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer1.2.aff1.bias_q` | `layer1.2.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer1.2.aff2.sign` | `layer1.2.aff2.log2_abs_k, layer1.2.aff2.sign_k` | `[16]` | `int8` | Signed-pow2 K sign |
| `layer1.2.aff2.exponent` | `layer1.2.aff2.log2_abs_k, layer1.2.aff2.sign_k` | `[16]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer1.2.aff2.bias_q` | `layer1.2.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer2.0.aff1.sign` | `layer2.0.aff1.log2_abs_k, layer2.0.aff1.sign_k` | `[32]` | `int8` | Signed-pow2 K sign |
| `layer2.0.aff1.exponent` | `layer2.0.aff1.log2_abs_k, layer2.0.aff1.sign_k` | `[32]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer2.0.aff1.bias_q` | `layer2.0.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer2.0.aff2.sign` | `layer2.0.aff2.log2_abs_k, layer2.0.aff2.sign_k` | `[32]` | `int8` | Signed-pow2 K sign |
| `layer2.0.aff2.exponent` | `layer2.0.aff2.log2_abs_k, layer2.0.aff2.sign_k` | `[32]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer2.0.aff2.bias_q` | `layer2.0.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer2.1.aff1.sign` | `layer2.1.aff1.log2_abs_k, layer2.1.aff1.sign_k` | `[32]` | `int8` | Signed-pow2 K sign |
| `layer2.1.aff1.exponent` | `layer2.1.aff1.log2_abs_k, layer2.1.aff1.sign_k` | `[32]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer2.1.aff1.bias_q` | `layer2.1.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer2.1.aff2.sign` | `layer2.1.aff2.log2_abs_k, layer2.1.aff2.sign_k` | `[32]` | `int8` | Signed-pow2 K sign |
| `layer2.1.aff2.exponent` | `layer2.1.aff2.log2_abs_k, layer2.1.aff2.sign_k` | `[32]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer2.1.aff2.bias_q` | `layer2.1.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer2.2.aff1.sign` | `layer2.2.aff1.log2_abs_k, layer2.2.aff1.sign_k` | `[32]` | `int8` | Signed-pow2 K sign |
| `layer2.2.aff1.exponent` | `layer2.2.aff1.log2_abs_k, layer2.2.aff1.sign_k` | `[32]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer2.2.aff1.bias_q` | `layer2.2.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer2.2.aff2.sign` | `layer2.2.aff2.log2_abs_k, layer2.2.aff2.sign_k` | `[32]` | `int8` | Signed-pow2 K sign |
| `layer2.2.aff2.exponent` | `layer2.2.aff2.log2_abs_k, layer2.2.aff2.sign_k` | `[32]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer2.2.aff2.bias_q` | `layer2.2.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer3.0.aff1.sign` | `layer3.0.aff1.log2_abs_k, layer3.0.aff1.sign_k` | `[64]` | `int8` | Signed-pow2 K sign |
| `layer3.0.aff1.exponent` | `layer3.0.aff1.log2_abs_k, layer3.0.aff1.sign_k` | `[64]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer3.0.aff1.bias_q` | `layer3.0.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer3.0.aff2.sign` | `layer3.0.aff2.log2_abs_k, layer3.0.aff2.sign_k` | `[64]` | `int8` | Signed-pow2 K sign |
| `layer3.0.aff2.exponent` | `layer3.0.aff2.log2_abs_k, layer3.0.aff2.sign_k` | `[64]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer3.0.aff2.bias_q` | `layer3.0.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer3.1.aff1.sign` | `layer3.1.aff1.log2_abs_k, layer3.1.aff1.sign_k` | `[64]` | `int8` | Signed-pow2 K sign |
| `layer3.1.aff1.exponent` | `layer3.1.aff1.log2_abs_k, layer3.1.aff1.sign_k` | `[64]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer3.1.aff1.bias_q` | `layer3.1.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer3.1.aff2.sign` | `layer3.1.aff2.log2_abs_k, layer3.1.aff2.sign_k` | `[64]` | `int8` | Signed-pow2 K sign |
| `layer3.1.aff2.exponent` | `layer3.1.aff2.log2_abs_k, layer3.1.aff2.sign_k` | `[64]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer3.1.aff2.bias_q` | `layer3.1.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer3.2.aff1.sign` | `layer3.2.aff1.log2_abs_k, layer3.2.aff1.sign_k` | `[64]` | `int8` | Signed-pow2 K sign |
| `layer3.2.aff1.exponent` | `layer3.2.aff1.log2_abs_k, layer3.2.aff1.sign_k` | `[64]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer3.2.aff1.bias_q` | `layer3.2.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer3.2.aff2.sign` | `layer3.2.aff2.log2_abs_k, layer3.2.aff2.sign_k` | `[64]` | `int8` | Signed-pow2 K sign |
| `layer3.2.aff2.exponent` | `layer3.2.aff2.log2_abs_k, layer3.2.aff2.sign_k` | `[64]` | `int16` | Signed-pow2 K power-of-two exponent |
| `layer3.2.aff2.bias_q` | `layer3.2.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `head_affine.sign` | `head_affine.k_latent` | `[64]` | `int8` | Signed-pow2 K sign |
| `head_affine.exponent` | `head_affine.k_latent` | `[64]` | `int16` | Signed-pow2 K power-of-two exponent |
| `head_affine.bias_q` | `head_affine.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | INT6 bias at per-module scale in model_config.json |
| `layer1.0.post_act.a_raw` | `layer1.0.post_act.a` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.0.post_act.xi1_raw` | `layer1.0.post_act.xi1` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.0.post_act.xi2_raw` | `layer1.0.post_act.xi2` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.0.qrprelu_output.slope_exponent` | `layer1.0.post_act.a` | `[16]` | `int16` | Negative-branch pow2 exponent |
| `layer1.0.qrprelu_output.xi1_q` | `layer1.0.post_act.xi1` | `[16]` | `int64` | QRP offset integer at H2 param_shift |
| `layer1.0.qrprelu_output.xi2_q` | `layer1.0.post_act.xi2` | `[16]` | `int64` | QRP offset integer at H2 param_shift |
| `layer1.1.post_act.a_raw` | `layer1.1.post_act.a` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.1.post_act.xi1_raw` | `layer1.1.post_act.xi1` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.1.post_act.xi2_raw` | `layer1.1.post_act.xi2` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.1.qrprelu_output.slope_exponent` | `layer1.1.post_act.a` | `[16]` | `int16` | Negative-branch pow2 exponent |
| `layer1.1.qrprelu_output.xi1_q` | `layer1.1.post_act.xi1` | `[16]` | `int64` | QRP offset integer at H2 param_shift |
| `layer1.1.qrprelu_output.xi2_q` | `layer1.1.post_act.xi2` | `[16]` | `int64` | QRP offset integer at H2 param_shift |
| `layer1.2.post_act.a_raw` | `layer1.2.post_act.a` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.2.post_act.xi1_raw` | `layer1.2.post_act.xi1` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.2.post_act.xi2_raw` | `layer1.2.post_act.xi2` | `[16]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.2.qrprelu_output.slope_exponent` | `layer1.2.post_act.a` | `[16]` | `int16` | Negative-branch pow2 exponent |
| `layer1.2.qrprelu_output.xi1_q` | `layer1.2.post_act.xi1` | `[16]` | `int64` | QRP offset integer at H2 param_shift |
| `layer1.2.qrprelu_output.xi2_q` | `layer1.2.post_act.xi2` | `[16]` | `int64` | QRP offset integer at H2 param_shift |
| `layer2.0.post_act.a_raw` | `layer2.0.post_act.a` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.0.post_act.xi1_raw` | `layer2.0.post_act.xi1` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.0.post_act.xi2_raw` | `layer2.0.post_act.xi2` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.0.qrprelu_output.slope_exponent` | `layer2.0.post_act.a` | `[32]` | `int16` | Negative-branch pow2 exponent |
| `layer2.0.qrprelu_output.xi1_q` | `layer2.0.post_act.xi1` | `[32]` | `int64` | QRP offset integer at H2 param_shift |
| `layer2.0.qrprelu_output.xi2_q` | `layer2.0.post_act.xi2` | `[32]` | `int64` | QRP offset integer at H2 param_shift |
| `layer2.1.post_act.a_raw` | `layer2.1.post_act.a` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.1.post_act.xi1_raw` | `layer2.1.post_act.xi1` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.1.post_act.xi2_raw` | `layer2.1.post_act.xi2` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.1.qrprelu_output.slope_exponent` | `layer2.1.post_act.a` | `[32]` | `int16` | Negative-branch pow2 exponent |
| `layer2.1.qrprelu_output.xi1_q` | `layer2.1.post_act.xi1` | `[32]` | `int64` | QRP offset integer at H2 param_shift |
| `layer2.1.qrprelu_output.xi2_q` | `layer2.1.post_act.xi2` | `[32]` | `int64` | QRP offset integer at H2 param_shift |
| `layer2.2.post_act.a_raw` | `layer2.2.post_act.a` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.2.post_act.xi1_raw` | `layer2.2.post_act.xi1` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.2.post_act.xi2_raw` | `layer2.2.post_act.xi2` | `[32]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.2.qrprelu_output.slope_exponent` | `layer2.2.post_act.a` | `[32]` | `int16` | Negative-branch pow2 exponent |
| `layer2.2.qrprelu_output.xi1_q` | `layer2.2.post_act.xi1` | `[32]` | `int64` | QRP offset integer at H2 param_shift |
| `layer2.2.qrprelu_output.xi2_q` | `layer2.2.post_act.xi2` | `[32]` | `int64` | QRP offset integer at H2 param_shift |
| `layer3.0.post_act.a_raw` | `layer3.0.post_act.a` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.0.post_act.xi1_raw` | `layer3.0.post_act.xi1` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.0.post_act.xi2_raw` | `layer3.0.post_act.xi2` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.0.qrprelu_output.slope_exponent` | `layer3.0.post_act.a` | `[64]` | `int16` | Negative-branch pow2 exponent |
| `layer3.0.qrprelu_output.xi1_q` | `layer3.0.post_act.xi1` | `[64]` | `int64` | QRP offset integer at H2 param_shift |
| `layer3.0.qrprelu_output.xi2_q` | `layer3.0.post_act.xi2` | `[64]` | `int64` | QRP offset integer at H2 param_shift |
| `layer3.1.post_act.a_raw` | `layer3.1.post_act.a` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.1.post_act.xi1_raw` | `layer3.1.post_act.xi1` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.1.post_act.xi2_raw` | `layer3.1.post_act.xi2` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.1.qrprelu_output.slope_exponent` | `layer3.1.post_act.a` | `[64]` | `int16` | Negative-branch pow2 exponent |
| `layer3.1.qrprelu_output.xi1_q` | `layer3.1.post_act.xi1` | `[64]` | `int64` | QRP offset integer at H2 param_shift |
| `layer3.1.qrprelu_output.xi2_q` | `layer3.1.post_act.xi2` | `[64]` | `int64` | QRP offset integer at H2 param_shift |
| `layer3.2.post_act.a_raw` | `layer3.2.post_act.a` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.2.post_act.xi1_raw` | `layer3.2.post_act.xi1` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.2.post_act.xi2_raw` | `layer3.2.post_act.xi2` | `[64]` | `float32` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.2.qrprelu_output.slope_exponent` | `layer3.2.post_act.a` | `[64]` | `int16` | Negative-branch pow2 exponent |
| `layer3.2.qrprelu_output.xi1_q` | `layer3.2.post_act.xi1` | `[64]` | `int64` | QRP offset integer at H2 param_shift |
| `layer3.2.qrprelu_output.xi2_q` | `layer3.2.post_act.xi2` | `[64]` | `int64` | QRP offset integer at H2 param_shift |
| `linear.weight_sign` | `linear.weight` | `[10, 64]` | `int8` | 64-to-10 signed-pow2 FC term sign |
| `linear.weight_exponent` | `linear.weight` | `[10, 64]` | `int16` | 64-to-10 signed-pow2 FC term exponent |
| `linear.bias_q` | `linear.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[10]` | `int8` | INT6 FC bias |

## Exact inference equations

**Binary encoding.** Stored bit `0 → -1`, bit `1 → +1`. `torch.sign(0)=0` is a masked third state; spatial padding is zero. Effective backbone signs are exported from the exact eval-time centered, variance-normalized, tau-clipped ReCU weights. Frozen `alpha_mode=one` makes checkpoint alpha tensors inference-inert.

**Thermometer R=8.** For UINT8 pixel `p`, `n=round_ties_even(p/8)`, clipped to `[0,32]`. For channel `c∈{R,G,B}` and index `j∈[0,31]`, `bit[c,j]=[j≥32−n_c]`, `bipolar[c,j]=2·bit−1`. Channel order is RGB then increasing `j`; output is 96×32×32. This is the exact `ThermometerEncoder.forward()` result for official `ToTensor()` input.

**Binary convolution.** Per 3×3 window, stride and padding come from the frozen graph. At each active input/weight pair, XNOR gives one matched bit. `P=popcount(XNOR & active_mask)`, `N=popcount(active_mask)`, `S=2P−N`. The mask excludes `sign(0)` and zero padding. Stem is INT11; Cin16 INT9; Cin32 INT10; Cin64 INT11. Measured non-H1 overflow is zero.

**Signed-pow2 affine.** For input integer `q_x` at scale shift `s_x`, per-channel sign `σ∈{−1,+1}`, exponent `e`, INT6 bias integer `q_B` at `s_B`, choose `s=max(s_B,s_x−min(e))`. Then `q_term=σ·(q_x << (s+e−s_x))`, `q_bias=q_B << (s−s_B)`, `q_y=q_term+q_bias`. H2A finite width checks precede H1MP boundary requantization. No multiplier or new rounding is used.

**ReCU double-skip and Option-A.** `x1 = H1(aff1(BConv(sign(x))) + shortcut(x))`; `z = H1(aff2(BConv(sign(H1(hardtanh(x1))))) + x1)`; `out = H1(QRP(z))`. Identity shortcuts inherit their input integer and scale. At layer2.0 and layer3.0, Option-A uses `x[:,:,::2,::2]` and symmetric zero channel padding of 8 and 16 channels per side respectively; no shortcut convolution exists.

### CURRENT EXACT QRPReLU EQUATION

The source `QuantizedRPReLU.forward()` is `x` when `x≥0`, otherwise `2^round(a)·(x+xi1)+xi2`. The frozen H2A-v2 path dispatches on the integer **input `q_in≥0`**, so the threshold is exactly zero and **xi1/xi2 are not folded away**. Per channel, let `e=round_ties_even(a)`, `s_m=max(s_in,24)`, `q_x=q_in << (s_m−s_in)`, `q_xi1=round_ties_even(xi1·2^s_m)`, `q_xi2=round_ties_even(xi2·2^s_m)`, and `q_inner=finite(q_x+q_xi1, inner_bits)`. Choose `s_out=max(s_m,s_m−min(e))`. Then `q_neg=finite((q_inner << (s_out+e−s_m)) + (q_xi2 << (s_out−s_m)), output_bits)`, `q_pos=q_in << (s_out−s_in)`, and `q_out=finite(where(q_in≥0,q_pos,q_neg),output_bits)`. The output is requantized to the frozen H1MP node shift with nearest-even rounding and saturated to that node width. This preserves horizontal offset, branch decision, pow2 branch, output offset, requantization and saturation exactly.

**H1MP representation.** Every one of the 58 frozen nodes has `q`, width, scale shift `s` and policy in `model_config.json`; real value is `q·2^(−s)`. At a normal H1 boundary, scale conversion rounds to nearest with ties to even and saturates. At each of 18 residual adds, branch alignment uses left shifts or symmetric nearest with ties **away from zero**, then exact addition and frozen H1 saturation. H2A adds its conservative finite-width check before that H1 boundary.

**GAP H2A-v2.** On final 64×8×8 integer state, `q_gap[c]=Σ_{h,w} q[c,h,w]` (INT16), `s_gap=s_input+6`. There is **no arithmetic >>6** and no GAP rounding. The /64 is exclusively scale metadata.

**Head and FC.** Head uses the affine equation above. For class `k` and 64 features, with FC sign `σ_ki`, exponent `e_ki`, head shift `s_h`, and common shift `s_c=11`, `term_ki=σ_ki·(q_i << (s_c+e_ki−s_h))`; `acc_k=Σ_i term_ki` is INT24. Align it to FC bias-add shift 11, add frozen INT6 bias at shift 6, then bound the final logit to INT24. Prediction is integer argmax over ten logits.

### Frozen H1MP 58-node plan

| Node | Width | Shift | Policy |
|---|---:|---:|---|
| `stem.post_hardtanh` | INT6 | 5 | mse |
| `layer1.0.first_affine` | INT6 | 4 | mse |
| `layer1.0.x1` | INT6 | 3 | mse |
| `layer1.0.pre_bconv2_hardtanh` | INT6 | 5 | mse |
| `layer1.0.second_affine` | INT6 | 4 | mse |
| `layer1.0.second_add` | INT6 | 3 | mse |
| `layer1.0.qrprelu_output` | INT7 | 4 | mse |
| `layer1.1.first_affine` | INT6 | 4 | mse |
| `layer1.1.x1` | INT6 | 3 | mse |
| `layer1.1.pre_bconv2_hardtanh` | INT6 | 5 | mse |
| `layer1.1.second_affine` | INT6 | 4 | mse |
| `layer1.1.second_add` | INT6 | 3 | mse |
| `layer1.1.qrprelu_output` | INT12 | 8 | mse |
| `layer1.2.first_affine` | INT6 | 4 | mse |
| `layer1.2.x1` | INT6 | 3 | mse |
| `layer1.2.pre_bconv2_hardtanh` | INT6 | 5 | mse |
| `layer1.2.second_affine` | INT7 | 4 | mse |
| `layer1.2.second_add` | INT8 | 4 | mse |
| `layer1.2.qrprelu_output` | INT6 | 3 | mse |
| `layer2.0.first_affine` | INT8 | 5 | absmax |
| `layer2.0.stage_shortcut` | INT6 | 3 | mse |
| `layer2.0.x1` | INT6 | 3 | mse |
| `layer2.0.pre_bconv2_hardtanh` | INT6 | 5 | mse |
| `layer2.0.second_affine` | INT6 | 3 | mse |
| `layer2.0.second_add` | INT6 | 3 | mse |
| `layer2.0.qrprelu_output` | INT6 | 2 | p99.99 |
| `layer2.1.first_affine` | INT7 | 5 | mse |
| `layer2.1.x1` | INT6 | 3 | mse |
| `layer2.1.pre_bconv2_hardtanh` | INT6 | 5 | mse |
| `layer2.1.second_affine` | INT7 | 4 | mse |
| `layer2.1.second_add` | INT8 | 3 | absmax |
| `layer2.1.qrprelu_output` | INT12 | 8 | absmax |
| `layer2.2.first_affine` | INT7 | 5 | mse |
| `layer2.2.x1` | INT6 | 2 | absmax |
| `layer2.2.pre_bconv2_hardtanh` | INT6 | 4 | absmax |
| `layer2.2.second_affine` | INT6 | 4 | mse |
| `layer2.2.second_add` | INT6 | 3 | mse |
| `layer2.2.qrprelu_output` | INT6 | 3 | mse |
| `layer3.0.first_affine` | INT6 | 4 | mse |
| `layer3.0.stage_shortcut` | INT6 | 2 | absmax |
| `layer3.0.x1` | INT6 | 3 | mse |
| `layer3.0.pre_bconv2_hardtanh` | INT6 | 5 | mse |
| `layer3.0.second_affine` | INT6 | 3 | mse |
| `layer3.0.second_add` | INT6 | 3 | mse |
| `layer3.0.qrprelu_output` | INT6 | 3 | mse |
| `layer3.1.first_affine` | INT7 | 4 | mse |
| `layer3.1.x1` | INT7 | 4 | mse |
| `layer3.1.pre_bconv2_hardtanh` | INT7 | 6 | mse |
| `layer3.1.second_affine` | INT7 | 4 | mse |
| `layer3.1.second_add` | INT7 | 3 | mse |
| `layer3.1.qrprelu_output` | INT10 | 7 | p99.9 |
| `layer3.2.first_affine` | INT7 | 4 | mse |
| `layer3.2.x1` | INT6 | 3 | mse |
| `layer3.2.pre_bconv2_hardtanh` | INT7 | 6 | mse |
| `layer3.2.second_affine` | INT7 | 4 | mse |
| `layer3.2.second_add` | INT7 | 3 | mse |
| `layer3.2.qrprelu_output` | INT10 | 6 | mse |
| `head.affine_output` | INT6 | 2 | mse |

## Verification

The verifier compares the independent NumPy reference with the frozen H2TraceAdapter on the same official CIFAR-10 test pixels. Its oracle capture includes Thermometer, binary accumulators, affine terms/outputs, Option-A, both residual adds per block, QRP branch/inner/output, 58 H1 states, GAP, head, FC shifted terms/accumulator and logits.

| Official TEST prefix | Compared integer nodes | Maximum absolute node error | Prediction mismatch |
|---:|---:|---:|---:|
| 1 | 245 | 0 | 0 |
| 10 | 245 | 0 | 0 |
| 100 | 245 | 0 | 0 |

| Metric | Result |
|---|---:|
| Thermometer mismatch | 0 |
| Maximum exact integer node error | 0 |
| QRP branch mismatch | 0 |
| Residual alignment status | 18/18 exact |
| Final logits mismatch | 0 |
| Prediction mismatch (including full TEST if run) | 0 |
| H2A exact-width overflow | 0 |
| H1MP boundary saturations during verifier | 109181803 |
| Official CIFAR-10 TEST reference accuracy | 85.01% |
| Current CPU formal implementation accuracy, same 10,000 images | 85.01% |
| Frozen report accuracy | 85.03% |
| Focused pytest checks | 6 passed |

### Node-by-node maximum integer error (largest debug prefix)

| Node | Max abs error | Mismatch elements |
|---|---:|---:|
| `thermometer` | 0 | 0 |
| `binary.stem.signed_acc` | 0 | 0 |
| `affine.stem.post_hardtanh.shifted_term` | 0 | 0 |
| `affine.stem.post_hardtanh.bias_aligned` | 0 | 0 |
| `affine.stem.post_hardtanh.add_output` | 0 | 0 |
| `h1.stem.post_hardtanh` | 0 | 0 |
| `binary.layer1.0.conv1_signed_acc` | 0 | 0 |
| `affine.layer1.0.first_affine.shifted_term` | 0 | 0 |
| `affine.layer1.0.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer1.0.first_affine.add_output` | 0 | 0 |
| `h1.layer1.0.first_affine` | 0 | 0 |
| `residual.layer1.0.x1.align_a` | 0 | 0 |
| `residual.layer1.0.x1.align_b` | 0 | 0 |
| `residual.layer1.0.x1.add_output` | 0 | 0 |
| `h1.layer1.0.x1` | 0 | 0 |
| `h1.layer1.0.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer1.0.conv2_signed_acc` | 0 | 0 |
| `affine.layer1.0.second_affine.shifted_term` | 0 | 0 |
| `affine.layer1.0.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer1.0.second_affine.add_output` | 0 | 0 |
| `h1.layer1.0.second_affine` | 0 | 0 |
| `residual.layer1.0.second_add.align_a` | 0 | 0 |
| `residual.layer1.0.second_add.align_b` | 0 | 0 |
| `residual.layer1.0.second_add.add_output` | 0 | 0 |
| `h1.layer1.0.second_add` | 0 | 0 |
| `qrprelu.layer1.0.branch_positive` | 0 | 0 |
| `qrprelu.layer1.0.inner_add` | 0 | 0 |
| `qrprelu.layer1.0.negative_branch` | 0 | 0 |
| `qrprelu.layer1.0.output` | 0 | 0 |
| `h1.layer1.0.qrprelu_output` | 0 | 0 |
| `layer1.0.stage_output` | 0 | 0 |
| `binary.layer1.1.conv1_signed_acc` | 0 | 0 |
| `affine.layer1.1.first_affine.shifted_term` | 0 | 0 |
| `affine.layer1.1.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer1.1.first_affine.add_output` | 0 | 0 |
| `h1.layer1.1.first_affine` | 0 | 0 |
| `residual.layer1.1.x1.align_a` | 0 | 0 |
| `residual.layer1.1.x1.align_b` | 0 | 0 |
| `residual.layer1.1.x1.add_output` | 0 | 0 |
| `h1.layer1.1.x1` | 0 | 0 |
| `h1.layer1.1.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer1.1.conv2_signed_acc` | 0 | 0 |
| `affine.layer1.1.second_affine.shifted_term` | 0 | 0 |
| `affine.layer1.1.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer1.1.second_affine.add_output` | 0 | 0 |
| `h1.layer1.1.second_affine` | 0 | 0 |
| `residual.layer1.1.second_add.align_a` | 0 | 0 |
| `residual.layer1.1.second_add.align_b` | 0 | 0 |
| `residual.layer1.1.second_add.add_output` | 0 | 0 |
| `h1.layer1.1.second_add` | 0 | 0 |
| `qrprelu.layer1.1.branch_positive` | 0 | 0 |
| `qrprelu.layer1.1.inner_add` | 0 | 0 |
| `qrprelu.layer1.1.negative_branch` | 0 | 0 |
| `qrprelu.layer1.1.output` | 0 | 0 |
| `h1.layer1.1.qrprelu_output` | 0 | 0 |
| `layer1.1.stage_output` | 0 | 0 |
| `binary.layer1.2.conv1_signed_acc` | 0 | 0 |
| `affine.layer1.2.first_affine.shifted_term` | 0 | 0 |
| `affine.layer1.2.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer1.2.first_affine.add_output` | 0 | 0 |
| `h1.layer1.2.first_affine` | 0 | 0 |
| `residual.layer1.2.x1.align_a` | 0 | 0 |
| `residual.layer1.2.x1.align_b` | 0 | 0 |
| `residual.layer1.2.x1.add_output` | 0 | 0 |
| `h1.layer1.2.x1` | 0 | 0 |
| `h1.layer1.2.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer1.2.conv2_signed_acc` | 0 | 0 |
| `affine.layer1.2.second_affine.shifted_term` | 0 | 0 |
| `affine.layer1.2.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer1.2.second_affine.add_output` | 0 | 0 |
| `h1.layer1.2.second_affine` | 0 | 0 |
| `residual.layer1.2.second_add.align_a` | 0 | 0 |
| `residual.layer1.2.second_add.align_b` | 0 | 0 |
| `residual.layer1.2.second_add.add_output` | 0 | 0 |
| `h1.layer1.2.second_add` | 0 | 0 |
| `qrprelu.layer1.2.branch_positive` | 0 | 0 |
| `qrprelu.layer1.2.inner_add` | 0 | 0 |
| `qrprelu.layer1.2.negative_branch` | 0 | 0 |
| `qrprelu.layer1.2.output` | 0 | 0 |
| `h1.layer1.2.qrprelu_output` | 0 | 0 |
| `layer1.2.stage_output` | 0 | 0 |
| `binary.layer2.0.conv1_signed_acc` | 0 | 0 |
| `affine.layer2.0.first_affine.shifted_term` | 0 | 0 |
| `affine.layer2.0.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer2.0.first_affine.add_output` | 0 | 0 |
| `h1.layer2.0.first_affine` | 0 | 0 |
| `residual.layer2.0.stage_shortcut_option_a` | 0 | 0 |
| `h1.layer2.0.stage_shortcut` | 0 | 0 |
| `residual.layer2.0.x1.align_a` | 0 | 0 |
| `residual.layer2.0.x1.align_b` | 0 | 0 |
| `residual.layer2.0.x1.add_output` | 0 | 0 |
| `h1.layer2.0.x1` | 0 | 0 |
| `h1.layer2.0.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer2.0.conv2_signed_acc` | 0 | 0 |
| `affine.layer2.0.second_affine.shifted_term` | 0 | 0 |
| `affine.layer2.0.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer2.0.second_affine.add_output` | 0 | 0 |
| `h1.layer2.0.second_affine` | 0 | 0 |
| `residual.layer2.0.second_add.align_a` | 0 | 0 |
| `residual.layer2.0.second_add.align_b` | 0 | 0 |
| `residual.layer2.0.second_add.add_output` | 0 | 0 |
| `h1.layer2.0.second_add` | 0 | 0 |
| `qrprelu.layer2.0.branch_positive` | 0 | 0 |
| `qrprelu.layer2.0.inner_add` | 0 | 0 |
| `qrprelu.layer2.0.negative_branch` | 0 | 0 |
| `qrprelu.layer2.0.output` | 0 | 0 |
| `h1.layer2.0.qrprelu_output` | 0 | 0 |
| `layer2.0.stage_output` | 0 | 0 |
| `binary.layer2.1.conv1_signed_acc` | 0 | 0 |
| `affine.layer2.1.first_affine.shifted_term` | 0 | 0 |
| `affine.layer2.1.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer2.1.first_affine.add_output` | 0 | 0 |
| `h1.layer2.1.first_affine` | 0 | 0 |
| `residual.layer2.1.x1.align_a` | 0 | 0 |
| `residual.layer2.1.x1.align_b` | 0 | 0 |
| `residual.layer2.1.x1.add_output` | 0 | 0 |
| `h1.layer2.1.x1` | 0 | 0 |
| `h1.layer2.1.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer2.1.conv2_signed_acc` | 0 | 0 |
| `affine.layer2.1.second_affine.shifted_term` | 0 | 0 |
| `affine.layer2.1.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer2.1.second_affine.add_output` | 0 | 0 |
| `h1.layer2.1.second_affine` | 0 | 0 |
| `residual.layer2.1.second_add.align_a` | 0 | 0 |
| `residual.layer2.1.second_add.align_b` | 0 | 0 |
| `residual.layer2.1.second_add.add_output` | 0 | 0 |
| `h1.layer2.1.second_add` | 0 | 0 |
| `qrprelu.layer2.1.branch_positive` | 0 | 0 |
| `qrprelu.layer2.1.inner_add` | 0 | 0 |
| `qrprelu.layer2.1.negative_branch` | 0 | 0 |
| `qrprelu.layer2.1.output` | 0 | 0 |
| `h1.layer2.1.qrprelu_output` | 0 | 0 |
| `layer2.1.stage_output` | 0 | 0 |
| `binary.layer2.2.conv1_signed_acc` | 0 | 0 |
| `affine.layer2.2.first_affine.shifted_term` | 0 | 0 |
| `affine.layer2.2.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer2.2.first_affine.add_output` | 0 | 0 |
| `h1.layer2.2.first_affine` | 0 | 0 |
| `residual.layer2.2.x1.align_a` | 0 | 0 |
| `residual.layer2.2.x1.align_b` | 0 | 0 |
| `residual.layer2.2.x1.add_output` | 0 | 0 |
| `h1.layer2.2.x1` | 0 | 0 |
| `h1.layer2.2.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer2.2.conv2_signed_acc` | 0 | 0 |
| `affine.layer2.2.second_affine.shifted_term` | 0 | 0 |
| `affine.layer2.2.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer2.2.second_affine.add_output` | 0 | 0 |
| `h1.layer2.2.second_affine` | 0 | 0 |
| `residual.layer2.2.second_add.align_a` | 0 | 0 |
| `residual.layer2.2.second_add.align_b` | 0 | 0 |
| `residual.layer2.2.second_add.add_output` | 0 | 0 |
| `h1.layer2.2.second_add` | 0 | 0 |
| `qrprelu.layer2.2.branch_positive` | 0 | 0 |
| `qrprelu.layer2.2.inner_add` | 0 | 0 |
| `qrprelu.layer2.2.negative_branch` | 0 | 0 |
| `qrprelu.layer2.2.output` | 0 | 0 |
| `h1.layer2.2.qrprelu_output` | 0 | 0 |
| `layer2.2.stage_output` | 0 | 0 |
| `binary.layer3.0.conv1_signed_acc` | 0 | 0 |
| `affine.layer3.0.first_affine.shifted_term` | 0 | 0 |
| `affine.layer3.0.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer3.0.first_affine.add_output` | 0 | 0 |
| `h1.layer3.0.first_affine` | 0 | 0 |
| `residual.layer3.0.stage_shortcut_option_a` | 0 | 0 |
| `h1.layer3.0.stage_shortcut` | 0 | 0 |
| `residual.layer3.0.x1.align_a` | 0 | 0 |
| `residual.layer3.0.x1.align_b` | 0 | 0 |
| `residual.layer3.0.x1.add_output` | 0 | 0 |
| `h1.layer3.0.x1` | 0 | 0 |
| `h1.layer3.0.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer3.0.conv2_signed_acc` | 0 | 0 |
| `affine.layer3.0.second_affine.shifted_term` | 0 | 0 |
| `affine.layer3.0.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer3.0.second_affine.add_output` | 0 | 0 |
| `h1.layer3.0.second_affine` | 0 | 0 |
| `residual.layer3.0.second_add.align_a` | 0 | 0 |
| `residual.layer3.0.second_add.align_b` | 0 | 0 |
| `residual.layer3.0.second_add.add_output` | 0 | 0 |
| `h1.layer3.0.second_add` | 0 | 0 |
| `qrprelu.layer3.0.branch_positive` | 0 | 0 |
| `qrprelu.layer3.0.inner_add` | 0 | 0 |
| `qrprelu.layer3.0.negative_branch` | 0 | 0 |
| `qrprelu.layer3.0.output` | 0 | 0 |
| `h1.layer3.0.qrprelu_output` | 0 | 0 |
| `layer3.0.stage_output` | 0 | 0 |
| `binary.layer3.1.conv1_signed_acc` | 0 | 0 |
| `affine.layer3.1.first_affine.shifted_term` | 0 | 0 |
| `affine.layer3.1.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer3.1.first_affine.add_output` | 0 | 0 |
| `h1.layer3.1.first_affine` | 0 | 0 |
| `residual.layer3.1.x1.align_a` | 0 | 0 |
| `residual.layer3.1.x1.align_b` | 0 | 0 |
| `residual.layer3.1.x1.add_output` | 0 | 0 |
| `h1.layer3.1.x1` | 0 | 0 |
| `h1.layer3.1.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer3.1.conv2_signed_acc` | 0 | 0 |
| `affine.layer3.1.second_affine.shifted_term` | 0 | 0 |
| `affine.layer3.1.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer3.1.second_affine.add_output` | 0 | 0 |
| `h1.layer3.1.second_affine` | 0 | 0 |
| `residual.layer3.1.second_add.align_a` | 0 | 0 |
| `residual.layer3.1.second_add.align_b` | 0 | 0 |
| `residual.layer3.1.second_add.add_output` | 0 | 0 |
| `h1.layer3.1.second_add` | 0 | 0 |
| `qrprelu.layer3.1.branch_positive` | 0 | 0 |
| `qrprelu.layer3.1.inner_add` | 0 | 0 |
| `qrprelu.layer3.1.negative_branch` | 0 | 0 |
| `qrprelu.layer3.1.output` | 0 | 0 |
| `h1.layer3.1.qrprelu_output` | 0 | 0 |
| `layer3.1.stage_output` | 0 | 0 |
| `binary.layer3.2.conv1_signed_acc` | 0 | 0 |
| `affine.layer3.2.first_affine.shifted_term` | 0 | 0 |
| `affine.layer3.2.first_affine.bias_aligned` | 0 | 0 |
| `affine.layer3.2.first_affine.add_output` | 0 | 0 |
| `h1.layer3.2.first_affine` | 0 | 0 |
| `residual.layer3.2.x1.align_a` | 0 | 0 |
| `residual.layer3.2.x1.align_b` | 0 | 0 |
| `residual.layer3.2.x1.add_output` | 0 | 0 |
| `h1.layer3.2.x1` | 0 | 0 |
| `h1.layer3.2.pre_bconv2_hardtanh` | 0 | 0 |
| `binary.layer3.2.conv2_signed_acc` | 0 | 0 |
| `affine.layer3.2.second_affine.shifted_term` | 0 | 0 |
| `affine.layer3.2.second_affine.bias_aligned` | 0 | 0 |
| `affine.layer3.2.second_affine.add_output` | 0 | 0 |
| `h1.layer3.2.second_affine` | 0 | 0 |
| `residual.layer3.2.second_add.align_a` | 0 | 0 |
| `residual.layer3.2.second_add.align_b` | 0 | 0 |
| `residual.layer3.2.second_add.add_output` | 0 | 0 |
| `h1.layer3.2.second_add` | 0 | 0 |
| `qrprelu.layer3.2.branch_positive` | 0 | 0 |
| `qrprelu.layer3.2.inner_add` | 0 | 0 |
| `qrprelu.layer3.2.negative_branch` | 0 | 0 |
| `qrprelu.layer3.2.output` | 0 | 0 |
| `h1.layer3.2.qrprelu_output` | 0 | 0 |
| `layer3.2.stage_output` | 0 | 0 |
| `gap.sum` | 0 | 0 |
| `gap.output` | 0 | 0 |
| `affine.head.affine_output.shifted_term` | 0 | 0 |
| `affine.head.affine_output.bias_aligned` | 0 | 0 |
| `affine.head.affine_output.add_output` | 0 | 0 |
| `h1.head.affine_output` | 0 | 0 |
| `fc.shifted_products` | 0 | 0 |
| `fc.accumulator` | 0 | 0 |
| `fc.bias_add` | 0 | 0 |
| `final_logits` | 0 | 0 |

The official 10,000-image TEST is run only after 1, 10 and 100 image exact checks. The full run compares predictions and accuracy; 245-node tensor capture is the prefix debug contract. Known H1MP saturation is expected and separate from H2A exact-width overflow.

**Accuracy anchor discrepancy.** The frozen H2A-v2 report records 85.03% (8,503/10,000). The independent reference and the current CPU H2TraceAdapter both produce 85.01% (8,501/10,000), with zero prediction mismatch. A second evaluation using the official `CIFAR10` `ToTensor()` loader, batch size 128 and repository `evaluate()` also produced 85.01%; raw pixels, labels and `ToTensor()` values were checked equal. The cause of the 2-image difference from the historical frozen result is not established; the historical run may have used a different device or numerical environment. We did not alter any arithmetic, parameters, or frozen results to force 85.03%. The 85.03% acceptance criterion therefore remains unmet in this workspace run.

No RTL or synthesis was started.
