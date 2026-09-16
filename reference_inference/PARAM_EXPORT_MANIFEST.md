# Parameter export manifest

Frozen R8B-H2A-v2 Safe Profile. All inference arrays are in `params_export/inference_params.npz`; scale and width metadata are in `model_config.json`.

Checkpoint: `experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt`

```json
{
  "exported_tensors": 136,
  "exported_scalar_values": 286650,
  "binary_weight_values": 281088,
  "affine_effective_values": 2256,
  "qrp_effective_values": 1008,
  "fc_effective_values": 1290
}
```

| Export tensor | Original state_dict key / source | Shape | dtype | Export file | Hardware meaning |
|---|---|---|---|---|---|
| `conv1.weight_sign` | `conv1.weight` | `[16, 96, 3, 3]` | `int8` | `params_export/inference_params.npz` | Stem binary kernel: -1/0/+1; 0 bit means -1, 1 bit means +1 when active |
| `layer1.0.conv1.weight_sign` | `layer1.0.conv1.weight, layer1.0.conv1.tau` | `[16, 16, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.0.conv2.weight_sign` | `layer1.0.conv2.weight, layer1.0.conv2.tau` | `[16, 16, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.1.conv1.weight_sign` | `layer1.1.conv1.weight, layer1.1.conv1.tau` | `[16, 16, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.1.conv2.weight_sign` | `layer1.1.conv2.weight, layer1.1.conv2.tau` | `[16, 16, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.2.conv1.weight_sign` | `layer1.2.conv1.weight, layer1.2.conv1.tau` | `[16, 16, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer1.2.conv2.weight_sign` | `layer1.2.conv2.weight, layer1.2.conv2.tau` | `[16, 16, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.0.conv1.weight_sign` | `layer2.0.conv1.weight, layer2.0.conv1.tau` | `[32, 16, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.0.conv2.weight_sign` | `layer2.0.conv2.weight, layer2.0.conv2.tau` | `[32, 32, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.1.conv1.weight_sign` | `layer2.1.conv1.weight, layer2.1.conv1.tau` | `[32, 32, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.1.conv2.weight_sign` | `layer2.1.conv2.weight, layer2.1.conv2.tau` | `[32, 32, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.2.conv1.weight_sign` | `layer2.2.conv1.weight, layer2.2.conv1.tau` | `[32, 32, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer2.2.conv2.weight_sign` | `layer2.2.conv2.weight, layer2.2.conv2.tau` | `[32, 32, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.0.conv1.weight_sign` | `layer3.0.conv1.weight, layer3.0.conv1.tau` | `[64, 32, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.0.conv2.weight_sign` | `layer3.0.conv2.weight, layer3.0.conv2.tau` | `[64, 64, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.1.conv1.weight_sign` | `layer3.1.conv1.weight, layer3.1.conv1.tau` | `[64, 64, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.1.conv2.weight_sign` | `layer3.1.conv2.weight, layer3.1.conv2.tau` | `[64, 64, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.2.conv1.weight_sign` | `layer3.2.conv1.weight, layer3.2.conv1.tau` | `[64, 64, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `layer3.2.conv2.weight_sign` | `layer3.2.conv2.weight, layer3.2.conv2.tau` | `[64, 64, 3, 3]` | `int8` | `params_export/inference_params.npz` | Eval-time ReCU clipped centered weight sign; alpha_mode=one |
| `stem_affine.sign` | `stem_affine.k_latent` | `[16]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `stem_affine.exponent` | `stem_affine.k_latent` | `[16]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `stem_affine.bias_q` | `stem_affine.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer1.0.aff1.sign` | `layer1.0.aff1.log2_abs_k, layer1.0.aff1.sign_k` | `[16]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer1.0.aff1.exponent` | `layer1.0.aff1.log2_abs_k, layer1.0.aff1.sign_k` | `[16]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer1.0.aff1.bias_q` | `layer1.0.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer1.0.aff2.sign` | `layer1.0.aff2.log2_abs_k, layer1.0.aff2.sign_k` | `[16]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer1.0.aff2.exponent` | `layer1.0.aff2.log2_abs_k, layer1.0.aff2.sign_k` | `[16]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer1.0.aff2.bias_q` | `layer1.0.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer1.1.aff1.sign` | `layer1.1.aff1.log2_abs_k, layer1.1.aff1.sign_k` | `[16]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer1.1.aff1.exponent` | `layer1.1.aff1.log2_abs_k, layer1.1.aff1.sign_k` | `[16]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer1.1.aff1.bias_q` | `layer1.1.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer1.1.aff2.sign` | `layer1.1.aff2.log2_abs_k, layer1.1.aff2.sign_k` | `[16]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer1.1.aff2.exponent` | `layer1.1.aff2.log2_abs_k, layer1.1.aff2.sign_k` | `[16]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer1.1.aff2.bias_q` | `layer1.1.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer1.2.aff1.sign` | `layer1.2.aff1.log2_abs_k, layer1.2.aff1.sign_k` | `[16]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer1.2.aff1.exponent` | `layer1.2.aff1.log2_abs_k, layer1.2.aff1.sign_k` | `[16]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer1.2.aff1.bias_q` | `layer1.2.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer1.2.aff2.sign` | `layer1.2.aff2.log2_abs_k, layer1.2.aff2.sign_k` | `[16]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer1.2.aff2.exponent` | `layer1.2.aff2.log2_abs_k, layer1.2.aff2.sign_k` | `[16]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer1.2.aff2.bias_q` | `layer1.2.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[16]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer2.0.aff1.sign` | `layer2.0.aff1.log2_abs_k, layer2.0.aff1.sign_k` | `[32]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer2.0.aff1.exponent` | `layer2.0.aff1.log2_abs_k, layer2.0.aff1.sign_k` | `[32]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer2.0.aff1.bias_q` | `layer2.0.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer2.0.aff2.sign` | `layer2.0.aff2.log2_abs_k, layer2.0.aff2.sign_k` | `[32]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer2.0.aff2.exponent` | `layer2.0.aff2.log2_abs_k, layer2.0.aff2.sign_k` | `[32]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer2.0.aff2.bias_q` | `layer2.0.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer2.1.aff1.sign` | `layer2.1.aff1.log2_abs_k, layer2.1.aff1.sign_k` | `[32]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer2.1.aff1.exponent` | `layer2.1.aff1.log2_abs_k, layer2.1.aff1.sign_k` | `[32]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer2.1.aff1.bias_q` | `layer2.1.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer2.1.aff2.sign` | `layer2.1.aff2.log2_abs_k, layer2.1.aff2.sign_k` | `[32]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer2.1.aff2.exponent` | `layer2.1.aff2.log2_abs_k, layer2.1.aff2.sign_k` | `[32]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer2.1.aff2.bias_q` | `layer2.1.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer2.2.aff1.sign` | `layer2.2.aff1.log2_abs_k, layer2.2.aff1.sign_k` | `[32]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer2.2.aff1.exponent` | `layer2.2.aff1.log2_abs_k, layer2.2.aff1.sign_k` | `[32]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer2.2.aff1.bias_q` | `layer2.2.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer2.2.aff2.sign` | `layer2.2.aff2.log2_abs_k, layer2.2.aff2.sign_k` | `[32]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer2.2.aff2.exponent` | `layer2.2.aff2.log2_abs_k, layer2.2.aff2.sign_k` | `[32]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer2.2.aff2.bias_q` | `layer2.2.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[32]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer3.0.aff1.sign` | `layer3.0.aff1.log2_abs_k, layer3.0.aff1.sign_k` | `[64]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer3.0.aff1.exponent` | `layer3.0.aff1.log2_abs_k, layer3.0.aff1.sign_k` | `[64]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer3.0.aff1.bias_q` | `layer3.0.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer3.0.aff2.sign` | `layer3.0.aff2.log2_abs_k, layer3.0.aff2.sign_k` | `[64]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer3.0.aff2.exponent` | `layer3.0.aff2.log2_abs_k, layer3.0.aff2.sign_k` | `[64]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer3.0.aff2.bias_q` | `layer3.0.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer3.1.aff1.sign` | `layer3.1.aff1.log2_abs_k, layer3.1.aff1.sign_k` | `[64]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer3.1.aff1.exponent` | `layer3.1.aff1.log2_abs_k, layer3.1.aff1.sign_k` | `[64]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer3.1.aff1.bias_q` | `layer3.1.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer3.1.aff2.sign` | `layer3.1.aff2.log2_abs_k, layer3.1.aff2.sign_k` | `[64]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer3.1.aff2.exponent` | `layer3.1.aff2.log2_abs_k, layer3.1.aff2.sign_k` | `[64]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer3.1.aff2.bias_q` | `layer3.1.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer3.2.aff1.sign` | `layer3.2.aff1.log2_abs_k, layer3.2.aff1.sign_k` | `[64]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer3.2.aff1.exponent` | `layer3.2.aff1.log2_abs_k, layer3.2.aff1.sign_k` | `[64]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer3.2.aff1.bias_q` | `layer3.2.aff1.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer3.2.aff2.sign` | `layer3.2.aff2.log2_abs_k, layer3.2.aff2.sign_k` | `[64]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `layer3.2.aff2.exponent` | `layer3.2.aff2.log2_abs_k, layer3.2.aff2.sign_k` | `[64]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `layer3.2.aff2.bias_q` | `layer3.2.aff2.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `head_affine.sign` | `head_affine.k_latent` | `[64]` | `int8` | `params_export/inference_params.npz` | Signed-pow2 K sign |
| `head_affine.exponent` | `head_affine.k_latent` | `[64]` | `int16` | `params_export/inference_params.npz` | Signed-pow2 K power-of-two exponent |
| `head_affine.bias_q` | `head_affine.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[64]` | `int8` | `params_export/inference_params.npz` | INT6 bias at per-module scale in model_config.json |
| `layer1.0.post_act.a_raw` | `layer1.0.post_act.a` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.0.post_act.xi1_raw` | `layer1.0.post_act.xi1` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.0.post_act.xi2_raw` | `layer1.0.post_act.xi2` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.0.qrprelu_output.slope_exponent` | `layer1.0.post_act.a` | `[16]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer1.0.qrprelu_output.xi1_q` | `layer1.0.post_act.xi1` | `[16]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer1.0.qrprelu_output.xi2_q` | `layer1.0.post_act.xi2` | `[16]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer1.1.post_act.a_raw` | `layer1.1.post_act.a` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.1.post_act.xi1_raw` | `layer1.1.post_act.xi1` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.1.post_act.xi2_raw` | `layer1.1.post_act.xi2` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.1.qrprelu_output.slope_exponent` | `layer1.1.post_act.a` | `[16]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer1.1.qrprelu_output.xi1_q` | `layer1.1.post_act.xi1` | `[16]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer1.1.qrprelu_output.xi2_q` | `layer1.1.post_act.xi2` | `[16]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer1.2.post_act.a_raw` | `layer1.2.post_act.a` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.2.post_act.xi1_raw` | `layer1.2.post_act.xi1` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.2.post_act.xi2_raw` | `layer1.2.post_act.xi2` | `[16]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer1.2.qrprelu_output.slope_exponent` | `layer1.2.post_act.a` | `[16]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer1.2.qrprelu_output.xi1_q` | `layer1.2.post_act.xi1` | `[16]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer1.2.qrprelu_output.xi2_q` | `layer1.2.post_act.xi2` | `[16]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer2.0.post_act.a_raw` | `layer2.0.post_act.a` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.0.post_act.xi1_raw` | `layer2.0.post_act.xi1` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.0.post_act.xi2_raw` | `layer2.0.post_act.xi2` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.0.qrprelu_output.slope_exponent` | `layer2.0.post_act.a` | `[32]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer2.0.qrprelu_output.xi1_q` | `layer2.0.post_act.xi1` | `[32]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer2.0.qrprelu_output.xi2_q` | `layer2.0.post_act.xi2` | `[32]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer2.1.post_act.a_raw` | `layer2.1.post_act.a` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.1.post_act.xi1_raw` | `layer2.1.post_act.xi1` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.1.post_act.xi2_raw` | `layer2.1.post_act.xi2` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.1.qrprelu_output.slope_exponent` | `layer2.1.post_act.a` | `[32]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer2.1.qrprelu_output.xi1_q` | `layer2.1.post_act.xi1` | `[32]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer2.1.qrprelu_output.xi2_q` | `layer2.1.post_act.xi2` | `[32]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer2.2.post_act.a_raw` | `layer2.2.post_act.a` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.2.post_act.xi1_raw` | `layer2.2.post_act.xi1` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.2.post_act.xi2_raw` | `layer2.2.post_act.xi2` | `[32]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer2.2.qrprelu_output.slope_exponent` | `layer2.2.post_act.a` | `[32]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer2.2.qrprelu_output.xi1_q` | `layer2.2.post_act.xi1` | `[32]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer2.2.qrprelu_output.xi2_q` | `layer2.2.post_act.xi2` | `[32]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer3.0.post_act.a_raw` | `layer3.0.post_act.a` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.0.post_act.xi1_raw` | `layer3.0.post_act.xi1` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.0.post_act.xi2_raw` | `layer3.0.post_act.xi2` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.0.qrprelu_output.slope_exponent` | `layer3.0.post_act.a` | `[64]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer3.0.qrprelu_output.xi1_q` | `layer3.0.post_act.xi1` | `[64]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer3.0.qrprelu_output.xi2_q` | `layer3.0.post_act.xi2` | `[64]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer3.1.post_act.a_raw` | `layer3.1.post_act.a` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.1.post_act.xi1_raw` | `layer3.1.post_act.xi1` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.1.post_act.xi2_raw` | `layer3.1.post_act.xi2` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.1.qrprelu_output.slope_exponent` | `layer3.1.post_act.a` | `[64]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer3.1.qrprelu_output.xi1_q` | `layer3.1.post_act.xi1` | `[64]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer3.1.qrprelu_output.xi2_q` | `layer3.1.post_act.xi2` | `[64]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer3.2.post_act.a_raw` | `layer3.2.post_act.a` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.2.post_act.xi1_raw` | `layer3.2.post_act.xi1` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.2.post_act.xi2_raw` | `layer3.2.post_act.xi2` | `[64]` | `float32` | `params_export/inference_params.npz` | Checkpoint QRP parameter before H2 finite-width integer interpretation |
| `layer3.2.qrprelu_output.slope_exponent` | `layer3.2.post_act.a` | `[64]` | `int16` | `params_export/inference_params.npz` | Negative-branch pow2 exponent |
| `layer3.2.qrprelu_output.xi1_q` | `layer3.2.post_act.xi1` | `[64]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `layer3.2.qrprelu_output.xi2_q` | `layer3.2.post_act.xi2` | `[64]` | `int64` | `params_export/inference_params.npz` | QRP offset integer at H2 param_shift |
| `linear.weight_sign` | `linear.weight` | `[10, 64]` | `int8` | `params_export/inference_params.npz` | 64-to-10 signed-pow2 FC term sign |
| `linear.weight_exponent` | `linear.weight` | `[10, 64]` | `int16` | `params_export/inference_params.npz` | 64-to-10 signed-pow2 FC term exponent |
| `linear.bias_q` | `linear.bias + H0_INTEGER_BIAS_SWEEP_RESULTS.json` | `[10]` | `int8` | `params_export/inference_params.npz` | INT6 FC bias |

The checkpoint's `conv*.alpha` tensors are retained for compatibility but unused because the frozen R4 blocks set `alpha_mode=one`. `conv*.tau` is consumed only during effective weight-sign export. Bias source is H0 INT6 JSON, which the formal loader applies after loading the checkpoint.
