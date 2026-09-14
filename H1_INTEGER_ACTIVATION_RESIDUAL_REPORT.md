# H1 Integer Activation / Residual Report

> R8B-H0 zero-shot multi-bit activation/residual state sweep. No fine-tuning was performed.

## A. Source R8B / H0 verification

- Source checkpoint: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r8b\recu_r8b_r7_pow2_fc_20260912_225827\best.pt`
- Stored R8B best: **85.40%** @ epoch **100**
- R8B actual reload before H0: **85.40%**
- H0 INT6 baseline after fixed q+shift reconstruction: **85.13%** (target 85.13% +/- 0.08 pp)
- Adapter disabled-quantization equivalence max logit error: `0`; PASS.
- R8B invariants: PASS; binary stem, signed-pow2 affine/FC weights, FC bias present, no BN2, 672 alpha parameters frozen.

## B. H0 INT6 bias reconstruction

- Source JSON: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H0_INTEGER_BIAS_SWEEP_RESULTS.json`
- Fixed targets: 21 tensors / 762 scalar biases.
- For every layer, the stored H0 `q_values` and `shift` were applied as `Bhat = q * 2^(-shift)`.
- Bias bit-width was not searched again; H0 bias remained fixed during every H1 width evaluation.

| Module | Class | Shift | q range | Parameters |
|---|---|---:|---:|---:|
| `stem_affine` | `StemFusedPow2Affine2d` | 6 | [-32, 31] | 16 |
| `layer1.0.aff1` | `FusedAffine2d` | 5 | [-32, 31] | 16 |
| `layer1.0.aff2` | `FusedAffine2d` | 5 | [-32, 31] | 16 |
| `layer1.1.aff1` | `FusedAffine2d` | 5 | [-32, 31] | 16 |
| `layer1.1.aff2` | `FusedAffine2d` | 4 | [-32, 31] | 16 |
| `layer1.2.aff1` | `FusedAffine2d` | 6 | [-32, 31] | 16 |
| `layer1.2.aff2` | `FusedAffine2d` | 4 | [-32, 31] | 16 |
| `layer2.0.aff1` | `FusedAffine2d` | 5 | [-32, 31] | 32 |
| `layer2.0.aff2` | `FusedAffine2d` | 5 | [-32, 31] | 32 |
| `layer2.1.aff1` | `FusedAffine2d` | 5 | [-32, 31] | 32 |
| `layer2.1.aff2` | `FusedAffine2d` | 5 | [-32, 31] | 32 |
| `layer2.2.aff1` | `FusedAffine2d` | 5 | [-32, 31] | 32 |
| `layer2.2.aff2` | `FusedAffine2d` | 5 | [-32, 31] | 32 |
| `layer3.0.aff1` | `FusedAffine2d` | 5 | [-32, 31] | 64 |
| `layer3.0.aff2` | `FusedAffine2d` | 5 | [-32, 31] | 64 |
| `layer3.1.aff1` | `FusedAffine2d` | 6 | [-32, 31] | 64 |
| `layer3.1.aff2` | `FusedAffine2d` | 5 | [-32, 31] | 64 |
| `layer3.2.aff1` | `FusedAffine2d` | 6 | [-32, 31] | 64 |
| `layer3.2.aff2` | `FusedAffine2d` | 5 | [-32, 31] | 64 |
| `head_affine` | `HeadFusedPow2Affine1d` | 2 | [-32, 31] | 64 |
| `linear` | `Pow2LinearSTE` | 6 | [-32, 31] | 10 |

## C. H1 quantization nodes

- Total quant nodes: **58**.
- The adapter uses controlled forward execution; it does not rely on generic module hooks.
- Binary-convolution accumulators and GAP accumulation are explicitly not quantized.

| Node | Tensor role | Description |
|---|---|---|
| `stem.post_hardtanh` | `stem_state` | stem affine + Hardtanh output passed to the first R4 block |
| `layer1.0.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer1.0.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer1.0.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer1.0.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer1.0.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer1.0.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer1.1.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer1.1.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer1.1.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer1.1.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer1.1.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer1.1.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer1.2.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer1.2.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer1.2.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer1.2.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer1.2.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer1.2.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer2.0.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer2.0.stage_shortcut` | `stage_transition_shortcut` | Option-A stage-transition shortcut output before the first residual add |
| `layer2.0.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer2.0.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer2.0.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer2.0.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer2.0.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer2.1.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer2.1.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer2.1.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer2.1.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer2.1.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer2.1.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer2.2.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer2.2.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer2.2.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer2.2.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer2.2.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer2.2.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer3.0.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer3.0.stage_shortcut` | `stage_transition_shortcut` | Option-A stage-transition shortcut output before the first residual add |
| `layer3.0.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer3.0.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer3.0.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer3.0.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer3.0.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer3.1.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer3.1.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer3.1.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer3.1.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer3.1.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer3.1.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `layer3.2.first_affine` | `residual_branch_a` | first BConv + fused affine branch before the first residual add |
| `layer3.2.x1` | `first_residual_add_output` | first residual-add output saved as x1 |
| `layer3.2.pre_bconv2_hardtanh` | `pre_second_bconv_state` | Hardtanh(x1) magnitude state feeding the second BConv |
| `layer3.2.second_affine` | `residual_branch_a` | second BConv + fused affine branch before the second residual add |
| `layer3.2.second_add` | `second_residual_add_output` | second residual-add output before QRPReLU |
| `layer3.2.qrprelu_output` | `block_output_state` | QRPReLU output and block output state |
| `head.affine_output` | `head_fc_input` | head signed-pow2 affine output and final FC input |

## D. Calibration methodology

- Dataset: CIFAR-10 TRAIN only; samples used: **10000**.
- Test data was not used to choose any shift. Test data was used only for final accuracy and saturation statistics.
- Shift search: `-16..24`.
- Selection: largest integer shift with zero calibration saturation based on observed absmax.
- Right-shift alignment rounding: nearest integer, ties away from zero.

## E. Train calibration ranges

| Node | Count | Min | Max | Absmax |
|---|---:|---:|---:|---:|
| `stem.post_hardtanh` | 163840000 | -1 | 1 | 1 |
| `layer1.0.first_affine` | 163840000 | -2.09375 | 2.40625 | 2.40625 |
| `layer1.0.x1` | 163840000 | -3.09375 | 3.34375 | 3.34375 |
| `layer1.0.pre_bconv2_hardtanh` | 163840000 | -1 | 1 | 1 |
| `layer1.0.second_affine` | 163840000 | -3.03125 | 2.6875 | 3.03125 |
| `layer1.0.second_add` | 163840000 | -4.40625 | 5.15625 | 5.15625 |
| `layer1.0.qrprelu_output` | 163840000 | -3.4093032 | 5.15625 | 5.15625 |
| `layer1.1.first_affine` | 163840000 | -2.4375 | 2.71875 | 2.71875 |
| `layer1.1.x1` | 163840000 | -4.4959645 | 5.53125 | 5.53125 |
| `layer1.1.pre_bconv2_hardtanh` | 163840000 | -1 | 1 | 1 |
| `layer1.1.second_affine` | 163840000 | -3.8125 | 2.6875 | 3.8125 |
| `layer1.1.second_add` | 163840000 | -6.5824909 | 6.78125 | 6.78125 |
| `layer1.1.qrprelu_output` | 163840000 | -6.1976275 | 6.78125 | 6.78125 |
| `layer1.2.first_affine` | 163840000 | -2.6875 | 1.859375 | 2.6875 |
| `layer1.2.x1` | 163840000 | -7.8851275 | 7.09375 | 7.8851275 |
| `layer1.2.pre_bconv2_hardtanh` | 163840000 | -1 | 1 | 1 |
| `layer1.2.second_affine` | 163840000 | -3.75 | 2.125 | 3.75 |
| `layer1.2.second_add` | 163840000 | -10.557003 | 6.40625 | 10.557003 |
| `layer1.2.qrprelu_output` | 163840000 | -3.6282945 | 6.40625 | 6.40625 |
| `layer2.0.first_affine` | 81920000 | -2.78125 | 3.125 | 3.125 |
| `layer2.0.stage_shortcut` | 81920000 | -3.526732 | 6.40625 | 6.40625 |
| `layer2.0.x1` | 81920000 | -3.0537376 | 7.53125 | 7.53125 |
| `layer2.0.pre_bconv2_hardtanh` | 81920000 | -1 | 1 | 1 |
| `layer2.0.second_affine` | 81920000 | -4.875 | 3.59375 | 4.875 |
| `layer2.0.second_add` | 81920000 | -7.3174095 | 8.03125 | 8.03125 |
| `layer2.0.qrprelu_output` | 81920000 | -6.0453043 | 8.03125 | 8.03125 |
| `layer2.1.first_affine` | 81920000 | -2.8125 | 3.1875 | 3.1875 |
| `layer2.1.x1` | 81920000 | -6.8890543 | 8.453125 | 8.453125 |
| `layer2.1.pre_bconv2_hardtanh` | 81920000 | -1 | 1 | 1 |
| `layer2.1.second_affine` | 81920000 | -4.4375 | 2.75 | 4.4375 |
| `layer2.1.second_add` | 81920000 | -8.7328043 | 8.078125 | 8.7328043 |
| `layer2.1.qrprelu_output` | 81920000 | -4.3844929 | 8.078125 | 8.078125 |
| `layer2.2.first_affine` | 81920000 | -3.53125 | 2.625 | 3.53125 |
| `layer2.2.x1` | 81920000 | -5.7362375 | 7.859375 | 7.859375 |
| `layer2.2.pre_bconv2_hardtanh` | 81920000 | -1 | 1 | 1 |
| `layer2.2.second_affine` | 81920000 | -3.65625 | 3.34375 | 3.65625 |
| `layer2.2.second_add` | 81920000 | -8.6112375 | 7.765625 | 8.6112375 |
| `layer2.2.qrprelu_output` | 81920000 | -4.0752516 | 7.765625 | 7.765625 |
| `layer3.0.first_affine` | 40960000 | -3.96875 | 3.8125 | 3.96875 |
| `layer3.0.stage_shortcut` | 40960000 | -3.7939327 | 7.765625 | 7.765625 |
| `layer3.0.x1` | 40960000 | -4.7158079 | 8.296875 | 8.296875 |
| `layer3.0.pre_bconv2_hardtanh` | 40960000 | -1 | 1 | 1 |
| `layer3.0.second_affine` | 40960000 | -4.4375 | 3.6875 | 4.4375 |
| `layer3.0.second_add` | 40960000 | -7.3820057 | 7.984375 | 7.984375 |
| `layer3.0.qrprelu_output` | 40960000 | -3.5393662 | 7.984375 | 7.984375 |
| `layer3.1.first_affine` | 40960000 | -5.328125 | 4 | 5.328125 |
| `layer3.1.x1` | 40960000 | -6.3330278 | 9.203125 | 9.203125 |
| `layer3.1.pre_bconv2_hardtanh` | 40960000 | -1 | 1 | 1 |
| `layer3.1.second_affine` | 40960000 | -6.71875 | 5.21875 | 6.71875 |
| `layer3.1.second_add` | 40960000 | -9.999279 | 8.609375 | 9.999279 |
| `layer3.1.qrprelu_output` | 40960000 | -4.8267488 | 8.609375 | 8.609375 |
| `layer3.2.first_affine` | 40960000 | -3.984375 | 4.578125 | 4.578125 |
| `layer3.2.x1` | 40960000 | -7.0705276 | 10.015625 | 10.015625 |
| `layer3.2.pre_bconv2_hardtanh` | 40960000 | -1 | 1 | 1 |
| `layer3.2.second_affine` | 40960000 | -5.40625 | 5.125 | 5.40625 |
| `layer3.2.second_add` | 40960000 | -10.508028 | 12.931768 | 12.931768 |
| `layer3.2.qrprelu_output` | 40960000 | -0.022504814 | 12.931768 | 12.931768 |
| `head.affine_output` | 640000 | -2.6410682 | 10.266491 | 10.266491 |

## F. Per-width shifts

### INT8

| Node | Shift | Range | Calibration zero-sat |
|---|---:|---:|---:|
| `stem.post_hardtanh` | 6 | [-128, 127] | True |
| `layer1.0.first_affine` | 5 | [-128, 127] | True |
| `layer1.0.x1` | 5 | [-128, 127] | True |
| `layer1.0.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer1.0.second_affine` | 5 | [-128, 127] | True |
| `layer1.0.second_add` | 4 | [-128, 127] | True |
| `layer1.0.qrprelu_output` | 4 | [-128, 127] | True |
| `layer1.1.first_affine` | 5 | [-128, 127] | True |
| `layer1.1.x1` | 4 | [-128, 127] | True |
| `layer1.1.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer1.1.second_affine` | 5 | [-128, 127] | True |
| `layer1.1.second_add` | 4 | [-128, 127] | True |
| `layer1.1.qrprelu_output` | 4 | [-128, 127] | True |
| `layer1.2.first_affine` | 5 | [-128, 127] | True |
| `layer1.2.x1` | 4 | [-128, 127] | True |
| `layer1.2.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer1.2.second_affine` | 5 | [-128, 127] | True |
| `layer1.2.second_add` | 3 | [-128, 127] | True |
| `layer1.2.qrprelu_output` | 4 | [-128, 127] | True |
| `layer2.0.first_affine` | 5 | [-128, 127] | True |
| `layer2.0.stage_shortcut` | 4 | [-128, 127] | True |
| `layer2.0.x1` | 4 | [-128, 127] | True |
| `layer2.0.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer2.0.second_affine` | 4 | [-128, 127] | True |
| `layer2.0.second_add` | 3 | [-128, 127] | True |
| `layer2.0.qrprelu_output` | 3 | [-128, 127] | True |
| `layer2.1.first_affine` | 5 | [-128, 127] | True |
| `layer2.1.x1` | 3 | [-128, 127] | True |
| `layer2.1.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer2.1.second_affine` | 4 | [-128, 127] | True |
| `layer2.1.second_add` | 3 | [-128, 127] | True |
| `layer2.1.qrprelu_output` | 3 | [-128, 127] | True |
| `layer2.2.first_affine` | 5 | [-128, 127] | True |
| `layer2.2.x1` | 4 | [-128, 127] | True |
| `layer2.2.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer2.2.second_affine` | 5 | [-128, 127] | True |
| `layer2.2.second_add` | 3 | [-128, 127] | True |
| `layer2.2.qrprelu_output` | 4 | [-128, 127] | True |
| `layer3.0.first_affine` | 5 | [-128, 127] | True |
| `layer3.0.stage_shortcut` | 4 | [-128, 127] | True |
| `layer3.0.x1` | 3 | [-128, 127] | True |
| `layer3.0.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer3.0.second_affine` | 4 | [-128, 127] | True |
| `layer3.0.second_add` | 3 | [-128, 127] | True |
| `layer3.0.qrprelu_output` | 3 | [-128, 127] | True |
| `layer3.1.first_affine` | 4 | [-128, 127] | True |
| `layer3.1.x1` | 3 | [-128, 127] | True |
| `layer3.1.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer3.1.second_affine` | 4 | [-128, 127] | True |
| `layer3.1.second_add` | 3 | [-128, 127] | True |
| `layer3.1.qrprelu_output` | 3 | [-128, 127] | True |
| `layer3.2.first_affine` | 4 | [-128, 127] | True |
| `layer3.2.x1` | 3 | [-128, 127] | True |
| `layer3.2.pre_bconv2_hardtanh` | 6 | [-128, 127] | True |
| `layer3.2.second_affine` | 4 | [-128, 127] | True |
| `layer3.2.second_add` | 3 | [-128, 127] | True |
| `layer3.2.qrprelu_output` | 3 | [-128, 127] | True |
| `head.affine_output` | 3 | [-128, 127] | True |

### INT7

| Node | Shift | Range | Calibration zero-sat |
|---|---:|---:|---:|
| `stem.post_hardtanh` | 5 | [-64, 63] | True |
| `layer1.0.first_affine` | 4 | [-64, 63] | True |
| `layer1.0.x1` | 4 | [-64, 63] | True |
| `layer1.0.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer1.0.second_affine` | 4 | [-64, 63] | True |
| `layer1.0.second_add` | 3 | [-64, 63] | True |
| `layer1.0.qrprelu_output` | 3 | [-64, 63] | True |
| `layer1.1.first_affine` | 4 | [-64, 63] | True |
| `layer1.1.x1` | 3 | [-64, 63] | True |
| `layer1.1.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer1.1.second_affine` | 4 | [-64, 63] | True |
| `layer1.1.second_add` | 3 | [-64, 63] | True |
| `layer1.1.qrprelu_output` | 3 | [-64, 63] | True |
| `layer1.2.first_affine` | 4 | [-64, 63] | True |
| `layer1.2.x1` | 2 | [-64, 63] | True |
| `layer1.2.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer1.2.second_affine` | 4 | [-64, 63] | True |
| `layer1.2.second_add` | 2 | [-64, 63] | True |
| `layer1.2.qrprelu_output` | 3 | [-64, 63] | True |
| `layer2.0.first_affine` | 4 | [-64, 63] | True |
| `layer2.0.stage_shortcut` | 3 | [-64, 63] | True |
| `layer2.0.x1` | 3 | [-64, 63] | True |
| `layer2.0.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer2.0.second_affine` | 3 | [-64, 63] | True |
| `layer2.0.second_add` | 2 | [-64, 63] | True |
| `layer2.0.qrprelu_output` | 2 | [-64, 63] | True |
| `layer2.1.first_affine` | 4 | [-64, 63] | True |
| `layer2.1.x1` | 2 | [-64, 63] | True |
| `layer2.1.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer2.1.second_affine` | 3 | [-64, 63] | True |
| `layer2.1.second_add` | 2 | [-64, 63] | True |
| `layer2.1.qrprelu_output` | 2 | [-64, 63] | True |
| `layer2.2.first_affine` | 4 | [-64, 63] | True |
| `layer2.2.x1` | 3 | [-64, 63] | True |
| `layer2.2.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer2.2.second_affine` | 4 | [-64, 63] | True |
| `layer2.2.second_add` | 2 | [-64, 63] | True |
| `layer2.2.qrprelu_output` | 3 | [-64, 63] | True |
| `layer3.0.first_affine` | 3 | [-64, 63] | True |
| `layer3.0.stage_shortcut` | 3 | [-64, 63] | True |
| `layer3.0.x1` | 2 | [-64, 63] | True |
| `layer3.0.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer3.0.second_affine` | 3 | [-64, 63] | True |
| `layer3.0.second_add` | 2 | [-64, 63] | True |
| `layer3.0.qrprelu_output` | 2 | [-64, 63] | True |
| `layer3.1.first_affine` | 3 | [-64, 63] | True |
| `layer3.1.x1` | 2 | [-64, 63] | True |
| `layer3.1.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer3.1.second_affine` | 3 | [-64, 63] | True |
| `layer3.1.second_add` | 2 | [-64, 63] | True |
| `layer3.1.qrprelu_output` | 2 | [-64, 63] | True |
| `layer3.2.first_affine` | 3 | [-64, 63] | True |
| `layer3.2.x1` | 2 | [-64, 63] | True |
| `layer3.2.pre_bconv2_hardtanh` | 5 | [-64, 63] | True |
| `layer3.2.second_affine` | 3 | [-64, 63] | True |
| `layer3.2.second_add` | 2 | [-64, 63] | True |
| `layer3.2.qrprelu_output` | 2 | [-64, 63] | True |
| `head.affine_output` | 2 | [-64, 63] | True |

### INT6

| Node | Shift | Range | Calibration zero-sat |
|---|---:|---:|---:|
| `stem.post_hardtanh` | 4 | [-32, 31] | True |
| `layer1.0.first_affine` | 3 | [-32, 31] | True |
| `layer1.0.x1` | 3 | [-32, 31] | True |
| `layer1.0.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer1.0.second_affine` | 3 | [-32, 31] | True |
| `layer1.0.second_add` | 2 | [-32, 31] | True |
| `layer1.0.qrprelu_output` | 2 | [-32, 31] | True |
| `layer1.1.first_affine` | 3 | [-32, 31] | True |
| `layer1.1.x1` | 2 | [-32, 31] | True |
| `layer1.1.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer1.1.second_affine` | 3 | [-32, 31] | True |
| `layer1.1.second_add` | 2 | [-32, 31] | True |
| `layer1.1.qrprelu_output` | 2 | [-32, 31] | True |
| `layer1.2.first_affine` | 3 | [-32, 31] | True |
| `layer1.2.x1` | 1 | [-32, 31] | True |
| `layer1.2.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer1.2.second_affine` | 3 | [-32, 31] | True |
| `layer1.2.second_add` | 1 | [-32, 31] | True |
| `layer1.2.qrprelu_output` | 2 | [-32, 31] | True |
| `layer2.0.first_affine` | 3 | [-32, 31] | True |
| `layer2.0.stage_shortcut` | 2 | [-32, 31] | True |
| `layer2.0.x1` | 2 | [-32, 31] | True |
| `layer2.0.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer2.0.second_affine` | 2 | [-32, 31] | True |
| `layer2.0.second_add` | 1 | [-32, 31] | True |
| `layer2.0.qrprelu_output` | 1 | [-32, 31] | True |
| `layer2.1.first_affine` | 3 | [-32, 31] | True |
| `layer2.1.x1` | 1 | [-32, 31] | True |
| `layer2.1.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer2.1.second_affine` | 2 | [-32, 31] | True |
| `layer2.1.second_add` | 1 | [-32, 31] | True |
| `layer2.1.qrprelu_output` | 1 | [-32, 31] | True |
| `layer2.2.first_affine` | 3 | [-32, 31] | True |
| `layer2.2.x1` | 1 | [-32, 31] | True |
| `layer2.2.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer2.2.second_affine` | 3 | [-32, 31] | True |
| `layer2.2.second_add` | 1 | [-32, 31] | True |
| `layer2.2.qrprelu_output` | 1 | [-32, 31] | True |
| `layer3.0.first_affine` | 2 | [-32, 31] | True |
| `layer3.0.stage_shortcut` | 1 | [-32, 31] | True |
| `layer3.0.x1` | 1 | [-32, 31] | True |
| `layer3.0.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer3.0.second_affine` | 2 | [-32, 31] | True |
| `layer3.0.second_add` | 1 | [-32, 31] | True |
| `layer3.0.qrprelu_output` | 1 | [-32, 31] | True |
| `layer3.1.first_affine` | 2 | [-32, 31] | True |
| `layer3.1.x1` | 1 | [-32, 31] | True |
| `layer3.1.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer3.1.second_affine` | 2 | [-32, 31] | True |
| `layer3.1.second_add` | 1 | [-32, 31] | True |
| `layer3.1.qrprelu_output` | 1 | [-32, 31] | True |
| `layer3.2.first_affine` | 2 | [-32, 31] | True |
| `layer3.2.x1` | 1 | [-32, 31] | True |
| `layer3.2.pre_bconv2_hardtanh` | 4 | [-32, 31] | True |
| `layer3.2.second_affine` | 2 | [-32, 31] | True |
| `layer3.2.second_add` | 1 | [-32, 31] | True |
| `layer3.2.qrprelu_output` | 1 | [-32, 31] | True |
| `head.affine_output` | 1 | [-32, 31] | True |

### INT5

| Node | Shift | Range | Calibration zero-sat |
|---|---:|---:|---:|
| `stem.post_hardtanh` | 3 | [-16, 15] | True |
| `layer1.0.first_affine` | 2 | [-16, 15] | True |
| `layer1.0.x1` | 2 | [-16, 15] | True |
| `layer1.0.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer1.0.second_affine` | 2 | [-16, 15] | True |
| `layer1.0.second_add` | 1 | [-16, 15] | True |
| `layer1.0.qrprelu_output` | 1 | [-16, 15] | True |
| `layer1.1.first_affine` | 2 | [-16, 15] | True |
| `layer1.1.x1` | 1 | [-16, 15] | True |
| `layer1.1.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer1.1.second_affine` | 1 | [-16, 15] | True |
| `layer1.1.second_add` | 1 | [-16, 15] | True |
| `layer1.1.qrprelu_output` | 1 | [-16, 15] | True |
| `layer1.2.first_affine` | 2 | [-16, 15] | True |
| `layer1.2.x1` | 0 | [-16, 15] | True |
| `layer1.2.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer1.2.second_affine` | 2 | [-16, 15] | True |
| `layer1.2.second_add` | 0 | [-16, 15] | True |
| `layer1.2.qrprelu_output` | 1 | [-16, 15] | True |
| `layer2.0.first_affine` | 2 | [-16, 15] | True |
| `layer2.0.stage_shortcut` | 1 | [-16, 15] | True |
| `layer2.0.x1` | 0 | [-16, 15] | True |
| `layer2.0.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer2.0.second_affine` | 1 | [-16, 15] | True |
| `layer2.0.second_add` | 0 | [-16, 15] | True |
| `layer2.0.qrprelu_output` | 0 | [-16, 15] | True |
| `layer2.1.first_affine` | 2 | [-16, 15] | True |
| `layer2.1.x1` | 0 | [-16, 15] | True |
| `layer2.1.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer2.1.second_affine` | 1 | [-16, 15] | True |
| `layer2.1.second_add` | 0 | [-16, 15] | True |
| `layer2.1.qrprelu_output` | 0 | [-16, 15] | True |
| `layer2.2.first_affine` | 2 | [-16, 15] | True |
| `layer2.2.x1` | 0 | [-16, 15] | True |
| `layer2.2.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer2.2.second_affine` | 2 | [-16, 15] | True |
| `layer2.2.second_add` | 0 | [-16, 15] | True |
| `layer2.2.qrprelu_output` | 0 | [-16, 15] | True |
| `layer3.0.first_affine` | 1 | [-16, 15] | True |
| `layer3.0.stage_shortcut` | 0 | [-16, 15] | True |
| `layer3.0.x1` | 0 | [-16, 15] | True |
| `layer3.0.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer3.0.second_affine` | 1 | [-16, 15] | True |
| `layer3.0.second_add` | 0 | [-16, 15] | True |
| `layer3.0.qrprelu_output` | 0 | [-16, 15] | True |
| `layer3.1.first_affine` | 1 | [-16, 15] | True |
| `layer3.1.x1` | 0 | [-16, 15] | True |
| `layer3.1.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer3.1.second_affine` | 1 | [-16, 15] | True |
| `layer3.1.second_add` | 0 | [-16, 15] | True |
| `layer3.1.qrprelu_output` | 0 | [-16, 15] | True |
| `layer3.2.first_affine` | 1 | [-16, 15] | True |
| `layer3.2.x1` | 0 | [-16, 15] | True |
| `layer3.2.pre_bconv2_hardtanh` | 3 | [-16, 15] | True |
| `layer3.2.second_affine` | 1 | [-16, 15] | True |
| `layer3.2.second_add` | 0 | [-16, 15] | True |
| `layer3.2.qrprelu_output` | 0 | [-16, 15] | True |
| `head.affine_output` | 0 | [-16, 15] | True |

### INT4

| Node | Shift | Range | Calibration zero-sat |
|---|---:|---:|---:|
| `stem.post_hardtanh` | 2 | [-8, 7] | True |
| `layer1.0.first_affine` | 1 | [-8, 7] | True |
| `layer1.0.x1` | 1 | [-8, 7] | True |
| `layer1.0.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer1.0.second_affine` | 1 | [-8, 7] | True |
| `layer1.0.second_add` | 0 | [-8, 7] | True |
| `layer1.0.qrprelu_output` | 0 | [-8, 7] | True |
| `layer1.1.first_affine` | 1 | [-8, 7] | True |
| `layer1.1.x1` | 0 | [-8, 7] | True |
| `layer1.1.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer1.1.second_affine` | 0 | [-8, 7] | True |
| `layer1.1.second_add` | 0 | [-8, 7] | True |
| `layer1.1.qrprelu_output` | 0 | [-8, 7] | True |
| `layer1.2.first_affine` | 1 | [-8, 7] | True |
| `layer1.2.x1` | -1 | [-8, 7] | True |
| `layer1.2.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer1.2.second_affine` | 0 | [-8, 7] | True |
| `layer1.2.second_add` | -1 | [-8, 7] | True |
| `layer1.2.qrprelu_output` | 0 | [-8, 7] | True |
| `layer2.0.first_affine` | 1 | [-8, 7] | True |
| `layer2.0.stage_shortcut` | 0 | [-8, 7] | True |
| `layer2.0.x1` | -1 | [-8, 7] | True |
| `layer2.0.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer2.0.second_affine` | 0 | [-8, 7] | True |
| `layer2.0.second_add` | -1 | [-8, 7] | True |
| `layer2.0.qrprelu_output` | -1 | [-8, 7] | True |
| `layer2.1.first_affine` | 1 | [-8, 7] | True |
| `layer2.1.x1` | -1 | [-8, 7] | True |
| `layer2.1.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer2.1.second_affine` | 0 | [-8, 7] | True |
| `layer2.1.second_add` | -1 | [-8, 7] | True |
| `layer2.1.qrprelu_output` | -1 | [-8, 7] | True |
| `layer2.2.first_affine` | 0 | [-8, 7] | True |
| `layer2.2.x1` | -1 | [-8, 7] | True |
| `layer2.2.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer2.2.second_affine` | 0 | [-8, 7] | True |
| `layer2.2.second_add` | -1 | [-8, 7] | True |
| `layer2.2.qrprelu_output` | -1 | [-8, 7] | True |
| `layer3.0.first_affine` | 0 | [-8, 7] | True |
| `layer3.0.stage_shortcut` | -1 | [-8, 7] | True |
| `layer3.0.x1` | -1 | [-8, 7] | True |
| `layer3.0.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer3.0.second_affine` | 0 | [-8, 7] | True |
| `layer3.0.second_add` | -1 | [-8, 7] | True |
| `layer3.0.qrprelu_output` | -1 | [-8, 7] | True |
| `layer3.1.first_affine` | 0 | [-8, 7] | True |
| `layer3.1.x1` | -1 | [-8, 7] | True |
| `layer3.1.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer3.1.second_affine` | 0 | [-8, 7] | True |
| `layer3.1.second_add` | -1 | [-8, 7] | True |
| `layer3.1.qrprelu_output` | -1 | [-8, 7] | True |
| `layer3.2.first_affine` | 0 | [-8, 7] | True |
| `layer3.2.x1` | -1 | [-8, 7] | True |
| `layer3.2.pre_bconv2_hardtanh` | 2 | [-8, 7] | True |
| `layer3.2.second_affine` | 0 | [-8, 7] | True |
| `layer3.2.second_add` | -1 | [-8, 7] | True |
| `layer3.2.qrprelu_output` | -1 | [-8, 7] | True |
| `head.affine_output` | -1 | [-8, 7] | True |

## G. Residual integer scale alignment

Every residual add aligns both integer branches to the common output shift. Positive deltas use integer left shifts; negative deltas use signed arithmetic right shifts with defined rounding. Output saturation is applied only at the designated W-bit output node.

### INT8

| Add node | Branch A s | Branch B s | Output s | Delta A/B | Test saturation | Equivalence |
|---|---:|---:|---:|---|---:|---|
| `layer1.0.x1` | 5 | 6 | 5 | 0/-1 | 0 | True |
| `layer1.0.second_add` | 5 | 5 | 4 | -1/-1 | 0 | True |
| `layer1.1.x1` | 5 | 4 | 4 | -1/0 | 0 | True |
| `layer1.1.second_add` | 5 | 4 | 4 | -1/0 | 0 | True |
| `layer1.2.x1` | 5 | 4 | 4 | -1/0 | 0 | True |
| `layer1.2.second_add` | 5 | 4 | 3 | -2/-1 | 0 | True |
| `layer2.0.x1` | 5 | 4 | 4 | -1/0 | 0 | True |
| `layer2.0.second_add` | 4 | 4 | 3 | -1/-1 | 0 | True |
| `layer2.1.x1` | 5 | 3 | 3 | -2/0 | 0 | True |
| `layer2.1.second_add` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer2.2.x1` | 5 | 3 | 4 | -1/1 | 0 | True |
| `layer2.2.second_add` | 5 | 4 | 3 | -2/-1 | 0 | True |
| `layer3.0.x1` | 5 | 4 | 3 | -2/-1 | 0 | True |
| `layer3.0.second_add` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer3.1.x1` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer3.1.second_add` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer3.2.x1` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer3.2.second_add` | 4 | 3 | 3 | -1/0 | 0 | True |

### INT7

| Add node | Branch A s | Branch B s | Output s | Delta A/B | Test saturation | Equivalence |
|---|---:|---:|---:|---|---:|---|
| `layer1.0.x1` | 4 | 5 | 4 | 0/-1 | 0 | True |
| `layer1.0.second_add` | 4 | 4 | 3 | -1/-1 | 0 | True |
| `layer1.1.x1` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer1.1.second_add` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer1.2.x1` | 4 | 3 | 2 | -2/-1 | 0 | True |
| `layer1.2.second_add` | 4 | 2 | 2 | -2/0 | 0 | True |
| `layer2.0.x1` | 4 | 3 | 3 | -1/0 | 0 | True |
| `layer2.0.second_add` | 3 | 3 | 2 | -1/-1 | 0 | True |
| `layer2.1.x1` | 4 | 2 | 2 | -2/0 | 0 | True |
| `layer2.1.second_add` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer2.2.x1` | 4 | 2 | 3 | -1/1 | 0 | True |
| `layer2.2.second_add` | 4 | 3 | 2 | -2/-1 | 0 | True |
| `layer3.0.x1` | 3 | 3 | 2 | -1/-1 | 0 | True |
| `layer3.0.second_add` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer3.1.x1` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer3.1.second_add` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer3.2.x1` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer3.2.second_add` | 3 | 2 | 2 | -1/0 | 0 | True |

### INT6

| Add node | Branch A s | Branch B s | Output s | Delta A/B | Test saturation | Equivalence |
|---|---:|---:|---:|---|---:|---|
| `layer1.0.x1` | 3 | 4 | 3 | 0/-1 | 0 | True |
| `layer1.0.second_add` | 3 | 3 | 2 | -1/-1 | 0 | True |
| `layer1.1.x1` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer1.1.second_add` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer1.2.x1` | 3 | 2 | 1 | -2/-1 | 0 | True |
| `layer1.2.second_add` | 3 | 1 | 1 | -2/0 | 0 | True |
| `layer2.0.x1` | 3 | 2 | 2 | -1/0 | 0 | True |
| `layer2.0.second_add` | 2 | 2 | 1 | -1/-1 | 0 | True |
| `layer2.1.x1` | 3 | 1 | 1 | -2/0 | 0 | True |
| `layer2.1.second_add` | 2 | 1 | 1 | -1/0 | 0 | True |
| `layer2.2.x1` | 3 | 1 | 1 | -2/0 | 0 | True |
| `layer2.2.second_add` | 3 | 1 | 1 | -2/0 | 0 | True |
| `layer3.0.x1` | 2 | 1 | 1 | -1/0 | 0 | True |
| `layer3.0.second_add` | 2 | 1 | 1 | -1/0 | 0 | True |
| `layer3.1.x1` | 2 | 1 | 1 | -1/0 | 0 | True |
| `layer3.1.second_add` | 2 | 1 | 1 | -1/0 | 0 | True |
| `layer3.2.x1` | 2 | 1 | 1 | -1/0 | 0 | True |
| `layer3.2.second_add` | 2 | 1 | 1 | -1/0 | 0 | True |

### INT5

| Add node | Branch A s | Branch B s | Output s | Delta A/B | Test saturation | Equivalence |
|---|---:|---:|---:|---|---:|---|
| `layer1.0.x1` | 2 | 3 | 2 | 0/-1 | 0 | True |
| `layer1.0.second_add` | 2 | 2 | 1 | -1/-1 | 0 | True |
| `layer1.1.x1` | 2 | 1 | 1 | -1/0 | 0 | True |
| `layer1.1.second_add` | 1 | 1 | 1 | 0/0 | 0 | True |
| `layer1.2.x1` | 2 | 1 | 0 | -2/-1 | 0 | True |
| `layer1.2.second_add` | 2 | 0 | 0 | -2/0 | 0 | True |
| `layer2.0.x1` | 2 | 1 | 0 | -2/-1 | 0 | True |
| `layer2.0.second_add` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer2.1.x1` | 2 | 0 | 0 | -2/0 | 0 | True |
| `layer2.1.second_add` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer2.2.x1` | 2 | 0 | 0 | -2/0 | 0 | True |
| `layer2.2.second_add` | 2 | 0 | 0 | -2/0 | 0 | True |
| `layer3.0.x1` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer3.0.second_add` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer3.1.x1` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer3.1.second_add` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer3.2.x1` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer3.2.second_add` | 1 | 0 | 0 | -1/0 | 0 | True |

### INT4

| Add node | Branch A s | Branch B s | Output s | Delta A/B | Test saturation | Equivalence |
|---|---:|---:|---:|---|---:|---|
| `layer1.0.x1` | 1 | 2 | 1 | 0/-1 | 0 | True |
| `layer1.0.second_add` | 1 | 1 | 0 | -1/-1 | 0 | True |
| `layer1.1.x1` | 1 | 0 | 0 | -1/0 | 0 | True |
| `layer1.1.second_add` | 0 | 0 | 0 | 0/0 | 0 | True |
| `layer1.2.x1` | 1 | 0 | -1 | -2/-1 | 0 | True |
| `layer1.2.second_add` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer2.0.x1` | 1 | 0 | -1 | -2/-1 | 0 | True |
| `layer2.0.second_add` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer2.1.x1` | 1 | -1 | -1 | -2/0 | 0 | True |
| `layer2.1.second_add` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer2.2.x1` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer2.2.second_add` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer3.0.x1` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer3.0.second_add` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer3.1.x1` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer3.1.second_add` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer3.2.x1` | 0 | -1 | -1 | -1/0 | 0 | True |
| `layer3.2.second_add` | 0 | -1 | -1 | -1/0 | 0 | True |

## H. Option-A handling

Stage2 and Stage3 shortcuts use the actual workspace Option-A implementation: `x[:, :, ::2, ::2]`, followed by symmetric zero channel padding. The integer adapter performs the same transform directly on q, inherits the input shift before any designated shortcut requantization, and verifies the resulting shape/alignment metadata.

| Block | Stride | Input channels | Output channels | Padding per side |
|---|---:|---:|---:|---:|
| `layer2.0` | 2 | 16 | 32 | 8 |
| `layer3.0` | 2 | 32 | 64 | 16 |

## I. Accuracy table

| Width | Accuracy | Delta vs H0 85.13% | Test saturation |
|---|---:|---:|---:|
| H0 reference | 85.13% | 0.00 pp | - |
| INT8 | 77.07% | -8.06 pp | 0 |
| INT7 | 68.82% | -16.31 pp | 0 |
| INT6 | 35.45% | -49.68 pp | 0 |
| INT5 | 10.00% | -75.13 pp | 0 |
| INT4 | 13.51% | -71.62 pp | 83 |

## J. Test saturation statistics

The JSON contains per-node saturation and error statistics for every width. The following table shows total saturation and the largest per-node count.

| Width | Total saturation | Node with largest saturation | Count |
|---|---:|---|---:|
| INT8 | 0 | `stem.post_hardtanh` | 0 |
| INT7 | 0 | `stem.post_hardtanh` | 0 |
| INT6 | 0 | `stem.post_hardtanh` | 0 |
| INT5 | 0 | `stem.post_hardtanh` | 0 |
| INT4 | 83 | `head.affine_output` | 65 |

## K. Sensitive-node analysis

- Most sensitive node by INT4 test quantization MSE proxy: **`layer3.0.qrprelu_output`**.
- Top residual state by INT4 saturation/MSE proxy: **`layer1.0.x1`**.
- INT5 accuracy drop: **75.13 pp**; INT4 accuracy drop: **71.62 pp**.
- Residual-output saturation total at INT4: **0**.
- Interpretation: no_residual_output_saturation; accuracy loss is not attributable to one saturation event.
- This is a quantization-error/saturation proxy, not a one-node ablation attribution: Sensitivity ranking is a quantization-error/saturation proxy, not a one-node ablation attribution.

## L. Recommended production width

- Lowest near-lossless (drop <= 0.10 pp): **none**.
- Recommended production (drop <= 0.30 pp): **none**.
- Absolute minimum acceptable (drop <= 0.50 pp): **none**.
- INT4 drop was 71.62 pp, so INT3 was not tested under the optional rule.

## M. Hardware interpretation

Affine offsets/classifier bias use fixed INT6 integer storage from H0, and the selected multi-bit activation/residual states have a verified W-bit integer representation with power-of-two scale metadata.

This does not mean the entire network is fully finite-width integer-only. Binary-convolution accumulators, GAP accumulation, final FC accumulation, and final logit width remain full precision for this H1 experiment and are reserved for later experiments.

- Binary weights, signed-pow2 K, FC signed-pow2 weights, Thermometer, QRPReLU function/parameters, and architecture are unchanged.
- No extra inference BMAC was introduced; H1 only changes representation boundaries of existing multi-bit states.
- No training, fine-tuning, KD, accumulator truncation, GAP quantization, FC accumulator quantization, RTL, or FPGA work was performed.
