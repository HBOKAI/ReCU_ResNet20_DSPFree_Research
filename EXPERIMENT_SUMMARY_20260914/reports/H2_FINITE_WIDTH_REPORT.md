# H2A-v2 GAP Deferred Scaling Report

> H2 closes finite-width accumulator, GAP, FC, bias-add, and logit arithmetic around the frozen R8B-H1MP plan. No retraining was performed.

## 1. Source verification and frozen scope

- R8B checkpoint: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r8b\recu_r8b_r7_pow2_fc_20260912_225827\best.pt`
- H0 INT6 bias export: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H0_INTEGER_BIAS_SWEEP_RESULTS.json`
- Frozen H1MP plan: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H1MP_SEARCH_RESULTS.json`
- H1MP source official-test accuracy reproduced: **85.03%**; expected 85.03% ±0.08 pp.
- H0 INT6 bias, H1MP 58-node bits/shifts/policies, binary weights, signed-pow2 K, FC weights, Thermometer R=8, and architecture were not modified.
- Official TEST was evaluation-only after the H2A plan was frozen; it was not used for any width decision.
- H2B was not run in this H2A-v2 experiment by design.

## 2. Binary-convolution exact widths

| Path | Terms | Popcount width | Signed accumulator | Mathematical range | Overflow |
|---|---:|---:|---:|---|---:|
| Stem Cin=96 | 864 | INT10 | INT11 | [-864, 864] | 0 |
| Cin=16 | 144 | INT8 | INT9 | [-144, 144] | 0 |
| Cin=32 | 288 | INT9 | INT10 | [-288, 288] | 0 |
| Cin=64 | 576 | INT10 | INT11 | [-576, 576] | 0 |

## 3. H2A-v2 exact/conservative plan

### Pow2-affine intermediate widths

| H1 boundary | Width | Integer shift | Range | Basis |
|---|---:|---:|---|---|
| `stem.post_hardtanh` | INT13 | 6 | [-3477, 3456] | worst-case signed input range plus exact INT6 bias range |
| `layer1.0.first_affine` | INT11 | 7 | [-564, 588] | worst-case signed input range plus exact INT6 bias range |
| `layer1.0.second_affine` | INT10 | 6 | [-328, 288] | worst-case signed input range plus exact INT6 bias range |
| `layer1.1.first_affine` | INT10 | 6 | [-290, 302] | worst-case signed input range plus exact INT6 bias range |
| `layer1.1.second_affine` | INT10 | 6 | [-348, 284] | worst-case signed input range plus exact INT6 bias range |
| `layer1.2.first_affine` | INT11 | 7 | [-636, 516] | worst-case signed input range plus exact INT6 bias range |
| `layer1.2.second_affine` | INT11 | 7 | [-824, 584] | worst-case signed input range plus exact INT6 bias range |
| `layer2.0.first_affine` | INT11 | 7 | [-636, 588] | worst-case signed input range plus exact INT6 bias range |
| `layer2.0.second_affine` | INT12 | 7 | [-1248, 1120] | worst-case signed input range plus exact INT6 bias range |
| `layer2.1.first_affine` | INT12 | 7 | [-1176, 1200] | worst-case signed input range plus exact INT6 bias range |
| `layer2.1.second_affine` | INT12 | 7 | [-1252, 1184] | worst-case signed input range plus exact INT6 bias range |
| `layer2.2.first_affine` | INT11 | 6 | [-592, 592] | worst-case signed input range plus exact INT6 bias range |
| `layer2.2.second_affine` | INT11 | 6 | [-596, 590] | worst-case signed input range plus exact INT6 bias range |
| `layer3.0.first_affine` | INT12 | 7 | [-1236, 1112] | worst-case signed input range plus exact INT6 bias range |
| `layer3.0.second_affine` | INT13 | 7 | [-2340, 2296] | worst-case signed input range plus exact INT6 bias range |
| `layer3.1.first_affine` | INT12 | 6 | [-1173, 1142] | worst-case signed input range plus exact INT6 bias range |
| `layer3.1.second_affine` | INT13 | 7 | [-2384, 2300] | worst-case signed input range plus exact INT6 bias range |
| `layer3.2.first_affine` | INT13 | 7 | [-2354, 2340] | worst-case signed input range plus exact INT6 bias range |
| `layer3.2.second_affine` | INT12 | 6 | [-1186, 1152] | worst-case signed input range plus exact INT6 bias range |
| `head.affine_output` | INT21 | 11 | [-539648, 518128] | worst-case signed input range plus exact INT6 bias range |

### Residual alignment/add intermediates

| Add | Branch A | Branch B | Output shift | Width | Range |
|---|---|---|---:|---:|---|
| `layer1.0.x1` | INT6/4 | INT6/5 | 3 | INT6 | [-24, 24] |
| `layer1.0.second_add` | INT6/4 | INT6/3 | 3 | INT7 | [-48, 47] |
| `layer1.1.x1` | INT6/4 | INT7/4 | 3 | INT7 | [-48, 48] |
| `layer1.1.second_add` | INT6/4 | INT6/3 | 3 | INT7 | [-48, 47] |
| `layer1.2.x1` | INT6/4 | INT12/8 | 3 | INT8 | [-80, 80] |
| `layer1.2.second_add` | INT7/4 | INT6/3 | 4 | INT8 | [-128, 125] |
| `layer2.0.x1` | INT8/5 | INT6/3 | 3 | INT7 | [-64, 63] |
| `layer2.0.second_add` | INT6/3 | INT6/3 | 3 | INT7 | [-64, 62] |
| `layer2.1.x1` | INT7/5 | INT6/2 | 3 | INT8 | [-80, 78] |
| `layer2.1.second_add` | INT7/4 | INT6/3 | 3 | INT7 | [-64, 63] |
| `layer2.2.x1` | INT7/5 | INT12/8 | 2 | INT7 | [-40, 40] |
| `layer2.2.second_add` | INT6/4 | INT6/2 | 3 | INT8 | [-80, 78] |
| `layer3.0.x1` | INT6/4 | INT6/2 | 3 | INT8 | [-80, 78] |
| `layer3.0.second_add` | INT6/3 | INT6/3 | 3 | INT7 | [-64, 62] |
| `layer3.1.x1` | INT7/4 | INT6/3 | 4 | INT8 | [-128, 125] |
| `layer3.1.second_add` | INT7/4 | INT7/4 | 3 | INT8 | [-64, 64] |
| `layer3.2.x1` | INT7/4 | INT10/7 | 3 | INT8 | [-64, 64] |
| `layer3.2.second_add` | INT7/4 | INT6/3 | 3 | INT7 | [-64, 63] |

### QRPReLU integer intermediates

| Block | Input | Parameter shift | Inner add | Negative/output width | Output shift |
|---|---:|---:|---:|---:|---:|
| `layer1.0.qrprelu_output` | INT6 | 24 | INT28 | INT36 | 32 |
| `layer1.1.qrprelu_output` | INT6 | 24 | INT28 | INT36 | 32 |
| `layer1.2.qrprelu_output` | INT8 | 24 | INT29 | INT37 | 32 |
| `layer2.0.qrprelu_output` | INT6 | 24 | INT28 | INT36 | 32 |
| `layer2.1.qrprelu_output` | INT8 | 24 | INT30 | INT38 | 32 |
| `layer2.2.qrprelu_output` | INT6 | 24 | INT28 | INT29 | 25 |
| `layer3.0.qrprelu_output` | INT6 | 24 | INT28 | INT35 | 32 |
| `layer3.1.qrprelu_output` | INT7 | 24 | INT29 | INT37 | 32 |
| `layer3.2.qrprelu_output` | INT7 | 24 | INT29 | INT36 | 32 |

### GAP

- Input: INT10 at shift 6, 64 values (8x8).
- Sum accumulator: INT16 with mathematical range [-32768, 32704].
- Divide-by-64: **deferred scale metadata**; q_gap = q_sum (no arithmetic right shift and no integer rounding), with s_gap = s_input + 6.
- GAP output: q_gap remains the INT16 sum representation at scale shift 12; the /64 factor is metadata-only.

### FC / bias-add / final logits

- FC input: INT6 at shift 2; common product scale shift 11.

| Class | Worst-case min | Worst-case max | FC accumulator width |
|---:|---:|---:|---:|
| 0 | -237670 | 237980 | INT19 |
| 1 | -262862 | 262180 | INT20 |
| 2 | -250768 | 250271 | INT19 |
| 3 | -179164 | 178739 | INT19 |
| 4 | -238974 | 238944 | INT19 |
| 5 | -214956 | 214452 | INT19 |
| 6 | -239688 | 240624 | INT19 |
| 7 | -273524 | 272560 | INT20 |
| 8 | -273654 | 274068 | INT20 |
| 9 | -244836 | 245052 | INT19 |

- H2A FC accumulator: **INT24**.
- FC INT6 bias-add: **INT24** at shift 11.
- H2A final logits: **INT24**.

## 4. H2A-v2 anchor

- Accuracy: **85.03%**.
- Delta vs reproduced H1MP source: **+0.00 pp**.
- H2A-v2 gate |delta| <=0.10 pp: **PASS**.
- H2A overflow total: **0**; saturation total including frozen H1MP boundaries: **107974934**.

### Bit-accurate GAP verification

- Reference: `(q_sum / 64) * 2^-s_input`.
- Integer datapath interpretation: `q_sum * 2^-(s_input+6)`.
- Mathematical equality: **True**; max absolute error: **0.000e+00**.
- Integer rounding applied by GAP division: **False**.
- Explicit implementation statement: GAP division by 64 is implemented as a scale metadata adjustment, not arithmetic right-shift rounding.

## 5. H2B

H2B was not run by design. H2A-v2 is a single-variable verification pass only.
