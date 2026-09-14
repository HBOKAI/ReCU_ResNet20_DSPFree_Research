# H0 Integer Bias Sweep Report

> Zero-shot pure-integer bias quantization on the formal R8B checkpoint. No fine-tuning was performed.

## A. Configuration

- Architecture: R8B unchanged; Thermometer `R=8`, raw RGB `[0,1]`, bipolar `{-1,+1}`.
- Bias widths tested: INT8, INT7, INT6, INT5, INT4.
- Shift search: `s=0..12`; negative extension to `-12` only when the initial range has no zero-saturation candidate.
- Per-layer encoding: `q = round(B * 2^s)`, signed integer `q`, hardware reconstruction `Bhat = q * 2^-s`.
- Only `stem_affine.bias`, the 18 R4 fused-affine biases, `head_affine.bias`, and `linear.bias` were changed in evaluation copies.
- K/weight tensors, activation, residual, GAP, accumulators, alpha, and architecture were not changed.

## B. Formal R8B reference

- Checkpoint: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r8b\recu_r8b_r7_pow2_fc_20260912_225827\best.pt`
- Summary: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r8b\recu_r8b_r7_pow2_fc_20260912_225827\summary.json`
- Stored best: **85.40%** @ epoch **100**
- Actual full-test reload: **85.40%**
- R8B invariant check: PASS; stem binary, all affine/head/FC K exact signed-pow2, FC bias present, 672 alpha parameters frozen, no BN2.

### Bias target inventory

| Module | Class | Parameters | Original min | Original max |
|---|---:|---:|---:|---:|
| `stem_affine` | `StemFusedPow2Affine2d` | 16 | -0.39958563 | 0.19490702 |
| `layer1.0.aff1` | `FusedAffine2d` | 16 | -0.52752298 | 0.74521744 |
| `layer1.0.aff2` | `FusedAffine2d` | 16 | -0.64302683 | 0.59875023 |
| `layer1.1.aff1` | `FusedAffine2d` | 16 | -0.78595251 | 0.60695291 |
| `layer1.1.aff2` | `FusedAffine2d` | 16 | -1.1003922 | 0.16347028 |
| `layer1.2.aff1` | `FusedAffine2d` | 16 | -0.46756229 | 0.42711928 |
| `layer1.2.aff2` | `FusedAffine2d` | 16 | -1.9072552 | 0.03804056 |
| `layer2.0.aff1` | `FusedAffine2d` | 32 | -0.53599936 | 0.53952652 |
| `layer2.0.aff2` | `FusedAffine2d` | 32 | -0.75009799 | 0.65777785 |
| `layer2.1.aff1` | `FusedAffine2d` | 32 | -0.53789687 | 0.60401821 |
| `layer2.1.aff2` | `FusedAffine2d` | 32 | -0.84077519 | 0.4706738 |
| `layer2.2.aff1` | `FusedAffine2d` | 32 | -0.63251233 | 0.56610864 |
| `layer2.2.aff2` | `FusedAffine2d` | 32 | -0.70718986 | 0.34931752 |
| `layer3.0.aff1` | `FusedAffine2d` | 64 | -0.65012586 | 0.5524016 |
| `layer3.0.aff2` | `FusedAffine2d` | 64 | -0.7500819 | 0.07007198 |
| `layer3.1.aff1` | `FusedAffine2d` | 64 | -0.46982303 | 0.49016124 |
| `layer3.1.aff2` | `FusedAffine2d` | 64 | -0.75475347 | -0.040348195 |
| `layer3.2.aff1` | `FusedAffine2d` | 64 | -0.48762628 | 0.35399422 |
| `layer3.2.aff2` | `FusedAffine2d` | 64 | -0.56715536 | 0.094937608 |
| `head_affine` | `HeadFusedPow2Affine1d` | 64 | -7.4480515 | -1.1490291 |
| `linear` | `Pow2LinearSTE` | 10 | -0.14814837 | 0.28566426 |

## C. Accuracy sweep

| Width | Accuracy | Delta vs R8B reload | Total saturation |
|---:|---:|---:|---:|
| INT8 | 85.12% | -0.28 pp | 0 |
| INT7 | 85.06% | -0.34 pp | 0 |
| INT6 | 85.13% | -0.27 pp | 0 |
| INT5 | 84.35% | -1.05 pp | 0 |
| INT4 | 83.24% | -2.16 pp | 0 |

## D. Per-layer integer export

The complete integer parameter tables (`q_values`) are in `H0_INTEGER_BIAS_SWEEP_RESULTS.json`. The table below records every target, selected layer shift, range, saturation, MSE, and maximum absolute reconstruction error.

### INT8

| Module | Shift | q range | Saturation | MSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| `stem_affine` | 8 | [-128, 127] | 0 | 0.000001 | 0.001641 |
| `layer1.0.aff1` | 7 | [-128, 127] | 0 | 0.000007 | 0.003727 |
| `layer1.0.aff2` | 7 | [-128, 127] | 0 | 0.000003 | 0.003263 |
| `layer1.1.aff1` | 7 | [-128, 127] | 0 | 0.000006 | 0.003517 |
| `layer1.1.aff2` | 6 | [-128, 127] | 0 | 0.000021 | 0.007220 |
| `layer1.2.aff1` | 8 | [-128, 127] | 0 | 0.000001 | 0.001841 |
| `layer1.2.aff2` | 6 | [-128, 127] | 0 | 0.000025 | 0.007577 |
| `layer2.0.aff1` | 7 | [-128, 127] | 0 | 0.000006 | 0.003802 |
| `layer2.0.aff2` | 7 | [-128, 127] | 0 | 0.000003 | 0.003845 |
| `layer2.1.aff1` | 7 | [-128, 127] | 0 | 0.000006 | 0.003767 |
| `layer2.1.aff2` | 7 | [-128, 127] | 0 | 0.000005 | 0.003737 |
| `layer2.2.aff1` | 7 | [-128, 127] | 0 | 0.000005 | 0.003807 |
| `layer2.2.aff2` | 7 | [-128, 127] | 0 | 0.000005 | 0.003892 |
| `layer3.0.aff1` | 7 | [-128, 127] | 0 | 0.000006 | 0.003844 |
| `layer3.0.aff2` | 7 | [-128, 127] | 0 | 0.000005 | 0.003898 |
| `layer3.1.aff1` | 8 | [-128, 127] | 0 | 0.000001 | 0.001923 |
| `layer3.1.aff2` | 7 | [-128, 127] | 0 | 0.000005 | 0.003876 |
| `layer3.2.aff1` | 8 | [-128, 127] | 0 | 0.000001 | 0.001947 |
| `layer3.2.aff2` | 7 | [-128, 127] | 0 | 0.000006 | 0.003854 |
| `head_affine` | 4 | [-128, 127] | 0 | 0.000302 | 0.031230 |
| `linear` | 8 | [-128, 127] | 0 | 0.000001 | 0.001908 |

### INT7

| Module | Shift | q range | Saturation | MSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| `stem_affine` | 7 | [-64, 63] | 0 | 0.000004 | 0.003439 |
| `layer1.0.aff1` | 6 | [-64, 63] | 0 | 0.000023 | 0.006720 |
| `layer1.0.aff2` | 6 | [-64, 63] | 0 | 0.000023 | 0.007494 |
| `layer1.1.aff1` | 6 | [-64, 63] | 0 | 0.000016 | 0.006525 |
| `layer1.1.aff2` | 5 | [-64, 63] | 0 | 0.000100 | 0.015511 |
| `layer1.2.aff1` | 7 | [-64, 63] | 0 | 0.000004 | 0.003548 |
| `layer1.2.aff2` | 5 | [-64, 63] | 0 | 0.000061 | 0.014278 |
| `layer2.0.aff1` | 6 | [-64, 63] | 0 | 0.000022 | 0.007348 |
| `layer2.0.aff2` | 6 | [-64, 63] | 0 | 0.000018 | 0.007766 |
| `layer2.1.aff1` | 6 | [-64, 63] | 0 | 0.000023 | 0.007583 |
| `layer2.1.aff2` | 6 | [-64, 63] | 0 | 0.000018 | 0.007636 |
| `layer2.2.aff1` | 6 | [-64, 63] | 0 | 0.000020 | 0.007555 |
| `layer2.2.aff2` | 6 | [-64, 63] | 0 | 0.000022 | 0.007806 |
| `layer3.0.aff1` | 6 | [-64, 63] | 0 | 0.000018 | 0.007565 |
| `layer3.0.aff2` | 6 | [-64, 63] | 0 | 0.000023 | 0.007617 |
| `layer3.1.aff1` | 7 | [-64, 63] | 0 | 0.000006 | 0.003865 |
| `layer3.1.aff2` | 6 | [-64, 63] | 0 | 0.000021 | 0.007676 |
| `layer3.2.aff1` | 7 | [-64, 63] | 0 | 0.000005 | 0.003879 |
| `layer3.2.aff2` | 6 | [-64, 63] | 0 | 0.000021 | 0.007680 |
| `head_affine` | 3 | [-64, 63] | 0 | 0.001366 | 0.062351 |
| `linear` | 7 | [-64, 63] | 0 | 0.000005 | 0.003493 |

### INT6

| Module | Shift | q range | Saturation | MSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| `stem_affine` | 6 | [-32, 31] | 0 | 0.000023 | 0.007407 |
| `layer1.0.aff1` | 5 | [-32, 31] | 0 | 0.000071 | 0.015029 |
| `layer1.0.aff2` | 5 | [-32, 31] | 0 | 0.000072 | 0.014594 |
| `layer1.1.aff1` | 5 | [-32, 31] | 0 | 0.000084 | 0.015421 |
| `layer1.1.aff2` | 4 | [-32, 31] | 0 | 0.000416 | 0.028947 |
| `layer1.2.aff1` | 6 | [-32, 31] | 0 | 0.000024 | 0.007353 |
| `layer1.2.aff2` | 4 | [-32, 31] | 0 | 0.000538 | 0.030839 |
| `layer2.0.aff1` | 5 | [-32, 31] | 0 | 0.000063 | 0.014660 |
| `layer2.0.aff2` | 5 | [-32, 31] | 0 | 0.000090 | 0.015363 |
| `layer2.1.aff1` | 5 | [-32, 31] | 0 | 0.000073 | 0.014063 |
| `layer2.1.aff2` | 5 | [-32, 31] | 0 | 0.000084 | 0.015417 |
| `layer2.2.aff1` | 5 | [-32, 31] | 0 | 0.000073 | 0.014619 |
| `layer2.2.aff2` | 5 | [-32, 31] | 0 | 0.000097 | 0.015337 |
| `layer3.0.aff1` | 5 | [-32, 31] | 0 | 0.000078 | 0.015587 |
| `layer3.0.aff2` | 5 | [-32, 31] | 0 | 0.000081 | 0.015345 |
| `layer3.1.aff1` | 6 | [-32, 31] | 0 | 0.000021 | 0.007685 |
| `layer3.1.aff2` | 5 | [-32, 31] | 0 | 0.000055 | 0.015474 |
| `layer3.2.aff1` | 6 | [-32, 31] | 0 | 0.000019 | 0.007678 |
| `layer3.2.aff2` | 5 | [-32, 31] | 0 | 0.000078 | 0.015241 |
| `head_affine` | 2 | [-32, 31] | 0 | 0.004624 | 0.124506 |
| `linear` | 6 | [-32, 31] | 0 | 0.000015 | 0.007523 |

### INT5

| Module | Shift | q range | Saturation | MSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| `stem_affine` | 5 | [-16, 15] | 0 | 0.000075 | 0.014367 |
| `layer1.0.aff1` | 4 | [-16, 15] | 0 | 0.000291 | 0.028594 |
| `layer1.0.aff2` | 4 | [-16, 15] | 0 | 0.000314 | 0.030802 |
| `layer1.1.aff1` | 4 | [-16, 15] | 0 | 0.000365 | 0.030527 |
| `layer1.1.aff2` | 3 | [-16, 15] | 0 | 0.001371 | 0.059901 |
| `layer1.2.aff1` | 5 | [-16, 15] | 0 | 0.000078 | 0.014988 |
| `layer1.2.aff2` | 3 | [-16, 15] | 0 | 0.001154 | 0.055683 |
| `layer2.0.aff1` | 4 | [-16, 15] | 0 | 0.000321 | 0.031224 |
| `layer2.0.aff2` | 4 | [-16, 15] | 0 | 0.000339 | 0.030786 |
| `layer2.1.aff1` | 4 | [-16, 15] | 0 | 0.000341 | 0.030932 |
| `layer2.1.aff2` | 4 | [-16, 15] | 0 | 0.000295 | 0.031207 |
| `layer2.2.aff1` | 4 | [-16, 15] | 0 | 0.000279 | 0.030858 |
| `layer2.2.aff2` | 4 | [-16, 15] | 0 | 0.000289 | 0.028599 |
| `layer3.0.aff1` | 4 | [-16, 15] | 0 | 0.000280 | 0.030837 |
| `layer3.0.aff2` | 4 | [-16, 15] | 0 | 0.000276 | 0.031031 |
| `layer3.1.aff1` | 4 | [-16, 15] | 0 | 0.000317 | 0.030333 |
| `layer3.1.aff2` | 4 | [-16, 15] | 0 | 0.000387 | 0.030567 |
| `layer3.2.aff1` | 5 | [-16, 15] | 0 | 0.000089 | 0.015088 |
| `layer3.2.aff2` | 4 | [-16, 15] | 0 | 0.000313 | 0.030670 |
| `head_affine` | 1 | [-16, 15] | 0 | 0.021716 | 0.248098 |
| `linear` | 5 | [-16, 15] | 0 | 0.000043 | 0.014862 |

### INT4

| Module | Shift | q range | Saturation | MSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| `stem_affine` | 4 | [-8, 7] | 0 | 0.000370 | 0.030290 |
| `layer1.0.aff1` | 3 | [-8, 7] | 0 | 0.000876 | 0.057766 |
| `layer1.0.aff2` | 3 | [-8, 7] | 0 | 0.001129 | 0.056622 |
| `layer1.1.aff1` | 3 | [-8, 7] | 0 | 0.001710 | 0.060871 |
| `layer1.1.aff2` | 2 | [-8, 7] | 0 | 0.005910 | 0.114459 |
| `layer1.2.aff1` | 4 | [-8, 7] | 0 | 0.000290 | 0.030062 |
| `layer1.2.aff2` | 2 | [-8, 7] | 0 | 0.004588 | 0.118999 |
| `layer2.0.aff1` | 3 | [-8, 7] | 0 | 0.000960 | 0.060373 |
| `layer2.0.aff2` | 3 | [-8, 7] | 0 | 0.000900 | 0.058532 |
| `layer2.1.aff1` | 3 | [-8, 7] | 0 | 0.001389 | 0.057235 |
| `layer2.1.aff2` | 3 | [-8, 7] | 0 | 0.001108 | 0.061912 |
| `layer2.2.aff1` | 3 | [-8, 7] | 0 | 0.001150 | 0.062128 |
| `layer2.2.aff2` | 3 | [-8, 7] | 0 | 0.001346 | 0.060501 |
| `layer3.0.aff1` | 3 | [-8, 7] | 0 | 0.001108 | 0.062297 |
| `layer3.0.aff2` | 3 | [-8, 7] | 0 | 0.001341 | 0.060930 |
| `layer3.1.aff1` | 3 | [-8, 7] | 0 | 0.001145 | 0.061520 |
| `layer3.1.aff2` | 3 | [-8, 7] | 0 | 0.001395 | 0.062304 |
| `layer3.2.aff1` | 4 | [-8, 7] | 0 | 0.000290 | 0.029692 |
| `layer3.2.aff2` | 3 | [-8, 7] | 0 | 0.001421 | 0.062191 |
| `head_affine` | 0 | [-8, 7] | 0 | 0.092820 | 0.499673 |
| `linear` | 4 | [-8, 7] | 0 | 0.000544 | 0.030830 |

## E. Interpretation and decision

- Lowest near-lossless width (drop <= 0.10 pp): **none**.
- Recommended production width (drop <= 0.30 pp): **6**.
- Absolute minimum acceptable width (drop <= 0.50 pp): **6**.
- INT4 drop was 2.16 pp; optional INT3 rule threshold was 0.50 pp, so INT3 was not tested.

This experiment answers the H0 question by measuring the accuracy cost of storing only affine offsets and the FC bias as signed integers with a per-layer power-of-two shift. It does not retrain or alter the R8B effective weights.

## F. Hardware

Inference architecture is unchanged from formal R8B.

- No extra inference cost.
- No extra BMAC and no extra parameters caused by this zero-shot representation change.
- Stem remains pure binary convolution: XNOR + popcount, general multiplier = 0, convolution DSP = 0.
- Bias hardware representation is integer `q` plus layer-shared power-of-two shift metadata; no floating-point bias storage, general multiplier, or floating-point adder is part of the exported H0 representation.

## G. Reproducibility

- Full CIFAR-10 test set was used for the R8B reference and every reported width.
- No fine-tuning, KD, activation/residual/GAP/accumulator quantization, RTL, FPGA, or architecture change was performed.
- Export verification: PASS for integer dtype, signed range, q+shift reconstruction, and unchanged R8B invariants.
