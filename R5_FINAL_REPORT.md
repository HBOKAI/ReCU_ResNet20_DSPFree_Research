# R5 Final Report — R4 + W1A8 Stem

Date: 2026-09-12  
Environment: `KenBnn_env`, PyTorch 2.5.1+cu121, RTX 4070  
Dataset protocol: official ReCU CIFAR-10 train/test split  

## 1. Executive result

R5 successfully replaces the first 3x3 stem convolution with a W1A8 implementation:

- latent floating-point stem weights are projected with STE to exactly `{-1, +1}`;
- normalized input activations are fake-quantized to signed INT8 with fixed scale `2^-5 = 0.03125`;
- the stem product is implemented as `+q` or `-q`, followed by accumulation;
- the stem BN is intentionally retained;
- the R4 binary backbone, fused power-of-two affine path, residual path, GAP, and FP final FC are unchanged.

The formal 100-epoch run completed successfully. Best/reloaded test accuracy is **83.77%** at epoch 98.

This is below the current 84.5% threshold for proceeding to final FC/BN elimination. Therefore, do not start the next hardware-elimination stage yet; improve the R5 stem/QAT behavior first.

## 2. Source R4 and formal run

Source R4 checkpoint:

`experiments/recu_r4/recu_r4_qrprelu_fused_pow2_affine_20260912_110737/best.pt`

| Item | Result |
|---|---:|
| R4 source best accuracy | 86.11% |
| R4 source best epoch | 98 |
| R5 formal epochs | 100 |
| R5 best epoch | 98 |
| R5 best checkpoint accuracy | 83.77% |
| R5 reload accuracy | 83.77% |
| R5 final epoch accuracy | 83.63% |
| R5 training time | 1492.81 s (24 min 52.81 s) |
| R5 trainable parameters | 270,858 |
| History records | 100 |

Formal run directory:

`experiments/recu_r5_w1a8_stem/recu_r5_w1a8_stem_20260912_122359/`

The formal `best.pt` was used for the reported result. It is not assumed to be identical to `last.pt`; the final verification confirmed that the two state dictionaries are different.

## 3. Accuracy comparison

| Model | Test accuracy | Difference vs R4 |
|---|---:|---:|
| Official ReCU | 87.28% | +1.17 pp |
| R4 fused pow2 affine | 86.11% | reference |
| R5 initial converted, epoch 0 | 22.54% | -63.57 pp |
| R5 20-epoch diagnostic best | 78.43% | -7.68 pp |
| R5 formal best, epoch 98 | **83.77%** | **-2.34 pp** |
| R5 formal final, epoch 100 | 83.63% | -2.48 pp |

The formal run did not collapse: accuracy recovered from the 22.54% converted initialization to 83.77%, and the stem binary signs remained balanced. However, the result remains below the research threshold for moving on to final FC/BN elimination.

## 4. Input INT8 verification

The input protocol is the existing normalized CIFAR-10 tensor followed by fixed fake quantization:

```text
q  = clip(round(x / 0.03125), -128, 127)
xq = q * 0.03125
```

| Measurement | Result |
|---|---:|
| Scale exponent | -5 |
| Scale | 0.03125 |
| Normalized input minimum | -2.11790395 |
| Normalized input maximum | 2.64000010 |
| Observed q minimum | -68 |
| Observed q maximum | 84 |
| Total values checked | 12,582,912 |
| Low saturation count | 0 |
| High saturation count | 0 |
| Saturation fraction | 0.0 |

The required saturation limit of `1e-4` is satisfied with a large margin. No scale adjustment was needed.

## 5. W1A8 stem verification

The stem contains 432 latent weight values. At the formal best checkpoint:

| Measurement | Result |
|---|---:|
| Binary positive weights | 211 |
| Binary negative weights | 221 |
| Latent zero weights | 0 |
| Latent minimum | -0.73815155 |
| Latent maximum | 0.65901124 |
| Latent mean | 0.00295131 |
| Latent standard deviation | 0.24134313 |
| Diagnostic scaled-binary MSE | 0.02424737 |
| Diagnostic alpha mean | 0.18176651 |

The stem has no sign collapse and uses only `+1/-1` in the forward path. The R4 compatibility alpha tensors total 672 values and were verified frozen; the reconstructed model has the expected 270,858 trainable parameters.

## 6. Hardware-oriented accounting

| Module | R5 operation | General multiplier / DSP candidate? | Current status |
|---|---|---|---|
| Input quantizer | round, clip to signed INT8, fixed binary-point scale `2^-5` | No general multiplier; shift/binary point | Implemented and verified |
| 3x3 stem convolution | `+q` / `-q` select-negate, then accumulate | No | W1A8 implemented |
| Stem BN | Batch-normalization affine transform | Yes | Retained; not eliminated |
| Binary convolution backbone | XNOR/XOR plus popcount | No | Preserved from R4 |
| Fused pow2 affine | signed shift, optional negate, add | No general multiplier | Preserved from R4 |
| Residual path | multi-bit addition | No | Preserved from R4 |
| Hardtanh | compare and clamp | No | Preserved from R4 |
| QRPReLU | add, shift, add | No general multiplier | Preserved from R4 |
| GAP | sum followed by division by 64 (`>>6`) | No | Preserved |
| Head BN | Batch-normalization affine transform | Yes | Retained; not eliminated |
| Final FC, 64 to 10 | multi-bit matrix multiply | Yes | Retained; not quantized |

For operation-count context, the original 3x3 stem has 442,368 conventional product sites. In R5 these are 442,368 signed-INT8 select/negate-and-accumulate terms rather than general weight-by-activation multiplications. The binary backbone retains 40,108,032 XNOR/popcount terms.

The input normalization is still a software/dataset preprocessing protocol. This report therefore does not claim a raw-RGB, end-to-end multiplier-free implementation.

## 7. Verification record

- R5 conversion verification: passed.
- Binary stem values: exactly `{-1, +1}`.
- INT8 saturation: 0.0 fraction.
- Smoke run: passed on CUDA, with save/reload.
- 20-epoch diagnostic: completed; no collapse or abnormal saturation.
- Formal 100-epoch run: completed.
- Formal history: 100/100 records, all finite.
- Formal best checkpoint reload: 83.77%, matching the recorded best.
- R4 compatibility alpha parameters: 672 values, all frozen.
- Unit tests: **18/18 passed**.

## 8. Decision

R5 proves that the stem convolution can be changed to W1A8 without a general multiplier/DSP in the convolution product itself. The accuracy cost is currently 2.34 percentage points versus R4 and 3.51 points versus Official ReCU.

Because the formal result is 83.77%, below the 84.5% go/no-go threshold, the recommended next action is to improve the W1A8 stem training or initialization and rerun the R5 ablation. Do not proceed to final FC quantization or BN elimination from this checkpoint.
