# R5T Final Report — Thermometer R=8 + W1A1 Binary Stem

Date: 2026-09-12  
Environment: `KenBnn_env`, PyTorch 2.5.1+cu121, RTX 4070  
Dataset protocol: CIFAR-10 raw RGB input with the existing augmentation policy  

## 1. Executive result

R5T replaces the failed 3-channel first-layer binary bottleneck with a FracBNN-style thermometer input representation:

```text
raw RGB [0, 1]
 -> R=8 thermometer encoding
 -> 96 bipolar binary channels
 -> 3x3 W1A1 binary stem, 96 -> 16
 -> unchanged R4 backbone and head
```

The two controlled stages completed successfully:

- T1 FP-stem adaptation: **85.29%**, best epoch 99;
- T2 W1A1 binary-stem fine-tune: **85.00%**, best epoch 97;
- T2 reload accuracy: **85.00%**;
- final T2 epoch: 84.67%.

The final R5T result is in the requested **85.0–85.5% acceptable** range. It improves over naive R5 and R5AB, but does not reach the 85.5% strong-success threshold.

## 2. Paper-based design

The implemented thermometer definition is:

- pixel intensity `p` is derived from raw `[0,1]` RGB and conceptually lies in `[0,255]`;
- resolution `R=8`;
- thermometer length `L=ceil(255/8)=32` per RGB channel;
- `n=round(p/R)` ones are placed in a length-32 thermometer vector;
- zeros are mapped to `-1` and ones to `+1`;
- RGB expands from 3 channels to `3 x 32 = 96` bipolar binary channels.

The input to the first convolution is therefore truly `A1 in {-1,+1}`, not normalized CIFAR tensors and not INT8 activations.

The implementation also passes the required reference case:

```text
R=32, p=109 -> round(109/32)=3 ones
```

The R4-to-thermometer kernel projection used for initialization is an engineering warm-start for this project. It is not claimed as a method from the FracBNN paper. The projection matches the RGB-dependent slope of the normalized R4 input and leaves the constant offset for the stem BN to adapt.

## 3. Source R4

The source was the formal R4 `best.pt`, not R5, R5AB, R3, R2, Official ReCU, or R4 `last.pt`:

`experiments/recu_r4/recu_r4_qrprelu_fused_pow2_affine_20260912_110737/best.pt`

| Item | Result |
|---|---:|
| Stored R4 best accuracy | 86.11% |
| Stored R4 best epoch | 98 |
| R4 reload accuracy | 86.11% |
| Source checkpoint | `best.pt` |

The workspace root is not a Git repository; the requested `git status` command returned that status. Existing Official, R1, R2, R3, R4, R5, and R5AB experiments were not deleted or overwritten.

## 4. T1 — FP stem with binary thermometer input

T1 keeps the stem weight floating-point and trains the R4 network to adapt to raw thermometer input. The R4 backbone, QRPReLU, fused power-of-two affine modules, BN, and FC are unchanged.

### T1 conversion and diagnostic

The full-test projected FP-stem zero-epoch accuracy was **84.66%**. The 20-epoch diagnostic reached **85.25%** best/reload at epoch 20, so formal T1 was allowed to proceed.

### T1 formal result

Run directory:

`experiments/recu_r5t/recu_r5t_thermometer_r8_20260912_143431/`

Best checkpoint:

`experiments/recu_r5t/recu_r5t_thermometer_r8_20260912_143431/t1_best.pt`

| Item | Result |
|---|---:|
| Epochs | 100 |
| Best accuracy | **85.29%** |
| Best epoch | 99 |
| Final accuracy | 85.03% |
| Reload accuracy | **85.29%** |
| Training time | 1726.66 s (28 min 46.66 s) |
| Trainable parameters | 284,250 |
| History | 100 records, all finite |

The T1 best checkpoint is the only source used to start T2.

## 5. T2 — W1A1 binary stem

T2 was initialized from the T1 best checkpoint above. It was not restarted from R4 or random initialization.

The stem keeps a floating-point latent weight for the optimizer but uses strict sign projection in inference and forward computation:

```text
W_forward = sign(W_latent) in {-1,+1}
```

### T2 zero-epoch decomposition

| State | Accuracy |
|---|---:|
| T1 best FP-stem accuracy | 85.29% |
| T2 zero-epoch W1A1 accuracy | **18.26%** |
| Binary shock | **-67.03 pp** |

The large initial shock is recovered by T2 fine-tuning; it does not indicate a thermometer encoding or data-pipeline failure.

### T2 diagnostic

The 20-epoch T2 diagnostic reached **84.39%** best/reload at epoch 20. It recovered from 18.26% without NaN/Inf or sign collapse:

- positive latent signs: 6,679;
- negative latent signs: 7,145;
- zero latent weights: 0.

### T2 formal result

Run directory:

`experiments/recu_r5t/recu_r5t_thermometer_r8_20260912_150935/`

Best checkpoint:

`experiments/recu_r5t/recu_r5t_thermometer_r8_20260912_150935/t2_best.pt`

| Item | Result |
|---|---:|
| Epochs | 100 |
| Best binary-stem accuracy | **85.00%** |
| Best epoch | 97 |
| Final binary-stem accuracy | 84.67% |
| Reload binary-stem accuracy | **85.00%** |
| Training time | 1866.78 s (31 min 06.78 s) |
| Final binary mode | enabled |
| Trainable parameters | 284,250 |
| History | 100 records, all finite |

The reloaded best model was explicitly checked with binary mode enabled. Its effective stem weights contain only `{-1,+1}`.

### Final binary stem statistics

| Measurement | Result |
|---|---:|
| Positive latent signs | 6,820 |
| Negative latent signs | 7,004 |
| Zero latent weights | 0 |
| Effective-weight unique values | `{-1.0, +1.0}` |

There is no sign collapse.

## 6. Accuracy comparison

| Model | Stem / input | Accuracy |
|---|---|---:|
| Official ReCU | original | 87.28% |
| R4 | FP RGB stem + multiplier-free binary backbone | 86.11% |
| R5 naive | 3-channel W1A8 stem | 83.77% |
| R5AB | progressive scaled W1A8 stem | 83.96% |
| R5T T1 | thermometer A1 + FP stem | 85.29% |
| **R5T T2** | **thermometer A1 + W1A1 stem** | **85.00%** |

R5T T2 deltas:

- vs Official ReCU: `85.00 - 87.28 = -2.28 pp`;
- vs R4: `85.00 - 86.11 = -1.11 pp`;
- vs naive R5: `85.00 - 83.77 = +1.23 pp`;
- vs R5AB: `85.00 - 83.96 = +1.04 pp`;
- vs T1 FP-stem thermometer: `85.00 - 85.29 = -0.29 pp`.

Therefore, thermometer encoding improves the first-layer binary result substantially relative to both R5 and R5AB, while the remaining binary-stem cost relative to T1 is only 0.29 pp after T2 fine-tuning.

## 7. Storage accounting

### R4 FP stem

```text
3 x 16 x 3 x 3 = 432 weights
432 x 32 bits = 13,824 bits
```

### R5T W1 stem

```text
96 x 16 x 3 x 3 = 13,824 binary weights
13,824 x 1 bit = 13,824 bits
```

The actual model verification reports 13,824 stem weights and confirms storage equivalence at **13,824 bits**. R5T trades 32x more binary weight values for the same raw stem weight storage budget while making the input representation binary and the stem computation BMAC-compatible.

## 8. Compute accounting

The actual model structure was counted, rather than hard-coding the expected total.

### R5T stem

```text
32 x 32 x 16 x 96 x 3 x 3
= 14,155,776 BMAC terms/image
```

These are binary XNOR/popcount terms, not conventional multiplications.

### R4 binary backbone

The 18 R4 binary convolution layers contribute:

```text
40,108,032 binary terms/image
```

### Total R5T convolution terms

```text
14,155,776 + 40,108,032
= 54,263,808 binary terms/image
```

Relative to the R4 binary backbone count, the added thermometer stem contributes a 35.29% increase in binary convolution terms. The R4 first layer was FP/multi-bit and is not included in the 40,108,032 binary-backbone figure.

## 9. Hardware interpretation

R5T uses:

- 96 binary input channels;
- 1-bit stem weights;
- XNOR for the product;
- popcount for accumulation;
- no general multiplier in the stem convolution core.

Because `96 = 3 x 32`, the input is naturally compatible with 32-bit packing:

- 3 packed channel groups per spatial location;
- 9 spatial positions in a 3x3 kernel;
- 27 packed XNOR/popcount groups per output pixel and output filter.

The thermometer stem can share a binary convolution datapath with the R4 binary backbone. The hardware claim supported by this experiment is:

**The R5T stem convolution is a pure binary MAC / XNOR-popcount core with zero general multipliers and zero DSP multiplier usage in that convolution core.**

This does not imply the entire network has DSP=0. Remaining implementation considerations include:

- stem BN;
- head BN;
- final FC;
- QRPReLU and fused affine realization details.

No BN/FC elimination or quantization was performed in this round.

## 10. Verification record

- Correct R4 `best.pt` source confirmed: 86.11%, best epoch 98, reload 86.11%.
- Raw RGB pipeline verified; no CIFAR Normalize tensor is passed to the thermometer encoder.
- R=8, L=32, and 96 input channels verified.
- Thermometer output is strictly bipolar `{-1,+1}`.
- `R=32, pixel=109` reference produces 3 ones.
- `R=8, pixel=255` produces 32 ones.
- R5T stem shape is `16 x 96 x 3 x 3`.
- R4 backbone state is copied and compatibility alpha parameters are frozen.
- CUDA smoke passed for T1 and T2, including forward/backward and save/reload.
- T1 diagnostic and formal runs completed.
- T2 diagnostic and formal runs completed.
- T1 and T2 histories are 100/100 and finite.
- Reloaded T2 effective stem weights are exactly `{-1,+1}`.
- Unit tests: **37/37 passed**.

## 11. Final decision

1. **Did thermometer encoding improve first-layer binary accuracy?** Yes. R5T T2 reaches 85.00%, improving over R5 by 1.23 pp and R5AB by 1.04 pp.
2. **Is R5T better than R5/R5AB?** Yes.
3. **Did it reach 85.5%?** No. It reaches the acceptable 85.0–85.5% range, not the strong-success range.
4. **Is the stem pure BMAC?** Yes, for the convolution core: binary input, binary weight, XNOR/popcount, no general multiplier.
5. **Is a later R=16 study worthwhile?** Yes, as a separate controlled follow-up because R5T is close to the strong-success threshold. It was not started here.
6. **Should BN/FC elimination start now?** No. This round stops after T1/T2 and this report, as requested.

No R=4, R=16, fractional activation, 2-bit activation, fused stem-BN quantization, final-FC binarization, RTL, FPGA synthesis, T18, or T90 experiment was started.
