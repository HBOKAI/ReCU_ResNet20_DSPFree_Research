# R6 Final Report

Date: 2026-09-12

## A. Source checkpoint

R6 was initialized only from the R5T-Long T2 best checkpoint:

`experiments/recu_r5t_long/recu_r5t_thermometer_r8_long_20260912_171239/t2_best.pt`

Source verification:

- Stored best accuracy: **85.14%**
- Stored best epoch: **200**
- Source full reload accuracy: **85.14%**
- Thermometer: `R=8`, `L=32`, `96` input channels
- Stem: binary W1A1 `3x3`, `96 -> 16`, bias `None`
- Source effective stem values: `{-1,+1}`
- Source stem signs: `+1=6839`, `-1=6985`

The old R5T T2, T1, R5, R5AB, and R4 checkpoints were not used as the R6 source.

The workspace is not a Git repository (`git status` returned “not a git repository”), so no Git commit is available for this run. Existing R5T/R5T-Long artifacts were kept intact; R6 uses the independent files and output directory listed below.

## B. R6 architecture and implementation

R6 changes only the R5T-Long stem BN path:

```text
RGB [0,1]
-> Thermometer R=8, bipolar {-1,+1}
-> W1A1 Conv 96->16, bias=None
-> signed-pow2 K * S + float B
-> Hardtanh
```

The R4 backbone, QRPReLU, backbone fused pow2 affine units, GAP, head BN, and final FC are unchanged.

Independent R6 files:

- `recu_hw/r6.py`
- `train_recu_r6.py`
- `tools/verify_r6_folding.py`
- `configs/recu_r6.json`
- `tests/test_r6.py`

Only `K` is quantized:

```text
Khat = sign(K_latent) * 2^round(log2(abs(K_latent)))
```

`B` remains floating-point/reference and was not quantized, removed, or fused into another structure. The R6 quantizer uses the repository's existing `round_ste` semantics, aligned with the R4 signed-pow2 implementation. No exponent clamp was used (`min_exp=null`, `max_exp=null`).

## C. Exact BN fold

For each stem channel:

```text
K = gamma / sqrt(running_var + eps)
B = beta - gamma * running_mean / sqrt(running_var + eps)
```

Comparison over 8 CIFAR-10 test batches:

```text
BN(BinaryConv(x))
versus
K * BinaryConv(x) + B
```

- Maximum absolute error: **9.5367431640625e-07**
- Required tolerance: `< 1e-5`
- Result: **PASS**

Training was allowed only after this exact-fold check passed.

## D. Folded K/B statistics

### D1. Source folded values before fine-tuning

| Quantity | Value |
|---|---:|
| K channels | 16 |
| K min | 0.0133008622 |
| K max | 0.0407633148 |
| abs(K) min | 0.0133008622 |
| abs(K) max | 0.0407633148 |
| mean(abs(K)) | 0.0283208359 |
| positive K | 16 |
| negative K | 0 |
| B min | -0.6572578549 |
| B max | 0.2314942628 |
| mean(B) | -0.1956568360 |

Initial signed-pow2 exponent histogram:

| Rounded exponent | Count |
|---:|---:|
| -6 | 3 |
| -5 | 13 |

Initial Khat absolute values were `{2^-6, 2^-5}` = `{0.015625, 0.03125}`.

### D2. Final best checkpoint values

| Quantity | Value |
|---|---:|
| K channels | 16 |
| K min | 0.0219314061 |
| K max | 0.0442362092 |
| abs(K) min | 0.0219314061 |
| abs(K) max | 0.0442362092 |
| mean(abs(K)) | 0.0358579755 |
| positive K | 16 |
| negative K | 0 |
| B min | -0.4436965585 |
| B max | 0.1852049381 |
| mean(B) | -0.0907891989 |

Final signed-pow2 exponent histogram:

| Rounded exponent | Count |
|---:|---:|
| -6 | 1 |
| -5 | 13 |
| -4 | 2 |

Final unique exponents are `{-6,-5,-4}`. Therefore final Khat absolute values are `{0.015625, 0.03125, 0.0625}`, all exact powers of two.

## E. Hard projection and training

### E1. Zero-epoch hard projection

Before any R6 fine-tuning:

- Source R5T-Long accuracy: **85.14%**
- R6 hard K projection accuracy: **84.90%**
- Projection delta: **-0.24 pp**

### E2. Diagnostic run

A separate 20-epoch diagnostic was run from the source checkpoint. It completed without NaN/Inf, exponent explosion, binary collapse, or frozen-parameter changes.

- Zero-epoch best/reload: 84.90%
- Diagnostic final accuracy: 81.18%
- Diagnostic checkpoint was not used for formal training.

### E3. Formal 100-epoch run

Run directory:

`experiments/recu_r6/recu_r6_r5tlong_stem_bn_pow2_20260912_203911`

Recipe:

- Epochs: `100`
- Optimizer: SGD
- Initial LR: `1e-3`
- Momentum: `0.9`
- Weight decay: `0`
- Scheduler: cosine annealing over 100 epochs
- ReCU tau: `0.99`
- Source: R5T-Long T2 `t2_best.pt`
- Best selection: true R6 inference with W1A1 stem, signed-pow2 K, float B, and no stem BN

Results:

- Best accuracy: **85.52%**
- Best epoch: **97**
- Final epoch accuracy: **85.10%**
- Reload accuracy: **85.52%**
- Training time: `2969.17 s` (~49 min 29 s)

Best checkpoint:

`experiments/recu_r6/recu_r6_r5tlong_stem_bn_pow2_20260912_203911/best.pt`

## F. Invariants and reload verification

The final best checkpoint was independently reloaded and evaluated over the full CIFAR-10 test set.

- Full independent reload accuracy: **85.52%**
- Checkpoint epoch: `97`
- Checkpoint best accuracy: `85.52%`
- K shape: `[16]`
- B shape: `[16]`
- Reload effective K unique values: `{0.015625, 0.03125, 0.0625}`
- Khat exact signed-pow2: **PASS**
- Stem effective weight unique values: `{-1,+1}`
- Final stem signs: `+1=6882`, `-1=6942`, `zero_latent=0`
- Stem Conv bias: `None`
- Standalone `bn1`: absent
- K/B/model values: finite
- Forward output shape: `[N,10]`
- Trainable parameters: `284,250`
- Frozen parameters: `672`
- Frozen category: `r4_compatibility_alpha`
- Frozen alpha tensors: `18`

Frozen alpha names:

```text
layer1.0.conv1.alpha, layer1.0.conv2.alpha
layer1.1.conv1.alpha, layer1.1.conv2.alpha
layer1.2.conv1.alpha, layer1.2.conv2.alpha
layer2.0.conv1.alpha, layer2.0.conv2.alpha
layer2.1.conv1.alpha, layer2.1.conv2.alpha
layer2.2.conv1.alpha, layer2.2.conv2.alpha
layer3.0.conv1.alpha, layer3.0.conv2.alpha
layer3.1.conv1.alpha, layer3.1.conv2.alpha
layer3.2.conv1.alpha, layer3.2.conv2.alpha
```

CUDA smoke passed forward/backward, finite loss, Khat exact-power-of-two, binary stem, and checkpoint save/reload checks. Final unit tests:

```text
Ran 48 tests ... OK
```

## G. Accuracy comparison

The required final metric is R6 best reload accuracy = **85.52%**.

| Reference | Calculation | Delta |
|---|---:|---:|
| R5T-Long | `85.52 - 85.14` | **+0.38 pp** |
| Old R5T | `85.52 - 85.00` | **+0.52 pp** |
| R5 | `85.52 - 83.77` | **+1.75 pp** |
| R5AB | `85.52 - 83.96` | **+1.56 pp** |
| R4 | `85.52 - 86.11` | **-0.59 pp** |
| Official ReCU | `85.52 - 87.28` | **-1.76 pp** |

R6 reaches the requested `>=84.8%` “very good” range. The exact fold itself is lossless within numerical tolerance; the initial hard K projection costs 0.24 pp, and fine-tuning recovers that cost and adds 0.38 pp over R5T-Long.

## H. Hardware interpretation

R6 does not change the convolution topology or BMAC count from R5T:

- Stem convolution: `14,155,776` BMAC/image
- R4 binary backbone: `40,108,032` BMAC/image
- Total convolution: `54,263,808` BMAC/image

Stem inference can be interpreted as:

```text
Binary S
-> XNOR + popcount convolution
-> signed arithmetic shift by exponent k
-> optional negate from sign(Khat)
-> channel-wise constant add B
-> Hardtanh
```

Therefore R6 supports the following scoped claim:

> The stem convolution plus normalization-scale path is general-multiplier-free.

The final checkpoint has only three exponent values (`-6`, `-5`, `-4`), so hardware can use fixed shift paths plus a small exponent-selection mux instead of a wide barrel shifter. This checkpoint has all positive K signs, but the implementation preserves signed-K support and optional negate.

This report does **not** claim that the entire network is DSP=0. Stem B remains a float/reference representation, and the head BN and final FC were not quantized or folded in this R6 experiment.

## I. Stop condition

R6 implementation, exact fold verification, smoke, diagnostic, formal training, reload verification, tests, and final report are complete. No B quantization, head BN folding, FC quantization, RTL, FPGA synthesis, T18/T90, R=16, FracConv, or additional training was started.
