# R7 Final Report — R6 + Head Signed-Pow2 Affine

Date: 2026-09-12  
Device: NVIDIA RTX 4070, conda environment `KenBnn_env`  
Dataset: CIFAR-10

## A. Configuration

R7 keeps the R6 model and changes only the final head:

```text
GAP → signed-pow2 affine (Khat, float B) → unchanged FC
```

The standalone head `BatchNorm1d(64)` is removed. The FC weights, FC bias,
dimension, and all R4 backbone parameters are retained.

| Item | Setting |
|---|---|
| Thermometer resolution | R=8 |
| Input | raw RGB [0,1] → Thermometer → bipolar {-1,+1} |
| Thermometer channels | 3×32 = 96 |
| Stem | W1A1, signed-pow2 affine, Conv bias=None |
| Backbone | R4 unchanged |
| Head K | signed power-of-two, no exponent clamp |
| Head B | floating point |
| Batch size | 128 |
| Formal epochs | 100 |
| Optimizer | SGD, momentum=0.9 |
| Weight decay | 0 |
| Scheduler | cosine, T_max=100 |
| ReCU tau | 0.99 |
| Formal executed LR | 1e-4 |

### Stability correction

The requested initial LR was 1e-3. A controlled probe showed a genuine
training collapse after removing train-mode BN2: the first full-network step
dropped test accuracy from 75.80% to 48.39%. A 20-epoch diagnostic with LR=1e-4
was stable and reached 83.46%. Therefore the formal run kept SGD, momentum,
weight decay, cosine scheduling, data, and architecture unchanged, but used
LR=1e-4 as the necessary post-BN-removal stability correction. This deviation
is recorded explicitly rather than hidden.

## B. Source and verification

Required source:

```text
experiments/recu_r6/recu_r6_r5tlong_stem_bn_pow2_20260912_203911/best.pt
```

Source verification:

- Stored R6 best accuracy: 85.52%
- Stored best epoch: 97
- Reload accuracy: 85.52%
- Thermometer input: 96 channels
- Stem effective weights: {-1,+1}
- Stem exponents: {-6,-5,-4}
- Stem Conv bias: absent
- Stem BN1: absent
- Frozen R4 compatibility alpha parameters: 672 values / 18 tensors

Exact head BN fold was checked over eight CIFAR-10 test batches:

```text
max_abs(GAP→BN2, GAP→fused affine) = 1.430511474609375e-06
fold tolerance = 1e-5
```

The fold passed.

## C. R7 formal result

Formal run directory:

```text
experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/
```

| Metric | Accuracy |
|---|---:|
| R6 source reload | 85.52% |
| R7 zero-epoch hard projection | 75.80% |
| R7 zero epoch vs R6 | -9.72 pp |
| R7 best | **85.22%** |
| Best epoch | **99** |
| R7 final epoch 100 | 85.20% |
| R7 best checkpoint reload | **85.22%** |
| R7 best vs R6 | -0.30 pp |

Independent reload verification reproduced the full test accuracy of 85.22%.

Final R7 invariants:

- Stem signs: +1 = 6,875; -1 = 6,949
- Stem K exponents: -6: 3, -5: 7, -4: 6
- Head K: 64/64 positive signed-pow2 values
- Head K exponents: 1: 2, 2: 4, 3: 19, 4: 30, 5: 9
- Head K effective values: {2, 4, 8, 16, 32}
- Final FC bias: present
- Standalone BN2: removed
- Trainable parameters: 284,250
- Frozen parameters: 672, all R4 compatibility alpha
- NaN/Inf: none

## D. Reference comparison

| Reference | Accuracy | R7 best delta |
|---|---:|---:|
| Official ReCU | 87.28% | -2.06 pp |
| R4 | 86.11% | -0.89 pp |
| R6 | 85.52% | -0.30 pp |
| R5T-Long | 85.14% | +0.08 pp |
| R5AB | 83.96% | +1.26 pp |
| R5 | 83.77% | +1.45 pp |

## E. Curve interpretation

1. The head-only hard projection starts at 75.80%, so signed-pow2 head
   quantization creates a large initial shock of -9.72 pp relative to R6.
2. The 100-epoch schedule was useful: the stable 20-epoch diagnostic reached
   83.46%, while the completed 100-epoch run reached 85.22% (+1.76 pp).
3. The best result occurred at epoch 99, very close to the schedule end. The
   model was still improving late in training; 100 epochs was not obviously
   excessive.
4. R7 best accuracy is 85.22%, just above the 85.20% very-good threshold. It is
   below R6 by 0.30 pp, so it does not replace R6 as the highest-accuracy main
   line, but it is a viable signed-pow2-head variant.
5. The R7 result is in the 85.2–85.5 range and does not justify additional
   R7-specific training tricks under the current stopping rule.

## F. Hardware and cost

Inference architecture and BMAC count remain unchanged from R5T/R6:

```text
Stem BMAC/image     = 14,155,776
Backbone BMAC/image = 40,108,032
Total BMAC/image    = 54,263,808
```

Hardware interpretation:

- Stem weights and activations remain binary and use XNOR + popcount.
- Stem Khat uses signed power-of-two shift/optional negate plus B add.
- R7 head Khat uses signed power-of-two shift/optional negate plus floating B
  add before the unchanged FC.
- No extra BMAC, parameters, or inference cost is introduced by longer
  training.
- This report does **not** claim DSP-free operation for the whole network;
  the unchanged FC and other non-binary operations remain outside the stem
  XNOR/popcount claim.

## G. Tests and artifacts

- Unit tests: 58 passed.
- Exact fold verifier: passed.
- Formal run: completed 100/100 epochs.
- Best checkpoint reload: passed at 85.22%.
- Signed-pow2, binary stem, FC bias, no-BN2, finite-parameter, and frozen-alpha
  checks: passed.

Primary artifacts:

```text
R7_FINAL_REPORT.md
experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/best.pt
experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/last.pt
experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/history.json
experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/summary.json
experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/conversion.json
```

R7 is complete. No R8, KD, FC quantization, RTL, or FPGA work was started.
