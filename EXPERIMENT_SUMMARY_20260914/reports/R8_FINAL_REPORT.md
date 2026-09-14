# R8 Final Report — R7 + Quantized FC Comparison

Date: 2026-09-13  
Device: NVIDIA RTX 4070, conda environment `KenBnn_env`  
Dataset: CIFAR-10

## 1. Source and common configuration

Both experiments start from the same formal R7 best checkpoint:

```text
experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/best.pt
```

Source verification:

- R7 stored best: 85.22% @ epoch 99
- R7 reload: 85.22%
- Thermometer: R=8, 96 bipolar input channels
- Stem: W1A1, strict {-1,+1}, signed-pow2 affine
- R4 backbone unchanged
- Head signed-pow2 affine unchanged
- Standalone BN2: absent
- Frozen R4 compatibility alpha: 672 values / 18 tensors

Common training recipe:

| Item | Setting |
|---|---|
| Epochs | 100 |
| Batch size | 128 |
| Optimizer | SGD |
| Initial LR | 1e-4 |
| Momentum | 0.9 |
| Weight decay | 0 |
| Scheduler | cosine |
| ReCU tau | 0.99 |
| FC bias | enabled |
| FC exponent clamp | none for R8B |

R8A and R8B were trained concurrently on the RTX 4070. Peak combined GPU
memory was approximately 4.1 GiB of 12.3 GiB, with no CUDA OOM or driver
instability.

## 2. R8A — W1 FC

Classifier forward:

```text
y_c = sum_j sign(W_cj) * x_j + b_c
```

Formal run:

```text
experiments/recu_r8a/recu_r8a_r7_w1_fc_20260912_225827/
```

| Metric | Result |
|---|---:|
| Zero-epoch projection | 69.06% |
| Zero-epoch shock vs R7 | -16.16 pp |
| Best accuracy | **82.84%** |
| Best epoch | **95** |
| Final epoch 100 | 81.96% |
| Best checkpoint reload | **82.84%** |
| Best vs R7 | -2.38 pp |
| Training time | 3927.12 s / 65.45 min |

FC invariants at the best checkpoint:

- Effective FC weights: exactly {-1,+1}
- FC weights: 640
- Positive signs: 229
- Negative signs: 411
- FC bias: present
- FC weight storage: 640 bits = 80 bytes
- FP32 bias storage: 320 bits = 40 bytes
- Estimated FC classifier storage: 960 bits = 120 bytes

Fine-tune recovery from the zero-epoch projection was:

```text
82.84 - 69.06 = +13.78 percentage points
```

## 3. R8B — Signed-Pow2 FC

Classifier forward:

```text
y_c = sum_j sign(W_cj) * 2^k_cj * x_j + b_c
```

Formal run:

```text
experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/
```

| Metric | Result |
|---|---:|
| Zero-epoch projection | 84.35% |
| Zero-epoch shock vs R7 | -0.87 pp |
| Best accuracy | **85.40%** |
| Best epoch | **100** |
| Final epoch 100 | **85.40%** |
| Best checkpoint reload | **85.40%** |
| Best vs R7 | **+0.18 pp** |
| Training time | 3937.01 s / 65.62 min |

FC invariants at the best checkpoint:

- Effective FC weights: all exact signed powers of two
- FC weights: 640
- Positive weights: 241
- Negative weights: 399
- FC bias: present
- Exponent clamp: none
- Exponent range: -9 to 0
- Unique exponent count: 10
- Signed exponent storage width used for estimate: 5 bits
- Weight storage: 640 × (1 sign bit + 5 exponent bits) = 3840 bits
- FP32 bias storage: 320 bits
- Estimated FC classifier storage: 4160 bits = 520 bytes

Exponent histogram:

| Exponent | Count |
|---:|---:|
| -9 | 4 |
| -8 | 7 |
| -7 | 10 |
| -6 | 22 |
| -5 | 54 |
| -4 | 74 |
| -3 | 162 |
| -2 | 181 |
| -1 | 97 |
| 0 | 29 |

Fine-tune recovery from the zero-epoch projection was:

```text
85.40 - 84.35 = +1.05 percentage points
```

## 4. Direct comparison

| Quantity | R8A W1 FC | R8B signed-Pow2 FC |
|---|---:|---:|
| Zero epoch | 69.06% | 84.35% |
| Best | 82.84% | **85.40%** |
| Best epoch | 95 | 100 |
| Final | 81.96% | **85.40%** |
| Reload | 82.84% | **85.40%** |
| Delta vs R7 85.22% | -2.38 pp | **+0.18 pp** |
| FC weight storage | 80 bytes | 480 bytes |
| FC + FP32 bias estimate | 120 bytes | 520 bytes |
| General multiplier | 0 | 0 |
| Shift hardware | none | required |

R8B is 2.56 pp better than R8A in best accuracy, while still eliminating the
general multiplier from the FC. R8A has the minimum storage and simplest
add/subtract-only datapath, but its accuracy loss is substantial.

## 5. Hardware interpretation

### R8A

- FC multiplication becomes add/subtract selected by the binary sign.
- General multiplier count: 0.
- Weight storage: 640 bits.
- Simplest FC RTL and smallest classifier storage.

### R8B

- Each FC term uses shift + optional negate + accumulate.
- General multiplier count: 0.
- Weight storage uses one sign bit plus a 5-bit signed exponent in this run.
- FC classifier storage is about 520 bytes including FP32 bias.
- Datapath control is more complex than R8A because of per-weight shifts.

The stem, head affine, R4 backbone, and BMAC counts remain unchanged from the
R7 hardware scope:

```text
Stem BMAC/image     = 14,155,776
Backbone BMAC/image = 40,108,032
Total BMAC/image    = 54,263,808
```

This does not claim that every operation in the complete network is DSP-free;
the statement is specifically that the R8 FC has no general multiplier and the
binary stem remains XNOR + popcount.

## 6. Recommendation

### Recommended final RTL: R8B

Choose **R8B** for the final RTL mainline. It improves over R7 by 0.18 pp and
over R8A by 2.56 pp, while retaining zero general multipliers. Its additional
cost is limited to per-term sign/exponent storage and shift control: about 520
bytes for the FC classifier including FP32 bias, versus about 120 bytes for
R8A.

Keep **R8A** as the ultra-minimal hardware fallback when the absolute simplest
add/sub-only FC and minimum storage are more important than accuracy.

## 7. Verification and artifacts

- Full unit tests: **62/62 passed**.
- R8A smoke: passed; 640 binary FC weights and bias verified.
- R8B smoke: passed; 640 signed-pow2 FC weights, exponent histogram, no clamp,
  and bias verified.
- Formal R8A: 100/100 epochs completed.
- Formal R8B: 100/100 epochs completed.
- Independent R8A reload verifier: passed at 82.84%.
- Independent R8B reload verifier: passed at 85.40%.
- Stem/Head pow2, binary stem, FC bias, no BN2, finite parameters, and frozen
  alpha checks: passed for both branches.

Primary artifacts:

```text
R8_FINAL_REPORT.md
experiments/recu_r8a/recu_r8a_r7_w1_fc_20260912_225827/best.pt
experiments/recu_r8a/recu_r8a_r7_w1_fc_20260912_225827/last.pt
experiments/recu_r8a/recu_r8a_r7_w1_fc_20260912_225827/history.json
experiments/recu_r8a/recu_r8a_r7_w1_fc_20260912_225827/summary.json
experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt
experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/last.pt
experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/history.json
experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/summary.json
```

R8A and R8B are complete. No bias quantization, FC bias removal, RTL,
FPGA synthesis, T18/T90, KD, or longer training was started.
