# R5AB Final Report — Progressive Scaled W1A8 Stem

Date: 2026-09-12  
Environment: `KenBnn_env`, PyTorch 2.5.1+cu121, RTX 4070  
Dataset protocol: official ReCU CIFAR-10 train/test split  

**All reported R5AB deployment accuracy uses lambda = 1.**

## A. Source R4

The experiment was warm-started directly from the formal R4 `best.pt`:

`experiments/recu_r4/recu_r4_qrprelu_fused_pow2_affine_20260912_110737/best.pt`

| Item | Result |
|---|---:|
| R4 best accuracy | 86.11% |
| R4 best epoch | 98 |
| R4 reload accuracy | 86.11% |
| R4 checkpoint selected | `best.pt` |

R5, R3, R2, Official ReCU, and any R4 `last.pt` were not used as the R5AB source. The workspace root is not a Git repository, so the requested `git status` audit returned “not a git repository”; no existing experiment directory was deleted or overwritten.

## B. R5AB definition

Only the stem training rule was changed relative to R5:

```text
alpha_c = mean(abs(W_c))
B_c     = sign(W_c)
W_eff   = (1 - lambda) * W + lambda * alpha_c * B_c
lambda  = min((epoch_index + 1) / 30, 1)
```

At inference and deployment evaluation, the model is always forced to:

```text
lambda = 1
W_deploy = alpha_c * sign(W_c)
```

The activation path is unchanged from R5:

```text
signed INT8, fixed scale = 2^-5 = 0.03125
```

No R4 backbone, final FC, BN, or activation quantizer was changed. R4 compatibility alpha tensors remain in the state interface but are frozen; the final model has 270,858 trainable parameters.

## C. Conversion and zero-epoch decomposition

Full-test zero-epoch conversion results:

| State | Meaning | Accuracy |
|---|---|---:|
| lambda = 0 | A8 input + original FP latent stem | **85.80%** |
| lambda = 1 | A8 input + scaled W1A8 stem | **32.06%** |
| Binary shock | `32.06 - 85.80` | **-53.74 pp** |

The lambda=0 result is only 0.31 pp below the R4 source, isolating the large initial loss to the W1A8 stem projection rather than the A8 input quantizer. Training later recovers most of this shock.

### A8 verification

| Measurement | Result |
|---|---:|
| Scale exponent | -5 |
| Scale | 0.03125 |
| Binary stem values | exactly `{-1, +1}` before alpha scaling |
| Saturation fraction | **0.0** |
| Required limit | `<= 1e-4` |

The A8 quantizer was not changed. The normalized CIFAR-10 input protocol and the previously verified observed range remain in use; no saturation adjustment was needed.

### Initial stem statistics

| Measurement | Result |
|---|---:|
| Latent minimum | -0.57664508 |
| Latent maximum | 0.57905954 |
| Latent mean | -0.00134845 |
| Latent standard deviation | 0.21318990 |
| Binary positive | 199 |
| Binary negative | 233 |
| Alpha minimum | 0.13027808 |
| Alpha maximum | 0.22794464 |
| Alpha mean | 0.16505811 |
| Alpha median | 0.16626321 |

## D. Progressive diagnostic

The 20-epoch diagnostic used a 30-epoch ramp, so lambda was below 1 throughout this diagnostic. Both accuracy types were recorded:

- `scheduled_test_acc`: the current training lambda;
- `binary_lambda1_test_acc`: forced deployment lambda=1, used for best-checkpoint selection.

| Epoch | Lambda | Scheduled | Forced lambda=1 |
|---:|---:|---:|---:|
| 1 | 0.0333 | 72.55% | 33.17% |
| 2 | 0.0667 | 77.24% | 41.05% |
| 3 | 0.1000 | 76.93% | 53.90% |
| 4 | 0.1333 | 78.40% | 56.76% |
| 5 | 0.1667 | 75.38% | 54.99% |
| 6 | 0.2000 | 79.52% | 48.80% |
| 7 | 0.2333 | 74.97% | 46.57% |
| 8 | 0.2667 | 78.01% | 62.99% |
| 9 | 0.3000 | 78.74% | 51.64% |
| 10 | 0.3333 | 78.97% | 65.81% |
| 11 | 0.3667 | 78.96% | 50.21% |
| 12 | 0.4000 | 78.53% | 49.11% |
| 13 | 0.4333 | 78.71% | 59.05% |
| 14 | 0.4667 | 77.43% | 63.13% |
| 15 | 0.5000 | 77.15% | 65.09% |
| 16 | 0.5333 | 78.11% | 63.04% |
| 17 | 0.5667 | 74.62% | **69.85%** |
| 18 | 0.6000 | 73.50% | 68.25% |
| 19 | 0.6333 | 77.15% | 67.41% |
| 20 | 0.6667 | 69.70% | 61.06% |

The forced deployment accuracy was noisy but recovered from 33.17% to 69.85% and did not show continuous deterioration, NaN/Inf, sign collapse, or INT8 saturation. Formal training was therefore allowed to proceed.

## E. Formal deployment result

Formal run directory:

`experiments/recu_r5ab/recu_r5ab_progressive_scaled_w1a8_20260912_132942/`

Best checkpoint:

`experiments/recu_r5ab/recu_r5ab_progressive_scaled_w1a8_20260912_132942/best.pt`

| Item | Result |
|---|---:|
| Epochs | 100 |
| Ramp epochs | 30 |
| Best forced lambda=1 accuracy | **83.96%** |
| Best epoch | **100** |
| Final forced lambda=1 accuracy | **83.96%** |
| Reload forced lambda=1 accuracy | **83.96%** |
| Final lambda | exactly 1.0 |
| Training time | 1677.10 s (27 min 57.10 s) |
| Trainable parameters | 270,858 |

The best checkpoint metadata records `selection_lambda=1.0`. Reload verification reproduced 83.96%. The 100-record history is finite, and all 672 R4 compatibility alpha parameters are frozen after reconstructing the model from the R4 source.

Training settings were unchanged from the requested configuration: SGD, learning rate 0.005, momentum 0.9, weight decay 5e-4, cosine schedule, ReCU tau 0.99, and A8 scale `2^-5`.

## F. Comparisons

| Model | Accuracy | Delta vs R4 |
|---|---:|---:|
| Official ReCU | 87.28% | +1.17 pp |
| R4 fused pow2 affine | 86.11% | reference |
| R5 naive W1A8 | 83.77% | -2.34 pp |
| R5AB progressive scaled W1A8 | **83.96%** | **-2.15 pp** |

R5AB deltas:

- vs Official ReCU: `83.96 - 87.28 = -3.32 pp`;
- vs R4: `83.96 - 86.11 = -2.15 pp`;
- vs naive R5: `83.96 - 83.77 = +0.19 pp`;
- recovered accuracy: **+0.19 pp**;
- original R5 loss vs R4: `86.11 - 83.77 = 2.34 pp`;
- recovery ratio: `0.19 / 2.34 = 8.12%`.

This is a real but small recovery over naive R5. It is below the 84.5% threshold for advancing to the next hardware-elimination stage and is not a strong or acceptable final result under the requested criteria.

### Final stem statistics

| Measurement | Result |
|---|---:|
| Latent minimum | -0.78387439 |
| Latent maximum | 0.71977013 |
| Latent mean | 0.00532666 |
| Latent standard deviation | 0.22532982 |
| Binary positive | 211 |
| Binary negative | 221 |
| Alpha minimum | 0.10385700 |
| Alpha maximum | 0.21595025 |
| Alpha mean | 0.15057454 |
| Alpha median | 0.15075168 |

The final binary signs remain balanced and no sign collapse occurred.

### Float alpha+BN folding statistics

For the stem BN, the exact float fold is:

```text
K = gamma * alpha / sqrt(running_var + epsilon)
B = beta - gamma * running_mean / sqrt(running_var + epsilon)
```

At the formal best checkpoint:

| Measurement | Result |
|---|---:|
| K minimum | 0.12090071 |
| K maximum | 0.26253557 |
| Mean absolute K | 0.17772204 |
| B minimum | -0.70567101 |
| B maximum | 0.19429940 |

These are float fused coefficients only. K was not quantized to power-of-two in R5AB.

## G. Hardware conclusion

1. Progressive mixing is **training-only**. It does not imply an FP branch, lambda multiplier, mixer, or interpolation datapath in deployment.
2. At inference, lambda is exactly 1 and the stem deploy weight is `alpha * sign(W)`.
3. The W1A8 convolution core receives signed INT8 `q` and computes only `+q` or `-q`, followed by accumulation. Thus:

   **W1A8 stem convolution core general multiplier count = 0.**

4. Per-filter alpha can be exactly folded into the stem BN as the float K/B coefficients above.
5. Because K has not been quantized to power-of-two, R5AB cannot claim the entire stem is multiplier-free and cannot claim the entire network has DSP=0.

The current claim is limited to:

- the W1A8 stem convolution core is general-multiplier-free;
- per-filter alpha can be folded into stem BN;
- fused-K power-of-two quantization has not yet been performed.

## H. Verification record and stopping point

- Correct R4 `best.pt` source confirmed: 86.11%, epoch 98, reload 86.11%.
- R5AB conversion verification passed.
- Binary stem signs are strictly `+1/-1`.
- A8 saturation fraction is 0.0.
- CUDA smoke passed: forward, backward, finite loss, alpha/latent finiteness, save/reload, forced lambda=1 evaluation.
- 20-epoch scheduled-vs-forced diagnostic completed.
- Formal 100-epoch run completed.
- Formal best/reload deployment accuracy is 83.96% at lambda=1.
- Unit tests after the R5AB audit patch: **26/26 passed**.

Stop here. Do not start fused stem K power-of-two QAT, final FC quantization, head BN elimination, RTL, FPGA synthesis, or T18/T90 from this result.
