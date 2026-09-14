# ReCU R1 / R2 / R3 Warm-start Hardware Ablation Report

Date: 2026-09-12  
Environment: `KenBnn_env`, PyTorch 2.5.1+cu121, NVIDIA GeForce RTX 4070  
Protocol: CIFAR-10, Official ReCU data protocol, 100-epoch warm-start fine-tuning, SGD lr=0.01, momentum=0.9, weight decay=5e-4, cosine scheduler, tau fixed at 0.99.

## A. Official source

Source checkpoint:

`C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_runs\recu_resnet20_official_repro_s123_20260911_234802\best.pt`

- best epoch: 600
- stored best/reload accuracy: 87.28%
- checkpoint config: `resnet20_1w1a`, activation `prelu`, alpha `float`, 600 epochs, tau 0.85→0.99 official schedule
- confirmed as the successful Official ReCU reproduction

The workspace has no root `AGENTS.md` and is not a Git repository, so `git status`/commit provenance was unavailable. The existing `AGENTS_RECU_APPEND.md` was read. A Windows-only DataLoader fallback (`workers=0`) was added; model topology and data transforms were unchanged.

## B. Final accuracy results

All accuracy deltas below use the stored unrounded values before display rounding.

| Model | Initial converted | Best | Best epoch | Final | Reload | Δ vs Official |
|---|---:|---:|---:|---:|---:|---:|
| Official ReCU | — | 87.28% | 600 | — | 87.28% | 0.00 pp |
| R1 Remove alpha | 8.44% | 87.02% | 100 | 87.02% | 87.02% | −0.26 pp |
| R2 Warm-start QRPReLU | 11.86% | 86.72% | 97 | 86.30% | 86.72% | −0.56 pp |
| R3 Fused pow2 affine | 73.34% | 85.50% | 100 | 85.50% | 85.50% | −1.78 pp |

All three formal histories contain 100 epochs and finite loss/accuracy values. The formal run directories are:

- R1: `experiments/recu_r1_remove_alpha/recu_r1_remove_alpha_20260912_094409`
- R2: `experiments/recu_r2_warmstart_qrprelu/recu_r2_warmstart_qrprelu_20260912_094409`
- R3: `experiments/recu_r3_fused_pow2_affine/recu_r3_fused_pow2_affine_20260912_094409`

## C. R1 — Remove alpha

The inference function is fixed to `alpha=1`; the 672-element alpha tensors remain in `state_dict` only for checkpoint compatibility and are frozen (`requires_grad=False`). Trainable parameters: 270,186.

- initial converted accuracy: 8.44%
- best/final/reload: 87.02% / 87.02% / 87.02%
- best epoch: 100
- delta vs Official: −0.26 percentage points

Conclusion: alpha is not safe to remove as a zero-shot drop-in conversion: the immediate accuracy loss is 78.84 pp. After warm-start fine-tuning, R1 recovers to within 0.26 pp of Official ReCU. Therefore alpha can be removed for a retrained hardware variant, but not by checkpoint surgery alone.

## D. R2 — Warm-start QRPReLU

Official PReLU slope statistics over 336 channels:

- min: −0.918749
- max: 0.834225
- mean: 0.228508
- median: 0.340588
- population std: 0.315575
- `p_i <= 0`: 89
- `p_i < 2^-8`: 96

Initialization used `a_i=log2(max(p_i, 2^-16))`, `xi1=0`, `xi2=0`, and the inference slope `2^round(a_i)`. The initial exponent histogram after the required floor was:

`-16:94, -12:1, -9:1, -5:1, -2:86, -1:147, 0:6`

For the 247 positive slopes, projection error was mean absolute 0.076043, mean relative 4.293596, and maximum relative 959.8926. The large relative-error maximum is caused by very small positive slopes near/below the `2^-16` floor. Using all channels after the initialization floor gives mean absolute error 0.055901, mean relative error 0.141113, and maximum relative error 0.413071.

The 89 non-positive slopes are a representational mismatch: positive power-of-two QRPReLU slopes cannot represent them. They were not silently ignored; they were clamped only during initialization. Their layer/channel locations are:

- `layer1.0.post_act`: channels 4, 9, 13
- `layer1.1.post_act`: channels 2, 3, 4, 7, 8, 11, 15
- `layer1.2.post_act`: channels 1, 9
- `layer2.0.post_act`: channels 4, 24, 30
- `layer2.1.post_act`: channels 6, 22, 31
- `layer3.1.post_act`: channels 1, 9, 18, 24, 25, 28, 45
- `layer3.2.post_act`: channels 0–63 (all 64 channels)

Training result:

- trainable parameters: 271,530
- initial converted accuracy: 11.86%
- best/final/reload: 86.72% / 86.30% / 86.72%
- best epoch: 97
- delta vs Official: −0.56 pp
- delta vs old H1 84.28%: +2.44 pp

The final QRPReLU exponent histogram was `-10:94, -7:1, -6:1, -3:1, -1:211, 0:28`.

Conclusion: warm-start projection clearly improves over old H1 (84.28%→86.72%) and is worth keeping as a hardware-aware activation candidate, although the 89 non-positive official slopes remain an explicit representational mismatch.

## E. R3 — Fold alpha + binary-branch BN, then quantize K

The float fold was verified before training:

`BN(alpha*S) == K*S+B`

- maximum absolute logit error: 2.74e-6 in the final audit; repeated audit maximum observed: 3.34e-6
- required threshold: 1e-4
- result: PASS

The initial 672 fused coefficients used `K_q=sign(K) * 2^round(log2(abs(K)))`, with exponent clamp `[-16, 8]`:

- K min/max: −0.0122711 / 0.0246094
- `|K|` mean/median: 0.00642089 / 0.00604767
- sign: 4 negative, 668 non-negative
- `log2|K|` range: −13.0030 to −5.34465
- initial exponent histogram: `-13:1, -9:6, -8:262, -7:380, -6:22, -5:1`
- mean absolute quantization error: 0.00114414
- mean relative quantization error: 0.178090
- maximum relative quantization error: 0.412563

Training result:

- trainable parameters: 270,186
- initial converted accuracy: 73.34%
- best/final/reload: 85.50% / 85.50% / 85.50%
- best epoch: 100
- delta vs Official: −1.78 pp
- delta vs old H2 84.60%: +0.90 pp
- final exponent histogram: `-8:1, -6:1, -5:359, -4:310, -3:1`

Conclusion: the float fold is algebraically correct and the power-of-two fused branch is effective after fine-tuning. It removes the general multiplier from the binary-branch alpha+BN datapath, but it does not remove all multipliers from the network.

## F. Parameter and hardware summary

Static per-image counts for this ResNet20 topology:

- first FP convolution: 442,368 multi-bit MACs
- binary-convolution terms: 40,108,032 XNOR/popcount terms
- final FP 64→10 linear: 640 multi-bit MACs
- binary-branch alpha scale locations in Official: 672 channels, 172,032 spatial elements/image
- PReLU/QRPReLU channels: 336, 86,016 activation elements/image
- global average pool: 8×8→64 values; `/64` can be implemented as `>>6`

| Model | Trainable params | Binary conv | General multiplier candidates | DSP candidate | Shift/add change | Residual |
|---|---:|---|---|---|---|---|
| Official ReCU | 270,858 | XNOR + popcount | first conv, alpha, PReLU, BN if unfused, final FC | Yes | no new shift-only replacement | multi-bit add + buffer, no DSP |
| R1 | 270,186 | XNOR + popcount | first conv, PReLU, remaining BN, final FC | Yes | alpha removed | multi-bit add + buffer, no DSP |
| R2 | 271,530 | XNOR + popcount | first conv, float alpha, remaining BN, final FC | Yes | QRPReLU: add → shift → add; no general multiplier in activation | multi-bit add + buffer, no DSP |
| R3 | 270,186 | XNOR + popcount | first conv, PReLU, stem/head BN, final FC | Yes | fused branch: optional negate → shift → add B; no general multiplier in binary branch | multi-bit add + buffer, no DSP |

The first convolution remains the original FP/multi-bit 3×3 layer; its weights are FP32, not 8-bit. The final FC also remains FP/multi-bit. Stem BN and head BN were intentionally not removed in R3. Only the binary-branch alpha+BN is fused there.

## G. Final decisions and next-stage boundary

- R1 alpha: keep as a viable retrained ablation; do not claim zero-shot removability.
- R2 warm-start QRPReLU: keep; +2.44 pp over old H1 is a clear improvement.
- R3 fused pow2 affine: keep as the deployment-correct alpha+BN hardware candidate; it reaches 85.50% and improves over old H2.
- No R1+R2, R2+R3, W1A8 stem, binary FC, residual quantization, RTL, ASIC synthesis, T18, or T90 was started.

Final status: R1, R2, and R3 are complete. The requested next phase is intentionally not started.
