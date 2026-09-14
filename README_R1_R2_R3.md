# ReCU R1/R2/R3 hardware-ablation extension

This extension assumes a validated Official ReCU ResNet20-1W1A best checkpoint.

## R1 — Remove alpha
Tests whether ReCU's per-channel alpha is functionally dispensable after
the network has already converged.

## R2 — Warm-start QRPReLU
Projects every learned official PReLU slope into A&B-style power-of-two
QRPReLU, rather than retraining QRPReLU from random initialization.

## R3 — Fold alpha+BN, then quantize fused K
This is the deployment-correct experiment.

Original:
`BN(alpha * S)`

Folded:
`K*S + B`

Then:
`K -> sign(K)*2^round(log2|K|)`

Hardware:
- sign = optional negate
- `2^k` = shift
- `+B` = add
- no general multiplier in this branch

R3 first verifies float-fold equivalence before any power-of-two quantization.

All three default to 100-epoch warm-start fine-tuning from the official best:
- SGD
- LR 0.01
- momentum .9
- weight decay 5e-4
- cosine
- ReCU tau fixed at .99
