# ReCU Research Extension — append to existing AGENTS.md

## Current primary research line

Official ReCU ResNet20-1W1A reproduction first, followed by hardware-aware multiplier elimination.

Do NOT alter official ReCU topology during reproduction.

Official reproduction must preserve:
- CIFAR-10
- ResNet20, channels 16/32/64
- double-skip block
- Hardtanh
- per-block channel-wise PReLU
- per-binary-conv alpha
- FP first convolution
- BN1d + FP final FC
- official tau schedule
- official training recipe

Official reproduction target:
- public ReCU result is about 87.5% top-1 for CIFAR-10 ResNet20-1W1A
- reproduction below 85% is a diagnostic failure, not a final research result

## Hardware-aware ablation order

Only after reproduction succeeds:

H1:
PReLU -> A&B-BNN-style Quantized RPReLU

H2:
ReCU alpha -> signed power-of-two quantization

Later, not yet:
H3 W1A8 first conv
H4 binary-weight integer-GAP classifier
H5 residual precision reduction
H6 ASIC XOR/XNOR/popcount PPA

Change one hardware variable at a time.

## Quantized RPReLU

For channel i:

positive:
y

negative:
2^round(a_i) * (y + xi1_i) + xi2_i

Deployment interpretation:
- 2^k multiplication is a bit shift
- xi1 / xi2 are additions
- target is zero general-purpose multiplier for this activation unit

Do not claim DSP-free merely because PyTorch uses a fake quantizer.
DSP-free is an inference hardware property that must be verified in RTL/synthesis later.
