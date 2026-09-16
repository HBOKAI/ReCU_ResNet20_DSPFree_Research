# ReCU ResNet-20 + DSP-Free Research Pack

For the current safe H2A-v2 baseline, experiment reports, branch roles, and GitHub archive exclusions, see [Research archive and discussion guide](GITHUB_RESEARCH_INDEX.md) on the archive branch.

This package has TWO strictly separated purposes.

## A. Official ReCU reproduction — do this first

Default config:

```bash
python train_recu.py --config configs/recu_official.json
```

The model reproduces the official CIFAR ReCU topology:

- FP Conv3x3 stem, 3->16
- BN + Hardtanh
- 3/3/3 ReCU residual blocks, channels 16/32/64
- each block:
  - ReCU BConv
  - BN
  - shortcut add
  - save x1
  - Hardtanh
  - ReCU BConv
  - BN
  - add x1
  - channel-wise PReLU
- GAP
- BN1d(64)
- FP Linear 64->10

Official training recipe mirrored from the public ReCU repository:

- CIFAR-10 train set for training
- CIFAR-10 test set for epoch evaluation, matching official code
- 600 epochs
- batch 256
- test batch 128
- SGD
- LR 0.1
- momentum 0.9
- weight decay 5e-4
- 5-epoch warm-up behavior
- cosine scheduler
- tau_min = 0.85
- tau_max = 0.99
- seed = 123

Official public target is ~87.5% top-1 for ResNet20-1W1A, with ~88.0% listed for finetune.

Important:
The official reproduction uses the CIFAR-10 test set during training evaluation because that is what the public ReCU code does. For a final thesis study, create a separate validation protocol after reproduction is confirmed.

## B. Hardware-aware experiments — only after reproduction works

### QRPReLU

```bash
python train_recu.py --config configs/recu_qrprelu.json
```

This replaces official PReLU with A&B-BNN-style Quantized RPReLU:

negative branch:

`2^round(a) * (x + xi1) + xi2`

Deployment:
- `2^k` -> bit shift
- xi1/xi2 -> additions
- no general multiplier for RPReLU

### QRPReLU + power-of-two alpha

```bash
python train_recu.py --config configs/recu_qrprelu_pow2alpha.json
```

This additionally quantizes ReCU alpha scaling to signed power-of-two magnitude.

This is experimental hardware-aware QAT, NOT official ReCU.

## Smoke / diagnostic

```bash
python tools/report_recu.py --config configs/recu_official.json
python -m unittest discover -s tests -v
python train_recu.py --config configs/recu_official.json --smoke
python train_recu.py --config configs/recu_official.json --diagnostic-epochs 20
```

Do not run the hardware-aware variants until the official reproduction curve is healthy.

## Sources used for implementation

ReCU official:
https://github.com/yuanchunyu/ReCU

A&B BNN official paper/code:
https://openaccess.thecvf.com/content/CVPR2024/html/Ma_AB_BNN_AddBit-Operation-Only_Hardware-Friendly_Binary_Neural_Network_CVPR_2024_paper.html
https://github.com/Ruichen0424/AB-BNN
