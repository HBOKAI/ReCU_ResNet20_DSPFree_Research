# Source audit anchors

Implementation was aligned against the public ReCU repository:

- `cifar/models_cifar/resnet.py`
  - `BasicBlock_1w1a`
  - FP stem
  - double skip
  - Hardtanh
  - channel-wise PReLU
  - GAP -> BN1d -> FP Linear
- `cifar/modules/binarized_modules.py`
  - per-filter weight centering / normalization
  - tau-dependent clipping
  - weight Sign STE
  - training-only activation variance normalization
  - triangular activation surrogate gradient
  - per-output-channel alpha
- `cifar/main.py`
  - 600 epochs
  - SGD 0.1 / momentum 0.9 / weight decay 5e-4
  - 5-epoch warmup behavior
  - cosine scheduler
  - exponential tau schedule
- `cifar/dataset/dataset.py`
  - RandomCrop(32,padding=4)
  - RandomHorizontalFlip
  - CIFAR-10 train/test protocol
  - ImageNet mean/std used by the official repository

A&B BNN hardware-aware extension:
- Quantized RPReLU:
  - positive: y
  - negative: 2^round(a_i) * (y + xi1_i) + xi2_i
- This enables slope multiplication to map to bit shift at inference.
