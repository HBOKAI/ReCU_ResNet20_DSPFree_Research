# R8B-LONG — 600-Epoch Control Results

> Independent controlled training run. Existing R8B/H0/H1MP/H2A/H2B artifacts were not modified.

## 1. Formal source and baseline

- Formal R8B checkpoint: `experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt`
- Baseline reload accuracy (CUDA, official CIFAR-10 TEST): **85.40%**; expected 85.40%; status **PASS**.
- Formal R8B checkpoint SHA-256: `c07ec3b29b88bbf70d3300f0bb19c7113abdadd305eb3d73cc12c7f40c2b0ba8`; parameters: `284922`; seed: `123`; TEST images: `10000`.
- Environment: Python `3.10.21`, torch `2.5.1+cu121`, torchvision `0.20.1+cu121`, device `NVIDIA GeForce RTX 4070`.
- Initialization source: `experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/best.pt` (the same R7-to-R8B construction used by the formal R8B recipe).
- R7 source SHA-256: `32b720c0dfd876fdabbf7d1004efe725bcfa08d35ebc66269a6c32120077b73b`
- R8B architecture, binary convolutions, Thermometer R=8, signed-power-of-two K, QRPReLU, FC, and initialization are unchanged.
- The only schedule change is `CosineAnnealingLR(T_max=600)` instead of the original 100-epoch horizon.

## 2. Fixed recipe

- Seed: `123`; validation split seed: `123`.
- CIFAR-10 TRAIN split: 45,000 train / 5,000 validation, fixed and disjoint; official TEST is never used for epoch selection.
- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor. Validation/TEST: ToTensor only.
- Optimizer: SGD, lr=1e-4, momentum=0.9, weight_decay=0.0; tau=0.99; batch size=128.
- Checkpoints: epochs 100, 200, 300, 400, 500, 600, plus best validation.

## 3. Epoch accuracy table

| Epoch | Train loss | Train acc. | Validation acc. | Official test acc. | LR | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---|
| 100 | 0.42296 | 85.15% | 84.56% | 82.81% | 9.33012702e-05 | `f4869e9afad74629daf713ce68fcb81b65d953a5c5b09be041aecb0b5fb56c41` |
| 200 | 0.42375 | 85.21% | 83.68% | 82.08% | 7.5e-05 | `301312c1cf006556ffc39e2ed7b7d9b47c72849753a9bf4047795aafbe16c297` |
| 300 | 0.42052 | 85.30% | 82.64% | 80.99% | 5e-05 | `b051ff08ffbaf3aeffb60edc5d3cc82dcc0eea1373b2dc41b399d506e2a847e4` |
| 400 | 0.41143 | 85.55% | 84.10% | 81.79% | 2.5e-05 | `517c465ad9d0bf7de40e08d565eb648d69e84c5be726e77139b2a1c46e2d3baf` |
| 500 | 0.38007 | 86.68% | 84.62% | 83.98% | 6.69872981e-06 | `a62cf43d66e2da8e7e40809821a794db4aa26a10c168055be6bd06e4f672b684` |
| 600 | 0.30057 | 89.52% | 87.42% | 84.91% | 0 | `d56e53606cd28e8ab2cd75f9794905576ed31ed81e088d8f08f7bc4ab0bf8e43` |

Best validation: **87.42%** at epoch **600**.
Best-validation checkpoint official TEST: **84.91%**.
Best official milestone TEST: **84.91%** at epoch **600**; delta vs R8B 85.40% = **-0.49 pp**.

## 4. Saturation, plateau, and overfitting

- Last epoch validation rising: **True**.
- Validation slope over epochs 501–600: `+0.017033` percentage points/epoch.
- Plateau detected after epoch 300: **False**.
- Overfitting signal: **False**.
- Train–validation gap at epoch 600: **+2.10 pp**; validation–TEST gap: **+2.51 pp**; generalization-gap signal: **True**.
- Interpretation: Validation is still improving at the end; official TEST remains -0.49 pp versus the original R8B baseline.

## 5. Required integrity checks

- Model parameter count: `284922` (284250 trainable + 672 frozen alpha compatibility parameters).
- Exact signed-pow2 K / FC constraints: **PASS** at initialization and final checkpoint.
- NaN/Inf check: **PASS**.
- Thermometer input: UINT8-equivalent CIFAR-10 RGB converted by the formal R=8 ToTensor path; no preprocessing change.
- No H0 INT6, H1MP, H2A, H2B, RTL, synthesis, retraining, or fine-tuning operation was started after this run.

## 6. Decision answers

A. Original R8B baseline: **85.40%**; 600-epoch best official result: **84.91%**.
B. Last epoch still rising: **True**; plateau: **False**.
C. Classic validation overfitting evidence: **False**; a train/TEST generalization gap is **True**.
D. Worth starting H0/H1/H2 from this control: **NO (do not start a downstream H0/H1/H2 study from this control)**.
E. Best checkpoint: `experiments/recu_r8b_long/r8b_long_s123_20260916_005803/best_validation.pt` (SHA-256 `d56e53606cd28e8ab2cd75f9794905576ed31ed81e088d8f08f7bc4ab0bf8e43`).

## 7. Files

- Run directory: `experiments/recu_r8b_long/r8b_long_s123_20260916_005803`
- Recipe: `reports/r8b_long/R8B_LONG_RECIPE.md`
- Curve CSV: `reports/r8b_long/R8B_LONG_CURVE.csv`
- JSON: `reports/r8b_long/R8B_LONG_RESULTS.json`
