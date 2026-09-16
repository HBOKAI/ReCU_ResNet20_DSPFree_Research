# R8B-LONG Recipe

## Original R8B

- Epochs: 100
- Initial learning rate: 1e-4
- Scheduler: cosine decay, `CosineAnnealingLR(T_max=100)`
- Warmup: none (the formal configuration has no warmup)
- Optimizer: SGD, momentum 0.9, weight decay 0.0
- Batch size: 128
- Augmentation: `RandomCrop(32,padding=4)`, `RandomHorizontalFlip`, `ToTensor`
- Loss: cross-entropy

## R8B-LONG

- Epochs: 600
- Initial learning rate: 1e-4
- Scheduler: same cosine shape with `CosineAnnealingLR(T_max=600)`
- Warmup: none
- Optimizer, batch size, augmentation, loss, quantization, signed-Pow2 constraints, QRPReLU, and model initialization: unchanged
- Validation: fixed seed=123 CIFAR-10 TRAIN split, 45,000 train / 5,000 validation
- Official TEST: evaluated only at epochs 100/200/300/400/500/600 and on the best validation checkpoint

The only training change is the schedule horizon. The run starts from the exact R7-derived R8B initialization used by the formal R8B recipe; it does not continue the old epoch-100 scheduler state. No H0 INT6 bias, H1MP activation quantization, H2A/H2B arithmetic, KD, EMA, Mixup, CutMix, label smoothing, retraining, RTL, or synthesis was used.

Run directory: `experiments/recu_r8b_long/r8b_long_s123_20260916_005803`
Fixed split indices and all optimizer/scheduler/RNG states are saved in that run directory.
