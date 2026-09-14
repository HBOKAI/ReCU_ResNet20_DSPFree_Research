# R5T — FracBNN-style Thermometer Binary Stem

Source:
- R4 best.pt
- accuracy = 86.11%
- best epoch = 98

Main configuration:
- raw CIFAR-10 RGB, no Normalize before thermometer encoding
- R = 8
- L = ceil(255/8) = 32 bins per RGB channel
- 3 x 32 = 96 bipolar binary input channels
- stem = 3x3, 96 -> 16
- T1: FP stem weights + binary thermometer activations
- T2: binary stem weights + binary thermometer activations

## 1. Verify

```bash
python tools/verify_r5t.py \
  --source-r4-checkpoint "<R4_BEST>" \
  --resolution 8
```

## 2. Unit tests

```bash
python -m unittest discover -s tests -v
```

## 3. Smoke both stages

```bash
python train_recu_r5t.py \
  --config configs/recu_r5t_thermometer_r8.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --stage both \
  --smoke
```

## 4. Formal T1 + T2

```bash
python train_recu_r5t.py \
  --config configs/recu_r5t_thermometer_r8.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --stage both
```

Important:
- T2 must warm-start from the best T1 state, not directly from R4.
- Final R5T accuracy is the T2 binary-stem accuracy.
- Do not use normalized CIFAR inputs for thermometer encoding.
- Do not introduce A8 into this experiment.
