# R5 — R4 + W1A8 Stem

Prerequisite:
Use R4 `best.pt`:
- R4 best/reload = 86.11%
- best epoch = 98

R5 changes only the stem convolution precision.

## Stem definition

Input activation:
- existing normalized CIFAR-10 tensor
- fake-quantized to signed INT8
- fixed power-of-two scale: 2^-5 by default

Stem weight:
- floating latent weight for training
- forward is strictly +/-1

Hardware product:
- +1 * q -> +q
- -1 * q -> -q

So stem Conv3x3 itself uses:
- select/sign negate
- add/accumulate
- no general multiplier / DSP

The input power-of-two scale is a binary-point/shift and can later be folded with
stem BN. R5 does NOT yet eliminate stem BN.

## 1. Verify conversion / INT8 range

```bash
python tools/verify_r5_conversion.py \
  --source-r4-checkpoint "<R4_BEST>" \
  --input-scale-exp -5
```

Required:
- binary stem weights only +/-1
- INT8 saturation fraction <= 1e-4

## 2. Unit tests

```bash
python -m unittest discover -s tests -v
```

## 3. Smoke

```bash
python train_recu_r5.py \
  --config configs/recu_r5_w1a8_stem.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --smoke
```

## 4. 20-epoch diagnostic

```bash
python train_recu_r5.py \
  --config configs/recu_r5_w1a8_stem.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --diagnostic-epochs 20
```

## 5. Formal 100-epoch fine-tune

```bash
python train_recu_r5.py \
  --config configs/recu_r5_w1a8_stem.json \
  --source-r4-checkpoint "<R4_BEST>"
```

R5 target:
- preserve R4 multiplier-free binary backbone
- eliminate general multipliers from the stem convolution
- do NOT yet modify final FC or remaining BN
