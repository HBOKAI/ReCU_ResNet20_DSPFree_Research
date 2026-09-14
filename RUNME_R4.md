# R4 run order

R2 source must be the successful warm-start QRPReLU best checkpoint:
- Best / reload: 86.72%

## 1. Conversion verification

```bash
python tools/verify_r4_conversion.py \
  --source-r2-checkpoint "<R2_BEST>"
```

Required:
- 18 alpha+BN pairs
- max abs float-fold error <= 1e-4

## 2. Unit tests

```bash
python -m unittest discover -s tests -v
```

## 3. Smoke

```bash
python train_recu_r4.py \
  --config configs/recu_r4_qrprelu_fused_pow2_affine.json \
  --source-r2-checkpoint "<R2_BEST>" \
  --smoke
```

## 4. 20 epochs diagnostic

```bash
python train_recu_r4.py \
  --config configs/recu_r4_qrprelu_fused_pow2_affine.json \
  --source-r2-checkpoint "<R2_BEST>" \
  --diagnostic-epochs 20
```

## 5. Formal 100 epochs

```bash
python train_recu_r4.py \
  --config configs/recu_r4_qrprelu_fused_pow2_affine.json \
  --source-r2-checkpoint "<R2_BEST>"
```

R4 means:
- QRPReLU inherited from R2
- binary branch alpha+BN folded
- fused K constrained to signed power-of-two
- binary residual backbone has no general multiplier

R4 does NOT yet remove multipliers from:
- FP stem
- stem BN
- head BN
- final FP FC
