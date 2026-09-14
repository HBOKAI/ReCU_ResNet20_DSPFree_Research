# R7 run order

Source: R6 best = 85.52% @ epoch 97.

1. Verify exact Head BN fold + hard projection:
```bash
python tools/verify_r7_folding.py \
  --source-r6-checkpoint "<R6_BEST>"
```

2. Tests:
```bash
python -m unittest discover -s tests -v
```

3. Smoke:
```bash
python train_recu_r7.py \
  --config configs/recu_r7.json \
  --source-r6-checkpoint "<R6_BEST>" \
  --smoke
```

4. Optional 20-epoch diagnostic:
```bash
python train_recu_r7.py \
  --config configs/recu_r7.json \
  --source-r6-checkpoint "<R6_BEST>" \
  --diagnostic-epochs 20
```

5. Formal 100 epochs:
```bash
python train_recu_r7.py \
  --config configs/recu_r7.json \
  --source-r6-checkpoint "<R6_BEST>"
```

R7 changes only Head BN scale:
- exact fold BN2 -> K*x+B
- K -> signed power-of-two
- B stays float/reference
- Final FC remains unchanged with bias
