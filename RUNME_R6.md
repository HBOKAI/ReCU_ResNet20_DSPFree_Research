# R6 run order

Source: R5T-Long T2 best = 85.14%.

1. Verify exact fold + hard projection:
```bash
python tools/verify_r6_folding.py \
  --source-r5t-long-checkpoint "<R5T_LONG_T2_BEST>"
```

2. Tests:
```bash
python -m unittest discover -s tests -v
```

3. Smoke:
```bash
python train_recu_r6.py \
  --config configs/recu_r6.json \
  --source-r5t-long-checkpoint "<R5T_LONG_T2_BEST>" \
  --smoke
```

4. Optional 20-epoch diagnostic:
```bash
python train_recu_r6.py \
  --config configs/recu_r6.json \
  --source-r5t-long-checkpoint "<R5T_LONG_T2_BEST>" \
  --diagnostic-epochs 20
```

5. Formal 100 epochs:
```bash
python train_recu_r6.py \
  --config configs/recu_r6.json \
  --source-r5t-long-checkpoint "<R5T_LONG_T2_BEST>"
```

Required invariants:
- exact BN fold max error < 1e-5
- stem W1A1 remains {-1,+1}
- stem Conv bias is None
- no separate stem bn1 in R6
- Khat is signed power-of-two
- B remains float/reference
