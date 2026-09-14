# R8 classifier ablation

Source: R7 best = 85.22% @ epoch 99.

Run both branches from exactly the same R7 best checkpoint.

## Tests
```bash
python -m unittest discover -s tests -v
```

## R8A smoke
```bash
python train_recu_r8.py \
  --variant r8a \
  --config configs/recu_r8a.json \
  --source-r7-checkpoint "<R7_BEST>" \
  --smoke
```

## R8B smoke
```bash
python train_recu_r8.py \
  --variant r8b \
  --config configs/recu_r8b.json \
  --source-r7-checkpoint "<R7_BEST>" \
  --smoke
```

## Formal R8A
```bash
python train_recu_r8.py \
  --variant r8a \
  --config configs/recu_r8a.json \
  --source-r7-checkpoint "<R7_BEST>"
```

## Formal R8B
```bash
python train_recu_r8.py \
  --variant r8b \
  --config configs/recu_r8b.json \
  --source-r7-checkpoint "<R7_BEST>"
```

Do not choose a winner until both are complete.
