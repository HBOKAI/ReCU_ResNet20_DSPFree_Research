# H0 Pure-Integer Bias Sweep

Source:
- R8B best / reload ≈ 85.40%

This is zero-shot evaluation, not model retraining.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Sweep

```bash
python tools/eval_integer_bias_sweep.py \
  --source-r8b-checkpoint "<R8B_BEST>"
```

The tool evaluates:
- INT8
- INT7
- INT6
- INT5
- INT4
- INT3 only if INT4 loses <= 0.50 pp

Each target is represented for hardware by:
- signed integer q
- one integer power-of-two shift metadata value s

No Q-format naming is used.
No fine-tuning is performed.
