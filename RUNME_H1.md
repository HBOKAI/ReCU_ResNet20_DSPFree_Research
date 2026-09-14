# H1 Multi-bit Activation / Residual Integer Sweep

Source:
- R8B best = 85.40%
- H0 INT6 bias fixed
- expected H0 baseline ≈ 85.13%

Important:
- Calibration uses CIFAR-10 TRAIN data only.
- CIFAR-10 TEST is used only for final accuracy evaluation.
- No fine-tuning.
- Binary-convolution accumulators remain exact.
- GAP accumulator remains full precision.
- Residual branches must be scale-aligned by integer shifts before add.

Codex must first implement:
`recu_hw/h1_model_adapter.py`
against the actual R4/R8B block implementation, following
`recu_hw/h1_adapter_contract.py`.

Then run:

```bash
python -m unittest discover -s tests -v
```

```bash
python tools/run_h1_integer_sweep.py \
  --source-r8b-checkpoint "<R8B_BEST>" \
  --h0-results-json "<H0_INTEGER_BIAS_SWEEP_RESULTS.json>"
```
