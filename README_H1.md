# H1 Integer Activation / Residual Sweep

Purpose:
Starting from R8B-H0, keep the H0 INT6 affine/FC bias representation fixed and
measure the accuracy impact of pure-integer multi-bit activation/residual state
widths.

Sweep:
INT8 -> INT7 -> INT6 -> INT5 -> INT4
and optionally INT3 if INT4 loses <= 0.50 pp.

Important:
- Calibration uses CIFAR-10 train data only.
- The test set is never used to choose integer shifts.
- Binary-convolution accumulators remain mathematically exact in H1.
- GAP accumulator remains full precision in H1.
- Residual branches must be aligned to a common integer scale by shifts before
  integer addition.
- No fine-tuning is performed.

This pack intentionally provides generic integer arithmetic utilities plus an
adapter contract. Codex must implement `recu_hw/h1_model_adapter.py` against
the actual R4/R8B block code so inline residual-add nodes are captured
faithfully.
