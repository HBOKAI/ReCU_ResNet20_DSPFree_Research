# H0 Integer Bias Sweep

Purpose:
Convert the remaining affine offsets and final FC bias of R8B into a
pure-integer storage representation, while leaving the rest of R8B unchanged.

Representation:
    q = round(B * 2^s)

RTL stores:
- signed integer q
- layer-level integer shift s

PyTorch uses:
    B_hat = q * 2^(-s)
only as a reference simulation of that exact integer+shift representation.

Sweep:
INT8 -> INT7 -> INT6 -> INT5 -> INT4
and optionally INT3.

This stage does not quantize activations, residuals, GAP or accumulators.
