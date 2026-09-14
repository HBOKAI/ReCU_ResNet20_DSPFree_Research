# R5 — W1A8 Stem

R5 starts from the validated R4 best checkpoint (~86.11%).

Only the stem convolution precision changes:

- normalized input -> signed INT8, fixed power-of-two scale
- stem weight -> 1-bit +/-1
- stem products -> +q / -q
- no general multiplier in the stem convolution

Default input quantization:
- signed INT8
- per-tensor
- scale = 2^-5

The rest of R4 is unchanged:
- multiplier-free binary residual backbone
- QRPReLU
- fused signed-power-of-two affine
- double skip

R5 still does NOT make the whole network multiplier-free because:
- stem BN remains
- head BN remains
- final FC remains multi-bit
- raw-RGB preprocessing is outside the current accelerator definition
