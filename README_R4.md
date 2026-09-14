# R4 — Multiplier-Free Binary Backbone

Source checkpoint:
R2 Warm-start QRPReLU best (~86.72%).

R4 combines:
1. trained QRPReLU from R2
2. alpha+BN folding
3. signed power-of-two fused scale

Binary residual backbone target arithmetic:
- XNOR/XOR
- Popcount
- Shift
- Negate
- Add
- Compare

No general multiplier should remain inside the binary residual backbone.

Still outside scope:
- FP stem
- stem BN
- head BN
- FP final FC
