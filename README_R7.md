# R7
Controlled experiment on top of R6 best 85.52%.

Only change:
Head BatchNorm1d(64) is exact-folded to K*x+B and K is quantized to signed power-of-two.

Unchanged:
- Thermometer R=8
- W1A1 stem
- Stem signed-pow2 affine
- R4 binary backbone
- Final FC 64->10, including its bias

Goal:
Measure the isolated accuracy cost of removing the general multiplier from the Head BN scale path.
