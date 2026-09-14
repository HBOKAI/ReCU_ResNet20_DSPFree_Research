# R6
Controlled experiment on top of R5T-Long 85.14%.

Only change:
Stem BatchNorm is exact-folded to K*S+B, and K is quantized to signed power-of-two.
B remains float/reference.

No change to:
Thermometer R=8, W1A1 stem, R4 backbone, Head BN, Final FC.

Goal:
Measure isolated accuracy cost of removing the general multiplier from the Stem BN scale path.
