# R8 Classifier Ablation

Controlled comparison from the same R7 best checkpoint:

- R8A: W1 FC
- R8B: signed-power-of-two FC

FC bias is preserved in both branches.
The purpose is to choose the final multiplier-free classifier for RTL.
