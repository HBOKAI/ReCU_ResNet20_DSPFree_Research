# R5AB run order

Source: R4 best.pt (86.11%, epoch 98)

Method:
- A8 unchanged
- alpha_c = mean(abs(W_c))
- W_eff = (1-lambda)W + lambda*alpha*sign(W)
- lambda ramps to 1 over 30 epochs
- epochs 30..100 use lambda=1

Critical:
Best checkpoint is selected only by forced lambda=1 deployment accuracy.

Commands:

```bash
python tools/verify_r5ab_conversion.py --source-r4-checkpoint "<R4_BEST>" --input-scale-exp -5
python -m unittest discover -s tests -v
python train_recu_r5ab.py --config configs/recu_r5ab_progressive_scaled_w1a8.json --source-r4-checkpoint "<R4_BEST>" --smoke
python train_recu_r5ab.py --config configs/recu_r5ab_progressive_scaled_w1a8.json --source-r4-checkpoint "<R4_BEST>" --diagnostic-epochs 20
python train_recu_r5ab.py --config configs/recu_r5ab_progressive_scaled_w1a8.json --source-r4-checkpoint "<R4_BEST>"
```

Target:
- >=85.5%: strong success
- 85.0–85.5%: acceptable
- <85.0%: inspect stem training before FC

Hardware:
- deployment lambda=1
- W1A8 conv core remains add/sub only
- alpha may be folded into stem BN
- fused K is not yet assumed pow2 in this experiment
