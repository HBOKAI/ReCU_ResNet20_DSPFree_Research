# R5T-Long Final Report

Date: 2026-09-12

## A. Configuration

This round kept the R5T architecture unchanged and retrained both stages from the required sources.

- Dataset: CIFAR-10 raw RGB
- Pipeline: augmentation -> `ToTensor` in `[0,1]` -> Thermometer -> bipolar `{-1,+1}`
- No `Normalize` before Thermometer
- Thermometer resolution: `R=8`
- Thermometer length per RGB channel: `L=32`
- Thermometer input channels: `3 x 32 = 96`
- Stem: `3x3`, `96 -> 16`
- Backbone: unchanged R4 backbone
- ReCU tau: `0.99`
- Batch size: `128`
- Optimizer: SGD, momentum `0.9`
- Initial LR: `1e-3`
- Scheduler: linear decay to zero over the full `200` epochs
- T1 weight decay: `1e-5`
- T2 weight decay: `0`
- T1 epochs: `200`, floating latent stem weight, A1 activation
- T2 epochs: `200`, binary Sign forward stem weight, A1 activation
- T2 initialization: `T1-Long best.pt`
- No KD, scaling, progressive lambda, A8, architecture change, or optimizer-family change

The long-run configuration is in `configs/recu_r5t_thermometer_r8_long.json`.
On Windows, the loader used the existing safety fallback of `num_workers=0` to avoid worker permission failures; this did not change the model, input representation, augmentation, or optimization settings.

## B. Reference Results

| Reference | Accuracy | Best epoch |
|---|---:|---:|
| Official ReCU | 87.28% | - |
| R4 | 86.11% | 98 |
| R5 naive W1A8 | 83.77% | - |
| R5AB | 83.96% | - |
| Old R5T T1, FP stem | 85.29% | 99 |
| Old R5T T2, W1A1 stem | 85.00% | 97 |

Old R5T T1 and T2 are the 100-epoch references used for this comparison.

## C. T1-Long

Source: R4 best checkpoint, 86.11% at epoch 98.

Run directory: `experiments/recu_r5t_long/recu_r5t_thermometer_r8_long_20260912_155005`

- Zero epoch projected FP-stem accuracy: **84.66%**
- Best accuracy: **85.89%**
- Best epoch: **200**
- Final epoch accuracy: **85.89%**
- Reload accuracy: **85.89%**
- Trainable parameters: `284,250`
- Stem latent sign statistics at best checkpoint: `+1=6548`, `-1=7276`, `zero_latent=0`

Milestones:

| Epoch | LR | Train accuracy | Test accuracy |
|---:|---:|---:|---:|
| 100 | 0.000495 | 85.34% | 81.81% |
| 125 | 0.000370 | 85.46% | 82.80% |
| 150 | 0.000245 | 86.238% | 81.39% |
| 175 | 0.000120 | 86.648% | 83.75% |
| 200 | 0 | 89.042% | **85.89%** |

Compared with old T1:

- `85.89 - 85.29 = +0.60 pp`
- This satisfies the requested `>=85.6%` strong training-length-benefit criterion.
- The best point is at the end of the 200-epoch schedule, so the old 100-epoch endpoint was too early for this T1 schedule.

## D. T2-Long

Source: `T1-Long best.pt` from epoch 200. The old T1, old T2, and R4 checkpoints were not used as the direct T2-Long source.

Run directory: `experiments/recu_r5t_long/recu_r5t_thermometer_r8_long_20260912_171239`

### T2 zero epoch

- T1-Long best FP-stem accuracy: `85.89%`
- After switching only the stem forward path to binary Sign: `19.87%`
- Binary shock: `19.87 - 85.89 = -66.02 pp`

The T2 stem stayed binary during the complete training stage. The optimizer retained the floating latent stem weights and used the existing STE backward path.

### T2 result

- Best accuracy: **85.14%**
- Best epoch: **200**
- Final epoch accuracy: **85.14%**
- Reload accuracy: **85.14%**
- Trainable parameters: `284,250`
- Final sign statistics: `+1=6839`, `-1=6985`, `zero_latent=0`

Milestones:

| Epoch | LR | Train accuracy | Test accuracy |
|---:|---:|---:|---:|
| 50 | 0.000745 | 84.188% | 80.53% |
| 100 | 0.000495 | 84.990% | 82.94% |
| 125 | 0.000370 | 85.204% | 82.12% |
| 150 | 0.000245 | 85.838% | 82.91% |
| 175 | 0.000120 | 86.080% | 83.10% |
| 200 | 0 | 88.354% | **85.14%** |

Compared with old T2:

- `85.14 - 85.00 = +0.14 pp`
- This is a limited gain in the requested `85.0–85.2%` range.
- The best point is again at the end of the schedule, but the final result does not reach `85.5%`.

## E. Curve Interpretation

1. **Was 100 epochs too short?**

   - T1: yes. The 200-epoch schedule improved the result from 85.29% to 85.89% (+0.60 pp), a clear training-length benefit.
   - T2: the longer schedule produced only a small +0.14 pp improvement, so training length alone is not a strong solution for the binary-stem gap.

2. **Did the best epoch remain near the schedule end?**

   Yes. Both T1-Long and T2-Long reached their best accuracy at epoch 200.

3. **How much did the longer schedule add?**

   - T1: `+0.60 pp` over old T1
   - T2: `+0.14 pp` over old T2

4. **Did R5T-Long reach `>=85.5%`?**

   No. The final T2-Long best/reload accuracy is **85.14%**, which is `0.36 pp` below 85.5%.

## F. Final Comparisons

Using the required final metric, T2-Long best reload accuracy = **85.14%**.

| Comparison | Calculation | Difference |
|---|---:|---:|
| vs R4 | `85.14 - 86.11` | **-0.97 pp** |
| vs old R5T | `85.14 - 85.00` | **+0.14 pp** |
| vs R5 | `85.14 - 83.77` | **+1.37 pp** |
| vs R5AB | `85.14 - 83.96` | **+1.18 pp** |

The T1-Long FP-stem result is 85.89%, which is 0.22 pp below R4 and 0.60 pp above old T1. The final binary-stem conversion and 200-epoch T2 training retain 85.14%, leaving a 0.75 pp gap from T1-Long.

## G. Hardware

Inference architecture is unchanged from R5T.

- Stem BMAC: `14,155,776` BMAC/image
- R4 binary backbone BMAC: `40,108,032` BMAC/image
- Total: `54,263,808` BMAC/image
- Final stem: A1, W1, XNOR + popcount
- Stem general multiplier: `0`
- Stem convolution DSP: `0`

Longer training adds:

- no inference cost
- no extra BMAC
- no extra inference parameters
- no architecture change

The final T2 stem effective weight is strictly `{-1,+1}`.

## H. Verification

- Thermometer unit checks: `R=8`, `L=32`, `96` input channels, bipolar output only
- R4-to-R5T backbone state compatibility: verified
- R4 compatibility alpha tensors: all frozen
- T1 checkpoint reload: passed, 85.89%
- T2 checkpoint reload: passed, 85.14%
- T2 effective stem unique values after reload: `[-1.0, 1.0]`
- T2 sign counts after reload: `+1=6839`, `-1=6985`, `zero_latent=0`
- T1/T2 parameters and forward outputs: finite
- Test suite: **37 tests, OK**

## Conclusion

R5T-Long answers the training-length question. Extending the FP-stem T1 stage from 100 to 200 epochs is clearly useful and reaches 85.89%. Extending the binary-stem T2 stage gives only a limited additional gain to 85.14%, below the 85.5% target. Both best checkpoints occur at epoch 200, but the binary stem remains the limiting factor for this fixed architecture.

Per the requested stop condition, no 300+300 training, resolution change, KD, fractional activation, FracConv, BN elimination, FC quantization, RTL, FPGA synthesis, or later experiment was started.
