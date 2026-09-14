# R4 — QRPReLU + Fused Signed Power-of-Two Alpha/BN

Date: 2026-09-12  
Environment: `KenBnn_env`, PyTorch 2.5.1+cu121, NVIDIA GeForce RTX 4070  
Training: 100-epoch warm-start fine-tuning, SGD lr=0.005, momentum=0.9, weight decay=5e-4, cosine scheduler, tau fixed at 0.99.

## A. R2 source checkpoint

Source:

`C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r2_warmstart_qrprelu\recu_r2_warmstart_qrprelu_20260912_094409\best.pt`

- source best epoch: 97
- source best/reload accuracy: 86.72% / 86.72%
- source was the formal R2 Warm-start QRPReLU checkpoint
- Official, old H1, old H2, R1, and R3 were not used as the R4 source

The workspace is not a Git repository, so `git status`/commit provenance is unavailable. Existing Official/R1/R2/R3 experiment directories were preserved.

## B. R4 conversion audit

R4 preserves R2's W1A1 binary convolutions, double skip, Hardtanh, trained QRPReLU parameters (`a`, `xi1`, `xi2`), FP stem, stem BN, head BN, and FP final FC. The only new model change is:

`S -> alpha -> BN` → `S -> sign(K) * 2^round(log2(abs(K))) + B`

Float folding was checked over all 18 binary-convolution alpha+BN pairs:

- maximum absolute float-fold error: `1.19e-7`
- required threshold: `1e-4`
- result: PASS

Initial 672 fused K coefficients:

- K min/max: −0.0116633 / 0.0192148
- `|K|` mean/median: 0.00693450 / 0.00658926
- negative/positive: 4 / 668
- initial exponent histogram: `-9:5, -8:194, -7:433, -6:40`
- mean absolute quantization error: 0.00123168
- maximum absolute quantization error: 0.00456504
- mean relative quantization error: 0.179200
- maximum relative quantization error: 0.412754

## C. Zero-epoch and formal accuracy

| Model | Accuracy | Difference |
|---|---:|---:|
| R2 source | 86.72% | reference |
| R4 zero-epoch converted | 70.11% | −16.61 pp vs R2 |
| R4 formal best/reload | 86.11% / 86.11% | +16.00 pp after fine-tuning |

Formal R4:

- best accuracy: 86.11%
- best epoch: 98
- final accuracy: 85.76%
- reload accuracy: 86.11%
- Δ vs Official 87.28%: −1.17 pp
- Δ vs R2 86.72%: −0.61 pp
- Δ vs R3 85.50%: +0.61 pp
- training time: 1617.66 seconds, approximately 26.96 minutes
- trainable parameters: 270,858

Accuracy decision: R4 reaches the requested `>=86.0%` “very good” category.

Formal run directory:

`experiments/recu_r4/recu_r4_qrprelu_fused_pow2_affine_20260912_110737`

Best checkpoint:

`experiments/recu_r4/recu_r4_qrprelu_fused_pow2_affine_20260912_110737/best.pt`

## D. Final quantization and sanity checks

Final 672 fused exponent histogram:

`-7:11, -6:452, -5:209`

This is not exponent collapse: three distinct exponents remain populated.

Final K statistics:

- K min/max: −0.0305379 / 0.0406455
- `|K|` mean/median: 0.0200010 / 0.0194471
- negative/positive: 4 / 668
- mean absolute quantization error: 0.00369037
- maximum absolute quantization error: 0.00939546
- mean relative quantization error: 0.179258
- maximum relative quantization error: 0.414183

Sanity checks on the formal best checkpoint:

- history: 100/100 epochs, all loss/accuracy values finite
- QRPReLU `a`, `xi1`, and `xi2`: all finite
- fused `log2|K|` and bias: all finite
- 672 alpha compatibility parameters: all frozen, not trainable
- trainable parameter count: 270,858
- unit tests: 13/13 PASS
- smoke: CUDA forward/backward, save, and reload PASS
- diagnostic: best/reload 81.83% / 81.83%, no NaN/Inf, no exponent collapse

## E. Hardware conclusion

### 1. Is the binary residual backbone general-multiplier-free?

Yes, under the deployment interpretation requested here. The binary residual backbone uses:

- binary convolution: XNOR/XOR + popcount
- fused alpha+BN: signed shift, optional negate, and add B
- residual paths: multi-bit add
- Hardtanh: compare/clamp
- QRPReLU: add, shift, add

None of these requires a general multiplier. Therefore the correct claim is:

`Binary residual backbone is general-multiplier-free.`

This is not a claim that the entire network is DSP-free.

### 2. Modules that still need general multipliers / DSP candidates

- FP/multi-bit stem Conv3×3: yes
- stem BN: not eliminated in R4
- head BN: not eliminated in R4
- FP/multi-bit final FC 64→10: yes

The first convolution remains FP32, not 8-bit. The final FC also remains multi-bit.

### 3. Modules using shift/add/compare instead

- binary convolution: XNOR/XOR + popcount
- fused alpha+BN: shift + optional negate + add
- QRPReLU: add + shift + add
- residual: multi-bit add, with buffer/adder/bit-width cost but no multiplier
- Hardtanh: compare/clamp
- GAP: 8×8 average to 64 values; division by 64 can be `>>6`

### 4. Is W1A8 stem worth entering next?

Yes, but only as the next isolated experiment and not started in this run. R4 reaches 86.11%, is only −0.61 pp from R2, improves R3 by +0.61 pp, and meets the `>=86.0%` target. The next logical study is W1A8 stem while keeping R4's binary residual backbone unchanged.

## F. Stop boundary

This R4 run is complete. No W1A8 stem, binary FC, two-shift coefficient, residual quantization, RTL, FPGA synthesis, ASIC synthesis, T18, or T90 was started.
