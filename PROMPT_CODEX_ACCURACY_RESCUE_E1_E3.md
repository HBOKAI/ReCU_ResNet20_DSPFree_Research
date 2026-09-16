# Codex Prompt — ReCU Accuracy Rescue E1–E3

Work only on branch `exp/accuracy-rescue-v1`, created from `archive/workspace-20260916`.

## Non-negotiable baseline protection

- Do not modify, rewrite, delete, or overwrite artifacts on `archive/workspace-20260916`.
- Frozen safe baseline: R8B-H2A-v2, archived official TEST 85.03%.
- R8B software baseline: 85.40%.
- H0 INT6: 85.13%.
- H1MP: 85.03%.
- H2A-v2 adds 0.00 pp loss vs H1MP and has H2 overflow 0.
- Existing R8B-LONG 600-epoch control is NOT a rescue baseline: official TEST 84.91%, i.e. -0.49 pp vs original R8B. Do not attribute gains from extra epochs alone to the new method.
- Do not run or implement SPoT/two-term coefficients yet.
- Do not change Thermometer R=8, model topology, dataset protocol, class count, or binary-conv semantics.
- Do not use H2B as the mainline.
- Official CIFAR-10 TEST must never be used for checkpoint selection or hyperparameter search.

Run the experiments independently in the following order. Do not combine them until each standalone effect is measured.

---

# E1 — H2A-v2 Exact-Deploy QAT

## Goal

Recover as much as possible of the R8B 85.40% -> H2A-v2 85.03% gap without changing deployed hardware.

## Source

Use the formal selected R8B checkpoint and the frozen H0/H1MP/H2A-v2 numeric contract already archived in the repository. Reuse existing loading/verification code from:

- `recu_hw/h1_model_adapter.py`
- `recu_hw/h1s_mp_quant.py`
- `recu_hw/h2_workspace.py`
- `recu_hw/h2_integer_accum.py`
- `tools/run_h2_finite_width.py`
- `reference_inference/`

Do not invent a new quantization plan.

## Required deployment contract during forward

Forward must behave as the existing safe deployment path:

1. Thermometer R=8, 96 bipolar input channels.
2. W1A1 stem and W1A1 ReCU binary convolutions.
3. Signed-pow2 affine K values.
4. H0-selected INT6 affine B and FC bias representation.
5. Frozen H1MP 58-node bits/shift/policy map exactly as archived.
6. Exact residual scale alignment with the repository's defined rounding rules.
7. Existing QRPReLU behavior for E1; do NOT introduce signed-QRP here.
8. H2A-v2 safe internal widths exactly as archived.
9. GAP uses deferred scale only: `q_gap=q_sum`, `s_gap=s_input+6`; never reintroduce rounded `>>6`.
10. FC remains signed-pow2 and uses the safe H2A-v2 accumulator/logit widths.

No deployment precision, node bit width, scale, exponent representation, or accumulator width may be relaxed to gain accuracy.

## Training method

Create a differentiable fake-quant / STE training adapter whose forward numerically matches the deploy contract at every quantized boundary, while gradients flow to the latent trainable parameters.

Important: `H2TraceAdapter` currently contains integer/reference operations that are not differentiable. Do NOT simply train through integer tensors. Build a QAT-specific adapter that applies the same quantization values using STE, e.g. conceptually `x_q = x + (dequant(quant(x)) - x).detach()`, while preserving the exact frozen bits/shifts/ranges and rounding/clipping semantics in the forward value.

Keep the frozen 672 compatibility alpha parameters frozen.

Recommended controlled schedule:

- optimizer: SGD
- momentum: 0.9
- weight decay: 0
- initial LR: 1e-5 and 5e-5 as TRAIN-derived candidates only
- cosine schedule
- first diagnostic: 20 epochs on train/validation
- formal: up to 100 epochs using the selected TRAIN-derived recipe
- seed: 123 unless the existing formal scripts require otherwise

Do not select by official TEST.

## E1 controls

Run at least:

- E1-C0: original frozen H2A-v2 evaluation (must reproduce current local result; archived official is 85.03%, independent reference currently reports 85.01%, and this 2-image discrepancy is unresolved).
- E1-C1: same R8B source with an equal-length low-LR fine-tune that does NOT apply deploy QAT, if needed to isolate the QAT effect. Existing `R8B_LONG` already proves simple 600-epoch extension is not automatically beneficial; cite it rather than rerunning 600 epochs.
- E1-QAT: exact-deploy QAT.

## E1 acceptance

Report both archived and locally reproduced baselines accurately. Success tiers:

- strong: >=85.35% official TEST with zero hardware change
- useful: >=85.20%
- neutral: 85.00–85.19%
- reject: <85.00% or any hardware-contract relaxation

The success label is only an experiment decision, not permission to alter H2A-v2 widths.

## E1 outputs

Create isolated files such as:

- `recu_hw/h2_qat.py`
- `tools/train_h2a_v2_qat.py`
- `configs/h2a_v2_qat.json`
- `tests/test_h2_qat.py`
- `reports/accuracy_rescue/E1_H2A_V2_QAT_REPORT.md`
- `reports/accuracy_rescue/E1_H2A_V2_QAT_RESULTS.json`

Record zero-epoch deploy accuracy, TRAIN validation curve, best epoch, final epoch, official TEST after freeze, reload accuracy, NaN/Inf, all hardware invariants, H2 overflow, residual alignment, GAP exactness, and whether any frozen bits/shifts changed. They must not.

---

# E2 — Progressive W1A1 Stem + KD

## Goal

Recover the R5T-Long T1 85.89% -> T2 85.14% binary-stem gap (-0.75 pp) with zero deployment hardware overhead.

## Fixed architecture

Teacher and student both use:

- CIFAR-10 raw RGB -> Thermometer R=8 -> 96 bipolar channels
- same R4-era backbone topology

Teacher source:

- formal R5T-Long T1 FP-stem best checkpoint, 85.89%

Student deployment target:

- strict W1A1 stem, effective weights exactly `{-1,+1}`
- same downstream architecture as the original R5T-Long T2 experiment

Do NOT change R, channel count, kernel size, stage topology, or add inference modules.

## E2-A progressive binarization

During training only, allow a progressive stem transition. A preferred implementation is an STE-compatible blend whose deployed endpoint is exactly sign weight:

`W_eff = (1-lambda) * W_latent + lambda * sign_ste(W_latent)`

Use a monotonic schedule from lambda=0 to lambda=1 over the progressive phase, then a hard-binary phase with lambda=1 throughout. Do not leave any floating blend in the exported model.

Start with a controlled recipe such as:

- progressive phase: 100 epochs, lambda 0 -> 1
- hard-binary phase: 100 epochs, lambda = 1
- SGD, momentum 0.9
- low LR derived from the successful R5T-Long schedule; tune only on TRAIN-derived validation

## E2-B KD

After E2-A is implemented and measured, add KD without changing inference.

Teacher is frozen and eval-only. Student is the progressive/hard W1A1 stem model.

Required KD baseline:

`L = CE(student_logits, y) + alpha * T^2 * KL(log_softmax(student/T), softmax(teacher/T))`

Use TRAIN-derived validation to choose a small candidate grid, e.g. T in {2,4}, alpha in {0.25,0.5}. Keep the grid small.

Optional feature KD is allowed only if it is clean and training-only. If used, prefer matching stem output and/or stage1 output. It must not add any inference module or parameter.

## E2 controls

Report independently:

- T1-Long teacher reload: 85.89% expected.
- original T2-Long reference: 85.14%.
- E2-A progressive only.
- E2-B progressive + logits KD.
- optional progressive + logits + feature KD only if the previous result justifies it.

## E2 acceptance

- strong: >=85.70% official TEST while exported stem is strict W1A1
- useful: >=85.45%
- partial: >85.14% but <85.45%
- reject: <=85.14% or any inference overhead

## E2 outputs

Create isolated modules/scripts/configs/tests and reports under `reports/accuracy_rescue/`. Verify after reload:

- stem effective unique values exactly `[-1,+1]`
- thermometer R=8, L=32, 96 channels
- no teacher object in exported student checkpoint
- no new inference parameters/modules
- unchanged BMAC count
- no NaN/Inf

---

# E3 — Signed-QRPReLU

## Goal

Address the explicit representational mismatch in R2: official ReCU PReLU has 336 slopes, including 89 non-positive slopes, while current QRPReLU only represents positive `2^k` slopes. Reduce the Official 87.28% -> R2 86.72% gap.

## Required new activation

Implement a separate class; do not silently change the existing `QuantizedRPReLU` semantics used by archived checkpoints.

Signed negative-branch slope:

`y = x`, if `x >= 0`

`y = s_i * 2^round(a_i) * (x + xi1_i) + xi2_i`, if `x < 0`

where `s_i in {-1,+1}` at deployment.

Training may keep a latent sign parameter with STE, but the effective deployed sign must be exactly bipolar. Do not use an arbitrary learned floating multiplier.

Hardware interpretation must remain:

- add xi1
- power-of-two shift
- optional negate controlled by one sign bit
- add xi2

No general multiplier.

## Initialization from official PReLU

For each PReLU slope `p_i`:

- effective sign initialized from `sign(p_i)`; define an explicit deterministic policy for exact zero
- exponent initialized from `round(log2(max(abs(p_i), floor)))`
- xi1 = 0
- xi2 = 0

Document the number of positive, negative, and zero source slopes and verify that previously negative PReLU channels are no longer forced to a positive slope.

## Source and isolation

Use the same official ReCU source/protocol as R2 so this is a clean replacement for the R2 activation ablation. Do not simultaneously change alpha, stem, BN, FC, thermometer, or downstream integerization.

Train with the same warm-start protocol as the original R2 as the primary control; additional low-LR extension may be TRAIN-validation-selected but must be reported separately.

## E3 controls

- Official ReCU: 87.28%.
- Existing R2 positive-only QRP: 86.72%.
- New Signed-QRP under matched training protocol.

## E3 acceptance

- strong: >=87.05%
- useful: >=86.90%
- partial: >86.72%
- reject: <=86.72% or use of a general multiplier

## E3 outputs

Suggested isolated files:

- `recu_hw/signed_qrprelu.py`
- a separate model/build path for signed-QRP warm-start
- `train_recu_signed_qrp.py`
- `configs/recu_signed_qrp.json`
- `tests/test_signed_qrprelu.py`
- `reports/accuracy_rescue/E3_SIGNED_QRP_REPORT.md`
- `reports/accuracy_rescue/E3_SIGNED_QRP_RESULTS.json`

Tests must verify exact effective sign values, exact integer shift exponents, NCHW behavior, positive branch identity, negative branch formula, checkpoint reload, and no NaN/Inf.

---

# Final comparison and stop condition

After E1, E2, and E3 are complete, create:

- `reports/accuracy_rescue/ACCURACY_RESCUE_COMPARISON.md`
- `reports/accuracy_rescue/ACCURACY_RESCUE_RESULTS.json`

Required final table:

| Experiment | Source | Baseline TEST | New TEST | Delta | Extra inference hardware | DSP/general multiplier | Decision |
|---|---|---:|---:|---:|---|---|---|
| E1 H2A-v2 exact-deploy QAT | H2A-v2/R8B | 85.03 archived / local reproduce separately | | | none | none | |
| E2 progressive W1 stem | T1/T2-Long | 85.14 student baseline | | | none | none | |
| E2 progressive + KD | same | 85.14 | | | none | none | |
| E3 Signed-QRP | Official/R2 | 86.72 R2 baseline | | | sign bit + conditional negate | none | |

Also answer:

1. How much of the 85.40 -> 85.03 integerization loss did E1 recover with zero hardware change?
2. How much of the 85.89 -> 85.14 binary-stem loss did E2 recover with zero hardware change?
3. How much of the 87.28 -> 86.72 QRP gap did E3 recover, and what is the exact added sign-storage/datapath cost?
4. Which results are statistically/experimentally clean enough to propagate into a future combined mainline?

STOP after producing these results. Do not implement or start SPoT, two-term coefficients, H2B, R change, FracConv, or a combined E1+E2+E3 model. Wait for the user to decide whether SPoT or a combined mainline is warranted.