# Codex Prompt — Selective 2-term SPoT Sensitivity Study

Work only on branch `exp/selective-2term-spot`.

## Status and non-negotiable scope

This is an isolated sensitivity study. It is **NOT** the selected mainline.

Selected deployment baseline remains:

- R8B-H2A-v2 + exact-deploy QAT
- official CIFAR-10 TEST = **85.51%**
- this result must not be overwritten, relabeled, or replaced by SPoT automatically.

Do NOT modify/archive-overwrite the frozen historical branches or existing E1/E2/E3 reports.
Do NOT change Thermometer R, binary-conv topology, QRPReLU, stem, FC, H1MP, H2A-v2 widths, or RTL in this study.
Do NOT combine this study with E1 QAT yet.
Do NOT push unless explicitly requested.
Do NOT use official TEST to choose coverage, ranking, hyperparameters, or checkpoints.

The purpose is only to answer:

> Can a selective subset of the backbone fused-affine K coefficients be upgraded from one-term signed power-of-two to 2-term signed power-of-two and recover part of the R2 -> R4 accuracy loss with a favorable hardware trade-off?

Relevant historical references:

- R2 positive-only QRP: **86.72%** official TEST
- R4 one-term signed-pow2 affine: **86.11%** official TEST
- gap targeted by this study: **-0.61 pp**
- `recu_hw/fused_affine.py` currently implements one-term `K = sign * 2^round(log2|K|)`.
- R4 matched training recipe is `configs/recu_r4_qrprelu_fused_pow2_affine.json`: 100 epochs, SGD lr=0.005, momentum=0.9, weight_decay=5e-4, cosine, tau=0.99, seed=123.

---

# 1. Exact SPoT definition

For selected backbone affine channels only, replace one-term:

`K1 = s * 2^k`

with canonical 2-term signed power-of-two:

`K2 = s1 * 2^k1 + s2 * 2^k2`

where:

- `s1, s2 ∈ {-1,+1}`
- integer exponents use the same safe exponent bounds as the affine module unless the source code proves a stricter bound is required
- canonical non-degenerate 2-term form requires `k1 > k2`
- if the best representation is already exactly one-term, retain one-term and mark the second term inactive; do not add a redundant second shift
- no arbitrary floating deployment multiplier is allowed
- deployment interpretation is shift/negate + shift/negate + add, then +B

Search the discrete 2-term representation exactly or exhaustively over the bounded exponent/sign space. Do not approximate the search with a heuristic if exact search is cheap.

The existing one-term `FusedAffine2d` semantics must remain unchanged for archived models. Implement a separate SPoT-capable class/path.

---

# 2. Scope of coefficients

This study initially targets only the **backbone fused-affine K values created by the R2 -> R4 conversion**.

Do not include:

- stem affine
- head affine
- FC
- QRPReLU slope
- bias B

Expected target population is the historical backbone set of **672 K coefficients**. Verify this programmatically; if the actual count differs, stop and report rather than forcing 672.

Bias B remains exactly the corresponding R4 floating bias during this R4-level sensitivity study. Do not introduce H0 INT6 yet because this study is isolating the R2 -> R4 coefficient restriction.

---

# 3. S0 — analytic coefficient audit

Load the formal R2 best checkpoint and reconstruct the exact folded floating coefficients using the existing `fold_recu_alpha_bn` path.

For every target coefficient record:

- module/layer name
- channel index
- `K_float`
- nearest legal one-term `K_1term`
- absolute error `|K_float-K_1term|`
- relative error `|K_float-K_1term| / max(|K_float|, eps)`
- exact best legal 2-term `K_2term`
- two exponents and signs
- absolute/relative 2-term error
- error reduction from one-term to 2-term
- whether second term is actually needed

Verify that the one-term reconstruction reproduces the current R4 coefficient quantization semantics exactly.

Create:

- `reports/selective_spot/S0_COEFFICIENT_AUDIT.csv`
- `reports/selective_spot/S0_COEFFICIENT_AUDIT.json`
- `reports/selective_spot/S0_COEFFICIENT_AUDIT.md`

Report summary statistics for one-term vs 2-term approximation: mean/median/max absolute and relative error, exact-match count, and number of channels materially improved by 2-term.

---

# 4. S0B — TRAIN-only activation-aware sensitivity metric

Coefficient error alone is not sufficient because a large K error on an inactive/small-S channel may not matter.

Using a fixed TRAIN-derived calibration subset only (never official TEST), collect the pre-affine binary-convolution accumulator `S` statistics for each target channel.

Compute at minimum:

`distortion_1term_i = E[((K_float_i - K_1term_i) * S_i)^2]`

`distortion_2term_i = E[((K_float_i - K_2term_i) * S_i)^2]`

and

`recoverable_distortion_i = distortion_1term_i - distortion_2term_i`.

Use a fixed seed and save the exact calibration indices or split metadata.

Primary ranking for selective coverage = descending `recoverable_distortion_i`.

Also retain a secondary coefficient-only ranking by relative one-term error for analysis. Do not silently switch ranking based on official TEST.

Create:

- `reports/selective_spot/S0B_ACTIVATION_AWARE_RANKING.csv`
- `reports/selective_spot/S0B_ACTIVATION_AWARE_RANKING.json`

---

# 5. S1 — coverage sweep without official TEST selection

Evaluate the following 2-term coverage levels using the TRAIN-derived validation split:

- 0% = one-term R4 control
- 5%
- 10%
- 25%
- 50%
- 100% = full 2-term control

Convert percentage to deterministic channel count from the verified target population, document the exact count and selected channels, and use the fixed activation-aware ranking.

Important:

- unselected channels remain exact one-term pow2
- selected channels use exact 2-term SPoT
- same R2 source
- same floating B
- same architecture and data protocol
- no official TEST during coverage selection

First perform zero-shot conversion diagnostics on TRAIN-derived validation and verify:

- no NaN/Inf
- one-term channels remain exact signed powers-of-two
- two-term channels exactly equal their discrete SPoT representation
- no general multipliers in the mathematical deployment form

Save a coverage manifest for every candidate with exact module/channel selections and SPoT terms.

---

# 6. S2 — short matched diagnostic training

Run short **20-epoch** diagnostics for:

- 10%
- 25%
- 50%
- 100%

5% may be included if its zero-shot validation is unexpectedly strong; otherwise keep it analysis-only.

Use the original R4 optimizer family and source but shorten only the horizon for diagnosis:

- SGD
- lr=0.005 unless a reproducibility check of the original R4 code indicates the short diagnostic needs scheduler rescaling
- momentum=0.9
- weight_decay=5e-4
- tau=0.99
- seed=123
- cosine schedule over the diagnostic horizon

Training must use an STE-compatible discrete SPoT forward. At deployment-effective forward values, selected K must always be exactly the two signed powers-of-two sum; unselected K must always be exactly one-term.

Do not allow a latent floating K to leak into forward inference.

Select the best selective coverage using TRAIN-derived validation only. Record zero-epoch, best, final, reload validation and epoch.

---

# 7. S3 — formal 100-epoch study

Formal training must include exactly these controls/candidates:

1. **one-term R4 matched control** using the known R4 recipe or a verified reload if exact retraining is unnecessary for attribution
2. **best selective coverage** chosen from S2 by TRAIN-derived validation
3. **100% full 2-term control**

Use the historical R4 100-epoch recipe:

- epochs=100
- SGD lr=0.005
- momentum=0.9
- weight_decay=5e-4
- cosine
- tau=0.99
- seed=123

Checkpoint selection is by TRAIN-derived validation only.

Only after the best-validation checkpoint for each formal candidate is frozen/reloaded may official TEST be evaluated once for reporting. Do not repeatedly inspect TEST during training.

Formal report must clearly distinguish TRAIN-derived validation from official TEST.

---

# 8. Hardware-cost accounting

This is not RTL synthesis yet. Do not claim actual LUT/Fmax/DSP numbers.

For every coverage report analytical deployment cost relative to one-term:

- number of channels using 2-term
- percentage coverage
- additional coefficient metadata: second exponent, second sign, active flag if needed
- number of second shift operations per full inference under a direct per-channel interpretation
- number of additional add/sub operations per inference
- whether a reused affine engine would need a second parallel shift path or could serialize the second term
- if serialized: estimated added affine cycles
- if parallel: qualitative critical-path/LUT implication
- worst-case numeric range growth at the 2-term add

Run a numeric width audit for all selected 2-term affine outputs. Do NOT modify H2A-v2 widths because H2A-v2 is out of scope; only state whether later H2 re-audit would be required.

Mathematically this study must remain multiplier-free, but explicitly state that physical `DSP48E1=0` is not proven until RTL/Vivado synthesis.

---

# 9. Decision rules

This experiment does not promote anything to mainline.

Use the R4 one-term official TEST 86.11% as the historical reference, but use the matched formal control if its reproduced value differs materially.

Classify results descriptively:

- meaningful selective gain: >= +0.20 pp over matched one-term control
- strong selective gain: >= +0.35 pp
- near-full recovery: >= +0.50 pp of the historical 0.61 pp R2->R4 gap
- negligible: < +0.10 pp
- regression: <= 0 pp

A selective candidate is especially interesting if it reaches at least 80% of the full-2term accuracy gain while using <=25% 2-term coverage.

Do not automatically merge/promote even if these thresholds are met.

---

# 10. Required files/tests

Suggested isolated implementation:

- `recu_hw/spot_affine.py`
- `recu_hw/selective_spot.py`
- `tools/audit_selective_spot.py`
- `tools/train_selective_spot.py`
- `configs/selective_spot.json`
- `tests/test_spot_affine.py`
- `tests/test_selective_spot.py`
- `reports/selective_spot/SELECTIVE_SPOT_REPORT.md`
- `reports/selective_spot/SELECTIVE_SPOT_RESULTS.json`

Tests must cover at minimum:

- exact exhaustive 2-term nearest-value search on known examples
- canonical `k1 > k2`
- exact one-term fallback
- signs ± independently supported
- positive/negative K
- selected/unselected mixed module behavior
- checkpoint reload
- exact effective K after reload
- no NaN/Inf
- deterministic ranking and selection
- target coefficient count verification

Run the full repository unit test suite after adding the new isolated path. Existing 108/108 accuracy-rescue tests must remain passing; if the count grows, report the new exact total rather than claiming 108/108.

---

# 11. Final comparison report

Produce a table like:

| Coverage | 2-term channels | Best TRAIN val | Official TEST (formal only) | Delta vs one-term TEST | Recovered fraction of 0.61 pp | Extra shift/add cost | Decision |
|---:|---:|---:|---:|---:|---:|---|---|

Also answer explicitly:

1. How much does exact 2-term representation reduce K approximation error vs one-term?
2. Is approximation-error reduction correlated with validation accuracy recovery?
3. Which channels/layers are repeatedly selected by activation-aware ranking?
4. Does <=25% selective coverage capture most of the full-2term gain?
5. Does 100% 2-term materially outperform selective coverage?
6. Is any observed gain large enough to justify later porting the idea into the E1 85.51% H2A-v2-QAT pipeline?
7. What width/cycle/datapath changes would need re-audit before such a port?

STOP after this report. Do not merge to mainline, do not modify E1/H2A-v2, do not start RTL, and do not start another SPoT variant without user approval.