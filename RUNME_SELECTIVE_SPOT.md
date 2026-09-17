# RUNME — Selective 2-term SPoT sensitivity study

Branch: `exp/selective-2term-spot`

This is an isolated R4-level study. **Do not merge/promote it into the E1 85.51% mainline automatically.**

The local formal R2 best checkpoint is required and is intentionally not committed. Replace `<R2_BEST_PT>` below with the existing formal R2 checkpoint path.

## 0. Checkout and preflight

```bash
git checkout exp/selective-2term-spot
python -m unittest tests.test_spot_affine tests.test_selective_spot tests.test_spot_hardware -v
python -m unittest discover -s tests -v
```

A branch-local CPU GitHub Actions workflow also runs these tests on push. Stop on any failure. Do not train around a failed invariant.

## 1. S0/S0B coefficient + TRAIN-only activation audit

```bash
python tools/audit_selective_spot.py \
  --config configs/selective_spot.json \
  --source-r2-checkpoint <R2_BEST_PT>
```

Required invariants before continuing:

- target K count = 672
- 0% SPoT K mismatch vs historical R4 `FusedAffine2d` = 0
- no official TEST dataset is loaded
- ranking file exists at `reports/selective_spot/S0B_ACTIVATION_AWARE_RANKING.json`

The primary ranking is TRAIN-calibration activation-aware recoverable distortion. It is not selected from TEST.

## 2. S1 zero-shot TRAIN-derived validation sweep

```bash
python tools/eval_selective_spot_sweep.py \
  --config configs/selective_spot.json \
  --source-r2-checkpoint <R2_BEST_PT>
```

This evaluates 0/5/10/25/50/100% coverage on the fixed TRAIN-derived validation split only. It writes an exact module/channel manifest for every coverage. It must not be used to report official TEST accuracy.

## 3. S2 20-epoch diagnostics

Run exactly the planned diagnostic coverages:

```bash
python tools/train_selective_spot.py --stage diagnostic --coverage 10  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage diagnostic --coverage 25  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage diagnostic --coverage 50  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage diagnostic --coverage 100 --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
```

Do not evaluate official TEST for these diagnostics.

Then select the best selective coverage automatically using TRAIN-derived validation only:

```bash
python tools/select_selective_spot_candidate.py \
  --config configs/selective_spot.json \
  --source-r2-checkpoint-placeholder <R2_BEST_PT>
```

This tool refuses to select if any planned diagnostic run is missing, excludes 100% from the selective-candidate competition, and uses deterministic tie breaking: higher TRAIN-val first, then lower coverage, then earlier best epoch.

It writes:

- `reports/selective_spot/S2_DIAGNOSTIC_SELECTION.json`
- `reports/selective_spot/S2_DIAGNOSTIC_SELECTION.md`

and prints the exact three formal commands to run next.

## 4. S3 formal 100-epoch runs

Run exactly:

1. 0% matched one-term control
2. the best selective coverage chosen by S2 TRAIN-derived validation
3. 100% full <=2-term control

Example if 25% wins S2:

```bash
python tools/train_selective_spot.py --stage formal --coverage 0   --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage formal --coverage 25  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage formal --coverage 100 --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
```

Checkpoint selection remains TRAIN-derived validation only. Do not substitute historical R4 TEST for the matched 0% control when computing the formal SPoT delta; report both, but attribution uses the matched formal control.

## 5. Official TEST — only after each formal best checkpoint is frozen

Run once for each of the three frozen `best.pt` checkpoints:

```bash
python tools/train_selective_spot.py \
  --stage official-test \
  --config configs/selective_spot.json \
  --checkpoint <FORMAL_RUN_DIR>/best.pt
```

The tool refuses a repeated TEST call when `official_test.json` already exists unless an explicit force flag is used for a documented reproducibility check.

## 6. Build the final report + hardware/width audit

After all three formal runs have exactly one `official_test.json`, run:

```bash
python tools/build_selective_spot_report.py \
  --config configs/selective_spot.json \
  --control-run <FORMAL_0PCT_RUN_DIR> \
  --selective-run <FORMAL_SELECTED_RUN_DIR> \
  --full-run <FORMAL_100PCT_RUN_DIR>
```

This tool verifies that the selective formal coverage matches `S2_DIAGNOSTIC_SELECTION.json`, then produces:

- `reports/selective_spot/SELECTIVE_SPOT_RESULTS.json`
- `reports/selective_spot/SELECTIVE_SPOT_REPORT.md`

The report includes:

- official TEST deltas vs matched 0% control
- recovered fraction of the historical R2->R4 0.61 pp gap
- fraction of full-2term gain captured by the selective candidate
- one-term vs <=2-term K approximation error reduction
- TRAIN-only coverage-level correlation between recoverable-distortion coverage and zero-shot validation gain
- selected/active 2-term concentration by stage/module
- extra shifts/adds per image
- dense-programmable and sparse/static coefficient metadata overhead
- conservative pre-bias K*S width/carry audit
- whether an H2 width/overflow re-audit is required before any E1 port

The width audit is intentionally limited to the multiplier-free K*S path because this R4-level study still keeps floating B. It does not claim a complete H2 fixed-point output width.

## 7. Interpretation

Historical references:

- R2: 86.72% official TEST
- R4 one-term: 86.11% official TEST
- target gap: 0.61 pp
- selected deployment mainline remains E1/H2A-v2-QAT: 85.51% official TEST

Do not directly compare TRAIN-derived validation percentages to official TEST percentages.

Useful labels for the matched formal R4 comparison:

- >= +0.20 pp: meaningful selective gain
- >= +0.35 pp: strong selective gain
- >= +0.50 pp: near-full recovery of the historical R2->R4 gap
- < +0.10 pp: negligible
- <= 0 pp: regression

Especially interesting: <=25% coverage captures >=80% of the full-2term TEST gain.

## Hardware interpretation guardrails

Mathematically, active 2-term K is implemented as:

`K*S = s1*(S << or >> k1) + s2*(S << or >> k2)`

so no arbitrary general multiplier is required. However:

- Python-level multiplier-free arithmetic is not physical proof
- `DSP48E1=0` requires RTL/Vivado synthesis
- parallel 2-term implementation adds a second shift/sign path plus add/sub
- serialized implementation can reuse the shift path but costs extra affine cycles
- any later E1/H2A-v2 port must re-audit scales, intermediate widths, overflow, rounding, and schedule

## Stop condition

After S3 + one-time official TEST + final report + full unit-test suite, stop. Do not merge to mainline, do not port SPoT into E1/H2A-v2, do not change H2 widths, and do not start RTL without explicit user approval.
