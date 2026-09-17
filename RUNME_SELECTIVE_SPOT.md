# RUNME — Selective 2-term SPoT sensitivity study

Branch: `exp/selective-2term-spot`

This is an isolated R4-level study. **Do not merge/promote it into the E1 85.51% mainline automatically.**

The local formal R2 best checkpoint is required and is intentionally not committed. Replace `<R2_BEST_PT>` below with the existing formal R2 checkpoint path.

## 0. Checkout and preflight

```bash
git checkout exp/selective-2term-spot
python -m unittest tests.test_spot_affine tests.test_selective_spot -v
python -m unittest discover -s tests -v
```

Stop on any failure. Do not train around a failed invariant.

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

## 2. S1 zero-shot TRAIN-derived validation sweep

```bash
python tools/eval_selective_spot_sweep.py \
  --config configs/selective_spot.json \
  --source-r2-checkpoint <R2_BEST_PT>
```

This evaluates 0/5/10/25/50/100% coverage on the fixed TRAIN-derived validation split only. It must not be used to report official TEST accuracy.

## 3. S2 20-epoch diagnostics

Run exactly the planned diagnostic coverages:

```bash
python tools/train_selective_spot.py --stage diagnostic --coverage 10  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage diagnostic --coverage 25  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage diagnostic --coverage 50  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage diagnostic --coverage 100 --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
```

Do not evaluate official TEST for these diagnostics.

Select the best **selective** coverage using only `best_train_derived_val_acc`. Keep 100% as the full-2term control, not as a selective candidate.

## 4. S3 formal 100-epoch runs

Run:

1. 0% matched one-term control
2. the best selective coverage chosen from S2 using TRAIN-derived validation
3. 100% full <=2-term control

Example if 25% wins S2:

```bash
python tools/train_selective_spot.py --stage formal --coverage 0   --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage formal --coverage 25  --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
python tools/train_selective_spot.py --stage formal --coverage 100 --config configs/selective_spot.json --source-r2-checkpoint <R2_BEST_PT>
```

Checkpoint selection remains TRAIN-derived validation only.

## 5. Official TEST — only after each formal best checkpoint is frozen

Run exactly once per frozen formal candidate:

```bash
python tools/train_selective_spot.py \
  --stage official-test \
  --config configs/selective_spot.json \
  --checkpoint <FORMAL_RUN_DIR>/best.pt
```

The tool refuses a repeated TEST call when `official_test.json` already exists unless an explicit force flag is used for a documented reproducibility check.

## 6. Interpretation

Historical references:

- R2: 86.72% official TEST
- R4 one-term: 86.11% official TEST
- target gap: 0.61 pp
- selected deployment mainline remains E1/H2A-v2-QAT: 85.51% official TEST

Do not directly compare TRAIN-derived validation percentages to official TEST percentages.

Useful labels for the matched formal R4 comparison:

- >= +0.20 pp: meaningful selective gain
- >= +0.35 pp: strong selective gain
- >= +0.50 pp: near-full recovery of the historical R2→R4 gap
- < +0.10 pp: negligible
- <= 0 pp: regression

Especially interesting: <=25% coverage captures >=80% of the full-2term TEST gain.

## Stop condition

After S3 + one-time official TEST + full unit-test suite, write the comparison report and stop. Do not merge to mainline, do not port SPoT into E1/H2A-v2, do not change H2 widths, and do not start RTL without explicit user approval.
