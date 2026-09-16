# Research archive and discussion guide

This private repository contains research code, configurations, tests, and recorded experiment results. It is not a complete dataset/checkpoint backup. Local datasets, training checkpoints, model-weight exports, logs, smoke-run outputs, and unpacked extension packs are intentionally excluded. Some historical reports contain local Windows paths; those identify local artifacts and are not downloadable GitHub links.

## Branches

- `main`: original local research snapshot before the GMP experiment; safe H2A-v2 baseline included.
- `exp/gmp-ablation`: isolated GAP-to-GMP experiment; not merged into main.
- `archive/workspace-20260916`: discussion/archive snapshot including GMP and subsequent workspace code/reports. This branch does not promote experimental results into the official baseline.

## Official safe numerical baseline

R8B-H2A-v2: official CIFAR-10 TEST **85.03%**, H2 finite-width overflow **0**. GAP uses `q_gap=sum(q_i)`, `s_gap=s_input+6`, without runtime division/rounding. No RTL/synthesis or measured DSP/area/power/timing result is claimed.

- [H2A-v2 report](H2_FINITE_WIDTH_REPORT.md)
- [H2A-v2 results](H2_FINITE_WIDTH_RESULTS.json)
- [H1MP report](H1MP_MIXED_PRECISION_REPORT.md)
- [H2B optimization report](H2B_WIDTH_OPTIMIZATION_REPORT.md): 84.93%, narrowed internal nodes can saturate; not a replacement for the conservative anchor.
- [GMP report](GMP_ABLATION_REPORT.md): trained epoch 100 66.48%; rejected as the next mainline.
- [R8B 600-epoch control](reports/r8b_long/R8B_LONG_RESULTS.md): 84.91%; independent control, not promoted.
- [Bias precision ablation](reports/B_PRECISION_ABLATION.md)
- [Reference inference](reference_inference/REFERENCE_INFERENCE_REPORT.md)

## Discussing the repository

A private repository link alone does not grant another conversation access. Authorize the repository through an available GitHub connection, or attach the relevant report/code files. Start discussions with the branch name, `GITHUB_RESEARCH_INDEX.md`, and the specific report; distinguish software accuracy, numerical hardware replay, and synthesized hardware claims.

For the original setup and training instructions, see [README](README.md). Existing recipe/report limitations remain authoritative; this archive does not rerun or independently validate later experiments.
