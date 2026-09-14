# H1-MP Mixed-Precision Search Report

> H1-S seeded mixed-precision search with H0 INT6 bias fixed. All policy, promotion, demotion, and final-plan decisions use TRAIN-derived data only.

## A. Source R8B / H0 verification

- R8B checkpoint: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r8b\recu_r8b_r7_pow2_fc_20260912_225827\best.pt`
- H0 INT6 bias JSON: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H0_INTEGER_BIAS_SWEEP_RESULTS.json`
- R8B stored/reload source verification: 85.4% / 85.40%.
- H0 search-validation baseline: **89.34%**.
- H0 official-test baseline: **85.13%**; test baseline was source verification only.
- H0 INT6 bias was reconstructed from stored q_values + shifts and remained fixed; R8B binary/pow2 invariants were unchanged.

## B. INT14 high-precision anchor

- All 58 nodes at INT14 search-validation accuracy: **89.48%**.
- Drop vs H0 search-validation baseline: **-0.14 pp**.
- Anchor saturation: **62820405**.
- Anchor gate (drop <=0.30 pp): **PASS**.

## C. H1-S sensitivity classification

| Class | Initial bits | Candidate ladder | Node count |
|---|---:|---|---:|
| `robust` | 6 | 6, 7, 8 | 38 |
| `mild` | 7 | 7, 8, 9 | 14 |
| `sensitive` | 10 | 8, 9, 10, 12 | 2 |
| `very_sensitive` | 12 | 10, 12, 14 | 4 |

Top H1-S nodes used to seed the plan:

| Rank | Node | H1-S drop | Class | Selected policy |
|---:|---|---:|---|---|
| 1 | `layer3.2.qrprelu_output` | +1.53 pp | `very_sensitive` | `mse` |
| 2 | `layer1.1.qrprelu_output` | +0.56 pp | `very_sensitive` | `mse` |
| 3 | `layer2.1.qrprelu_output` | +0.44 pp | `very_sensitive` | `absmax` |
| 4 | `layer3.1.qrprelu_output` | +0.37 pp | `very_sensitive` | `p99.9` |
| 5 | `layer2.0.first_affine` | +0.20 pp | `sensitive` | `absmax` |
| 6 | `layer2.1.second_add` | +0.12 pp | `sensitive` | `absmax` |
| 7 | `layer1.0.qrprelu_output` | +0.08 pp | `mild` | `mse` |
| 8 | `layer1.2.second_add` | +0.08 pp | `mild` | `mse` |
| 9 | `layer3.1.second_affine` | +0.07 pp | `mild` | `mse` |
| 10 | `layer3.2.second_add` | +0.06 pp | `mild` | `mse` |

## D. Final frozen 58-node plan

| Node | Class | Policy | Bits | Shift |
|---|---|---|---:|---:|
| `stem.post_hardtanh` | `robust` | `mse` | 6 | 5 |
| `layer1.0.first_affine` | `robust` | `mse` | 6 | 4 |
| `layer1.0.x1` | `robust` | `mse` | 6 | 3 |
| `layer1.0.pre_bconv2_hardtanh` | `robust` | `mse` | 6 | 5 |
| `layer1.0.second_affine` | `robust` | `mse` | 6 | 4 |
| `layer1.0.second_add` | `robust` | `mse` | 6 | 3 |
| `layer1.0.qrprelu_output` | `mild` | `mse` | 7 | 4 |
| `layer1.1.first_affine` | `robust` | `mse` | 6 | 4 |
| `layer1.1.x1` | `robust` | `mse` | 6 | 3 |
| `layer1.1.pre_bconv2_hardtanh` | `robust` | `mse` | 6 | 5 |
| `layer1.1.second_affine` | `robust` | `mse` | 6 | 4 |
| `layer1.1.second_add` | `robust` | `mse` | 6 | 3 |
| `layer1.1.qrprelu_output` | `very_sensitive` | `mse` | 12 | 8 |
| `layer1.2.first_affine` | `robust` | `mse` | 6 | 4 |
| `layer1.2.x1` | `robust` | `mse` | 6 | 3 |
| `layer1.2.pre_bconv2_hardtanh` | `robust` | `mse` | 6 | 5 |
| `layer1.2.second_affine` | `mild` | `mse` | 7 | 4 |
| `layer1.2.second_add` | `mild` | `mse` | 8 | 4 |
| `layer1.2.qrprelu_output` | `robust` | `mse` | 6 | 3 |
| `layer2.0.first_affine` | `sensitive` | `absmax` | 8 | 5 |
| `layer2.0.stage_shortcut` | `robust` | `mse` | 6 | 3 |
| `layer2.0.x1` | `robust` | `mse` | 6 | 3 |
| `layer2.0.pre_bconv2_hardtanh` | `robust` | `mse` | 6 | 5 |
| `layer2.0.second_affine` | `robust` | `mse` | 6 | 3 |
| `layer2.0.second_add` | `robust` | `mse` | 6 | 3 |
| `layer2.0.qrprelu_output` | `robust` | `p99.99` | 6 | 2 |
| `layer2.1.first_affine` | `mild` | `mse` | 7 | 5 |
| `layer2.1.x1` | `robust` | `mse` | 6 | 3 |
| `layer2.1.pre_bconv2_hardtanh` | `robust` | `mse` | 6 | 5 |
| `layer2.1.second_affine` | `mild` | `mse` | 7 | 4 |
| `layer2.1.second_add` | `sensitive` | `absmax` | 8 | 3 |
| `layer2.1.qrprelu_output` | `very_sensitive` | `absmax` | 12 | 8 |
| `layer2.2.first_affine` | `mild` | `mse` | 7 | 5 |
| `layer2.2.x1` | `robust` | `absmax` | 6 | 2 |
| `layer2.2.pre_bconv2_hardtanh` | `robust` | `absmax` | 6 | 4 |
| `layer2.2.second_affine` | `robust` | `mse` | 6 | 4 |
| `layer2.2.second_add` | `robust` | `mse` | 6 | 3 |
| `layer2.2.qrprelu_output` | `robust` | `mse` | 6 | 3 |
| `layer3.0.first_affine` | `robust` | `mse` | 6 | 4 |
| `layer3.0.stage_shortcut` | `robust` | `absmax` | 6 | 2 |
| `layer3.0.x1` | `robust` | `mse` | 6 | 3 |
| `layer3.0.pre_bconv2_hardtanh` | `robust` | `mse` | 6 | 5 |
| `layer3.0.second_affine` | `robust` | `mse` | 6 | 3 |
| `layer3.0.second_add` | `robust` | `mse` | 6 | 3 |
| `layer3.0.qrprelu_output` | `robust` | `mse` | 6 | 3 |
| `layer3.1.first_affine` | `mild` | `mse` | 7 | 4 |
| `layer3.1.x1` | `mild` | `mse` | 7 | 4 |
| `layer3.1.pre_bconv2_hardtanh` | `mild` | `mse` | 7 | 6 |
| `layer3.1.second_affine` | `mild` | `mse` | 7 | 4 |
| `layer3.1.second_add` | `robust` | `mse` | 7 | 3 |
| `layer3.1.qrprelu_output` | `very_sensitive` | `p99.9` | 10 | 7 |
| `layer3.2.first_affine` | `mild` | `mse` | 7 | 4 |
| `layer3.2.x1` | `robust` | `mse` | 6 | 3 |
| `layer3.2.pre_bconv2_hardtanh` | `mild` | `mse` | 7 | 6 |
| `layer3.2.second_affine` | `mild` | `mse` | 7 | 4 |
| `layer3.2.second_add` | `mild` | `mse` | 7 | 3 |
| `layer3.2.qrprelu_output` | `very_sensitive` | `mse` | 10 | 6 |
| `head.affine_output` | `robust` | `mse` | 6 | 2 |

## E. Search-validation history

| Step | Action | Node | Bits | Accuracy | Drop |
|---:|---|---|---|---:|---:|
| 0 | `initial` | `-` | - | 88.47% | +0.87 pp |
| 1 | `promote` | `layer3.2.qrprelu_output` | 12->14 | 88.46% | +0.88 pp |
| 2 | `promote` | `layer1.1.qrprelu_output` | 12->14 | 88.71% | +0.63 pp |
| 3 | `promote` | `layer2.1.qrprelu_output` | 12->14 | 88.63% | +0.71 pp |
| 4 | `promote` | `layer3.1.qrprelu_output` | 12->14 | 88.56% | +0.78 pp |
| 5 | `promote` | `layer2.0.first_affine` | 10->12 | 88.56% | +0.78 pp |
| 6 | `promote` | `layer2.1.second_add` | 10->12 | 88.56% | +0.78 pp |
| 7 | `promote` | `layer1.2.second_add` | 7->8 | 88.37% | +0.97 pp |
| 8 | `promote` | `layer1.2.second_add` | 8->9 | 88.37% | +0.97 pp |
| 9 | `promote` | `layer1.0.qrprelu_output` | 7->8 | 88.63% | +0.71 pp |
| 10 | `promote` | `layer1.0.qrprelu_output` | 8->9 | 88.41% | +0.93 pp |
| 11 | `promote` | `layer3.1.second_affine` | 7->8 | 88.32% | +1.02 pp |
| 12 | `promote` | `layer3.1.second_affine` | 8->9 | 88.32% | +1.02 pp |
| 13 | `promote` | `layer3.2.second_add` | 7->8 | 88.35% | +0.99 pp |
| 14 | `promote` | `layer3.2.second_add` | 8->9 | 88.35% | +0.99 pp |
| 15 | `promote` | `layer3.2.second_affine` | 7->8 | 88.37% | +0.97 pp |
| 16 | `promote` | `layer3.2.second_affine` | 8->9 | 88.37% | +0.97 pp |
| 17 | `promote` | `layer3.2.pre_bconv2_hardtanh` | 7->8 | 88.37% | +0.97 pp |
| 18 | `promote` | `layer3.2.pre_bconv2_hardtanh` | 8->9 | 88.37% | +0.97 pp |
| 19 | `promote` | `layer3.1.x1` | 7->8 | 88.39% | +0.95 pp |
| 20 | `promote` | `layer3.1.x1` | 8->9 | 88.39% | +0.95 pp |
| 21 | `promote` | `layer3.1.pre_bconv2_hardtanh` | 7->8 | 88.39% | +0.95 pp |
| 22 | `promote` | `layer3.1.pre_bconv2_hardtanh` | 8->9 | 88.39% | +0.95 pp |
| 23 | `promote` | `layer2.1.first_affine` | 7->8 | 88.67% | +0.67 pp |
| 24 | `promote` | `layer2.1.first_affine` | 8->9 | 88.66% | +0.68 pp |
| 25 | `promote` | `layer3.2.first_affine` | 7->8 | 88.73% | +0.61 pp |
| 26 | `promote` | `layer3.2.first_affine` | 8->9 | 88.72% | +0.62 pp |
| 27 | `promote` | `layer3.1.first_affine` | 7->8 | 88.65% | +0.69 pp |
| 28 | `promote` | `layer3.1.first_affine` | 8->9 | 88.67% | +0.67 pp |
| 29 | `promote` | `layer2.2.first_affine` | 7->8 | 88.62% | +0.72 pp |
| 30 | `promote` | `layer2.2.first_affine` | 8->9 | 88.39% | +0.95 pp |
| 31 | `promote` | `layer2.1.second_affine` | 7->8 | 88.45% | +0.89 pp |
| 32 | `promote` | `layer2.1.second_affine` | 8->9 | 88.47% | +0.87 pp |
| 33 | `promote` | `layer1.2.second_affine` | 7->8 | 88.68% | +0.66 pp |
| 34 | `promote` | `layer1.2.second_affine` | 8->9 | 88.97% | +0.37 pp |
| 35 | `promote` | `layer2.0.qrprelu_output` | 6->7 | 88.98% | +0.36 pp |
| 36 | `promote` | `layer2.0.qrprelu_output` | 7->8 | 88.39% | +0.95 pp |
| 37 | `promote` | `layer1.0.second_affine` | 6->7 | 88.39% | +0.95 pp |
| 38 | `promote` | `layer1.0.second_affine` | 7->8 | 88.88% | +0.46 pp |
| 39 | `promote` | `stem.post_hardtanh` | 6->7 | 88.64% | +0.70 pp |
| 40 | `promote` | `stem.post_hardtanh` | 7->8 | 88.64% | +0.70 pp |
| 41 | `promote` | `layer3.2.x1` | 6->7 | 88.59% | +0.75 pp |
| 42 | `promote` | `layer3.2.x1` | 7->8 | 88.59% | +0.75 pp |
| 43 | `promote` | `layer3.1.second_add` | 6->7 | 88.81% | +0.53 pp |
| 44 | `promote` | `layer3.1.second_add` | 7->8 | 88.93% | +0.41 pp |
| 45 | `promote` | `layer3.0.x1` | 6->7 | 88.95% | +0.39 pp |
| 46 | `promote` | `layer3.0.x1` | 7->8 | 88.95% | +0.39 pp |
| 47 | `promote` | `layer3.0.stage_shortcut` | 6->7 | 88.89% | +0.45 pp |
| 48 | `promote` | `layer3.0.stage_shortcut` | 7->8 | 88.89% | +0.45 pp |
| 49 | `promote` | `layer3.0.second_affine` | 6->7 | 89.27% | +0.07 pp |
| 50 | `demote_accept` | `stem.post_hardtanh` | 8->7 | 89.27% | +0.07 pp |
| 51 | `demote_accept` | `layer3.0.stage_shortcut` | 8->7 | 89.27% | +0.07 pp |
| 52 | `demote_accept` | `layer3.0.x1` | 8->7 | 89.27% | +0.07 pp |
| 53 | `demote_reject` | `layer3.0.second_affine` | 7->6 | 88.89% | +0.45 pp |
| 54 | `demote_reject` | `layer3.1.second_add` | 8->7 | 88.93% | +0.41 pp |
| 55 | `demote_accept` | `layer3.2.x1` | 8->7 | 89.20% | +0.14 pp |
| 56 | `demote_reject` | `layer1.0.second_affine` | 8->7 | 88.72% | +0.62 pp |
| 57 | `demote_accept` | `layer2.0.qrprelu_output` | 8->7 | 89.22% | +0.12 pp |
| 58 | `demote_reject` | `layer1.2.second_affine` | 9->8 | 88.86% | +0.48 pp |
| 59 | `demote_accept` | `layer2.1.second_affine` | 9->8 | 89.13% | +0.21 pp |
| 60 | `demote_accept` | `layer2.2.first_affine` | 9->8 | 89.40% | -0.06 pp |
| 61 | `demote_accept` | `layer3.1.first_affine` | 9->8 | 89.17% | +0.17 pp |
| 62 | `demote_accept` | `layer3.2.first_affine` | 9->8 | 89.17% | +0.17 pp |
| 63 | `demote_accept` | `layer2.1.first_affine` | 9->8 | 89.21% | +0.13 pp |
| 64 | `demote_accept` | `layer3.1.x1` | 9->8 | 89.44% | -0.10 pp |
| 65 | `demote_accept` | `layer3.1.pre_bconv2_hardtanh` | 9->8 | 89.44% | -0.10 pp |
| 66 | `demote_accept` | `layer3.2.pre_bconv2_hardtanh` | 9->8 | 89.44% | -0.10 pp |
| 67 | `demote_accept` | `layer3.2.second_affine` | 9->8 | 89.37% | -0.03 pp |
| 68 | `demote_accept` | `layer3.2.second_add` | 9->8 | 89.34% | +0.00 pp |
| 69 | `demote_accept` | `layer3.1.second_affine` | 9->8 | 89.34% | +0.00 pp |
| 70 | `demote_accept` | `layer1.0.qrprelu_output` | 9->8 | 89.28% | +0.06 pp |
| 71 | `demote_reject` | `layer1.2.second_add` | 9->8 | 88.96% | +0.38 pp |
| 72 | `demote_accept` | `layer2.1.second_add` | 12->10 | 89.28% | +0.06 pp |
| 73 | `demote_accept` | `layer2.0.first_affine` | 12->10 | 89.28% | +0.06 pp |
| 74 | `demote_accept` | `layer3.1.qrprelu_output` | 14->12 | 89.26% | +0.08 pp |
| 75 | `demote_accept` | `layer2.1.qrprelu_output` | 14->12 | 89.21% | +0.13 pp |
| 76 | `demote_accept` | `layer1.1.qrprelu_output` | 14->12 | 89.23% | +0.11 pp |
| 77 | `demote_accept` | `layer3.2.qrprelu_output` | 14->12 | 89.23% | +0.11 pp |
| 78 | `demote_accept` | `stem.post_hardtanh` | 7->6 | 89.56% | -0.22 pp |
| 79 | `demote_accept` | `layer3.0.stage_shortcut` | 7->6 | 89.21% | +0.13 pp |
| 80 | `demote_accept` | `layer3.0.x1` | 7->6 | 89.37% | -0.03 pp |
| 81 | `demote_accept` | `layer3.0.second_affine` | 7->6 | 89.58% | -0.24 pp |
| 82 | `demote_accept` | `layer3.1.second_add` | 8->7 | 89.07% | +0.27 pp |
| 83 | `demote_accept` | `layer3.2.x1` | 7->6 | 89.31% | +0.03 pp |
| 84 | `demote_accept` | `layer1.0.second_affine` | 8->7 | 89.20% | +0.14 pp |
| 85 | `demote_accept` | `layer2.0.qrprelu_output` | 7->6 | 89.25% | +0.09 pp |
| 86 | `demote_reject` | `layer1.2.second_affine` | 9->8 | 88.99% | +0.35 pp |
| 87 | `demote_accept` | `layer2.1.second_affine` | 8->7 | 89.18% | +0.16 pp |
| 88 | `demote_accept` | `layer2.2.first_affine` | 8->7 | 89.14% | +0.20 pp |
| 89 | `demote_reject` | `layer3.1.first_affine` | 8->7 | 88.91% | +0.43 pp |
| 90 | `demote_accept` | `layer3.2.first_affine` | 8->7 | 89.13% | +0.21 pp |
| 91 | `demote_accept` | `layer2.1.first_affine` | 8->7 | 89.41% | -0.07 pp |
| 92 | `demote_accept` | `layer3.1.x1` | 8->7 | 89.43% | -0.09 pp |
| 93 | `demote_accept` | `layer3.1.pre_bconv2_hardtanh` | 8->7 | 89.43% | -0.09 pp |
| 94 | `demote_accept` | `layer3.2.pre_bconv2_hardtanh` | 8->7 | 89.43% | -0.09 pp |
| 95 | `demote_accept` | `layer3.2.second_affine` | 8->7 | 89.47% | -0.13 pp |
| 96 | `demote_accept` | `layer3.2.second_add` | 8->7 | 89.33% | +0.01 pp |
| 97 | `demote_accept` | `layer3.1.second_affine` | 8->7 | 89.20% | +0.14 pp |
| 98 | `demote_accept` | `layer1.0.qrprelu_output` | 8->7 | 89.32% | +0.02 pp |
| 99 | `demote_reject` | `layer1.2.second_add` | 9->8 | 88.92% | +0.42 pp |
| 100 | `demote_accept` | `layer2.1.second_add` | 10->9 | 89.32% | +0.02 pp |
| 101 | `demote_accept` | `layer2.0.first_affine` | 10->9 | 89.46% | -0.12 pp |
| 102 | `demote_accept` | `layer3.1.qrprelu_output` | 12->10 | 89.35% | -0.01 pp |
| 103 | `demote_reject` | `layer2.1.qrprelu_output` | 12->10 | 88.22% | +1.12 pp |
| 104 | `demote_reject` | `layer1.1.qrprelu_output` | 12->10 | 88.79% | +0.55 pp |
| 105 | `demote_accept` | `layer3.2.qrprelu_output` | 12->10 | 89.13% | +0.21 pp |
| 106 | `demote_reject` | `layer3.1.second_add` | 7->6 | 88.66% | +0.68 pp |
| 107 | `demote_accept` | `layer1.0.second_affine` | 7->6 | 89.13% | +0.21 pp |
| 108 | `demote_accept` | `layer1.2.second_affine` | 9->8 | 89.22% | +0.12 pp |
| 109 | `demote_accept` | `layer3.1.first_affine` | 8->7 | 89.09% | +0.25 pp |
| 110 | `demote_reject` | `layer1.2.second_add` | 9->8 | 88.61% | +0.73 pp |
| 111 | `demote_reject` | `layer2.1.second_add` | 9->8 | 88.92% | +0.42 pp |
| 112 | `demote_accept` | `layer2.0.first_affine` | 9->8 | 89.15% | +0.19 pp |
| 113 | `demote_reject` | `layer2.1.qrprelu_output` | 12->10 | 88.05% | +1.29 pp |
| 114 | `demote_reject` | `layer1.1.qrprelu_output` | 12->10 | 88.77% | +0.57 pp |
| 115 | `demote_reject` | `layer3.1.second_add` | 7->6 | 88.74% | +0.60 pp |
| 116 | `demote_accept` | `layer1.2.second_affine` | 8->7 | 89.17% | +0.17 pp |
| 117 | `demote_accept` | `layer1.2.second_add` | 9->8 | 89.17% | +0.17 pp |
| 118 | `demote_accept` | `layer2.1.second_add` | 9->8 | 89.18% | +0.16 pp |
| 119 | `demote_reject` | `layer2.1.qrprelu_output` | 12->10 | 87.90% | +1.44 pp |
| 120 | `demote_reject` | `layer1.1.qrprelu_output` | 12->10 | 88.69% | +0.65 pp |
| 121 | `demote_reject` | `layer3.1.second_add` | 7->6 | 88.60% | +0.74 pp |
| 122 | `demote_reject` | `layer1.2.second_add` | 8->7 | 88.90% | +0.44 pp |
| 123 | `demote_reject` | `layer2.1.qrprelu_output` | 12->10 | 87.90% | +1.44 pp |
| 124 | `demote_reject` | `layer1.1.qrprelu_output` | 12->10 | 88.69% | +0.65 pp |

## F. Final accuracy and storage metrics

- Final search-validation accuracy: **89.18%**.
- Final search-validation drop vs H0: **+0.16 pp**.
- Production target <=0.30 pp: **PASS**.
- Official test accuracy after freeze: **85.03%**.
- Official test delta vs H0 85.13%: **-0.10 pp**.
- Unweighted average node bits: **6.6897**.
- Activation-volume-weighted average bits: **6.6240**.
- Nodes at 6 bits: `stem.post_hardtanh`, `layer1.0.first_affine`, `layer1.0.x1`, `layer1.0.pre_bconv2_hardtanh`, `layer1.0.second_affine`, `layer1.0.second_add`, `layer1.1.first_affine`, `layer1.1.x1`, `layer1.1.pre_bconv2_hardtanh`, `layer1.1.second_affine`, `layer1.1.second_add`, `layer1.2.first_affine`, `layer1.2.x1`, `layer1.2.pre_bconv2_hardtanh`, `layer1.2.qrprelu_output`, `layer2.0.stage_shortcut`, `layer2.0.x1`, `layer2.0.pre_bconv2_hardtanh`, `layer2.0.second_affine`, `layer2.0.second_add`, `layer2.0.qrprelu_output`, `layer2.1.x1`, `layer2.1.pre_bconv2_hardtanh`, `layer2.2.x1`, `layer2.2.pre_bconv2_hardtanh`, `layer2.2.second_affine`, `layer2.2.second_add`, `layer2.2.qrprelu_output`, `layer3.0.first_affine`, `layer3.0.stage_shortcut`, `layer3.0.x1`, `layer3.0.pre_bconv2_hardtanh`, `layer3.0.second_affine`, `layer3.0.second_add`, `layer3.0.qrprelu_output`, `layer3.2.x1`, `head.affine_output`.
- Nodes at 7 bits: `layer1.0.qrprelu_output`, `layer1.2.second_affine`, `layer2.1.first_affine`, `layer2.1.second_affine`, `layer2.2.first_affine`, `layer3.1.first_affine`, `layer3.1.x1`, `layer3.1.pre_bconv2_hardtanh`, `layer3.1.second_affine`, `layer3.1.second_add`, `layer3.2.first_affine`, `layer3.2.pre_bconv2_hardtanh`, `layer3.2.second_affine`, `layer3.2.second_add`.
- Nodes at 10/12/14 bits: `layer3.1.qrprelu_output`, `layer3.2.qrprelu_output`, `layer1.1.qrprelu_output`, `layer2.1.qrprelu_output`.
- `layer3.0.qrprelu_output` final width: **6 bits**.

## G. Mixed residual alignment

All 18 residual adds were checked with per-node bits/shifts, common output scale, integer alignment, saturation, and equivalence.

| Add node | A bits/shift | B bits/shift | Output bits/shift | Delta A/B | Saturation | PASS |
|---|---|---|---|---|---:|---|
| `layer1.0.x1` | 6/4 | 6/5 | 6/3 | -1/-2 | 0 | True |
| `layer1.0.second_add` | 6/4 | 6/3 | 6/3 | -1/0 | 7605 | True |
| `layer1.1.x1` | 6/4 | 7/4 | 6/3 | -1/-1 | 32589 | True |
| `layer1.1.second_add` | 6/4 | 6/3 | 6/3 | -1/0 | 266948 | True |
| `layer1.2.x1` | 6/4 | 12/8 | 6/3 | -1/-5 | 293714 | True |
| `layer1.2.second_add` | 7/4 | 6/3 | 8/4 | 0/1 | 0 | True |
| `layer2.0.x1` | 8/5 | 6/3 | 6/3 | -2/0 | 2324 | True |
| `layer2.0.second_add` | 6/3 | 6/3 | 6/3 | 0/0 | 62433 | True |
| `layer2.1.x1` | 7/5 | 6/2 | 6/3 | -2/1 | 55291 | True |
| `layer2.1.second_add` | 7/4 | 6/3 | 8/3 | -1/0 | 0 | True |
| `layer2.2.x1` | 7/5 | 12/8 | 6/2 | -3/-6 | 0 | True |
| `layer2.2.second_add` | 6/4 | 6/2 | 6/3 | -1/1 | 170830 | True |
| `layer3.0.x1` | 6/4 | 6/2 | 6/3 | -1/1 | 17228 | True |
| `layer3.0.second_add` | 6/3 | 6/3 | 6/3 | 0/0 | 27066 | True |
| `layer3.1.x1` | 7/4 | 6/3 | 7/4 | 0/1 | 9975 | True |
| `layer3.1.second_add` | 7/4 | 7/4 | 7/3 | -1/-1 | 0 | True |
| `layer3.2.x1` | 7/4 | 10/7 | 6/3 | -1/-4 | 42801 | True |
| `layer3.2.second_add` | 7/4 | 6/3 | 7/3 | -1/0 | 0 | True |

## H. Test leakage and excluded arithmetic

- Calibration and search-validation were non-overlapping CIFAR-10 TRAIN subsets with seed 20260913.
- Official TEST was not used to choose shifts, policies, ranking, promotions, demotions, or the final plan.
- Binary-convolution accumulators, GAP accumulation, FC accumulation, and final logits were not quantized.

## I. Hardware interpretation

H0 affine/classifier bias remains fixed INT6. The final plan stores each selected activation/residual state as signed integer q with node-level integer shift metadata. This is not a claim that the entire network is fully finite-width integer-only; accumulator widths remain a later experiment.
