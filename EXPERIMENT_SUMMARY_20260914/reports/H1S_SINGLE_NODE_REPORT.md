# H1-S Single-Node INT8 Sensitivity Report

> Exactly one verified H1 node was quantized at a time. All policy and ranking decisions use held-out CIFAR-10 TRAIN data only.

## Source and methodology

- R8B checkpoint: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\experiments\recu_r8b\recu_r8b_r7_pow2_fc_20260912_225827\best.pt`
- H0 INT6 bias JSON: `C:\Users\MCLAB\OneDrive - 國立臺灣科技大學\ForLab\ReCU_ResNet20_DSPFree_Research\H0_INTEGER_BIAS_SWEEP_RESULTS.json`
- H0 search-validation baseline: **89.34%**
- H0 official-test baseline (source verification only): **85.13%**
- Calibration: CIFAR-10 TRAIN subset, 10000 samples.
- Search-validation: a non-overlapping CIFAR-10 TRAIN subset, 10000 samples, deterministic raw-domain ToTensor.
- Split seed: `20260913`; official test used for search: **False**.
- H0 INT6 bias remained fixed. The 58 verified H1 boundaries and 18 residual sites were reused.

## Single-node ranking

| Rank | Node | Selected policy | Shift | Only-this-node INT8 accuracy | Drop vs search baseline | Saturation | Class |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `layer3.2.qrprelu_output` | `mse` | 4 | 87.81% | +1.53 pp | 219 | `very_sensitive` |
| 2 | `layer1.1.qrprelu_output` | `mse` | 5 | 88.78% | +0.56 pp | 122473 | `very_sensitive` |
| 3 | `layer2.1.qrprelu_output` | `absmax` | 4 | 88.90% | +0.44 pp | 0 | `very_sensitive` |
| 4 | `layer3.1.qrprelu_output` | `p99.9` | 5 | 88.97% | +0.37 pp | 16494 | `very_sensitive` |
| 5 | `layer2.0.first_affine` | `absmax` | 5 | 89.14% | +0.20 pp | 0 | `sensitive` |
| 6 | `layer2.1.second_add` | `absmax` | 3 | 89.22% | +0.12 pp | 0 | `sensitive` |
| 7 | `layer1.0.qrprelu_output` | `mse` | 5 | 89.26% | +0.08 pp | 5884 | `mild` |
| 8 | `layer1.2.second_add` | `mse` | 4 | 89.26% | +0.08 pp | 2239 | `mild` |
| 9 | `layer3.1.second_affine` | `mse` | 5 | 89.27% | +0.07 pp | 13679 | `mild` |
| 10 | `layer3.2.second_add` | `mse` | 4 | 89.28% | +0.06 pp | 306 | `mild` |
| 11 | `layer3.2.second_affine` | `mse` | 5 | 89.30% | +0.04 pp | 2601 | `mild` |
| 12 | `layer2.1.first_affine` | `mse` | 6 | 89.31% | +0.03 pp | 21897 | `mild` |
| 13 | `layer3.1.x1` | `mse` | 4 | 89.31% | +0.03 pp | 17 | `mild` |
| 14 | `layer3.1.pre_bconv2_hardtanh` | `mse` | 7 | 89.31% | +0.03 pp | 2329316 | `mild` |
| 15 | `layer3.2.pre_bconv2_hardtanh` | `mse` | 7 | 89.31% | +0.03 pp | 2606993 | `mild` |
| 16 | `layer1.2.second_affine` | `mse` | 5 | 89.32% | +0.02 pp | 0 | `mild` |
| 17 | `layer2.1.second_affine` | `mse` | 5 | 89.32% | +0.02 pp | 13 | `mild` |
| 18 | `layer2.2.first_affine` | `mse` | 5 | 89.32% | +0.02 pp | 0 | `mild` |
| 19 | `layer3.1.first_affine` | `mse` | 5 | 89.32% | +0.02 pp | 146 | `mild` |
| 20 | `layer3.2.first_affine` | `mse` | 5 | 89.32% | +0.02 pp | 88 | `mild` |
| 21 | `layer1.0.second_affine` | `mse` | 6 | 89.33% | +0.01 pp | 41835 | `robust` |
| 22 | `layer2.0.qrprelu_output` | `p99.99` | 4 | 89.33% | +0.01 pp | 0 | `robust` |
| 23 | `stem.post_hardtanh` | `mse` | 6 | 89.34% | +0.00 pp | 0 | `robust` |
| 24 | `layer1.1.first_affine` | `mse` | 6 | 89.34% | +0.00 pp | 1848 | `robust` |
| 25 | `layer3.0.pre_bconv2_hardtanh` | `mse` | 7 | 89.34% | +0.00 pp | 2014116 | `robust` |
| 26 | `head.affine_output` | `mse` | 4 | 89.34% | +0.00 pp | 353 | `robust` |
| 27 | `layer1.0.first_affine` | `mse` | 6 | 89.35% | -0.01 pp | 226 | `robust` |
| 28 | `layer3.1.second_add` | `mse` | 4 | 89.35% | -0.01 pp | 484 | `robust` |
| 29 | `layer2.0.pre_bconv2_hardtanh` | `mse` | 7 | 89.36% | -0.02 pp | 4182464 | `robust` |
| 30 | `layer1.0.second_add` | `mse` | 5 | 89.37% | -0.03 pp | 5895 | `robust` |
| 31 | `layer1.0.pre_bconv2_hardtanh` | `mse` | 6 | 89.38% | -0.04 pp | 0 | `robust` |
| 32 | `layer3.0.second_add` | `mse` | 4 | 89.38% | -0.04 pp | 0 | `robust` |
| 33 | `layer3.2.x1` | `mse` | 4 | 89.43% | -0.09 pp | 23 | `robust` |
| 34 | `layer1.1.x1` | `mse` | 5 | 89.44% | -0.10 pp | 27862 | `robust` |
| 35 | `layer1.1.pre_bconv2_hardtanh` | `mse` | 6 | 89.44% | -0.10 pp | 0 | `robust` |
| 36 | `layer3.0.qrprelu_output` | `mse` | 5 | 89.45% | -0.11 pp | 7383 | `robust` |
| 37 | `layer3.0.first_affine` | `mse` | 5 | 89.48% | -0.14 pp | 3 | `robust` |
| 38 | `layer1.1.second_affine` | `mse` | 5 | 89.49% | -0.15 pp | 0 | `robust` |
| 39 | `layer2.1.pre_bconv2_hardtanh` | `mse` | 7 | 89.49% | -0.15 pp | 4420619 | `robust` |
| 40 | `layer2.2.second_affine` | `mse` | 5 | 89.49% | -0.15 pp | 0 | `robust` |
| 41 | `layer1.2.qrprelu_output` | `mse` | 5 | 89.51% | -0.17 pp | 8803 | `robust` |
| 42 | `layer2.2.second_add` | `mse` | 4 | 89.51% | -0.17 pp | 4 | `robust` |
| 43 | `layer3.0.stage_shortcut` | `absmax` | 4 | 89.51% | -0.17 pp | 0 | `robust` |
| 44 | `layer2.1.x1` | `mse` | 4 | 89.55% | -0.21 pp | 8 | `robust` |
| 45 | `layer2.2.qrprelu_output` | `mse` | 5 | 89.55% | -0.21 pp | 23477 | `robust` |
| 46 | `layer3.0.second_affine` | `mse` | 5 | 89.55% | -0.21 pp | 39 | `robust` |
| 47 | `layer1.2.pre_bconv2_hardtanh` | `mse` | 7 | 89.56% | -0.22 pp | 21265805 | `robust` |
| 48 | `layer2.0.second_affine` | `mse` | 5 | 89.57% | -0.23 pp | 1737 | `robust` |
| 49 | `layer1.2.x1` | `mse` | 4 | 89.60% | -0.26 pp | 1 | `robust` |
| 50 | `layer2.2.x1` | `absmax` | 4 | 89.61% | -0.27 pp | 0 | `robust` |
| 51 | `layer2.0.second_add` | `mse` | 4 | 89.62% | -0.28 pp | 0 | `robust` |
| 52 | `layer1.0.x1` | `mse` | 5 | 89.63% | -0.29 pp | 0 | `robust` |
| 53 | `layer2.2.pre_bconv2_hardtanh` | `absmax` | 6 | 89.69% | -0.35 pp | 0 | `robust` |
| 54 | `layer1.2.first_affine` | `mse` | 6 | 89.71% | -0.37 pp | 2087 | `robust` |
| 55 | `layer3.0.x1` | `mse` | 5 | 89.71% | -0.37 pp | 19854 | `robust` |
| 56 | `layer2.0.stage_shortcut` | `mse` | 5 | 89.74% | -0.40 pp | 2256 | `robust` |
| 57 | `layer2.0.x1` | `mse` | 5 | 89.74% | -0.40 pp | 9805 | `robust` |
| 58 | `layer1.1.second_add` | `mse` | 4 | 89.80% | -0.46 pp | 0 | `robust` |

## Top-10 scale-policy ablation

### `layer3.2.qrprelu_output`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `mse` | 4 | 87.81% | +1.53 pp | 219 |
| `p99.9` | 4 | 87.81% | +1.53 pp | 219 |
| `p99.99` | 4 | 87.81% | +1.53 pp | 219 |
| `absmax` | 3 | 87.64% | +1.70 pp | 0 |

### `layer2.1.qrprelu_output`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `absmax` | 4 | 88.90% | +0.44 pp | 0 |
| `p99.99` | 4 | 88.90% | +0.44 pp | 0 |
| `mse` | 5 | 88.76% | +0.58 pp | 19591 |
| `p99.9` | 5 | 88.76% | +0.58 pp | 19591 |

### `layer1.1.qrprelu_output`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `mse` | 5 | 88.78% | +0.56 pp | 122473 |
| `p99.9` | 5 | 88.78% | +0.56 pp | 122473 |
| `absmax` | 4 | 86.64% | +2.70 pp | 0 |
| `p99.99` | 4 | 86.64% | +2.70 pp | 0 |

### `layer3.1.qrprelu_output`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `p99.9` | 5 | 88.97% | +0.37 pp | 16494 |
| `mse` | 4 | 88.91% | +0.43 pp | 5 |
| `p99.99` | 4 | 88.91% | +0.43 pp | 5 |
| `absmax` | 3 | 88.78% | +0.56 pp | 0 |

### `layer2.0.first_affine`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `absmax` | 5 | 89.14% | +0.20 pp | 0 |
| `p99.99` | 5 | 89.14% | +0.20 pp | 0 |
| `mse` | 6 | 89.10% | +0.24 pp | 13104 |
| `p99.9` | 6 | 89.10% | +0.24 pp | 13104 |

### `layer2.0.qrprelu_output`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `p99.99` | 4 | 89.33% | +0.01 pp | 0 |
| `absmax` | 3 | 89.16% | +0.18 pp | 0 |
| `mse` | 5 | 89.13% | +0.21 pp | 30751 |
| `p99.9` | 5 | 89.13% | +0.21 pp | 30751 |

### `layer2.2.x1`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `absmax` | 4 | 89.61% | -0.27 pp | 0 |
| `p99.99` | 4 | 89.61% | -0.27 pp | 0 |
| `mse` | 5 | 89.14% | +0.20 pp | 55783 |
| `p99.9` | 5 | 89.14% | +0.20 pp | 55783 |

### `layer2.1.second_add`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `absmax` | 3 | 89.22% | +0.12 pp | 0 |
| `mse` | 4 | 89.21% | +0.13 pp | 17 |
| `p99.9` | 4 | 89.21% | +0.13 pp | 17 |
| `p99.99` | 4 | 89.21% | +0.13 pp | 17 |

### `layer3.0.stage_shortcut`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `absmax` | 4 | 89.51% | -0.17 pp | 0 |
| `p99.99` | 4 | 89.51% | -0.17 pp | 0 |
| `mse` | 5 | 89.23% | +0.11 pp | 5346 |
| `p99.9` | 5 | 89.23% | +0.11 pp | 5346 |

### `layer2.2.pre_bconv2_hardtanh`

| Policy | Shift | Search-validation accuracy | Drop | Saturation |
|---|---:|---:|---:|---:|
| `absmax` | 6 | 89.69% | -0.35 pp | 0 |
| `p99.9` | 6 | 89.69% | -0.35 pp | 0 |
| `p99.99` | 6 | 89.69% | -0.35 pp | 0 |
| `mse` | 7 | 89.25% | +0.09 pp | 4820431 |

## Interpretation

- Most sensitive nodes: `layer3.2.qrprelu_output`, `layer1.1.qrprelu_output`, `layer2.1.qrprelu_output`, `layer3.1.qrprelu_output`, `layer2.0.first_affine`, `layer2.1.second_add`, `layer1.0.qrprelu_output`, `layer1.2.second_add`, `layer3.1.second_affine`, `layer3.2.second_add`.
- `layer3.0.qrprelu_output` remains rank 36 with drop -0.11 pp.
- Robust/near-unaffected nodes (drop <=0.02 pp): `layer1.0.second_affine`, `layer2.0.qrprelu_output`, `stem.post_hardtanh`, `layer1.1.first_affine`, `layer3.0.pre_bconv2_hardtanh`, `head.affine_output`, `layer1.0.first_affine`, `layer3.1.second_add`, `layer2.0.pre_bconv2_hardtanh`, `layer1.0.second_add`, `layer1.0.pre_bconv2_hardtanh`, `layer3.0.second_add`, `layer3.2.x1`, `layer1.1.x1`, `layer1.1.pre_bconv2_hardtanh`, `layer3.0.qrprelu_output`, `layer3.0.first_affine`, `layer1.1.second_affine`, `layer2.1.pre_bconv2_hardtanh`, `layer2.2.second_affine`, `layer1.2.qrprelu_output`, `layer2.2.second_add`, `layer3.0.stage_shortcut`, `layer2.1.x1`, `layer2.2.qrprelu_output`, `layer3.0.second_affine`, `layer1.2.pre_bconv2_hardtanh`, `layer2.0.second_affine`, `layer1.2.x1`, `layer2.2.x1`, `layer2.0.second_add`, `layer1.0.x1`, `layer2.2.pre_bconv2_hardtanh`, `layer1.2.first_affine`, `layer3.0.x1`, `layer2.0.stage_shortcut`, `layer2.0.x1`, `layer1.1.second_add`.
- Clearly affected nodes (drop >0.30 pp): `layer3.2.qrprelu_output`, `layer1.1.qrprelu_output`, `layer2.1.qrprelu_output`, `layer3.1.qrprelu_output`.
- Policy ablation improved validation accuracy over MSE for: `layer2.1.qrprelu_output`, `layer3.1.qrprelu_output`, `layer2.0.first_affine`, `layer2.0.qrprelu_output`, `layer2.2.x1`, `layer2.1.second_add`, `layer3.0.stage_shortcut`, `layer2.2.pre_bconv2_hardtanh`.
- Official test was not used to select policies, rank nodes, or choose any mixed-precision width.
