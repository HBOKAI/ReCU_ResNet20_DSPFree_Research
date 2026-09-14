# 完整準確率比較

下表除有 `TRAIN validation` 或 `歷史回報` 標記外，皆為 **CIFAR-10 official TEST**；數字使用 best checkpoint 或量化後官方 test 評估。`Δ source` 是 accuracy 減 source accuracy，pp；scratch/historical 無同一來源，故記 `—`。**不能拿 TRAIN validation 的 89.xx% 與 official TEST 的 85.xx% 當同資料切分比較。** 機器可讀的 36-row 完整 schema（含 Params、Binary Ops、bias/activation/accumulator precision、training/status/decision/evidence）在 [experiment_results.csv](tables/experiment_results.csv) 和 [experiment_results.json](tables/experiment_results.json)。

## Baseline 與模型 ablation

| 實驗 | Source | 主要變因 | Best/TEST | Δ source | 判斷 |
|---|---|---|---:|---:|---|
| FP32 ResNet20 | prior baseline | 全精度參照 | *91.06%* | — | 歷史 prompt 數值；原始結果 **NOT FOUND**。 |
| Minimal W1A1 | prior baseline | 最小二值模型 | *72.60%* | — | 歷史 prompt 數值；原始結果 **NOT FOUND**。 |
| Early failed ReCU adaptation | prior adaptation | 非官方移植 | *43.29%* | — | historical/failed；不是 official reproduction，原始結果 **NOT FOUND**。 |
| Early failed SiMaN adaptation | prior adaptation | 非官方移植 | — | — | historical/failed；數值及原始結果 **NOT FOUND**。 |
| Official ReCU | scratch | 600-epoch official recipe | **87.28%** | — | reproduced reference。 |
| H1 QRPReLU scratch | scratch | PReLU→QRPReLU | 84.28% | — | 歷史 scratch ablation。 |
| H2 QRPReLU+pow2alpha scratch | scratch | QRPReLU + pow2 alpha | 84.60% | — | 歷史 scratch ablation；非 H2 finite-width。 |
| R1 | Official 87.28 | remove alpha | 87.02% | −0.26 | warm-start 後可回復，非直轉。 |
| R2 | Official 87.28 | QRPReLU warm-start | 86.72% | −0.56 | R4 source。 |
| R3 | Official 87.28 | fused signed-pow2 affine，PReLU 保留 | 85.50% | −1.78 | fold 路徑驗證。 |
| R4 | R2 86.72 | multiplier-free binary backbone | **86.11%** | −0.61 | 接受作 stem ablation source。 |
| R5 | R4 86.11 | raw RGB, 3-channel naive W1A8 stem | 83.77% | −2.34 | reject。 |
| R5AB | R4 86.11 | scaled/progressive W1A8 stem | 83.96% | −2.15 | 比 R5 +0.19，但仍 reject。 |
| R5T T1 | R4 86.11 | Thermometer R=8, 96-channel FP stem | 85.29% | −0.82 | T2 初始化。 |
| R5T T2 | T1 85.29 | 96-channel W1A1 stem | **85.00%** | −0.29 | binary stem。 |
| R5T-Long T1 | R4 86.11 | 同架構，200-epoch FP stem | 85.89% | −0.22 | vs 舊 T1 +0.60。 |
| R5T-Long T2 | T1-Long 85.89 | 同架構，200-epoch W1A1 stem | **85.14%** | −0.75 | vs 舊 T2 +0.14；R6 source。 |
| R6 | T2-Long 85.14 | stem BN→pow2 affine | **85.52%** | +0.38 | 接受。 |
| R7 | R6 85.52 | head BN→pow2 affine | 85.22% | −0.30 | R8 source；降低 LR 後穩定。 |
| R8A | R7 85.22 | W1 FC | 82.84% | −2.38 | reject。 |
| **R8B** | R7 85.22 | signed-pow2 FC | **85.40%** | **+0.18** | selected software model；vs R8A **+2.56**。 |

R1/R2/R3、R5/R5AB、R8A/R8B 為**平行分支**，並非前一列接下一列。R5 stem 使用 W1A8、442,368 select/negate sites，不能和 R5T stem 的 14,155,776 binary XNOR/popcount terms 當同一種 MAC 比。[架構與 workload](02_MODEL_ARCHITECTURE.md)

## H0 / H1 / H2 官方 TEST

| 實驗 | Source | INT/數值變因 | TEST | Δ source | Saturation / 結論 |
|---|---|---|---:|---:|---|
| H0 INT8 bias | R8B 85.40 | 21 bias tensors | 85.12% | −0.28 | 0；對照。 |
| H0 INT7 bias | R8B 85.40 | 同上 | 85.06% | −0.34 | 0；對照。 |
| **H0 INT6 bias** | R8B 85.40 | 同上 | **85.13%** | **−0.27** | 0；selected production bias。 |
| H0 INT5 bias | R8B 85.40 | 同上 | 84.35% | −1.05 | 0；不選。 |
| H0 INT4 bias | R8B 85.40 | 同上 | 83.24% | −2.16 | 0；不選。 |
| H1 uniform INT8 | H0 85.13 | 58 nodes all INT8 | 77.07% | −8.06 | saturation 0；仍顯著失準。 |
| H1 uniform INT7 | H0 85.13 | all INT7 | 68.82% | −16.31 | saturation 0。 |
| H1 uniform INT6 | H0 85.13 | all INT6 | 35.45% | −49.68 | saturation 0。 |
| H1 uniform INT5 | H0 85.13 | all INT5 | 10.00% | −75.13 | saturation 0。 |
| H1 uniform INT4 | H0 85.13 | all INT4 | 13.51% | −71.62 | 83 events；uniform rejected。 |
| H1-S | H0 INT6 | single-node TRAIN analysis | — | — | 58/58 覆蓋；**無 final official TEST**。 |
| **H1-MP** | H0 85.13 | 58-node mixed plan | **85.03%** | **−0.10** | 18/18 residual alignment。 |
| H2A original | H1MP 85.03 | rounded GAP `>>6` | *84.92%* | *−0.11* | 歷史執行回報；原始 result/report **NOT FOUND**，未採納。 |
| **H2A-v2 Safe** | H1MP 85.03 | GAP deferred scaling | **85.03%** | **0.00** | H2 overflow 0、18/18 align、92/92 tests。 |
| **H2B Aggressive** | H2A-v2 85.03 | narrower internal widths | **84.93%** | **−0.10** | H2 finite saturation 2,078,706；binary overflow 0、92/92 tests。 |

H1 uniform INT8 的 failure **不是飽和次數增加才造成**：INT8 整體 saturation=0 仍 −8.06 pp，表明量化誤差、scale/敏感節點的累積影響。H1S/H1MP 的 scale search 使用不重疊的 TRAIN-derived calibration / validation；H2B 的寬度 accept/reject 亦只看 TRAIN-derived validation，TEST 在 freeze 後才評估。

## H1-S Top-10（僅 TRAIN search-validation）

H0 TRAIN search-validation baseline **89.34%**；下表是**只量化該單一節點為 INT8**的 accuracy 與 drop，絕非整網 official TEST。完整 58-node 順序與 absmax/mse/p99.9/p99.99 policy ablation 在 [H1S result](results_json/H1S_SINGLE_NODE_RESULTS.json)及[原始報告](reports/H1S_SINGLE_NODE_REPORT.md)。

| Rank | Node | Selected policy / shift | 單點 validation | Drop vs 89.34 | Saturation |
|---:|---|---|---:|---:|---:|
| 1 | `layer3.2.qrprelu_output` | mse / 4 | 87.81% | +1.53 pp | 219 |
| 2 | `layer1.1.qrprelu_output` | mse / 5 | 88.78% | +0.56 pp | 122,473 |
| 3 | `layer2.1.qrprelu_output` | absmax / 4 | 88.90% | +0.44 pp | 0 |
| 4 | `layer3.1.qrprelu_output` | p99.9 / 5 | 88.97% | +0.37 pp | 16,494 |
| 5 | `layer2.0.first_affine` | absmax / 5 | 89.14% | +0.20 pp | 0 |
| 6 | `layer2.1.second_add` | absmax / 3 | 89.22% | +0.12 pp | 0 |
| 7 | `layer1.0.qrprelu_output` | mse / 5 | 89.26% | +0.08 pp | 5,884 |
| 8 | `layer1.2.second_add` | mse / 4 | 89.26% | +0.08 pp | 2,239 |
| 9 | `layer3.1.second_affine` | mse / 5 | 89.27% | +0.07 pp | 13,679 |
| 10 | `layer3.2.second_add` | mse / 4 | 89.28% | +0.06 pp | 306 |

四種 scale policy 的**原始 Top-10 trial set**如下。這個 trial set 由最初敏感度篩選，與上面「採最佳 policy 後重新排序的 final Top-10」不完全相同；數值均是單節點 INT8 的 **TRAIN search-validation accuracy**，不是 official TEST。括號內為該節點最終選定 policy。

| Policy trial node（selected） | absmax | mse | p99.9 | p99.99 |
|---|---:|---:|---:|---:|
| `layer3.2.qrprelu_output`（mse） | 87.64% | **87.81%** | 87.81% | 87.81% |
| `layer2.1.qrprelu_output`（absmax） | **88.90%** | 88.76% | 88.76% | 88.90% |
| `layer1.1.qrprelu_output`（mse） | 86.64% | **88.78%** | 88.78% | 86.64% |
| `layer3.1.qrprelu_output`（p99.9） | 88.78% | 88.91% | **88.97%** | 88.91% |
| `layer2.0.first_affine`（absmax） | **89.14%** | 89.10% | 89.10% | 89.14% |
| `layer2.0.qrprelu_output`（p99.99） | 89.16% | 89.13% | 89.13% | **89.33%** |
| `layer2.2.x1`（absmax） | **89.61%** | 89.14% | 89.14% | 89.61% |
| `layer2.1.second_add`（absmax） | **89.22%** | 89.21% | 89.21% | 89.21% |
| `layer3.0.stage_shortcut`（absmax） | **89.51%** | 89.23% | 89.23% | 89.51% |
| `layer2.2.pre_bconv2_hardtanh`（absmax） | **89.69%** | 89.25% | 89.69% | 89.69% |

Tie 情況依原始 [H1S result](results_json/H1S_SINGLE_NODE_RESULTS.json)的最終 policy selection；完整 shift 與 saturation 見[H1S report](reports/H1S_SINGLE_NODE_REPORT.md)。

## H1-MP / H2B TRAIN-only selection curves

H1-MP：INT14 anchor TRAIN validation **89.48%**；最終 TRAIN validation **89.18%**。相對 H0 TRAIN baseline 89.34% 為 **−0.16 pp**；其 official TEST 85.03% 則相對 H0 TEST 85.13% 為 **−0.10 pp**。58-node 分布：INT6×37、INT7×14、INT8×3、INT10×2、INT12×2，平均 6.6897 bit、volume-weighted 6.6240 bit。[完整 bits/shifts/policies](tables/hardware_widths.csv)及[search trace](results_json/H1MP_SEARCH_RESULTS.json)。

H2B：H2A-v2 的 TRAIN validation baseline **88.95%**，最終 frozen 計畫 **88.65%**，準確率變化 **−0.30 pp**（若以正數 *drop* 記則 **+0.30 pp**）。各單 family 的逐 bit accept/reject 點見[數值演進](04_HARDWARE_NUMERIC_EVOLUTION.md)與[原始 JSON](results_json/H2B_WIDTH_OPTIMIZATION_RESULTS.json)。在這些 TRAIN 決定完成後，official TEST 僅用於 final 評估 **84.93%**。
