# 硬體友善數值化演進：R8B → H2B

這條線**不再重訓模型**。R8B 固定 architecture/weights/pow2 exponents，H0 固定 bias，H1-S/H1-MP 以 TRAIN-derived subsets 搜尋 activation/residual 的 scale/位寬，H2A-v2/H2B 再明確約束內部 accumulator。各階段可消除的只是**軟體數值契約中的 float/reference arithmetic**；真實 DSP/面積/時序仍需 synthesis。

| 階段 | 此步的唯一主要數值變化 | Official TEST | 仍待處理 / 判斷 |
|---|---|---:|---|
| **R8B** | W1A1 Conv、signed-pow2 stem/backbone/head affine K 與 signed-pow2 FC K | 85.40% | 算法層面 general-multiplier-free；B/FC bias 仍 FP/reference，activation/residual 及內部 accumulator 尚未 finite-width。 |
| **H0** | 21 個 affine B + FC bias，762 values → 帶 per-tensor power-of-two scale 的整數；選 INT6 | 85.13% | 偏置已 pure integer，58 個 activation/residual 邊界仍待規格化。 |
| **H1 uniform** | 同時把 58 邊界套相同 INT8→INT4 | INT8 77.07% | 均一位寬失敗；不能因 INT8 表面上足夠就全網設成 INT8。 |
| **H1-S** | 逐一 INT8 節點測 sensitivity，四種 scale policy | — | 58/58 node 排名為 H1-MP 提供先驗；只有 TRAIN validation。 |
| **H1-MP** | 每節點 frozen bits/shift/policy；18 個 residual 的 integer scale alignment | 85.03% | 58-node 平均 6.6897 bit、volume 6.6240 bit；binary accumulator、GAP、FC、logits 尚需 H2 finite width。 |
| **H2A 原版** | 保守 exact widths，但 GAP sum 後 rounded arithmetic `>>6` | *84.92%* | *−0.11 pp 歷史回報*，原版獨立 artifact NOT FOUND；主要新 rounding point 是 GAP，不是 binary overflow 或 residual alignment。 |
| **H2A-v2** | 僅改 GAP 為 deferred scale：保留 `q_sum`，scale +6 | **85.03%** | GAP 數學誤差 0、H2 overflow 0；保守 finite-width **safe profile**。 |
| **H2B** | TRAIN-only 序列縮減 QRPReLU、FC、logits、GAP 位寬 | **84.93%** | 更小位寬有 H2 finite saturation 2,078,706；**aggressive alternative**，不覆蓋 safe。 |

## GAP `/64` 的數學修正

H2A 原版在 64 個空間值的整數和 `q_sum` 後執行 symmetric rounding 的 arithmetic right shift 6，等於新增一次實際取整；影響 official TEST 的歷史回報為 85.03→84.92%（−0.11 pp）。H2A-v2 改成：

```text
q_gap = q_sum
s_gap = s_input + 6
physical value = q_gap × 2^(−s_gap)
               = (q_sum / 64) × 2^(−s_input)
```

`GAP /64` **只由 scale metadata 表示**，不執行 `>>6` rounding。現存 [H2A-v2 result](results_json/H2_FINITE_WIDTH_RESULTS.json)記錄 float/reference vs integer-implicit-scale 的最大 GAP 誤差 **0**；後續 head signed-pow2 affine 按新 scale 對齊，仍是 shift + optional negate + integer bias add。H0 INT6 與 H1-MP 58-node bits/shifts/policies 不變。

## H1-S/H1-MP 為何必要

H1 uniform 的 INT8 **77.07%**，即使總 saturation 0，仍比 H0 低 8.06 pp；等寬截斷及各節點 scale/rounding 誤差的串接，不能用「無 overflow」判定精度安全。H1-S 的 Top-1 `layer3.2.qrprelu_output` 單節點 INT8 就讓 TRAIN validation 89.34→87.81%，drop 1.53 pp。H1-MP 根據敏感度先 promotion，再逐步 demotion，最後留下 INT6×37、INT7×14、INT8×3、INT10×2、INT12×2；其 58-node `bits/shift/policy` 與 H2 profiles 完整列於[hardware_widths.csv](tables/hardware_widths.csv)。搜尋使用 TRAIN-derived calibration/search-validation，official TEST 只評估 frozen plan。

## H2B 逐類縮位決策（只看 TRAIN validation）

H2A-v2 在 H2B 搜尋切分的 TRAIN validation baseline **88.95%**。依內部程式的 fixed family order，一次只變動一族，其餘 freeze；`drop` 是**正數損失**。完整每 bit 的範圍、saturation、accept/reject 見[H2B JSON](results_json/H2B_WIDTH_OPTIMIZATION_RESULTS.json)與[原始報告](reports/H2B_WIDTH_OPTIMIZATION_REPORT.md)。

| Family | Accept 到 | 下一 bit reject / rollback | 最終 TRAIN validation（該步） | 最終 width |
|---|---|---|---:|---:|
| QRPReLU inner | INT29→28→27 | INT26：88.32%，drop 0.63 pp | 88.90% | **INT27** |
| QRPReLU output | INT37→36→35 | INT34：85.44%，drop 3.51 pp | 88.80% | **INT35** |
| FC accumulator | INT23→…→16 | INT15：81.69%，drop 7.26 pp | 88.75% | **INT16** |
| Final logits | INT23→…→16 | INT15：83.40%，drop 5.55 pp | 88.75% | **INT16** |
| GAP accumulator | INT15→14→13 | INT12：87.37%，drop 1.58 pp | **88.65%** | **INT13** |

最終 TRAIN validation 88.65%，相對 baseline **−0.30 pp**；plan freeze 後才跑一次 official TEST 得 **84.93%**，相對 H2A-v2 official TEST **−0.10 pp**。H2B 表中的 `saturation_total` 包含既有 H1MP 邊界：TEST 合計 **110,040,676**；真正因 H2 narrowed finite nodes 的 TEST saturation 是 **2,078,706**。Binary Conv accumulator 按 exact width 規格維持 **0 overflow**，但 H2B 全域 `overflow_total`/H2 finite saturation 不為 0，兩者不可混稱。[兩 profile 詳細規格](05_CURRENT_HARDWARE_SPEC.md)
