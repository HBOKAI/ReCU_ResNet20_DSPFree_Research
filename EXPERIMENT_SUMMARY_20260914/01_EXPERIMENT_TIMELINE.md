# 實驗時間線與單一主要變因

時間線依研究/模型依賴順序，而非所有診斷 run 的 wall-clock timestamp。官方 TEST、TRAIN search-validation、早期歷史回報均分別標註；完整每列 source/delta/decision 見 [experiment_results.csv](tables/experiment_results.csv)。`reports/` 和 `results_json/` 保留正式來源副本。

| 順序 | 實驗 / source | 目的與主要變因 | 結果及判斷 |
|---:|---|---|---|
| 0 | FP32、minimal W1A1、failed ReCU/SiMaN adaptation | 早期參照與非正式移植 | 91.06%、72.60%、43.29% 為先前 prompt 記載；SiMaN 數值與各原始 artifact **NOT FOUND**。標為 historical/failed，絕非 official ReCU reproduction。 |
| 1 | Official ReCU / scratch | 600 epochs 重現 3+3+3 double-skip 網路 | best **87.28% @ epoch 600**，reload 一致；270,858 trainable，backbone 40,108,032 binary terms/image。 |
| 2 | H1 QRPReLU scratch / scratch | PReLU→QRPReLU，從頭 600 epochs | **84.28% @ 598**；弱於 warm-start。 |
| 3 | H2 QRPReLU+pow2 alpha scratch / scratch | 再加 pow2 alpha，從頭 600 epochs | **84.60% @ 597**；歷史探索，非後續 H2 finite-width。 |
| 4 | R1 / Official ReCU best | alpha 設 1、100-epoch warm-start | **87.02% @100**，相對 source −0.26 pp；直接轉換只有 8.44%，需 fine-tune。 |
| 5 | R2 / Official ReCU best | PReLU→QRPReLU，100-epoch warm-start | **86.72% @97**，−0.56 pp；選為 R4 source。 |
| 6 | R3 / Official ReCU best | alpha+BN fold→signed-pow2 K，PReLU 保留 | **85.50% @100**，−1.78 pp；浮點 fold 誤差 2.74e−6，證明硬體分支可行。 |
| 7 | R4 / R2 best | 18 條 binary branch 改 fused signed-pow2 affine | **86.11% @98**，相對 R2 −0.61 pp；fold 最大 1.19e−7，13/13 tests，選用。 |
| 8 | R5 / R4 best | 原始 RGB 3-channel W1A8 naive stem | **83.77% @98**，−2.34 pp；不選。 |
| 9 | R5AB / R4 best | scaled W1A8 + progressive lambda | **83.96% @100**，比 R5 +0.19 pp 但仍未達門檻；不選。 |
| 10 | R5T T1→T2 / R4 best→T1 best | Thermometer R=8 / FP stem 預適應→binary W1A1 stem | **85.29% @99 → 85.00% @97**；binary stem 96→16，選擇此 input representation。 |
| 11 | R5T-Long T1→T2 / R4 best→T1-Long best | 相同架構與 optimizer，完整 200+200 epoch schedule | **85.89% @200 → 85.14% @200**。T1 比舊版 +0.60 pp，T2 僅 +0.14 pp；T2-Long 作 R6 source。 |
| 12 | R6 / R5T-Long T2 best | stem BN→signed-pow2 affine，fine-tune | hard projection 84.90%，best **85.52% @97**；fold 9.536743e−7、K exponent −6/−5/−4。 |
| 13 | R7 / R6 best | head BN→獨立 signed-pow2 affine，FC 不變 | hard projection 75.80%；LR1e−3 首步失穩，正式 LR1e−4；best **85.22% @99**、fold 1.430511e−6、58 tests。 |
| 14 | R8A vs R8B / R7 best | classifier W1 FC vs signed-pow2 FC | **82.84% @95 vs 85.40% @100**；R8B +2.56 pp，正式 software baseline，62/62 tests。 |
| 15 | H0 / R8B best | 只量化 21 個 affine B + FC bias，INT8→4 sweep | INT6 **85.13%**、saturation 0，選 production INT6；沒有重訓。 |
| 16 | H1 uniform / H0 INT6 | 58 activation/residual 節點同位寬 INT8→4 | INT8 僅 **77.07%**；uniform 失敗，18/18 residual 對齊已驗證。 |
| 17 | H1-S / H0 INT6 | TRAIN-derived 58/58 單節點及四 scale policies | search-validation baseline 89.34%；Top-1 `layer3.2.qrprelu_output` 單節點 INT8 drop 1.53 pp。**不是 official TEST。** |
| 18 | H1-MP / H0 INT6 | 依敏感度分配 58 node bits/shifts/policies、promotion/demotion | INT14 anchor TRAIN val 89.48%、final TRAIN val 89.18%；plan freeze 後 TEST **85.03%**，18/18 align。 |
| 19 | H2A 原版 / H1-MP | 首版 exact/conservative width，GAP `/64` 用 rounded `>>6` | *84.92% 歷史回報*，−0.11 pp；原版報告/JSON **NOT FOUND**，不與 v2 混用。 |
| 20 | H2A-v2 / H1-MP | 唯一修正 GAP deferred scaling | exact GAP error 0，TEST **85.03%**、H2 overflow 0、18/18 align、92/92 tests；**safe**。 |
| 21 | H2B / H2A-v2 | 依序只縮 QRP inner/output、FC、logits、GAP | TRAIN val 88.95→88.65%；freeze 後 TEST **84.93%**，H2 finite saturation 2,078,706，binary overflow 0、92/92 tests；**aggressive alternative**。 |

R1/R2/R3 是三個從 Official best 出發的分支，並非 R1→R2→R3 串接；R5/R5AB 也是平行對照。R5T 與 R5T-Long 的 T2 各自從**同一輪 T1 best** 初始化，並非由前一輪舊 T2 接訓。H0/H1/H2 是**數值化實驗**，不涉及重新訓練模型。H1/S/MP 的 search-validation 與 official TEST 不可直接相減成同一組 delta。

Checkpoint metadata 複核：R1、R2、R3、R4、R5、R5AB、R5T T1/T2、R5T-Long T1/T2、R6、R7、R8A、R8B 共 **14 個 formal best checkpoint**，逐一讀取 `epoch`/`best_acc`；它們與正式 report 所列最佳 epoch/accuracy 相符。R5T 的檔名為 `t1_best.pt`/`t2_best.pt`，其餘多為 `best.pt`。Official ReCU 的 best checkpoint 另核對為 epoch 600、87.28%。歸檔只留 JSON summary/history，**不複製 checkpoint**。
