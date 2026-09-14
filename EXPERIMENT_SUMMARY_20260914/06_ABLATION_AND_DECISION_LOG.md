# Ablation、選型與來源衝突紀錄

此頁採 `Decision / Reason / Evidence / Rejected alternative` 形式。準確率均為 official TEST，除非明確標為 TRAIN validation 或歷史回報。Source of truth 順序：formal final report → machine result JSON → checkpoint metadata → console/log；若只有先前 prompt/對話而沒有目前 workspace artifact，不能提高證據等級。

| Decision | Reason | Evidence | Rejected alternative |
|---|---|---|---|
| 不選 Minimal W1A1 | 先前回報 72.60%，低於後續可重現 ReCU 87.28%。但 minimal 原始檔缺失，僅作背景。 | [R1–R3 報告](reports/R1_R2_R3_FINAL_REPORT.md)確認 official；minimal 數字只見 prior prompt `PROMPT_CODEX_R1_R2_R3.md`，原始結果 **NOT FOUND**。 | 把 minimal 當正式 binary baseline。 |
| 選 Official ReCU 作研究起點 | 600-epoch official architecture/recipe 重現至 87.28%@600；有 history 與 checkpoint metadata。早期 failed ReCU/SiMaN adaptation 不可混入。 | [official history](results_json/official_recu_history.json)、[R1–R3 報告](reports/R1_R2_R3_FINAL_REPORT.md)。 | 使用非官方失敗移植（ReCU 43.29% 歷史回報；SiMaN 數值未找到）。 |
| 選 QRPReLU 路徑 | QRPReLU 的 slope 為 pow2 可用 shift 實作；R2 warm-start 86.72%，較 scratch H1 84.28% 更適合後續 R4。R2 初始轉換 11.86% 且有 89 個 non-positive PReLU slopes，故需 fine-tune。 | [R1–R3 報告](reports/R1_R2_R3_FINAL_REPORT.md)、[H1 scratch history](results_json/h1_qrprelu_scratch_history.json)。 | 直接採 scratch H1 或零訓練轉換。 |
| 接受 R4 | 從 R2 的 18 條 `alpha+BN` branch fold 成 signed-pow2 affine，最高 float-fold error 1.19e−7；最佳準確率 86.11%，以 −0.61 pp 換取 multiplier-free backbone。 | [R4 報告](reports/R4_FINAL_REPORT.md)與[summary](results_json/r4_summary.json)：86.11%@98、reload 86.11%、13/13 tests、672 K exponents −7:11/−6:452/−5:209。 | 保留 binary branch 的一般浮點係數/BN；或採獨立 R3 85.50%。 |
| 不選 R5/R5AB | 3-channel raw-RGB W1A8 stem 相對 R4 損失 2.34/2.15 pp；R5AB 只比 R5 高 0.19 pp，均低於預定 84.5% 門檻。 | [R5 報告](reports/R5_FINAL_REPORT.md)、[R5AB 報告](reports/R5AB_FINAL_REPORT.md)。 | R5 83.77%、R5AB 83.96%。 |
| 選 Thermometer R=8 | 輸入轉為 96 bipolar channels 後可用 W1A1 binary stem，R5T T2 85.00% 高於 R5/R5AB，同時將 stem 變成 XNOR/popcount 語意。 | [R5T 報告](reports/R5T_FINAL_REPORT.md)：T1 85.29%、T2 85.00%，stem 14,155,776 binary terms/image。 | 仍使用 3-channel W1A8 stem。此決策不宣稱 R=4/16 已完整比較。 |
| 以 R5T-Long T2 作 R6 source | controlled 200+200 epoch schedule：T1 85.89%（比舊 T1 +0.60），T2 binary 85.14%（比舊 T2 +0.14）；R6 必須從具 binary stem 的 T2-Long best 而非較高的 FP-stem T1。 | [R5T-Long 報告](reports/R5T_LONG_FINAL_REPORT.md)與兩個 [T1](results_json/r5t_long_t1_summary.json)/[T2](results_json/r5t_long_t2_summary.json) summaries。 | 直接用 FP stem T1 或舊 T2 當 R6 source。 |
| 接受 R6 stem affine | Exact BN fold 9.536743e−7；hard projection 84.90%，正式 fine-tune 至 85.52%，K exponent −6/−5/−4；移除 stem BN。 | [R6 報告](reports/R6_FINAL_REPORT.md)：best @97、reload、48 tests。 | 仍保留 stem BN；誤稱 hard projection 本身無損。 |
| R7 保留獨立 head affine，不併入 FC | 正式 R7 單變因是移除 head BN2，同時**保持 FC weights/bias 不變**；獨立 affine 讓 exact BN fold 和下一步 R8A/R8B classifier ablation 可分別核對。這是受控比較的設計理由，**不是**宣稱數學上不可併入 FC。 | [R7 報告](reports/R7_FINAL_REPORT.md)明列 `GAP→signed-pow2 affine→unchanged FC`、fold error 1.430511e−6。 | 在 R7 同時重寫/折入 FC、破壞單變因與 R8 classifier 對照。 |
| R7 正式 LR1e−4 | 移除 train-mode BN2 後，LR1e−3 第一完整網路 step 將 75.80% 降至 48.39%；LR1e−4 的 20-epoch 診斷穩定，正式 best 85.22%@99、58 tests。 | [R7 報告](reports/R7_FINAL_REPORT.md)。 | 繼續使用失穩的 LR1e−3。 |
| 選 R8B、不選 R8A | 兩者共用 R7 source，binary FC 82.84%，signed-pow2 FC 85.40%，差 **+2.56 pp**；後者仍只需 shift/optional negate。 | [R8 報告](reports/R8_FINAL_REPORT.md)與 [R8A](results_json/r8a_summary.json)/[R8B](results_json/r8b_summary.json) summaries；62/62 tests。 | 最小 FC storage 的 W1 方案。 |
| H0 選 INT6 | 21 bias tensors/762 values；INT6 85.13%、全部 0 saturation，相對 R8B −0.27 pp，INT5/4 明顯更差。 | [H0 JSON](results_json/H0_INTEGER_BIAS_SWEEP_RESULTS.json)、[H0 報告](reports/H0_INTEGER_BIAS_SWEEP_REPORT.md)。報告記錄 export/invariant verification PASS，但**未提供可核對的 full unit-test 數量**。 | INT8 85.12、INT7 85.06 或過窄 INT5 84.35、INT4 83.24。 |
| 不採 uniform INT8 activation，改 mixed precision | 58 節點 uniform INT8 77.07%，相對 H0 −8.06 pp，且 saturation 0；單一節點可有高敏感度。 | [H1 報告](reports/H1_INTEGER_ACTIVATION_RESIDUAL_REPORT.md)、[H1-S 報告](reports/H1S_SINGLE_NODE_REPORT.md)：Top-1 單點 TRAIN drop 1.53 pp。 | 把全網 activation/residual 一律設 INT8；將 INT4 MSE proxy 誤作單節點因果測試。 |
| 接受 H1-MP | 由 TRAIN-derived sensitivity、policy、promotion/demotion 定 58-node bits/shifts/policies，18/18 residual alignment PASS；final official TEST 85.03%，只比 H0 低 0.10 pp。 | [H1MP JSON](results_json/H1MP_SEARCH_RESULTS.json)、[H1MP 報告](reports/H1MP_MIXED_PRECISION_REPORT.md)。 | uniform precision；用 official TEST 搜尋 scale/width。 |
| GAP 不做 rounded `>>6` | 原版 H2A 84.92% 為 −0.11 pp 的歷史執行回報；H2A-v2 `q_gap=q_sum, s_gap=s_input+6` 精確等價，回到 85.03%。原版獨立 result/report **NOT FOUND**，其歸因以先前執行紀錄為限。 | 現存 [H2A-v2 JSON](results_json/H2_FINITE_WIDTH_RESULTS.json)及[報告](reports/H2_FINITE_WIDTH_REPORT.md)證實 v2 GAP error=0 / delta=0。 | 真正右移 6 位並做對稱 rounding。 |
| 保留 H2A-v2 為 safe | 所有 H1MP/H0/binary widths 不變，H2 finite overflow 0，18/18 residual align；official TEST 與 H1MP 同為 85.03%。 | [H2A-v2 JSON](results_json/H2_FINITE_WIDTH_RESULTS.json)；既有執行回報 tests 92/92。 | 讓原版 H2A 的 −0.11 pp 或後續 H2B 蓋掉 safe。 |
| H2B 只作 aggressive alternative | TRAIN validation 88.95→88.65（−0.30 pp）定最小可接受位寬；官方 TEST 84.93%（safe −0.10 pp），有 H2 finite saturation 2,078,706。 | [H2B JSON](results_json/H2B_WIDTH_OPTIMIZATION_RESULTS.json)、[報告](reports/H2B_WIDTH_OPTIMIZATION_REPORT.md)：92/92 tests、binary overflow 0。 | 宣稱 H2B 為零飽和或取代 H2A-v2。 |

## 來源缺口、看似衝突的數值及處理

| Sources / 表面不一致 | 各自數字或內容 | 較可信的結論與理由 |
|---|---|---|
| 先前 prompt vs 當前 workspace early baseline files | FP32 91.06%、minimal 72.60%、failed ReCU 43.29% 僅先前 prompt；SiMaN 數值缺；找不到正式 summary/checkpoint。 | 僅標為 *historical reported*；官方可核對起點是 ReCU 87.28%。在[manifest](08_FILE_MANIFEST.md)列 `NOT FOUND`。 |
| 舊 H2A 84.92% vs 現在根目錄 `H2_FINITE_WIDTH_RESULTS.json` 85.03% | 舊 H2A 84.92%/−0.11 pp 為先前執行回報；現在根檔的 `status=H2A_V2_PASS`，內容為 v2 GAP deferred scaling 85.03%。 | **不同版本**，非同一次 run 的數字衝突；現存 JSON/report 只證實 v2。舊版獨立原始檔 **NOT FOUND**，不能拿 v2 副本假扮舊版。 |
| H1MP report 的 `drop +0.16 pp` vs accuracy 89.34→89.18 | 報告用正數 *drop* = +0.16；按「final − baseline」記是 **−0.16 pp**。 | 同一 TRAIN validation 資料的正負號慣例差異，非結果矛盾；archive 統一 accuracy delta 為 −0.16。 |
| H2B JSON/report 的 validation `drop +0.30 pp` vs 88.95→88.65 | 正數損失 +0.30；accuracy delta **−0.30 pp**。 | 同上；不可拿 H2B TRAIN 88.65 與 TEST 85.03/84.93 直接相減。 |
| H2B official saturation `110,040,676` vs H2 finite `2,078,706` vs binary overflow 0 | 前者含 frozen H1MP 邊界；中者是縮位 H2 節點；後者只涵 binary Conv accumulator。 | 都可同時成立；依 [H2B JSON](results_json/H2B_WIDTH_OPTIMIZATION_RESULTS.json) 分範圍報告。H2B 不可稱「全域 overflow 0」。 |
| R8B params 284,250 vs registered 284,922 | 284,250 trainable；另有 672 frozen compatibility alpha。 | 實際 R8B model 重新計數，兩個總數口徑不同；以 trainable **284,250** 為本文參數主數。 |
| R8 報告 FC bias FP32 vs H0/H2 INT6 | R8B 當時 bias FP/reference；H0 後轉 INT6，FC weights 與 topology 不變。 | 實驗**階段不同**，不是互相否定；硬體數值主線採 H0/H1MP/H2A-v2 INT6。 |
| R5T T2 best 85.00% vs final epoch 84.67% | best/reload 85.00%，最後一 epoch 84.67%。 | 做跨實驗比較用 best reload，不將 final epoch 當 best。 |

除以上版本覆寫、缺檔與統計口徑差異，所比對的現存 formal report / JSON / checkpoint metadata 之主要 best/TEST 數值未發現兩份**同版同口徑**互相矛盾的結果。若未來找到舊 H2A 或 early baseline 原始檔，應追加而非改寫本歸檔的來源缺口紀錄。
