# ReCU ResNet20 DSP-free research — experiment archive (2026-09-14)

本專案以 CIFAR-10 的 ReCU ResNet20 為起點，逐步把推論資料路徑改成二值卷積、signed-power-of-two 係數與明確位寬的整數運算；目標是**演算法層面不需要通用乘法器**，並為後續 RTL 提供可核對的數值規格。這份資料夾只歸檔已完成實驗，不含新訓練或硬體實作。

目前選定的軟體模型是 **R8B**：Thermometer R=8、W1A1 stem/backbone、signed-pow2 affine/head/FC。原始 R8B 的 CIFAR-10 official TEST 為 **85.40%**；量化後保留兩個**互不取代**的數值 profile：

| Profile | Official TEST | 用途 | 關鍵差異 |
|---|---:|---|---|
| **H2A-v2 Safe** | **85.03%** | 零額外準確率損失的主要數值規格 | H0 INT6 bias、H1MP 58-node mixed precision、保守 finite-width；GAP 平均由 scale metadata 處理 |
| **H2B Aggressive** | **84.93%** | 最小位寬替代方案 | 同一模型/量化計畫，縮減 QRPReLU、FC、logits、GAP 內部位寬；TEST 相對 safe −0.10 pp |

模型有 **284,250 trainable parameters**（另有 672 個 frozen compatibility alpha），每張影像 **54,263,808 個 binary-convolution terms**，分類器另有 **640 個 signed-pow2 shift terms**。這些不是一般 INT8 MAC，亦不是量測過的 FPGA DSP 數。H2A-v2 safe profile 的 H2 overflow 為 0；H2B 的 finite-width saturation 為 2,078,706，不能誤稱零 overflow。

主要取捨：R5/R5AB 的 3-channel W1A8 stem 準確率不足；R5T 的 96-channel Thermometer binary stem 改善，R5T-Long 為 R6 提供較佳起點；R8B signed-pow2 FC 比 R8A binary FC 高 **2.56 pp**；uniform INT8 activation 只有 **77.07%**，因此改用 H1-S 敏感度分析與 H1-MP mixed precision；H2A 原版的 GAP `>>6` 引入 rounding，H2A-v2 改用 deferred scaling 回到 **85.03%**。

閱讀路徑：

1. [00_MASTER_SUMMARY.md](00_MASTER_SUMMARY.md) — 一頁研究摘要。
2. [01_EXPERIMENT_TIMELINE.md](01_EXPERIMENT_TIMELINE.md) — 完整實驗順序、source checkpoint 與變因。
3. [02_MODEL_ARCHITECTURE.md](02_MODEL_ARCHITECTURE.md) — 現行網路、參數與 binary workload。
4. [03_ACCURACY_RESULTS.md](03_ACCURACY_RESULTS.md) — official TEST、TRAIN validation、Top-10 敏感節點及比較表。
5. [04_HARDWARE_NUMERIC_EVOLUTION.md](04_HARDWARE_NUMERIC_EVOLUTION.md) — R8B 至 H2B 的數值化演進。
6. [05_CURRENT_HARDWARE_SPEC.md](05_CURRENT_HARDWARE_SPEC.md) — safe/aggressive 位寬及尚未驗證的硬體聲明。
7. [06_ABLATION_AND_DECISION_LOG.md](06_ABLATION_AND_DECISION_LOG.md) — 每次決策與來源缺口/口徑差異。
8. [07_CURRENT_STATUS_AND_NEXT_STEP.md](07_CURRENT_STATUS_AND_NEXT_STEP.md) — 當前進度與下一正式階段。
9. [08_FILE_MANIFEST.md](08_FILE_MANIFEST.md) — 拷貝來源、UTC 修改時間、大小及 SHA256。

`tables/` 包含完整 [experiment_results.csv](tables/experiment_results.csv)、[experiment_results.json](tables/experiment_results.json)、[hardware_widths.csv](tables/hardware_widths.csv) 和 [accuracy_progression.csv](tables/accuracy_progression.csv)；`reports/`、`results_json/`、`configs/` 是保留原內容的輕量副本。數據優先順序：final report → result JSON → checkpoint metadata → log；TRAIN validation 不可視為 official TEST。早期 baseline 及 H2A 原版缺獨立現存結果檔，已標記 `NOT FOUND`。

下一正式階段是 RTL specification → SystemVerilog → 功能驗證 → FPGA synthesis → ASIC T18/T90 PPA；**本次未開始**。在 synthesis 之前，不宣稱實體 FPGA DSP48=0、ASIC multiplier cells=0、timing、FPS 或 area/power 達標。
