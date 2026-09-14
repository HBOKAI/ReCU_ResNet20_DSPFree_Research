# 當前完成範圍與下一正式階段

截至 2026-09-14，已完成的**軟體模型**主線是 **R8B signed-pow2 FC，official CIFAR-10 TEST 85.40%**；已完成的**硬體數值化**主線是 **R8B-H2A-v2 Safe，85.03%**，並同時保留 **R8B-H2B Aggressive，84.93%**。H2B 是依 TRAIN validation 凍結的可選最小位寬 profile，不取代零新增準確率損失的 H2A-v2。已有 [總結](00_MASTER_SUMMARY.md)、[架構](02_MODEL_ARCHITECTURE.md)、[完整比較](03_ACCURACY_RESULTS.md)、[兩套數值規格](05_CURRENT_HARDWARE_SPEC.md)及[SHA256 manifest](08_FILE_MANIFEST.md)。

目前可以合理宣稱：

- Binary Conv 使用 W1A1 XNOR/popcount semantics；54,263,808 binary-conv terms/image。
- Signed-pow2 stem/backbone/head affine K 及 FC effective weights，以 integer shifts、optional negate 和加法實現，演算法上無需 general multiplier。
- H0 affine B/FC bias 為 pure integer INT6；H1MP activation/residual 有 frozen 58-node finite integer bits/shifts/policies，18/18 residual alignment 已驗證。
- H2A-v2 提供 explicit finite-width accumulator/intermediate 的軟體數值契約；GAP `/64` 是**scale metadata +6** 而非實際 `>>6` rounding，bit-accurate GAP equality max error 0；official TEST 85.03%、H2 overflow 0。
- H2B 有 TRAIN-derived minimum-width alternative；official TEST 84.93%，narrowed H2 finite saturation 2,078,706，binary accumulator overflow 0。以上均是**software/numeric** 結論。

目前**不能**宣稱 FPGA DSP48=0、ASIC multiplier cells=0、timing met、final FPGA FPS、FPGA LUT/BRAM/FF 用量、ASIC T18/T90 面積/功耗/PPA。Binary workload 不是 MAC 指標，也不能直接換算硬體吞吐；cache、記憶體、adder tree、控制與 routing 仍須在實際設計中驗證。歷史 early baseline 及 H2A 原版獨立 artifact 缺失，限制已記在[決策與來源紀錄](06_ABLATION_AND_DECISION_LOG.md)。

下一正式階段的依賴順序：**RTL Specification → SystemVerilog implementation → functional verification（與 frozen H2A-v2/H2B software reference 對照）→ FPGA synthesis → ASIC T18/T90 synthesis/PPA**。應先選擇 safe 為主規格、aggressive 為比較規格，並維持 H0/H1MP/H2 plans 的 frozen provenance。本次工作**僅搜尋、核對、整理及複製小型文件**；沒有開始 RTL、Vivado、FPGA/ASIC synthesis、訓練、QAT 或新量化實驗，也沒有刪除/改名原始實驗資料夾。
