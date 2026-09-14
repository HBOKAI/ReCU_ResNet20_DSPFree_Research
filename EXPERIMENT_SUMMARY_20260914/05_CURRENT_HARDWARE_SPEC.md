# 當前硬體數值規格：SAFE 與 AGGRESSIVE 並存

此為 R8B 模型的 **software finite-width inference contract**，不是經 RTL 或綜合證實的 FPGA/ASIC 實作。共同點：Thermometer R=8，96 個 bipolar inputs；stem/backbone binary Conv 使用 XNOR+popcount 語意；affine/head/FC 的有效 K 是 signed powers of two（shift、optional negate），H0 21 個 B/FC bias tensors **INT6**（762 values）；H1-MP 58-node bits/shifts/policies 全部 frozen；18/18 residual scale alignment 已驗證。參數/運算量見[模型架構](02_MODEL_ARCHITECTURE.md)，機器可讀的逐節點明細見[hardware_widths.csv](tables/hardware_widths.csv)。

| 數值欄位 | **Profile A — H2A-v2 Safe** | **Profile B — H2B Aggressive** |
|---|---:|---:|
| Official CIFAR-10 TEST | **85.03%** | **84.93%** |
| Δ vs H1-MP TEST 85.03 | **0.00 pp** | **−0.10 pp** |
| Bias / offset | INT6 | INT6；與 safe 完全相同 |
| H1MP activation/residual | 58-node frozen mixed plan | 同一 frozen plan |
| H1MP mean / volume-weighted bits | 6.6897 / 6.6240 | 同左 |
| Stem binary accumulator | **INT11** | **INT11**；exact、不可縮 |
| Backbone Cin16 accumulator | **INT9** | **INT9**；exact、不可縮 |
| Backbone Cin32 accumulator | **INT10** | **INT10**；exact、不可縮 |
| Backbone Cin64 accumulator | **INT11** | **INT11**；exact、不可縮 |
| GAP sum accumulator | **INT16** | **INT13** |
| GAP division by 64 | `q_gap=q_sum`, `s_gap=s_input+6` | **同左**，不得重新導入 `>>6` rounding |
| FC signed-pow2 accumulator | **INT24** | **INT16**（可飽和） |
| FC INT6 bias-add intermediate | **INT24** | **INT24**（H2B 不縮此項） |
| Final logits | **INT24** | **INT16**（可飽和） |
| QRPReLU max inner | **INT30** | **INT27** |
| QRPReLU output internal | **INT38** | **INT35** |
| H2 finite-width saturation / overflow, official TEST | **0** | **2,078,706**（H2 narrowed nodes） |
| Binary Conv accumulator overflow | **0** | **0** |
| Residual alignment | **18/18 PASS** | **18/18 PASS** |
| Full tests（既有實驗） | **92/92 PASS**（既有執行回報） | **92/92 PASS**（H2B JSON 記錄） |
| 選用狀態 | **主要安全規格** | **較小位寬備選；不取代 safe** |

H1MP frozen bits 的分布為 **INT6×37、INT7×14、INT8×3、INT10×2、INT12×2**。`hardware_widths.csv` 包含兩套 profile 的每個 activation/residual node 的精確 bits、scale shift 和 policy，以及 binary accumulator、9 個 QRPReLU internal 欄位、GAP/FC/bias。原始 frozen `bits_by_node/shift_by_node/policy_by_node` 亦見 [H1MP JSON](results_json/H1MP_SEARCH_RESULTS.json)；H2A-v2/H2B 完整 plan 見 [H2A-v2 JSON](results_json/H2_FINITE_WIDTH_RESULTS.json)與[H2B JSON](results_json/H2B_WIDTH_OPTIMIZATION_RESULTS.json)。

## 精確算術邊界與 saturation 口徑

H2A-v2 的 GAP sum 以 INT16 保留原 `q_sum`，輸出 scale 從 `s_input=6` 改為 `s_gap=12`；不做真正的整數除法、右移取整或額外 rounding。H2B 只把 sum accumulator 縮至 INT13，仍沿用相同 scale 語義。FC 以 signed-pow2 權重的 per-term shift 對齊 common scale 後累加，bias 是 H0 INT6 的 integer add，沒有 general multiplier。

H2A-v2 報告的 `saturation_total including frozen H1MP boundaries` 為 **107,974,934**；這不是 H2 overflow，H2 的新增 finite overflow 為 **0**。H2B official TEST 的總報告 saturation **110,040,676**，其中 narrowed H2 finite nodes **2,078,706**；因此絕不應寫「H2B zero overflow」。其 TEST accuracy 減 0.10 pp 是 TRAIN-only width selection 後的獨立 final 評估。H2A-v2 GAP exact verification max error **0**。[H2A-v2 原始報告](reports/H2_FINITE_WIDTH_REPORT.md)；[H2B 原始報告](reports/H2B_WIDTH_OPTIMIZATION_REPORT.md)

## 可主張與不可主張

可主張：在已驗證的軟體路徑中，binary Conv 為 XNOR/popcount 語意；signed-pow2 K/FC 只需 shift/optional negate 與加法；bias 是 pure integer；activation/residual 有 finite integer 表示；H2A-v2 定義 explicit finite-width intermediate/accumulator 及 bit-accurate GAP deferred scaling；整體**演算法層面**不要求通用乘法器。不可主張：已知 FPGA DSP48 使用量為 0、ASIC multiplier cells 為 0、timing closure、FPS、FPGA 資源數、ASIC area/power。這些都要後續 SystemVerilog、功能驗證及 FPGA/ASIC synthesis 才能判定。
