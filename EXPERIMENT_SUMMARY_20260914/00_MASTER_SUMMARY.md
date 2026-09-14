# 研究進度總結（截至 2026-09-14）

## 研究目標與證據邊界

CIFAR-10 / ResNet20 分類。從 official ReCU W1A1 reproduction 出發，逐步移除推論所需的通用乘法，然後固定 bias、activation/residual 和內部 accumulator 的整數表示，形成 RTL 前的 software numeric specification。所有以下準確率若未標註，均指 **CIFAR-10 official TEST**；百分點差異記作 pp。模型 checkpoint 沒有複製進歸檔，僅核對 metadata。[完整比較](03_ACCURACY_RESULTS.md)與[原始報告](reports/)可追溯。

## 正式架構與計算量

`32×32×3 RGB → Thermometer R=8（96 個 bipolar channels）→ W1A1 3×3 stem 96→16 → 3+3+3 ReCU double-skip blocks → GAP → signed-pow2 head affine → signed-pow2 FC 64→10`。Stem/Backbone 的 convolution 為 XNOR+popcount 語意，affine/FC K 為 shift、optional negate 與 integer bias add。R8B 實際 checkpoint 為 **284,250 trainable + 672 frozen alpha = 284,922 registered parameters**；binary Conv weights **281,088**，FC weight/bias **640+10**。Stem **14,155,776**、backbone **40,108,032**、合計 **54,263,808 binary-conv terms/image**，外加 **640 FC signed-pow2 shifted terms**；54,264,448 只能稱混合算術 sites，**不能稱一般 MAC 或 INT8 MAC**。[架構與拆解](02_MODEL_ARCHITECTURE.md)

## 模型進程

| 關鍵節點 | Best / TEST | 關鍵結論 |
|---|---:|---|
| Official ReCU | 87.28% | 600-epoch 可重現參照；二值 backbone，但 stem/head/FC 仍有 FP 路徑。 |
| R4 | 86.11% | R2 QRPReLU → 18 個 fused signed-pow2 affine，backbone 無通用乘法。 |
| R5 / R5AB | 83.77% / 83.96% | 3-channel W1A8 stem 不足，未選。 |
| R5T / R5T-Long T2 | 85.00% / 85.14% | Thermometer R=8 + W1A1 stem；長訓練僅 +0.14 pp。 |
| R6 / R7 | 85.52% / 85.22% | 分別移除 stem BN、head BN；保留 signed-pow2 affine。 |
| R8A / **R8B** | 82.84% / **85.40%** | Signed-pow2 FC 比 W1 FC 高 2.56 pp，故選 R8B。 |

## 量化與 finite-width 進程

| 節點 | Official TEST | 規格 / 意義 |
|---|---:|---|
| H0 INT6 | 85.13% | 21 個 bias/offset tensors、762 scalar values，saturation 0；相對 R8B −0.27 pp。 |
| H1 uniform INT8 | 77.07% | 58 個節點全部等寬不可行；不是單純 overflow 問題。 |
| H1-S | — | TRAIN-derived 58/58 single-node 分析及四種 scale policy；不報 final TEST。 |
| H1-MP | 85.03% | 58-node frozen bits/shifts/policies；平均 6.6897 bit、volume weighted 6.6240 bit；18/18 residual alignment。 |
| H2A 原版 | *84.92% 歷史回報* | Rounded `GAP >>6` 造成 −0.11 pp；原版獨立 result/report **NOT FOUND**。 |
| **H2A-v2 Safe** | **85.03%** | `q_gap=q_sum`, `s_gap=s_input+6`；GAP exact error 0，H2 overflow 0，與 H1-MP 無 loss。 |
| **H2B Aggressive** | **84.93%** | TRAIN validation 決定更小內部位寬；相對 safe −0.10 pp，保留為替代規格。 |

H2A-v2 的保守寬度：binary accum stem/Cin16/Cin32/Cin64 = **INT11/9/10/11**，GAP sum **INT16**，FC accumulator/logits **INT24/24**，QRPReLU max inner/output **INT30/38**。H2B 保留 binary accum、H0、H1MP、GAP deferred scaling，縮為 GAP **INT13**、FC/logits **INT16/16**、QRPReLU max inner/output **INT27/35**。H2B official TEST 的 H2 finite saturation 為 **2,078,706**，binary accumulator overflow **0**。[完整規格](05_CURRENT_HARDWARE_SPEC.md)

## 結論與剩餘工作

**選定模型：R8B。選定 safe numeric mainline：R8B-H2A-v2（85.03%）。H2B（84.93%）作 aggressive alternative，不覆蓋 safe。** 到目前為止可主張軟體層面的 finite-width integer 表示與演算法層面無通用乘法器；實際 DSP、timing、FPS、ASIC PPA 仍待 RTL/綜合量測。下一階段按順序為 RTL specification、SystemVerilog、functional verification、FPGA synthesis、ASIC T18/T90 PPA；本歸檔工作沒有執行其中任何一項。[限制與下一步](07_CURRENT_STATUS_AND_NEXT_STEP.md)
