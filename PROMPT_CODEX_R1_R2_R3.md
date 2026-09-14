目前 workspace 已經完成：

- FP32 ResNet20 test = 91.06%
- Vanilla W1A1 test = 72.60%
- Official ReCU reproduction best/reload = 87.28%
- H1 QRPReLU from-scratch best = 84.28%
- H2 QRPReLU + pow2-alpha from-scratch best = 84.60%

舊 ReCU 43.29% 是 failed adaptation，不列入公平比較。

現在不要再從頭訓練 600 epochs。
這一輪要從已成功的 Official ReCU 87.28% `best.pt` 做 warm-start hardware ablation。

請直接執行、檢查、修正、訓練與整理結果，不要只給我指令，也不要逐步詢問我。

# 核心目標

只做三個實驗：

R1:
Remove alpha

R2:
Official PReLU -> warm-start A&B-style Quantized RPReLU

R3:
Fold ReCU alpha + BN into deployment affine K*S+B,
then quantize K to signed power-of-two.

禁止這一輪做：
- W1A8 first conv
- binary FC
- residual quantization
- SiMaN
- KD
- widening
- thermometer input
- XOR/XNOR RTL
- ASIC synthesis

--------------------------------------------------
PHASE 0 — Find Official checkpoint
--------------------------------------------------

自動在目前 experiments 下找到成功 Official ReCU：

best accuracy = 87.28%
best epoch = 600
checkpoint reload = 87.28%

使用它的 `best.pt`。

不要誤用：
- H1
- H2
- 舊 43.29% ReCU adaptation

先印出：
- checkpoint path
- stored best_acc
- stored best_epoch

--------------------------------------------------
PHASE 1 — Read and audit new code
--------------------------------------------------

請閱讀：

- `recu_hw/ablation.py`
- `recu_hw/fused_affine.py`
- `train_recu_ablation.py`
- `tools/verify_ablation_conversion.py`
- `RUNME_R1_R2_R3.md`

並閱讀原本：
- `recu_hw/model.py`
- `recu_hw/layers.py`
- `recu_hw/qrprelu.py`

先 `git status`。
不得破壞 Official ReCU / H1 / H2 舊結果。

--------------------------------------------------
PHASE 2 — Unit tests
--------------------------------------------------

執行：

```bash
python -m unittest discover -s tests -v
```

必須全部 PASS。

特別確認：

1. R1：
   - alpha inference function 等於 1
   - alpha 不再 trainable

2. R2：
   Official PReLU slope p_i 必須 warm-start 到：

   a_i = log2(p_i)

   xi1 = 0
   xi2 = 0

   因此 QRPReLU 初始 quantized slope：

   2^round(log2(p_i))

   不准全部重新初始化為固定 -2。

3. R3：
   官方 binary branch：

   BN(alpha*S)

   必須先 fold 成：

   K*S+B

   K = gamma*alpha/sqrt(running_var+eps)

   B = beta - gamma*running_mean/sqrt(running_var+eps)

--------------------------------------------------
PHASE 3 — Critical R3 folding verification
--------------------------------------------------

執行：

```bash
python tools/verify_ablation_conversion.py --checkpoint "<OFFICIAL_BEST>"
```

這一步非常重要。

先建立 `float folded affine`，不要量化 K。

比較：

Official ReCU eval logits

vs

Float-folded ReCU eval logits

要求：

max abs logit error <= 1e-4

最好接近 numerical precision。

如果超過：
- 禁止開始 R3 training
- 找 folding / running-stat / alpha / BN epsilon 問題
- 修正後重跑

只有 float folding 等價成立後，
才允許：

K -> sign(K) * 2^round(log2|K|)

--------------------------------------------------
PHASE 4 — R1 Remove Alpha
--------------------------------------------------

R1 的唯一 architecture/function change：

原：

BinaryConv output = alpha_c * S

改：

BinaryConv output = S

也就是 alpha=1。

其餘全部維持 Official：
- BN
- double skip
- Hardtanh
- PReLU
- FP stem
- head BN
- FP FC

從 Official best checkpoint warm-start。

先取得「轉換後、尚未 fine-tune」accuracy。
這個數字必須記錄。

然後 smoke：

```bash
python train_recu_ablation.py \
  --config configs/recu_r1_remove_alpha.json \
  --source-checkpoint "<OFFICIAL_BEST>" \
  --smoke
```

再 20 epoch diagnostic。

若正常，正式 fine-tune 100 epochs。

預設：
- SGD
- LR 0.01
- momentum 0.9
- WD 5e-4
- cosine
- tau 固定 0.99

注意：
這是已收斂模型的 fine-tune。
不要把 tau 重設為 0.85。

記錄：

- initial converted acc
- best acc
- best epoch
- final acc
- reload acc
- Δ vs Official 87.28%
- trainable params
- training time

--------------------------------------------------
PHASE 5 — R2 Warm-start QRPReLU
--------------------------------------------------

這個實驗是要回答：

之前 H1 從頭訓練掉到 84.28%，
是否主要是因為 QRPReLU initialization / optimization 不佳？

唯一修改：

Official PReLU -> QRPReLU

不要改 alpha。

每個 channel：

Official:
PReLU negative slope = p_i

QRPReLU initialization：

a_i = log2(max(p_i, epsilon))
xi1_i = 0
xi2_i = 0

Quantized negative slope：

2^round(a_i)

先統計 Official 336 個 PReLU slopes：

- min
- max
- mean
- median
- <=0 數量
- 投影後 exponent histogram
- 投影量化誤差

如果存在 p_i <= 0：
- 不要靜默忽略
- 列出數量
- QRPReLU 無法直接表示負 slope，因此用小正值 clamp 僅作初始化
- final report 必須說明

先取得轉換後 0-epoch accuracy。

再：
- smoke
- 20 epoch diagnostic
- 100 epoch fine-tune

最後比較：

Official = 87.28%
old H1 from-scratch = 84.28%
new R2 warm-start = ?

我要特別知道：

new R2 - old H1

如果能回升，證明 warm-start projection 有效。

--------------------------------------------------
PHASE 6 — R3 Fused alpha+BN Pow2 Affine
--------------------------------------------------

這次不要再量化 alpha 本身。

原始 binary branch：

S
-> * alpha_c
-> BatchNorm

deployment fold：

y = K_c*S + B_c

其中：

K_c =
gamma_c * alpha_c /
sqrt(running_var_c + eps)

B_c =
beta_c -
gamma_c * running_mean_c /
sqrt(running_var_c + eps)

然後只量化：

K_c -> sign(K_c) * 2^k_c

k_c = round(log2(abs(K_c)))

硬體對應：

S
-> shift |k|
-> optional sign negate
-> + B

不需要 general multiplier。

要求：

- sign(K) 保留
- exponent QAT 用 STE
- bias B 可 fine-tune
- exponent 有合理 clamp
- 每 channel exponent 可 export
- binary branch 原 BN 被 fused affine 取代
- 不得同時換 QRPReLU
- PReLU 保持 Official

也就是 R3 只測：
`alpha + BN multiplier -> signed shift + add`

先記錄：

1. Float fold equivalence error
2. Pow2 quantization 前 K distribution
3. 初始 exponent histogram
4. K quantization relative/absolute error
5. 0-epoch converted accuracy

再 smoke / diagnostic / 100 epoch fine-tune。

最後輸出 final exponent histogram。

--------------------------------------------------
PHASE 7 — Compare
--------------------------------------------------

整理：

| Model | Best Acc | Δ vs ReCU | Main change |
|---|---:|---:|---|
| Official ReCU | 87.28 | 0 | reference |
| Old H1 QRPReLU | 84.28 | -3.00 | from scratch |
| Old H2 QRPReLU+pow2 alpha | 84.60 | -2.68 | from scratch |
| R1 no-alpha | ? | ? | remove alpha |
| R2 warm QRPReLU | ? | ? | shift/add activation |
| R3 fused pow2 affine | ? | ? | shift/add alpha+BN |

--------------------------------------------------
PHASE 8 — Hardware interpretation
--------------------------------------------------

請精確區分：

1. Binary Conv
   XNOR/XOR + Popcount
   general multiplier = NO

2. Residual
   multi-bit ADD
   general multiplier = NO
   但有 buffer/adder cost

3. R1 alpha
   removed
   multiplier = NO

4. R2 QRPReLU
   2^k slope
   shift + add
   general multiplier = NO

5. R3 fused affine
   signed 2^k scale + B
   shift + negate + add
   general multiplier = NO

6. Stem
   目前仍是 FP/multi-bit Conv
   multiplier/DSP candidate = YES

7. Final FC
   目前仍是 FP/multi-bit
   multiplier/DSP candidate = YES

8. Stem BN / Head BN
   這一輪尚未一起消除
   請單獨標示。

因此即使 R1/R2/R3 成功，
這一輪也不能宣稱 entire network DSP=0。

--------------------------------------------------
PHASE 9 — Stop
--------------------------------------------------

三個實驗完成後停止。

不要自行開始：
- W1A8 stem
- binary-weight FC
- complete DSP-free network
- RTL
- ASIC

我要先看 R1/R2/R3 accuracy 再決定。

--------------------------------------------------
FINAL REPORT
--------------------------------------------------

最後請輸出：

A. Official source checkpoint
- path
- 87.28% 驗證

B. R1
- zero-epoch acc
- best/final/reload
- Δ Acc
- alpha 是否可刪

C. R2
- Official PReLU slope distribution
- projected exponent distribution
- nonpositive slopes
- zero-epoch acc
- best/final/reload
- Δ vs Official
- Δ vs old H1 84.28%

D. R3
- float folding max error
- K distribution
- exponent histogram before/after
- zero-epoch acc
- best/final/reload
- Δ Acc

E. Hardware table
逐模組標：
- arithmetic
- bit width（能合理推導就列）
- multiplier?
- DSP candidate?
- replacement
- current status

不要用「理論上應該」代替實際訓練結果。
