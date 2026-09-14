目前已完成：

- Official ReCU = 87.28%
- R2 Warm-start QRPReLU = 86.72%
- R3 Fused pow2 affine = 85.50%
- R4 QRPReLU + fused pow2 affine = 86.11%
- R4 best epoch = 98
- R4 reload = 86.11%

R4 已確認：
`binary residual backbone is general-multiplier-free`

本輪只做：

R5 = R4 + W1A8 Stem

目標：

把第一層 FP32 Conv3x3 3→16 改成：
- Weight = 1-bit {-1,+1}
- Activation = signed INT8
- product = +q / -q
- no general multiplier / DSP in stem convolution

其他 R4 結構全部不改。

不要做：
- final FC quantization
- head BN elimination
- stem BN elimination
- residual bit-width quantization
- KD
- SiMaN
- two-shift affine
- RTL
- FPGA synthesis
- T18/T90

請直接執行、檢查、修正、訓練與整理結果。
不要只提供指令，不要逐步詢問我。

--------------------------------------------------
PHASE 1 — 找正確 R4 checkpoint
--------------------------------------------------

自動找到成功的 R4：

- experiment: QRPReLU + fused pow2 affine
- best accuracy = 86.11%
- best epoch = 98
- reload accuracy = 86.11%

必須使用該 run 的 `best.pt`。

禁止使用：
- Official
- R1
- R2
- R3
- R4 last.pt（除非證明與 best 一樣）

先輸出：
- checkpoint path
- best_acc
- best_epoch

--------------------------------------------------
PHASE 2 — Merge / Audit R5
--------------------------------------------------

閱讀並 merge：

- `recu_hw/r5.py`
- `train_recu_r5.py`
- `tools/verify_r5_conversion.py`
- `configs/recu_r5_w1a8_stem.json`
- `tests/test_r5.py`
- `RUNME_R5.md`

這些是 extension。

不得刪除或覆寫：
- Official
- R1
- R2
- R3
- R4 experiments

先執行：

```bash
git status
```

--------------------------------------------------
PHASE 3 — R5 precision definition
--------------------------------------------------

R5 只修改 stem precision。

原 R4：

FP32 / multi-bit Conv3x3 3→16

改成：

W1A8 Stem

### Weight

保留 FP latent weight 給 optimizer。

Forward：

w_b =
+1, w >= 0
-1, w < 0

使用 STE backward。

### Activation

注意：
不能只是 binary weight + FP32 activation 然後稱 W1A8。

進入 stem 前必須明確 fake-quantize 成 signed INT8。

目前沿用既有 CIFAR-10 Normalize protocol，
接著使用固定 power-of-two symmetric INT8 quantizer：

scale = 2^-5

q = clamp(round(x / scale), -128, 127)

training reference：

x_q = q * scale

Hardware interpretation：

真正送入 W1 stem MAC datapath 的是 signed INT8 q。

scale = 2^-5 是 binary-point / shift，
不是 general multiplier；
後續可再 fold 到 stem BN。

本輪先不要移除 stem BN。

--------------------------------------------------
PHASE 4 — Hardware meaning
--------------------------------------------------

每個 stem product：

w in {-1,+1}

q in signed INT8

所以：

w*q =
+q if w=+1
-q if w=-1

硬體只能需要：

- sign select / negate
- adder tree
- accumulator

不得以一般 8x8 multiplier 描述 R5 stem。

因此：

Stem Conv general multiplier = NO
Stem Conv DSP candidate = NO

但注意：

Stem BN 本輪仍存在。

所以不能因為 Stem Conv W1A8 就直接宣稱整網 DSP=0。

--------------------------------------------------
PHASE 5 — Conversion verification
--------------------------------------------------

執行：

```bash
python tools/verify_r5_conversion.py \
  --source-r4-checkpoint "<R4_BEST>" \
  --input-scale-exp -5
```

必須確認：

1. source checkpoint 是 R4 86.11%
2. binary stem weights forward 只包含：
   -1
   +1
3. activation quantizer 是 signed INT8
4. input scale exponent = -5
5. CIFAR-10 normalized input 的 saturation fraction

要求：

saturation_fraction <= 1e-4

如果超過：
- 不得直接 training
- 測試鄰近 power-of-two scale exponent
- 優先選「最細、但不造成顯著 saturation」的 exponent
- final report 必須說明最終選擇

不要使用任意 floating scale。

scale 必須是：

2^k

以維持 hardware shift/binary-point interpretation。

同時回報：

- normalized input min/max
- observed INT8 q min/max
- saturation fraction
- stem latent weight min/max/std
- sign positive/negative counts

--------------------------------------------------
PHASE 6 — Unit tests
--------------------------------------------------

執行：

```bash
python -m unittest discover -s tests -v
```

所有 tests 必須 PASS。

特別確認：

- R5 output shape [N,10]
- input activation 落在 integer INT8 grid
- W1 stem forward weight strictly ±1
- R4 backbone/head parameters 正確複製到 R5

--------------------------------------------------
PHASE 7 — 0-Epoch Accuracy
--------------------------------------------------

從 R4 轉成 R5 後，不 fine-tune，直接測 CIFAR-10。

記錄：

R4 = 86.11%

R5 zero-epoch = ?

計算：

Δ0 = R5_zero_epoch - R4

這個數字要保留。

它代表：

FP stem → W1A8 stem

立即造成多少 loss。

--------------------------------------------------
PHASE 8 — Smoke
--------------------------------------------------

執行：

```bash
python train_recu_r5.py \
  --config configs/recu_r5_w1a8_stem.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --smoke
```

確認：

- RTX 4070 / CUDA
- forward/backward
- no NaN/Inf
- stem latent weight finite
- QRPReLU finite
- fused affine finite
- checkpoint save
- reload

--------------------------------------------------
PHASE 9 — 20-Epoch Diagnostic
--------------------------------------------------

執行：

```bash
python train_recu_r5.py \
  --config configs/recu_r5_w1a8_stem.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --diagnostic-epochs 20
```

檢查：

- accuracy 是否回升
- binary stem latent weights 是否正常更新
- positive/negative sign 是否沒有 collapse
- input saturation 是否維持正常
- no NaN/Inf

如果明顯 collapse：
- 不要直接跑正式 100 epochs
- 先診斷
- 不得自行改 backbone

--------------------------------------------------
PHASE 10 — Formal R5
--------------------------------------------------

diagnostic 正常後：

```bash
python train_recu_r5.py \
  --config configs/recu_r5_w1a8_stem.json \
  --source-r4-checkpoint "<R4_BEST>"
```

預設：

- epochs = 100
- SGD
- LR = 0.005
- momentum = 0.9
- weight decay = 5e-4
- cosine
- ReCU tau = 0.99 fixed
- input scale exponent = -5

這是 R4 warm-start fine-tune。

不得重新跑 600 epochs。

--------------------------------------------------
PHASE 11 — Accuracy 判定
--------------------------------------------------

Reference：

Official = 87.28%
R4 = 86.11%

R5：

- >=85.5%：很好
- 85.0–85.5%：可以接受
- 84.5–85.0%：仍有硬體研究價值，但 stem binarization 需要改善
- <84.5%：先不要做 final FC，優先改善 stem QAT

所有 Δ 用原始 accuracy 算後再顯示兩位小數。

--------------------------------------------------
PHASE 12 — MAC / Hardware accounting
--------------------------------------------------

原 FP stem conventional MAC：

32*32*16*3*3*3

= 442,368 MAC/image

R5 不得再把這 442,368 個 products 算成一般 multiplication。

應報告為：

442,368 W1A8 add/sub terms

其中每個：

± INT8 activation

硬體：

- negate / sign select
- accumulation

General multiplier:

0

FPGA DSP for stem convolution:

0

同時保留 binary backbone：

40,108,032 binary terms
→ XNOR/XOR + Popcount

--------------------------------------------------
PHASE 13 — Remaining multiplier sources
--------------------------------------------------

R5 如果成功：

### Stem Conv
W1A8
multiplier = NO

### Binary residual backbone
R4 structure
multiplier = NO

### GAP
sum + /64
可 shift
multiplier = NO

但是仍須明確列出：

### Stem BN
本輪未 eliminate

### Head BN
本輪未 eliminate

### Final FC
仍為 multi-bit 64→10
general multiplier candidate = YES

另外：
input Normalize 是目前 dataset/software preprocessing protocol。

不要把 software preprocessing 自動算成 accelerator 已經 multiplier-free。

如果論文要聲稱 end-to-end raw-RGB hardware multiplier-free，
後續還需要定義 raw RGB preprocessing hardware。

--------------------------------------------------
PHASE 14 — Final report
--------------------------------------------------

最後輸出：

A. Source R4
- checkpoint
- best epoch
- best/reload accuracy

B. INT8 input quantization
- scale exponent
- scale
- input min/max
- q min/max
- saturation count/fraction

C. W1 stem
- latent weight stats
- binary +1/-1 count
- 0-epoch accuracy

D. Formal R5
- best accuracy
- best epoch
- final accuracy
- reload accuracy
- Δ vs R4 86.11%
- Δ vs Official 87.28%
- training time
- trainable params

E. Hardware table
對以下逐項列：
- arithmetic
- multiplier?
- DSP?
- replacement

項目：
- Input quantization
- W1A8 stem
- Stem BN
- Binary Conv
- Fused Pow2 affine
- Residual
- Hardtanh
- QRPReLU
- GAP
- Head BN
- Final FC

F. 結論
明確回答：

1. R5 是否成功消除 stem convolution multiplier？
2. Accuracy cost 是多少？
3. 是否值得進入下一階段 final FC / BN 處理？

完成後停止。

不要自行開始：
- final FC binarization
- BN elimination
- raw RGB preprocessing hardware
- RTL
- FPGA synthesis
- T18/T90
