目前 workspace 是專案根目錄。

請先閱讀既有 `AGENTS.md`、本包 `README.md`、`RUNME.md`，以及目前 repository 的既有 FP32 / Vanilla W1A1 baseline。

這次不要只給我指令。請直接檢查、執行、修正、重跑與紀錄。
不要逐步詢問我。

# 最重要的研究規則

這一輪先做「官方 ReCU reproduction」。
官方 reproduction 沒正常以前，禁止開始 QRPReLU / power-of-two alpha / W1A8 first / binary FC。

禁止：
- widening
- ResNet-18
- ImageNet stem
- KD
- thermometer input
- SiMaN
- 自行簡化 double skip
- 自行刪除 alpha
- 自行把 PReLU 換掉
- 自行改 first/last precision

# 官方依據

請把以下官方 repository 當主要 source of truth：
https://github.com/yuanchunyu/ReCU

必須仔細核對官方：
- cifar/models_cifar/resnet.py
- cifar/modules/binarized_modules.py
- cifar/main.py
- cifar/dataset/dataset.py
- cifar/utils/options.py

不要憑記憶實作。

A&B BNN 只作後續硬體改造參考：
https://github.com/Ruichen0424/AB-BNN
https://openaccess.thecvf.com/content/CVPR2024/html/Ma_AB_BNN_AddBit-Operation-Only_Hardware-Friendly_Binary_Neural_Network_CVPR_2024_paper.html

# PHASE 1 — Repository integration

1. 先 `git status`。
2. 檢查目前 workspace 是否已經有：
   - recu implementation
   - train_recu.py
   - 同名 config
3. 不要粗暴覆蓋有效舊實驗。
4. 若本包與現有程式衝突，採最小侵入整合。
5. 保留先前：
   - FP32 91.06% baseline
   - Vanilla W1A1 72.60% baseline
   - 舊 ReCU 43.29% adaptation 結果
   舊 43.29% 必須標成 failed adaptation，不得覆寫或當 official reproduction。

# PHASE 2 — Source audit

逐項比對這包的 official ReCU implementation 與官方 GitHub：

Model block 必須是：
BConv1 -> BN1 -> shortcut add -> save x1 -> Hardtanh
-> BConv2 -> BN2 -> add x1 -> PReLU

Stem：
FP Conv3x3 3->16 -> BN -> Hardtanh

Head：
GAP -> BN1d(64) -> FP Linear 64->10

ReCU BinaryConv 必須包含：
- per-filter weight mean removal
- per-filter variance normalization
- tau-dependent clamp
- sign binary weight
- training-only activation variance normalization
- ReCU activation binary surrogate gradient
- per-output-channel alpha

如發現本包跟官方不一致：
- 修正
- 在 final report 列出差異
- 重新跑 tests

# PHASE 3 — Parameter and unit verification

執行：

`python tools/report_recu.py --config configs/recu_official.json`

Official reproduction 預期 trainable params：

- total = 270858
- alpha = 672
- PReLU = 336

然後：

`python -m unittest discover -s tests -v`

所有 test 必須通過。

另外新增/確認 tests：
- 18 個 ReCU binary conv
- tau(0) = 0.85
- tau(600) = 0.99
- forward shape [N,10]
- checkpoint save/reload
- alpha count 672
- official PReLU count 336

# PHASE 4 — Smoke

執行：

`python train_recu.py --config configs/recu_official.json --smoke`

確認：
- CUDA 正常
- forward/backward 正常
- no NaN/Inf
- checkpoint 正常
- reload 正常
- registry 正常

Windows 若 DataLoader multiprocessing 有問題：
- smoke 可用 workers=0
- 不要改模型

# PHASE 5 — 20-epoch diagnostic

執行：

`python train_recu.py --config configs/recu_official.json --diagnostic-epochs 20`

檢查：
- loss 是否下降
- train accuracy 是否上升
- test accuracy curve 是否合理
- alpha 是否 finite
- weight 是否 finite
- tau 是否依 schedule 上升
- learning rate warm-up/cosine 是否與官方 semantics 一致

若 20 epoch 已明顯 collapse：
- 停止
- 找 implementation mismatch
- 修正後重跑 diagnostic
- 不要直接燒 600 epochs

# PHASE 6 — Official 600-epoch training

只有 diagnostic 正常時才執行：

`python train_recu.py --config configs/recu_official.json`

官方公開 target：
- ResNet20-1W1A vanilla 約 87.5%
- finetune 約 88.0%

不要求完全一樣，但：
- >=85%：可視為 reproduction 基本合理，進入下一階段
- <85%：先做 reproduction diagnosis，不准直接做硬體 QAT

請記錄：
- best test accuracy
- best epoch
- final epoch accuracy
- total training time
- peak GPU memory（容易取得則記）
- checkpoint path
- config
- git commit
- parameter count

注意：
官方 ReCU code 直接使用 CIFAR-10 test set 做 epoch evaluation。
為了 reproduction 必須先照做。
後續 thesis final protocol 再建立 train/val/test 嚴格版本，兩者不要混淆。

# PHASE 7 — Hardware-aware branch（只有 Official ReCU >=85% 才能做）

先不要一次全部改。

## Experiment H1 — QRPReLU only

從 official best checkpoint / recipe 出發，建立：

`configs/recu_qrprelu.json`

只改：
PReLU -> A&B-style Quantized RPReLU

負半軸：
2^round(a_i) * (x + xi1_i) + xi2_i

要求：
- a_i, xi1_i, xi2_i learnable
- round 用 STE
- inference slope 只能是 power-of-two
- hardware mapping 註明：shift + add，不是 general multiplier

先 smoke，再正式訓練 / fine-tune。
不要同時改 alpha。

## Experiment H2 — QRPReLU + power-of-two alpha

H1 完成後才做：

`configs/recu_qrprelu_pow2alpha.json`

只額外把每 channel alpha 量化成 signed power-of-two magnitude。

要求：
- QAT
- 記錄 accuracy drop
- 匯出每層 shift exponent distribution
- 不得只做 post-training rounding 後直接宣稱有效

# PHASE 8 — Stop

這一輪不要做：
- W1A8 first layer
- binary-weight FC
- residual bit-width quantization
- ASIC RTL
- XOR vs XNOR
- SiMaN
- KD

這些等 QRPReLU / alpha ablation 有結果再做。

# FINAL REPORT

最後請回報：

A. Official ReCU reproduction
- source audit 差異
- params
- test accuracy
- best epoch
- training time
- checkpoint
- 是否 >=85%
- 是否接近官方 87.5%

B. QRPReLU（只有達門檻才執行）
- params
- accuracy
- relative drop/gain
- power-of-two slope exponent統計

C. QRPReLU + pow2 alpha（只有前一步正常才執行）
- params
- accuracy
- relative drop/gain
- alpha exponent統計

D. Hardware interpretation
對三個版本分別列：
- binary conv terms
- general multiplier candidates
- 哪些可用 shift
- 哪些只需 add
- 預期是否可移除 PReLU/alpha DSP

不要把 training-time PyTorch multiplication 誤稱為 inference hardware multiplier。
