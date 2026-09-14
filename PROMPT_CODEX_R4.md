目前已完成：

- Official ReCU = 87.28%
- R1 Remove alpha = 87.02%
- R2 Warm-start QRPReLU = 86.72%
- R3 Fused pow2 affine = 85.50%

本輪只做：

R4 = R2 QRPReLU + fused alpha+BN signed power-of-two affine

目標：

驗證「multiplier-free binary backbone」。

不要開始 W1A8 stem、binary FC、residual quantization、RTL 或 ASIC。

請直接執行、檢查、修正、訓練並整理結果，不要只提供指令，也不要逐步詢問我。

## 1. 找正確 R2 checkpoint

自動找到 R2 Warm-start QRPReLU：

- best/reload = 86.72%
- 使用 `best.pt`

禁止使用：
- Official 87.28%
- old H1 84.28%
- old H2 84.60%
- R1
- R3

先回報：
- R2 best.pt path
- best_acc
- best_epoch

## 2. Merge / audit R4 files

閱讀：
- `recu_hw/r4.py`
- `train_recu_r4.py`
- `tools/verify_r4_conversion.py`
- `configs/recu_r4_qrprelu_fused_pow2_affine.json`
- `tests/test_r4.py`
- `RUNME_R4.md`

這些檔案是 extension，merge 到目前 workspace。
不得覆寫或刪除既有 Official/R1/R2/R3 experiment folders。

先執行：

```bash
git status
```

## 3. R4 architecture definition

R4 必須從 R2 轉換。

保留 R2：
- W1A1 binary conv
- double skip
- Hardtanh
- 已訓練的 QRPReLU:
  - a_i
  - xi1_i
  - xi2_i
- FP stem
- stem BN
- head BN
- FP FC

唯一新增變更：

每個 binary branch：

`S -> alpha -> BN`

改成：

`S -> signed pow2 fused affine`

先 fold：

K = gamma * alpha / sqrt(running_var + eps)

B = beta - gamma * running_mean / sqrt(running_var + eps)

再量化：

K_q = sign(K) * 2^round(log2(abs(K)))

硬體：
- shift
- optional negate
- add B

不得再有 general multiplier。

## 4. 先驗證 float folding

執行：

```bash
python tools/verify_r4_conversion.py \
  --source-r2-checkpoint "<R2_BEST>"
```

要求：
- 18 個 binary-conv alpha+BN pairs
- max abs float-fold error <= 1e-4

若超過：
- 停止 R4 training
- 修正 fold
- 重跑驗證

同時記錄：
- 672 個 K
- K min/max
- |K| mean/median
- positive/negative count
- exponent histogram
- mean/max absolute quantization error
- mean/max relative quantization error

## 5. Unit tests

執行：

```bash
python -m unittest discover -s tests -v
```

所有 tests 必須 PASS。

特別確認：
- R2 QRPReLU parameters 完整複製
- R4 alpha 不再 trainable
- R4 output shape 正確
- fused K power-of-two projection 正確

## 6. 0-epoch accuracy

轉換完但尚未 fine-tune 前直接 test。

記錄：

R2 = 86.72%

R4 zero-epoch = ?

計算：

Δ0 = R4_zero_epoch - R2

## 7. Smoke

```bash
python train_recu_r4.py \
  --config configs/recu_r4_qrprelu_fused_pow2_affine.json \
  --source-r2-checkpoint "<R2_BEST>" \
  --smoke
```

確認：
- CUDA
- forward/backward
- no NaN/Inf
- QRPReLU finite
- fused affine finite
- checkpoint save/reload

## 8. 20-epoch diagnostic

```bash
python train_recu_r4.py \
  --config configs/recu_r4_qrprelu_fused_pow2_affine.json \
  --source-r2-checkpoint "<R2_BEST>" \
  --diagnostic-epochs 20
```

確認：
- accuracy 有恢復
- exponent 不異常 collapse
- QRPReLU a/xi1/xi2 finite
- fused log2|K| finite
- no NaN/Inf

如果明顯 collapse：
- 不要跑正式 100 epochs
- 找原因並重跑 diagnostic

## 9. Formal 100 epochs

```bash
python train_recu_r4.py \
  --config configs/recu_r4_qrprelu_fused_pow2_affine.json \
  --source-r2-checkpoint "<R2_BEST>"
```

固定：
- 100 epochs
- SGD
- LR = 0.005
- momentum = 0.9
- weight decay = 5e-4
- cosine
- tau = 0.99

這是 warm-start fine-tuning，不得重跑 600 epoch ReCU schedule。

## 10. Accuracy interpretation

Reference：

- Official = 87.28%
- R2 = 86.72%
- R3 = 85.50%

R4：

- >= 86.0%：很好
- 85.5–86.0%：可接受
- 85.0–85.5%：有研究價值，但 fused-K quantization 仍需改善
- < 85.0%：先不要往 stem/FC 前進

使用原始未四捨五入 accuracy 算 Δ。

## 11. Hardware conclusion

R4 binary backbone：

Binary Conv:
- XNOR/XOR + Popcount
- multiplier = NO

Fused alpha+BN:
- signed shift + optional negate + bias add
- multiplier = NO

Residual:
- multi-bit ADD
- multiplier = NO

Hardtanh:
- compare/clamp
- multiplier = NO

QRPReLU:
- add + shift + add
- multiplier = NO

如果驗證成功，可稱：

`binary residual backbone is general-multiplier-free`

但是不能稱 entire network DSP-free，因為還有：

Stem:
- FP/multi-bit Conv
- multiplier = YES

Stem BN:
- 本輪未 eliminate

Head BN:
- 本輪未 eliminate

Final FC:
- multi-bit
- multiplier = YES

## 12. Final report

請輸出：

A. R2 source
- checkpoint
- best epoch
- best/reload accuracy

B. Conversion
- float-fold error
- zero-epoch R4 acc
- Δ0 vs R2
- initial K/exponent stats

C. Formal R4
- best acc
- best epoch
- final acc
- reload acc
- Δ vs Official
- Δ vs R2
- Δ vs R3
- training time
- trainable params

D. Final quantization
- exponent histogram
- K quantization errors
- 是否 exponent collapse
- QRPReLU parameter sanity

E. Hardware statement
- binary backbone 是否 general-multiplier-free
- 還剩哪些 whole-network multiplier sources

完成後停止。

不要自行開始：
- W1A8 stem
- binary FC
- two-shift coefficient
- residual quantization
- RTL
- FPGA synthesis
- T18/T90
