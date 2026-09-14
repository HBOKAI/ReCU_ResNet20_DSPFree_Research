目前結果：

- Official ReCU = 87.28%
- R4 = 86.11%
- R5 naive W1A8 = 83.77%
- R5AB progressive scaled W1A8 = 83.96%
- R5AB 的 A8+FP stem (lambda=0) = 85.80%
- R5AB 的 scaled-W1A8 zero-epoch = 32.06%
- INT8 saturation = 0

這些結果顯示：
A8 input 本身只造成約 0.31 pp 損失，但 3-channel stem weight binarization 造成主要 accuracy loss。

本輪改採附件 FracBNN 論文的 thermometer binary input layer 思路。

本輪只做：

R5T = R4 + Thermometer Encoding (R=8) + Binary 3x3 Stem

不要使用 W2。
不要沿用 R5 的 A8 stem。
不要開始 final FC、BN elimination、RTL、FPGA synthesis 或 T18/T90。

==================================================
1. Source checkpoint
==================================================

自動找到 R4 best.pt：
- best/reload = 86.11%
- best epoch = 98

必須從 R4 warm-start。

禁止直接從 R5/R5AB warm-start。

先回報：
- path
- best_acc
- best_epoch

==================================================
2. 先核對 FracBNN thermometer 定義
==================================================

附件 FracBNN Section 3.2 的 thermometer input layer：

- pixel p in [0,255]
- resolution R
- vector length L = ceil(255/R)
- n = round(p/R)
- thermometer vector 有 n 個 1，其餘為 0
- 再把 0 -> -1、1 -> +1

論文實驗使用：

R = 8

因此：

L = 32 per RGB channel
3 RGB channels -> 96 bipolar binary channels

重要：
Thermometer 必須基於 raw pixel intensity。
不要先套 CIFAR Normalize 再 thermometer encode。

Data augmentation 可以保留：
- random crop
- horizontal flip

但模型輸入在 ToTensor 後仍保持 [0,1]，由 thermometer encoder 轉回 p=round(255*x)。

==================================================
3. Merge / audit files
==================================================

閱讀並 merge：

- recu_hw/r5t.py
- recu_hw/r5t_data.py
- train_recu_r5t.py
- tools/verify_r5t.py
- configs/recu_r5t_thermometer_r8.json
- tests/test_r5t.py
- RUNME_R5T.md

先執行：

git status

不得覆蓋既有 R1~R5AB results。

==================================================
4. 檢查既有 CIFAR normalization constants
==================================================

R5T pack 裡的 R4->thermometer projected initialization 使用 CIFAR std reference：

[0.2470, 0.2435, 0.2616]

先閱讀既有 recu_hw/data.py，確認 R4/Official ReCU 實際使用的 CIFAR-10 Normalize mean/std。

如果現有專案數值不同：
- 將 config/reference 改成專案實際 std
- 記錄修改

不要擅自改 dataset protocol 其他部分。

這個 projected initialization 是本專案的 engineering warm-start，不是 FracBNN 論文原方法。

==================================================
5. Verification
==================================================

執行：

python tools/verify_r5t.py \
  --source-r4-checkpoint "<R4_BEST>" \
  --resolution 8

必須確認：

- R=8
- L=32
- input channels=96
- thermometer output strictly bipolar {-1,+1}
- binary stem effective weight strictly {-1,+1}
- FracBNN Fig. 6 的示意：R=32, pixel 109 -> 3 ones

另外報告：

- new binary stem weight count = 96*16*3*3 = 13,824 bits
- original R4 FP32 stem storage = 3*16*3*3*32 = 13,824 bits

兩者 storage bits 應相同。

==================================================
6. Unit tests
==================================================

執行：

python -m unittest discover -s tests -v

所有 tests 必須 PASS。

==================================================
7. 兩階段 training
==================================================

FracBNN 論文 training 思路是：
先 binary activation + FP weight，再 binary weight。
本專案採用縮短但保持相同方向的兩階段流程。

-------------------------
Stage T1
-------------------------

Input activation：
- thermometer bipolar {-1,+1}

Stem weight：
- FP latent / FP forward

Stem：
- Conv3x3 96->16

初始化：
- 從 R4 3->16 FP stem 投影成 96->16 thermometer stem
- R4 backbone/head 全部 warm-start

預設：
- 100 epochs
- batch 128
- SGD
- LR 1e-3
- momentum 0.9
- weight decay 1e-5
- linear decay to zero
- tau = 0.99

目的：
讓 network 先適應 thermometer input representation。

記錄：
- zero-epoch projected FP thermometer accuracy
- T1 best/final/reload

-------------------------
Stage T2
-------------------------

必須從 T1 best checkpoint warm-start。

Input activation：
- thermometer bipolar {-1,+1}

Stem weight：
- sign(W), STE

即真正：

W1A1 stem

預設：
- 100 epochs
- batch 128
- SGD
- LR 1e-3
- momentum 0.9
- weight decay 0
- linear decay to zero
- tau = 0.99

記錄：
- T1 best -> T2 binary zero-epoch accuracy
- T2 best/final/reload
- +1/-1 counts
- no sign collapse

==================================================
8. Smoke first
==================================================

先執行：

python train_recu_r5t.py \
  --config configs/recu_r5t_thermometer_r8.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --stage both \
  --smoke

確認：
- CUDA
- T1 forward/backward
- T2 forward/backward
- checkpoint save/reload
- no NaN/Inf
- raw input沒有 Normalize
- thermometer values only +/-1

==================================================
9. Formal T1 + T2
==================================================

Smoke 正常後執行：

python train_recu_r5t.py \
  --config configs/recu_r5t_thermometer_r8.json \
  --source-r4-checkpoint "<R4_BEST>" \
  --stage both

完成後 T2 best accuracy 才是 R5T 正式結果。

==================================================
10. Accuracy interpretation
==================================================

Reference：
- Official = 87.28%
- R4 = 86.11%
- R5 = 83.77%
- R5AB = 83.96%

R5T T2：

>=85.5%：成功，thermometer 值得作為 main stem
85.0~85.5%：可接受，值得做硬體/進一步 R ablation
84.5~85.0%：部分成功
<=83.96%：沒有足夠收益

計算：
- Δ vs R4
- Δ vs R5
- Δ vs R5AB
- Δ vs Official

==================================================
11. Hardware accounting
==================================================

R=8：
- 3 RGB -> 96 binary channels
- stem binary weight shape = 16 x 96 x 3 x 3
- BMAC/image = 32*32*16*96*3*3 = 14,155,776

每個 stem product：
- activation = 1 bit bipolar
- weight = 1 bit bipolar
- operation = XNOR + popcount

因此 stem convolution general multiplier = 0。

與 R4 binary backbone 約 40,108,032 binary terms 合計時，請分開報：
- thermometer stem BMAC
- backbone BMAC
- total binary terms

不要把 14.16M BMAC 當 conventional MAC。

Thermometer encoder：
R=8=2^3，可用 shift/round + mask generation。
硬體不需要 general multiplier。

但本輪仍有：
- stem BN
- head BN
- final FC

所以仍不能宣稱 entire-network DSP=0。

==================================================
12. Final report
==================================================

建立：

R5T_FINAL_REPORT.md

至少包含：

A. Source R4
- path
- best acc/epoch

B. Thermometer
- R
- L
- input channels
- encoding examples
- raw-input preprocessing definition

C. Initialization
- original R4 normalization std
- projected initialization formula
- zero-epoch projected FP thermometer accuracy

D. T1
- best/final/reload
- best epoch
- training time

E. T2
- binary zero-epoch accuracy
- best/final/reload
- best epoch
- training time
- sign counts

F. Comparison
- vs Official
- vs R4
- vs R5
- vs R5AB

G. Hardware
- stem BMAC count
- weight storage bits
- XNOR/popcount interpretation
- total binary terms
- remaining non-binary modules

H. Conclusion
回答：
1. Thermometer 是否成功解決 3-channel W1 stem bottleneck？
2. 是否值得取代 W1A8？
3. 是否值得下一步測 R=16？

完成後停止。

不要自行開始：
- R=4 / R=16 正式訓練
- fractional activation
- final FC
- BN elimination
- RTL
- FPGA synthesis
- T18/T90
