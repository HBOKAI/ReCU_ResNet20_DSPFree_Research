目前正式主線：

- Official ReCU = 87.28%
- R4 = 86.11%
- R5T-Long = 85.14%
- R6 = **85.52% @ epoch 97**
- R6 final = 85.10%
- R6 reload = **85.52%**
- R6 Stem exponent = {-6,-5,-4}
- Stem W1A1 / Thermometer R=8
- Stem Conv bias=False
- 672 frozen compatibility alpha parameters
- R6 tests = 48/48 PASS

現在只做：

# R7 = R6 + Head BN Signed-Pow2 Affine

請直接整合 extension pack、檢查、修正、執行、訓練、驗證，最後建立 `R7_FINAL_REPORT.md`。
不要只回 command。

完成 R7 後停止。

--------------------------------------------------
1. Source
--------------------------------------------------

自動找到 R6 正確的 `best.pt`。

必須確認：

- best_acc = 85.52% 左右
- best_epoch = 97
- reload = 85.52%
- Stem effective weights only {-1,+1}
- Stem Khat signed-pow2
- Stem exponent set 約 {-6,-5,-4}
- Stem Conv bias=None

若不是此 checkpoint，停止並重新尋找。

--------------------------------------------------
2. 不覆蓋既有結果
--------------------------------------------------

先：

git status

新增/整合：

- recu_hw/r7.py
- train_recu_r7.py
- tools/verify_r7_folding.py
- configs/recu_r7.json
- tests/test_r7.py

不要覆蓋 R6/R5T-Long。

--------------------------------------------------
3. 唯一變因
--------------------------------------------------

R6 Head：

GAP
-> BatchNorm1d(64)
-> Linear(64,10,bias=True)

R7 Head：

GAP
-> signed-pow2 affine
-> Linear(64,10,bias=True)

只改 Head BN。

不得改：

- Thermometer
- Stem
- Stem pow2 affine
- R4 backbone
- GAP
- FC weight
- FC bias

--------------------------------------------------
4. Head BN exact fold
--------------------------------------------------

對 64 個 Head channels：

K_j =
gamma_j / sqrt(running_var_j + eps)

B_j =
beta_j
-
gamma_j * running_mean_j
/
sqrt(running_var_j + eps)

所以：

BN(x_j) = K_j*x_j + B_j

注意：
K 不是 gamma。
B 不是 beta。

--------------------------------------------------
5. Exact fold 必須先 PASS
--------------------------------------------------

執行：

python tools/verify_r7_folding.py \
  --source-r6-checkpoint "<R6_BEST>"

量化 K 前，比較：

source:
bn2(GAP_features)

vs

folded:
K*GAP_features+B

要求：

max_abs_error < 1e-5

至少多個 CIFAR batches。

若失敗：
停止 training，先修 folding。

--------------------------------------------------
6. 只量化 Head K
--------------------------------------------------

Khat_j =
sign(K_j) * 2^round(log2(abs(K_j)))

B 本輪維持 float/reference。

禁止：

- quantize B
- remove B
- fold Head BN into FC
- quantize FC
- binarize FC

這輪只隔離 Head BN scale 的 pow2 cost。

--------------------------------------------------
7. 不要 fold Head BN into FC
--------------------------------------------------

雖然數學可以把 BN fold 到 Linear，
但本研究後續可能要處理 FC precision。

因此本輪 Head affine 必須保留成獨立：

GAP
-> Pow2 affine
-> FC

這樣後續 FC 可單獨做 controlled experiment。

--------------------------------------------------
8. Head exponent analysis
--------------------------------------------------

不要預設 exponent range。

先統計 64 個 Head K：

- K min/max
- |K| min/max/mean
- positive/negative count
- rounded exponent histogram
- unique exponent set/count

config 預設不 clamp。

--------------------------------------------------
9. Zero-epoch hard projection
--------------------------------------------------

Source R6：

85.52%

不 fine-tune，直接 Head：

K -> Khat

測：

R7_zero_epoch_hard_projection_acc

計算：

Delta_projection =
R7_zero - 85.52

--------------------------------------------------
10. R6 invariants 必須全部保留
--------------------------------------------------

R7 training/reload 都確認：

- Thermometer R=8
- Stem effective weights only {-1,+1}
- Stem Conv bias=None
- Stem K signed-pow2
- Stem exponent distribution正常
- 672 compatibility alpha frozen
- backbone architecture unchanged

R7 不得再有 standalone `bn2`。

Path：

GAP -> head_affine -> linear

--------------------------------------------------
11. Final FC 完全不改
--------------------------------------------------

Linear：

64 -> 10

必須保留：

- existing FP/multi-bit weight
- existing 10 bias values
- `bias=True`

這輪不要碰 classifier。

測試需確認：

`linear.bias is not None`

--------------------------------------------------
12. Tests
--------------------------------------------------

執行：

python -m unittest discover -s tests -v

所有既有 + R7 tests PASS。

至少新增/確認：

1. exact Head BN fold
2. Head K/B shape = 64
3. Head Khat exact signed power-of-two
4. signed Head K supported
5. no standalone bn2
6. Stem remains binary
7. Stem K remains pow2
8. Stem Conv bias=None
9. FC bias remains present
10. output shape [N,10]
11. reload invariants

--------------------------------------------------
13. Smoke
--------------------------------------------------

python train_recu_r7.py \
  --config configs/recu_r7.json \
  --source-r6-checkpoint "<R6_BEST>" \
  --smoke

確認：

- CUDA
- finite loss
- forward/backward
- Head latent K finite
- Head Khat pow2
- Head B finite
- R6 invariants preserved
- save/reload normal

--------------------------------------------------
14. Diagnostic
--------------------------------------------------

可跑 20 epochs：

python train_recu_r7.py \
  --config configs/recu_r7.json \
  --source-r6-checkpoint "<R6_BEST>" \
  --diagnostic-epochs 20

若沒有 catastrophic issue，進 formal 100 epochs。

--------------------------------------------------
15. Formal training
--------------------------------------------------

預設：

- epochs = 100
- LR = 1e-3
- SGD
- momentum = 0.9
- weight_decay = 0
- cosine
- tau = 0.99

優先保持 R6 successful recipe，不要另開新的 training trick。

--------------------------------------------------
16. Best checkpoint
--------------------------------------------------

Best 只能依真正 R7 inference：

Thermometer
-> W1A1 Stem
-> Stem Pow2 affine
-> R4 backbone
-> GAP
-> Head Pow2 affine
-> unchanged FP FC

選擇。

Reload 再驗證所有 invariants。

--------------------------------------------------
17. Accuracy target
--------------------------------------------------

Source R6：

85.52%

R7：

>=85.2%
非常好，Head BN multiplier removal cost <=0.32 pp。

84.9–85.2%
可接受。

84.5–84.9%
有代價，但仍值得分析。

<84.5%
停止，不做 FC。

計算：

- Delta vs R6 85.52
- Delta vs R5T-Long 85.14
- Delta vs R4 86.11
- Delta vs Official 87.28

--------------------------------------------------
18. Hardware interpretation
--------------------------------------------------

R7 Head affine：

Y_j = Khat_j*x_j + B_j

Khat_j = sign(K_j)*2^k

因此：

x
-> arithmetic shift
-> optional negate
-> +B

Head BN scale：

general multiplier = 0

如果 exponent 種類很少，請指出 fixed shift paths + mux 的實作可能性。

--------------------------------------------------
19. R7 成功後可以宣稱什麼
--------------------------------------------------

可以說：

- Stem convolution is pure XNOR/popcount.
- Stem normalization scale path is signed-pow2 shift/add.
- Binary backbone uses binary convolution + pow2/shift-add operations.
- Head BN scale path is signed-pow2 shift/add.
- All BN scale multipliers considered so far have been eliminated.

還不能說：

entire network DSP=0

因為 Final FC 仍是 FP/multi-bit classifier。

--------------------------------------------------
20. Final report
--------------------------------------------------

建立：

R7_FINAL_REPORT.md

至少包含：

A. Source R6
- path
- 85.52%
- epoch 97
- reload

B. Exact head fold
- max_abs_error

C. Head K/B statistics
- ranges
- sign counts
- exponent histogram

D. Hard projection
- zero epoch accuracy
- projection delta

E. Formal training
- recipe
- best / epoch
- final
- reload
- training time

F. Invariants
- stem binary sign
- stem exponent
- frozen alpha
- no bn2
- FC bias preserved

G. Accuracy comparisons

H. Hardware conclusion

--------------------------------------------------
21. Stop
--------------------------------------------------

完成 R7 後停止。

不要自行開始：

- FC W1
- FC pow2
- FC bias removal
- Head B quantization
- Stem B quantization
- RTL
- FPGA synthesis
- ASIC T18/T90
