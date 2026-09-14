目前結果：
- Official ReCU = 87.28%
- R4 = 86.11%
- R5 naive W1A8 = 83.77%
- R5 相較 R4 = -2.34 pp
- R5 INT8 saturation = 0
- R5 無 sign collapse

本輪只做：
R5AB = R4 + Progressive Scaled W1A8 Stem

必須從 R4 best 86.11% warm-start，不要從 R5 warm-start。

核心：
alpha_c = mean(abs(W_c))
B_c = sign(W_c)
W_eff = (1-lambda)W + lambda*alpha_c*B_c

lambda schedule：
- epoch 1..30 線性升到 1
- epoch >=30 固定 1

A8 quantizer 維持 R5：
- signed INT8
- scale=2^-5
- 不要先改 activation quantizer

最重要規則：
每一 epoch 都要額外 force lambda=1 做 deployment evaluation。
Best checkpoint 只能依 `binary_lambda1_test_acc` 選。
不能用 lambda<1 的 scheduled accuracy 當作 R5AB 成績。
best.pt 與 reload 都必須驗證 lambda=1。

請依序：
1. 自動找到 R4 best.pt（86.11%, epoch 98）
2. git status，確認不覆蓋既有 R1~R5 results
3. merge/audit R5AB files
4. 跑 verify_r5ab_conversion.py
5. 跑全部 unit tests
6. smoke
7. 20-epoch diagnostic
8. 若正常，跑正式 100 epochs
9. 建立 `R5AB_FINAL_REPORT.md`
10. 完成後停止

Conversion report 必須包含：
- source R4 path/best acc/best epoch
- lambda=0 A8+FP stem accuracy
- lambda=1 scaled-W1A8 zero-epoch accuracy
- binary shock
- A8 saturation
- alpha min/max/mean/median
- +1/-1 sign counts

正式結果必須包含：
- best forced-lambda1 accuracy
- best epoch
- final forced-lambda1 accuracy
- reload forced-lambda1 accuracy
- final lambda exactly 1
- training time
- trainable params
- Δ vs R4 86.11
- Δ vs naive R5 83.77
- Δ vs Official 87.28
- recovered pp = R5AB - 83.77
- recovery ratio = (R5AB - 83.77)/2.34

Hardware 結論要精確：
training 的 progressive mixing 是 training-only。
deployment lambda=1，因此 stem conv core 為：
signed INT8 q × sign weight = +q / -q
=> convolution core general multiplier = 0。

但 alpha 仍需處理。
因為 stem 後面是 BN，可 fold：
K = gamma*alpha/sqrt(var+eps)
B = beta - gamma*mu/sqrt(var+eps)
=> K*S+B

本輪只做 exact float fold 統計，不要把 K 擅自量成 pow2。
因此只能宣稱：
- W1A8 stem convolution core general-multiplier-free
- alpha can be folded into stem BN
不能宣稱整個 stem post-processing 已 multiplier-free。

不要自行開始：
- fused stem K pow2 QAT
- final FC
- BN elimination
- RTL
- FPGA synthesis
- T18/T90
