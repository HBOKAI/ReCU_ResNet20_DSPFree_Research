目前主線：
- Official ReCU 87.28%
- R4 86.11%
- R5T-Long T2 85.14% @ epoch 200，reload 85.14%
- Thermometer R=8
- Stem W1A1 3x3, 96->16
- Stem effective weights only {-1,+1}
- Stem sign +1=6839 / -1=6985
- Total convolution 54,263,808 BMAC/image

現在只做：
R6 = R5T-Long + Stem BN exact fold + signed-pow2 K。

請直接整合 extension pack、檢查、修正、執行、訓練與建立 `R6_FINAL_REPORT.md`，不要只回 command。

1. 先 `git status`，不得覆蓋 R5T/R5T-Long。
2. 自動找到正確 R5T-Long `t2_best.pt`，驗證 best_acc≈85.14、best_epoch=200、binary stem reload 正常。
3. Stem source：
   `Thermometer -> W1A1 Conv(bias=False) -> BN2d(16) -> Hardtanh`
   R6：
   `Thermometer -> W1A1 Conv(bias=False) -> fused affine -> Hardtanh`
4. Exact fold：
   K = gamma/sqrt(running_var+eps)
   B = beta - gamma*running_mean/sqrt(running_var+eps)
   先禁止 K quantization，驗證 `BN(S)` 與 `K*S+B`，max_abs_error < 1e-5。
   若失敗，停止並修 folding bug。
5. 只量化 K：
   `Khat = sign(K_latent)*2^round(log2(abs(K_latent)))`
   B 本輪保持 float/reference。
6. 不要預設 exponent range。先統計 16 個 K 的 range、sign、rounded exponent histogram、unique exponents。
7. 做 zero-epoch hard projection，記錄 accuracy 與相較 85.14% 的 drop。
8. 優先檢查現有 R4 signed-pow2 quantizer。如果語意一致，對齊/重用其 STE；不得產生兩套語意不同 quantizer。
9. 保留 R5T-Long 既有 frozen policy，尤其 672 個 R4 compatibility alpha 不得意外解凍。輸出 frozen parameter count/category。
10. 跑全部 tests。R6 至少驗證：
   - exact fold
   - K/B shape 16
   - Khat exact signed pow2
   - stem effective weight only {-1,+1}
   - stem Conv bias=None
   - R6 無 standalone bn1
   - output [N,10]
   - reload invariant
11. Smoke 正常後可跑 20-epoch diagnostic，再正式 100 epochs。
12. Formal defaults：SGD, lr=1e-3, momentum=.9, weight_decay=0, cosine, tau=.99。若 R4 successful pow2 recipe 有明確不同，先比較後才調整並寫入報告。
13. Best checkpoint 只能依真正 R6 inference：W1A1 stem + signed-pow2 K + float B + no stem BN。
14. Reload 再驗證：
   - Khat power-of-two
   - stem {-1,+1}
   - Conv bias None
   - no bn1
15. Accuracy 判定：
   - >=84.8% 很好
   - 84.5–84.8% 可接受
   - 84.0–84.5% 有代價
   - <84.0% 停止，不往 Head BN/FC
16. Final report 至少包含：
   source、exact fold error、K/B stats、exponent histogram、zero-epoch projection、
   training recipe、best/final/reload、sign/frozen invariants、與 R5T-Long/R5/R4/Official 比較、
   hardware interpretation。
17. 硬體允許宣稱：
   Stem Conv = XNOR+popcount；
   Stem BN scale = arithmetic shift + optional negate；
   B = channel-wise constant add；
   因此 Stem Conv + normalization scale path general-multiplier-free。
   還不能宣稱 entire network DSP=0。
18. 完成 R6 後停止。禁止自行開始 B quantization、Head BN、FC、RTL、FPGA、T18/T90。
