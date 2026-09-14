目前正式 source：

- R7 best = 85.22% @ epoch 99
- R7 final = 85.20%
- R7 reload = 85.22%
- R7 Head BN signed-pow2 已完成
- Stem W1A1 + Thermometer R=8
- Stem signed-pow2 affine
- R4 binary backbone
- Head signed-pow2 affine
- Final FC 目前仍是 Linear(64,10,bias=True)
- 672 compatibility alpha frozen
- R7 tests 58/58 PASS

本輪做一個公平的 classifier ablation：

# R8A = R7 + W1 FC
# R8B = R7 + Signed-Pow2 FC

兩個 branch 必須從完全相同的 R7 best checkpoint 開始。
請實際整合、驗證、訓練、reload、比較並產生 `R8_FINAL_REPORT.md`。
不要只回 command。

1. 先 git status，不覆蓋 R7。
2. 自動找到正確 R7 best.pt：
   - best≈85.22%
   - epoch=99
   - reload≈85.22%
3. 先跑全部 tests。
4. R8A：
   - FC weight forward only {-1,+1}
   - latent FP weight + STE
   - FC bias保留
   - zero-epoch projection accuracy
   - smoke
   - formal 100 epochs
5. R8B：
   - FC effective weight = sign(W)*2^round(log2|W|)
   - 不預設 exponent clamp
   - 先輸出 640 weights 的 exponent histogram
   - FC bias保留
   - zero-epoch projection accuracy
   - smoke
   - formal 100 epochs
6. 兩個 branch 完全保持：
   - Thermometer R=8
   - Stem W1A1
   - Stem pow2 affine
   - R4 backbone
   - Head pow2 affine
   - 672 frozen alpha
7. LR 優先使用 R7 已驗證穩定的 1e-4：
   - SGD
   - momentum=.9
   - wd=0
   - cosine
   - tau=.99
8. R8A invariants：
   - effective FC only {-1,+1}
   - 640 binary weights
   - FC bias=True
9. R8B invariants：
   - all 640 effective weights exact signed-pow2
   - report unique exponent set/count/histogram
   - FC bias=True
10. Best checkpoint 只能依真正 quantized classifier inference 選。
11. Reload accuracy 必須等於 reported best。
12. 比較至少：
   - source R7 85.22%
   - R8A best/reload
   - R8B best/reload
   - zero-epoch drops
   - training recovery
   - classifier hardware cost
13. Hardware interpretation：
   R8A:
   - FC multiply becomes conditional add/sub
   - no general multiplier
   - lowest weight storage: 640 bits
   R8B:
   - FC multiply becomes shift + optional negate + accumulation
   - no general multiplier
   - storage includes sign + exponent per weight
14. Winner selection：
   不要只看 accuracy。
   請用：
   - accuracy
   - projection shock
   - reload stability
   - exponent complexity
   - storage
   - adder/shift complexity
   綜合判定。
15. 建立 `R8_FINAL_REPORT.md`，清楚回答：
   - 哪個 branch 適合最終 RTL？
   - 哪個 branch accuracy 較高？
   - 哪個 branch hardware 最簡單？
   - 相較 R7 loss 幾 pp？
   - 是否已可宣稱主要 inference datapath 無 general multiplier？
16. 完成兩個 branch 後停止。
不要自行做：
   - bias quantization/removal
   - RTL
   - FPGA synthesis
   - T18/T90
   - KD
   - longer training
