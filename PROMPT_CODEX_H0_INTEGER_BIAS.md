目前正式模型：

- R8B best = 85.40%
- R8B 使用 Signed-Pow2 FC
- Stem W1A1 + Thermometer R=8
- Stem affine K = signed-pow2
- Backbone fused affine K = signed-pow2
- Head affine K = signed-pow2
- FC weight = signed-pow2
- FC bias 仍存在
- 672 compatibility alpha frozen

我已提供 H0 integer bias sweep extension pack。

請你直接在目前 workspace 中整合、檢查、修正並執行。
不要只回 command，也不要要求我手動改程式。

本輪不是重新訓練模型。
本輪是從同一個 R8B best checkpoint 做 zero-shot pure-integer bias sweep。

==================================================
1. Workspace safety
==================================================

先執行：

git status

不得覆蓋：
- R8B best checkpoint
- R8_FINAL_REPORT.md
- R8 training files
- R8 experiment directories

優先新增/整合：

- recu_hw/integer_bias.py
- tools/eval_integer_bias_sweep.py
- tests/test_integer_bias.py
- configs/h0_integer_bias_sweep.json

如果 extension pack API 與目前 workspace 有小幅差異，
請自行依實際 R8B implementation 修正 import / constructor / module name。

但不得改變 H0 實驗定義。

==================================================
2. Source checkpoint
==================================================

自動找到正式 R8B best checkpoint。

必須驗證：

- best_acc 約 85.40%
- reload 約 85.40%
- Stem effective weight only {-1,+1}
- Stem K signed-pow2
- Backbone K signed-pow2
- Head K signed-pow2
- FC effective weights signed-pow2
- FC bias exists
- 672 compatibility alpha frozen

若不是正確 R8B checkpoint，停止並重新找。

先回報：
- source path
- stored best_acc
- stored best_epoch
- actual reload accuracy

==================================================
3. 本輪只量化 B / bias
==================================================

只處理：

1. Stem affine B
2. Backbone 所有 fused affine B
3. Head affine B
4. Final FC bias

禁止修改：

- Thermometer R=8
- binary weights
- Stem W1A1
- Stem K
- Backbone K
- QRPReLU
- Head K
- FC signed-pow2 weights
- pow2 exponents
- activation precision
- residual precision
- accumulator precision
- GAP precision

==================================================
4. Pure-integer representation
==================================================

不要使用 Q3.5 / Q4.4 之類命名。

每個 bias tensor 使用：

q = round(B * 2^s)

其中：
- q 是 signed integer
- s 是該 layer 共用的 integer shift metadata

RTL 最終只保存：
- q
- s

Python accuracy simulation 可以使用：

B_hat = q * 2^(-s)

但這只是 reference simulation。

硬體 interpretation 必須是：
integer + shift only，
不得需要 floating-point storage / multiplier / adder。

==================================================
5. Target discovery
==================================================

請先列出實際找到的所有 bias targets：

- module name
- class name
- parameter count

必須至少包含：

- stem_affine.bias
- backbone fused affine biases
- head_affine.bias
- linear.bias

如果自動 selector 漏掉任何 R4 backbone fused affine B，
請依 workspace 的真實 module class 修正 selector。

不要誤把：
- QRPReLU trainable parameter
- PReLU-like parameter
- 非 affine offset

當作本輪 bias target。

==================================================
6. Bit-width sweep
==================================================

從完全相同的 R8B best checkpoint 分別測：

INT8
INT7
INT6
INT5
INT4

signed ranges：

INT8 [-128,127]
INT7 [-64,63]
INT6 [-32,31]
INT5 [-16,15]
INT4 [-8,7]

如果 INT4 相較 source 85.40% 的 drop <= 0.50 pp，
再測 INT3 [-4,3]。

禁止 fine-tune。

==================================================
7. Per-layer integer shift search
==================================================

每一個 bit-width 都重新搜尋每一層的 s。

初始搜尋：

s = 0 ~ 12

如果此範圍完全無法 zero-saturation，
可自動往負 shift 擴展到 -12。

每個 layer 優先順序：

1. zero saturation
2. minimum saturation count
3. minimum MSE
4. minimum max absolute error
5. shift 不要不必要地過大

==================================================
8. Evaluation
==================================================

每個 width 跑完整 CIFAR-10 test set。

輸出：

| Width | Accuracy | Delta vs R8B | Saturation |
|---|---:|---:|---:|
| Float reference | source reload | 0 | - |
| INT8 | ? | ? | ? |
| INT7 | ? | ? | ? |
| INT6 | ? | ? | ? |
| INT5 | ? | ? | ? |
| INT4 | ? | ? | ? |
| INT3 | optional | optional | optional |

==================================================
9. Per-layer report
==================================================

每個 width、每個 target 輸出：

- module name
- class name
- parameter count
- original float min/max
- integer bits
- selected shift s
- q min/max
- q values or exported integer parameter table
- saturation count
- MSE
- max abs quantization error

禁止使用 Q-format 命名。

==================================================
10. Accuracy criteria
==================================================

相較實際 R8B source reload：

drop <= 0.10 pp:
near-lossless

drop <= 0.30 pp:
recommended range

drop <= 0.50 pp:
minimum acceptable

最後明確回答：

1. lowest near-lossless bit-width
2. recommended production bit-width
3. absolute minimum acceptable bit-width

==================================================
11. Integer export verification
==================================================

確認每個 target：

- q dtype 是 integer
- q 落在 W-bit signed range
- export 中保存 q + shift
- 不需要 float storage
- 不需要 general multiplier
- 不需要 floating-point adder

同時確認所有 R8B invariants 沒改。

==================================================
12. Tests
==================================================

跑：

python -m unittest discover -s tests -v

全部 existing tests + H0 tests 都要 PASS。

至少確認：

- signed range
- rounding
- clipping
- saturation detection
- shift search
- integer export/reconstruction equivalence
- R8B invariants

==================================================
13. Final report
==================================================

建立：

H0_INTEGER_BIAS_SWEEP_REPORT.md

至少包含：

A. R8B source verification
B. discovered bias target inventory
C. INT8 -> INT4 / optional INT3 accuracy table
D. each layer's integer shift
E. q ranges
F. saturation
G. quantization error
H. lowest near-lossless width
I. recommended production width
J. absolute minimum width
K. hardware interpretation

並保存 machine-readable：

H0_INTEGER_BIAS_SWEEP_RESULTS.json

==================================================
14. Important wording
==================================================

本輪成功後可以說：

“All affine offsets and classifier bias have a pure-integer representation
using signed integers and power-of-two shift metadata.”

還不能說：

“Entire network is integer-only.”

因為 activation / residual / GAP / accumulator finite bit-width 尚未完成。

==================================================
15. Stop
==================================================

完成 H0 report 後停止。

不要開始：
- fine-tuning
- activation quantization
- residual quantization
- accumulator truncation
- GAP quantization
- RTL
- FPGA synthesis
- T18/T90
- KD
- longer training
