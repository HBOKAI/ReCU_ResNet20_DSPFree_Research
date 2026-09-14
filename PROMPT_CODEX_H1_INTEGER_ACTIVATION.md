目前正式硬體數值 baseline：

- R8B best = 85.40%
- H0 已完成 pure-integer bias sweep
- H0 production 選擇 = INT6 bias/offset
- H0 INT6 accuracy = 85.13%
- 相較 R8B = -0.27 pp
- 21 個 bias tensors / 762 bias parameters
- H0 所有 INT6 bias 都是 0 saturation
- H0 tests = 69/69 PASS

現在做：

# H1 = R8B-H0 + Multi-bit Activation / Residual Pure-Integer Sweep

我已提供 H1 extension pack。

請你直接在目前 workspace：
- 整合
- 檢查實際 R4/R8B block forward
- 實作 workspace-specific adapter
- 跑 tests
- calibration
- INT8 -> INT4 sweep
- 建立完整 report

不要只回 command，也不要要求我手動修改程式。

本輪禁止重新訓練 / fine-tune。

==================================================
1. Source 必須固定
==================================================

Source：

R8B best checkpoint
+
H0 INT6 bias export

必須先重建：

R8B-H0 INT6 bias baseline

並驗證完整 CIFAR-10 test accuracy 約：

85.13%

建議容許誤差：

±0.08 pp

若無法 reproduce H0 85.13%：

停止 H1，
先找出 H0 INT6 export / module mapping / source checkpoint 問題。

不要直接往 activation sweep 繼續。

==================================================
2. 使用 H0 已輸出的 INT6 integer bias
==================================================

請讀取：

H0_INTEGER_BIAS_SWEEP_RESULTS.json

找到：

bits = 6

使用其中每個 layer：

- q_values
- shift

重新套用：

B_hat = q * 2^(-shift)

這只是 Python reference simulation。

硬體 interpretation 仍然是：

- signed INT6 q
- integer shift metadata
- no floating-point storage

本輪不得重新搜尋 bias bit-width。

H0 bias 固定不動。

==================================================
3. H1 只量化 multi-bit state
==================================================

本輪 target 是真正會在 RTL 保存 / 傳遞的 multi-bit activation/residual state。

至少檢查並包含：

1. Stem affine + Hardtanh 後、供後續 shortcut/BConv 使用的 state
2. 每個 residual block 的第一個 residual add 後保存的 x1
3. 每個 block 第二個 affine + x1 residual add 後的 state
4. 每個 block QRPReLU output / block output
5. Stage transition 的 shortcut / residual state
6. Head signed-pow2 affine output，也就是 FC input

請依實際 R4/R8B implementation 命名並列出所有 quant nodes。

==================================================
4. 本輪不要量化的東西
==================================================

明確禁止在 H1 截斷：

- Binary convolution popcount / signed accumulator
- Stem binary-conv accumulator
- Backbone binary-conv accumulator
- GAP accumulation sum
- Final FC internal accumulator

這些留到後面的 accumulator experiment。

也不要改：

- binary weights
- signed-pow2 K
- FC signed-pow2 weights
- H0 INT6 bias
- Thermometer
- architecture
- QRPReLU functional form

==================================================
5. 不要只靠 generic forward hook
==================================================

R4/ReCU block 有 inline residual add / double-skip。

Generic module hook 很可能抓不到真正的：

x1
residual add
second residual add

所以請先閱讀目前 workspace 的：

- R4 block implementation
- QRPReLU implementation
- R7/R8B forward

然後建立：

recu_hw/h1_model_adapter.py

依：

recu_hw/h1_adapter_contract.py

實作真正的 quantization boundaries。

如果需要：
- subclass block
- wrapper block
- controlled forward override

都可以。

但不要改變 mathematical architecture。

==================================================
6. Pure integer representation
==================================================

不要使用：

Q3.5
Q4.4
fractional-bit 命名

每個 activation node i 使用：

q_i = round(x_i * 2^s_i)

其中：

- q_i 是 signed W-bit integer
- s_i 是 node-level integer shift metadata

Python 模擬：

x_hat = q_i * 2^(-s_i)

只用來模擬該 integer+shift representation。

RTL 最終只需要：

- integer q_i
- integer shift s_i

==================================================
7. Calibration 不得使用 test set
==================================================

這點很重要。

Scale / shift calibration：

只能使用 CIFAR-10 TRAIN data。

預設：

10000 train samples

如果執行成本合理，可使用完整 50000 train samples，
但不得因 test accuracy 去挑 shift。

CIFAR-10 TEST：

只能最後評估 accuracy。

這樣避免 test-set leakage。

==================================================
8. Shift 選擇
==================================================

對每個 quant node、每個 bit-width：

根據 calibration train data 的：

absmax

選：

largest integer shift s

使 calibration observed range：

zero saturation

搜尋邊界：

s = -16 ~ +24

若實際 range 有需要可以擴大，
但必須寫入 report。

不要使用 test accuracy 搜尋 s。

==================================================
9. Bit-width sweep
==================================================

固定同一個：

R8B-H0 baseline = 85.13%

依序測：

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

如果 INT4 相較 H0 85.13%：

drop <= 0.50 pp

才額外測 INT3。

禁止 fine-tune。

==================================================
10. Residual addition 必須做 integer scale alignment
==================================================

這是 H1 最重要的 correctness requirement。

假設兩個 branch：

a_real ~= qa * 2^(-sa)

b_real ~= qb * 2^(-sb)

不能直接：

qa + qb

除非：

sa == sb

相加前必須對齊到共同 output scale：

s_out

使用：

- integer left shift
- arithmetic right shift with defined rounding
- optional saturation only at the designated output W-bit node

再：

q_out = qa_aligned + qb_aligned

請對每個 residual add 建立 verification：

- branch A shift
- branch B shift
- common output shift
- required shift amount
- integer-add equivalence
- saturation count

==================================================
11. Option-A shortcut
==================================================

Stage2 / Stage3 downsample 的 Option-A shortcut 必須特別驗證。

不要假設它和普通 identity shortcut 相同。

檢查實際 implementation：

- spatial downsample
- channel padding
- scale inheritance
- integer alignment

並寫入 residual alignment report。

==================================================
12. Hardtanh / sign 關係
==================================================

不要因為下一個 BConv 內部會 sign activation，
就誤刪 multi-bit residual state。

ReCU/R4 residual path 仍會重用 magnitude。

必須依目前 block 真實 dataflow 保留：

- residual magnitude
- x1 lifetime
- block output magnitude

Binary convolution input 可以取 sign，
但 residual state 必須維持 H1 定義的 W-bit integer representation。

==================================================
13. QRPReLU
==================================================

QRPReLU 本身已是 hardware-friendly shift/add approximation。

本輪：

- 不改 QRPReLU parameter
- 不重新量化其 coefficient
- 只量化它的 multi-bit output state

如果 QRPReLU 內部 intermediate 目前仍需要更寬位元，
先保留 full internal precision。

H1 只在 output boundary 截成 W-bit。

==================================================
14. Head / FC boundary
==================================================

Head signed-pow2 affine output：

是 Final Pow2 FC 的輸入。

本輪量化此 multi-bit tensor。

Final FC：

- signed-pow2 weight 不變
- INT6 FC bias 不變
- FC accumulation 本輪保持 full precision

==================================================
15. Calibration report
==================================================

對每個 node 輸出：

- node name
- tensor role
- calibration sample count
- min
- max
- absmax

並對每個 W：

- selected shift s
- representable integer range
- test saturation count

Test saturation 可以統計，
但不得回頭拿 test data 修改 s。

==================================================
16. Accuracy table
==================================================

建立：

| Width | Accuracy | Delta vs H0 85.13 | Test saturation |
|---|---:|---:|---:|
| H0 reference | 85.13 | 0 | - |
| INT8 | ? | ? | ? |
| INT7 | ? | ? | ? |
| INT6 | ? | ? | ? |
| INT5 | ? | ? | ? |
| INT4 | ? | ? | ? |
| INT3 | optional | optional | optional |

判定：

drop <= 0.10 pp
= near-lossless

drop <= 0.30 pp
= recommended range

drop <= 0.50 pp
= minimum acceptable

==================================================
17. Tests
==================================================

整合 extension pack 後跑：

python -m unittest discover -s tests -v

除了 existing tests，
至少確認：

- signed integer ranges
- float->integer quantization
- shift selection
- signed right-shift rounding
- integer requantization
- residual scale alignment
- integer residual add
- Option-A adapter behavior
- H0 INT6 bias still fixed
- R8B binary/pow2 invariants unchanged

全部 PASS 才接受 H1。

==================================================
18. 請特別檢查 H1 adapter
==================================================

請輸出：

adapter.list_quant_nodes()

確認沒有：

- 漏掉真正 residual state
- 重複 quant 同一 state
- 誤量 binary accumulator
- 誤量 GAP accumulator

如果 adapter contract 與目前 code 不符，
請自行修正 adapter，
而不是改變實驗目標。

==================================================
19. 最終推薦
==================================================

請回答：

1. Lowest near-lossless activation/residual width
   drop <= 0.10 pp

2. Recommended production width
   drop <= 0.30 pp

3. Absolute minimum acceptable width
   drop <= 0.50 pp

4. 哪個 node 最敏感？

5. 是否存在某個 residual node 導致 INT5/INT4 明顯崩潰？

==================================================
20. Report
==================================================

建立：

H1_INTEGER_ACTIVATION_RESIDUAL_REPORT.md

以及 machine-readable：

H1_INTEGER_ACTIVATION_SWEEP_RESULTS.json

Report 至少包含：

A. Source R8B / H0 verification
B. H0 INT6 bias reconstruction
C. H1 quant node list
D. Calibration methodology
E. Train calibration ranges
F. Per-width shifts
G. Residual integer scale-alignment table
H. Option-A handling
I. Accuracy table
J. Test saturation statistics
K. Sensitive-node analysis
L. Recommended production bit-width
M. Hardware interpretation

==================================================
21. 本輪完成後可以說什麼
==================================================

若成功：

可以說：

“Affine offsets/classifier bias use INT6 integer storage, and the selected
multi-bit residual/activation states have a verified W-bit integer
representation with power-of-two scale metadata.”

還不能說：

“Entire network is fully finite-width integer-only.”

因為以下尚未完成：

- binary-conv accumulator truncation
- GAP accumulator width
- FC accumulator width
- final logit width

==================================================
22. Stop
==================================================

完成 H1 後停止。

禁止自行開始：

- fine-tuning
- accumulator truncation
- GAP quantization
- FC accumulator quantization
- RTL
- FPGA synthesis
- T18/T90
- KD
- longer training
