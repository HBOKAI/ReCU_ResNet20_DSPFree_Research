# 現行 R8B 模型架構與工作量

依 [R8 final report](reports/R8_FINAL_REPORT.md)、現有 `recu_hw` model implementation 與 R8B `best.pt` 的實際參數 metadata 核對。R8B 是**軟體模型架構**；H0/H1MP/H2A-v2/H2B 只改其推論數值表示，並未建立新拓樸或重新訓練權重。

## 逐層資料流

| 順序 | 空間/通道 | 運算 |
|---|---|---|
| Input | 32×32×3 | CIFAR-10 RGB；`ToTensor` 後 `[0,1]`，不先 Normalize 再 Thermometer。 |
| Thermometer | 32×32×96 | R=8，每 RGB channel 32 個 bipolar `{-1,+1}` channels。 |
| Stem | 32×32×16 | W1A1 binary Conv3×3, 96→16, stride1；signed-pow2 stem affine + Hardtanh。 |
| Stage 1 | 32×32×16 | 3 個 ReCU blocks；每 block 兩個 binary Conv。 |
| Stage 2 | 16×16×32 | 3 個 blocks；首個 16→32、stride2，Option-A shortcut。 |
| Stage 3 | 8×8×64 | 3 個 blocks；首個 32→64、stride2，Option-A shortcut。 |
| GAP | 64 values | 8×8 每通道求和並以 scale metadata 表示 `/64`，見 H2A-v2。 |
| Head | 64 values | BN2 已移除；獨立 signed-pow2 affine。 |
| Classifier | 10 logits | Signed-pow2 FC 64→10；FC bias 存在，H0 後為 INT6。 |

每個 ReCU block 的**double-skip**：`x → BConv1 → pow2 affine1 → + shortcut(x) = x1 → Hardtanh → BConv2 → pow2 affine2 → + x1 → QRPReLU`。第二次 residual 加的是第一個 add 產生的 `x1`，不是原始 x；Stage 2/3 首 block 的 shortcut 為實際 Option-A：空間隔點抽樣加對稱 zero channel padding。H1MP/H2 均依此結構做 18 個 residual alignment 檢查。[H1 原始報告](reports/H1_INTEGER_ACTIVATION_RESIDUAL_REPORT.md)

## 實際參數核對

從現有 R8B best checkpoint 載入模型後重新計數：

| 類別 | 數量 | 說明 |
|---|---:|---|
| Binary stem Conv weights | **13,824** | 96×16×3×3。 |
| Binary backbone Conv weights | **267,264** | 9 blocks × 2 BConv 的總和。 |
| Binary Conv weights total | **281,088** | `13,824 + 267,264`。 |
| FC weights + bias | **650** | 64×10 signed-pow2 weights + 10 bias。 |
| 其他 trainable | **2,512** | stem affine K/B 32、backbone affine K/B 1,344、QRPReLU 1,008、head affine K/B 128。 |
| **Trainable total** | **284,250** | 281,088 + 650 + 2,512。 |
| Frozen compatibility alpha | **672** | 保留在 checkpoint/state_dict，但 R8B forward 不使用。 |
| Registered total | **284,922** | trainable + frozen；不可拿此數與 284,250 混稱。 |

R8B 的 640 FC 權重有效值皆為 signed powers of two，報告記錄 exponent `−9…0`；權重估計儲存 640×(1 sign + 5 exponent)=3,840 bits。R8B 階段 FC bias 還是 FP32，H0 才量化為 INT6；不能用 R8 報告的含 FP32 bias 儲存估算當成 H0/H2 的 final storage 數。

## Binary convolution workload（每張 32×32 影像）

| 類別 | Binary-conv terms/image |
|---|---:|
| Stem 96→16, 3×3, 32×32 | **14,155,776** |
| Backbone 18 BConv | **40,108,032** |
| **Binary Conv total** | **54,263,808** |
| FC signed-pow2 shifted terms | **640**（非 binary-conv terms） |

形式上把上述異質算術 sites 相加為 **54,264,448**，只適合比較工作量，不是 standard FLOPs、INT8 MAC 或實際 cycle/FPS。Binary convolution 的 bitwise XNOR+popcount、signed-pow2 affine/FC 的 shift/optional negate/add，仍需 RTL 決定平行度、memory、adder tree 與時序。H2A-v2/H2B 的不同 accumulator 位寬**不改變**模型參數數或 binary terms 數。
