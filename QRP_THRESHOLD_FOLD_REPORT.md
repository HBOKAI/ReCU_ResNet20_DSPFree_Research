# QRPReLU Threshold Folding Exact-Equivalence — FAIL / 不適用

本輪**未**修改模型、checkpoint、H0 INT6、H1MP 58-node plan、H2A-v2/H2B 位寬或 GAP deferred scaling；未重訓、未做 output offset folding、未開始 RTL/synthesis。依使用者指定的先決條件，確認無法建立精確的 threshold-folded path 後，**停止於可行性與整數反例檢查；沒有執行 folded official CIFAR-10 TEST**。既有 H2A-v2 official TEST 85.03% 僅是參照，不能冒稱本輪重新測得。

## 1. CURRENT QRPReLU EXACT EQUATION（真正使用的程式）

R8B 繼承 R7 的 9 個 block activation；真實 class/forward 位於 [`recu_hw/qrprelu.py`](recu_hw/qrprelu.py)，H2A-v2/H2B integer replay 位於 [`recu_hw/h2_workspace.py`](recu_hw/h2_workspace.py)。前者 `forward()` 第 28–47 行，後者 `_qrprelu_integer()` 第 411–507 行；H1MP `second_add`/`qrprelu_output` 量化邊界見 [`recu_hw/h1_model_adapter.py`](recu_hw/h1_model_adapter.py)與 [`recu_hw/h1s_mp_quant.py`](recu_hw/h1s_mp_quant.py)。

對每個 channel `i`，目前浮點 forward **確切**是：

```text
k_i = round_ste(a_i)       # forward 時為整數 exponent
n_i = 2^(k_i) * (x_i + xi1_i) + xi2_i
y_i = x_i,  if x_i >= 0
    = n_i,  if x_i < 0
```

- `x_i` 是第二次 residual add 後、QRPReLU 前的輸入；**branch condition 是 `x_i >= 0`**。
- `a_i` 決定負分支 signed-power-of-two slope；`xi1_i` 僅在**負分支內**加到 x，是該分支的 horizontal offset；`xi2_i` 是負分支 output offset。
- `a`、`xi1`、`xi2` 都是 per-channel，9 個節點通道數 `16×3 + 32×3 + 64×3 = 336`，各參數類型各 336 個 scalar。
- **不存在名為 gamma 或具同義功能的 learned branch-threshold parameter。真正 branch threshold 已是固定零。** `xi1` 不控制 branch decision，也不是在比較前做的 threshold subtract。

H2 software integer path 對 `second_add` 的 `q_in`（代表 `q_in·2^(−s_in)`）採固定 QRP parameter shift `P=24`，先在建置/推論表示中計算：

```text
Q1_i = round(xi1_i * 2^P)
Q2_i = round(xi2_i * 2^P)
Qx_i = q_in_i << (P - s_in)
I_i  = saturate_to_inner_width(Qx_i + Q1_i)
O    = max(P, P - min_i(k_i))
N_before_offset_i = I_i << (O + k_i - P)
N_after_offset_i  = N_before_offset_i + (Q2_i << (O - P))
Pos_i = q_in_i << (O - s_in)
Qpre_i = where(q_in_i >= 0, Pos_i, N_after_offset_i)
Qout_i = saturate_to_output_width(Qpre_i)
```

`Qout·2^(−O)` 經現有 float32 dequantization，再由 frozen H1MP `qrprelu_output` bits/shift 做 `round→clamp` requantization。H2 trace 另外記錄負分支的 finite-width 值，但實際選擇是依 `q_in >= 0`，最後 `Qout` 自身再飽和；以上方程以真正輸出路徑為準。`xi2` 及其 add **完全保留**。

## 2. 為什麼 requested threshold fold 無法成立

若前級第二個 affine/residual 的結果是 `r = K·z + B + x1`，程式真正的輸出是 `r>=0` 時直接回傳 `r`，否則回傳 `2^k(r+xi1)+xi2`。因此：

1. 若把 threshold 寫成 `gamma`，此模型的 `gamma=0` 是**常數**；`B_folded=B−gamma=B` 只是 no-op，沒有 threshold parameter/read/subtract 可刪。
2. 若把 `xi1` 誤認成 `−gamma`，試著做 `B_folded=B+xi1`，令 `r'=r+xi1`，則原本的 branch condition 是 `r' >= xi1`，**不是** `r'>=0`。正分支原本輸出 `r`，fold 後若直接輸出 `r'` 亦差 `xi1`；要補回必須改分支或新增 subtract/offset，違反本輪限制。負分支的 xi1 add 雖可形式上移動，**整個 piecewise function 無法維持一致**。
3. 整數 scale 也無損不相容。9 個 `second_add` 的 frozen H1MP 輸入 shift 為 `3` 或 `4`，而 H2 `xi1` 表示於 shift `24`；本次實際 checkpoint 的 **336/336 個 Q1 都非零，且 0/336 能被 `2^(24−s_in)` 整除**。把 xi1 移到既有 `q_in` grid 需要新取整；前級 H0 INT6 B 另有本身 scale、之後還有 residual 對齊與 H1MP requantization，不能直接在不同整數 scale 相減。即使某值恰可對齊，前述分支/identity 阻礙仍成立。

另一種改寫負分支常數可能涉及修改 output offset，**屬使用者明確排除的不同實驗**；本輪沒有建立或測試。

## 3. 逐節點 scale 與 bit-accurate 反例

以現存 R8B `best.pt`、H1MP frozen plan、H2A-v2 safe 與 H2B plan 做**唯讀**核對，所有 9 個 QRPReLU nodes 均已檢查。表中的 mismatch 是**合成反例**：令每個 channel 的原始 `q_in=0`，原路徑由正分支輸出零；診斷性錯誤提案把 `xi1` 取整到 incoming q grid、移到前級、在 zero 比較並保留相同 `xi2`/位寬。此提案本身**不符合 no-new-rounding**，因此不是可部署 folded path；不代表 official TEST 數據。

| QRP node | C | H1MP in shift | H2A inner/out bits, out shift | H2B inner/out bits | Q1 可無損落在 in grid | 合成反例 QRP q mismatch | 合成反例 H1MP q mismatch |
|---|---:|---:|---|---|---:|---:|---:|
| `layer1.0` | 16 | 3 | 28/36, 32 | 27/35 | 0 | 11 | 11 |
| `layer1.1` | 16 | 3 | 28/36, 32 | 27/35 | 0 | 3 | 3 |
| `layer1.2` | 16 | 4 | 29/37, 32 | 27/35 | 0 | 13 | 8 |
| `layer2.0` | 32 | 3 | 28/36, 32 | 27/35 | 0 | 14 | 4 |
| `layer2.1` | 32 | 3 | 30/38, 32 | 27/35 | 0 | 12 | 12 |
| `layer2.2` | 32 | 3 | 28/29, 25 | 27/29 | 0 | 17 | 17 |
| `layer3.0` | 64 | 3 | 28/35, 32 | 27/35 | 0 | 6 | 6 |
| `layer3.1` | 64 | 3 | 29/37, 32 | 27/35 | 0 | 9 | 9 |
| `layer3.2` | 64 | 3 | 29/36, 32 | 27/35 | 0 | 0 | 0 |
| **合計** | **336** | — | — | — | **0** | **85** | **70** |

同一個 `q_in=0` 診斷，safe 與 H2B 皆為：pre-branch q 有 **85** channel 不一致；branch decision **1** channel 不一致；負分支 shifted term **336** channel 不一致；選定值在 output offset 前/後各 **85** channel 不一致；final internal QRP q **85** channel 不一致；H1MP requantized QRP output **70** channel 不一致。Safe diagnostic 最大 raw QRP integer error 為 **2,684,354,560**（不同 node 的 scale 不同，此數只是 raw integer 最大值，非共同物理單位的 error）。即使部分渠道最終 requantization 偶然掩蔽差異，**內部 exact-equivalence 已被否定**。完整 per-node/freeze hash 見 [QRP_THRESHOLD_FOLD_RESULTS.json](QRP_THRESHOLD_FOLD_RESULTS.json)；可重現的唯讀檢查見 [`tools/audit_qrp_threshold_folding.py`](tools/audit_qrp_threshold_folding.py)。

## 4. PASS gate、accuracy 與硬體後果

| 指標 | 結果 |
|---|---|
| QRP nodes 已核對 | **9/9** 可行性與 scale；有效 folded path **0/9**。 |
| 有效 folded path 的 max_abs_error / branch mismatch / QRP mismatch | **未測（null）**；因根本沒有合法 folded path，不能填 0。合成反例見上表。 |
| Folded prediction mismatch、folded official TEST accuracy、delta vs 85.03% | **未測（null）**；bit-accurate gate 未通過，依規格停止。 |
| 既有 H2A-v2 official TEST 參照 | **85.03%**，本輪未重跑。 |
| 新增 runtime rounding | 原路徑不變；所試 upstream xi1 移動**必須新增取整**，不符合要求。 |
| 可消除 threshold parameters / runtime threshold subtract / comparator operand storage | **0 / 0 / 0**；這三項在現行 datapath 原本就不存在。 |
| Branch 是否可用 sign bit | **已經可以**：負分支 `q_in<0`、其餘含 0 走正分支。 |
| `xi1` / `xi2` | 336 個負分支 xi1 值與 336 個 output-offset xi2 值均保留；沒有宣稱其加法消失。 |
| 實體 area / power | **未量測，不宣稱改善**。 |

**結論：FAIL（requested threshold folding 不適用於此實際 QRPReLU 方程）。** 不把 no-op 的 `gamma=0` 包裝成最佳化，不改原 QRPReLU implementation；H2A-v2 與 H2B 原數值規格保持 frozen。沒有做 output offset folding、full official test、RTL 或 synthesis。
