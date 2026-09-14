# GAP → GMP 單一結構變因實驗報告

## 結論

GMP 讓 pooling 暫存器從 GAP 的 **signed INT16** 縮至 **signed INT10**，也不需 GAP 的 `+6` scale metadata；但以相同 R8B 類型訓練設定完成 100 epochs 後，GMP 軟體模型的官方 CIFAR-10 TEST 僅 **66.48%**，比既有 GAP R8B 軟體模型 **85.40% 低 18.92 pp**。按預定門檻屬 **Reject**；目前不建議投入 GMP-H0/H1MP/H2。沒有開始 RTL 或 synthesis，不能宣稱實際面積、功耗或時脈收益。

| 層級 | 模型／評估 | 官方 TEST | 比較 |
|---|---|---:|---:|
| 安全數值基線 | GAP R8B-H2A-v2 | 85.03% | 基線 |
| Phase A | 既有 GAP 權重直接換 GMP，凍結 H2A-v2 整數重播 | 23.95% | −61.08 pp vs 85.03% |
| 可比軟體基線 | GAP R8B signed-pow2 FC | 85.40% | 基線 |
| Phase B，訓練後 | GMP R8B-level，固定第 100 epoch | **66.48%** | **−18.92 pp vs 85.40%** |

Phase A 是 *GAP→GMP direct replacement* 的分布轉移測試，不能當成 GMP 架構最終訓練性能。Phase B 的 66.48% 是有實際經過 100 epochs 訓練的結果；尚未做 GMP 版 H0/H1MP/H2 量化，因此也不能把 66.48% 稱為 GMP 硬體整數推論準確率。

## 1. Git、安全與來源

執行前依序檢查了 `git status`、`git branch --show-current`、`git rev-parse HEAD`：三者都因 workspace **原本沒有 `.git`** 而失敗，因此原先不存在可記錄的 branch/HEAD 或 Git dirty-tree 狀態。為建立真正獨立的分支，先以本地 `main` 建立現有程式、設定、小型報告與結果檔的基線快照，再建立 `exp/gmp-ablation`。基線 commit 為 `34f8a86df2df0654ac2af56fdbd1189d4689af83`，提交身分 `Codex <codex@local.invalid>` 是因本機沒有設定 Git 作者而採用的本地封存身分，不代表既有歷史。沒有 reset、discard checkout、刪檔或 merge。

既有 `data/`、`logs/`、checkpoint、舊 `experiments/` 與 extension pack 均原位保留，不追蹤大型生成檔；新 GMP checkpoint 只在 `experiments/gmp_ablation/`。本次只修改 GMP 專用程式／測試與新報告／結果，沒有改動既有 R8B/H0/H1/H1MP/H2 來源檔、plan、checkpoint。執行後重驗下列 SHA-256，與 Phase A 記錄一致：

| 凍結來源 | SHA-256 |
|---|---|
| R8B best checkpoint | `c07ec3b29b88bbf70d3300f0bb19c7113abdadd305eb3d73cc12c7f40c2b0ba8` |
| R7 best checkpoint（Phase B 初始化） | `32b720c0dfd876fdabbf7d1004efe725bcfa08d35ebc66269a6c32120077b73b` |
| H0 結果 | `b608bc9d5325ee9cfa79802c3d8bf4904365c8bf49ac4ad82235357e57465ac3` |
| H1MP 結果 | `83da34d5c4f77643c2e9b2f69d6a8f478f70b2eb8673bbb37457d5e59efda82d` |
| H2A-v2 結果 | `13e7889628432bcdf2b3419503492e72c7604ba858993b4b9008746aec857f12` |

來源：`experiments/recu_r8b/recu_r8b_r7_pow2_fc_20260912_225827/best.pt`、`experiments/recu_r7/recu_r7_r6_head_bn_pow2_20260912_220758/best.pt`、`H0_INTEGER_BIAS_SWEEP_RESULTS.json`、`H1MP_SEARCH_RESULTS.json`、`H2_FINITE_WIDTH_RESULTS.json`。其他後續出現的未追蹤 `reference_inference/` 與 `tests/test_reference_inference.py` 不屬於本 GMP 實驗，未碰觸或納入 GMP 提交。

## 2. Phase A：凍結 H2A-v2 的 zero-shot 直接替換

在同一份正式 R8B checkpoint 上，GAP 與 GMP 整數重播都走相同 Thermometer R=8、W1A1 stem、ReCU blocks、H0 INT6 bias、H1MP 58-node bits/shifts/policies、exact binary-convolution accumulator 與 signed-pow2 head/FC；只以 64 個 Stage3 整數值的 `max` 取代 GAP `sum`。評估前後模型 state SHA-256 完全相同，沒有重新訓練。完整 10,000 張官方 TEST 重播 GAP **85.03%**，確認來源一致。

| Phase A 指標 | 結果 |
|---|---:|
| GMP zero-shot 官方 TEST | **23.95%** |
| Δ vs GAP H2A-v2 | **−61.08 pp** |
| GAP/GMP 預測不同張數 | 7,665 / 10,000 |
| logits 絕對差：平均／最大 | 7.63665 / 33.34375 |
| GAP head 輸入真值：min / max / mean | 0 / 2.32104 / 0.27501 |
| GMP head 輸入真值：min / max / mean | 0 / 7.875 / 1.79164 |

GMP head 輸入平均約為 GAP 的 6.5 倍；這是從數值 trace 推論出的嚴重分布轉移，不是對 GMP 從頭訓練可達準確率的估計。完整 GMP 輸出整數 histogram、range、logit 統計與來源 checksum 見 `experiments/gmp_ablation/zero_shot.json`。

## 3. GMP finite-width trace

H1MP Stage3 最終 tensor 是每張 `[64,8,8]`，亦即每 channel 64 個值。實測 H1MP 整數 representation 為 **signed INT10、shift 6**；完整 TEST 的 40,960,000 個輸入元素觀測 `q∈[-1,504]`。GMP 對每 channel 取最大值，產生 `[64]`；640,000 個 GMP 輸出觀測 `q∈[0,504]`、平均 `114.66468`、標準差 `71.20295`。

GMP 的 signed max register **INT10**、輸出 **INT10**、shift **6**。INT10 的數學範圍是 `[-512,511]`；`max` 只傳遞其中一個輸入，不可能產生超出輸入 signed width 的新值，因此不需位寬成長。此次 GMP max 節點 saturation **0**、H2 有限位寬 overflow **0**。這些數字只針對新增的 GMP 節點與 H2 overflow；既有凍結 H1MP boundary quantization 的 saturation 不應誤寫成零。

原 exact binary accumulator 維持：stem Cin96 **INT11**、Cin16 **INT9**、Cin32 **INT10**、Cin64 **INT11**。訓練後 GMP 模型未移植／重搜 H0/H1MP/H2；上列 finite-width trace 僅驗證 Phase A 凍結數值來源的 GMP reducer。

## 4. GAP 與 GMP 的硬體概念比較

| 性質 | H2A-v2 GAP | GMP |
|---|---|---|
| 定義 | `q_gap = Σ(i=0..63) q_i` | `q_gmp = max(i=0..63) q_i` |
| scale metadata | `s_gap = s_stage3 + 6 = 12` | `s_gmp = s_stage3 = 6` |
| 核心運算 | signed addition | signed comparison/select |
| 暫存器 | signed **INT16** sum accumulator | signed **INT10** max register |
| 第一元素之後的逐項運算 | 63 次加法 | 63 次比較／選擇 |
| 單 lane、每 cycle 一輸入的抽象排程 | 每 channel 64 次讀取，約 64 cycles | 每 channel 64 次讀取，約 64 cycles |
| 控制 | index/counter，無 runtime `/64` | index/counter，無 `/64` |
| 位寬成長 | INT10 → INT16 | INT10 → INT10 |

GAP 的 `/64` 是 **deferred scaling**：`q_gap=q_sum`、`s_gap=s_input+6`，沒有 arithmetic right shift 或新 rounding。GMP 既不 sum、也不除以 64、更不做 `+6`；其 scale 與 Stage3 相同。兩者下游仍是 signed-pow2 affine（位移、可選負號、整數 bias add）及 signed-pow2 FC，並未因 GMP 加入一般乘法器。上表 cycles 只是假設單一逐項 datapath 的操作級估算；尚無 RTL/synthesis，不能量化真實 area、power、frequency、gate count 或完整系統延遲。

## 5. Phase B：R8B-level GMP 訓練

方法是 **GMP fine-tuned from GAP R7 source, not scratch**：用既有 R7 best checkpoint（85.22%@99）建立 GAP R8B 對照初始化與 GMP R8B 初始化，兩者所有 state tensor 經逐項檢查完全相同；僅 `global average pooling` 改為 `global max pooling`，並按照 R8B 原 pipeline 把 FC 投影成 signed-pow2。訓練時不改 Thermometer、stem/backbone、Option-A shortcut、QRPReLU、signed-pow2 affine/FC 形式或凍結的 672 個 compatibility-alpha 參數。

| 設定 | 本次 GMP |
|---|---|
| Seed／epoch／batch | 123／100／128 |
| Optimizer | SGD，LR `1e-4`，momentum `0.9`，weight decay `0` |
| Schedule | `CosineAnnealingLR(T_max=100)` |
| ReCU tau | `0.99` |
| Augmentation | `RandomCrop(32,padding=4)`、`RandomHorizontalFlip`、`ToTensor`；無 Normalize |
| Train／validation | CIFAR-10 **TRAIN** 決定性、不重疊 45,000／5,000 張；只用 validation 選 checkpoint |
| Official TEST | 完成 100 epochs 並固定兩個 checkpoint 角色後才讀取；未用於模型選擇 |

| Phase B 指標 | 結果 |
|---|---:|
| Zero-epoch validation | 67.88% |
| 全部 epoch（含 0）最佳 validation / epoch / reload | 67.88% / 0 / 67.88% |
| 該 epoch 0 checkpoint 官方 TEST | 65.20%（**未訓練模型**） |
| **實際訓練 epoch 1–100 最佳 validation / epoch** | **67.68% / 100** |
| **固定最後 epoch 100 validation / 官方 TEST** | **67.68% / 66.48%** |
| Δ vs 可比 GAP R8B 軟體 85.40% | **−18.92 pp** |
| Δ vs 安全 GAP H2A-v2 85.03% | −18.55 pp（跨量化層級，僅參考） |

固定最終 checkpoint：`experiments/gmp_ablation/gmp_r8b_from_r7_20260914_231241/last.pt`，SHA-256 `6c0475a4ae072db99c288c6a7c12fdcc6a15d038c99c447cd467a0ea54d8bb62`。`best.pt` 是全局驗證最佳 epoch 0，不能把其 65.20% TEST 當作「訓練後」結果；訓練後最高 validation 剛好是最後的 epoch 100，因此固定最後模型同時也是最佳**已訓練**模型。兩個 checkpoint 角色事先固定，兩次 TEST 都在訓練結束後執行，不用 TEST 結果反選權重。完整曲線、split 與 metadata 在同一 run directory 的 `history.json`、`train_validation_split.json`、`setup.json`、`summary.json`。

比較限制：原歷史 GAP R8B 的 85.40% 使用全部 50,000 張 TRAIN，且當時每 epoch 用 TEST 選最佳；本 GMP run 為避免 TEST 選模，保留 TRAIN 的 5,000 張做 validation。因此 85.40% 是可比 R8B-level **外部參考**，不是在相同 45k/5k split 上重跑的嚴格配對 GAP control。不過 18.92 pp 的落差遠超預設 0.50 pp Reject 界線；不應將其解讀為細微隨機波動。

## 6. 模型複雜度與驗證

從實際 GMP 模型重新計算：**284,250 trainable parameters + 672 frozen compatibility-alpha parameters = 284,922 total state parameters**，與 GAP 同構模型完全相同。Binary convolution 仍是 stem **14,155,776**、backbone **40,108,032**，合計 **54,263,808 BMAC/image**；GMP 沒有新增 binary-convolution 參數或 BMAC，變化只在 pooling datapath。

新增 8 個 GMP 專項測試，涵蓋 signed max、64 元素/channel、全負數、正負混合、scale 不 `+6`、不誤做 GAP `/64`、`[N,64]`、`[N,10]` classifier。完整 `unittest discover -s tests -v` 為 **100/100 PASS**，包含原有 GAP/H0/H1/H2 測試。Phase A GAP 85.03% 完整 TEST reload 與 checkpoint/plan hash 均已確認。

## 7. 判定與停止點

預先門檻：相較可比 GAP，drop ≤0.10 pp 為 Excellent、≤0.30 pp 為 Acceptable、0.30–0.50 pp 為 Borderline、>0.50 pp 為 Reject。本次 GMP **−18.92 pp**，屬 **Reject**。GMP 確有 INT16→INT10 的 pooling 暫存器與 scale 簡化，但沒有綜合結果可證明硬體效益足以抵銷大幅準確率損失。依規格在此停止，不啟動 GMP-H0、GMP-H1MP、GMP-H2、RTL、synthesis 或其他實驗。
