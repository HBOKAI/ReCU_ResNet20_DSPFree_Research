# 無腦跑順序

## 0. 放到你現有專案

建議把這包內容 merge 到目前 workspace。
不要刪掉你之前的 FP32 / Vanilla baseline。

## 1. 安裝

```bash
python -m pip install -r requirements.txt
```

## 2. 靜態驗證

```bash
python tools/report_recu.py --config configs/recu_official.json
python -m unittest discover -s tests -v
```

官方 ReCU 模型參數應看到：

```text
total = 270858
alpha = 672
prelu = 336
```

## 3. Smoke

```bash
python train_recu.py --config configs/recu_official.json --smoke
```

## 4. 先跑 20 epochs diagnostic

```bash
python train_recu.py --config configs/recu_official.json --diagnostic-epochs 20
```

如果 curve 明顯異常，先不要燒 600 epochs。

## 5. 正式官方 ReCU

```bash
python train_recu.py --config configs/recu_official.json
```

目標不是強迫剛好 87.5%，但應該接近官方 87–88% 區間。
若 <85%，先查 reproduction，不要直接進硬體改造。

## 6. 官方 ReCU 成功後，再做 QRPReLU

```bash
python train_recu.py --config configs/recu_qrprelu.json
```

## 7. 再做 QRPReLU + power-of-two alpha

```bash
python train_recu.py --config configs/recu_qrprelu_pow2alpha.json
```

這一步開始才是你的 DSP-free / multiplier-free ablation。
