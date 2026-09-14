# R1 / R2 / R3 執行順序

先找已成功的 Official ReCU 87.28% `best.pt`。

以下用：

`OFFICIAL_BEST=/path/to/recu_resnet20_official_repro.../best.pt`

表示。

## 0. 驗證轉換

```bash
python tools/verify_ablation_conversion.py --checkpoint "$OFFICIAL_BEST"
```

R3 的 float fold 必須先通過：
- max abs logit error <= 1e-4

否則禁止跑 R3。

## 1. R1 Remove alpha

Smoke：

```bash
python train_recu_ablation.py \
  --config configs/recu_r1_remove_alpha.json \
  --source-checkpoint "$OFFICIAL_BEST" \
  --smoke
```

20 epoch diagnostic：

```bash
python train_recu_ablation.py \
  --config configs/recu_r1_remove_alpha.json \
  --source-checkpoint "$OFFICIAL_BEST" \
  --diagnostic-epochs 20
```

正式 fine-tune：

```bash
python train_recu_ablation.py \
  --config configs/recu_r1_remove_alpha.json \
  --source-checkpoint "$OFFICIAL_BEST"
```

## 2. R2 Warm-start QRPReLU

```bash
python train_recu_ablation.py \
  --config configs/recu_r2_warmstart_qrprelu.json \
  --source-checkpoint "$OFFICIAL_BEST" \
  --smoke
```

```bash
python train_recu_ablation.py \
  --config configs/recu_r2_warmstart_qrprelu.json \
  --source-checkpoint "$OFFICIAL_BEST" \
  --diagnostic-epochs 20
```

```bash
python train_recu_ablation.py \
  --config configs/recu_r2_warmstart_qrprelu.json \
  --source-checkpoint "$OFFICIAL_BEST"
```

## 3. R3 Fused alpha+BN -> signed power-of-two affine

R3 不再量 alpha 本身。

先把官方：

`BN(alpha*S)`

依 running stats fold 成：

`K*S+B`

再量：

`K -> sign(K)*2^round(log2|K|)`

正式：

```bash
python train_recu_ablation.py \
  --config configs/recu_r3_fused_pow2_affine.json \
  --source-checkpoint "$OFFICIAL_BEST"
```

R3 中：
- binary branch BN 已被 deployment-oriented fused affine 取代
- stem BN 與 head BN 這一輪仍保留
- PReLU 保持官方版本
- first Conv / final FC 仍保持官方版本

## Fine-tune recipe

三個實驗預設：
- 100 epochs
- SGD
- lr 0.01
- momentum 0.9
- weight decay 5e-4
- cosine
- tau 固定 0.99

原因：
這三個都是從已收斂的 Official ReCU 600-epoch checkpoint 做硬體化轉換，
不應重新把 tau 拉回 0.85 或從頭訓練。
