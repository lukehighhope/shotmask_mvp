# 只用 CNN 是不是更好？

## 结论（基于 01032026 五段视频评估）

在当前数据和同一阈值（min_confidence_threshold=0.26）下：

| 模式 | 精确率 P | 召回率 R | F1 |
|------|----------|----------|-----|
| **LogReg+CNN**（默认） | 27.5% | 55.1% | **36.7%** |
| **CNN-only** | **49.6%** | 50.0% | **49.8%** |

- **CNN-only 更好**：F1 更高（49.8% vs 36.7%），误报更少（P 49.6% vs 27.5%），召回略降（50% vs 55%）。
- **1.mp4**：CNN-only 可达 **F1 89.2%、召回 100%**；LogReg+CNN 为 57.1% / 96.6%。

## 怎么用

- **配置**：在 `calibrated_detector_params.json` 里加 `"use_cnn_only": true`，则主流程只用 CNN 打分（不用 LogReg）。
- **评估**：  
  - 默认（LogReg+CNN）：`python evaluate_multivideo.py --folder 01032026`  
  - 只用 CNN：`python evaluate_multivideo.py --folder 01032026 --cnn-only`

## 原理区别

- **LogReg+CNN**：Stage1 候选 → LogReg 用 9 维手工特征打分 → 聚类/筛选 → 再对剩余 shot 用 CNN 做融合（conf = (1-w)*logreg + w*cnn）。
- **CNN-only**：Stage1 候选 → **每个候选**用 CNN(Mel 谱) 直接得到 P(枪声) 作为分数 → 聚类/筛选，不再用 LogReg。

CNN-only 依赖 CNN 在 Mel 谱上学到的模式，同一批训练数据下在本数据集上表现更好；若换到差异很大的场景，可再对比两种模式。

## 如何提高 CNN 准确度 / 继续训练

1. **多训几轮**：`--epochs 50` 或 `60`（默认 30）。训练里已加**学习率衰减**（loss 不降则 lr 减半），多训一般更稳。
2. **数据增强**：加 `--augment`，对 Mel 做随机噪声和时移，减轻过拟合，推荐：  
   `python train_cnn_gunshot.py --folder 01032026 --epochs 50 --augment --out outputs/cnn_gunshot.pt --save-config`
3. **在已有模型上继续训**：`--resume` 从当前 `--out` 读权重，再训 `--epochs` 轮：  
   `python train_cnn_gunshot.py --folder 01032026 --epochs 20 --resume --out outputs/cnn_gunshot.pt`
4. **调阈值**：训完后在 `calibrated_detector_params.json` 里调 `min_confidence_threshold`（如 0.2～0.35），用 `evaluate_multivideo.py --cnn-only` 看 P/R 变化。
