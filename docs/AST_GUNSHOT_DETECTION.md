# AST-style 枪声检测（Transformer）

## 概述

在 `detectors/shot_ast.py` 中实现了一个 **AST（Audio Spectrogram Transformer）风格** 的模型：把 Mel 谱图切成 patch，用 Transformer 编码，再用 CLS token 做二分类（枪声 / 非枪声）。输入与现有 CNN 一致：以候选时刻为中心的 0.35s 片段的 64×32 Mel 谱图，可直接复用 CNN 的数据与流程。

## 结构

- **Patch**：4×4 → 64×32 得到 16×8 = 128 个 patch
- **Embedding**：Conv2d(patch) + CLS token + 可学习位置编码
- **Transformer**：默认 4 层、4 头、embed_dim=192、mlp_ratio=2
- **Head**：Linear(embed_dim, 1)，输出单 logit，sigmoid 为 P(枪声)

## 训练

```bash
# 仅室内数据
python train_ast_gunshot.py --folder "traning data/01032026" --epochs 40 --augment --out outputs/ast_gunshot.pt

# 多目录（与 CNN 相同数据）
python train_ast_gunshot.py --folder "traning data" --recursive --epochs 30 --augment --out outputs/ast_gunshot.pt --save-config
```

`--save-config` 会把 `ast_gunshot_path` 写入 `calibrated_detector_params.json`。

**在 pipeline 里用 AST 打分**：在 `calibrated_detector_params.json` 中设置 `ast_gunshot_path` 为训练得到的 `.pt` 路径，并设 `use_cnn_only: true`。若同时存在 `cnn_gunshot_path`，需设 `use_ast_gunshot: true` 才会用 AST 打分；若只配置了 `ast_gunshot_path` 未配 CNN，则自动用 AST。

## 与 CNN 的对比

|        | CNN (shot_cnn)     | AST (shot_ast)        |
|--------|--------------------|------------------------|
| 输入   | (1, 64, 32) mel    | 同左                   |
| 参数量 | 较小               | 较大（Transformer）    |
| 数据需求 | 中等             | 建议更多或加增强       |
| 推理   | 略快               | 略慢                   |

## 使用预训练 AST（如 Hugging Face）

若要使用 **Hugging Face 上的 AST**（例如 AudioSet 预训练），需单独接一版：

- 输入多为 128 mel × 1024 帧（约 10s @ 10fps），与当前 64×32 短片段不同。
- 做法：要么把片段拉长并 resample 成 128×128 再微调；要么只用预训练 AST 做特征提取，再接一个小头做二分类（需改 `shot_ast.py` 和训练脚本）。

当前仓库内实现的是 **轻量 ViT on 64×32**，不依赖 HF，与现有 pipeline 即插即用。
