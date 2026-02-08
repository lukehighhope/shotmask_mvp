# 提高枪声检测准确度的可选方案

当前问题：单视频、单次训练得到的 LogReg 准确度不够（误报多）。以下方案按**实现成本**从低到高排列，可组合使用。

---

## 一、低成本、可立即做

### 1. 多视频联合训练 LogReg（推荐先做）

**思路**：用多段带 ref 的视频（如 `01032026/1.mp4` … `5.mp4`）一起训练，让模型见到更多“真枪声”和“误报”，学出更稳的边界。

**做法**：
- 已提供脚本 `train_logreg_multivideo.py`：对多视频逐个取候选 + ref 标号，合并 (X, y, weights) 后训练一个 9 维 LogReg 并写回 `calibrated_detector_params.json`。
- 使用方式：`python train_logreg_multivideo.py --videos "01032026/1.mp4" "01032026/2.mp4" ...` 或 `--folder 01032026`。

**预期**：更多负样本（各视频里的 FP）可压低误报；更多正样本可稳住召回。

---

### 2. 提高默认阈值 / 严格过滤

**思路**：不改模型，只改“用模型的方式”，优先保证可接受的精确率。

**做法**：
- 在 `calibrated_detector_params.json` 里把 **min_confidence_threshold** 设为 **0.65～0.75**（你之前 grid-search 最佳约 0.7）。
- 或在主流程里默认加 **--shots-filter balanced** / **strict**（只保留运动确认或高置信度），减少误报。

**预期**：检测数会减少，但精确率明显提升；召回会略降。

---

### 3. 两阶段：候选 + 二次分类器

**思路**：当前 pipeline 做“高召回候选”，再单独训一个**只做二分类**的模型（gunshot vs not），对每个候选做一次判别，筛掉明显误报。

**做法**：
- 用现有候选特征（9 维 + 可加 spectral contrast、zero-crossing 等）标成 TP/FP（有 ref 的用 ref，无 ref 的可手标少量）。
- 用 **RandomForest / XGBoost** 训一个二分类器，保存成 pkl；在 `shot_audio` 里对每个候选先算 9 维，再调用该分类器，只有预测为 1 的才保留。
- 文献支持：用分类器做“验证阶段”可显著降 FP（“What makes audio event detection harder than classification”）。

**预期**：树模型对少量标注和混合特征通常比单层 LogReg 更稳，有利于进一步压 FP。

---

## 二、中等成本（需要一点新代码/依赖）

### 4. 用 YAMNet 做迁移学习（含枪声）

**思路**：用预训练音频事件模型（如 YAMNet）提 embedding，再在上面训一个小头（全连接层）做“枪声 vs 非枪声”。

**做法**：
- 依赖：TensorFlow、tensorflow_hub（或本地 yamnet）。
- 对每个候选切一短段（如 ±0.1s），用 YAMNet 提 embedding，标上 1/0（有 ref 的按时间标），训一个线性层或 2 层 MLP。
- 推理时：候选段 → YAMNet embedding → 小头 → P(gunshot)；可与现有 LogReg 做融合（例如两者都 > 0.5 才保留）。

**参考**：文献 “Enhancing Gun Detection With Transfer Learning and YAMNet” 报告枪型分类约 95% 准确率；我们只做二分类检测通常更容易。

---

### 5. 小 CNN 做谱图二分类（已实现）

**思路**：对每个候选截取短时窗，做 Mel 谱图，用 2D CNN 学习特征并二分类（枪声/非枪声），与 LogReg 置信度融合。

**做法**（已接入项目）：
- **模块** `detectors/shot_cnn.py`：Mel 谱图截取（librosa）、小 2D CNN 定义、加载/推理。
- **训练**：`python train_cnn_gunshot.py --folder 01032026 --epochs 30 --out outputs/cnn_gunshot.pt --save-config`  
  使用 1.txt～5.txt 真值标候选为正/负，训练 CNN；`--save-config` 会把 `cnn_gunshot_path` 写入 `calibrated_detector_params.json`。
- **推理**：主流程加载配置中的 `cnn_gunshot_path` 后，对每个最终候选截 Mel、跑 CNN 得到 P(枪声)，与现有置信度按 `cnn_fusion_weight`（默认 0.5）融合：`conf = (1-w)*logreg + w*cnn`。
- 依赖：`pip install torch`（已加入 requirements.txt）。

**预期**：由网络从谱图学习特征，跨场景泛化通常优于纯手工特征+LogReg。

---

## 三、较高成本 / 长期

### 6. 端到端声学事件检测模型（CRNN / Transformer）

**思路**：不用“先候选再分类”，而是直接做“帧级/段级”的枪声检测（类似 DCASE）。

**做法**：用 CRNN 或小 Transformer 做 SED，输出每帧/每段的 P(gunshot)，再后处理成事件列表。需要较多标注（段级或事件级）和训练时间。

### 7. 人工反馈 + 增量学习

**思路**：在波形/列表上让人标“这是误报”，把这些当负样本加入训练集，定期重新训练 LogReg 或第二级分类器。

**做法**：在 `extract_audio_plot` 或单独小工具里导出候选列表 + 时间；用户勾选 FP；下一版训练时把这些时间对应的候选特征标为 0，合并进现有 (X,y) 再训。

---

## 建议执行顺序

1. **先跑多视频联合训练**（`train_logreg_multivideo.py`），用 01032026 下多段视频，看 P/R/F1 是否明显改善。
2. 若仍不满意，**把 min_confidence_threshold 提到 0.7** 并视情况加 **--shots-filter balanced**，保证“可用”的精确率。
3. 若还有空间，**加第二级分类器**（RandomForest/XGBoost）或 **YAMNet 迁移**，进一步压误报。

如需我帮你写多视频训练脚本或第二级分类器接口，可以指定用哪几个视频和哪种分类器。
