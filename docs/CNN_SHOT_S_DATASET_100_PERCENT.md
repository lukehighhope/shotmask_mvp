# 为什么 CNN 枪声检测在 S 数据集上接近 100% 正确？

## 原因：训练集与评估集重叠（Data Leakage）

当前 **CNN 枪声模型**（`cnn_gunshot.pt`）的训练脚本支持两种用法：

1. **单目录**：只用一个文件夹里的视频 + 参考时间  
   ```bash
   python train_cnn_gunshot.py --folder "traning data/01032026"
   ```
   此时训练数据只有 **1.mp4～5.mp4**（01032026）。

2. **递归多目录**：用 `traning data` 下**所有**含 .mp4 的子目录  
   ```bash
   python train_cnn_gunshot.py --folder "traning data" --recursive
   ```
   此时训练数据 = **01032026（1～5） + outdoor（S1～S8）**。

如果你曾经用 **`--recursive`** 训练过，那么 **S1-main.mp4～S8-main.mp4 已经参与训练**：  
每个 S 视频的「候选时刻 + 是否在 ref 0.04s 内」被标成 0/1，用来训练 CNN。  
在 S 上评估时，相当于在**训练集**或**同分布数据**上测，所以会出现接近甚至 100% 的匹配率，这是**数据泄露**导致的乐观估计，不能代表在「从未见过的视频」上的表现。

## 如何得到更真实的泛化表现？

- **方案 A：训练时不用 S**  
  只用室内数据训练，在 S 上做测试：  
  ```bash
  python train_cnn_gunshot.py --folder "traning data/01032026" --epochs 50 --out outputs/cnn_gunshot.pt
  ```  
  然后用当前流程在 S1～S8 上跑 shot 检测，看与 S1.txt～S8.txt 的匹配率。

- **方案 B：S 上做留一法**  
  训练时：01032026 + S 里 7 个（例如去掉 S3）；测试时：只在 S3 上评估。  
  需要改 `train_cnn_gunshot.py` 支持 `--exclude-folder` 或类似参数，或在脚本里写死排除某一个 S 子目录。

- **方案 C：严格区分 train / val / test**  
  把 S 固定为 **test**，只在不含 S 的数据上训练和调参（包括 01032026 再拆成 train/val）。

## 小结

| 训练命令 | S 是否参与训练 | 在 S 上“100%”是否可信 |
|----------|----------------|------------------------|
| `--folder "traning data/01032026"` | 否 | 是，可视为泛化表现 |
| `--folder "traning data" --recursive` | 是 | 否，属于训练集上的表现 |

若你希望「S 数据集上的数字」代表**真实泛化能力**，请用**未包含 S 的数据**重新训练 CNN（例如仅 01032026），再在 S 上评估。
