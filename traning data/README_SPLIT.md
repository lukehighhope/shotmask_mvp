# Training / Validation 划分

划分规则在 `dataset_split.json` 中配置。

## 当前规则：每个文件夹最后一个视频为 Val

- **split_type**: `last_video_per_folder`
- **folders**: 留空 `[]` 表示**自动发现**：`traning data` 下所有含有 `.mp4` 的子文件夹都参与，**新增 data 文件夹会自动按同一规则划分**。若需只使用部分文件夹，可在 `folders` 里写明子文件夹名列表。

在每个文件夹内，按**文件名排序**，**最后一个** `.mp4` 作为 **validation**，其余作为 **training**。

| 文件夹 | Train 视频 | Val 视频 |
|--------|------------|----------|
| 01032026 | 1.mp4, 2.mp4, 3.mp4, 4.mp4 | **5.mp4** |
| jeff 03-04 | S1-main.mp4 … S7-main.mp4 | **S8-main.mp4** |
| jeff 05-06 | S1-main.mp4 … S7-main.mp4 | **S8-main.mp4** |
| outdoor-... | S1-main.mp4 … S7-main.mp4 | **S8-main.mp4** |

## 使用方式（脚本）

```python
from dataset_split import get_train_video_paths, get_val_video_paths

train_paths = get_train_video_paths()  # 所有“非最后一个”视频的绝对路径
val_paths   = get_val_video_paths()   # 每个文件夹最后一个视频的绝对路径
```

- 训练时只用 `train_paths` 里的视频。
- 验证/早停在 `val_paths` 上计算指标。

修改划分：编辑 `dataset_split.json`（例如改 `split_type` 或 `folders` 列表）。
