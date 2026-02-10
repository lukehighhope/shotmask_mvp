# 数据命名规则 (Naming Convention)

## 路径约定

探测某个视频时，**视频路径会在入口处被规范为绝对路径**（`os.path.abspath(os.path.normpath(video))`），并检查文件存在。  
因此 **ref/beep 的查找目录 = 该视频所在目录**，与当前工作目录 (cwd) 无关；用相对路径传入时，会先按 cwd 解析成绝对路径再使用。  
**输出**（波形图、viewer HTML）默认也写在**视频所在目录**（如 `2.mp4` → `2_waveform.png`、`2_waveform_viewer_envelope.html`），便于按视频整理；可用 `--output` 指定到其他路径。

同一目录下，与主视频配套的文件按以下规则命名：

| 文件模式 | 含义 | 说明 |
|----------|------|------|
| **\*.mp4** | 主射击视频 | 拍摄的射击过程视频。 |
| **\*.txt** | Split 时刻数据 | 从 **beep 为起点** 的分段间隔（秒），一行一个数：第 1 个 = beep→第 1 枪，第 2 个 = 第 1 枪→第 2 枪，依此类推。参考枪声时刻 = `beep_t + cumsum(splits)`。**仅用同名 .txt，不先检测同名 .jpg。** |
| **\*beep.txt** | Beep 时刻 | Beep 声**相对于视频起点**的时刻（秒），单行一个数字。**仅用同名 *beep.txt**（不先检测 .jpg）。例如 `2beep.txt` 内容为 `3.4276` 表示 beep 在视频第 3.43 秒。 |

## 示例

- `2.mp4`：视频  
- `2.txt`：18 个 split（beep→1 枪、1→2、…），用于得到 18 个参考枪声时刻  
- `2beep.txt`：一行 `3.4276`，表示 beep 在 2.mp4 的 3.43 秒处  

代码中 ref 计算：`ref_times = beep_t + np.cumsum(parse(2.txt))`，其中 `beep_t` 来自 `2beep.txt` 或 `beep_overrides.json` 或算法检测。
