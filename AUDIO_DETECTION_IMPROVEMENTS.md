# 音频检测算法改进说明

## 改进概述

根据你提供的建议，我已经重写了音频检测算法，实现了**两阶段多特征融合**的方法，解决了原算法的5个主要问题。

## 主要改进点

### 1. ✅ 多频段并行特征（替代固定带通）

**原问题**: 固定 400-6000Hz 带通可能削弱真正的 shot（压缩/设备差异）

**改进**: 
- 使用**多频段能量比**：E_low (50-300Hz), E_mid (300-3000Hz), E_high (3000-9000Hz)
- 计算比值特征：`r1 = E_mid / E_low`（枪声特征），`r2 = E_high / E_mid`（金属/尖叫惩罚）
- 不再依赖单一带通，而是通过频段比值识别枪声

### 2. ✅ Onset/Flux 候选检测（替代纯能量峰值）

**原问题**: 能量峰值会把关门、拍手、碰撞、喊叫都当 shot

**改进**:
- **Stage 1**: 使用 **Spectral Flux（频谱突变）** 找候选点
- STFT 参数：n_fft=1024（~21ms），hop=256（~5.3ms）
- Flux 对瞬态事件更敏感，对长时间噪声更鲁棒
- 高召回率：尽量不漏检

### 3. ✅ 滑动窗自适应阈值（替代全局百分位）

**原问题**: 92% 百分位在"枪声很少"的视频里会偏高，导致漏检

**改进**:
- 使用 **MAD（Median Absolute Deviation）** + 滑动窗
- 窗口大小：2秒
- 公式：`threshold = median + k * MAD * 1.4826`（k=6）
- 阈值随环境噪声自适应调整，不会"前半段漏、后半段乱报"

### 4. ✅ 事件聚类去重（替代硬 distance）

**原问题**: `distance=0.12s` 过于武断，会误杀连发，也会错配 peak

**改进**:
- **聚类窗口**: 250ms 内的 peaks 视为同一簇（回声/反射/AGC回弹）
- 每簇选 **score 最高者**输出
- 速率限制只做"异常保护"（>6 shot/s 才截断），不作为核心逻辑

### 5. ✅ 多特征软融合打分（替代 height/prominence）

**原问题**: 置信度只基于 height/prominence，"谁大谁赢"，但真实 shot 不一定最大（AGC压缩）

**改进**:
- **多特征打分**:
  ```
  score = 0.35 * onset_norm +      # 频谱突变强度（主要）
          0.25 * r1_norm +         # 中/低频比（枪声特征）
          0.15 * flatness_norm +   # 谱平坦度（噪声型）
          0.15 * attack_norm -     # 上升沿斜率（AGC鲁棒）
          0.10 * r2_penalty        # 高频惩罚（金属/尖叫）
  ```
- 能量只占一部分，不再"谁大谁赢"

## 算法流程

### Stage 1: 高召回候选检测
1. 计算 Spectral Flux（频谱突变）
2. 平滑处理（10ms窗口）
3. 滑动窗 MAD 自适应阈值
4. `find_peaks` 找候选点（最小间隔50ms，高召回）

### Stage 2: 多特征验证与打分
对每个候选点：
1. 提取窗口特征（[-25ms, +40ms]）
2. 计算多频段能量比（r1, r2）
3. 计算谱特征（flatness, centroid）
4. 计算攻击斜率（attack slope，AGC鲁棒）
5. 多特征融合打分
6. 归一化置信度

### Stage 3: 聚类去重
1. 按时间聚类（250ms窗口）
2. 每簇选最高分
3. 异常保护（>6 shot/s 才限制）

## 使用方法

### 启用改进算法

当前已默认启用改进算法。参数文件 `calibrated_detector_params.json`:

```json
{
  "use_improved": true,
  "threshold_percentile": null,    // null = 使用 MAD 自适应阈值
  "cluster_window_sec": 0.25,      // 聚类窗口（替代 min_dist_sec）
  "export_diagnostics": false      // 是否导出诊断曲线
}
```

### 导出诊断曲线（推荐）

启用诊断导出，查看算法中间结果：

```json
{
  "use_improved": true,
  "export_diagnostics": true
}
```

然后运行：
```bash
python extract_audio_plot.py --video 1.mp4 --no-show
```

会生成 `outputs/audio_diagnostics.json`，包含：
- `envelope`: 包络曲线
- `flux`: 频谱突变曲线
- `scores`: 最终打分曲线
- `shot_times`: 检测到的 shot 时间

**诊断方法**:
1. 把参考 shot 时间画成竖线
2. 对比三条曲线：
   - **漏检**: flux 有响应但 threshold 太高？还是特征压根没响应（带通/AGC问题）？
   - **误报**: 哪类事件的 score 被打高了？查看 r1/r2 比值能帮你区分

### 回退到旧算法

如果需要回退：
```json
{
  "use_improved": false,
  "threshold_percentile": 92,
  "min_dist_sec": 0.12,
  "prominence_frac": 0.4
}
```

## 参数调优建议

### 如果漏检多：
1. 降低 MAD 阈值：`k` 从 6.0 降到 4.0-5.0（在代码中修改）
2. 或使用 percentile 回退：`threshold_percentile: 85-90`
3. 检查诊断曲线：flux 是否有响应？

### 如果误报多：
1. 增加聚类窗口：`cluster_window_sec: 0.30`
2. 调整打分权重：降低 `onset_norm` 权重，增加 `r1_norm` 权重
3. 检查诊断曲线：误报的 r1/r2 比值特征是什么？

## 性能优化

- STFT 计算已优化：共享一次 STFT，避免重复计算
- 如果仍慢，可以：
  1. 降低 `n_fft`（1024 → 512，但精度下降）
  2. 增加 `hop_length`（256 → 512，但时间分辨率下降）
  3. 或使用 librosa（如果已安装，会自动使用，更快）

## 下一步

1. **测试改进算法**：运行 `python extract_audio_plot.py --video 1.mp4 --no-show`
2. **查看结果**：对比检测数量和匹配率
3. **启用诊断**：设置 `export_diagnostics: true`，分析中间曲线
4. **微调参数**：根据诊断结果调整阈值或权重
