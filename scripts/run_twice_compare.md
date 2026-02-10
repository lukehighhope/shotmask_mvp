# 检查两次运行是否一致

在项目根目录执行（PowerShell）：

```powershell
cd c:\Users\Cao\Downloads\shotmask_mvp

# 第一次
python extract_audio_plot.py --video "traning data/01032026/2.mp4" --no-show 2>&1 | Tee-Object -FilePath run1.txt

# 第二次（不改任何东西）
python extract_audio_plot.py --video "traning data/01032026/2.mp4" --no-show 2>&1 | Tee-Object -FilePath run2.txt

# 对比关键行
Select-String -Path run1.txt, run2.txt -Pattern "Ref|Detector|Shot time|Energy-detected|Matched"
```

若 run1 和 run2 里上述几行完全一致，说明**同一命令、同一目录下结果是可复现的**，之前的“差别很大”就来自：  
- 不同目录 / 不同 `--video` 路径，或  
- 不同配置（如曾改过 `calibrated_detector_params.json` 或 ref 文件）。
