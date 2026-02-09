"""
用 *beep.txt 里的 beep 时间为准，取 [beep_t, beep_t+0.3]s 窗口计算主频，列出表格。
假定 beep 持续 0.3s。只处理有 beep txt 的片段。

Usage:
  python beep_frequency_report.py
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from extract_audio_plot import get_ffmpeg_cmd, get_fps_from_video
from evaluate_beep_detector import collect_beep_gt


def dominant_frequency_hz(audio_path, start_s, duration_s=0.3, freq_low=200, freq_high=6000):
    """
    从 audio_path 中取 [start_s, start_s+duration_s] 段，计算主频 (Hz)。
    用 FFT 找 freq_low--freq_high Hz 内幅度最大的频率。
    """
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float64)
    n_start = int(start_s * sr)
    n_len = int(duration_s * sr)
    if n_start + n_len > len(data):
        n_len = len(data) - n_start
    if n_len <= 0:
        return None
    segment = data[n_start : n_start + n_len]
    # Hann window
    segment = segment * np.hanning(len(segment))
    n_fft = max(2048, 2 ** int(np.ceil(np.log2(len(segment)))))
    spec = np.abs(np.fft.rfft(segment, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    lo = np.searchsorted(freqs, freq_low)
    hi = np.searchsorted(freqs, freq_high)
    if hi <= lo:
        return None
    idx = lo + np.argmax(spec[lo:hi])
    return float(freqs[idx])


def main():
    # 只处理有 *beep.txt 的片段，beep 时间以 txt 为准
    entries = collect_beep_gt()
    if not entries:
        print("No *beep.txt entries found.")
        return 1

    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("ffmpeg not found")
        return 1

    os.makedirs("tmp", exist_ok=True)
    rows = []
    for video_path, beep_t_from_txt, label in entries:
        beep_t = float(beep_t_from_txt)
        safe = label.replace(".mp4", "").replace("-", "_").replace(" ", "_")
        audio_path = os.path.join("tmp", f"beep_eval_{safe}.wav")
        if not os.path.isfile(audio_path) or os.path.getmtime(audio_path) < os.path.getmtime(video_path):
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", video_path, "-ac", "1", "-ar", "48000", "-vn", audio_path],
                    check=True, capture_output=True
                )
            except Exception as e:
                print(f"  {label}: extract failed {e}")
                continue
        freq = dominant_frequency_hz(audio_path, beep_t, duration_s=0.3)
        rows.append((label, beep_t, freq))

    # 打印表格（beep 时间来自 *beep.txt，主频为 [beep_t, beep_t+0.3]s 窗口）
    print("Beep time from *beep.txt, 0.3s window -> dominant frequency")
    print("-" * 60)
    print(f"{'clip':<20} {'beep(s)':>10} {'freq(Hz)':>12}")
    print("-" * 60)
    for label, beep_t, freq in rows:
        if freq is not None:
            print(f"{label:<20} {beep_t:>10.4f} {freq:>12.1f}")
        else:
            print(f"{label:<20} {beep_t:>10.4f} {'(no peak)':>12}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
