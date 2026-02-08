"""Diagnose why beep is not detected in 5.mp4. Run: python debug_beep_5.py"""
import os
import sys
import subprocess

# extract audio for 5.mp4
video = os.path.join(os.path.dirname(__file__), "01032026", "5.mp4")
if not os.path.isfile(video):
    print("5.mp4 not found")
    sys.exit(1)
out_dir = "tmp"
os.makedirs(out_dir, exist_ok=True)
base = "5"
wav = os.path.join(out_dir, f"audio_{base}.wav")
ffmpeg = "ffmpeg"
for p in [r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe", r"C:\ffmpeg\bin\ffmpeg.exe"]:
    if os.path.exists(p):
        ffmpeg = p
        break
subprocess.run([ffmpeg, "-y", "-i", video, "-ac", "1", "-ar", "48000", "-vn", wav], check=True, capture_output=True)
print(f"Extracted: {wav}")

import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter, find_peaks

def bandpass(data, sr, low=1500, high=4000):
    b, a = butter(4, [low/(sr/2), high/(sr/2)], btype='band')
    return lfilter(b, a, data)

data, sr = sf.read(wav)
if data.ndim > 1:
    data = data.mean(axis=1)
duration = len(data) / sr
print(f"Duration: {duration:.2f}s, sr={sr}")

filtered = bandpass(data, sr)
energy = np.abs(filtered)
win = max(1, int(sr * 0.01))
smooth = np.convolve(energy, np.ones(win) / win, mode="same")

# Stats in first 3s
n_3s = int(3 * sr)
smooth_3 = smooth[:n_3s]
mean_s = float(np.mean(smooth_3))
std_s = float(np.std(smooth_3))
max_s = float(np.max(smooth_3))
threshold_8 = mean_s + 8 * std_s
threshold_6 = mean_s + 6 * std_s
threshold_4 = mean_s + 4 * std_s
print(f"First 3s: mean={mean_s:.4f} std={std_s:.4f} max={max_s:.4f}")
print(f"Threshold (8*std): {threshold_8:.4f}  (6*std): {threshold_6:.4f}  (4*std): {threshold_4:.4f}")
print(f"Max above 8*std? {max_s >= threshold_8}  6*std? {max_s >= threshold_6}  4*std? {max_s >= threshold_4}")

for name, th in [("8*std", threshold_8), ("6*std", threshold_6), ("4*std", threshold_4)]:
    peaks, _ = find_peaks(smooth, height=th, distance=int(sr * 0.2))
    in_window = [p for p in peaks if (p / sr) <= 3.0]
    times = [p / sr for p in (in_window if in_window else peaks)[:5]]
    print(f"  {name}: {len(peaks)} peaks total, {len(in_window)} in 0-3s, first times (s): {[round(t,3) for t in times]}")

# Current detector
from detectors.beep import detect_beeps
fps = 30.0  # dummy for this check
beeps = detect_beeps(wav, fps)
print(f"detect_beeps result: {beeps}")
