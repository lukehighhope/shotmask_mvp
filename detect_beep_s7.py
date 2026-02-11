"""Detect beep near 4.29s and 8.5s for S7-main, save to S7beep.txt."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_logreg_multivideo import extract_audio, get_fps_duration
from detectors.beep import detect_beep_near

def main():
    folder = os.path.join(os.path.dirname(__file__), "traning data", "jeff 03-04")
    video_path = os.path.join(folder, "S7-main.mp4")
    beep_path = os.path.join(folder, "S7beep.txt")
    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        return 1
    audio_path = extract_audio(video_path)
    fps, _ = get_fps_duration(video_path)
    window_sec = 0.6
    t1 = detect_beep_near(audio_path, fps, 4.29, window_sec=window_sec)
    t2 = detect_beep_near(audio_path, fps, 8.5, window_sec=window_sec)
    if t1 is None:
        t1 = 4.29
        print("First beep: no peak near 4.29s, using 4.29")
    else:
        print(f"First beep near 4.29s: {t1:.4f}s")
    if t2 is None:
        t2 = 8.5
        print("Second beep: no peak near 8.5s, using 8.5")
    else:
        print(f"Second beep near 8.5s: {t2:.4f}s")
    with open(beep_path, "w", encoding="utf-8") as f:
        f.write(f"{t1:.4f}\n")
        f.write(f"{t2:.4f}\n")
    print(f"Saved: {beep_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
