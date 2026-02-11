#!/usr/bin/env python3
"""
对 S1–S8 做枪声探测并生成三个图：1) waveform PNG  2) envelope+视频 viewer  3) waveform+video+calibration viewer。
Beep 来自各视频同目录的 *beep.txt；若有 2 个及以上 beep，split 起点采用最后一个 beep。

Usage: python run_s1_s8_shots_and_plots.py
       python run_s1_s8_shots_and_plots.py --folder "traning data/jeff 03-04"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_audio_plot import main as extract_main

def main():
    import argparse
    ap = argparse.ArgumentParser(description="S1-S8 gunshot detection + 3 figures per video (waveform, envelope viewer, calibration viewer)")
    ap.add_argument("--folder", default=os.path.join(os.path.dirname(__file__), "traning data", "jeff 03-04"),
                    help="Folder containing S1-main.mp4 ... S8-main.mp4 and S1beep.txt ... S8beep.txt")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1
    videos = []
    for i in range(1, 9):
        v = os.path.join(folder, f"S{i}-main.mp4")
        if os.path.isfile(v):
            videos.append(v)
    if not videos:
        print(f"No S1-main.mp4 .. S8-main.mp4 found in {folder}", file=sys.stderr)
        return 1
    print(f"Processing {len(videos)} videos (gunshot detection + 3 figures each)...\n")
    for vp in videos:
        name = os.path.basename(vp)
        print(f"--- {name} ---")
        try:
            extract_main(
                vp,
                output_image=None,
                show_plot=False,
                use_ref=False,
                use_calibrate=False,
                use_train_logreg=False,
                no_viewers=False,
                use_calibration_viewer=False,
            )
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
        print()
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
