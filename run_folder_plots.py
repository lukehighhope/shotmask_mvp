#!/usr/bin/env python3
"""
对指定文件夹内所有 .mp4 做枪声探测并生成三个图：
1) waveform PNG  2) envelope+视频 viewer  3) waveform+video+calibration viewer（不启动服务器）

Usage: python run_folder_plots.py --folder "test data/indoor"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_audio_plot import main as extract_main


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Gunshot detection + 3 figures per video in folder")
    ap.add_argument("--folder", required=True, help="Folder containing .mp4 files")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1
    videos = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    )
    if not videos:
        print(f"No .mp4 found in {folder}", file=sys.stderr)
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
                use_viewer=True,
                use_calibration=True,
                start_calibration_server=False,
            )
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
        print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
