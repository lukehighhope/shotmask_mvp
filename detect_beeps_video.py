#!/usr/bin/env python3
"""
通用 beep 探测脚本。
- 指定 --near 时：仅在这些时间附近探测（detect_beep_near），精确定位。
- 不指定 --near 时：全视频探测（先粗后精），输出所有通过验证的 beep；精探后会在该时刻再次检查主频 1.25–5.2kHz 与 CNN，过滤枪声等误检。
- 多 beep 时也可用 --near 指定近似时间做精探。
结果始终写入视频同目录的 *beep.txt。

用法:
  python detect_beeps_video.py video.mp4
  python detect_beeps_video.py video.mp4 --near 4.29 8.5
  python detect_beeps_video.py video.mp4 --min-gap 2 --rule
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_logreg_multivideo import extract_audio, get_fps_duration
from detectors.beep import detect_all_beeps, detect_beep_near


def _beep_txt_path(video_path):
    folder = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    if "-" in base and base.split("-")[0].upper().startswith("S"):
        key = base.split("-")[0]
    else:
        key = base
    return os.path.join(folder, f"{key}beep.txt")


def main():
    ap = argparse.ArgumentParser(
        description="Beep detection: near given times or full video. Writes *beep.txt."
    )
    ap.add_argument("video", help="Path to video file")
    ap.add_argument(
        "--near",
        type=float,
        nargs="+",
        metavar="SEC",
        help="Detect only near these times (e.g. --near 4.29 8.5). Omit for full-video detection.",
    )
    ap.add_argument(
        "--window",
        type=float,
        default=0.6,
        metavar="SEC",
        help="Half-window in sec when using --near (default: 0.6)",
    )
    ap.add_argument(
        "--min-gap",
        type=float,
        default=1.5,
        metavar="SEC",
        help="Min gap between beeps for full-video coarse merge (default: 1.5)",
    )
    ap.add_argument(
        "--refine-window",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Refine window half-width in sec for full-video (default: 0.5)",
    )
    ap.add_argument(
        "--rule",
        action="store_true",
        help="Full-video: use rule-based coarse instead of CNN+tonal",
    )
    args = ap.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"Not a file: {video_path}", file=sys.stderr)
        return 1

    print("Extracting audio and FPS...")
    audio_path = extract_audio(video_path)
    fps, duration = get_fps_duration(video_path)
    print(f"FPS={fps:.2f} duration={duration:.2f}s")

    beep_times = []  # list of float (seconds)

    if args.near is not None:
        # 指定时间：仅在附近探测
        print(f"Detecting beep(s) near {args.near}s (window ±{args.window}s)...")
        for t_approx in args.near:
            t_refined = detect_beep_near(
                audio_path, fps, t_approx_sec=t_approx, window_sec=args.window
            )
            if t_refined is not None:
                beep_times.append(t_refined)
                print(f"  near {t_approx}s -> {t_refined:.4f}s")
            else:
                beep_times.append(t_approx)
                print(f"  near {t_approx}s -> no peak, using {t_approx:.4f}s")
    else:
        # 未指定时间：全视频探测（先粗后精）
        print("Detecting beeps (full video: coarse then refine)...")
        beeps = detect_all_beeps(
            audio_path,
            fps,
            min_gap_s=args.min_gap,
            refine_window_s=args.refine_window,
            cnn_tonal_primary=not args.rule,
        )
        beep_times = [b["t"] for b in beeps]
        if beep_times:
            for i, t in enumerate(beep_times, 1):
                print(f"  {i}. {t:.4f}s")
        else:
            print("  No beeps detected.")

    out_path = _beep_txt_path(video_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in beep_times:
            f.write(f"{t:.4f}\n")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
