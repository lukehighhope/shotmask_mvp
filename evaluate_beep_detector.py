"""
Evaluate beep detector on all *beep.txt ground truth.
Collects (video, gt_beep) from traning data (and any folder with *beep.txt),
extracts audio, runs detect_beeps, reports MAE and per-clip errors.

Usage:
  python evaluate_beep_detector.py
  python evaluate_beep_detector.py --folders "traning data/01032026" "traning data/outdoor-20260208T235429Z-1-001"
"""
import os
import sys
import subprocess
import tempfile

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_audio_plot import get_ffmpeg_cmd, get_fps_from_video, extract_audio
from detectors.beep import detect_beeps


def get_ffprobe_cmd():
    common_paths = [
        "ffprobe", "ffprobe.exe",
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
    ]
    for path in common_paths:
        if path in ("ffprobe", "ffprobe.exe") or os.path.exists(path):
            return path
    return None


def collect_beep_gt(folders=None):
    """
    Scan folders for *beep.txt; return list of (video_path, gt_beep_sec, label).
    Video naming: 1beep.txt -> 1.mp4; S1beep.txt -> S1-main.mp4.
    """
    if folders is None:
        folders = [
            os.path.join(os.path.dirname(__file__), "traning data", "01032026"),
            os.path.join(os.path.dirname(__file__), "traning data", "outdoor-20260208T235429Z-1-001"),
        ]
    root = os.path.dirname(__file__)
    entries = []
    for folder in folders:
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            continue
        # S1beep.txt -> S1-main.mp4
        if os.path.isfile(os.path.join(folder, "S1beep.txt")):
            for i in range(1, 9):
                base = f"S{i}"
                beep_txt = os.path.join(folder, base + "beep.txt")
                if not os.path.isfile(beep_txt):
                    continue
                vp = os.path.join(folder, base + "-main.mp4")
                if not os.path.isfile(vp):
                    continue
                try:
                    with open(beep_txt, "r", encoding="utf-8") as f:
                        line = f.readline().strip()
                    if line:
                        gt = float(line)
                        entries.append((vp, gt, f"{base}-main.mp4"))
                except Exception:
                    pass
        else:
            for i in range(1, 20):
                base = str(i)
                beep_txt = os.path.join(folder, base + "beep.txt")
                if not os.path.isfile(beep_txt):
                    continue
                vp = os.path.join(folder, base + ".mp4")
                if not os.path.isfile(vp):
                    continue
                try:
                    with open(beep_txt, "r", encoding="utf-8") as f:
                        line = f.readline().strip()
                    if line:
                        gt = float(line)
                        entries.append((vp, gt, f"{base}.mp4"))
                except Exception:
                    pass
    return entries


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate beep detector on *beep.txt GT")
    ap.add_argument("--folders", nargs="*", default=None,
                    help="Folders to scan (default: traning data/01032026 and outdoor)")
    ap.add_argument("--quiet", action="store_true", help="Only print MAE and summary")
    args = ap.parse_args()

    entries = collect_beep_gt(args.folders)
    if not entries:
        print("No *beep.txt + video pairs found.")
        return 1

    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("ffmpeg not found")
        return 1

    os.makedirs("tmp", exist_ok=True)
    results = []
    for video_path, gt, label in entries:
        fps = get_fps_from_video(video_path)
        if fps is None:
            fps = 30.0
        # Use a stable temp path per video so we can cache
        safe = label.replace(".mp4", "").replace("-", "_")
        audio_path = os.path.join("tmp", f"beep_eval_{safe}.wav")
        if not os.path.isfile(audio_path) or os.path.getmtime(audio_path) < os.path.getmtime(video_path):
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", video_path, "-ac", "1", "-ar", "48000", "-vn", audio_path],
                    check=True, capture_output=True
                )
            except Exception as e:
                if not args.quiet:
                    print(f"  {label}: extract audio failed: {e}")
                continue
        try:
            beeps = detect_beeps(audio_path, fps)
            pred = float(beeps[0]["t"]) if beeps else None
        except Exception as e:
            pred = None
            if not args.quiet:
                print(f"  {label}: detect_beeps failed: {e}")
        if pred is None:
            err = float("inf")
            results.append((label, gt, None, err))
            continue
        err = abs(pred - gt)
        results.append((label, gt, pred, err))

    if not results:
        print("No results.")
        return 1

    valid = [r for r in results if r[2] is not None]
    n_fail = len(results) - len(valid)
    if valid:
        mae = sum(r[3] for r in valid) / len(valid)
        max_err = max(r[3] for r in valid)
    else:
        mae = max_err = float("nan")

    if not args.quiet:
        print("Beep detector evaluation (GT from *beep.txt)")
        print("-" * 60)
        for label, gt, pred, err in results:
            if pred is not None:
                print(f"  {label:20s}  GT={gt:8.4f}s  pred={pred:8.4f}s  err={err:6.4f}s")
            else:
                print(f"  {label:20s}  GT={gt:8.4f}s  pred=FAIL  err=---")
        print("-" * 60)
    print(f"MAE = {mae:.4f}s   max_error = {max_err:.4f}s   failed = {n_fail}/{len(results)}")
    return 0


def tune_params():
    """Grid search over key params; print best config."""
    entries = collect_beep_gt()
    if not entries:
        return
    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        return
    import subprocess
    os.makedirs("tmp", exist_ok=True)
    # Pre-extract all audio
    for video_path, gt, label in entries:
        safe = label.replace(".mp4", "").replace("-", "_")
        audio_path = os.path.join("tmp", f"beep_eval_{safe}.wav")
        if not os.path.isfile(audio_path) or os.path.getmtime(audio_path) < os.path.getmtime(video_path):
            subprocess.run(
                [ffmpeg, "-y", "-i", video_path, "-ac", "1", "-ar", "48000", "-vn", audio_path],
                check=True, capture_output=True
            )
    from detectors.beep import detect_beeps, BEEP_CONFIG
    best_mae = float("inf")
    best_cfg = None
    for min_frac in (0.25, 0.35, 0.45, 0.55):
        for use_probe in (True, False):
            for late_after in (5.5, 6.5, 8.0):
                overrides = {"min_height_frac": min_frac}
                if not use_probe:
                    overrides["late_probe_after_s"] = 999.0  # disable
                else:
                    overrides["late_probe_after_s"] = late_after
                errs = []
                for video_path, gt, label in entries:
                    safe = label.replace(".mp4", "").replace("-", "_")
                    audio_path = os.path.join("tmp", f"beep_eval_{safe}.wav")
                    fps = get_fps_from_video(video_path) or 30.0
                    try:
                        beeps = detect_beeps(audio_path, fps, **overrides)
                        pred = float(beeps[0]["t"]) if beeps else None
                    except Exception:
                        pred = None
                    if pred is not None:
                        errs.append(abs(pred - gt))
                if errs:
                    mae = sum(errs) / len(errs)
                    if mae < best_mae:
                        best_mae = mae
                        best_cfg = dict(min_height_frac=min_frac, use_probe=use_probe, late_probe_after_s=late_after if use_probe else None)
    print("Best tune:", best_cfg, "MAE =", round(best_mae, 4))


if __name__ == "__main__":
    if "--tune" in sys.argv:
        sys.argv.remove("--tune")
        tune_params()
        sys.exit(0)
    sys.exit(main())
