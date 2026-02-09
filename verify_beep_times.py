"""
用约略 beep 时间在附近窗口重新探测，验证并更新真实 beep 时间。
读取 folder 下的 beep_overrides.json（约略时间），对每个视频在 t±window 内找 beep 峰，写回 verified 时间。

Usage:
  python verify_beep_times.py --folder "traning data/outdoor-20260208T235429Z-1-001"
  python verify_beep_times.py --folder "traning data/outdoor-20260208T235429Z-1-001" --window 1.5
"""
import argparse
import json
import os

from detectors.beep import detect_beep_near
from train_logreg_multivideo import extract_audio, get_fps_duration


def main():
    ap = argparse.ArgumentParser(description="Verify beep times from approximate values in beep_overrides.json")
    ap.add_argument("--folder", required=True, help="Folder containing beep_overrides.json and S1-main.mp4 etc.")
    ap.add_argument("--window", type=float, default=2.0, help="Search window ± seconds around approximate time")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    override_path = os.path.join(folder, "beep_overrides.json")
    if not os.path.isfile(override_path):
        print(f"No beep_overrides.json in {folder}")
        return 1
    with open(override_path, "r", encoding="utf-8") as f:
        overrides = json.load(f)

    updated = {}
    for key in sorted(overrides.keys()):
        t_approx = float(overrides[key])
        vp = None
        for name in os.listdir(folder):
            if not name.lower().endswith(".mp4"):
                continue
            if name.startswith(key + "-") or name == key + ".mp4":
                vp = os.path.join(folder, name)
                break
        if not vp or not os.path.isfile(vp):
            print(f"  {key}: no video, keep approx {t_approx}s")
            updated[key] = t_approx
            continue
        try:
            audio_path = extract_audio(vp)
            fps, _ = get_fps_duration(vp)
        except Exception as e:
            print(f"  {key}: extract error {e}, keep approx {t_approx}s")
            updated[key] = t_approx
            continue
        t_verified = detect_beep_near(audio_path, fps, t_approx, window_sec=args.window)
        if t_verified is not None:
            updated[key] = t_verified
            print(f"  {key}: approx {t_approx}s -> verified {t_verified}s")
        else:
            updated[key] = t_approx
            print(f"  {key}: no peak near {t_approx}s, keep approx {t_approx}s")

    with open(override_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    print(f"\nUpdated {override_path} with verified beep times.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
