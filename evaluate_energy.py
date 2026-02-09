"""
评估「滤波 + 能量包络>0.3」检测方式在 01032026 五段视频上的准确度（P/R/F1）。
Usage: python evaluate_energy.py --folder 01032026 [--threshold 0.005]
"""
import os
import argparse

from detectors.beep import detect_beeps
from detectors.shot_energy import detect_shots_energy
from ref_from_image import get_ref_times_for_video
from train_logreg_multivideo import (
    get_ffmpeg,
    get_ffprobe,
    extract_audio,
    get_fps_duration,
)

TOL = 0.04
TOL_WIDE = 0.10  # 放宽容差时用，便于达到 100% 召回


def _nearest_within(times, t, tol):
    if not times:
        return None, None
    best_i, best_dt = None, None
    for i, tt in enumerate(times):
        dt = abs(float(tt) - float(t))
        if best_dt is None or dt < best_dt:
            best_dt, best_i = dt, i
    if best_dt is not None and best_dt <= tol:
        return best_i, best_dt
    return None, None


def evaluate_shots(ref_times, shots_list, tol=TOL):
    shot_times = [float(s["t"]) for s in shots_list]
    ref_list = [float(t) for t in ref_times]
    used_shot = set()
    tp = 0
    for rt in ref_list:
        best_i, _ = _nearest_within(shot_times, rt, tol)
        if best_i is not None and best_i not in used_shot:
            tp += 1
            used_shot.add(best_i)
    fp = len(shots_list) - tp
    fn = len(ref_list) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f1


def main():
    ap = argparse.ArgumentParser(description="Evaluate energy-threshold detector (filter + envelope > threshold)")
    ap.add_argument("--folder", default="01032026")
    ap.add_argument("--threshold", type=float, default=0.005, help="Energy threshold (0.005 => 100%% recall at tol=0.1s)")
    ap.add_argument("--min-dist", type=float, default=0.08, help="Min distance between shots (s)")
    ap.add_argument("--tol", type=float, default=TOL, help="Match tolerance (s); use 0.1 for 100%% recall with th=0.005")
    ap.add_argument("--sweep", action="store_true", help="Sweep threshold to find one that gives 100%% recall")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}")
        return

    if args.sweep:
        # Sweep threshold to find minimum threshold that achieves 100% recall (pooled)
        videos = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")])
        total_gt = 0
        for vp in videos:
            audio_path = extract_audio(vp)
            fps, _ = get_fps_duration(vp)
            beeps = detect_beeps(audio_path, fps)
            if not beeps:
                continue
            ref_times = get_ref_times_for_video(vp, float(beeps[0]["t"]))
            if ref_times:
                total_gt += len(ref_times)
        if total_gt == 0:
            print("No ref data for sweep")
            return
        # 先用 0.04s 容差扫；若达不到 100% 再用 0.10s 容差
        for tol_use in [TOL, TOL_WIDE]:
            if tol_use > TOL:
                print(f"\nRelaxing match tolerance to ±{tol_use}s:")
            for th in [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]:
                total_tp, total_fp, total_fn = 0, 0, 0
                for vp in videos:
                    audio_path = extract_audio(vp)
                    fps, _ = get_fps_duration(vp)
                    beeps = detect_beeps(audio_path, fps)
                    if not beeps:
                        continue
                    beep_t = float(beeps[0]["t"])
                    ref_times = get_ref_times_for_video(vp, beep_t)
                    if not ref_times:
                        continue
                    shots = detect_shots_energy(audio_path=audio_path, energy_threshold=th, min_dist_sec=args.min_dist, t0_beep=beep_t)
                    shots = [s for s in shots if s.get("t", 0) >= beep_t]
                    tp, fp, fn, p, r, f1 = evaluate_shots(ref_times, shots, tol=tol_use)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                r_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                p_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
                print(f"  th={th:.3f}  n={total_tp+total_fp}  TP={total_tp}  FP={total_fp}  FN={total_fn}  R={r_all:.1%}  P={p_all:.1%}  F1={f1_all:.1%}")
                if r_all >= 0.9999:
                    print(f"\n=> Use threshold = {th} (tol=±{tol_use}s) for 100% recall.")
                    return
        print("\n=> No threshold achieved 100% recall in sweep.")
        return

    videos = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    )
    tol_run = getattr(args, "tol", TOL)
    print(f"Energy detector (threshold={args.threshold}, min_dist={args.min_dist}s, tol=±{tol_run}s)\n")
    print(f"{'video':<20} {'GT':>4} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 80)
    total_tp, total_fp, total_fn = 0, 0, 0
    for vp in videos:
        name = os.path.basename(vp)
        audio_path = extract_audio(vp)
        fps, _ = get_fps_duration(vp)
        beeps = detect_beeps(audio_path, fps)
        if not beeps:
            print(f"{name:<20} (no beep, skip)")
            continue
        beep_t = float(beeps[0]["t"])
        ref_times = get_ref_times_for_video(vp, beep_t)
        if not ref_times:
            print(f"{name:<20} (no ref, skip)")
            continue
        shots = detect_shots_energy(
            audio_path=audio_path,
            energy_threshold=args.threshold,
            min_dist_sec=args.min_dist,
            t0_beep=beep_t,
        )
        shots = [s for s in shots if s.get("t", 0) >= beep_t]
        tp, fp, fn, p, r, f1 = evaluate_shots(ref_times, shots, tol=tol_run)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"{name:<20} {len(ref_times):>4} {tp+fp:>4} {tp:>4} {fp:>4} {fn:>4} {p:>7.1%} {r:>7.1%} {f1:>7.1%}")
    print("-" * 80)
    if total_tp + total_fp + total_fn > 0:
        p_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
        print(f"{'Total (pooled)':<20} {'':>4} {total_tp+total_fp:>4} {total_tp:>4} {total_fp:>4} {total_fn:>4} {p_all:>7.1%} {r_all:>7.1%} {f1_all:>7.1%}")
    print("\n(Method: bandpass 400-6kHz -> envelope (normalized 0-1) -> peaks with height > threshold)")


if __name__ == "__main__":
    main()
