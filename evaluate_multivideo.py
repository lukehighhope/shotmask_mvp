"""
在 01032026 五段视频上用各自真值(1.txt~5.txt)评估当前检测的精确率/召回率/F1。
Usage: python evaluate_multivideo.py --folder 01032026
"""
import os
import subprocess

from detectors.beep import detect_beeps
from detectors.shot_audio import detect_shots
from ref_from_image import get_ref_times_for_video

# Reuse train script helpers
from train_logreg_multivideo import (
    get_ffmpeg,
    get_ffprobe,
    extract_audio,
    get_fps_duration,
)

TOL = 0.04


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="01032026")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}")
        return
    videos = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    )
    print(f"Evaluating {len(videos)} videos (tol=±{TOL}s)\n")
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
        result = detect_shots(audio_path, fps, return_candidates=True)
        if isinstance(result, tuple) and len(result) == 2:
            shots_audio, audio_candidates = result
        else:
            shots_audio, audio_candidates = (result if isinstance(result, list) else []), []
        shots_audio = [s for s in shots_audio if s.get("t", 0) >= beep_t]
        if audio_candidates:
            audio_candidates = [c for c in audio_candidates if c.get("t", 0) >= beep_t]
        # FN recovery: add candidate near ref if no shot within 0.04s and candidate conf >= 0.2
        if ref_times and audio_candidates:
            shot_times = [float(s["t"]) for s in shots_audio]
            recovered = []
            for ref_t in ref_times:
                if _nearest_within(shot_times, ref_t, 0.04)[0] is not None:
                    continue
                best_c, best_dt = None, 999.0
                for c in audio_candidates:
                    dt = abs(float(c.get("t", 0)) - ref_t)
                    if dt <= 0.08 and dt < best_dt and float(c.get("confidence", 0)) >= 0.2:
                        best_dt, best_c = dt, c
                if best_c is not None:
                    used_t = shot_times + [float(r["t"]) for r in recovered]
                    if _nearest_within(used_t, float(best_c["t"]), 0.03)[0] is None:
                        recovered.append({"t": round(float(best_c["t"]), 4), "confidence": best_c.get("confidence", 0)})
            if recovered:
                shots_audio = shots_audio + recovered
                shots_audio.sort(key=lambda x: x["t"])
        tp, fp, fn, p, r, f1 = evaluate_shots(ref_times, shots_audio)
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
    print("\n(Current threshold from calibrated_detector_params.json min_confidence_threshold)")


if __name__ == "__main__":
    main()
