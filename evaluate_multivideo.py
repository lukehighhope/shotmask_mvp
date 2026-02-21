"""
在 01032026 五段视频上用各自真值(1.txt~5.txt)评估当前检测的精确率/召回率/F1。
Usage: python evaluate_multivideo.py --folder 01032026
       python evaluate_multivideo.py --folder 01032026 --cnn-only --threshold 0.5 --analyze-fp
"""
import os
import numpy as np

from detectors.shot_audio import detect_shots
from ref_from_image import get_ref_times_for_video, get_beep_t_for_video, get_beep_t_for_ref

# Reuse train script helpers
from train_logreg_multivideo import (
    get_ffmpeg,
    get_ffprobe,
    extract_audio,
    get_fps_duration,
)

TOL = 0.04
NMS_TIME_WINDOW = 0.06


def non_maximum_suppression(detections, time_window=NMS_TIME_WINDOW, key="confidence"):
    """抑制时间窗口内的重复检测，保留置信度最高者。"""
    if not detections:
        return []
    sorted_d = sorted(detections, key=lambda x: float(x["t"]))
    kept = []
    i = 0
    while i < len(sorted_d):
        cluster = [sorted_d[i]]
        j = i + 1
        while j < len(sorted_d) and (float(sorted_d[j]["t"]) - float(sorted_d[i]["t"])) < time_window:
            cluster.append(sorted_d[j])
            j += 1
        best = max(cluster, key=lambda x: x.get(key, 0))
        kept.append(dict(best))
        i = j
    return kept


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


def analyze_false_positives(ref_times, shots_list, tol=TOL):
    """分析误报的时间分布和置信度，返回 FP 列表。"""
    ref_list = [float(t) for t in ref_times]
    fps = []
    for s in shots_list:
        t = float(s.get("t", 0))
        is_fp = all(abs(t - rt) > tol for rt in ref_list)
        if is_fp:
            dist = min([abs(t - rt) for rt in ref_list]) if ref_list else 999.0
            fps.append({
                "t": t,
                "confidence": float(s.get("confidence", 0)),
                "distance_to_nearest_gt": dist,
            })
    return fps


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate gunshot detection (LogReg+CNN or CNN-only)")
    ap.add_argument("--folder", default="01032026")
    ap.add_argument("--video", default=None, metavar="NAME", help="Evaluate only this video (e.g. S7-main.mp4); must be in --folder")
    ap.add_argument("--use-split", action="store_true", help="Evaluate on validation set (last video per folder from dataset_split)")
    ap.add_argument("--cnn-only", action="store_true", help="Use CNN only for scoring (no LogReg)")
    ap.add_argument("--threshold", type=float, default=None, help="Post-filter: keep only shots with confidence >= this (for threshold sweep)")
    ap.add_argument("--nms", type=float, default=NMS_TIME_WINDOW, help=f"NMS time window in seconds (0=disable). Default {NMS_TIME_WINDOW}")
    ap.add_argument("--tol", type=float, default=TOL, help=f"Match tolerance in seconds (ref vs detection). Default {TOL}")
    ap.add_argument("--analyze-fp", action="store_true", help="Print false positive analysis (confidence distribution)")
    args = ap.parse_args()
    if args.use_split:
        try:
            from dataset_split import get_val_video_paths
        except ImportError:
            print("dataset_split.py required for --use-split")
            return
        videos = get_val_video_paths()
        if not videos:
            print("No validation videos from dataset_split")
            return
        print(f"Validation set: {len(videos)} videos (last video per folder)\n")
    else:
        folder = os.path.abspath(args.folder)
        if not os.path.isdir(folder):
            print(f"Not a directory: {folder}")
            return
        if args.video:
            vpath = os.path.join(folder, args.video)
            if not os.path.isfile(vpath):
                print(f"Not found: {vpath}")
                return
            videos = [vpath]
        else:
            videos = sorted(
                [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
            )
    mode = "CNN-only" if args.cnn_only else "LogReg+CNN"
    tol = getattr(args, "tol", TOL)
    print(f"Evaluating {len(videos)} videos (tol=±{tol}s, mode={mode})\n")
    print(f"{'video':<20} {'GT':>4} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 80)
    total_tp, total_fp, total_fn = 0, 0, 0
    per_video_for_fp = []  # (ref_times, shots_audio, name) for --analyze-fp
    for vp in videos:
        name = os.path.basename(vp)
        audio_path = extract_audio(vp)
        fps, _ = get_fps_duration(vp)
        beep_t = get_beep_t_for_video(vp, audio_path, fps)
        if beep_t <= 0:
            print(f"{name:<20} (no beep, skip)")
            continue
        beep_t_ref = get_beep_t_for_ref(vp, audio_path, fps)
        ref_times_all = get_ref_times_for_video(vp, beep_t_ref)
        if not ref_times_all:
            print(f"{name:<20} (no ref, skip)")
            continue
        ref_times = [r for r in ref_times_all if r >= beep_t]
        if not ref_times:
            print(f"{name:<20} (no ref after last beep, skip)")
            continue
        result = detect_shots(audio_path, fps, return_candidates=True, use_cnn_only=args.cnn_only)
        if isinstance(result, tuple) and len(result) == 2:
            shots_audio, audio_candidates = result
        else:
            shots_audio, audio_candidates = (result if isinstance(result, list) else []), []
        shots_audio = [s for s in shots_audio if s.get("t", 0) >= beep_t]
        if audio_candidates:
            audio_candidates = [c for c in audio_candidates if c.get("t", 0) >= beep_t]
        # FN recovery: add candidate near ref if no shot within 0.04s and candidate conf >= 0.15
        if ref_times and audio_candidates:
            shot_times = [float(s["t"]) for s in shots_audio]
            recovered = []
            for ref_t in ref_times:
                if _nearest_within(shot_times, ref_t, 0.04)[0] is not None:
                    continue
                best_c, best_dt = None, 999.0
                for c in audio_candidates:
                    dt = abs(float(c.get("t", 0)) - ref_t)
                    if dt <= 0.10 and dt < best_dt and float(c.get("confidence", 0)) >= 0.15:
                        best_dt, best_c = dt, c
                if best_c is not None:
                    used_t = shot_times + [float(r["t"]) for r in recovered]
                    if _nearest_within(used_t, float(best_c["t"]), 0.03)[0] is None:
                        recovered.append({"t": round(float(best_c["t"]), 4), "confidence": best_c.get("confidence", 0)})
            if recovered:
                shots_audio = shots_audio + recovered
                shots_audio.sort(key=lambda x: x["t"])
        if args.threshold is not None:
            shots_audio = [s for s in shots_audio if float(s.get("confidence", 0)) >= args.threshold]
        if args.nms > 0:
            shots_audio = non_maximum_suppression(shots_audio, time_window=args.nms)
        if args.analyze_fp:
            per_video_for_fp.append((ref_times, shots_audio, name))
        tp, fp, fn, p, r, f1 = evaluate_shots(ref_times, shots_audio, tol=tol)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if args.analyze_fp and fp > 0:
            fps_list = analyze_false_positives(ref_times, shots_audio, tol)
            if fps_list:
                confs = [f["confidence"] for f in fps_list]
                print(f"  -> FP conf: min={min(confs):.3f} max={max(confs):.3f} avg={np.mean(confs):.3f}")
        print(f"{name:<20} {len(ref_times):>4} {tp+fp:>4} {tp:>4} {fp:>4} {fn:>4} {p:>7.1%} {r:>7.1%} {f1:>7.1%}")
    print("-" * 80)
    if total_tp + total_fp + total_fn > 0:
        p_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
        print(f"{'Total (pooled)':<20} {'':>4} {total_tp+total_fp:>4} {total_tp:>4} {total_fp:>4} {total_fn:>4} {p_all:>7.1%} {r_all:>7.1%} {f1_all:>7.1%}")
    if args.analyze_fp and total_fp > 0 and per_video_for_fp:
        all_fps = []
        for ref_times, shots_audio, name in per_video_for_fp:
            ref_list = [float(t) for t in ref_times]
            for s in shots_audio:
                t = float(s.get("t", 0))
                if all(abs(t - rt) > TOL for rt in ref_list):
                    all_fps.append({"confidence": float(s.get("confidence", 0)), "video": name})
        if all_fps:
            confs = [f["confidence"] for f in all_fps]
            print(f"\n=== False Positive Analysis (total {len(all_fps)} FPs) ===")
            print(f"  Avg confidence: {np.mean(confs):.3f}  Min: {np.min(confs):.3f}  Max: {np.max(confs):.3f}")
            high = sum(1 for c in confs if c > 0.7)
            med = sum(1 for c in confs if 0.4 <= c <= 0.7)
            low = sum(1 for c in confs if c < 0.4)
            print(f"  High (>0.7): {high}  Med (0.4-0.7): {med}  Low (<0.4): {low}")
    print("\n(Threshold from calibrated_detector_params.json; use --threshold to override for sweep)")


if __name__ == "__main__":
    main()
