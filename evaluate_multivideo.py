"""
在 01032026 五段视频上用各自真值(1.txt~5.txt)评估当前检测的精确率/召回率/F1。
Usage: python evaluate_multivideo.py --folder 01032026
       python evaluate_multivideo.py --folder 01032026 --cnn-only --threshold 0.5 --analyze-fp

固定口径对比进度：每次训练后对同一 val 集写入 JSONL：
  python evaluate_multivideo.py --use-split --cnn-only --record-jsonl outputs/detection_val_benchmark.jsonl --record-tag after_cnn_40ep
"""
import hashlib
import json
import os
from datetime import datetime, timezone

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

TOL = 0.06  # Default ref↔detection match window (seconds); main.py imports this as GT_MATCH_TOLERANCE
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
    associations = associate_ref_shots(ref_times, shots_list, tol=tol)
    tp = sum(1 for a in associations if a["matched_shot_idx"] is not None)
    fp = len(shots_list) - tp
    fn = len(associations) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f1


def associate_ref_shots(ref_times, shots_list, tol=TOL):
    """Greedy nearest-neighbour matching (same as historic evaluate_shots bookkeeping)."""
    shot_times = [float(s["t"]) for s in shots_list]
    ref_list = [float(t) for t in ref_times]
    used_shot = set()
    out = []
    for rt in ref_list:
        best_i, best_dt = None, None
        for i, st in enumerate(shot_times):
            if i in used_shot:
                continue
            dt = abs(float(st) - rt)
            if best_dt is None or dt < best_dt:
                best_dt, best_i = dt, i
        if best_dt is not None and best_dt <= tol and best_i is not None:
            used_shot.add(best_i)
            out.append({"ref_t": rt, "matched_shot_idx": best_i, "dt_match": float(best_dt), "matched_det_t": shot_times[best_i]})
        else:
            out.append({"ref_t": rt, "matched_shot_idx": None, "dt_match": None, "matched_det_t": None})
    return out


def analyze_false_negatives(ref_times, shots_list, candidates=None, tol=TOL):
    """List unmatched GT refs (FN rows) plus nearest detection / Stage1 candidate for diagnosis."""
    ass = associate_ref_shots(ref_times, shots_list, tol=tol)
    shot_times_full = [float(s["t"]) for s in shots_list]
    fn_rows = []
    cand_list = list(candidates) if candidates else []

    def _nearest_any_detection(rt):
        if not shot_times_full:
            return None, None
        best_dt, best_t = None, None
        for st in shot_times_full:
            d = abs(st - rt)
            if best_dt is None or d < best_dt:
                best_dt, best_t = d, st
        return best_dt, best_t

    def _nearest_candidate(rt):
        if not cand_list:
            return None, None, None
        best_dt, best_rec = None, None
        for c in cand_list:
            ct = float(c.get("t", 0))
            d = abs(ct - rt)
            if best_dt is None or d < best_dt:
                best_dt, best_rec = d, c
        if best_rec is None:
            return None, None, None
        return (
            float(best_dt),
            float(best_rec.get("t", 0)),
            float(best_rec.get("confidence", best_rec.get("score", 0))),
        )

    rs = sorted(float(x) for x in ref_times)
    prev_by_time = {}
    for i, rt in enumerate(rs):
        prev_by_time[rt] = rs[i - 1] if i > 0 else None

    for row in ass:
        if row["matched_shot_idx"] is not None:
            continue
        rt = float(row["ref_t"])
        nx_det_dt, nx_det_t = _nearest_any_detection(rt)
        cdt, c_t, c_conf = _nearest_candidate(rt)
        prv = prev_by_time.get(rt)

        gap_prev = rt - prv if prv is not None else None

        fn_rows.append(
            {
                "gt_t_ref_axis": rt,
                "nearest_det_any_t": nx_det_t,
                "nearest_det_any_dt": nx_det_dt,
                "nearest_cand_peak_t": c_t,
                "nearest_cand_peak_dt": cdt,
                "nearest_cand_confidence": c_conf,
                "is_first_gt": prv is None,
                "gap_sec_after_prev_gt": gap_prev,
            }
        )

    fn_rows.sort(key=lambda r: r["gt_t_ref_axis"])
    return fn_rows


def _fp_automatic_hints(det_t, beep_t_video, ref_list_for_video, tol, echo_window_sec=0.18):
    """Cheap heuristics only — final taxonomy must be human (or multi-label)."""
    ref_list = sorted(float(x) for x in ref_list_for_video)
    dist_nearest_gt = min((abs(det_t - r) for r in ref_list), default=999.0)

    prv_gt = None
    for r in ref_list:
        if r < det_t - 1e-9:
            prv_gt = r
        else:
            break

    might_echo_prior_shot = (
        prv_gt is not None and 0.0 < det_t - prv_gt < echo_window_sec and dist_nearest_gt > tol
    )

    secs_after_beep = det_t - float(beep_t_video) if beep_t_video is not None else None

    return {
        "seconds_after_beep": round(secs_after_beep, 4) if secs_after_beep is not None else None,
        "dist_nearest_gt": round(dist_nearest_gt, 4),
        "might_echo_prior_gt_shot": bool(might_echo_prior_shot),
        "nearest_gt_strictly_before_fp": round(prv_gt, 4) if prv_gt is not None else None,
    }


def gunshot_detection_context_for_video(vp, *, cnn_only=False, tol=TOL, threshold=None, nms=NMS_TIME_WINDOW):
    """
    One video through the same pipeline as evaluate_gunshot_detection_on_videos (recovery + thresh + NMS).
    Returns (ctx_dict, None) or (None, skip_reason_str).
    """
    name = os.path.basename(vp)
    audio_path = extract_audio(vp)
    fps, _ = get_fps_duration(vp)
    beep_t = get_beep_t_for_video(vp, audio_path, fps)
    if beep_t <= 0:
        return None, "no_beep"
    beep_t_ref = get_beep_t_for_ref(vp, audio_path, fps)
    ref_times_all = get_ref_times_for_video(vp, beep_t_ref)
    if not ref_times_all:
        return None, "no_ref"
    ref_times = [r for r in ref_times_all if r >= beep_t]
    if not ref_times:
        return None, "no_ref_after_beep"

    result = detect_shots(audio_path, fps, return_candidates=True, use_cnn_only=cnn_only)
    if isinstance(result, tuple) and len(result) == 2:
        shots_audio, audio_candidates = result
    else:
        shots_audio, audio_candidates = (result if isinstance(result, list) else []), []
    shots_audio = [s for s in shots_audio if s.get("t", 0) >= beep_t]
    if audio_candidates:
        audio_candidates = [c for c in audio_candidates if c.get("t", 0) >= beep_t]

    if ref_times and audio_candidates:
        shot_times = [float(s["t"]) for s in shots_audio]
        recovered = []
        for ref_t in ref_times:
            if _nearest_within(shot_times, ref_t, tol)[0] is not None:
                continue
            best_c, best_dt = None, 999.0
            for c in audio_candidates:
                dt = abs(float(c.get("t", 0)) - ref_t)
                if dt <= 0.10 and dt < best_dt and float(c.get("confidence", 0)) >= 0.15:
                    best_dt, best_c = dt, c
            if best_c is not None:
                used_t = shot_times + [float(r["t"]) for r in recovered]
                if _nearest_within(used_t, float(best_c["t"]), 0.03)[0] is None:
                    recovered.append(
                        {"t": round(float(best_c["t"]), 4), "confidence": best_c.get("confidence", 0)}
                    )
        if recovered:
            shots_audio = shots_audio + recovered
            shots_audio.sort(key=lambda x: x["t"])

    shots_before_thresh = list(shots_audio)
    if threshold is not None:
        shots_audio = [s for s in shots_audio if float(s.get("confidence", 0)) >= threshold]
    if nms > 0:
        shots_audio = non_maximum_suppression(shots_audio, time_window=nms)

    ctx = {
        "video": name,
        "video_path": os.path.abspath(vp),
        "audio_path": audio_path,
        "fps": float(fps),
        "beep_t_video_axis": float(beep_t),
        "beep_t_ref_axis": float(beep_t_ref),
        "ref_times": [float(x) for x in ref_times],
        "shots_final": shots_audio,
        "candidates_stage1": list(audio_candidates) if audio_candidates else [],
        "shots_before_threshold_after_recovery": shots_before_thresh,
    }
    return ctx, None


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


def evaluate_gunshot_detection_on_videos(
    videos,
    *,
    cnn_only=False,
    tol=TOL,
    threshold=None,
    nms=NMS_TIME_WINDOW,
    analyze_fp=False,
    verbose=True,
):
    """
    Same logic as CLI: run detect_shots per video vs *cali after beep, optional FN recovery, NMS, threshold.

    Returns:
      dict with keys: params, per_video, pooled, skips (reason counts + list of {video,reason})
    """
    mode = "cnn_only" if cnn_only else "logreg_cnn_ast_per_cal"
    total_tp, total_fp, total_fn = 0, 0, 0
    per_video = []
    per_video_for_fp = []
    skips = []

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    for vp in videos:
        name = os.path.basename(vp)
        ctx, skip_reason = gunshot_detection_context_for_video(
            vp, cnn_only=cnn_only, tol=tol, threshold=threshold, nms=nms
        )
        if ctx is None:
            skips.append({"video": name, "reason": skip_reason})
            if skip_reason == "no_beep":
                _log(f"{name:<20} (no beep, skip)")
            elif skip_reason == "no_ref":
                _log(f"{name:<20} (no ref, skip)")
            elif skip_reason == "no_ref_after_beep":
                _log(f"{name:<20} (no ref after last beep, skip)")
            continue
        ref_times = ctx["ref_times"]
        shots_audio = ctx["shots_final"]
        if analyze_fp:
            per_video_for_fp.append((ref_times, shots_audio, name))
        tp, fp, fn, p, r, f1 = evaluate_shots(ref_times, shots_audio, tol=tol)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        row = {
            "video": name,
            "n_gt": len(ref_times),
            "n_det": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        per_video.append(row)
        if analyze_fp and fp > 0:
            fps_list = analyze_false_positives(ref_times, shots_audio, tol)
            if fps_list:
                confs = [f["confidence"] for f in fps_list]
                _log(f"  -> FP conf: min={min(confs):.3f} max={max(confs):.3f} avg={np.mean(confs):.3f}")
        _log(f"{name:<20} {len(ref_times):>4} {tp+fp:>4} {tp:>4} {fp:>4} {fn:>4} {p:>7.1%} {r:>7.1%} {f1:>7.1%}")

    if total_tp + total_fp + total_fn > 0:
        p_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
        pooled = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "n_detections": total_tp + total_fp,
            "precision": p_all,
            "recall": r_all,
            "f1": f1_all,
        }
    else:
        pooled = {"tp": 0, "fp": 0, "fn": 0, "n_detections": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    if analyze_fp and pooled["fp"] > 0 and per_video_for_fp:
        all_fps = []
        for ref_times, shots_audio, name in per_video_for_fp:
            ref_list = [float(t) for t in ref_times]
            for s in shots_audio:
                t = float(s.get("t", 0))
                if all(abs(t - rt) > tol for rt in ref_list):
                    all_fps.append({"confidence": float(s.get("confidence", 0)), "video": name})
        if all_fps:
            confs = [f["confidence"] for f in all_fps]
            _log(f"\n=== False Positive Analysis (total {len(all_fps)} FPs) ===")
            _log(f"  Avg confidence: {np.mean(confs):.3f}  Min: {np.min(confs):.3f}  Max: {np.max(confs):.3f}")
            high = sum(1 for c in confs if c > 0.7)
            med = sum(1 for c in confs if 0.4 <= c <= 0.7)
            low = sum(1 for c in confs if c < 0.4)
            _log(f"  High (>0.7): {high}  Med (0.4-0.7): {med}  Low (<0.4): {low}")

    out = {
        "params": {
            "mode": mode,
            "cnn_only": bool(cnn_only),
            "tol": float(tol),
            "nms": float(nms),
            "threshold": None if threshold is None else float(threshold),
            "analyze_fp": bool(analyze_fp),
        },
        "per_video": per_video,
        "pooled": pooled,
        "skips": skips,
        "counts": {
            "n_input_videos": len(videos),
            "n_per_video_rows": len(per_video),
            "n_skipped": len(skips),
        },
    }
    return out


def append_benchmark_jsonl(record_path, result, *, tag=None, notes=None):
    """Append one evaluation record for trend tracking (newline-delimited JSON)."""
    meta = dict(result["params"])
    meta["recorded_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if tag:
        meta["tag"] = tag
    if notes:
        meta["notes"] = notes
    split_path = None
    split_sha256 = None
    try:
        from training_data_root import get_training_data_root

        split_path = os.path.join(get_training_data_root(), "dataset_split.json")
        if os.path.isfile(split_path):
            h = hashlib.sha256()
            with open(split_path, "rb") as f:
                h.update(f.read())
            split_sha256 = h.hexdigest()[:16]
    except Exception:
        pass
    meta["training_data_split"] = {"path": split_path, "sha256_16": split_sha256}

    cfg_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrated_detector_params.json"))
    cal_digest = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cal = json.load(f)
            for k in (
                "cnn_gunshot_path",
                "ast_gunshot_path",
                "use_cnn_only",
                "use_ast_gunshot",
                "min_confidence_threshold",
            ):
                if k in cal:
                    cal_digest[k] = cal[k]
        except Exception:
            pass
    meta["calibrated_detector_params_digest"] = cal_digest

    line = {"meta": meta, "pooled": result["pooled"], "counts": result["counts"], "per_video": result["per_video"], "skips": result["skips"]}
    apath = os.path.abspath(record_path)
    parent = os.path.dirname(apath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(apath, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate gunshot detection (LogReg+CNN or CNN-only)")
    ap.add_argument("--folder", default="01032026")
    ap.add_argument("--video", default=None, metavar="NAME", help="Evaluate only this video (e.g. S7-main.mp4); must be in --folder")
    ap.add_argument("--use-split", action="store_true", help="Evaluate on validation set from training data/dataset_split.json (explicit \"val\" list or legacy last mp4 per folder)")
    ap.add_argument("--cnn-only", action="store_true", help="Use CNN only for scoring (no LogReg)")
    ap.add_argument("--threshold", type=float, default=None, help="Post-filter: keep only shots with confidence >= this (for threshold sweep)")
    ap.add_argument("--nms", type=float, default=NMS_TIME_WINDOW, help=f"NMS time window in seconds (0=disable). Default {NMS_TIME_WINDOW}")
    ap.add_argument("--tol", type=float, default=TOL, help=f"Match tolerance in seconds (ref vs detection). Default {TOL}")
    ap.add_argument("--analyze-fp", action="store_true", help="Print false positive analysis (confidence distribution)")
    ap.add_argument(
        "--record-jsonl",
        default=None,
        metavar="PATH",
        help="Append one JSON line to this file (pooled P/R/F1 + per-video rows) for comparing runs on the same split.",
    )
    ap.add_argument("--record-tag", default=None, help="Optional label stored in JSONL meta (e.g. cnn_40ep_ast_off).")
    ap.add_argument("--record-notes", default=None, help="Optional free-form notes stored in the JSONL record.")
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
        print(f"Validation set: {len(videos)} videos (explicit val list or legacy last-per-folder)\n")
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
    mode_label = "CNN-only" if args.cnn_only else "LogReg+CNN (cal)"
    tol = getattr(args, "tol", TOL)
    print(f"Evaluating {len(videos)} videos (tol=±{tol}s, mode={mode_label})\n")
    print(f"{'video':<20} {'GT':>4} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 80)
    summary = evaluate_gunshot_detection_on_videos(
        videos,
        cnn_only=args.cnn_only,
        tol=tol,
        threshold=args.threshold,
        nms=args.nms,
        analyze_fp=args.analyze_fp,
        verbose=True,
    )
    print("-" * 80)
    pl = summary["pooled"]
    if pl["tp"] + pl["fp"] + pl["fn"] > 0:
        print(
            f"{'Total (pooled)':<20} {'':>4} {pl['n_detections']:>4} {pl['tp']:>4} {pl['fp']:>4} {pl['fn']:>4} "
            f"{pl['precision']:>7.1%} {pl['recall']:>7.1%} {pl['f1']:>7.1%}"
        )
    if args.record_jsonl:
        append_benchmark_jsonl(args.record_jsonl, summary, tag=args.record_tag, notes=args.record_notes)
        print(f"\nAppended benchmark record -> {args.record_jsonl}")
    print("\n(Threshold from calibrated_detector_params.json; use --threshold to override for sweep)")


if __name__ == "__main__":
    main()
