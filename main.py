import os
import json
import shutil
import subprocess
import argparse

from detectors.beep import detect_beeps
from detectors.shot_audio import detect_shots, cluster_peaks_by_time
from detectors.shot_motion import detect_shots_from_motion_roi_auto, detect_shots_from_motion_improved
from reference_splits import ref_shot_times as get_ref_shot_times
from ref_from_image import get_ref_times_for_video
from overlay.render_frames import render_overlay_frames
from overlay.encode_overlay import encode_webm


def _nearest_within(times, t, tol):
    """Return (idx, dt) for nearest element within tol, else (None, None)."""
    if not times:
        return None, None
    best_i = None
    best_dt = None
    for i, tt in enumerate(times):
        dt = abs(float(tt) - float(t))
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_i = i
    if best_dt is not None and best_dt <= tol:
        return best_i, best_dt
    return None, None


def non_maximum_suppression(detections, time_threshold_s=0.06):
    """
    Merge detections within time_threshold_s; keep the one with highest confidence per cluster.
    Reduces duplicate detections for the same physical shot (e.g. reverb/echo).
    """
    if not detections or time_threshold_s <= 0:
        return list(detections)
    sorted_d = sorted(detections, key=lambda x: float(x["t"]))
    kept = []
    i = 0
    while i < len(sorted_d):
        cluster = [sorted_d[i]]
        j = i + 1
        while j < len(sorted_d) and (float(sorted_d[j]["t"]) - float(sorted_d[i]["t"])) < time_threshold_s:
            cluster.append(sorted_d[j])
            j += 1
        best = max(cluster, key=lambda x: x.get("confidence", 0))
        kept.append(dict(best))
        i = j
    return kept


def _print_gt_diagnostics(ref_times, shots, candidates, tol=0.04):
    """
    Diagnostics against GT (ref_times):
    - candidate_coverage
    - per-FN: is there candidate within tol, its P(shot)=score, and cluster rank (1st/2nd/...)
    - FN-candidate score vs FP-candidate score distribution
    """
    if not ref_times:
        print("\n[GT diag] No reference (GT) times available.")
        return

    shot_times = [float(s["t"]) for s in (shots or [])]
    cand_times = [float(c["t"]) for c in (candidates or [])]
    cand_scores = [float(c.get("score", 0.0)) for c in (candidates or [])]

    covered = 0
    for gt in ref_times:
        _, _dt = _nearest_within(cand_times, gt, tol)
        if _dt is not None:
            covered += 1

    print("\n" + "=" * 50)
    print("[GT diag] Candidate coverage")
    print("=" * 50)
    print(f"candidate_coverage = {covered}/{len(ref_times)}  (tol=±{tol:.3f}s)")

    fns = []
    for gt in ref_times:
        _, dt = _nearest_within(shot_times, gt, tol)
        if dt is None:
            fns.append(float(gt))

    print("\n" + "=" * 50)
    print("[GT diag] False Negatives (FN)")
    print("=" * 50)
    print(f"FN count = {len(fns)}/{len(ref_times)}  (tol=±{tol:.3f}s)")
    if not fns:
        return

    cluster_rank_by_cand = {}
    if candidates:
        clusters = cluster_peaks_by_time(cand_times, cluster_window_sec=0.25)
        for cluster in clusters:
            if not cluster:
                continue
            ranked = sorted(cluster, key=lambda i: cand_scores[i], reverse=True)
            for r, i in enumerate(ranked, start=1):
                cluster_rank_by_cand[i] = r

    # Label candidates matched to any GT as "GT-covered"; rest treated as FP candidates
    gt_matched_cand = set()
    for gt in ref_times:
        ci, _dt = _nearest_within(cand_times, gt, tol)
        if ci is not None:
            gt_matched_cand.add(ci)

    fn_scores = []
    fp_scores = [cand_scores[i] for i in range(len(cand_scores)) if i not in gt_matched_cand]

    for gt in fns:
        ci, cdt = _nearest_within(cand_times, gt, tol)
        if ci is None:
            print(f"- FN @ GT={gt:.4f}s: no candidate within ±{tol:.3f}s")
            continue
        score = cand_scores[ci]
        rank = cluster_rank_by_cand.get(ci, None)
        rank_str = f"{rank}" if rank is not None else "?"
        fn_scores.append(score)
        print(
            f"- FN @ GT={gt:.4f}s: has_candidate=yes (dt={cdt:.4f}s) "
            f"P(shot)={score:.4f}  cluster_rank={rank_str}"
        )

    def _summ_stats(xs):
        if not xs:
            return "n=0"
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        p50 = xs_sorted[n // 2]
        p10 = xs_sorted[max(0, int(round(0.10 * (n - 1))))]
        p90 = xs_sorted[min(n - 1, int(round(0.90 * (n - 1))))]
        return f"n={n}, min={xs_sorted[0]:.4f}, p10={p10:.4f}, p50={p50:.4f}, p90={p90:.4f}, max={xs_sorted[-1]:.4f}"

    print("\n" + "=" * 50)
    print("[GT diag] P(shot) distribution (FN-candidates vs FP-candidates)")
    print("=" * 50)
    print(f"FN-candidates: {_summ_stats(fn_scores)}")
    print(f"FP-candidates: {_summ_stats(fp_scores)}")


def _evaluate_shots(ref_times, shots_list, tol=0.04):
    """Compute TP/FP/FN and P/R/F1 for an already filtered list of shots (no confidence filter)."""
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


def _precision_recall_at_threshold(ref_times, shots_with_confidence, conf_threshold, tol=0.04):
    """Given ref (GT) times and list of shots with 't' and 'confidence', filter by conf >= conf_threshold, compute TP/FP/FN and P/R."""
    filtered = [s for s in shots_with_confidence if s.get("confidence", 0) >= conf_threshold]
    return _evaluate_shots(ref_times, filtered, tol)


def _sweep_confidence_threshold(ref_times, shots_audio, tol=0.04, thresholds=None):
    """Print Precision / Recall / F1 for different confidence thresholds (for tuning)."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n" + "=" * 60)
    print("[Threshold sweep] Precision / Recall / F1 (tol=±{:.2f}s)".format(tol))
    print("=" * 60)
    print(f"{'thresh':>8} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 60)
    best_f1 = -1.0
    best_thresh = None
    for th in thresholds:
        tp, fp, fn, p, r, f1 = _precision_recall_at_threshold(ref_times, shots_audio, th, tol)
        n = tp + fp
        print(f"{th:>8.2f} {n:>4} {tp:>4} {fp:>4} {fn:>4} {p:>8.2%} {r:>8.2%} {f1:>8.2%}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
    print("-" * 60)
    if best_thresh is not None:
        print(f"Best F1 = {best_f1:.2%} at threshold = {best_thresh}")
    print("=" * 60)


def _grid_search_audio_postprocess(ref_times, shots_audio, tol=0.04, conf_grid=None, nms_grid=None):
    """
    Grid search over (min_confidence, NMS window). For each (c, n): filter by confidence >= c,
    apply NMS with window n, then compute P/R/F1. Print table and best params.
    """
    if conf_grid is None:
        conf_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if nms_grid is None:
        nms_grid = [0.0, 0.04, 0.06, 0.08]
    ref_list = [float(t) for t in ref_times]
    best_f1 = -1.0
    best_params = None
    rows = []
    for c in conf_grid:
        filtered = [s for s in shots_audio if s.get("confidence", 0) >= c]
        for n in nms_grid:
            if n > 0:
                list_nms = non_maximum_suppression(filtered, time_threshold_s=n)
            else:
                list_nms = list(filtered)
            tp, fp, fn, p, r, f1 = _evaluate_shots(ref_times, list_nms, tol)
            rows.append((c, n, len(list_nms), tp, fp, fn, p, r, f1))
            if f1 > best_f1:
                best_f1 = f1
                best_params = (c, n)
    print("\n" + "=" * 78)
    print("[Grid search] Audio post-process: min_confidence × NMS (tol=±{:.2f}s)".format(tol))
    print("=" * 78)
    print(f"{'conf':>6} {'nms':>6} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 78)
    for c, n, n_det, tp, fp, fn, p, r, f1 in rows:
        print(f"{c:>6.2f} {n:>6.2f} {n_det:>4} {tp:>4} {fp:>4} {fn:>4} {p:>8.2%} {r:>8.2%} {f1:>8.2%}")
    print("-" * 78)
    if best_params is not None:
        c_best, n_best = best_params
        print("Best F1 = {:.2%}  =>  min_confidence = {:.2f},  NMS = {:.2f}s".format(best_f1, c_best, n_best))
        print("  Suggested: set calibrated_detector_params.json min_confidence_threshold = {:.2f}".format(c_best))
        if n_best > 0:
            print("  Run with: --nms {:.2f}".format(n_best))
    print("=" * 78)

def get_ffmpeg_cmd():
    """Get ffmpeg command (prefer PATH, otherwise use full path)"""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    # Common Windows paths
    common_paths = [
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None

def get_ffprobe_cmd():
    """Get ffprobe command (prefer PATH, otherwise use full path)"""
    if shutil.which("ffprobe"):
        return "ffprobe"
    # Common Windows paths
    common_paths = [
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def ffprobe_info(video):
    ffprobe = get_ffprobe_cmd()
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found, please ensure ffmpeg is installed")
    cmd = (
        f'"{ffprobe}" -v error -select_streams v:0 '
        f'-show_entries stream=r_frame_rate,duration '
        f'-of json "{video}"'
    )
    out = subprocess.check_output(cmd, shell=True)
    data = json.loads(out)
    r = data["streams"][0]["r_frame_rate"]
    fps = eval(r)
    duration = float(data["streams"][0]["duration"])
    return fps, duration

def extract_audio(video, ffmpeg):
    """Extract audio file"""
    os.makedirs("tmp", exist_ok=True)
    audio = "tmp/audio.wav"
    print(f"Extracting audio: {audio}")
    run(f'"{ffmpeg}" -y -i "{video}" -ac 1 -ar 48000 -vn "{audio}"')
    return audio


def cross_validate_shots(audio_shots, motion_shots, max_diff_s=0.15, adaptive_window=True, use_confidence=True):
    """
    Cross-validate audio-detected shots with motion-detected shots.
    If adaptive_window=True, set max_diff_s from median inter-shot interval (OPTIMIZATION_GUIDE).
    use_confidence: include combined confidence in returned matched pairs.
    Returns matched pairs and unmatched shots.
    """
    audio_times = [s["t"] for s in audio_shots]
    motion_times = [s["t"] for s in motion_shots]
    
    if adaptive_window and (len(audio_times) + len(motion_times)) > 1:
        all_times = sorted([float(t) for t in audio_times] + [float(t) for t in motion_times])
        intervals = [all_times[i + 1] - all_times[i] for i in range(len(all_times) - 1)]
        if intervals:
            import numpy as np
            median_interval = float(np.median(intervals))
            max_diff_s = min(0.15, max(0.04, median_interval * 0.5))
    
    matched_audio = []
    matched_motion = []
    matched_confidence = []  # (audio_conf + motion_conf) / 2 when use_confidence
    unmatched_audio = []
    unmatched_motion = []
    
    used_motion = set()
    for i, a_t in enumerate(audio_times):
        best_j = None
        best_diff = float("inf")
        for j, m_t in enumerate(motion_times):
            if j in used_motion:
                continue
            diff = abs(a_t - m_t)
            if diff < max_diff_s and diff < best_diff:
                best_diff = diff
                best_j = j
        if best_j is not None:
            matched_audio.append(i)
            matched_motion.append(best_j)
            used_motion.add(best_j)
            if use_confidence:
                a_conf = audio_shots[i].get("confidence", 0.5)
                m_conf = motion_shots[best_j].get("confidence", 0.5)
                matched_confidence.append(round((a_conf + m_conf) / 2.0, 3))
        else:
            unmatched_audio.append(i)
    
    for j in range(len(motion_times)):
        if j not in used_motion:
            unmatched_motion.append(j)
    
    result = {
        "matched_audio": matched_audio,
        "matched_motion": matched_motion,
        "unmatched_audio": unmatched_audio,
        "unmatched_motion": unmatched_motion,
        "max_diff_s": max_diff_s,
    }
    if use_confidence and matched_confidence:
        result["matched_confidence"] = matched_confidence
    return result

def main(video, mode="all", shots_filter="all", nms_s=0.0, sweep_threshold=False, grid_search=False):
    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("ffmpeg not detected, please install first.")
        print("  Windows: Download ffmpeg-release-essentials.zip from https://www.gyan.dev/ffmpeg/builds/")
        print("  Extract (e.g., C:\\ffmpeg-8.0.1-essentials_build),")
        print("  Add bin folder path (e.g., C:\\ffmpeg-8.0.1-essentials_build\\bin)")
        print("  to system PATH environment variable, then reopen command window.")
        print("  Or: https://ffmpeg.org/download.html")
        raise SystemExit(1)

    os.makedirs("tmp", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Extract audio
    audio = extract_audio(video, ffmpeg)
    
    # Get video info
    fps, duration = ffprobe_info(video)
    print(f"Video FPS: {fps}, Duration: {duration:.2f} seconds\n")

    # Mode: detect beep only
    if mode == "beep":
        print("=" * 50)
        print("Detecting beep time")
        print("=" * 50)
        beeps = detect_beeps(audio, fps)
        
        print(f"\nDetected {len(beeps)} beep(s):")
        for i, beep in enumerate(beeps, 1):
            print(f"  Beep {i}: {beep['t']:.4f} seconds (frame {beep['frame']})")
        
        if beeps:
            print(f"\nFirst beep time (t0): {beeps[0]['t']:.4f} seconds")
            print(f"\nAll beep time sequence:")
            for beep in beeps:
                print(f"  {beep['t']:.4f}")
        else:
            print("\nWarning: No beep detected")
        return

    # Mode: detect shots only
    if mode == "shots":
        print("=" * 50)
        print("Detecting shot times")
        print("=" * 50)
        shots = detect_shots(audio, fps)
        
        print(f"\nDetected {len(shots)} shot(s):")
        for i, shot in enumerate(shots, 1):
            print(f"  Shot {i}: {shot['t']:.4f} seconds (frame {shot['frame']}, confidence {shot['confidence']:.2f})")
        
        if shots:
            print(f"\nAll shot time sequence:")
            for shot in shots:
                print(f"  {shot['t']:.4f}")
        else:
            print("\nWarning: No shots detected")
        return
    
    # Mode: detect motion only
    if mode == "motion":
        print("=" * 50)
        print("Detecting shots from video motion")
        print("=" * 50)
        motion_shots = detect_shots_from_motion_roi_auto(video, fps, method="diff")
        
        print(f"\nDetected {len(motion_shots)} shot(s) from motion:")
        for i, shot in enumerate(motion_shots, 1):
            print(f"  Shot {i}: {shot['t']:.4f} seconds (frame {shot['frame']}, confidence {shot['confidence']:.2f})")
        
        if motion_shots:
            print(f"\nAll motion-detected shot time sequence:")
            for shot in motion_shots:
                print(f"  {shot['t']:.4f}")
        else:
            print("\nWarning: No motion detected")
        return

    # Mode: full pipeline (default)
    # Save original video fps for audio detection (audio is extracted from original video)
    original_fps, original_duration = ffprobe_info(video)
    print(f"Original video: FPS={original_fps}, Duration={original_duration:.2f}s")
    
    cfr_video = "tmp/cfr.mp4"
    # Stream copy to avoid libx264 malloc issues on some systems; use source as CFR input for motion
    run(f'"{ffmpeg}" -y -i "{video}" -c:v copy -c:a aac "{cfr_video}"')

    # CFR video fps for motion detection (motion detection uses CFR video)
    cfr_fps, cfr_duration = ffprobe_info(cfr_video)
    print(f"CFR video: FPS={cfr_fps}, Duration={cfr_duration:.2f}s")
    print(f"Note: Audio detection uses original FPS ({original_fps}), motion detection uses CFR FPS ({cfr_fps})")
    print(f"Time axis: Both use absolute time (seconds), should be aligned.\n")

    # Audio detection: use original fps (audio is extracted from original video)
    # Note: Time 't' is calculated from audio sample rate, fps is only used for frame calculation
    beeps = detect_beeps(audio, original_fps)
    # Audio detection: return candidates for GT diagnostics
    shots_audio_result = detect_shots(audio, original_fps, return_candidates=True)
    if isinstance(shots_audio_result, tuple) and len(shots_audio_result) == 2:
        shots_audio, audio_candidates = shots_audio_result
    else:
        shots_audio, audio_candidates = shots_audio_result, []
    
    # Shots all happen after beep: drop any detection before beep (video start noise)
    t0_beep = float(beeps[0]["t"]) if beeps else 0.0
    if beeps and len(shots_audio) > 0:
        shots_audio = [s for s in shots_audio if s["t"] >= t0_beep]
        if audio_candidates:
            audio_candidates = [c for c in audio_candidates if c.get("t", 0) >= t0_beep]
    
    # FN recovery: when ref available, add near-miss candidates (confidence >= 0.2, within 0.08s of ref)
    if beeps and video and audio_candidates:
        ref_times_recovery = get_ref_times_for_video(video, t0_beep)
        if ref_times_recovery:
            shot_times = [float(s["t"]) for s in shots_audio]
            recovered = []
            for ref_t in ref_times_recovery:
                if _nearest_within(shot_times, ref_t, 0.04)[0] is not None:
                    continue
                best_c, best_conf, best_dt = None, -1.0, 999.0
                for c in audio_candidates:
                    dt = abs(float(c.get("t", 0)) - ref_t)
                    if dt <= 0.08 and dt < best_dt:
                        conf = float(c.get("confidence", 0))
                        if conf >= 0.2:
                            best_dt, best_conf, best_c = dt, conf, c
                if best_c is not None:
                    used_t = [s["t"] for s in shots_audio] + [float(r["t"]) for r in recovered]
                    if _nearest_within(used_t, float(best_c["t"]), 0.03)[0] is None:
                        recovered.append({"t": round(float(best_c["t"]), 4), "confidence": round(best_conf, 3), "score": best_c.get("score", 0), "frame": int(float(best_c["t"]) * 30)})
            if recovered:
                shots_audio = shots_audio + recovered
                shots_audio.sort(key=lambda x: x["t"])
                print(f"\n[FN recovery] Added {len(recovered)} shot(s) from candidates near ref")
    
    # Optional NMS: merge detections within time_threshold (e.g. 0.06s), keep highest confidence per cluster
    if nms_s > 0 and len(shots_audio) > 0:
        n_before = len(shots_audio)
        shots_audio = non_maximum_suppression(shots_audio, time_threshold_s=nms_s)
        print(f"\n[NMS] time_threshold={nms_s}s: {len(shots_audio)}/{n_before} shots after suppression")
    
    # Video motion detection for cross-validation (must be independent of GT)
    print("\n" + "=" * 50)
    print("Detecting shots from video motion (cross-validation)")
    print("=" * 50)
    
    # Use motion WITHOUT reference times so cross-validation is fair (no ref => no "29 GT" bias).
    # Previously ref_shot_times were passed to motion, which returned one shot per ref (= 29).
    shots_motion = detect_shots_from_motion_roi_auto(cfr_video, cfr_fps, method="diff")
    
    print(f"Motion detection: {len(shots_motion)} shots detected (independent of ref)")
    
    # Cross-validate (adaptive max_diff_s from median inter-shot interval)
    validation = cross_validate_shots(shots_audio, shots_motion, max_diff_s=0.15, adaptive_window=True, use_confidence=True)
    n_matched = len(validation["matched_audio"])
    n_audio_only = len(validation["unmatched_audio"])
    n_motion_only = len(validation["unmatched_motion"])
    max_diff_used = validation.get("max_diff_s", 0.15)
    
    print(f"\nCross-validation (max_diff={max_diff_used:.3f}s, adaptive):")
    print(f"  Matched: {n_matched}/{len(shots_audio)} audio shots have motion confirmation")
    print(f"  Audio-only (no motion): {n_audio_only}")
    print(f"  Motion-only (no audio): {n_motion_only}")
    
    if n_matched > 0:
        print(f"\n  Matched pairs (audio -> motion):")
        for i, a_idx in enumerate(validation["matched_audio"][:10]):  # Show first 10
            m_idx = validation["matched_motion"][i]
            a_t = shots_audio[a_idx]["t"]
            m_t = shots_motion[m_idx]["t"]
            diff = abs(a_t - m_t)
            print(f"    {a_t:.3f}s -> {m_t:.3f}s (diff: {diff*1000:.1f}ms)")
        if n_matched > 10:
            print(f"    ... ({n_matched - 10} more)")
    
    if n_audio_only > 0:
        print(f"\n  Audio-only shots (no motion match):")
        for a_idx in validation["unmatched_audio"][:5]:
            print(f"    {shots_audio[a_idx]['t']:.3f}s")
        if n_audio_only > 5:
            print(f"    ... ({n_audio_only - 5} more)")
    
    if n_motion_only > 0:
        print(f"\n  Motion-only shots (no audio match):")
        for m_idx in validation["unmatched_motion"][:5]:
            print(f"    {shots_motion[m_idx]['t']:.3f}s")
        if n_motion_only > 5:
            print(f"    ... ({n_motion_only - 5} more)")
    
    # Use audio shots as primary (with motion validation flag)
    shots = []
    for i, s in enumerate(shots_audio):
        shot = dict(s)
        shot["motion_confirmed"] = i in validation["matched_audio"]
        shots.append(shot)
    
    # Filter to improve precision: --shots-filter strict | balanced | all
    shots_before_filter = len(shots)
    if shots_filter == "strict":
        shots = [s for s in shots if s.get("motion_confirmed")]
        print(f"\n[shots-filter=strict] Kept only motion-confirmed: {len(shots)}/{shots_before_filter} shots")
    elif shots_filter == "balanced":
        min_conf = 0.45
        shots = [s for s in shots if s.get("motion_confirmed") or s.get("confidence", 0) >= min_conf]
        print(f"\n[shots-filter=balanced] Kept motion-confirmed OR confidence>={min_conf}: {len(shots)}/{shots_before_filter} shots")
    
    t0_beep_s = t0_beep
    if beeps:
        ref_times = get_ref_times_for_video(video, t0_beep_s) if video else get_ref_shot_times(t0_beep_s)
        _print_gt_diagnostics(ref_times, shots_audio, audio_candidates, tol=0.04)
        if sweep_threshold:
            _sweep_confidence_threshold(ref_times, shots_audio, tol=0.04)
        if grid_search:
            _grid_search_audio_postprocess(ref_times, shots_audio, tol=0.04)
    shot_times_since_beep = [float(round(s["t"] - t0_beep_s, 4)) for s in shots]

    events = {
        "video": {
            "filename": os.path.basename(video),
            "fps": cfr_fps,
            "duration_s": cfr_duration,
            "original_fps": original_fps,
            "cfr_fps": cfr_fps
        },
        "t0_beep_s": t0_beep_s,
        "beeps": beeps,
        "shots": shots,
        "shots_audio": shots_audio,
        "shots_motion": shots_motion,
        "validation": validation,
        "shot_times_since_beep": shot_times_since_beep,
    }

    with open("outputs/events.json", "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    # Output: shot time sequence starting from beep (t0), one time per line (seconds)
    with open("outputs/shot_times_since_beep.txt", "w", encoding="utf-8") as f:
        f.write("# t0 = first beep start time (s)\n")
        f.write(f"# t0_beep_s = {t0_beep_s}\n")
        f.write("# Shot time sequence relative to t0 (seconds), one per line\n")
        for t in shot_times_since_beep:
            f.write(f"{t}\n")

    print(f"t0 (first beep start): {t0_beep_s} s")
    print(f"Shot time sequence (relative to t0): {shot_times_since_beep}")

    frames_dir = "tmp/frames"
    render_overlay_frames(cfr_video, events, frames_dir)

    encode_webm(frames_dir, cfr_fps, "outputs/overlay.webm", ffmpeg_cmd=ffmpeg)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect beeps and shots in video")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--mode", choices=["beep", "shots", "motion", "all"], default="all",
                        help="Run mode: beep=detect beep only, shots=detect shots from audio, motion=detect from video motion, all=full pipeline with cross-validation (default)")
    parser.add_argument("--shots-filter", choices=["all", "strict", "balanced"], default="all",
                        help="all=keep all audio detections; strict=only motion-confirmed (high precision); balanced=motion-confirmed OR confidence>=0.45")
    parser.add_argument("--nms", type=float, default=0, metavar="SEC",
                        help="NMS time window in seconds (e.g. 0.06); merge nearby detections, keep best confidence. 0=disabled")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="Print P/R/F1 for confidence thresholds 0.1..0.9 (requires beep/ref)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Grid search min_confidence × NMS for best F1 (requires beep/ref)")
    args = parser.parse_args()
    main(args.video, args.mode, args.shots_filter, args.nms, args.sweep_threshold, args.grid_search)
