"""
Train a single LogReg (with MFCC) on multiple videos that have beep + ref.
More (diverse) positives and negatives than single-video training → better precision/recall.

Usage:
  python train_logreg_multivideo.py --folder 01032026
  python train_logreg_multivideo.py --videos 1.mp4 01032026/2.mp4 01032026/3.mp4
"""
import argparse
import json
import os
import subprocess
import numpy as np
import soundfile as sf

from detectors.shot_audio import (
    detect_shots_improved,
    compute_feature_at_time,
    load_calibrated_params,
    CALIBRATED_PARAMS_FILENAME,
)
from detectors.shot_logreg import train_logreg
from ref_from_image import get_ref_times_for_video, get_beep_t_for_video


def get_ffmpeg():
    import shutil
    exe = shutil.which("ffmpeg") or os.environ.get("FFMPEG")
    if exe:
        return exe
    for p in [
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.exists(p):
            return p
    return "ffmpeg"


def get_ffprobe():
    import shutil
    exe = shutil.which("ffprobe") or os.environ.get("FFPROBE")
    if exe:
        return exe
    for p in [
        r"C:\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
    ]:
        if os.path.exists(p):
            return p
    return "ffprobe"


def extract_audio(video_path, out_dir="tmp"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join(out_dir, f"audio_{base}.wav")
    ffmpeg = get_ffmpeg()
    subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-ac", "1", "-ar", "48000", "-vn", out],
        check=True,
        capture_output=True,
    )
    return out


def get_fps_duration(video_path):
    import json as _json
    ffprobe = get_ffprobe()
    out = subprocess.check_output(
        [
            ffprobe,
            "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration",
            "-of", "json",
            video_path,
        ],
        text=True,
    )
    d = _json.loads(out)
    r = d["streams"][0]["r_frame_rate"]
    fps = eval(r)
    duration = float(d["streams"][0]["duration"])
    return fps, duration


GT_COVER_TOL = 0.04
MISS_WEIGHT = 4.5
window_before = 0.02
window_after = 0.08


def build_xy_weights_for_video(audio_path, fps, ref_times, cal_cfg):
    """Return (X, y, weights) for one video. ref_times = list of GT shot times (abs)."""
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float64)

    shots_result = detect_shots_improved(
        data, sr, fps,
        cluster_window_sec=cal_cfg.get("cluster_window_sec", 0.25),
        mad_k=cal_cfg.get("mad_k", 6.0),
        candidate_min_dist_ms=cal_cfg.get("candidate_min_dist_ms", 50),
        score_weights=cal_cfg.get("score_weights"),
        min_confidence_threshold=None,
        logreg_model=None,
        return_candidates=True,
        return_feature_context=True,
        use_mfcc=True,
    )
    if not isinstance(shots_result, tuple) or len(shots_result) != 3:
        return [], [], []
    shots_cur, candidates, context = shots_result
    if not candidates:
        return [], [], []

    X, y, weights = [], [], []
    cand_times = [float(c.get("t", 0.0)) for c in candidates]
    shot_times_cur = [float(s.get("t", 0.0)) for s in (shots_cur or [])]

    # Candidates: TP or FP
    for c in candidates:
        t = c["t"]
        is_pos = any((t >= rt - window_before) and (t <= rt + window_after) for rt in ref_times)
        y.append(1 if is_pos else 0)
        weights.append(1.0)
        X.append([
            float(c.get("onset_log", 0.0)),
            float(c.get("r1_log", 0.0)),
            float(c.get("r2_log", 0.0)),
            float(c.get("flatness", 0.0)),
            float(c.get("attack_norm", 0.0)),
            float(c.get("E_low_ratio", 0.0)),
            1.0 if c.get("hard_neg_slam", False) else 0.0,
            1.0 if c.get("hard_neg_metal", False) else 0.0,
            float(c.get("mfcc_band", 0.0)),
        ])

    # FN: GT with no detection within tol; add nearest candidate as positive with MISS_WEIGHT
    for rt in ref_times:
        if any(abs(st - rt) <= GT_COVER_TOL for st in shot_times_cur):
            continue
        idx = int(np.argmin(np.abs(np.asarray(cand_times) - float(rt))))
        t_peak = float(cand_times[idx])
        if abs(t_peak - float(rt)) < window_after:
            c = candidates[idx]
            y.append(1)
            weights.append(MISS_WEIGHT)
            X.append([
                float(c.get("onset_log", 0.0)),
                float(c.get("r1_log", 0.0)),
                float(c.get("r2_log", 0.0)),
                float(c.get("flatness", 0.0)),
                float(c.get("attack_norm", 0.0)),
                float(c.get("E_low_ratio", 0.0)),
                1.0 if c.get("hard_neg_slam", False) else 0.0,
                1.0 if c.get("hard_neg_metal", False) else 0.0,
                float(c.get("mfcc_band", 0.0)),
            ])

    # Miss: ref with no candidate in window → synthesize feature at GT (mfcc=0)
    for rt in ref_times:
        if any(abs(c["t"] - rt) <= window_after for c in candidates):
            continue
        feat = compute_feature_at_time(context, rt, window_before, window_after)
        y.append(1)
        weights.append(MISS_WEIGHT)
        X.append([
            float(feat.get("onset_log", 0.0)),
            float(feat.get("r1_log", 0.0)),
            float(feat.get("r2_log", 0.0)),
            float(feat.get("flatness", 0.0)),
            float(feat.get("attack_norm", 0.0)),
            float(feat.get("E_low_ratio", 0.0)),
            1.0 if feat.get("hard_neg_slam", False) else 0.0,
            1.0 if feat.get("hard_neg_metal", False) else 0.0,
            0.0,
        ])

    return X, y, weights


def main():
    ap = argparse.ArgumentParser(description="Train LogReg (9-dim with MFCC) on multiple videos with ref")
    ap.add_argument("--videos", nargs="+", help="Video paths")
    ap.add_argument("--folder", help="Folder of mp4 files (e.g. 01032026)")
    ap.add_argument("--out", default=None, help="Output json path (default: calibrated_detector_params.json)")
    args = ap.parse_args()

    videos = []
    if args.videos:
        videos = [os.path.abspath(v) for v in args.videos]
    if args.folder:
        folder = os.path.abspath(args.folder)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".mp4"):
                videos.append(os.path.join(folder, f))
    if not videos:
        print("Provide --videos or --folder")
        return 1

    cal_cfg = load_calibrated_params() or {}
    all_X, all_y, all_w = [], [], []

    for vp in videos:
        if not os.path.isfile(vp):
            print("Skip (not found):", vp)
            continue
        print("Processing:", vp)
        try:
            audio_path = extract_audio(vp)
            fps, _ = get_fps_duration(vp)
        except Exception as e:
            print("  Error extract:", e)
            continue
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        beep_t = get_beep_t_for_video(vp, audio_path, fps)
        if beep_t <= 0:
            print("  No beep (*beep.txt / overrides / detect), skip")
            continue
        ref_times = get_ref_times_for_video(vp, beep_t)
        if not ref_times:
            print("  No ref times, skip")
            continue
        print(f"  Ref: {len(ref_times)} shots (from jpg or global)")
        X, y, w = build_xy_weights_for_video(audio_path, fps, ref_times, cal_cfg)
        if not X:
            print("  No candidates, skip")
            continue
        n_pos = sum(1 for yi in y if yi == 1)
        n_neg = sum(1 for yi in y if yi == 0)
        print(f"  Candidates: {len(X)}, pos={n_pos}, neg={n_neg}")
        all_X.extend(X)
        all_y.extend(y)
        all_w.extend(w)

    if not all_X or len(set(all_y)) < 2:
        print("Not enough data or class variety to train.")
        return 1

    X = np.asarray(all_X, dtype=np.float64)
    y = np.asarray(all_y, dtype=np.float64)
    w = np.asarray(all_w, dtype=np.float64)
    print(f"\nTotal samples: {len(X)}, pos={int(y.sum())}, neg={len(y)-int(y.sum())}")
    model = train_logreg(X, y, sample_weight=w)
    out_path = args.out or os.path.join(os.getcwd(), CALIBRATED_PARAMS_FILENAME)
    cal = load_calibrated_params() or {}
    cal["logreg_model"] = model
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"LogReg (9-dim with MFCC) saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
