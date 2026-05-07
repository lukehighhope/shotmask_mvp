"""
Extract shell casing visual features from videos around annotated shot times.

Strategy:
  - Camera is GoPro-style, mounted on/near the gun (first-person view)
  - Shell ejects from the RIGHT side of the gun, flies RIGHT (+x) and UP (-y)
  - For each shot time t in *cali.txt, extract frames in [t-0.05s, t+0.4s]
  - Use optical flow with camera-motion compensation in the ejection zone
  - For non-shot windows (negative samples), extract clips from quiet periods
  - Save features to outputs/shell_features_<split>.npz

Usage:
    python extract_shell_features.py
    python extract_shell_features.py --split val
    python extract_shell_features.py --split all --out outputs/shell_features_all.npz
"""

import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path


DATA_ROOT  = Path("traning data")
SPLIT_JSON = DATA_ROOT / "dataset_split.json"

# Window around each shot time to search for shell casing
WINDOW_BEFORE = 0.05   # seconds before shot
WINDOW_AFTER  = 0.40   # seconds after shot

# Ejection zone: shell flies from gun ejection port (right side) UP and to the RIGHT.
# Hands/gun are in lower-center frame (~y: 45-90%). Shell appears in upper-right (~y: 5-42%).
EJECT_ZONE_X0 = 0.30
EJECT_ZONE_X1 = 0.85
EJECT_ZONE_Y0 = 0.05
EJECT_ZONE_Y1 = 0.42

# Optical flow thresholds (after camera-motion compensation)
MIN_RELATIVE_FLOW  = 5.0   # pixels/frame relative rightward motion to count as shell
MIN_BRIGHT_IN_ZONE = 90    # brightness threshold inside the ejection zone


def load_times(txt_path):
    """Load shot/beep times from a text file (one float per line).

    Skips empty lines, lines starting with '#', and non-float lines.
    """
    times = []
    try:
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    times.append(float(line))
                except ValueError:
                    pass
    except Exception:
        pass
    return sorted(times)


def get_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def extract_frame_window(video_path, t_start, t_end, fps):
    """Extract frames from video between t_start and t_end (seconds)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_start = max(0, int(t_start * fps))
    frame_end   = int(t_end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frames = []
    for _ in range(frame_end - frame_start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def detect_shell_optical_flow(frames, shot_offset_frames=None):
    """
    Detect shell casings using optical flow with camera-motion compensation.

    Context: GoPro-style camera mounted on/near the gun. Shell ejects from the
    RIGHT side of the gun, flies RIGHT (+x) and UP (-y) in the frame.

    Algorithm per consecutive frame pair:
      1. Estimate camera motion using sparse Lucas-Kanade flow across the full frame
         (median of all flow vectors = camera translation)
      2. Compute dense Farneback flow in the EJECTION ZONE only (right 65%, upper 60%)
      3. Subtract camera motion -> residual flow = real object motion
      4. Shell score = sum of (brightness * rightward_residual_flow) in zone

    Returns 8-dim feature vector.
    """
    if len(frames) < 3:
        return np.zeros(8, dtype=np.float32)

    height, width = frames[0].shape[:2]
    n = len(frames)

    if shot_offset_frames is None:
        # Shell appears 50-150ms after the bang; skip 2 frames for muzzle flash
        shot_offset_frames = max(1, int(WINDOW_BEFORE * 30) + 2)

    # Ejection zone pixel bounds
    x0 = int(width  * EJECT_ZONE_X0)
    x1 = int(width  * EJECT_ZONE_X1)
    y0 = int(height * EJECT_ZONE_Y0)
    y1 = int(height * EJECT_ZONE_Y1)

    frame_scores = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, n):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # --- Camera motion estimation (sparse LK over full frame) ---
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01,
                                      minDistance=10, blockSize=7)
        cam_dx, cam_dy = 0.0, 0.0
        if pts is not None and len(pts) >= 5:
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
            )
            good = status.ravel() == 1
            if good.sum() >= 5:
                diff_vecs = (pts_next[good] - pts[good]).reshape(-1, 2)
                cam_dx = float(np.median(diff_vecs[:, 0]))
                cam_dy = float(np.median(diff_vecs[:, 1]))

        # --- Dense flow in ejection zone only ---
        zone_prev = prev_gray[y0:y1, x0:x1]
        zone_curr = curr_gray[y0:y1, x0:x1]
        flow = cv2.calcOpticalFlowFarneback(
            zone_prev, zone_curr, None,
            pyr_scale=0.5, levels=2, winsize=12, iterations=2,
            poly_n=5, poly_sigma=1.2, flags=0
        )

        # Residual flow after subtracting camera motion
        rel_fx = flow[:, :, 0] - cam_dx   # + = rightward relative to background
        rel_fy = flow[:, :, 1] - cam_dy   # - = upward relative to background

        # Shell indicator: bright pixels moving RIGHT or UP relative to background
        # Score = FRACTION of zone pixels that satisfy both conditions (scale-invariant)
        zone_bright = curr_gray[y0:y1, x0:x1]
        bright_ok   = zone_bright > MIN_BRIGHT_IN_ZONE

        rightward_ok = rel_fx > MIN_RELATIVE_FLOW    # relative rightward motion
        upward_ok    = rel_fy < -MIN_RELATIVE_FLOW   # relative upward motion
        directional_ok = rightward_ok | upward_ok

        shell_pixels = bright_ok & directional_ok
        score = float(np.mean(shell_pixels))  # fraction of zone (0.0 = none, 0.1 = 10% of zone)
        frame_scores.append(score)

        prev_gray = curr_gray

    frame_scores = np.array(frame_scores, dtype=np.float32)
    n_scored = len(frame_scores)

    before = frame_scores[:shot_offset_frames] if shot_offset_frames <= n_scored else frame_scores
    after  = frame_scores[shot_offset_frames:] if shot_offset_frames < n_scored  else np.array([0.0])

    score_before = float(np.max(before)) if len(before) > 0 else 0.0
    score_after  = float(np.max(after))  if len(after)  > 0 else 0.0
    mean_after   = float(np.mean(after)) if len(after)  > 0 else 0.0

    # Spike: how much bigger is the after-shot peak vs the pre-shot baseline
    spike = score_after / (score_before + 1e-6)
    peak_frame = int(np.argmax(frame_scores))
    peak_after = 1.0 if peak_frame >= shot_offset_frames else 0.0

    # How many after-shot frames have elevated activity
    thresh = max(score_before * 1.5, 1e-5)
    active_frac = float(np.mean(after > thresh)) if len(after) > 0 else 0.0

    # Score = fraction of ejection zone pixels with shell-like motion (0.0 to ~0.15)
    # NORM calibrated so that a shell covering ~8% of zone gives output ~1.0
    NORM = 0.08
    return np.array([
        min(1.0, score_after  / NORM),          # 0: peak score after shot  (0=silent, 1=clear shell)
        min(1.0, score_before / NORM),           # 1: baseline before shot   (low = clean trigger)
        min(1.0, (score_after - score_before) / NORM),  # 2: absolute spike (after minus before)
        min(1.0, mean_after   / NORM),           # 3: mean activity after shot
        peak_after,                               # 4: peak frame is after shot time (0/1)
        active_frac,                              # 5: fraction of after-shot frames with activity
        float(peak_frame) / max(n_scored-1, 1),  # 6: normalized time of peak
        min(1.0, max(0.0, spike - 1.0) / 5.0),  # 7: relative spike above baseline (>1 = improvement)
    ], dtype=np.float32)


# Alias so process_video still works
detect_shell_blobs_in_window = detect_shell_optical_flow


def get_negative_windows(shot_times, video_duration, window_len, n_neg=None):
    """Sample negative windows (no shot) from quiet periods."""
    min_gap = 1.0
    candidates = []
    t = 0.0
    while t + window_len < video_duration:
        too_close = any(abs(t + window_len / 2 - s) < min_gap for s in shot_times)
        if not too_close:
            candidates.append((t, t + window_len))
        t += window_len / 2

    if n_neg is None:
        n_neg = len(shot_times)

    if len(candidates) == 0:
        return []
    step = max(1, len(candidates) // n_neg)
    return candidates[::step][:n_neg]


def process_video(video_path, cali_path, split_label):
    """Process one video: extract positive (shot) and negative (no-shot) features."""
    shot_times = load_times(cali_path)
    if not shot_times:
        print(f"  No shot times, skipping.")
        return None, None

    fps = get_fps(video_path)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps

    window_len = WINDOW_BEFORE + WINDOW_AFTER
    shot_offset_frames = max(1, int(WINDOW_BEFORE * fps)) + 2

    X_pos, X_neg = [], []

    for t in shot_times:
        t_start = max(0, t - WINDOW_BEFORE)
        t_end   = min(duration, t + WINDOW_AFTER)
        frames  = extract_frame_window(video_path, t_start, t_end, fps)
        if len(frames) < 3:
            continue
        feat = detect_shell_optical_flow(frames, shot_offset_frames)
        X_pos.append(feat)

    neg_windows = get_negative_windows(shot_times, duration, window_len)
    for t_start, t_end in neg_windows:
        frames = extract_frame_window(video_path, t_start, t_end, fps)
        if len(frames) < 3:
            continue
        # For negatives: shot_offset=0 means "no specific shot event"
        # We still want to measure if there's any rightward-bright motion, but
        # the spike feature (feat[2]) will not be meaningful without a real shot.
        feat = detect_shell_optical_flow(frames, shot_offset_frames=0)
        X_neg.append(feat)

    if not X_pos and not X_neg:
        return None, None

    X = np.array(X_pos + X_neg, dtype=np.float32)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg), dtype=np.int32)

    print(f"  +{len(X_pos)} positive, -{len(X_neg)} negative samples")
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "all"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(SPLIT_JSON) as f:
        split_data = json.load(f)

    if args.split == "all":
        video_list = split_data.get("train", []) + split_data.get("val", [])
    else:
        video_list = split_data.get(args.split, [])

    if not video_list:
        print(f"No videos found for split '{args.split}'")
        return

    out_path = args.out or f"outputs/shell_features_{args.split}.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_X, all_y, video_labels = [], [], []

    for rel_path in video_list:
        video_path = DATA_ROOT / rel_path
        base       = str(video_path).replace(".mp4", "")
        cali_path  = Path(base + "cali.txt")

        if not video_path.exists():
            print(f"[SKIP] Video not found: {video_path}")
            continue
        if not cali_path.exists():
            print(f"[SKIP] No cali.txt for: {video_path.name}")
            continue

        print(f"Processing: {rel_path}")
        X, y = process_video(video_path, cali_path, args.split)
        if X is None:
            continue

        all_X.append(X)
        all_y.append(y)
        video_labels.extend([rel_path] * len(X))

    if not all_X:
        print("No features extracted.")
        return

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    np.savez(out_path, X=X_all, y=y_all, video_labels=np.array(video_labels))
    print(f"\nSaved {len(X_all)} samples ({y_all.sum()} positive, {(y_all==0).sum()} negative)")
    print(f"Features shape: {X_all.shape}")
    print(f"Output: {out_path}")
    print("\nFeature columns:")
    print("  0: score_after   peak shell score after shot (optical flow in ejection zone)")
    print("  1: score_before  baseline shell activity before shot")
    print("  2: spike_ratio   after/before ratio (high = shell appeared)")
    print("  3: mean_after    mean shell activity after shot")
    print("  4: peak_after    1 if the peak frame is after shot time")
    print("  5: active_frac   fraction of after-shot frames with elevated activity")
    print("  6: peak_time     normalized time of peak (0=early, 1=late in window)")
    print("  7: peak_sharp    how sharp/localized the peak is")


if __name__ == "__main__":
    main()
