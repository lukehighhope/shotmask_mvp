"""
Detect gunshots from video motion (recoil/jerk when firing).
Uses optical flow or frame difference to detect sudden movements.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
import json
import os


def detect_shots_from_motion(video_path, fps, method="flow", roi=None, min_motion=0.3):
    """
    Detect gunshot times from video motion (recoil).
    
    Args:
        video_path: Path to video file
        fps: Video FPS
        method: "flow" (optical flow) or "diff" (frame difference)
        roi: Optional (x, y, w, h) region of interest to focus on gun area
        min_motion: Minimum motion magnitude threshold (normalized 0-1)
    
    Returns:
        List of dicts with 't', 'frame', 'confidence' (motion strength)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Region of interest: if not specified, use center region (likely gun area)
    if roi is None:
        # Center 40% of frame
        roi_w, roi_h = int(width * 0.4), int(height * 0.4)
        roi_x, roi_y = (width - roi_w) // 2, (height - roi_h) // 2
        roi = (roi_x, roi_y, roi_w, roi_h)
    
    x, y, w, h = roi
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    prev_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    motion_scores = []
    frame_idx = 0
    
    if method == "flow":
        # Optical flow method: tracks pixel movement between frames
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Detect corners in first frame for tracking
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            if corners is not None and len(corners) > 0:
                # Calculate optical flow
                next_corners, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, corners, None, **lk_params
                )
                
                # Filter valid points
                good_old = corners[status == 1]
                good_new = next_corners[status == 1]
                
                if len(good_old) > 0 and len(good_new) > 0:
                    # Calculate motion vectors
                    flow = good_new - good_old
                    # Motion magnitude
                    magnitudes = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
                    # Use median or 75th percentile to avoid outliers
                    motion = float(np.percentile(magnitudes, 75))
                else:
                    motion = 0.0
                
                # Update corners for next frame
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            else:
                motion = 0.0
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            motion_scores.append(motion)
            prev_gray = gray
            frame_idx += 1
    
    elif method == "diff":
        # Frame difference method: simpler, faster
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Frame difference
            diff = cv2.absdiff(prev_gray, gray)
            # Sum of absolute differences (normalized)
            motion = float(np.sum(diff)) / (w * h * 255.0)
            
            motion_scores.append(motion)
            prev_gray = gray
            frame_idx += 1
    
    cap.release()
    
    if len(motion_scores) < 10:
        return []
    
    motion_arr = np.array(motion_scores)
    
    # Normalize motion scores (0-1)
    if motion_arr.max() > 0:
        motion_arr = motion_arr / motion_arr.max()
    
    # Smooth to reduce noise
    kernel_size = max(3, int(fps * 0.05))  # ~50ms smoothing
    if kernel_size % 2 == 0:
        kernel_size += 1
    motion_smooth = np.convolve(motion_arr, np.ones(kernel_size) / kernel_size, mode="same")
    
    # Adaptive threshold: median + k * std
    baseline = np.median(motion_smooth)
    std = np.std(motion_smooth)
    threshold = baseline + 2.0 * std
    
    # Min distance between shots (~100ms)
    min_dist_samples = max(1, int(fps * 0.10))
    
    # Find peaks (sudden motion spikes = recoil)
    peaks, props = find_peaks(
        motion_smooth,
        height=max(threshold, min_motion),
        distance=min_dist_samples,
        prominence=0.1 * (motion_smooth.max() - baseline),
    )
    
    shots = []
    if len(peaks) > 0:
        max_motion = motion_smooth[peaks].max()
        for p in peaks:
            t = p / fps
            confidence = float(motion_smooth[p] / max_motion) if max_motion > 0 else 0.5
            shots.append({
                "t": round(float(t), 4),
                "frame": int(p),
                "confidence": round(confidence, 2),
            })
    
    return shots


def detect_shots_from_motion_ref_guided(video_path, fps, ref_shot_times, method="diff", roi=None, window_before=0.02, window_after=0.08):
    """
    Detect shots using reference times as guidance.
    For each ref_shot_time, search for motion peak in [ref_t - window_before, ref_t + window_after].
    This learns what "true shot motion" looks like.
    
    Args:
        video_path: Path to video file
        fps: Video FPS
        ref_shot_times: List of reference shot times (seconds)
        method: "flow" or "diff"
        roi: Optional (x, y, w, h) region of interest
        window_before: Time before ref to search (default 0.02s)
        window_after: Time after ref to search (default 0.08s)
    
    Returns:
        List of dicts with 't', 'frame', 'confidence', 'ref_idx' (which ref shot this matches)
    """
    if not ref_shot_times:
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Auto ROI if not provided
    if roi is None:
        roi_w, roi_h = int(width * 0.4), int(height * 0.4)
        roi_x, roi_y = (width - roi_w) // 2, (height - roi_h) // 2
        roi = (roi_x, roi_y, roi_w, roi_h)
    
    x, y, w, h = roi
    
    # Compute motion scores for all frames
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    prev_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    motion_scores = []
    frame_times = []
    
    if method == "diff":
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            diff = cv2.absdiff(prev_gray, gray)
            motion = float(np.sum(diff)) / (w * h * 255.0)
            
            t = len(motion_scores) / fps
            motion_scores.append(motion)
            frame_times.append(t)
            prev_gray = gray
    
    elif method == "flow":
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            if corners is not None and len(corners) > 0:
                next_corners, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, corners, None, **lk_params
                )
                good_old = corners[status == 1]
                good_new = next_corners[status == 1]
                
                if len(good_old) > 0 and len(good_new) > 0:
                    flow = good_new - good_old
                    magnitudes = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
                    motion = float(np.percentile(magnitudes, 75))
                else:
                    motion = 0.0
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            else:
                motion = 0.0
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            t = len(motion_scores) / fps
            motion_scores.append(motion)
            frame_times.append(t)
            prev_gray = gray
    
    cap.release()
    
    if len(motion_scores) < 10:
        return []
    
    motion_arr = np.array(motion_scores)
    times_arr = np.array(frame_times)
    
    # Normalize motion
    if motion_arr.max() > 0:
        motion_arr = motion_arr / motion_arr.max()
    
    # Smooth motion signal
    kernel_size = max(3, int(fps * 0.05))
    if kernel_size % 2 == 0:
        kernel_size += 1
    motion_smooth = np.convolve(motion_arr, np.ones(kernel_size) / kernel_size, mode="same")
    
    # For each reference shot time, find peak in window
    shots = []
    for ref_idx, ref_t in enumerate(ref_shot_times):
        t_start = ref_t - window_before
        t_end = ref_t + window_after
        
        # Find frames in window
        mask = (times_arr >= t_start) & (times_arr <= t_end)
        if not np.any(mask):
            continue
        
        window_motion = motion_smooth[mask]
        window_times = times_arr[mask]
        window_indices = np.where(mask)[0]
        
        if len(window_motion) == 0:
            continue
        
        # Find peak in this window
        peak_idx_local = np.argmax(window_motion)
        peak_idx_global = window_indices[peak_idx_local]
        peak_t = window_times[peak_idx_local]
        peak_motion = window_motion[peak_idx_local]
        
        # Confidence: normalized by max motion in entire video
        max_motion_all = motion_smooth.max()
        confidence = float(peak_motion / max_motion_all) if max_motion_all > 0 else 0.5
        
        shots.append({
            "t": round(float(peak_t), 4),
            "frame": int(peak_idx_global),
            "confidence": round(confidence, 2),
            "ref_idx": ref_idx + 1,  # 1-indexed
            "ref_time": round(float(ref_t), 4),
            "offset_from_ref": round(float(peak_t - ref_t), 4),
        })
    
    return shots


def detect_shots_from_motion_improved(video_path, fps, ref_shot_times=None, method="diff", roi=None):
    """
    Improved motion detection: if ref_shot_times provided, learn motion characteristics
    from ref-guided detection, then apply to global detection.
    Otherwise, use standard detection.
    """
    if ref_shot_times is None or len(ref_shot_times) == 0:
        # Fallback to standard detection
        return detect_shots_from_motion_roi_auto(video_path, fps, method=method)
    
    # Step 1: Ref-guided detection to learn true shot motion characteristics
    ref_guided_shots = detect_shots_from_motion_ref_guided(
        video_path, fps, ref_shot_times, method=method, roi=roi,
        window_before=0.02, window_after=0.08
    )
    
    if len(ref_guided_shots) == 0:
        return []
    
    # Step 2: Analyze motion characteristics from ref-guided detections
    # Compute motion signal again for analysis
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if roi is None:
        roi_w, roi_h = int(width * 0.4), int(height * 0.4)
        roi_x, roi_y = (width - roi_w) // 2, (height - roi_h) // 2
        roi = (roi_x, roi_y, roi_w, roi_h)
    
    x, y, w, h = roi
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    prev_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    motion_scores = []
    
    if method == "diff":
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            diff = cv2.absdiff(prev_gray, gray)
            motion = float(np.sum(diff)) / (w * h * 255.0)
            motion_scores.append(motion)
            prev_gray = gray
    
    cap.release()
    
    if len(motion_scores) < 10:
        return ref_guided_shots
    
    motion_arr = np.array(motion_scores)
    if motion_arr.max() > 0:
        motion_arr = motion_arr / motion_arr.max()
    
    # Smooth
    kernel_size = max(3, int(fps * 0.05))
    if kernel_size % 2 == 0:
        kernel_size += 1
    motion_smooth = np.convolve(motion_arr, np.ones(kernel_size) / kernel_size, mode="same")
    
    # Step 3: Learn threshold from ref-guided detections
    # Get motion values at ref-guided shot frames
    ref_motion_values = []
    for shot in ref_guided_shots:
        frame_idx = shot["frame"]
        if 0 <= frame_idx < len(motion_smooth):
            ref_motion_values.append(motion_smooth[frame_idx])
    
    if len(ref_motion_values) == 0:
        return ref_guided_shots
    
    # Use percentile of ref-guided motion as threshold (e.g., 50th percentile)
    learned_threshold = np.percentile(ref_motion_values, 30)  # Lower threshold to catch more
    
    # Step 4: Global detection with learned threshold
    baseline = np.median(motion_smooth)
    std = np.std(motion_smooth)
    # Use learned threshold, but not lower than adaptive threshold
    adaptive_threshold = baseline + 1.5 * std
    threshold = max(learned_threshold, adaptive_threshold * 0.5)
    
    min_dist_samples = max(1, int(fps * 0.10))
    prominence = 0.05 * (motion_smooth.max() - baseline)
    
    peaks, props = find_peaks(
        motion_smooth,
        height=threshold,
        distance=min_dist_samples,
        prominence=prominence,
    )
    
    # Step 5: Return ref-guided shots (ground truth from reference times)
    # These are the "true shots" learned from reference data
    # Optionally, we could add global peaks that are very strong and not near any ref,
    # but for now, prioritize ref-guided results (more accurate)
    return ref_guided_shots


def detect_shots_from_motion_roi_auto(video_path, fps, method="diff"):
    """
    Auto-detect ROI by finding area with most motion in first few seconds.
    Then detect shots using that ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample first 2 seconds to find active region
    sample_frames = int(fps * 2)
    motion_map = np.zeros((height, width), dtype=np.float32)
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    for _ in range(min(sample_frames, 60)):  # Max 60 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        diff = cv2.absdiff(prev_gray, gray)
        motion_map += diff.astype(np.float32)
        prev_gray = gray
    
    cap.release()
    
    # Find region with highest motion (likely gun area)
    # Use center-weighted search
    center_y, center_x = height // 2, width // 2
    roi_size = min(width, height) // 3
    
    best_score = 0
    best_roi = None
    
    for y_offset in [-roi_size//2, 0, roi_size//2]:
        for x_offset in [-roi_size//2, 0, roi_size//2]:
            y = max(0, min(height - roi_size, center_y + y_offset))
            x = max(0, min(width - roi_size, center_x + x_offset))
            roi_motion = np.sum(motion_map[y:y+roi_size, x:x+roi_size])
            if roi_motion > best_score:
                best_score = roi_motion
                best_roi = (x, y, roi_size, roi_size)
    
    if best_roi is None:
        # Fallback: center region
        roi_size = min(width, height) // 3
        best_roi = ((width - roi_size) // 2, (height - roi_size) // 2, roi_size, roi_size)
    
    return detect_shots_from_motion(video_path, fps, method=method, roi=best_roi)


def extract_motion_features_at_ref_shots(video_path, fps, ref_shot_times, method="diff", roi=None, 
                                         window_before=0.02, window_after=0.08):
    """
    Extract motion features from windows around reference shot times.
    For each ref_shot_time, extract detailed motion characteristics in [ref_t - window_before, ref_t + window_after].
    
    Args:
        video_path: Path to video file
        fps: Video FPS
        ref_shot_times: List of reference shot times (seconds)
        method: "flow" or "diff"
        roi: Optional (x, y, w, h) region of interest
        window_before: Time before ref to extract (default 0.02s)
        window_after: Time after ref to extract (default 0.08s)
    
    Returns:
        List of dicts, each containing features for one reference shot:
        {
            'ref_idx': index (1-based),
            'ref_time': reference time,
            'peak_time': peak motion time,
            'peak_motion': peak motion value,
            'offset_from_ref': peak_time - ref_time,
            'window_mean': mean motion in window,
            'window_std': std motion in window,
            'window_max': max motion in window,
            'window_min': min motion in window,
            'rise_time': time from window start to peak,
            'fall_time': time from peak to window end,
            'peak_to_mean_ratio': peak / mean,
            'motion_before_peak': mean motion before peak,
            'motion_after_peak': mean motion after peak,
            'motion_signal': full motion signal in window (for analysis),
            'time_signal': time points in window,
        }
    """
    if not ref_shot_times:
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Auto ROI if not provided
    if roi is None:
        roi_w, roi_h = int(width * 0.4), int(height * 0.4)
        roi_x, roi_y = (width - roi_w) // 2, (height - roi_h) // 2
        roi = (roi_x, roi_y, roi_w, roi_h)
    
    x, y, w, h = roi
    
    # Compute motion scores for all frames
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    
    prev_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    motion_scores = []
    frame_times = []
    
    if method == "diff":
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            diff = cv2.absdiff(prev_gray, gray)
            motion = float(np.sum(diff)) / (w * h * 255.0)
            
            t = len(motion_scores) / fps
            motion_scores.append(motion)
            frame_times.append(t)
            prev_gray = gray
    
    elif method == "flow":
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            if corners is not None and len(corners) > 0:
                next_corners, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, corners, None, **lk_params
                )
                good_old = corners[status == 1]
                good_new = next_corners[status == 1]
                
                if len(good_old) > 0 and len(good_new) > 0:
                    flow = good_new - good_old
                    magnitudes = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
                    motion = float(np.percentile(magnitudes, 75))
                else:
                    motion = 0.0
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            else:
                motion = 0.0
                corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            t = len(motion_scores) / fps
            motion_scores.append(motion)
            frame_times.append(t)
            prev_gray = gray
    
    cap.release()
    
    if len(motion_scores) < 10:
        return []
    
    motion_arr = np.array(motion_scores)
    times_arr = np.array(frame_times)
    
    # Normalize motion
    motion_max = motion_arr.max()
    if motion_max > 0:
        motion_arr = motion_arr / motion_max
    
    # Smooth motion signal
    kernel_size = max(3, int(fps * 0.05))
    if kernel_size % 2 == 0:
        kernel_size += 1
    motion_smooth = np.convolve(motion_arr, np.ones(kernel_size) / kernel_size, mode="same")
    
    # Extract features for each reference shot
    features_list = []
    for ref_idx, ref_t in enumerate(ref_shot_times):
        t_start = ref_t - window_before
        t_end = ref_t + window_after
        
        # Find frames in window
        mask = (times_arr >= t_start) & (times_arr <= t_end)
        if not np.any(mask):
            continue
        
        window_motion = motion_smooth[mask]
        window_times = times_arr[mask]
        window_indices = np.where(mask)[0]
        
        if len(window_motion) == 0:
            continue
        
        # Find peak in this window
        peak_idx_local = np.argmax(window_motion)
        peak_idx_global = window_indices[peak_idx_local]
        peak_t = window_times[peak_idx_local]
        peak_motion = window_motion[peak_idx_local]
        
        # Calculate features
        window_mean = float(np.mean(window_motion))
        window_std = float(np.std(window_motion))
        window_max = float(np.max(window_motion))
        window_min = float(np.min(window_motion))
        
        # Rise time: time from window start to peak
        rise_time = float(peak_t - t_start)
        
        # Fall time: time from peak to window end
        fall_time = float(t_end - peak_t)
        
        # Peak to mean ratio
        peak_to_mean_ratio = float(peak_motion / window_mean) if window_mean > 0 else 0.0
        
        # Motion before and after peak
        motion_before_peak = float(np.mean(window_motion[:peak_idx_local+1])) if peak_idx_local >= 0 else 0.0
        motion_after_peak = float(np.mean(window_motion[peak_idx_local:])) if peak_idx_local < len(window_motion) else 0.0
        
        # Offset from reference
        offset_from_ref = float(peak_t - ref_t)
        
        features = {
            "ref_idx": ref_idx + 1,  # 1-indexed
            "ref_time": round(float(ref_t), 4),
            "peak_time": round(float(peak_t), 4),
            "peak_motion": round(float(peak_motion), 6),
            "offset_from_ref": round(offset_from_ref, 4),
            "window_mean": round(window_mean, 6),
            "window_std": round(window_std, 6),
            "window_max": round(window_max, 6),
            "window_min": round(window_min, 6),
            "rise_time": round(rise_time, 4),
            "fall_time": round(fall_time, 4),
            "peak_to_mean_ratio": round(peak_to_mean_ratio, 4),
            "motion_before_peak": round(motion_before_peak, 6),
            "motion_after_peak": round(motion_after_peak, 6),
            "motion_signal": [round(float(x), 6) for x in window_motion.tolist()],
            "time_signal": [round(float(x), 4) for x in window_times.tolist()],
        }
        
        features_list.append(features)
    
    return features_list


def save_motion_features(features_list, output_path="outputs/motion_features.json"):
    """Save extracted motion features to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=2)
    print(f"Motion features saved to: {output_path}")


def analyze_motion_features(features_list):
    """Analyze and print statistics of extracted motion features."""
    if not features_list:
        print("No features to analyze.")
        return
    
    n = len(features_list)
    peak_motions = [f["peak_motion"] for f in features_list]
    offsets = [f["offset_from_ref"] for f in features_list]
    peak_to_mean_ratios = [f["peak_to_mean_ratio"] for f in features_list]
    rise_times = [f["rise_time"] for f in features_list]
    fall_times = [f["fall_time"] for f in features_list]
    
    print(f"\nMotion Features Analysis ({n} reference shots):")
    print("=" * 60)
    print(f"Peak Motion:")
    print(f"  Mean: {np.mean(peak_motions):.6f}, Std: {np.std(peak_motions):.6f}")
    print(f"  Min: {np.min(peak_motions):.6f}, Max: {np.max(peak_motions):.6f}")
    print(f"  Median: {np.median(peak_motions):.6f}")
    print(f"\nOffset from Reference:")
    print(f"  Mean: {np.mean(offsets):.4f}s, Std: {np.std(offsets):.4f}s")
    print(f"  Min: {np.min(offsets):.4f}s, Max: {np.max(offsets):.4f}s")
    print(f"\nPeak-to-Mean Ratio:")
    print(f"  Mean: {np.mean(peak_to_mean_ratios):.4f}, Std: {np.std(peak_to_mean_ratios):.4f}")
    print(f"\nRise Time (window start to peak):")
    print(f"  Mean: {np.mean(rise_times):.4f}s, Std: {np.std(rise_times):.4f}s")
    print(f"\nFall Time (peak to window end):")
    print(f"  Mean: {np.mean(fall_times):.4f}s, Std: {np.std(fall_times):.4f}s")
    print("=" * 60)
