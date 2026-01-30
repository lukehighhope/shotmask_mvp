import json
import os
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from detectors.shot_logreg import predict_logreg
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

CALIBRATED_PARAMS_FILENAME = "calibrated_detector_params.json"


def load_calibrated_params():
    """Load threshold_coef, min_dist_sec, prominence_frac from project root if present."""
    for base in [os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
        path = os.path.join(base, CALIBRATED_PARAMS_FILENAME)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

def bandpass(data, sr, low=400, high=6000):
    """Bandpass filter to emphasize gunshot crack (mid-high)."""
    nyq = sr / 2.0
    low = max(0.1, min(low, nyq - 1))
    high = max(low + 100, min(high, nyq - 1))
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, data.astype(np.float64))


def compute_spectral_flux(stft_mag, power=1.0, freq_mask=None, log_mag=True):
    """
    Compute spectral flux: measure of spectral change between consecutive frames (vectorized).
    Higher flux indicates onset/transient events like gunshots.
    
    Args:
        stft_mag: Magnitude spectrogram (freq_bins x time_frames)
        power: Power for flux calculation (2.0 = energy, 1.0 = magnitude)
        freq_mask: Optional 1D boolean mask (length = freq_bins); if set, flux only from these bins.
    
    Returns:
        flux: length (n_frames - 1), flux[t] = change from frame t to t+1
    """
    mag = np.asarray(stft_mag, dtype=np.float64)
    if log_mag:
        mag = np.log1p(mag)
    if freq_mask is not None:
        mag = np.where(freq_mask[:, np.newaxis], mag, 0.0)
    diff = mag[:, 1:] - mag[:, :-1]
    flux = np.sum(np.maximum(diff, 0), axis=0)
    return flux


def compute_onset_strength(y, sr, hop_length=256, n_fft=1024, band_low_hz=1000, band_high_hz=6000):
    """
    Compute onset strength using spectral flux.
    When band_low_hz/band_high_hz are set (e.g. 1k-6kHz), flux is computed only in that band for gunshot crack.
    """
    if HAS_LIBROSA and (band_low_hz is None or band_high_hz is None):
        try:
            return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        except Exception:
            pass
    
    window = np.hanning(n_fft)
    hop_samples = hop_length
    n_frames = max(1, (len(y) - n_fft) // hop_samples + 1)
    stft_mag = np.zeros((n_fft // 2 + 1, n_frames))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    
    for i in range(n_frames):
        start = i * hop_samples
        end = start + n_fft
        if end > len(y):
            frame = np.zeros(n_fft)
            frame[:len(y)-start] = y[start:]
            frame[:len(y)-start] *= window[:len(y)-start]
        else:
            frame = y[start:end] * window
        fft = np.fft.rfft(frame)
        stft_mag[:, i] = np.abs(fft)
    
    freq_mask = None
    if band_low_hz is not None and band_high_hz is not None:
        freq_mask = (freqs >= band_low_hz) & (freqs <= band_high_hz)
    flux = compute_spectral_flux(stft_mag, power=1.0, freq_mask=freq_mask, log_mag=True)
    expected_len = len(y) // hop_samples + 1
    onset = np.zeros(expected_len)
    onset[:len(flux)] = flux
    return onset


def compute_multi_band_energy(y, sr, hop_length=256, n_fft=1024):
    """
    Compute energy in multiple frequency bands for each time frame.
    Returns: (E_low, E_mid, E_high) where each is array of energy per frame.
    """
    # Frequency bands
    freq_bins = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    
    # Band indices
    low_idx = (freqs >= 50) & (freqs < 300)
    mid_idx = (freqs >= 300) & (freqs < 3000)
    high_idx = (freqs >= 3000) & (freqs < 9000)
    
    # Simple STFT
    window = np.hanning(n_fft)
    hop_samples = hop_length
    n_frames = (len(y) - n_fft) // hop_samples + 1
    
    E_low = np.zeros(n_frames)
    E_mid = np.zeros(n_frames)
    E_high = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_samples
        end = start + n_fft
        if end > len(y):
            break
        frame = y[start:end] * window
        fft = np.fft.rfft(frame)
        mag = np.abs(fft)
        
        E_low[i] = np.sum(mag[low_idx]**2)
        E_mid[i] = np.sum(mag[mid_idx]**2)
        E_high[i] = np.sum(mag[high_idx]**2)
    
    return E_low, E_mid, E_high


def compute_spectral_features(y, sr, hop_length=256, n_fft=1024):
    """
    Compute spectral features: centroid, flatness, rolloff.
    Returns arrays per time frame.
    """
    window = np.hanning(n_fft)
    hop_samples = hop_length
    n_frames = (len(y) - n_fft) // hop_samples + 1
    freq_bins = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    
    centroid = np.zeros(n_frames)
    flatness = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_samples
        end = start + n_fft
        if end > len(y):
            break
        frame = y[start:end] * window
        fft = np.fft.rfft(frame)
        mag = np.abs(fft)
        mag_sq = mag**2
        
        # Spectral centroid
        total_energy = np.sum(mag_sq)
        if total_energy > 1e-9:
            centroid[i] = np.sum(freqs * mag_sq) / total_energy
        else:
            centroid[i] = 0
        
        # Spectral flatness (geometric mean / arithmetic mean)
        mag_nonzero = mag[mag > 1e-9]
        if len(mag_nonzero) > 0:
            geo_mean = np.exp(np.mean(np.log(mag_nonzero)))
            arith_mean = np.mean(mag)
            flatness[i] = geo_mean / (arith_mean + 1e-9)
        else:
            flatness[i] = 0
    
    return centroid, flatness


def compute_attack_slope(envelope, sr, window_ms=10):
    """
    Compute attack slope: rate of rise in envelope.
    Useful for detecting compressed/AGC-processed gunshots.
    """
    window_samples = max(1, int(sr * window_ms / 1000.0))
    attack = np.zeros(len(envelope))
    
    for i in range(window_samples, len(envelope)):
        window = envelope[i-window_samples:i+1]
        if len(window) > 1:
            # Max rise over window
            attack[i] = np.max(np.diff(window)) / (window_samples / sr)
        else:
            attack[i] = 0
    
    return attack


def adaptive_threshold_mad(signal, window_sec=2.0, k=6.0, sr=None):
    """
    Adaptive threshold using sliding window + MAD (Median Absolute Deviation).
    More robust than global percentile for varying noise levels.
    Optimized: compute threshold at regular intervals, then interpolate.
    
    Args:
        signal: Input signal
        window_sec: Window size in seconds
        k: Multiplier for MAD (typically 6-10)
        sr: Sample rate (if None, assumes signal is already per-sample)
    
    Returns:
        threshold: Adaptive threshold array (same length as signal)
    """
    if sr is None:
        window_samples = int(len(signal) * window_sec / signal.shape[0] if hasattr(signal, 'shape') else window_sec)
    else:
        window_samples = int(sr * window_sec)
    
    window_samples = max(10, min(window_samples, len(signal) // 4))
    
    # Optimize: compute threshold at regular intervals (every window_samples/4)
    step = max(1, window_samples // 4)
    threshold_points = []
    indices = []
    
    for i in range(0, len(signal), step):
        start = max(0, i - window_samples // 2)
        end = min(len(signal), i + window_samples // 2)
        window = signal[start:end]
        
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        sigma = mad * 1.4826 if mad > 0 else np.std(window)
        
        threshold_points.append(median + k * sigma)
        indices.append(i)
    
    # Interpolate to full length
    if len(indices) == 1:
        threshold = np.full(len(signal), threshold_points[0])
    else:
        threshold = np.interp(np.arange(len(signal)), indices, threshold_points)
    
    return threshold


def adaptive_threshold_multi_scale(signal, sr, short_window=0.5, long_window=3.0, k_short=4.0, k_long=2.5):
    """
    Multi-scale adaptive threshold: step + interp (O(n)) like adaptive_threshold_mad.
    Short window catches transients; long window adapts to background noise.
    Returns per-sample (or per-frame) threshold: max(short_thresh, long_thresh).
    """
    n = len(signal)
    if n == 0:
        return np.array([])
    signal = np.asarray(signal, dtype=np.float64)
    short_win = max(10, min(int(sr * short_window), n // 4))
    long_win = max(short_win, min(int(sr * long_window), n // 2))
    step = max(1, long_win // 4)
    indices = []
    short_pts = []
    long_pts = []
    for i in range(0, n, step):
        start_s = max(0, i - short_win // 2)
        end_s = min(n, i + short_win // 2)
        win_s = signal[start_s:end_s]
        med_s = np.median(win_s)
        mad_s = np.median(np.abs(win_s - med_s)) + 1e-12
        short_pts.append(med_s + k_short * mad_s * 1.4826)
        start_l = max(0, i - long_win // 2)
        end_l = min(n, i + long_win // 2)
        win_l = signal[start_l:end_l]
        med_l = np.median(win_l)
        mad_l = np.median(np.abs(win_l - med_l)) + 1e-12
        long_pts.append(med_l + k_long * mad_l * 1.4826)
        indices.append(i)
    if len(indices) == 1:
        thresh = max(short_pts[0], long_pts[0])
        return np.full(n, thresh)
    short_thresh = np.interp(np.arange(n), indices, short_pts)
    long_thresh = np.interp(np.arange(n), indices, long_pts)
    return np.maximum(short_thresh, long_thresh)


# Scene presets for score_weights [onset, r1, flatness, attack, r2_penalty]
SCENE_WEIGHTS = {
    "default": [0.26, 0.34, 0.15, 0.15, 0.10],
    "indoor": [0.28, 0.32, 0.18, 0.12, 0.10],
    "outdoor": [0.22, 0.38, 0.12, 0.18, 0.10],
    "near": [0.22, 0.38, 0.15, 0.15, 0.10],
    "far": [0.30, 0.28, 0.16, 0.16, 0.10],
}


def _resolve_score_weights(score_weights, scene_config=None):
    """Resolve (w_onset, w_r1, w_flat, w_attack, w_r2_penalty). Prefer explicit weights, else scene preset."""
    if score_weights is not None and len(score_weights) >= 5:
        return tuple(score_weights[:5])
    if scene_config and scene_config in SCENE_WEIGHTS:
        return tuple(SCENE_WEIGHTS[scene_config])
    return tuple(SCENE_WEIGHTS["default"])


def cluster_peaks_by_time(peak_times, cluster_window_sec=0.25):
    """
    Cluster peaks that are close in time (within cluster_window_sec).
    Returns list of clusters, each cluster is list of peak indices.
    """
    if len(peak_times) == 0:
        return []
    
    peak_times = np.array(peak_times)
    sorted_indices = np.argsort(peak_times)
    sorted_times = peak_times[sorted_indices]
    
    clusters = []
    current_cluster = [sorted_indices[0]]
    
    for i in range(1, len(sorted_times)):
        if sorted_times[i] - sorted_times[i-1] <= cluster_window_sec:
            # Add to current cluster
            current_cluster.append(sorted_indices[i])
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [sorted_indices[i]]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def dynamic_cluster_window(detected_times, min_window=0.08, max_window=0.30):
    """
    Dynamic cluster window from OPTIMIZATION_GUIDE / shot_audio_improved.
    Fast bursts use smaller window, slower shots use larger window.
    """
    if len(detected_times) < 3:
        return min_window
    times = np.array(detected_times, dtype=np.float64)
    intervals = np.diff(np.sort(times))
    median_interval = float(np.median(intervals))
    window = float(np.clip(median_interval * 0.4, min_window, max_window))
    return window


def detect_shots(
    audio_path,
    fps,
    threshold_coef=5.0,
    threshold_percentile=None,
    min_dist_sec=0.10,
    prominence_frac=0.5,
    smooth_ms=6.0,
    use_calibrated=True,
    min_confidence=0.0,
    post_filter=True,
    use_improved=True,
    export_diagnostics=False,
    return_candidates=False,
):
    """
    Detect gunshot times from audio using improved two-stage multi-feature approach.
    
    Stage 1: High-recall candidate detection using onset/flux
    Stage 2: Multi-feature verification and scoring
    
    Args:
        use_improved: If True, use new two-stage approach; else use legacy method
        export_diagnostics: If True, export diagnostic curves (env, flux, score) for analysis
    """
    if use_calibrated:
        cal = load_calibrated_params()
        if cal:
            threshold_coef = cal.get("threshold_coef", threshold_coef)
            threshold_percentile = cal.get("threshold_percentile", threshold_percentile)
            min_dist_sec = cal.get("min_dist_sec", min_dist_sec)
            prominence_frac = cal.get("prominence_frac", prominence_frac)
            smooth_ms = cal.get("smooth_ms", smooth_ms)
            min_confidence = cal.get("min_confidence", min_confidence)
            post_filter = cal.get("post_filter", post_filter)
            use_improved = cal.get("use_improved", True)  # Default to improved method
    
    # Load audio
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float64)
    
    # Use improved method if requested
    if use_improved:
        # Get improved method parameters from calibration
        cal = load_calibrated_params() if use_calibrated else {}
        cluster_window = cal.get("cluster_window_sec", 0.25)
        use_dynamic_cluster = cal.get("use_dynamic_cluster", False)
        use_multi_scale_threshold = cal.get("use_multi_scale_threshold", False)
        mad_k = cal.get("mad_k", 7.0)  # Increased from 6.0 to reduce false positives
        candidate_min_dist_ms = cal.get("candidate_min_dist_ms", 80)  # Increased from 50ms
        score_weights = cal.get("score_weights", None)  # [onset, r1, flatness, attack, r2_penalty]
        min_score_threshold = cal.get("min_score_threshold", None)
        logreg_model = cal.get("logreg_model", None)
        scene_config = cal.get("scene_config", None)  # indoor/outdoor/near/far
        # min_score_threshold is used as min_confidence filter on final shots; keep from cal even when LR is present
        
        return detect_shots_improved(
            data, sr, fps,
            threshold_percentile=threshold_percentile,
            cluster_window_sec=cluster_window,
            use_dynamic_cluster=use_dynamic_cluster,
            use_multi_scale_threshold=use_multi_scale_threshold,
            export_diagnostics=export_diagnostics,
            audio_path=audio_path,
            mad_k=mad_k,
            candidate_min_dist_ms=candidate_min_dist_ms,
            score_weights=score_weights,
            min_score_threshold=min_score_threshold,
            scene_config=scene_config,
            logreg_model=logreg_model,
            return_candidates=return_candidates,
        )
    
    # Legacy method (original implementation)
    filtered = bandpass(data, sr, low=400, high=6000)
    env_win = max(1, int(sr * 0.003))
    energy = np.abs(filtered)
    envelope = np.convolve(energy, np.ones(env_win) / env_win, mode="same")
    smooth_win = max(1, int(sr * smooth_ms / 1000.0))
    smooth = np.convolve(envelope, np.ones(smooth_win) / smooth_win, mode="same")

    baseline = np.median(smooth)
    std = np.std(smooth)
    if threshold_percentile is not None:
        threshold = float(np.percentile(smooth, threshold_percentile))
    else:
        threshold = float(baseline + threshold_coef * max(std, 1e-9))

    upper = max(std, (np.percentile(smooth, 95) - baseline) * 0.3)
    prominence = float(prominence_frac * upper)
    prominence = max(prominence, 1e-9)

    min_dist_samples = max(1, int(sr * min_dist_sec))
    peaks, props = find_peaks(
        smooth,
        height=threshold,
        distance=min_dist_samples,
        prominence=prominence,
    )

    shots = []
    if len(peaks) > 0:
        max_h = max(props["peak_heights"])
        peak_heights = props["peak_heights"]
        prominences = props.get("prominences", peak_heights)
        
        for p, h, prom in zip(peaks, peak_heights, prominences):
            t = p / sr
            height_conf = h / max_h if max_h > 0 else 0.5
            prom_conf = prom / max(prominences) if len(prominences) > 0 and max(prominences) > 0 else 0.5
            confidence = 0.6 * height_conf + 0.4 * prom_conf
            
            shots.append({
                "t": round(float(t), 4),
                "frame": int(t * fps),
                "confidence": round(float(confidence), 3),
                "height": round(float(h), 4),
                "prominence": round(float(prom), 4),
            })
        
        if min_confidence > 0:
            shots = [s for s in shots if s["confidence"] >= min_confidence]
        
        if post_filter and len(shots) > 5:
            confidences = [s["confidence"] for s in shots]
            median_conf = np.median(confidences)
            min_conf_threshold = max(0.18, 0.4 * median_conf)
            shots = [s for s in shots if s["confidence"] >= min_conf_threshold]
            
            if len(shots) > 1:
                filtered_shots = []
                shots_sorted = sorted(shots, key=lambda x: x["t"])
                i = 0
                while i < len(shots_sorted):
                    current = shots_sorted[i]
                    j = i + 1
                    while j < len(shots_sorted) and shots_sorted[j]["t"] - current["t"] < 0.15:
                        if shots_sorted[j]["confidence"] > current["confidence"]:
                            current = shots_sorted[j]
                        j += 1
                    filtered_shots.append(current)
                    i = j
                shots = filtered_shots
            
            if len(shots) > 10:
                duration = shots[-1]["t"] - shots[0]["t"] if len(shots) > 1 else 1.0
                max_shots = int(duration * 1.5) + 3
                if len(shots) > max_shots:
                    shots.sort(key=lambda x: x["confidence"], reverse=True)
                    shots = shots[:max_shots]
                    shots.sort(key=lambda x: x["t"])

    return shots

def compute_feature_at_time(context, t, window_before=0.025, window_after=0.040):
    """
    Compute feature vector at a given time t (seconds) using precomputed context.
    Returns dict with same fields as candidate_features.
    """
    sr = context["sr"]
    hop = context["hop_length"]
    n_frames = context["n_frames"]
    onset_smooth = context["onset_smooth"]
    E_low = context["E_low"]
    E_mid = context["E_mid"]
    E_high = context["E_high"]
    flatness = context["flatness"]
    attack_frames = context["attack_frames"]
    eps = 1e-9

    peak_idx = int(round((t * sr) / hop))
    peak_idx = max(0, min(n_frames - 1, peak_idx))
    window_before_frames = max(1, int((sr * window_before) / hop))
    window_after_frames = max(1, int((sr * window_after) / hop))
    start_f = max(0, peak_idx - window_before_frames)
    end_f = min(n_frames, peak_idx + window_after_frames + 1)

    E_low_win = np.mean(E_low[start_f:end_f])
    E_mid_win = np.mean(E_mid[start_f:end_f])
    E_high_win = np.mean(E_high[start_f:end_f])
    r1 = E_mid_win / (E_low_win + eps)
    r2 = E_high_win / (E_mid_win + eps)
    r1_log = np.log(E_mid_win + eps) - np.log(E_low_win + eps)
    r2_log = np.log(E_high_win + eps) - np.log(E_mid_win + eps)
    r1_penalty = max(0.0, 0.7 - r1_log)
    flatness_val = float(np.mean(flatness[start_f:end_f]))
    attack_val = float(np.max(attack_frames[start_f:end_f]))
    attack_ref = np.percentile(attack_frames, 99.5) + eps
    attack_norm = min(attack_val / attack_ref, 1.0)
    onset_val = float(onset_smooth[peak_idx])
    onset_log = np.log1p(onset_val)
    E_total_win = E_low_win + E_mid_win + E_high_win + eps
    E_low_ratio = E_low_win / E_total_win
    hard_neg_slam = (E_low_ratio > 0.65) or ((E_low_ratio > 0.55) and (flatness_val < 0.22))
    hard_neg_metal = (r2_log > 1.2) and (flatness_val > 0.5) and (attack_norm < 0.4)

    return {
        "t": float(t),
        "peak_idx": int(peak_idx),
        "onset_raw": onset_val,
        "onset_log": onset_log,
        "r1": r1,
        "r2": r2,
        "r1_log": r1_log,
        "r2_log": r2_log,
        "flatness": flatness_val,
        "attack": attack_val,
        "attack_norm": attack_norm,
        "hard_neg_slam": hard_neg_slam,
        "hard_neg_metal": hard_neg_metal,
        "r1_penalty": r1_penalty,
        "E_low_ratio": E_low_ratio,
    }


def detect_shots_improved(
    data, sr, fps,
    threshold_percentile=None,
    cluster_window_sec=0.25,
    use_dynamic_cluster=False,
    use_multi_scale_threshold=False,
    export_diagnostics=False,
    audio_path=None,
    mad_k=6.0,
    candidate_min_dist_ms=50,
    score_weights=None,
    min_score_threshold=None,
    scene_config=None,
    logreg_model=None,
    return_candidates=False,
    return_feature_context=False,
):
    """
    Improved two-stage gunshot detection with multi-feature fusion.
    
    Stage 1: High-recall candidate detection using onset/flux (1k-6kHz band).
    Stage 2: Multi-feature verification, robust+sigmoid confidence, hard-negative filtering.
    
    Args:
        use_dynamic_cluster: If True, set cluster_window from median inter-candidate interval.
        mad_k: MAD threshold multiplier (higher = stricter).
        candidate_min_dist_ms: Minimum distance between candidate peaks (ms).
        score_weights: [onset, r1, flatness, attack, r2_penalty] or None to use scene_config.
        min_score_threshold: Minimum confidence (float only; None disables).
        scene_config: Preset for score_weights: "indoor"|"outdoor"|"near"|"far"|"default".
        logreg_model: Optional logistic regression model dict (weights/bias/mean/std).
        return_candidates: If True, return (shots, candidate_features).
    """
    # STFT parameters; n_frames is the single source of truth for frame axis
    hop_length = 256  # ~5.3ms at 48kHz
    n_fft = 1024      # ~21ms at 48kHz
    hop_samples = hop_length
    n_frames = max(1, (len(data) - n_fft) // hop_samples + 1)
    
    # ========== Single STFT (compute once, reuse for onset + all features) ==========
    window = np.hanning(n_fft)
    stft_mag = np.zeros((n_fft // 2 + 1, n_frames))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    for i in range(n_frames):
        start = i * hop_samples
        end = start + n_fft
        if end > len(data):
            frame = np.zeros(n_fft)
            frame[: len(data) - start] = data[start:]
            frame[: len(data) - start] *= window[: len(data) - start]
        else:
            frame = data[start:end] * window
        fft = np.fft.rfft(frame)
        stft_mag[:, i] = np.abs(fft)
    
    # Onset = spectral flux in 1k-6kHz band, aligned to n_frames (frame axis unified)
    freq_mask_1k_6k = (freqs >= 1000) & (freqs <= 6000)
    flux = compute_spectral_flux(stft_mag, power=1.0, freq_mask=freq_mask_1k_6k, log_mag=True)
    onset = np.zeros(n_frames, dtype=np.float64)
    onset[1:] = flux[:]
    
    # Smooth onset (10ms ~= 2 frames at 48k/256 hop)
    smooth_onset_win = max(1, int((sr * 0.01) / hop_samples))
    onset_smooth = np.convolve(onset, np.ones(smooth_onset_win) / smooth_onset_win, mode="same")
    
    # ========== STAGE 1: Candidate Detection (High Recall) ==========
    if threshold_percentile is not None:
        threshold_onset = np.percentile(onset_smooth, threshold_percentile)
        threshold_onset_arr = np.full(n_frames, threshold_onset)
    elif use_multi_scale_threshold:
        sr_frames = sr / hop_samples
        threshold_onset_arr = adaptive_threshold_multi_scale(
            onset_smooth, sr_frames, short_window=0.5, long_window=3.0, k_short=4.0, k_long=2.5
        )
    else:
        sr_frames = sr / hop_samples
        threshold_onset_arr = adaptive_threshold_mad(onset_smooth, window_sec=2.0, k=mad_k, sr=sr_frames)
    
    min_dist_frames = max(1, int((sr * candidate_min_dist_ms / 1000.0) / hop_samples))
    candidate_peaks, _ = find_peaks(onset_smooth, distance=min_dist_frames)
    if len(candidate_peaks) > 0:
        candidate_peaks = np.array(
            [p for p in candidate_peaks if p < n_frames and onset_smooth[p] >= threshold_onset_arr[p]],
            dtype=int
        )
    
    # Second-shot recovery: AGC/compression often lowers the next shot; search with relaxed threshold
    recovery_ratio = 0.50
    recovery_start_frames = max(1, int((sr * 0.04) / hop_samples))
    recovery_end_frames = min(n_frames - 1, int((sr * 0.20) / hop_samples))
    existing_set = set(int(p) for p in candidate_peaks)
    for p in np.sort(candidate_peaks):
        start_f = min(n_frames - 1, p + recovery_start_frames)
        end_f = min(n_frames - 1, p + recovery_end_frames)
        if start_f >= end_f:
            continue
        for f in range(start_f, end_f + 1):
            if f in existing_set:
                continue
            if onset_smooth[f] < recovery_ratio * threshold_onset_arr[f]:
                continue
            if (f > 0 and onset_smooth[f] < onset_smooth[f - 1]) or (f < n_frames - 1 and onset_smooth[f] < onset_smooth[f + 1]):
                continue
            too_near_other = any(abs(f - p2) < min_dist_frames for p2 in existing_set if p2 != p)
            if too_near_other:
                continue
            existing_set.add(f)
            candidate_peaks = np.append(candidate_peaks, f)
    candidate_peaks = np.unique(candidate_peaks).astype(int)
    
    if len(candidate_peaks) == 0:
        if return_feature_context:
            return ([], [], {})
        return ([], []) if return_candidates else []
    
    # ========== STAGE 2: Feature Extraction & Scoring (same n_frames / stft_mag) ==========
    
    # Extract multi-band energy from STFT
    low_idx = (freqs >= 50) & (freqs < 300)
    mid_idx = (freqs >= 300) & (freqs < 3000)
    high_idx = (freqs >= 3000) & (freqs < 9000)
    
    E_low = np.sum(stft_mag[low_idx, :]**2, axis=0)
    E_mid = np.sum(stft_mag[mid_idx, :]**2, axis=0)
    E_high = np.sum(stft_mag[high_idx, :]**2, axis=0)
    
    # Compute spectral features from STFT
    mag_sq = stft_mag**2
    centroid = np.zeros(n_frames)
    flatness = np.zeros(n_frames)
    
    for i in range(n_frames):
        mag = stft_mag[:, i]
        mag_sq_frame = mag_sq[:, i]
        
        # Spectral centroid
        total_energy = np.sum(mag_sq_frame)
        if total_energy > 1e-9:
            centroid[i] = np.sum(freqs * mag_sq_frame) / total_energy
        
        # Spectral flatness
        mag_nonzero = mag[mag > 1e-9]
        if len(mag_nonzero) > 0:
            geo_mean = np.exp(np.mean(np.log(mag_nonzero)))
            arith_mean = np.mean(mag)
            flatness[i] = geo_mean / (arith_mean + 1e-9)
    
    # Compute envelope for attack slope (per-sample), then aggregate to frame domain
    env_win = max(1, int(sr * 0.003))
    energy = np.abs(data)
    envelope = np.convolve(energy, np.ones(env_win) / env_win, mode="same")
    attack_slope = compute_attack_slope(envelope, sr, window_ms=10)
    attack_frames = np.zeros(n_frames)
    for i in range(n_frames):
        start_idx = i * hop_samples
        end_idx = min(len(attack_slope), start_idx + hop_samples)
        if end_idx > start_idx:
            attack_frames[i] = np.max(attack_slope[start_idx:end_idx])
    
    # Evaluate each candidate (frame-domain)
    candidate_scores = []
    candidate_features = []
    candidate_onset = []
    candidate_flatness = []
    candidate_r1_log = []
    candidate_r2_log = []
    candidate_attack = []
    
    window_before_frames = max(1, int((sr * 0.025) / hop_samples))
    window_after_frames = max(1, int((sr * 0.040) / hop_samples))
    
    for peak_idx in candidate_peaks:
        t = (peak_idx * hop_samples) / sr
        start_f = max(0, peak_idx - window_before_frames)
        end_f = min(n_frames, peak_idx + window_after_frames + 1)
        
        # Multi-band energy ratios (frame domain)
        E_low_win = np.mean(E_low[start_f:end_f])
        E_mid_win = np.mean(E_mid[start_f:end_f])
        E_high_win = np.mean(E_high[start_f:end_f])
        
        eps = 1e-9
        r1 = E_mid_win / (E_low_win + eps)
        r2 = E_high_win / (E_mid_win + eps)
        r1_log = np.log(E_mid_win + eps) - np.log(E_low_win + eps)
        r2_log = np.log(E_high_win + eps) - np.log(E_mid_win + eps)
        
        # Penalize low r1_log (more stable than linear ratios)
        r1_penalty = max(0.0, 0.7 - r1_log)
        
        # Spectral flatness
        flatness_val = float(np.mean(flatness[start_f:end_f]))
        
        # Attack slope (frame-aggregated); normalize by p99.5 for robustness
        attack_val = float(np.max(attack_frames[start_f:end_f]))
        attack_ref = np.percentile(attack_frames, 99.5) + eps
        attack_norm = min(attack_val / attack_ref, 1.0)
        
        # Onset value (frame domain)
        onset_val = float(onset_smooth[peak_idx])
        onset_log = np.log1p(onset_val)
        
        # Hard negatives
        E_total_win = E_low_win + E_mid_win + E_high_win + eps
        E_low_ratio = E_low_win / E_total_win
        hard_neg_slam = (E_low_ratio > 0.65) or ((E_low_ratio > 0.55) and (flatness_val < 0.22))
        hard_neg_metal = (r2_log > 1.2) and (flatness_val > 0.5) and (attack_norm < 0.4)
        
        candidate_onset.append(onset_log)
        candidate_flatness.append(flatness_val)
        candidate_r1_log.append(r1_log)
        candidate_r2_log.append(r2_log)
        candidate_attack.append(attack_norm)
        
        candidate_features.append({
            "t": t,
            "peak_idx": int(peak_idx),
            "onset_raw": onset_val,
            "onset_log": onset_log,
            "r1": r1,
            "r2": r2,
            "r1_log": r1_log,
            "r2_log": r2_log,
            "flatness": flatness_val,
            "attack": attack_val,
            "attack_norm": attack_norm,
            "hard_neg_slam": hard_neg_slam,
            "hard_neg_metal": hard_neg_metal,
            "r1_penalty": r1_penalty,
            "E_low_ratio": E_low_ratio,
        })

    # Build reusable feature context (for miss samples)
    feature_context = {
        "hop_length": hop_length,
        "sr": sr,
        "n_frames": n_frames,
        "onset_smooth": onset_smooth,
        "E_low": E_low,
        "E_mid": E_mid,
        "E_high": E_high,
        "flatness": flatness,
        "attack_frames": attack_frames,
    }
    
    # Robust normalize features across candidates (reduce device/AGC effects)
    def _robust_z(vals):
        arr = np.asarray(vals, dtype=np.float64)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-9
        scale = mad * 1.4826
        return (arr - med) / (scale + 1e-9)
    
    if len(candidate_features) > 0:
        onset_norms = 1.0 / (1.0 + np.exp(-_robust_z(candidate_onset)))
        flatness_norms = 1.0 / (1.0 + np.exp(-_robust_z(candidate_flatness)))
        r1_norms = 1.0 / (1.0 + np.exp(-_robust_z(candidate_r1_log)))
        r2_norms = 1.0 / (1.0 + np.exp(-_robust_z(candidate_r2_log)))
    else:
        onset_norms, flatness_norms, r1_norms, r2_norms = [], [], [], []
    
    # Score candidates (hand-tuned)
    w_onset, w_r1, w_flat, w_attack, w_r2_penalty = _resolve_score_weights(score_weights, scene_config)
    for i, feat in enumerate(candidate_features):
        onset_norm = onset_norms[i]
        flatness_norm = flatness_norms[i]
        r1_norm = r1_norms[i]
        r2_norm = r2_norms[i]
        attack_norm = candidate_attack[i]
        
        score = (
            w_onset * onset_norm
            + w_r1 * r1_norm
            - w_r1 * 0.3 * feat["r1_penalty"]
            + w_flat * flatness_norm
            + w_attack * attack_norm
            - w_r2_penalty * r2_norm
        )
        if feat["hard_neg_slam"]:
            score -= 0.10
        if feat["hard_neg_metal"]:
            score -= 0.25
        
        candidate_scores.append(score)
        feat["score"] = score
    
    # If logistic regression model is provided, use it to score candidates
    if logreg_model and len(candidate_features) > 0:
        X = []
        for i, feat in enumerate(candidate_features):
            X.append([
                float(feat["onset_log"]),
                float(feat["r1_log"]),
                float(feat["r2_log"]),
                float(feat["flatness"]),
                float(feat["attack_norm"]),
                float(feat["E_low_ratio"]),
                1.0 if feat["hard_neg_slam"] else 0.0,
                1.0 if feat["hard_neg_metal"] else 0.0,
            ])
        probs = predict_logreg(logreg_model, np.asarray(X))
        candidate_scores = probs.tolist()
        for i, feat in enumerate(candidate_features):
            feat["score"] = float(candidate_scores[i])
    
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    # Normalized absolute score per candidate (0–1) for product-friendly confidence
    scores_arr = np.array(candidate_scores, dtype=np.float64)
    med_s = np.median(scores_arr)
    mad_s = np.median(np.abs(scores_arr - med_s)) + 1e-9
    z = (scores_arr - med_s) / (mad_s * 1.4826 + 1e-9)
    score_norm = 1.0 / (1.0 + np.exp(-np.clip(z, -5, 5)))
    
    # ========== Clustering & Selection ==========
    candidate_times = [f["t"] for f in candidate_features]
    if use_dynamic_cluster and len(candidate_times) >= 3:
        cluster_window_sec = dynamic_cluster_window(candidate_times, min_window=0.08, max_window=0.30)
    clusters = cluster_peaks_by_time(candidate_times, cluster_window_sec=cluster_window_sec)
    
    shots = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        
        cluster_raw_scores = [candidate_scores[i] for i in cluster]
        best_local = np.argmax(cluster_raw_scores)
        best_idx_in_cluster = cluster[best_local]
        best_candidate = candidate_features[best_idx_in_cluster]
        
        margin = cluster_raw_scores[best_local] - np.median(cluster_raw_scores)
        confidence_margin = float(_sigmoid(margin / 0.15))
        confidence_absolute = float(score_norm[best_idx_in_cluster])
        confidence = 0.5 * confidence_absolute + 0.5 * confidence_margin
        
        shots.append({
            "t": round(float(best_candidate["t"]), 4),
            "frame": int(best_candidate["t"] * fps),
            "confidence": round(float(confidence), 3),
            "onset": round(float(best_candidate.get("onset_raw", 0.0)), 3),
            "r1": round(float(best_candidate["r1"]), 2),
            "r2": round(float(best_candidate["r2"]), 2),
            "flatness": round(float(best_candidate["flatness"]), 3),
            "attack": round(float(best_candidate["attack"]), 3),
            "score": round(float(best_candidate["score"]), 3),
        })
    
    # Cap shots by plausible fire rate using raw score ranking
    if len(shots) > 2:
        max_rate = 2.2  # shots/sec (outdoor pistol baseline)
        duration = shots[-1]["t"] - shots[0]["t"] if len(shots) > 1 else 1.0
        max_allowed = int(duration * max_rate) + 2
        if len(shots) > max_allowed:
            shots.sort(key=lambda x: x["score"], reverse=True)
            shots = shots[:max_allowed]
            shots.sort(key=lambda x: x["t"])
    
    # Optional fixed confidence filter (only if explicitly set to a number)
    if len(shots) > 0 and isinstance(min_score_threshold, (int, float)):
        shots = [s for s in shots if s["confidence"] >= min_score_threshold]
    
    # Rate limit protection (only for extreme cases)
    if len(shots) > 0:
        duration = shots[-1]["t"] - shots[0]["t"] if len(shots) > 1 else 1.0
        shots_per_sec = len(shots) / duration if duration > 0 else 0
        if shots_per_sec > 6.0:  # Extreme case: >6 shots/sec
            # Keep top N by confidence
            shots.sort(key=lambda x: x["confidence"], reverse=True)
            max_shots = int(duration * 3.0) + 5  # Max 3 shots/sec
            shots = shots[:max_shots]
            shots.sort(key=lambda x: x["t"])
    
    # Export diagnostics if requested
    if export_diagnostics and audio_path:
        print("FP candidates (confidence < 0.5):")
        for s in shots:
            if s["confidence"] < 0.5:
                print(s)
        scores_per_sample = np.zeros(len(data))
        for i, feat in enumerate(candidate_features):
            idx = int(feat["peak_idx"] * hop_length)
            if 0 <= idx < len(scores_per_sample):
                scores_per_sample[idx] = candidate_scores[i]
        flux_for_plot = np.interp(
            np.arange(len(data)),
            np.linspace(0, len(data) - 1, num=max(1, onset_smooth.shape[0])),
            onset_smooth
        )
        export_diagnostic_curves(
            audio_path, data, sr, envelope, flux_for_plot,
            scores_per_sample, shots, hop_length
        )
    
    if return_feature_context:
        return shots, candidate_features, feature_context
    if return_candidates:
        return shots, candidate_features
    return shots


def export_diagnostic_curves(audio_path, data, sr, envelope, flux, scores, shots, hop_length):
    """
    Export diagnostic curves: envelope, flux, and scores for analysis.
    Helps identify where algorithm fails (missed detections vs false positives).
    """
    import os
    import json
    
    # Downsample for export (max 5000 points)
    max_points = 5000
    step = max(1, len(data) // max_points)
    
    time_axis = np.arange(len(data))[::step] / sr
    envelope_ds = envelope[::step]
    flux_ds = flux[::step] if len(flux) == len(data) else flux[::step*hop_length][:len(time_axis)]
    scores_ds = scores[::step] if len(scores) == len(data) else np.zeros(len(time_axis))
    
    # Shot times
    shot_times = [s["t"] for s in shots]
    
    diagnostics = {
        "time": time_axis.tolist(),
        "envelope": envelope_ds.tolist(),
        "flux": flux_ds.tolist() if len(flux_ds) == len(time_axis) else [0.0] * len(time_axis),
        "scores": scores_ds.tolist() if len(scores_ds) == len(time_axis) else [0.0] * len(time_axis),
        "shot_times": shot_times,
        "sample_rate": float(sr),
    }
    
    output_path = audio_path.replace(".wav", "_diagnostics.json").replace("tmp/", "outputs/")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    
    print(f"Diagnostic curves exported to: {output_path}")
    print("  Use this to visualize: envelope, flux, scores, and detected shots")
    print("  Helps identify: missed detections (threshold too high?) vs false positives (feature issue?)")
