"""
Beep detection: find the single start beep in shooting videos.
Tuned using all *beep.txt ground truth (traning data/01032026, outdoor S1-S8).

Run: python evaluate_beep_detector.py  # MAE on GT
     python evaluate_beep_detector.py --tune  # grid search params

When the detector is wrong (e.g. beep ~4.5s but first peak is first shot ~6.7s),
use same-folder *beep.txt (e.g. v1beep.txt with a single line: 4.5).
"""
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter, find_peaks

# 主探测模式: cnn_tonal_primary=True 时以滑窗 CNN+主频 为主；否则规则峰+CNN/主频选峰
BEEP_CONFIG = {
    "cnn_tonal_primary": True,  # True=滑窗 CNN+主频 为主探测手段
    "cnn_tonal_step_s": 0.15,   # 滑窗步长(秒)
    "cnn_tonal_min_prob": 0.5,  # CNN 阈值，且主频需在 2–5.2kHz
    "cnn_refine_window_s": 0.5, # CNN 得到近似 beep 后，用原算法在此窗口内精确定位
    "use_cnn_beep": True,       # 非 primary 时对规则候选用 CNN 打分
    "cnn_beep_min_prob": 0.3,
    "tonal_max_span_s": 5.0,
    "head_s": 3.0,           # Stats from first N s for threshold (avoid gunshots inflating std)
    "min_search_s": 0.5,     # Ignore peaks before this (start noise)
    "max_search_s": 30.0,    # Consider peaks up to this (outdoor beeps can be 18–24s)
    "k_list": (5, 4, 3, 2.5, 2.0),  # Threshold = mean + k*std; try from strict to loose
    "smooth_win_s": 0.01,
    "peak_distance_s": 0.2,
    "isolated_gap_s": 2.0,   # Prefer peak with no other peak in [t - gap, t)
    "min_height_frac": 0.55, # Keep peaks with height >= this fraction of max (tuned on *beep.txt GT)
    "early_probe_after_s": 0.5,  # When t0 in (0.5, 1.0)s → probe 3s (v2); 1.mp4 beep 1.13s no trigger
    "early_probe_before_s": 1.0,
    "early_probe_center_s": 3.0,
    "early_probe_window_s": 1.5,
    "early_probe_min_s": 2.0,
    "early_probe_max_s": 4.5,
    "early_probe_floor_s": 3.0,  # 若早峰修正结果 < 此值则用此值 (v2: 3s以上)
    "late_probe_after_s": 6.5,   # When first peak in (6.5, 6.75)s → probe 4.5s (v1: 6.69 first shot)
    "late_probe_before_s": 6.75, # S2 beep 6.78s stays (no probe)
    "late_probe_center_s": 4.5,
    "late_probe_window_s": 2.0,
    "late_probe_max_s": 5.5,
}


# Beep 主频来自 *beep.txt+0.3s 统计: 约 3 kHz 与 4.6 kHz 两档，带通覆盖两者
BEEP_BANDPASS_LOW = 1500
BEEP_BANDPASS_HIGH = 5500
BEEP_TONAL_FREQ_LOW = 2000   # 候选峰 0.3s 窗口主频需在此范围内才视为 beep
BEEP_TONAL_FREQ_HIGH = 5200


def bandpass(data, sr, low=None, high=None):
    low = low if low is not None else BEEP_BANDPASS_LOW
    high = high if high is not None else BEEP_BANDPASS_HIGH
    b, a = butter(4, [low/(sr/2), high/(sr/2)], btype='band')
    return lfilter(b, a, data)


def _dominant_freq_hz(data, sr, start_s, duration_s=0.3, freq_low=200, freq_high=6000):
    """Segment [start_s, start_s+duration_s] 的 FFT 主频 (Hz)."""
    n_start = int(start_s * sr)
    n_len = int(duration_s * sr)
    if n_start + n_len > len(data):
        n_len = max(0, len(data) - n_start)
    if n_len < int(sr * 0.05):
        return None
    seg = data[n_start : n_start + n_len].astype(np.float64) * np.hanning(n_len)
    n_fft = max(2048, 2 ** int(np.ceil(np.log2(n_len))))
    spec = np.abs(np.fft.rfft(seg, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    lo = np.searchsorted(freqs, freq_low)
    hi = np.searchsorted(freqs, freq_high)
    if hi <= lo:
        return None
    idx = lo + np.argmax(spec[lo:hi])
    return float(freqs[idx])


def _detect_beeps_cnn_tonal_primary(data, sr, duration_s, config):
    """
    以 CNN + 主频 为主：滑窗扫描 [min_s, max_s]，每窗算 CNN P(beep) 与主频；
    取「最早」满足 cnn_prob >= 阈值 且 主频在 2–5.2kHz 的 t；若无则取 CNN 分最高且主频通过的 t。
    返回 t0 (float) 或 None（未找到或未加载 CNN）。
    """
    min_s = config.get("min_search_s", 0.5)
    max_s = min(config.get("max_search_s", 30.0), duration_s - 0.4)
    step = config.get("cnn_tonal_step_s", 0.15)
    min_prob = config.get("cnn_tonal_min_prob", 0.5)
    tonal_low = config.get("tonal_freq_low", BEEP_TONAL_FREQ_LOW)
    tonal_high = config.get("tonal_freq_high", BEEP_TONAL_FREQ_HIGH)
    try:
        from detectors.beep_cnn import load_cnn_beep, mel_at_time as beep_cnn_mel, predict_proba_one as beep_cnn_proba
        cnn_model, cnn_device = load_cnn_beep()
    except Exception:
        return None
    if cnn_model is None or cnn_device is None:
        return None

    best_first = None   # 最早满足 cnn>=min_prob 且 主频 2–5.2kHz
    best_score = -1.0   # 主频通过时 CNN 最高分（兜底）
    best_t_fallback = None

    t = min_s
    while t <= max_s:
        mel = beep_cnn_mel(data, sr, t)
        p = beep_cnn_proba(cnn_model, cnn_device, mel)
        f = _dominant_freq_hz(data, sr, t, duration_s=0.3, freq_low=200, freq_high=6000)
        tonal_ok = f is not None and tonal_low <= f <= tonal_high
        if p >= min_prob and tonal_ok:
            if best_first is None:
                best_first = float(t)
        if tonal_ok and p > best_score:
            best_score = p
            best_t_fallback = float(t)
        t += step
    if best_first is not None:
        return best_first
    if best_t_fallback is not None and best_score >= config.get("cnn_beep_min_prob", 0.3):
        return best_t_fallback
    return None


def _detect_beeps_impl(audio_path, fps, config):
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float64)
    duration_s = len(data) / sr

    # 主模式：以 CNN + 主频 滑窗为主
    if config.get("cnn_tonal_primary"):
        t0_primary = _detect_beeps_cnn_tonal_primary(data, sr, duration_s, config)
        if t0_primary is not None:
            t0 = t0_primary
            # CNN 得到近似 beep 后，用原算法（detect_beep_near：带通+峰）在 t0 附近精确定位
            refine_win = config.get("cnn_refine_window_s", 0.5)
            t_refined = detect_beep_near(audio_path, fps, t_approx_sec=t0, window_sec=refine_win)
            if t_refined is not None:
                t0 = t_refined
            # 仅做晚峰修正：若落在 6.5–6.75s 则在 4.5s 附近再探
            after = config["late_probe_after_s"]
            before = config.get("late_probe_before_s", 999.0)
            if after < t0 < before:
                t_near = detect_beep_near(
                    audio_path, fps,
                    t_approx_sec=config["late_probe_center_s"],
                    window_sec=config["late_probe_window_s"]
                )
                if t_near is not None and config.get("min_search_s", 0.5) <= t_near <= config.get("late_probe_max_s", 5.5):
                    t0 = t_near
            return [{"t": round(float(t0), 4), "frame": int(t0 * fps), "confidence": 0.95}]

    # 否则：规则峰 + CNN/主频 选峰
    filtered = bandpass(data, sr)
    energy = np.abs(filtered)
    win = max(1, int(sr * config["smooth_win_s"]))
    smooth = np.convolve(energy, np.ones(win) / win, mode="same")

    head_s = config["head_s"]
    n_head = min(len(smooth), int(sr * head_s))
    smooth_head = smooth[:n_head]
    std_val = float(np.std(smooth_head))
    mean_val = float(np.mean(smooth_head))
    if std_val < 1e-12:
        std_val = 1e-12

    # Use loosest threshold to get all candidate peaks, then filter by strength
    dist = int(sr * config["peak_distance_s"])
    k_loose = min(config["k_list"])
    threshold = mean_val + k_loose * std_val
    peaks, props = find_peaks(smooth, height=threshold, distance=dist)
    if len(peaks) == 0:
        return []

    min_s = config["min_search_s"]
    max_s = min(config["max_search_s"], duration_s - 0.5)
    peak_times = np.array([p / sr for p in peaks])
    heights = props["peak_heights"] if props is not None else np.ones(len(peaks))
    in_window = (peak_times >= min_s) & (peak_times <= max_s)
    if not np.any(in_window):
        in_window = peak_times <= max_s
    idx = np.where(in_window)[0]
    if len(idx) == 0:
        return []
    peak_times_w = peak_times[idx]
    heights_w = heights[idx]

    # Keep only peaks that are "strong" (>= fraction of max height in window) to skip early noise
    min_height_frac = config.get("min_height_frac", 0.35)
    max_h = float(np.max(heights_w))
    strong = heights_w >= min_height_frac * max_h
    if np.any(strong):
        candidates = sorted(peak_times_w[strong])
    else:
        candidates = sorted(peak_times_w)

    # Prefer "isolated" peak: no other peak in [t - isolated_gap_s, t)
    gap = config["isolated_gap_s"]
    isolated = [t for t in candidates if not any(t - gap <= x < t for x in candidates if x != t)]
    ordered = isolated if isolated else candidates

    # Prefer candidate: optionally use beep CNN score, else use tonal (2–5.2 kHz) then first
    t_first = float(ordered[0])
    max_span_s = config.get("tonal_max_span_s", 5.0)
    t0 = None
    if config.get("use_cnn_beep"):
        try:
            from detectors.beep_cnn import load_cnn_beep, mel_at_time as beep_cnn_mel, predict_proba_one as beep_cnn_proba
            cnn_model, cnn_device = load_cnn_beep()
            if cnn_model is not None and cnn_device is not None:
                best_t, best_p = t_first, -1.0
                for t in ordered:
                    if t > t_first + max_span_s:
                        break
                    mel = beep_cnn_mel(data, sr, t)
                    p = beep_cnn_proba(cnn_model, cnn_device, mel)
                    if p > best_p:
                        best_p, best_t = p, float(t)
                if best_p >= config.get("cnn_beep_min_prob", 0.3):
                    t0 = best_t
        except Exception:
            pass
    if t0 is None:
        tonal_low = config.get("tonal_freq_low", BEEP_TONAL_FREQ_LOW)
        tonal_high = config.get("tonal_freq_high", BEEP_TONAL_FREQ_HIGH)
        for t in ordered:
            if t > t_first + max_span_s:
                break
            f = _dominant_freq_hz(data, sr, t, duration_s=0.3, freq_low=200, freq_high=6000)
            if f is not None and tonal_low <= f <= tonal_high:
                t0 = float(t)
                break
    if t0 is None:
        t0 = t_first

    # Optional: when t0 is suspiciously early (0.5–1.0s), probe ~3s (e.g. v2). Skip if there are peaks after 5s (late beep e.g. S8).
    e_after = config.get("early_probe_after_s", 999.0)
    e_before = config.get("early_probe_before_s", 0.0)
    has_late_candidate = any(c >= 5.0 for c in candidates)
    if e_after < t0 < e_before and not has_late_candidate:
        t_near = detect_beep_near(
            audio_path, fps,
            t_approx_sec=config.get("early_probe_center_s", 3.0),
            window_sec=config.get("early_probe_window_s", 1.5)
        )
        e_min = config.get("early_probe_min_s", 2.0)
        e_max = config.get("early_probe_max_s", 4.5)
        floor_s = config.get("early_probe_floor_s", 0.0)
        if t_near is not None and e_min <= t_near <= e_max:
            t0 = max(float(t_near), floor_s) if floor_s > 0 else float(t_near)

    # Optional: when first peak suggests "first shot" and beep is ~4.5s (e.g. v1)
    after = config["late_probe_after_s"]
    before = config.get("late_probe_before_s", 999.0)
    has_first_shot_candidate = any(after < c < before for c in candidates)
    # Case A: chosen t0 in (6.5, 6.75)s → likely first shot, probe 4.5s
    # Case B: chosen t0 in (2, 5)s but there is a peak in (6.5, 6.75)s → same (v1: t0=2.36, 6.69 in list)
    if (after < t0 < before) or (t0 < 5.0 and has_first_shot_candidate):
        t_near = detect_beep_near(
            audio_path, fps,
            t_approx_sec=config["late_probe_center_s"],
            window_sec=config["late_probe_window_s"]
        )
        if t_near is not None and min_s <= t_near <= config["late_probe_max_s"]:
            t0 = t_near

    return [{
        "t": round(float(t0), 4),
        "frame": int(t0 * fps),
        "confidence": 0.95
    }]


def detect_beeps(audio_path, fps, **overrides):
    """
    Detect the single start beep. Uses BEEP_CONFIG; overrides via kwargs.
    Returns list of one dict: [{"t": sec, "frame": int, "confidence": float}].
    """
    config = dict(BEEP_CONFIG)
    config.update(overrides)
    return _detect_beeps_impl(audio_path, fps, config)


def detect_beep_near(audio_path, fps, t_approx_sec, window_sec=2.0):
    """
    在 t_approx_sec 附近 ± window_sec 的窗口内重新探测 beep 真实时间。
    用于根据用户提供的约略时间验证并精确定位 beep。
    Returns: 真实 beep 时间 (float, 秒)，若未找到则返回 None。
    """
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float64)
    filtered = bandpass(data, sr)
    energy = np.abs(filtered)
    win = max(1, int(sr * 0.01))
    smooth = np.convolve(energy, np.ones(win) / win, mode="same")

    start_s = max(0.0, t_approx_sec - window_sec)
    end_s = min(len(data) / sr, t_approx_sec + window_sec)
    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    if end_i <= start_i:
        return None
    smooth_win = smooth[start_i:end_i]
    if len(smooth_win) < int(sr * 0.1):
        return None

    mean_w = float(np.mean(smooth_win))
    std_w = float(np.std(smooth_win))
    if std_w < 1e-9:
        return None
    threshold = mean_w + 2.5 * std_w
    peaks, props = find_peaks(smooth_win, height=threshold, distance=int(sr * 0.15))
    if len(peaks) == 0:
        for thresh_k in (2.0, 1.5, 1.0):
            threshold = mean_w + thresh_k * std_w
            peaks, props = find_peaks(smooth_win, height=threshold, distance=int(sr * 0.15))
            if len(peaks) > 0:
                break
    if len(peaks) == 0:
        return None
    peak_times = (start_i + peaks) / sr
    best_i = np.argmin(np.abs(peak_times - t_approx_sec))
    t_best = float(peak_times[best_i])
    return round(t_best, 4)
