import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter, find_peaks

def bandpass(data, sr, low=1500, high=4000):
    b, a = butter(4, [low/(sr/2), high/(sr/2)], btype='band')
    return lfilter(b, a, data)

def detect_beeps(audio_path, fps):
    data, sr = sf.read(audio_path)
    filtered = bandpass(data, sr)

    # Goal: find only the single starting beep around 1s at the beginning
    # Approach:
    # - Bandpass filter then take amplitude envelope
    # - Light smoothing to avoid detecting one beep as multiple peaks
    # - Only take the earliest peak within the first few seconds window as beep
    energy = np.abs(filtered)
    # ~10ms smoothing window
    win = max(1, int(sr * 0.01))
    smooth = np.convolve(energy, np.ones(win) / win, mode="same")

    # Threshold: 6*std balances sensitivity (e.g. 5.mp4 beep) vs false positives; 8*std missed some valid beeps
    threshold = float(np.mean(smooth) + 6 * np.std(smooth))

    # distance: at least 200ms apart to avoid multiple triggers for one beep
    peaks, _ = find_peaks(smooth, height=threshold, distance=int(sr * 0.2))

    if len(peaks) == 0:
        return []

    # Only search within the beginning window (default 3 seconds, covers the described "around 1s")
    max_search_s = 3.0
    peaks_in_window = [p for p in peaks if (p / sr) <= max_search_s]
    p0 = int(min(peaks_in_window) if peaks_in_window else min(peaks))
    t0 = p0 / sr

    return [{
        "t": round(float(t0), 4),
        "frame": int(t0 * fps),
        "confidence": 0.95
    }]
