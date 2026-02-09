"""
枪声检测：先滤波，再对能量包络做阈值判断，能量 > threshold 的峰记为一枪。
简单、可解释，适合作为 baseline 或与 shot_audio 对比。
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, find_peaks


def bandpass(data, sr, low=400, high=6000):
    """带通滤波，突出枪声频段。"""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    nyq = sr / 2.0
    low = max(0.1, min(low, nyq - 1))
    high = max(low + 100, min(high, nyq - 1))
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, data)


def energy_envelope(data, sr, smooth_sec=0.01):
    """滤波后取绝对值，再短时平滑得到能量包络。"""
    env = np.abs(data)
    win = max(1, int(sr * smooth_sec))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    return env.astype(np.float64)


def detect_shots_energy(
    audio_path=None,
    data=None,
    sr=None,
    fps=30.0,
    filter_low=400,
    filter_high=6000,
    energy_threshold=0.005,
    min_dist_sec=0.08,
    smooth_sec=0.01,
    normalize=True,
    t0_beep=None,
):
    """
    先滤波，再在能量包络上做阈值检测：包络 > energy_threshold 的峰记为一枪。

    Args:
        audio_path: 音频文件路径（与 data/sr 二选一）
        data, sr: 若不用 audio_path，可直接传入波形和采样率
        fps: 用于计算 frame 字段
        filter_low, filter_high: 带通滤波范围 (Hz)
        energy_threshold: 归一化能量包络阈值，大于此值的峰记为 shot（默认 0.005≈100% 召回，评估容差 ±0.1s；精确率约 15%，可适当提高以减 FP）
        min_dist_sec: 两枪最小间隔 (s)，避免同一枪多峰
        smooth_sec: 包络平滑窗长 (s)
        normalize: 是否将包络归一化到 [0,1]，再与 energy_threshold 比较
        t0_beep: 若给出 beep 时刻，只保留 t >= t0_beep 的检测

    Returns:
        list of {"t", "frame", "confidence", "energy"}
    """
    if audio_path is not None:
        data, sr = sf.read(audio_path)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    if data is None or sr is None:
        return []

    # 1) 滤波
    filtered = bandpass(data, sr, low=filter_low, high=filter_high)
    # 2) 能量包络
    env = energy_envelope(filtered, sr, smooth_sec=smooth_sec)
    if normalize:
        env_max = np.max(env)
        if env_max > 1e-12:
            env = env / env_max
        else:
            return []

    # 3) 找包络 > threshold 的峰，且峰间至少 min_dist_sec
    min_dist_samples = max(1, int(sr * min_dist_sec))
    peaks, _ = find_peaks(env, height=energy_threshold, distance=min_dist_samples)
    if len(peaks) == 0:
        return []

    times = peaks.astype(np.float64) / sr
    if t0_beep is not None:
        keep = times >= float(t0_beep)
        peaks = peaks[keep]
        times = times[keep]

    return [
        {
            "t": round(float(t), 4),
            "frame": int(round(float(t) * fps)),
            "confidence": 1.0,
            "energy": round(float(env[p]), 4),
        }
        for p, t in zip(peaks, times)
    ]


def get_energy_waveform(audio_path=None, data=None, sr=None, filter_low=400, filter_high=6000, smooth_sec=0.01, normalize=True):
    """
    仅计算滤波后的能量包络（用于画图）。返回 (t_axis, energy_axis)。
    """
    if audio_path is not None:
        data, sr = sf.read(audio_path)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    if data is None or sr is None:
        return np.array([]), np.array([])
    filtered = bandpass(data, sr, low=filter_low, high=filter_high)
    env = energy_envelope(filtered, sr, smooth_sec=smooth_sec)
    if normalize:
        env_max = np.max(env)
        if env_max > 1e-12:
            env = env / env_max
    t_axis = np.arange(len(env), dtype=np.float64) / sr
    return t_axis, env.astype(np.float64)


if __name__ == "__main__":
    import argparse
    import os
    # 从视频提音频需要 ffmpeg，这里仅支持直接传音频路径
    ap = argparse.ArgumentParser(description="Energy-threshold shot detection: filter -> envelope > 0.3 = shot")
    ap.add_argument("audio", nargs="?", default=None, help="Audio file path (.wav)")
    ap.add_argument("--threshold", type=float, default=0.3, help="Energy threshold (default 0.3)")
    ap.add_argument("--min-dist", type=float, default=0.08, help="Min distance between shots (s)")
    ap.add_argument("--t0-beep", type=float, default=None, help="Only keep shots after this time (s)")
    args = ap.parse_args()
    if not args.audio or not os.path.isfile(args.audio):
        print("Usage: python -m detectors.shot_energy path/to/audio.wav [--threshold 0.3] [--min-dist 0.08] [--t0-beep 1.13]")
        raise SystemExit(1)
    shots = detect_shots_energy(
        audio_path=args.audio,
        energy_threshold=args.threshold,
        min_dist_sec=args.min_dist,
        t0_beep=args.t0_beep,
    )
    print(f"Detected {len(shots)} shots (energy > {args.threshold}):")
    for s in shots:
        print(f"  t={s['t']:.3f}s  energy={s['energy']:.3f}")
