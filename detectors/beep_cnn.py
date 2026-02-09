"""
Beep 二分类：输入 0.35s 音频段的 Mel 谱图，2D CNN 输出 P(beep)。
用于区分 beep（持续单音）与枪声/噪声（短脉冲/宽带）。
"""
import os
import json
import numpy as np

# 与训练一致
BEEP_SEGMENT_DURATION = 0.35   # 秒
BEEP_MEL_N_MELS = 64
BEEP_MEL_HOP = 512
BEEP_MEL_N_FFT = 2048
BEEP_MEL_N_FRAMES = 32
SR_TARGET = 48000


def _mel_spec(audio_1d, sr, n_mels=BEEP_MEL_N_MELS, hop=BEEP_MEL_HOP, n_fft=BEEP_MEL_N_FFT):
    import librosa
    if sr != SR_TARGET:
        audio_1d = librosa.resample(audio_1d, orig_sr=sr, target_sr=SR_TARGET)
        sr = SR_TARGET
    mel = librosa.feature.melspectrogram(
        y=audio_1d, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=0, fmax=8000
    )
    mel = np.log1p(np.maximum(mel, 1e-9))
    return mel.astype(np.float32)


def audio_segment_from_start(audio_1d, sr, t_start_sec, duration_sec=BEEP_SEGMENT_DURATION):
    """从 t_start_sec 开始取 duration_sec 秒；不足则补零。"""
    n = int(sr * duration_sec)
    start_idx = int(round(t_start_sec * sr))
    end_idx = start_idx + n
    if start_idx < 0:
        pad_left = -start_idx
        start_idx = 0
    else:
        pad_left = 0
    if end_idx > len(audio_1d):
        pad_right = end_idx - len(audio_1d)
        end_idx = len(audio_1d)
    else:
        pad_right = 0
    seg = audio_1d[start_idx:end_idx]
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right), mode="constant", constant_values=0.0)
    return seg.astype(np.float32)


def mel_at_time(audio_1d, sr, t_start_sec, duration_sec=BEEP_SEGMENT_DURATION,
                n_mels=BEEP_MEL_N_MELS, hop=BEEP_MEL_HOP, n_fft=BEEP_MEL_N_FFT, n_frames=BEEP_MEL_N_FRAMES):
    """
    从 t_start_sec 起截取 duration_sec 秒，做 Mel 谱图，裁剪/填充到 (n_mels, n_frames)。
    返回 (1, n_mels, n_frames) 供 CNN。
    """
    seg = audio_segment_from_start(audio_1d, sr, t_start_sec, duration_sec)
    mel = _mel_spec(seg, SR_TARGET, n_mels=n_mels, hop=hop, n_fft=n_fft)
    if mel.shape[1] >= n_frames:
        mel = mel[:, :n_frames]
    else:
        mel = np.pad(mel, ((0, 0), (0, n_frames - mel.shape[1])), mode="constant", constant_values=np.log1p(1e-9))
    return mel[np.newaxis, ...].astype(np.float32)


def build_beep_cnn(n_mels=BEEP_MEL_N_MELS, n_frames=BEEP_MEL_N_FRAMES, num_classes=1):
    """小型 2D CNN：输入 (1, n_mels, n_frames)，输出 1 个 logit。"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    class BeepCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc(h).squeeze(-1)

    return BeepCNN()


def load_cnn_beep(path=None):
    """加载训练好的 beep CNN。path 可为 .pt 或通过 calibrated_detector_params.json 的 cnn_beep_path。"""
    try:
        import torch
    except ImportError:
        return None, None
    if path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "calibrated_detector_params.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            path = cfg.get("cnn_beep_path")
        if not path or not os.path.isfile(path):
            return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    model = build_beep_cnn()
    if model is None:
        return None, None
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_proba_batch(model, device, mel_batch):
    """mel_batch: list of (1, n_mels, n_frames) 或 (B, 1, n_mels, n_frames)。返回 (B,) P(beep)。"""
    try:
        import torch
    except ImportError:
        return None
    if not mel_batch:
        return np.array([])
    if isinstance(mel_batch, list):
        mel_batch = np.stack(mel_batch, axis=0)
    x = torch.from_numpy(mel_batch).to(device)
    with torch.no_grad():
        logits = model(x)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def predict_proba_one(model, device, mel):
    """单条 mel (1, n_mels, n_frames)，返回 float P(beep)。"""
    p = predict_proba_batch(model, device, [mel])
    return float(p[0]) if len(p) > 0 else 0.0
