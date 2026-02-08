"""
用 CNN 在 Mel 谱图上做枪声二分类，由网络学习特征（替代/辅助手工特征+LogReg）。
输入：以候选时刻为中心的短时窗 → Mel 谱图 → 2D CNN → P(枪声)。
"""
import os
import json
import numpy as np
import soundfile as sf

# Mel 谱图固定尺寸，与训练时一致
MEL_N_MELS = 64
MEL_HOP = 512
MEL_N_FFT = 2048
SEGMENT_DURATION = 0.35   # 以候选时刻为中心的前后窗总长 (s)
SR_TARGET = 48000


def _mel_spec(audio_1d, sr, n_mels=MEL_N_MELS, hop=MEL_HOP, n_fft=MEL_N_FFT):
    import librosa
    if sr != SR_TARGET:
        audio_1d = librosa.resample(audio_1d, orig_sr=sr, target_sr=SR_TARGET)
        sr = SR_TARGET
    mel = librosa.feature.melspectrogram(
        y=audio_1d, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, fmin=0, fmax=8000
    )
    # log scale
    mel = np.log1p(np.maximum(mel, 1e-9))
    return mel.astype(np.float32)


def audio_segment_at(audio_1d, sr, t_center_sec, duration_sec=SEGMENT_DURATION):
    """取以 t_center_sec 为中心、总长 duration_sec 的音频段；不足则补零。"""
    n = int(sr * duration_sec)
    half = n // 2
    center_idx = int(round(t_center_sec * sr))
    start = center_idx - half
    end = center_idx + half
    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0
    if end > len(audio_1d):
        pad_right = end - len(audio_1d)
        end = len(audio_1d)
    else:
        pad_right = 0
    seg = audio_1d[start:end]
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right), mode="constant", constant_values=0.0)
    return seg.astype(np.float32)


def mel_at_time(audio_1d, sr, t_sec, n_mels=MEL_N_MELS, hop=MEL_HOP, n_fft=MEL_N_FFT, n_frames=32):
    """
    在 t_sec 处截取 SEGMENT_DURATION 秒，做 Mel 谱图，并裁剪/填充到 (n_mels, n_frames)。
    返回 shape (1, n_mels, n_frames) 供 CNN 输入。
    """
    seg = audio_segment_at(audio_1d, sr, t_sec, duration_sec=SEGMENT_DURATION)
    mel = _mel_spec(seg, SR_TARGET, n_mels=n_mels, hop=hop, n_fft=n_fft)
    # mel: (n_mels, time)
    if mel.shape[1] >= n_frames:
        # 居中裁剪
        start = (mel.shape[1] - n_frames) // 2
        mel = mel[:, start : start + n_frames]
    else:
        mel = np.pad(mel, ((0, 0), (0, n_frames - mel.shape[1])), mode="constant", constant_values=np.log1p(1e-9))
    return mel[np.newaxis, ...].astype(np.float32)  # (1, n_mels, n_frames)


def build_cnn_model(n_mels=MEL_N_MELS, n_frames=32, num_classes=1):
    """构建小 2D CNN：输入 (1, n_mels, n_frames)，输出 1 个 logit（二分类）。"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    class GunshotCNN(nn.Module):
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
            # x: (B, 1, n_mels, n_frames)
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc(h).squeeze(-1)

    return GunshotCNN()


def load_cnn_gunshot(path=None):
    """
    加载训练好的 CNN。path 可为 .pt 或包含 cnn_gunshot_path 的 json。
    返回 (model, device) 或 (None, None)。
    """
    try:
        import torch
    except ImportError:
        return None, None
    if path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "calibrated_detector_params.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            path = cfg.get("cnn_gunshot_path")
        if not path or not os.path.isfile(path):
            return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn_model()
    if model is None:
        return None, None
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_proba_batch(model, device, mel_batch):
    """mel_batch: list of (1, n_mels, n_frames) 或 np array (B, 1, n_mels, n_frames)。返回 (B,) 概率。"""
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
    """单条 mel (1, n_mels, n_frames)，返回 float P(枪声)。"""
    p = predict_proba_batch(model, device, [mel])
    return float(p[0]) if len(p) > 0 else 0.0
