"""
用 CNN 在 Mel 谱图上做枪声二分类，由网络学习特征（替代/辅助手工特征+LogReg）。
输入：以候选时刻 t 为锚的短时窗 [t − pre, t + post]（默认前 0.05s / 后 0.08s）→ Mel → 2D CNN → P(枪声)。
"""
import os
import json
import numpy as np

from .config_paths import resolve_model_path
import soundfile as sf

# Mel 谱图固定尺寸，与训练时一致
MEL_N_MELS = 64
MEL_HOP = 512
MEL_N_FFT = 2048
# 不对称窗：更偏因果侧（枪口声后尾随能量/回声）
SEGMENT_PRE_SEC = 0.05   # t 之前
SEGMENT_POST_SEC = 0.08  # t 之后
SEGMENT_DURATION = SEGMENT_PRE_SEC + SEGMENT_POST_SEC  # 0.13s（供外部读取总长）
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


def audio_segment_at(
    audio_1d,
    sr,
    t_center_sec,
    pre_sec=SEGMENT_PRE_SEC,
    post_sec=SEGMENT_POST_SEC,
    *,
    duration_sec=None,
):
    """
    取 [t_center_sec - pre_sec, t_center_sec + post_sec]；越界样本用零填满。
    若传入 duration_sec（旧 API），退化为对称窗：总长 duration_sec、t 居中。
    """
    if duration_sec is not None:
        n = max(1, int(sr * float(duration_sec)))
        half = n // 2
        c = int(round(float(t_center_sec) * sr))
        start_idx, end_idx = c - half, c + half
    else:
        start_idx = int(round((float(t_center_sec) - float(pre_sec)) * sr))
        end_idx = int(round((float(t_center_sec) + float(post_sec)) * sr))
    la = len(audio_1d)
    out_len = max(0, end_idx - start_idx)
    if out_len == 0:
        return np.array([], dtype=np.float32)
    out = np.zeros(out_len, dtype=np.float64)
    src_start = max(0, min(start_idx, la))
    src_end = max(src_start, min(end_idx, la))
    if src_end > src_start:
        dst0 = src_start - start_idx
        chunk = audio_1d[src_start:src_end].astype(np.float64, copy=False)
        out[dst0 : dst0 + len(chunk)] = chunk
    return out.astype(np.float32)


def mel_at_time(audio_1d, sr, t_sec, n_mels=MEL_N_MELS, hop=MEL_HOP, n_fft=MEL_N_FFT, n_frames=32):
    """
    在 t_sec 处截取 [t−pre, t+post]（默认 0.05s / 0.08s），做 Mel 谱图，并裁剪/填充到 (n_mels, n_frames)。
    返回 shape (1, n_mels, n_frames) 供 CNN 输入。
    """
    seg = audio_segment_at(audio_1d, sr, t_sec)
    mel = _mel_spec(seg, SR_TARGET, n_mels=n_mels, hop=hop, n_fft=n_fft)
    # mel: (n_mels, time)
    if mel.shape[1] >= n_frames:
        # 居中裁剪
        start = (mel.shape[1] - n_frames) // 2
        mel = mel[:, start : start + n_frames]
    else:
        mel = np.pad(mel, ((0, 0), (0, n_frames - mel.shape[1])), mode="constant", constant_values=np.log1p(1e-9))
    return mel[np.newaxis, ...].astype(np.float32)  # (1, n_mels, n_frames)


def build_cnn_model(n_mels=MEL_N_MELS, n_frames=32, num_classes=1, arch="default"):
    """
    构建 2D CNN：输入 (1, n_mels, n_frames)，输出 1 个 logit（二分类）。
    arch: "default" = 小网络 (32->64->64); "deeper" = 更深+Dropout (32->64->128, fc 128->64->1)。
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    if arch == "deeper":
        return _build_cnn_deeper(num_classes)

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
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc(h).squeeze(-1)

    return GunshotCNN()


def _build_cnn_deeper(num_classes=1):
    """更深 CNN：3 个 block (32->64->128)，Dropout2d + fc 128->64->1。"""
    import torch.nn as nn

    class GunshotCNNDeeper(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc(h).squeeze(-1)

    return GunshotCNNDeeper()


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
    path = resolve_model_path(path)
    if not path or not os.path.isfile(path):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    arch = state.get("arch", "default") if isinstance(state, dict) else "default"
    model = build_cnn_model(arch=arch)
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
