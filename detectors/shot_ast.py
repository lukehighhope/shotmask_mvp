"""
AST-style Audio Spectrogram Transformer for gunshot binary classification.
Input: same as shot_cnn — (1, n_mels, n_frames) mel at candidate time (e.g. 64×32).
Architecture: patch embedding → transformer encoder → classifier head (CLS token).
Can be used as drop-in replacement for CNN in shot_audio (config: ast_gunshot_path).
"""
import os
import json
import numpy as np

# Reuse same mel spec as CNN so we can share data and swap model
from .shot_cnn import (
    mel_at_time,
    MEL_N_MELS,
    MEL_HOP,
    SEGMENT_DURATION,
)

# Default AST-style layout: 64×32 mel → 4×4 patches → 16×8 = 128 patches
PATCH_H = 4
PATCH_W = 4


def build_ast_model(
    n_mels=MEL_N_MELS,
    n_frames=32,
    patch_h=PATCH_H,
    patch_w=PATCH_W,
    embed_dim=192,
    depth=4,
    num_heads=4,
    mlp_ratio=2.0,
    num_classes=1,
):
    """
    Build AST-style ViT: patchify mel → transformer encoder → CLS head.
    Input shape: (B, 1, n_mels, n_frames). Patches: (n_mels//patch_h) * (n_frames//patch_w).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    n_ph = n_mels // patch_h
    n_pw = n_frames // patch_w
    num_patches = n_ph * n_pw
    patch_dim = 1 * patch_h * patch_w

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(1, embed_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))

        def forward(self, x):
            # x: (B, 1, H, W) -> (B, num_patches, embed_dim)
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return x

    class TransformerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    class TransformerBlock(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=2.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
            )

        def forward(self, x):
            # x: (B, N, C)
            x = x + self._attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

        def _attn(self, x):
            attn_out, _ = self.attn(x, x, x)
            return attn_out

    class AST(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.transformer = TransformerEncoder()
            self.head = nn.Linear(embed_dim, num_classes)
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)

        def forward(self, x):
            # x: (B, 1, n_mels, n_frames)
            B = x.shape[0]
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
            x = x + self.pos_embed
            x = self.transformer(x)
            cls_out = x[:, 0]
            return self.head(cls_out).squeeze(-1)

    return AST()


def load_ast_gunshot(path=None):
    """
    Load trained AST. path = .pt or from calibrated_detector_params.json ast_gunshot_path.
    Returns (model, device) or (None, None).
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
            path = cfg.get("ast_gunshot_path")
        if not path or not os.path.isfile(path):
            return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_kwargs" in state:
        kwargs = state["model_kwargs"]
    else:
        kwargs = {"n_mels": MEL_N_MELS, "n_frames": 32}
    model = build_ast_model(**kwargs)
    if model is None:
        return None, None
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_proba_one(model, device, mel):
    """Single mel (1, n_mels, n_frames) -> float P(gunshot). Same API as shot_cnn."""
    p = predict_proba_batch(model, device, [mel])
    return float(p[0]) if len(p) > 0 else 0.0


def predict_proba_batch(model, device, mel_batch):
    """Batch of mels -> (B,) probabilities. Same API as shot_cnn."""
    try:
        import torch
    except ImportError:
        return np.array([])
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
