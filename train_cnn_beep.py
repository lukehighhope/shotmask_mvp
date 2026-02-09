"""
用 *beep.txt 真值训练小型 2D CNN：输入 0.35s 段的 Mel 谱图，二分类 beep / 非 beep。
正样本：每段视频 [beep_t, beep_t+0.35]；负样本：随机时间窗口（避开 beep 附近）。

Usage:
  python train_cnn_beep.py
  python train_cnn_beep.py --folders "traning data/01032026" "traning data/outdoor-20260208T235429Z-1-001"
  python train_cnn_beep.py --epochs 50 --out outputs/cnn_beep.pt --augment
"""
import os
import sys
import subprocess
import argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_beep_detector import collect_beep_gt
from extract_audio_plot import get_ffmpeg_cmd
from detectors.beep_cnn import (
    mel_at_time,
    build_beep_cnn,
    BEEP_SEGMENT_DURATION,
    BEEP_MEL_N_MELS,
    BEEP_MEL_N_FRAMES,
)

NEGATIVES_PER_VIDEO = 10   # 每段视频采样的负样本数
AVOID_BEEP_MARGIN = 0.8   # 负样本与 beep 时间至少间隔（秒）


def extract_audio(video_path, out_dir="tmp"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0].replace("-", "_")
    wav = os.path.join(out_dir, f"beep_train_{base}.wav")
    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        return None
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", video_path, "-ac", "1", "-ar", "48000", "-vn", wav],
            check=True, capture_output=True
        )
        return wav
    except Exception:
        return None


def build_beep_dataset(folders=None):
    """
    从 *beep.txt 收集 (video_path, beep_t)；对每段视频提取 1 个正样本 + N 个负样本的 Mel。
    返回 (mels, labels)。
    """
    entries = collect_beep_gt(folders)
    if not entries:
        return [], []

    mels, labels = [], []
    rng = np.random.default_rng(42)

    for video_path, beep_t, _ in entries:
        wav = extract_audio(video_path)
        if not wav or not os.path.isfile(wav):
            print(f"  Skip (no audio): {video_path}")
            continue
        data, sr = sf.read(wav)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = np.asarray(data, dtype=np.float64)
        duration_s = len(data) / sr

        if beep_t + BEEP_SEGMENT_DURATION > duration_s or beep_t < 0:
            print(f"  Skip (beep out of range): {video_path} beep_t={beep_t:.2f}")
            continue

        # 正样本
        mel_pos = mel_at_time(data, sr, beep_t)
        mels.append(mel_pos)
        labels.append(1)

        # 负样本：随机起点，避开 [beep_t - margin, beep_t + margin + duration]
        avoid_start = beep_t - AVOID_BEEP_MARGIN
        avoid_end = beep_t + BEEP_SEGMENT_DURATION + AVOID_BEEP_MARGIN
        valid_start_min = max(0.5, 0)
        valid_start_max = max(0.5, duration_s - BEEP_SEGMENT_DURATION - 0.5)
        if valid_start_max <= valid_start_min:
            continue
        for _ in range(NEGATIVES_PER_VIDEO):
            t = rng.uniform(valid_start_min, valid_start_max)
            if avoid_start <= t <= avoid_end:
                continue
            if t + BEEP_SEGMENT_DURATION > duration_s:
                continue
            mel_neg = mel_at_time(data, sr, t)
            mels.append(mel_neg)
            labels.append(0)

    return mels, labels


def main():
    ap = argparse.ArgumentParser(description="Train 2D CNN for beep vs non-beep (mel 0.35s)")
    ap.add_argument("--folders", nargs="*", default=None, help="Folders with *beep.txt (default: traning data)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/cnn_beep.pt")
    ap.add_argument("--augment", action="store_true", help="Add noise + time shift on mel")
    ap.add_argument("--save-config", action="store_true", help="Write cnn_beep_path to calibrated_detector_params.json")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch required: pip install torch")
        return 1

    print("Collecting beep dataset from *beep.txt ...")
    mels, labels = build_beep_dataset(args.folders)
    if not mels or len(set(labels)) < 2:
        print("Not enough data or only one class.")
        return 1

    X = np.stack(mels, axis=0)
    y = np.asarray(labels, dtype=np.float32)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    print(f"Total: {len(X)} samples, pos={n_pos}, neg={n_neg}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_beep_cnn()
    if model is None:
        print("Failed to build model")
        return 1
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    best_loss, best_epoch = float("inf"), -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_b = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if args.augment:
                batch_x = batch_x + 0.03 * torch.randn_like(batch_x, device=device)
                roll = int(torch.randint(-2, 3, (1,)).item())
                if roll != 0:
                    batch_x = torch.roll(batch_x, roll, dims=-1)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        avg_loss = total_loss / max(n_b, 1)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss, best_epoch = avg_loss, epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_mels": BEEP_MEL_N_MELS,
        "n_frames": BEEP_MEL_N_FRAMES,
        "epoch": args.epochs,
    }, args.out)
    print(f"\nSaved: {args.out} (best loss {best_loss:.4f} @ epoch {best_epoch})")

    if args.save_config:
        cfg_path = "calibrated_detector_params.json"
        import json
        cfg = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["cnn_beep_path"] = os.path.abspath(args.out)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Updated {cfg_path} with cnn_beep_path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
