"""
用 01032026 五段视频的真值(1.txt~5.txt)，从候选处截取 Mel 谱图，训练小 CNN 做枪声二分类。
网络从谱图学习特征，替代/辅助手工特征+LogReg。

提高准确度：多训几轮(--epochs)、开数据增强(--augment)、学习率衰减(默认开)、继续训练(--resume)。
Usage:
  python train_cnn_gunshot.py --folder 01032026 --epochs 50 --out outputs/cnn_gunshot.pt --save-config
  python train_cnn_gunshot.py --folder 01032026 --epochs 20 --resume --out outputs/cnn_gunshot.pt   # 在已有模型上再训 20 轮
  python train_cnn_gunshot.py --folder "traning data" --recursive   # 用 traning data 下所有子目录（01032026 + outdoor 等，含绝对时间 txt）
"""
import os
import sys
import argparse
import numpy as np
import soundfile as sf

from detectors.shot_audio import detect_shots_improved, load_calibrated_params
from detectors.shot_cnn import (
    mel_at_time,
    build_cnn_model,
    SEGMENT_DURATION,
    MEL_N_MELS,
    MEL_HOP,
)
from ref_from_image import get_ref_times_for_video, get_beep_t_for_video
from train_logreg_multivideo import (
    get_ffmpeg,
    get_ffprobe,
    extract_audio,
    get_fps_duration,
)

GT_TOL = 0.04
WINDOW_AFTER = 0.08


def build_mel_dataset(folder, cal_cfg=None):
    """对 folder 下每个 mp4：取 ref（同目录 .txt，支持绝对时间或 splits）、候选，对每个候选截 Mel 并标 0/1。返回 (mels, labels)."""
    folder = os.path.abspath(folder)
    cal_cfg = cal_cfg or load_calibrated_params() or {}
    mels, labels = [], []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".mp4"):
            continue
        vp = os.path.join(folder, f)
        if not os.path.isfile(vp):
            continue
        print("Processing:", vp)
        try:
            audio_path = extract_audio(vp)
            fps, _ = get_fps_duration(vp)
        except Exception as e:
            print("  Error:", e)
            continue
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = np.asarray(data, dtype=np.float64)
        beep_t = get_beep_t_for_video(vp, audio_path, fps)
        if beep_t <= 0:
            print("  No beep (and no override), skip")
            continue
        ref_times = get_ref_times_for_video(vp, beep_t)
        if not ref_times:
            print("  No ref .txt, skip")
            continue
        override_path = os.path.join(os.path.dirname(vp), "beep_overrides.json")
        if os.path.isfile(override_path):
            print(f"  Beep t={beep_t:.2f}s (from beep_overrides.json)")
        result = detect_shots_improved(
            data, sr, fps,
            cluster_window_sec=cal_cfg.get("cluster_window_sec", 0.25),
            mad_k=cal_cfg.get("mad_k", 6.0),
            candidate_min_dist_ms=cal_cfg.get("candidate_min_dist_ms", 50),
            score_weights=cal_cfg.get("score_weights"),
            min_confidence_threshold=None,
            logreg_model=None,
            return_candidates=True,
            return_feature_context=False,
            use_mfcc=True,
        )
        if isinstance(result, tuple) and len(result) >= 2:
            shots_cur, candidates = result[0], result[1]
        else:
            candidates = []
        if not candidates:
            print("  No candidates, skip")
            continue
        n_pos, n_neg = 0, 0
        for c in candidates:
            t = float(c["t"])
            if t < beep_t:
                continue
            label = 1 if any(abs(t - rt) <= GT_TOL for rt in ref_times) else 0
            mel = mel_at_time(data, sr, t)
            mels.append(mel)
            labels.append(label)
            if label == 1:
                n_pos += 1
            else:
                n_neg += 1
        print(f"  Candidates: {len(candidates)}, pos={n_pos}, neg={n_neg}")
    return mels, labels


def main():
    ap = argparse.ArgumentParser(description="Train CNN on mel-spectrograms for gunshot classification")
    ap.add_argument("--folder", default="01032026", help="Single folder with .mp4 and .txt ref (or root when --recursive)")
    ap.add_argument("--recursive", action="store_true", help="Use all immediate subfolders of --folder (e.g. traning data -> 01032026, outdoor-...)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/cnn_gunshot.pt")
    ap.add_argument("--save-config", action="store_true", help="Write cnn_gunshot_path into calibrated_detector_params.json")
    ap.add_argument("--augment", action="store_true", help="Data augmentation: noise + time shift on mel (recommended for accuracy)")
    ap.add_argument("--resume", action="store_true", help="Load checkpoint from --out and train for --epochs more")
    ap.add_argument("--arch", choices=["default", "deeper"], default="default", help="CNN architecture: default (small) or deeper (3 blocks + Dropout)")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch required: pip install torch")
        return 1

    if args.recursive:
        root = os.path.abspath(args.folder)
        subfolders = [
            os.path.join(root, d)
            for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))
            and any(f.lower().endswith(".mp4") for f in os.listdir(os.path.join(root, d)))
        ]
        if not subfolders:
            print(f"No subfolders with .mp4 under {root}")
            return 1
        print(f"Using {len(subfolders)} folder(s): {subfolders}\n")
        mels, labels = [], []
        for folder in subfolders:
            m, l = build_mel_dataset(folder)
            mels.extend(m)
            labels.extend(l)
    else:
        mels, labels = build_mel_dataset(args.folder)
    if not mels or len(set(labels)) < 2:
        print("Not enough data or only one class.")
        return 1

    X = np.stack(mels, axis=0)  # (N, 1, n_mels, n_frames)
    y = np.asarray(labels, dtype=np.float32)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    print(f"\nTotal: {len(X)} samples, pos={n_pos}, neg={n_neg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = args.arch
    ckpt = None
    if args.resume and os.path.isfile(args.out):
        ckpt = torch.load(args.out, map_location=device)
        arch = ckpt.get("arch", arch)
    model = build_cnn_model(arch=arch)
    if model is None:
        print("Failed to build model")
        return 1
    model.to(device)
    start_epoch = 0
    if ckpt is not None and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.out} (arch={arch}, epoch {start_epoch}), training {args.epochs} more epochs")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    best_loss, best_epoch = float("inf"), -1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss, n_b = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if args.augment:
                batch_x = batch_x + 0.04 * torch.randn_like(batch_x, device=device)
                roll = int(torch.randint(-3, 4, (1,)).item())
                if roll != 0:
                    batch_x = torch.roll(batch_x, roll, dims=-1)
                # SpecAugment: freq mask (dim=2 = n_mels)
                if torch.rand(1, device=device).item() < 0.5:
                    f_size = int(torch.randint(1, 8, (1,), device=device).item())
                    f_size = min(f_size, batch_x.size(2) - 1)
                    if f_size > 0:
                        f_start = int(torch.randint(0, batch_x.size(2) - f_size + 1, (1,), device=device).item())
                        batch_x[:, :, f_start : f_start + f_size, :] = batch_x.min()
                # SpecAugment: time mask (dim=3 = n_frames)
                if torch.rand(1, device=device).item() < 0.5:
                    t_size = int(torch.randint(1, 4, (1,), device=device).item())
                    t_size = min(t_size, batch_x.size(3) - 1)
                    if t_size > 0:
                        t_start = int(torch.randint(0, batch_x.size(3) - t_size + 1, (1,), device=device).item())
                        batch_x[:, :, :, t_start : t_start + t_size] = batch_x.min()
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
            best_loss, best_epoch = avg_loss, epoch + 1
        if (epoch - start_epoch + 1) % 5 == 0 or epoch == start_epoch:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1}/{start_epoch + args.epochs} loss={avg_loss:.4f} lr={lr_now:.2e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_mels": MEL_N_MELS,
        "n_frames": 32,
        "epoch": start_epoch + args.epochs,
        "arch": arch,
    }, args.out)
    print(f"Saved: {args.out} (best loss {best_loss:.4f} @ epoch {best_epoch})")

    if args.save_config:
        cfg_path = "calibrated_detector_params.json"
        if os.path.isfile(cfg_path):
            import json
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["cnn_gunshot_path"] = os.path.abspath(args.out)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"Updated {cfg_path} with cnn_gunshot_path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
