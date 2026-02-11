"""
Train AST-style Spectrogram Transformer for gunshot binary classification.
Uses same data pipeline as train_cnn_gunshot (mel segments from candidates + ref labels). Ref: *cali.txt preferred, else *.txt.
Usage:
  python train_ast_gunshot.py --folder "traning data/01032026" --epochs 40 --out outputs/ast_gunshot.pt
  python train_ast_gunshot.py --folder "traning data" --recursive --epochs 30 --augment --out outputs/ast_gunshot.pt --save-config
  python train_ast_gunshot.py --use-split --epochs 40   # dataset_split: last video per folder = val, new folders auto-included
"""
import os
import sys
import argparse
import numpy as np
import soundfile as sf

from detectors.shot_audio import detect_shots_improved, load_calibrated_params
from detectors.shot_cnn import mel_at_time, MEL_N_MELS, SEGMENT_DURATION
from detectors.shot_ast import build_ast_model, PATCH_H, PATCH_W
from ref_from_image import get_ref_times_and_source, get_beep_t_for_video
from train_logreg_multivideo import (
    get_ffmpeg,
    get_ffprobe,
    extract_audio,
    get_fps_duration,
)

GT_TOL = 0.04


def build_mel_dataset(folder, cal_cfg=None, only_videos=None):
    """Same as train_cnn_gunshot: per-video ref (*cali.txt else *.txt) + candidates → (mels, labels).
    only_videos: if set, only process these basenames (for train/val split)."""
    folder = os.path.abspath(folder)
    cal_cfg = cal_cfg or load_calibrated_params() or {}
    mels, labels = [], []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".mp4"):
            continue
        if only_videos is not None and f not in only_videos:
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
            print("  No beep, skip")
            continue
        ref_times, ref_src = get_ref_times_and_source(vp, beep_t)
        if not ref_times:
            print("  No ref (*cali.txt / *.txt), skip")
            continue
        print("  Ref:", ref_src)
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
            _, candidates = result[0], result[1]
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
    ap = argparse.ArgumentParser(description="Train AST (Spectrogram Transformer) for gunshot classification")
    ap.add_argument("--folder", default="traning data/01032026", help="Folder with .mp4 and .txt ref (or root when --recursive)")
    ap.add_argument("--recursive", action="store_true", help="Use all subfolders of --folder")
    ap.add_argument("--use-split", action="store_true", help="Use dataset_split: train on all-but-last video per folder under traning data. New folders auto-included.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="outputs/ast_gunshot.pt")
    ap.add_argument("--save-config", action="store_true", help="Write ast_gunshot_path into calibrated_detector_params.json")
    ap.add_argument("--augment", action="store_true", help="Noise + time shift + SpecAugment")
    ap.add_argument("--resume", action="store_true", help="Load --out and train --epochs more")
    ap.add_argument("--embed-dim", type=int, default=192, help="Transformer embed dim")
    ap.add_argument("--depth", type=int, default=4, help="Number of transformer layers")
    ap.add_argument("--heads", type=int, default=4, help="Attention heads")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch required: pip install torch")
        return 1

    if args.use_split:
        try:
            from dataset_split import get_train_folders_with_videos
        except ImportError:
            print("dataset_split.py required for --use-split (same dir as train_ast_gunshot.py)")
            return 1
        folders_with_videos = get_train_folders_with_videos()
        if not folders_with_videos:
            print("No train folders from dataset_split (traning data empty or no .mp4?)")
            return 1
        print(f"Using dataset_split (last video per folder = val): {len(folders_with_videos)} folder(s)\n")
        mels, labels = [], []
        for folder, only_videos in folders_with_videos:
            m, l = build_mel_dataset(folder, only_videos=only_videos)
            mels.extend(m)
            labels.extend(l)
    elif args.recursive:
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
        print(f"Using {len(subfolders)} folder(s)\n")
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

    X = np.stack(mels, axis=0)
    y = np.asarray(labels, dtype=np.float32)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    print(f"\nTotal: {len(X)} samples, pos={n_pos}, neg={n_neg}")

    model_kwargs = {
        "n_mels": MEL_N_MELS,
        "n_frames": 32,
        "patch_h": PATCH_H,
        "patch_w": PATCH_W,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.heads,
        "mlp_ratio": 2.0,
        "num_classes": 1,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = None
    if args.resume and os.path.isfile(args.out):
        ckpt = torch.load(args.out, map_location=device)
        if isinstance(ckpt, dict) and "model_kwargs" in ckpt:
            model_kwargs = ckpt["model_kwargs"]
    model = build_ast_model(**model_kwargs)
    if model is None:
        print("Failed to build AST model")
        return 1
    model.to(device)
    start_epoch = 0
    if ckpt is not None and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.out}, training {args.epochs} more epochs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
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
                batch_x = batch_x + 0.03 * torch.randn_like(batch_x, device=device)
                roll = int(torch.randint(-4, 5, (1,)).item())
                if roll != 0:
                    batch_x = torch.roll(batch_x, roll, dims=-1)
                if torch.rand(1, device=device).item() < 0.5:
                    f_size = min(8, batch_x.size(2) - 1)
                    if f_size > 0:
                        f_start = int(torch.randint(0, batch_x.size(2) - f_size + 1, (1,), device=device).item())
                        batch_x[:, :, f_start : f_start + f_size, :] = batch_x.min()
                if torch.rand(1, device=device).item() < 0.5:
                    t_size = min(4, batch_x.size(3) - 1)
                    if t_size > 0:
                        t_start = int(torch.randint(0, batch_x.size(3) - t_size + 1, (1,), device=device).item())
                        batch_x[:, :, :, t_start : t_start + t_size] = batch_x.min()
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        scheduler.step()
        avg_loss = total_loss / max(n_b, 1)
        if avg_loss < best_loss:
            best_loss, best_epoch = avg_loss, epoch + 1
        if (epoch - start_epoch + 1) % 5 == 0 or epoch == start_epoch:
            print(f"  Epoch {epoch+1}/{start_epoch + args.epochs} loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "epoch": start_epoch + args.epochs,
    }, args.out)
    print(f"Saved: {args.out} (best loss {best_loss:.4f} @ epoch {best_epoch})")

    if args.save_config:
        cfg_path = "calibrated_detector_params.json"
        if os.path.isfile(cfg_path):
            import json
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["ast_gunshot_path"] = os.path.abspath(args.out)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"Updated {cfg_path} with ast_gunshot_path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
