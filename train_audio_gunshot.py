"""
Train audio gunshot classifier using dataset_split.json.

Uses *cali.txt directly as ground truth (no beep calibration needed).
Architecture: AST (Audio Spectrogram Transformer) on mel spectrograms.
Evaluates precision / recall / F1 on val set after every epoch.

Usage:
    python train_audio_gunshot.py                      # train + val
    python train_audio_gunshot.py --epochs 60 --augment
    python train_audio_gunshot.py --resume             # continue training
    python train_audio_gunshot.py --device cpu          # avoid Windows GPU TDR timeouts
"""

import os
import sys
import json
sys.stdout.reconfigure(line_buffering=True)
import argparse
import numpy as np
import soundfile as sf
import subprocess
import shutil
from pathlib import Path

DATA_ROOT  = Path("traning data")
SPLIT_JSON = DATA_ROOT / "dataset_split.json"
GT_TOL     = 0.05   # seconds: candidate within ±50ms of cali.txt = positive

# ─── helpers ─────────────────────────────────────────────────────────────────

def get_ffmpeg():
    exe = shutil.which("ffmpeg") or os.environ.get("FFMPEG")
    if exe:
        return exe
    for p in [r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
              r"C:\ffmpeg\bin\ffmpeg.exe"]:
        if os.path.exists(p):
            return p
    return "ffmpeg"


def extract_audio(video_path, tmp_dir="tmp"):
    os.makedirs(tmp_dir, exist_ok=True)
    base = Path(video_path).stem
    out  = os.path.join(tmp_dir, f"audio_{base}.wav")
    subprocess.run(
        [get_ffmpeg(), "-y", "-i", str(video_path),
         "-ac", "1", "-ar", "48000", "-vn", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    return out


def load_cali(video_path):
    """Load *cali.txt ground-truth shot times next to the video."""
    base = str(video_path).replace(".mp4", "")
    p    = Path(base + "cali.txt")
    if not p.exists():
        return []
    times = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                times.append(float(line))
            except ValueError:
                pass
    return sorted(times)


# ─── dataset building ────────────────────────────────────────────────────────

def build_dataset(video_paths, label="?"):
    """
    For each video: run audio detector → label candidates vs *cali.txt → extract mels.
    Returns (mels_array, labels_array).
    """
    from detectors.shot_audio  import detect_shots_improved, load_calibrated_params
    from detectors.shot_cnn    import mel_at_time, MEL_N_MELS

    cal_cfg = load_calibrated_params() or {}
    all_mels, all_labels = [], []

    for vp in video_paths:
        vp = Path(vp)
        if not vp.exists():
            print(f"  [SKIP] not found: {vp}")
            continue

        gt_times = load_cali(vp)
        if not gt_times:
            print(f"  [SKIP] no cali.txt: {vp.name}")
            continue

        print(f"  {label} {vp.name}  gt={len(gt_times)} shots")
        try:
            audio_path = extract_audio(vp)
        except Exception as e:
            print(f"    audio error: {e}")
            continue

        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = np.asarray(data, dtype=np.float64)

        try:
            result = detect_shots_improved(
                data, sr, fps=30.0,
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
        except Exception as e:
            print(f"    detect error: {e}")
            continue

        if isinstance(result, tuple) and len(result) >= 2:
            candidates = result[1]
        else:
            candidates = []

        if not candidates:
            print(f"    no candidates found")
            continue

        n_pos, n_neg = 0, 0
        for c in candidates:
            t     = float(c["t"])
            label_y = 1 if any(abs(t - gt) <= GT_TOL for gt in gt_times) else 0
            try:
                mel = mel_at_time(data, sr, t)
            except Exception:
                continue
            all_mels.append(mel)
            all_labels.append(label_y)
            if label_y == 1:
                n_pos += 1
            else:
                n_neg += 1

        print(f"    candidates={len(candidates)}  pos={n_pos}  neg={n_neg}")

    if not all_mels:
        return None, None
    return np.stack(all_mels, axis=0), np.array(all_labels, dtype=np.float32)


# ─── training ────────────────────────────────────────────────────────────────

def train(args):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("  pip install torch scikit-learn")
        return 1

    from detectors.shot_ast import build_ast_model, PATCH_H, PATCH_W
    from detectors.shot_cnn import MEL_N_MELS

    # ── load split ──
    with open(SPLIT_JSON, encoding="utf-8-sig") as f:
        split = json.load(f)

    train_paths = [DATA_ROOT / p for p in split.get("train", [])]
    val_paths   = [DATA_ROOT / p for p in split.get("val",   [])]

    print(f"Train: {len(train_paths)} videos   Val: {len(val_paths)} videos\n")

    # ── build / load cached datasets ──
    cache_tr  = Path("outputs/cache_train.npz")
    cache_val = Path("outputs/cache_val.npz")

    def load_or_build(cache, paths, label):
        if not args.rebuild and cache.exists():
            d = np.load(cache)
            print(f"  Loaded from cache: {cache}  ({len(d['y'])} samples)")
            return d["X"], d["y"]
        X, y = build_dataset(paths, label=label)
        if X is not None:
            os.makedirs("outputs", exist_ok=True)
            np.savez(cache, X=X, y=y)
            print(f"  Saved cache: {cache}")
        return X, y

    print("=== Building TRAIN dataset ===")
    X_tr, y_tr = load_or_build(cache_tr, train_paths, "TR")
    if X_tr is None:
        print("No train data.")
        return 1

    print(f"\nTrain total: {len(X_tr)}  pos={int(y_tr.sum())}  neg={int((y_tr==0).sum())}")

    print("\n=== Building VAL dataset ===")
    X_val, y_val = load_or_build(cache_val, val_paths, "VA")
    has_val = X_val is not None and len(y_val) > 0 and len(set(y_val)) == 2
    if not has_val:
        print("  No val data or only one class — skipping val evaluation.")

    # ── model ──
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but unavailable; using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model_kwargs = {
        "n_mels"    : MEL_N_MELS,
        "n_frames"  : 32,
        "patch_h"   : PATCH_H,
        "patch_w"   : PATCH_W,
        "embed_dim" : args.embed_dim,
        "depth"     : args.depth,
        "num_heads" : args.heads,
        "mlp_ratio" : 2.0,
        "num_classes": 1,
    }

    start_epoch = 0
    if args.resume and os.path.isfile(args.out):
        ckpt = torch.load(args.out, map_location=device)
        if isinstance(ckpt, dict) and "model_kwargs" in ckpt:
            model_kwargs = ckpt["model_kwargs"]
        model = build_ast_model(**model_kwargs).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.out} (epoch {start_epoch})")
    else:
        model = build_ast_model(**model_kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch, shuffle=True, num_workers=0
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    best_f1   = -1.0
    best_epoch = -1

    print(f"\n{'Ep':>4}  {'Loss':>7}  {'Val P':>7}  {'Val R':>7}  {'Val F1':>7}  {'Val AUC':>8}")
    print("-" * 50)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # ── train ──
        model.train()
        total_loss, n_b = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            if args.augment:
                # Gaussian noise
                bx = bx + 0.03 * torch.randn_like(bx)
                # Time shift
                shift = int(torch.randint(-4, 5, (1,)).item())
                if shift:
                    bx = torch.roll(bx, shift, dims=-1)
                # SpecAugment: freq mask
                if torch.rand(1).item() < 0.5:
                    f0 = int(torch.randint(0, max(1, bx.size(2)-8), (1,)).item())
                    bx[:, :, f0:f0+8, :] = bx.min()
                # SpecAugment: time mask
                if torch.rand(1).item() < 0.5:
                    t0 = int(torch.randint(0, max(1, bx.size(3)-4), (1,)).item())
                    bx[:, :, :, t0:t0+4] = bx.min()
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        scheduler.step()
        avg_loss = total_loss / max(n_b, 1)

        # ── val ──
        val_str = ""
        if has_val:
            model.eval()
            bs = max(1, args.batch)
            probs_chunks = []
            with torch.no_grad():
                for i in range(0, len(X_val), bs):
                    xv = torch.from_numpy(X_val[i : i + bs]).to(device)
                    logits = model(xv)
                    probs_chunks.append(
                        torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                    )
            probs = np.concatenate(probs_chunks, axis=0)
            preds = (probs >= 0.5).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y_val.astype(int), preds, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(y_val.astype(int), probs)
            except Exception:
                auc = float("nan")
            val_str = f"  {p:>7.3f}  {r:>7.3f}  {f1:>7.3f}  {auc:>8.3f}"

            if f1 > best_f1:
                best_f1    = f1
                best_epoch = epoch + 1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_kwargs"    : model_kwargs,
                    "epoch"           : epoch + 1,
                    "val_f1"          : best_f1,
                }, args.out)
        else:
            val_str = "  (no val)"
            # save every epoch if no val
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_kwargs"    : model_kwargs,
                "epoch"           : epoch + 1,
            }, args.out)

        print(f"{epoch+1:>4}  {avg_loss:>7.4f}{val_str}")

    print(f"\nBest val F1={best_f1:.3f} at epoch {best_epoch}")
    print(f"Model saved -> {args.out}")

    if args.save_config:
        cfg_path = "calibrated_detector_params.json"
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["ast_gunshot_path"] = os.path.abspath(args.out)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"Updated {cfg_path}")
    return 0


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train audio gunshot AST from dataset_split.json")
    ap.add_argument("--epochs",      type=int,   default=50)
    ap.add_argument("--batch",       type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--embed-dim",   type=int,   default=192)
    ap.add_argument("--depth",       type=int,   default=4)
    ap.add_argument("--heads",       type=int,   default=4)
    ap.add_argument("--augment",     action="store_true", help="SpecAugment + noise")
    ap.add_argument("--resume",      action="store_true", help="Load and continue")
    ap.add_argument("--out",         default="outputs/ast_gunshot.pt")
    ap.add_argument("--rebuild",     action="store_true", help="Ignore cache, rebuild dataset")
    ap.add_argument("--save-config", action="store_true",
                    help="Write path into calibrated_detector_params.json")
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help='Force device (default: auto). Use cpu if GPU hits Windows TDR timeouts.',
    )
    args = ap.parse_args()
    sys.exit(train(args))


if __name__ == "__main__":
    main()
