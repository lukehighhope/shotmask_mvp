"""
Compare CNN vs AST gunshot detector on the same dataset (ref-based metrics).
Usage: python compare_cnn_ast.py --folder "traning data/01032026"
"""
import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors.shot_audio import detect_shots, load_calibrated_params
from ref_from_image import get_ref_times_for_video, get_beep_t_for_video
from extract_audio_plot import calibration_metrics
from train_logreg_multivideo import get_ffmpeg, get_ffprobe, extract_audio, get_fps_duration

MAX_MATCH_S = 0.12


def get_ref_for_video(video_path):
    """Return (ref_times, beep_t) or (None, None)."""
    try:
        audio_path = extract_audio(video_path)
        fps, _ = get_fps_duration(video_path)
    except Exception:
        return None, None
    beep_t = get_beep_t_for_video(video_path, audio_path, fps)
    if beep_t <= 0:
        return None, None
    ref = get_ref_times_for_video(video_path, beep_t)
    return ref, beep_t


def main():
    ap = argparse.ArgumentParser(description="Compare CNN vs AST shot detection on folder with ref .txt")
    ap.add_argument("--folder", default="traning data/01032026", help="Folder with 1.mp4..5.mp4 and 1.txt..5.txt")
    ap.add_argument("--match-tol", type=float, default=0.12, help="Match tolerance in seconds (default 0.12)")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    max_match_s = args.match_tol

    # Collect videos and refs
    videos = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".mp4"):
            continue
        vp = os.path.join(folder, f)
        if not os.path.isfile(vp):
            continue
        ref, beep_t = get_ref_for_video(vp)
        if ref is None or len(ref) == 0:
            print(f"Skip {f}: no ref or beep")
            continue
        try:
            audio_path = extract_audio(vp)
            fps, _ = get_fps_duration(vp)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue
        videos.append({"path": vp, "name": f, "audio": audio_path, "fps": fps, "ref": ref})
    if not videos:
        print("No videos with ref found.")
        return 1

    total_ref = sum(len(v["ref"]) for v in videos)
    results_cnn = []
    results_ast = []

    for v in videos:
        det_cnn = detect_shots(v["audio"], v["fps"], use_ast_gunshot=False)
        det_ast = detect_shots(v["audio"], v["fps"], use_ast_gunshot=True)
        t_cnn = [float(s["t"]) for s in det_cnn]
        t_ast = [float(s["t"]) for s in det_ast]
        n_cnn, mae_cnn = calibration_metrics(v["ref"], t_cnn, max_match_s=max_match_s)
        n_ast, mae_ast = calibration_metrics(v["ref"], t_ast, max_match_s=max_match_s)
        results_cnn.append((v["name"], len(v["ref"]), len(t_cnn), n_cnn, mae_cnn))
        results_ast.append((v["name"], len(v["ref"]), len(t_ast), n_ast, mae_ast))

    # Summary: total matched, global MAE = weighted avg of per-video MAE by matched count
    n_cnn_all = sum(r[3] for r in results_cnn)
    n_ast_all = sum(r[3] for r in results_ast)
    sum_mae_cnn = sum(r[3] * r[4] for r in results_cnn if r[3] > 0)
    sum_mae_ast = sum(r[3] * r[4] for r in results_ast if r[3] > 0)
    mae_cnn_global = sum_mae_cnn / n_cnn_all if n_cnn_all else float("inf")
    mae_ast_global = sum_mae_ast / n_ast_all if n_ast_all else float("inf")

    print("\n" + "=" * 70)
    print("CNN vs AST gunshot detection (match_tol = {:.2f}s)".format(max_match_s))
    print("=" * 70)
    print(f"{'Video':<20} {'Ref':<6} {'CNN_n':<8} {'CNN_MAE':<10} {'AST_n':<8} {'AST_MAE':<10}")
    print("-" * 70)
    for (name, nref, _, n_cnn, mae_cnn), (_, _, _, n_ast, mae_ast) in zip(results_cnn, results_ast):
        print(f"{name:<20} {nref:<6} {n_cnn:<8} {mae_cnn:.4f}s    {n_ast:<8} {mae_ast:.4f}s")
    print("-" * 70)
    print(f"{'Total matched':<20} {total_ref:<6} {n_cnn_all:<8} {mae_cnn_global:.4f}s    {n_ast_all:<8} {mae_ast_global:.4f}s")
    print("=" * 70)
    print("\nConclusion: n = matched count (ref shots with a detection within tol), MAE = mean absolute error of matched pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
