"""
Check if beep (and thus ref times) might be wrong for 2 and 3.
- Compares beep from file/override vs detect_beeps().
- Shows ref time range vs detection time range (if they don't overlap, beep/ref is likely wrong).
- For 2 and 3: scan ref offset to see if a constant shift improves match (suggests beep/splits off).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_from_image import get_ref_times_for_video, get_beep_t_for_video
from detectors.beep import detect_beeps
from detectors.shot_audio import detect_shots
from train_logreg_multivideo import extract_audio, get_fps_duration
from extract_audio_plot import calibration_metrics

def main():
    folder = os.path.join(os.path.dirname(__file__), "traning data", "01032026")
    if not os.path.isdir(folder):
        print("Folder not found:", folder)
        return 1
    print("Video    beep_file  beep_detect  delta_beep   ref_range              det_range (CNN)   overlap?")
    print("-" * 95)
    tol = 0.12
    for name in ("1", "2", "3", "4", "5"):
        vp = os.path.join(folder, name + ".mp4")
        if not os.path.isfile(vp):
            continue
        try:
            audio_path = extract_audio(vp)
            fps, _ = get_fps_duration(vp)
        except Exception as e:
            print(name, "extract error:", e)
            continue
        beep_file = get_beep_t_for_video(vp, audio_path, fps)
        beeps = detect_beeps(audio_path, fps)
        beep_det = float(beeps[0]["t"]) if beeps else None
        ref = get_ref_times_for_video(vp, beep_file)
        if not ref:
            print(name, "no ref")
            continue
        det = detect_shots(audio_path, fps, use_ast_gunshot=False)
        t_det = [float(s["t"]) for s in det]
        ref_min, ref_max = min(ref), max(ref)
        det_min = min(t_det) if t_det else 0
        det_max = max(t_det) if t_det else 0
        overlap = "yes" if t_det and ref_max >= det_min and ref_min <= det_max else "no"
        delta = (beep_det - beep_file) if beep_det is not None else float("nan")
        print("{:6}   {:8.3f}   {:10.3f}   {:+.3f}s     [{:.2f}, {:.2f}]    [{:.2f}, {:.2f}]   {}".format(
            name + ".mp4", beep_file, beep_det or 0, delta,
            ref_min, ref_max, det_min, det_max, overlap))
    # For 2 and 3: try shifting ref by offset; if match improves a lot, beep/splits may be off
    print("\n--- Ref offset scan (if best offset != 0, beep or splits may be wrong) ---")
    for name in ("2", "3"):
        vp = os.path.join(folder, name + ".mp4")
        audio_path = extract_audio(vp)
        fps, _ = get_fps_duration(vp)
        beep_file = get_beep_t_for_video(vp, audio_path, fps)
        ref = get_ref_times_for_video(vp, beep_file)
        det = detect_shots(audio_path, fps, use_ast_gunshot=False)
        t_det = np.array([float(s["t"]) for s in det])
        best_n, best_off, best_mae = 0, 0.0, float("inf")
        for off in np.arange(-0.5, 0.55, 0.05):
            n, mae = calibration_metrics(np.array(ref) + off, t_det, max_match_s=tol)
            if n > best_n or (n == best_n and mae < best_mae):
                best_n, best_off, best_mae = n, off, mae
        n0, mae0 = calibration_metrics(ref, t_det, max_match_s=tol)
        print("{}: current match {} MAE {:.4f}s  |  best offset {:.2f}s -> match {} MAE {:.4f}s".format(
            name + ".mp4", n0, mae0, best_off, best_n, best_mae))
    return 0

if __name__ == "__main__":
    sys.exit(main())
