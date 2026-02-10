"""
Parse reference shot splits from a JPG image (e.g. 1.jpg = ref for 1.mp4).
Image format: lines of decimal numbers (beep→1st shot, then inter-shot intervals);
optional last line with "0.56 (29) AMG 95D3" where (N) = total shot count.

Naming convention (same folder as *.mp4):
  *.mp4       = main shooting video
  *cali.txt   = calibration splits (preferred over *.txt when present); ref_times = beep_t + cumsum(splits)
  *.txt       = split data (intervals from beep), one number per line; used when *cali.txt absent
  *beep.txt   = beep time relative to video start (seconds), single line

Returns list of floats = splits; ref_times = beep_t + np.cumsum(splits).
Requires: pip install pytesseract Pillow; Tesseract-OCR installed on system.
"""
import os
import re


def _try_pytesseract(image_path, psm=6):
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    if not os.path.isfile(image_path):
        return None
    try:
        img = Image.open(image_path)
        img = img.convert("L")  # grayscale
        config = rf"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789. ()"
        text = pytesseract.image_to_string(img, config=config)
        return text
    except Exception:
        return None


def parse_ref_splits_from_txt(txt_path):
    """Parse splits from a sidecar .txt: one number per line or space-separated. Returns list of floats."""
    if not os.path.isfile(txt_path):
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    numbers = re.findall(r"\d+\.\d+", raw)
    return [float(x) for x in numbers] if numbers else None


def _looks_like_absolute_times(numbers):
    """True if numbers look like absolute timestamps (e.g. 2.13, 2.69, 3.18) rather than splits."""
    if not numbers or len(numbers) < 2:
        return False
    arr = [float(x) for x in numbers]
    if min(arr) < 0.2:
        return False
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1] - 0.01:
            return False
    return True


def parse_ref_splits_from_image_ocr_only(jpg_path):
    """
    Parse splits from a JPG using OCR only (no .txt sidecar).
    Returns list of floats, or None if OCR failed or no numbers found.
    Tries PSM 6 (block), then 11 (sparse), then 13 (raw line).
    """
    for psm in (6, 11, 13):
        text = _try_pytesseract(jpg_path, psm=psm)
        if not text:
            continue
        numbers = re.findall(r"\d+\.\d+", text)
        if numbers:
            return [float(x) for x in numbers]
    return None


def parse_ref_splits_from_image(jpg_path):
    """
    Parse reference splits from a JPG (decimal numbers in order).
    Returns list of floats (splits), or None if parse failed.
    First tries sidecar .txt (e.g. 1.jpg -> 1.txt) to avoid OCR dependency.
    """
    base = os.path.splitext(jpg_path)[0]
    txt_path = base + ".txt"
    splits = parse_ref_splits_from_txt(txt_path)
    if splits:
        return splits
    text = _try_pytesseract(jpg_path)
    if not text:
        return None
    numbers = re.findall(r"\d+\.\d+", text)
    if not numbers:
        return None
    return [float(x) for x in numbers]


def ref_shot_times_from_splits(beep_t, splits):
    """Compute ref shot times (abs seconds): beep_t + cumsum(splits)."""
    import numpy as np
    return (beep_t + np.cumsum(splits)).tolist()


def get_beep_t_for_video(video_path, audio_path, fps):
    """
    Return beep time in seconds (relative to video start).
    Priority: 1) same-folder *beep.txt (first line); 2) beep_overrides.json; 3) detect_beeps().
    *beep.txt = beep time relative to video start (one number per file).
    Paths: video_path is normalized to absolute so *beep.txt is always looked up next to the video file.
    """
    try:
        from detectors.beep import detect_beeps
    except Exception:
        return 0.0
    video_path = os.path.abspath(os.path.normpath(video_path)) if video_path else ""
    dirname = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0] if video_path else ""
    key = base.split("-")[0] if "-" in base else base
    beep_txt = os.path.join(dirname, key + "beep.txt")
    if os.path.isfile(beep_txt):
        try:
            with open(beep_txt, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            if line:
                return float(line)
        except Exception:
            pass
    override_path = os.path.join(dirname, "beep_overrides.json")
    if os.path.isfile(override_path):
        try:
            import json
            with open(override_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            if key in overrides:
                return float(overrides[key])
        except Exception:
            pass
    beeps = detect_beeps(audio_path, fps)
    return float(beeps[0]["t"]) if beeps else 0.0


def get_ref_times_and_source(video_path, beep_t, ref_image_path=None):
    """
    Get reference shot times and a short source description.
    Only uses same-name .txt for splits (no same-name .jpg). ref = beep_t + cumsum(splits).
    Returns (ref_times_list, source_str).
    Paths: video_path is normalized to absolute so *.txt is always looked up next to the video file.
    """
    from reference_splits import ref_shot_times

    video_path = os.path.abspath(os.path.normpath(video_path)) if video_path else ""
    dirname = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0] if video_path else ""
    base_try = base.split("-")[0] if "-" in base else base

    # Only same-name .txt (no same-name .jpg). If ref_image_path given, use its same-base .txt when it's an image.
    if ref_image_path and os.path.isfile(ref_image_path):
        if ref_image_path.lower().endswith((".jpg", ".jpeg")) or not ref_image_path.lower().endswith(".txt"):
            splits = parse_ref_splits_from_image(ref_image_path)
            if splits:
                txt_path = os.path.splitext(ref_image_path)[0] + ".txt"
                src = "{} (splits)".format(os.path.basename(txt_path))
                return ref_shot_times_from_splits(beep_t, splits), src
        else:
            numbers = parse_ref_splits_from_txt(ref_image_path)
            if numbers:
                if _looks_like_absolute_times(numbers):
                    arr = [float(x) for x in numbers]
                    if beep_t > 0 and arr[0] < beep_t:
                        return [beep_t + x for x in arr], "{} (absolute since beep)".format(os.path.basename(ref_image_path))
                    return arr, "{} (absolute)".format(os.path.basename(ref_image_path))
                return ref_shot_times_from_splits(beep_t, numbers), "{} (splits)".format(os.path.basename(ref_image_path))
    if dirname and base:
        for b in (base, base_try) if base_try != base else (base,):
            cali_path = os.path.join(dirname, b + "cali.txt")
            txt_path = os.path.join(dirname, b + ".txt")
            path_to_try = cali_path if os.path.isfile(cali_path) else txt_path
            numbers = parse_ref_splits_from_txt(path_to_try)
            if numbers:
                if _looks_like_absolute_times(numbers):
                    arr = [float(x) for x in numbers]
                    if beep_t > 0 and arr[0] < beep_t:
                        return [beep_t + x for x in arr], "{} (absolute since beep)".format(os.path.basename(path_to_try))
                    return arr, "{} (absolute)".format(os.path.basename(path_to_try))
                return ref_shot_times_from_splits(beep_t, numbers), "{} (splits)".format(os.path.basename(path_to_try))
    ref = ref_shot_times(beep_t)
    return ref, "reference_splits(beep)"

def get_ref_times_for_video(video_path, beep_t, ref_image_path=None):
    """
    Get reference shot times. Only same-name .txt for splits (no same-name .jpg). ref = beep_t + cumsum(splits).
    - If ref_image_path given: use that file (image -> same-base .txt, or .txt directly).
    - Else same-base .txt only (e.g. 2.mp4 -> 2.txt).
    - Else global reference_splits.ref_shot_times(beep_t).
    """
    ref, _ = get_ref_times_and_source(video_path, beep_t, ref_image_path)
    return ref
