"""
Parse reference shot splits from a JPG image (e.g. 1.jpg = ref for 1.mp4).
Image format: lines of decimal numbers (beep→1st shot, then inter-shot intervals);
optional last line with "0.56 (29) AMG 95D3" where (N) = total shot count.

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
    Return beep time (seconds). Priority:
    1) Same-folder {key}beep.txt (e.g. S1beep.txt, 1beep.txt) first line;
    2) beep_overrides.json with key S1/1/...;
    3) detect_beeps(audio_path, fps).
    """
    try:
        from detectors.beep import detect_beeps
    except Exception:
        return 0.0
    dirname = os.path.dirname(os.path.abspath(video_path))
    base = os.path.splitext(os.path.basename(video_path))[0]
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


def get_ref_times_for_video(video_path, beep_t, ref_image_path=None):
    """
    Get reference shot times for a video.
    - If ref_image_path is given and parseable, use splits from that image (or same-base .txt).
    - Else look for same-base jpg next to video (e.g. 1.mp4 -> 1.jpg), then parse image or base.txt.
    - Else try same-base .txt (e.g. 1.mp4 -> 1.txt) with one number per line or space-separated.
    - Else use global reference_splits.ref_shot_times(beep_t).
    """
    from reference_splits import ref_shot_times

    dirname = os.path.dirname(os.path.abspath(video_path)) if video_path else ""
    base = os.path.splitext(os.path.basename(video_path))[0] if video_path else ""

    jpg = ref_image_path
    if not jpg and video_path and dirname:
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            candidate = os.path.join(dirname, base + ext)
            if os.path.isfile(candidate):
                jpg = candidate
                break
    if jpg:
        splits = parse_ref_splits_from_image(jpg)
        if splits:
            return ref_shot_times_from_splits(beep_t, splits)
    if dirname and base:
        for base_try in (base, base.split("-")[0] if "-" in base else None):
            if not base_try:
                continue
            txt_path = os.path.join(dirname, base_try + ".txt")
            numbers = parse_ref_splits_from_txt(txt_path)
            if numbers:
                if _looks_like_absolute_times(numbers):
                    arr = [float(x) for x in numbers]
                    # If beep detected and first ref is before beep, treat txt as "seconds since beep" (e.g. outdoor S1-S8)
                    if beep_t > 0 and arr[0] < beep_t:
                        return [beep_t + x for x in arr]
                    return arr
                return ref_shot_times_from_splits(beep_t, numbers)
    return ref_shot_times(beep_t)
