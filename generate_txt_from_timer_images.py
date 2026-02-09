"""
从 training data 文件夹里的 *-timer 图片（JPG/PNG）中识别时间数字，生成同名 .txt（每行一个时间，绝对秒数）。
用于 outdoor 等子目录下的 S1-timer.JPG, S2-timer.PNG 等。

Usage:
  python generate_txt_from_timer_images.py --folder "traning data"
  python generate_txt_from_timer_images.py --folder "traning data/outdoor-20260208T235429Z-1-001"
"""
import argparse
import os
import re


def _ocr_timer_image(image_path):
    """OCR 识别图片中的小数，返回列表。支持 .jpg / .png。"""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    if not os.path.isfile(image_path):
        return None
    try:
        img = Image.open(image_path)
        img = img.convert("L")
        for psm in (6, 11, 13):
            config = rf"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789., ()"
            text = pytesseract.image_to_string(img, config=config)
            numbers = re.findall(r"\d+\.\d+", text)
            if len(numbers) >= 2:
                return [float(x) for x in numbers]
    except Exception:
        pass
    return None


def _base_from_timer_filename(name):
    """S1-timer.JPG -> S1, S2-timer.PNG -> S2."""
    base = os.path.splitext(name)[0]
    if "-timer" in base.lower():
        return base.split("-")[0] or base
    return base


def process_folder(folder):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return 0
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    written = 0
    for name in sorted(os.listdir(folder)):
        if "timer" not in name.lower():
            continue
        if os.path.splitext(name)[1] not in exts:
            continue
        img_path = os.path.join(folder, name)
        if not os.path.isfile(img_path):
            continue
        base = _base_from_timer_filename(name)
        times = _ocr_timer_image(img_path)
        if not times:
            print(f"  {name} -> OCR failed or <2 numbers, skip")
            continue
        # 若最后一个数是整数且等于个数，多为 (19) 这种计数，去掉
        if len(times) >= 2 and abs(times[-1] - round(times[-1])) < 1e-6:
            n = int(round(times[-1]))
            if n == len(times) or n == len(times) - 1:
                times = times[:n] if n <= len(times) else times[:-1]
        txt_path = os.path.join(folder, base + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{t:.2f}" for t in times))
        print(f"  {name} -> {txt_path} ({len(times)} times)")
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description="From *-timer images in folder, OCR times and write base.txt")
    ap.add_argument("--folder", default="traning data", help="Root folder (will process this and subfolders if --recursive)")
    ap.add_argument("--recursive", action="store_true", help="Process subfolders too")
    args = ap.parse_args()
    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        return 1
    total = 0
    if args.recursive:
        for dirpath, _, _ in os.walk(root):
            n = process_folder(dirpath)
            total += n
    else:
        total = process_folder(root)
    print(f"Done: wrote {total} .txt file(s).")
    if total == 0:
        print("Tip: install pytesseract and Tesseract-OCR; images should be named *-timer.JPG/PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
