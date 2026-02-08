"""
从同目录下的每个 .jpg 用 OCR 解析枪声间隔，生成同名 .txt（每行一个数字）。
训练时会优先使用这些 .txt 作为每视频的真值。
若 OCR 不可用，则 1.jpg 使用 reference_splits 生成 1.txt。

Usage:
  python generate_ref_txt_from_jpg.py --folder 01032026
  python generate_ref_txt_from_jpg.py --folder .
"""
import argparse
import os

from ref_from_image import parse_ref_splits_from_image_ocr_only

# 1.jpg 对应 29 枪，与 REFERENCE_SPLITS 一致；OCR 失败时用此生成 1.txt
from reference_splits import REFERENCE_SPLITS


def write_splits_txt(txt_path, splits):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in splits))


def main():
    ap = argparse.ArgumentParser(description="Generate .txt ref from .jpg (OCR) in folder")
    ap.add_argument("--folder", default="01032026", help="Folder containing 1.jpg, 2.jpg, ...")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}")
        return
    exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
    written = 0
    for name in sorted(os.listdir(folder)):
        base, ext = os.path.splitext(name)
        if ext not in exts:
            continue
        jpg_path = os.path.join(folder, name)
        if not os.path.isfile(jpg_path):
            continue
        splits = parse_ref_splits_from_image_ocr_only(jpg_path)
        if not splits:
            # 1.jpg 与 REFERENCE_SPLITS 一致，OCR 失败时用其生成 1.txt
            if base == "1":
                splits = REFERENCE_SPLITS
            else:
                print(f"  {name} -> OCR failed or no numbers, skip")
                continue
        txt_path = os.path.join(folder, base + ".txt")
        write_splits_txt(txt_path, splits)
        print(f"  {name} -> {txt_path} ({len(splits)} splits)")
        written += 1
    print(f"Done: wrote {written} .txt file(s).")
    if written < len([n for n in os.listdir(folder) if os.path.splitext(n)[1] in exts]):
        print("Tip: install pytesseract and Tesseract-OCR to generate .txt from 2.jpg, 3.jpg, ...")


if __name__ == "__main__":
    main()
