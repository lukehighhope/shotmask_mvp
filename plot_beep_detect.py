"""
生成一整张图：从上到下依次 1.mp4～5.mp4 的波形，每行标出 beep 时刻。
默认保存到输入文件夹内：<folder>/beep_detect.png。

Usage:
  python plot_beep_detect.py --folder "traning data/01032026"
  python plot_beep_detect.py --folder "traning data/01032026" --output 其他路径/beep_detect.png
"""
import os
import argparse
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from extract_audio_plot import get_ffmpeg_cmd, extract_audio


def read_beep_txt(folder, base):
    path = os.path.join(folder, base + "beep.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            return float(line) if line else None
    except Exception:
        return None


def draw_one_row(data, sr, beep_t, label, row_width=1200, row_height=180):
    """Draw one row: waveform + green vertical line at beep_t. Returns PIL Image."""
    duration = len(data) / sr
    max_pts = 4000
    if len(data) > max_pts:
        step = len(data) // max_pts
        data = data[::step]
    n = len(data)
    max_val = np.max(np.abs(data)) or 1.0
    normalized = data.astype(np.float64) / max_val

    img = Image.new("RGB", (row_width, row_height), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    left, right = 50, row_width - 30
    top, bottom = 30, row_height - 30
    plot_w = right - left
    plot_h = bottom - top
    mid_y = (top + bottom) // 2

    # Waveform
    for i in range(n - 1):
        x1 = left + int(i * plot_w / max(1, n - 1))
        y1 = mid_y - int(normalized[i] * (plot_h // 2))
        x2 = left + int((i + 1) * plot_w / max(1, n - 1))
        y2 = mid_y - int(normalized[i + 1] * (plot_h // 2))
        y1 = max(top, min(bottom, y1))
        y2 = max(top, min(bottom, y2))
        draw.line([(x1, y1), (x2, y2)], fill=(70, 130, 180), width=1)

    # Beep vertical line
    if duration > 0 and 0 <= beep_t <= duration:
        x_beep = left + int(beep_t / duration * plot_w)
        draw.line([(x_beep, top), (x_beep, bottom)], fill=(0, 180, 0), width=2)
        draw.text((x_beep + 4, top), f"beep {beep_t:.2f}s", fill=(0, 120, 0))

    # Label (e.g. "1.mp4")
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((left, 6), label, fill=(0, 0, 0), font=font)
    return img


def main():
    ap = argparse.ArgumentParser(description="One image: 1-5 waveforms with beep, top to bottom")
    ap.add_argument("--folder", default="traning data/01032026")
    ap.add_argument("--output", default=None, help="默认保存到 folder/beep_detect.png")
    args = ap.parse_args()
    folder = os.path.abspath(args.folder)
    out_path = os.path.abspath(args.output) if args.output else os.path.join(folder, "beep_detect.png")

    ffmpeg = get_ffmpeg_cmd()
    if not ffmpeg:
        print("ffmpeg not found")
        return 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Auto-detect naming: S1beep.txt → S1-main.mp4; else 1beep.txt → 1.mp4
    entries = []
    if os.path.isfile(os.path.join(folder, "S1beep.txt")):
        for i in range(1, 9):
            base = f"S{i}"
            vp = os.path.join(folder, base + "-main.mp4")
            if not os.path.isfile(vp):
                continue
            beep_t = read_beep_txt(folder, base)
            if beep_t is None:
                continue
            entries.append((f"{base}-main.mp4", vp, beep_t))
    else:
        for i in range(1, 10):
            base = str(i)
            vp = os.path.join(folder, base + ".mp4")
            if not os.path.isfile(vp):
                continue
            beep_t = read_beep_txt(folder, base)
            if beep_t is None:
                continue
            entries.append((f"{base}.mp4", vp, beep_t))

    if not entries:
        print("No videos with beep .txt found.")
        return 1

    row_width = 1200
    row_height = 180
    rows = []
    for label, vp, beep_t in entries:
        base_id = label.replace(".mp4", "").replace("-main", "")
        audio_path = extract_audio(vp, ffmpeg, f"tmp/audio_beep_{base_id}.wav", channels=1)
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        row_img = draw_one_row(data, sr, beep_t, label, row_width, row_height)
        rows.append(row_img)

    # Stack vertically
    total_h = sum(im.height for im in rows)
    title_h = 44
    out = Image.new("RGB", (row_width, title_h + total_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    title = f"Beep Detect ({len(rows)} clips)"
    draw.text((row_width // 2 - len(title) * 4, 12), title, fill=(0, 0, 0), font=font)
    y = title_h
    for im in rows:
        out.paste(im, (0, y))
        y += im.height
    out.save(out_path)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
