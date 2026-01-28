import os
import cv2
from PIL import Image, ImageDraw, ImageFont

def render_overlay_frames(video, events, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    fps = events["video"]["fps"]
    shots = events["shots"]
    beeps = events["beeps"]

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    font = ImageFont.load_default()

    shot_frames = {}
    shot_motion_confirmed = {}
    for i, s in enumerate(shots):
        shot_frames[s["frame"]] = i+1
        shot_motion_confirmed[s["frame"]] = s.get("motion_confirmed", False)
    beep_frames = {b["frame"] for b in beeps}

    idx = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break

        img = Image.new("RGBA", (width, height), (0,0,0,0))
        d = ImageDraw.Draw(img)

        t = idx / fps
        d.text((20, 20), f"{t:.3f}s", fill=(255,255,255,200), font=font)

        if idx in beep_frames:
            d.text((20, 60), "BIP", fill=(0,255,0,255), font=font)

        if idx in shot_frames:
            shot_num = shot_frames[idx]
            motion_ok = shot_motion_confirmed.get(idx, False)
            # Green if motion-confirmed, red if audio-only
            color = (0, 255, 0, 255) if motion_ok else (255, 0, 0, 255)
            label = f"SHOT #{shot_num}"
            if motion_ok:
                label += " ✓"
            d.text(
                (width//2 - 50, height//2),
                label,
                fill=color,
                font=font
            )

        img.save(os.path.join(out_dir, f"{idx:06d}.png"))
        idx += 1

    cap.release()
