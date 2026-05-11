#!/usr/bin/env python3
"""Probe trimmed PD gunshot clip under CPU Torch (simulate CI).

Run:
  CUDA_VISIBLE_DEVICES= python scripts/probe_golden_trim.py

PowerShell example:
  $env:CUDA_VISIBLE_DEVICES=""; python scripts/probe_golden_trim.py

Requires tmp/golden_src.ogg — download via scripts/build_real_golden_wav.py (saved as tmp/_golden_dl_gunshots.ogg)
or replicate with curl against Wikimedia Commons.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import imageio_ffmpeg as iif  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectors import config_paths  # noqa: E402
from detectors import shot_audio  # noqa: E402
from detectors.shot_audio import detect_shots  # noqa: E402


def trim(dur: float, og: Path, out_wav: Path) -> None:
    exe = iif.get_ffmpeg_exe()
    subprocess.run(
        [
            exe,
            "-y",
            "-i",
            str(og),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-ss",
            "0",
            "-to",
            str(dur),
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )


def main():
    repo = Path(config_paths.project_root())
    og = repo / "tmp" / "golden_src.ogg"
    if not og.is_file():
        alt = repo / "tmp" / "_golden_dl_gunshots.ogg"
        og = alt if alt.is_file() else og
    if not og.is_file():
        print("missing ogg — run scripts/build_real_golden_wav.py once")
        sys.exit(1)

    probe = repo / "tests" / "fixtures" / "_probe_trim.wav"

    raw = json.loads((repo / "calibrated_detector_params.json").read_text(encoding="utf-8"))
    cal = dict(raw)
    cal["cnn_gunshot_path"] = str((repo / "outputs" / "cnn_gunshot.pt").resolve())
    cal["ast_gunshot_path"] = None
    cal["use_ast_gunshot"] = False
    cal["use_cnn_only"] = True

    orig = shot_audio.load_calibrated_params
    shot_audio.load_calibrated_params = lambda: shot_audio._resolve_cal_paths(dict(cal))

    try:
        for dur in [1.05, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5]:
            trim(dur, og, probe)
            shots = detect_shots(str(probe), fps=30)
            xs = [(round(float(s["t"]), 4), round(float(s["confidence"]), 3)) for s in shots]
            hi = sum(1 for _, c in xs if c >= 0.65)
            print(f"dur={dur:.2f}s n={len(xs)} hi65={hi} {xs}")
    finally:
        shot_audio.load_calibrated_params = orig


if __name__ == "__main__":
    main()
